import argparse
import os

import torch
import tqdm
import time
import torchvision
import transferattack
# from adverattack.attacks import load_adver_attacks
from transferattack.utils import *
import torch.nn.functional as F
import torchvision.transforms.functional as TF
import Transform
from Transform import *
from PIL import Image
os.environ['CUDA_VISIBLE_DEVICES'] = '1'

def get_parser():
    parser = argparse.ArgumentParser(description='Generating transferable adversaria examples')
    parser.add_argument('--attack', default='raa', type=str, help='the attack algorithm', choices=transferattack.attack_zoo.keys())
    parser.add_argument('--epoch', default=100, type=int, help='the iterations for updating the adversarial patch')
    parser.add_argument('--batchsize', default=4, type=int, help='the bacth size')
    parser.add_argument('--epsilon', default= 16 /255, type=float, help='the stepsize to update the perturbation')
    parser.add_argument('--alpha', default=1.6 / 255, type=float, help='the stepsize to update the perturbation')
    parser.add_argument('--momentum', default=0., type=float, help='the decay factor for momentum based attack')
    parser.add_argument('--ensemble', action='store_true', help='enable ensemble attack')
    parser.add_argument('--random_start', default=False, type=bool, help='set random start')
    parser.add_argument('--input_dir', default='./data', type=str, help='the path for custom benign images, default: untargeted attack data')
    parser.add_argument('--output_dir', default='./results', type=str, help='the path to store the adversarial patches')
    parser.add_argument('--targeted', default=0, action='store_true', help='targeted attack')
    parser.add_argument('--GPU_ID', default='cuda:0', type=str)
    parser.add_argument('--max_test_num', default=1000, type=int)
    parser.add_argument('--transform', type=str)
    parser.add_argument('--transform_par', type=float)
    return parser.parse_args()


def save_image(tensor, path, name):
    """
    Save a tensor as an image file.

    Arguments:
        tensor (torch.Tensor): The tensor to be saved as an image.
        path (str): The directory path where the image will be saved.
        name (str): The filename for the saved image.

    Returns:
        None
    """
    if not os.path.exists(path):
        os.makedirs(path)
    image = TF.to_pil_image(tensor.cpu())
    image.save(os.path.join(path, name))


def combine_images(images, path, name, gap=10):
    """
    Combine multiple images into a single horizontal collage with white gaps in between.

    Arguments:
        images (list of PIL.Image): A list of images to be combined.
        path (str): The directory path where the combined image will be saved.
        name (str): The filename for the combined image.
        gap (int, optional): The width of the white gap between images in pixels. Defaults to 10.

    Returns:
        None
    """
    if not os.path.exists(path):
        os.makedirs(path)
    widths, heights = zip(*(img.size for img in images))
    total_width = sum(widths) + gap * (len(images) - 1)
    max_height = max(heights)
    new_image = Image.new('RGB', (total_width, max_height), (255, 255, 255))
    x_offset = 0
    for img in images:
        new_image.paste(img, (x_offset, 0))
        x_offset += img.width + gap
    new_image.save(os.path.join(path, name))

def add_noise(x_adv, x, l2):
    """
    The add_noise function, which adds L2-norm constrained noise to an adversarial example.

    Arguments:
        x_adv (torch.cuda.Tensor): The adversarial example tensor.
        x (torch.cuda.Tensor): The original input tensor, used to ensure the noise is added to the correct device and type.
        l2 (float): The desired L2 norm of the noise to be added.

    Returns:
        torch.cuda.Tensor: The adversarial example with added noise, moved to the CPU.
    """
    noise = torch.randn_like(x)
    current_l2_norm = torch.norm(noise, p=2)
    scaled_noise = (l2 / current_l2_norm) * noise
    x_adv = x_adv + scaled_noise.to(args.GPU_ID)
    return x_adv


def reshape_image(tensor, target_size=(224, 224)):
    """
        The reshape function, which is used to resize the transformed images to a specified size.

        Arguments:
            tensor (torch.cuda.Tensor): The input tensor containing the transformed images with shape (batch_size, channels, height, width)
            target_size (tuple, optional): The desired size of the output images. Defaults to (224, 224).

        Returns:
            torch.cuda.Tensor: The resized tensor with shape (batch_size, channels, 224, 224) on the CUDA device.
    """
    batch_size, channels, orig_height, orig_width = tensor.shape
    resized_tensor = torch.empty((batch_size, channels, target_size[0], target_size[1]), dtype=tensor.dtype).to(args.GPU_ID)
    
    for i in range(batch_size):
        # Get current image
        current_image = tensor[i].to(args.GPU_ID)
        
        if current_image.size(1) > target_size[0] or current_image.size(2) > target_size[1]:
            # image_size > target_size : cutting
            diffH = current_image.size(1) - target_size[0]
            diffW = current_image.size(2) - target_size[1]
            crop_top = diffH // 2
            crop_left = diffW // 2
            current_image = current_image[:, crop_top:crop_top+target_size[0], crop_left:crop_left+target_size[1]]
        else:
            # image_size < target_size : padding
            current_image = TF.center_crop(current_image, target_size)
            padding = [0, 0, 0, 0]
            if current_image.size(1) < target_size[0]:
                padding[2] = target_size[0] - current_image.size(1)
            if current_image.size(2) < target_size[1]:
                padding[3] = target_size[1] - current_image.size(2)
            current_image = TF.pad(current_image, padding, fill=0)  # padding black

        resized_tensor[i] = current_image
    
    return resized_tensor

def main(args):
    """
    The main function for performing adversarial attacks, generating adversarial examples with additional perturbations, 
    saving the images, and evaluating the accuracy.

    This function implements the basic workflow of adversarial attack processing, including generating adversarial samples, 
    applying secondary perturbations, saving images, and evaluating the success rate (ASR).

    Arguments:
        args (object): An object containing various arguments and parameters for the attack, including:
            - input_dir (str): The directory path of the input dataset.
            - output_dir (str): The directory path for saving the output results.
            - targeted (bool): Whether the attack is targeted or untargeted.
            - model (str): The name of the model used for the attack.
            - attack (str): The name of the attack method.
            - batchsize (int): The batch size for processing data.
            - max_test_num (int): The maximum number of samples to test.
            - transform_dict (dict): A dictionary of transformations and their parameters.
            - GPU_ID (int): The ID of the GPU to use.

    Note:
        This function assumes that the necessary directories and files are properly set up, and that the required libraries 
        (e.g., PyTorch, torchvision) are installed and imported.
    """
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
   
    dataset = AdvDataset(input_dir=args.input_dir, output_dir=args.output_dir, targeted=args.targeted, eval=False)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=args.batchsize, shuffle=False, num_workers=0)

    if args.ensemble or len(args.model.split(',')) > 1:
        args.model = args.model.split(',')

    attacker = transferattack.load_attack_class(args.attack)(model_name=args.model, targeted=args.targeted)

    cnt = 0
    correct = 0
    total = 0
    total_time = 0
    correct_list = []
    asr_list = []
    for batch_idx, [images, labels, filenames] in tqdm.tqdm(enumerate(dataloader)):
        cnt_1 = 0
        if((batch_idx+1)*len(labels)>= args.max_test_num):break
        start_time = time.time()

        perturbations, model = attacker.forward(images, (labels)) #model
        images = images.cuda()
        adv = images.detach()+perturbations.detach()
        if(args.targeted): labels = labels[1]

        for transform in transform_dict.keys():
            args.transform = transform
            par_list = transform_dict[transform]
            for par in par_list:
                args.transform_par = par
                print('now testing attack:{}  model:{} transform:{} par:{}'.format(args.attack , args.model,args.transform,par))
                
                if (transform == "resize_image" or "Ranom_Affine"):
                    transform = getattr(Transform,args.transform)
                    adv_test = reshape_image(transform(adv,args.transform_par))
                    # adv_test = (transform(adv.cuda(),args.transform_par))
                else:
                    transform = getattr(Transform,args.transform)
                    adv_test = (transform(adv.cuda(),args.transform_par))

                pred = model(adv_test.cuda())
                # label[0] is for accuracy without target, labels[1] is for ASR with target
                correct = np.sum(labels.cpu().numpy() == pred.argmax(dim=1).detach().cpu().numpy())
                if(batch_idx == 0):
                    correct_list.append(correct)
                    length = len(correct_list)
                else:
                    correct_list[cnt_1] += correct 
                    cnt_1 +=1

        # Save the original image as a picture
        original_image = TF.to_pil_image(images[0].cpu())
        save_image(images[0], "./pic/imgnet", f"{args.attack}_original_{batch_idx}.png")

        # Get the adversarial example and save it
        adv_image = TF.to_pil_image(adv[0].cpu())
        save_image(adv[0], "./pic/imgnet", f"{args.attack}_new_{batch_idx}.png")
        # Save the transformed image
        adv_test_image = TF.to_pil_image(adv_test[0].cpu())
        save_image(adv_test[0], "./pic/imgnet", f"{args.attack}_transformed_{batch_idx}.png")

        # Combine three images
        combined_image_name = f"{args.attack}_combined_{batch_idx}.png"
        combine_images([original_image, adv_image, adv_test_image], "./pic/imgnet", combined_image_name)        
        total += labels.shape[0]
                
        end_time = time.time()
        total_time += end_time - start_time

    asr_list = [x / total for x in correct_list]

    # Output each ASR value
    for transform in transform_dict.keys():
                args.transform = transform
                par_list = transform_dict[transform]
                for par in par_list:
                    args.transform_par = par
                    with open('results_eval.txt', 'a') as f:
                        f.write(args.attack + '_' + args.model + '_' +args.transform+'_'+ str(args.transform_par) + '_' + str(asr_list[cnt]) + '   time:' + str(total_time)+ '\n')
                    print("ASR:{}".format(asr_list[cnt]))
                    cnt += 1
                    

if __name__ == '__main__':
    args = get_parser()
    # model_list supports ['vgg19','resnet34','densenet121','resnet101','inception_v3']
    model_list = ['vgg19']
    attack_list = ['raa']
    transform_dict = {
        # neighbor trans
        'adjust_brightness':[0.05,0.1,0.25,0.5],
        'adjust_contrast':[0.05,0.1,0.25,0.5],
        'JPEG_transform': [30,50,70,90],
        'Gaussian_blur':[[5,1.0]],
        # None-neighbor trans
        'resize_image':[0.5,0.75,0.9,1.1,1.25,1.5,2],
        'rotate_image':[0,1,2,5,10,15,20,30,45],
        'Random_perspective':[[0.25,1.0],[0.5,1.0]],
        'Ranom_Affine':[[10,[0.05,0.05],[0.9,0.9]]],

    }

    for model in model_list:
        args.model = model
        for attack in attack_list:
            args.attack = attack
            main(args)
