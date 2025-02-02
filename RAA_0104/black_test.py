import argparse
import os

import torch
import tqdm
import time
import torchvision
import transferattack
from transferattack.utils import *
import torch.nn.functional as F
import torchvision.transforms.functional as TF
import Transform
from Transform import *


def get_parser():
    parser = argparse.ArgumentParser(description='Generating transferable adversaria examples')
    parser.add_argument('-e', '--eval', default=False, action='store_true', help='attack/evluation')
    parser.add_argument('--attack', default='mifgsm', type=str, help='the attack algorithm', choices=transferattack.attack_zoo.keys())
    parser.add_argument('--epoch', default=10, type=int, help='the iterations for updating the adversarial patch')
    parser.add_argument('--batchsize', default=5, type=int, help='the bacth size')
    # parser.add_argument('--epsilon', default= 16 / 255, type=float, help='the stepsize to update the perturbation')
    parser.add_argument('--epsilon', default= 16/255, type=float, help='the stepsize to update the perturbation')
    parser.add_argument('--alpha', default=1.6 / 255, type=float, help='the stepsize to update the perturbation')
    parser.add_argument('--momentum', default=0., type=float, help='the decay factor for momentum based attack')
    parser.add_argument('--model', default='resnet18', type=str, help='the source surrogate model')
    parser.add_argument('--target_model', default='resnet18', type=str)
    parser.add_argument('--ensemble', action='store_true', help='enable ensemble attack')
    parser.add_argument('--random_start', default=False, type=bool, help='set random start')
    parser.add_argument('--input_dir', default='./data', type=str, help='the path for custom benign images, default: untargeted attack data')
    parser.add_argument('--output_dir', default='./results', type=str, help='the path to store the adversarial patches')
    parser.add_argument('--targeted', default=0, action='store_true', help='targeted attack')
    parser.add_argument('--GPU_ID', default='1', type=str)
    parser.add_argument('--max_test_num', default=1000, type=int)
    parser.add_argument('--transform', type=str)
    parser.add_argument('--transform_par', type=float)
    return parser.parse_args()

def add_noise(x_adv, x, l2):
    noise = torch.randn_like(x)
    current_l2_norm = torch.norm(noise, p=2)
    scaled_noise = (l2 / current_l2_norm) * noise
    x_adv = x_adv.cpu() + scaled_noise
    return x_adv

def reshape_image(tensor, target_size=(224, 224)):
    """
    Reshape the input tensor to the target size by either cropping or padding.

    Arguments:
        tensor (torch.Tensor): The input tensor with shape (batch_size, channels, height, width).
        target_size (tuple, optional): The target size (height, width). Defaults to (224, 224).

    Returns:
        torch.Tensor: The reshaped tensor with the target size.
    """
    batch_size, channels, orig_height, orig_width = tensor.shape
    resized_tensor = torch.empty((batch_size, channels, target_size[0], target_size[1]), dtype=tensor.dtype).to('cuda')
    
    for i in range(batch_size):
        # Get the current image
        current_image = tensor[i].to('cuda')
        
        # Calculate cropping or padding
        if current_image.size(1) > target_size[0] or current_image.size(2) > target_size[1]:
            # If the image is too large, perform center cropping
            diffH = current_image.size(1) - target_size[0]
            diffW = current_image.size(2) - target_size[1]
            crop_top = diffH // 2
            crop_left = diffW // 2
            current_image = current_image[:, crop_top:crop_top+target_size[0], crop_left:crop_left+target_size[1]]
        else:
            # If the image is too small, perform padding
            current_image = TF.center_crop(current_image, target_size)
            padding = [0, 0, 0, 0]
            if current_image.size(1) < target_size[0]:
                padding[2] = target_size[0] - current_image.size(1)
            if current_image.size(2) < target_size[1]:
                padding[3] = target_size[1] - current_image.size(2)
            current_image = TF.pad(current_image, padding, fill=0)  # Pad with zeros
        
        # Place the resized image into the result tensor
        resized_tensor[i] = current_image
    
    return resized_tensor


def test(args):
    """
    Test the adversarial attack on a dataset and evaluate the accuracy.
    There are two different models as source model and target model

    Arguments:
        args (object): An object containing the following attributes:
            - GPU_ID (str): The ID of the GPU to use.
            - output_dir (str): The directory to save the output results.
            - input_dir (str): The directory of the input dataset.
            - targeted (bool): Whether the attack is targeted.
            - max_test_num (int): The maximum number of samples to test.
            - batchsize (int): The batch size for the dataloader.
            - target_model (str): The name of the target model.
            - model (str): The name of the source model.
            - attack (str): The name of the attack method.

    Returns:
        None
    """
    # Set the GPU ID
    os.environ["CUDA_VISIBLE_DEVICES"] = args.GPU_ID

    # Create the output directory if it does not exist
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    # Initialize variables
    asr_lim = 1
    l2 = 0

    # Load the dataset
    dataset = AdvDataset(input_dir=args.input_dir, output_dir=args.output_dir, targeted=args.targeted, eval=False)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=args.batchsize, shuffle=False, num_workers=0)

    # Load the target model
    target_model = getattr(models, args.target_model)(pretrained=True)
    target_model = target_model.cuda().eval()

    # Load the source model(s)
    if args.ensemble or len(args.model.split(',')) > 1:
        args.model = args.model.split(',')

    # Load the attack class
    attacker = transferattack.load_attack_class(args.attack)(model_name=args.model, targeted=args.targeted)

    # Initialize counters
    correct = 0
    total = 0
    total_time = 0

    # Test the attack on the dataset
    for batch_idx, [images, labels, filenames] in tqdm.tqdm(enumerate(dataloader)):
        if (batch_idx * args.batchsize >= args.max_test_num):
            break
        start_time = time.time()

        # Perform the attack
        if args.attack in ['LinfPGD', 'EOTPGD', 'LinfBasicIterative', 'LinfMomentumIterative']:
            if args.targeted:
                labels = labels[1]
            adv = attacker.perturb(images.cuda(), torch.tensor(labels).cuda())
        else:
            perturbations, model = attacker.forward(images, labels)
            images = images.cuda()
            adv = images.detach() + perturbations.detach()
            if args.targeted:
                labels = labels[1]

        # Evaluate the attack on the target model
        pred = target_model(adv.cuda())
        correct += np.sum(labels.cpu().numpy() == pred.argmax(dim=1).detach().cpu().numpy())
        total += labels.shape[0]

        # Measure the time
        end_time = time.time()
        total_time += end_time - start_time

    # Output the accuracy and time
    acc = correct / total
    with open('black_result.txt', 'a') as f:
        f.write(args.attack + '_' + args.model + '_' + args.target_model + '_' + str(args.targeted) + '   acc:{}'.format(acc) + '   time:{}'.format(total_time) + '\n')
    print("ACC:{}".format(acc))

        


if __name__ == '__main__':
    args = get_parser()
    # model_list = ['resnet34','vgg19','densenet121','resnet101','inception_v3']
    model_list = ['vgg19']
    target_list = ['resnet34','densenet121']
    #attack_list supports ['raa','raa_dta','raa_input','raa_smifgrm']
    attack_list = ['raa']


    for model in model_list:
        args.model = model
        for tmodel in target_list:
            args.target_model = tmodel
            for attack in attack_list:
                args.attack = attack
                test(args)
