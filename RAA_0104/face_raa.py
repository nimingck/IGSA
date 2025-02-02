import os
import cv2
import numpy as np
import onnx
import onnx2pytorch
from insightface.app import FaceAnalysis
from insightface.data import get_image as ins_get_image
import torchvision.transforms.functional as TF
import Transform
import torch
from insightface.model_zoo import model_zoo
from insightface.utils import face_align
from PIL import Image
from torchvision import transforms
import transferattack

os.environ['CUDA_VISIBLE_DEVICES'] = '1'

def save_image(image, path, name):
    if not os.path.exists(path):
        os.makedirs(path)
    image.save(os.path.join(path, name))

def reshape_image(tensor, target_size=(112, 112)):
    """
        The reshape function, which is used to resize the transformed images to a specified size.

        Arguments:
            tensor (torch.cuda.Tensor): The input tensor containing the transformed images with shape (batch_size, channels, height, width)
            target_size (tuple, optional): The desired size of the output images. Defaults to (112, 112).

        Returns:
            torch.cuda.Tensor: The resized tensor with shape (batch_size, channels, 112, 112) on the CUDA device.
    """
    batchsize, channels, orig_height, orig_width = tensor.shape
    resized_tensor = torch.empty((batchsize, channels, target_size[0], target_size[1]), dtype=tensor.dtype).to('cuda')
    
    for i in range(batchsize):
        # Get current image
        current_image = tensor[i].to('cuda')
 
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

# load_identities
def load_identities(file_path):
    identities = {}
    with open(file_path, 'r') as f:
        for line in f:
            values = line.strip().split()
            identities[values[0]] = values[1]
    return identities

# load_attributes
def load_attributes(file_path):
    attributes = {}
    with open(file_path, 'r') as f:
        for line in f:
            values = line.strip().split()
            attributes[values[0]] = values[1:]
    return attributes

def load_dataset():
    # Load dataset e.g. CelebA
    dataset_dir = '/path/to/your/Img'
    input_size = (112, 112)  # ArcFace needs size of pics is 112x112
    img_align_celeba_dir = os.path.join(dataset_dir, 'img_test')

    return input_size,img_align_celeba_dir

def load_models():
    """
    Load face recognition models and convert an ONNX model to a PyTorch model.

    This function initializes a face recognition model using the `FaceAnalysis` class and loads an ONNX model, 
    which is then converted to a PyTorch model using the `onnx2pytorch` library.

    Note:
        - The function assumes that the `buffalo_l` or `buffalo_s` model pack is available. The corresponding ONNX 
          model file should be specified in the `onnx.load` function.
        - The `FaceAnalysis` class and `onnx2pytorch` library should be properly installed and configured.

    Returns:
        tuple:
            - model (torch.nn.Module): The converted PyTorch model.
            - app (FaceAnalysis): The initialized face recognition model.
    """
    # There are 'buffalo_l' and 'buffalo_s' model packs available. The corresponding ONNX model files are different.
    # You can modify the model pack name in the `constant` of `FaceAnalysis` initialization if needed.
    model_pack_name = 'buffalo_l'
    
    # Initialize the face recognition model
    app = FaceAnalysis(model_pack_name=model_pack_name)
    app.prepare(ctx_id=2, det_size=(640, 640))
    
    # Load the ONNX model file
    onnxm = onnx.load('/path/to/your/files/*.onnx')
    
    # Convert the ONNX model to a PyTorch model
    model = onnx2pytorch.ConvertModel(onnxm)

    return model, app


def main():
    """
    Main function for extracting features and generating adversarial samples with transformations.

    This function performs the following steps:
    1. Initializes the face recognition model and the adversarial attack.
    2. Extracts features from images in a dataset.
    3. Generates adversarial samples using the RAA (Randomized Adversarial Attack) method.
    4. Applies various transformations to the adversarial samples.
    5. Saves the original, adversarial, and transformed images.
    6. Computes and outputs the similarity between the original and transformed features.

    Returns:
        None
    """
    features = []
    feats_raa_img = []
    max_num = 100
    cnt = 0
    device = "cuda:0"

    input_size, img_align_celeba_dir = load_dataset()
    model, app = load_models()
    transform_dict = {
        # Neighbor transformations
        'adjust_brightness': [0.05, 0.1, 0.25, 0.5],
        'adjust_contrast': [0.05, 0.1, 0.25, 0.5],
        'JPEG_transform': [30, 50, 70, 90],
        'Gaussian_blur': [[5, 1.0]],
        # Non-neighbor transformations
        'resize_image': [0.5, 0.75, 0.9, 1.1, 1.25, 1.5, 2],
        'rotate_image': [0, 1, 2, 5, 10, 15, 20, 30, 45],
        'Random_perspective': [[0.25, 1.0], [0.5, 1.0]],
        'Ranom_Affine': [[10, [0.05, 0.05], [0.9, 0.9]]],
    }
    attacker = transferattack.load_attack_class('raa_face')(model_name=model, targeted=False, loss='pairwise', device=device)

    # Extract features
    for img_name in os.listdir(img_align_celeba_dir):
        features_raa = []
        cnt += 1
        if cnt > max_num:
            break
        img_path = os.path.join(img_align_celeba_dir, img_name)
        
        # Ensure the file is an image
        if not img_name.lower().endswith(('.png', '.jpg', '.jpeg')):
            continue
        
        # Load the image using cv2
        img = cv2.imread(img_path)
        if img is None:
            continue
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # Convert color space to RGB
        img = cv2.resize(img, input_size)

        # Detect faces and extract features
        feats = app.get(img)  # Directly obtain the feature list
        if feats:
            feat = feats[0].embedding  

        perturbations, _ = attacker.feat_forward(img, feat)
        new_img = 255 * perturbations + 255 * transforms.ToTensor()(img).to(device)
        
        for transform in transform_dict.keys():
            transform_name = transform
            par_list = transform_dict[transform]
            for par in par_list:
                par_name = par

                transform = getattr(Transform, transform_name)
                trans_img = transform(new_img.to(torch.uint8), par_name)

                if transform_name == "Ranom_Affine":
                    adv_test = reshape_image(trans_img).squeeze(0)
                else:
                    adv_test = trans_img.squeeze(0)

                feats_raa = app.get(adv_test.cpu().permute(1, 2, 0).numpy())
                if feats_raa:
                    feat_raa = feats_raa[0].embedding 
                    features_raa.append(feat_raa)

        # Save the original, adversarial, and transformed images
        if len(features_raa) >= 2:
            features.append(feat)
            feats_raa_img.append(features_raa)

        img_pil = Image.fromarray(img)
        new_img_pil = Image.fromarray(new_img.squeeze(0).cpu().numpy().transpose(1, 2, 0).astype(np.uint8))
        trans_img_pil = Image.fromarray(adv_test.cpu().numpy().transpose(1, 2, 0).astype(np.uint8))

        # Save images
        save_image(img_pil, "./pic/face", f"dim_original_{img_name}")
        save_image(new_img_pil, "./pic/face", f"dim_new_{img_name}")
        save_image(trans_img_pil, "./pic/face", f"dim_transformed_{img_name}")

        # Create a horizontal collage
        total_width = img_pil.width * 3 + 20  # Total width of three images plus two 10px gaps
        max_height = max(img_pil.height, new_img_pil.height, trans_img_pil.height)
        combined_img = Image.new('RGB', (total_width, max_height), (255, 255, 255))  # White background

        combined_img.paste(img_pil, (0, 0))
        combined_img.paste(new_img_pil, (img_pil.width + 10, 0))
        combined_img.paste(trans_img_pil, (img_pil.width * 2 + 20, 0))

        save_image(combined_img, "./pic/face", f"dim_combined_{img_name}")

        print(f'{img_name} extract done, {cnt}/{max_num} \n')

    # Number of feature vectors and labels
    num_features = len(features)
    print(f'Total number of features: {num_features}')

    # Compute and output similarity (example: compute similarity between the first and second feature vectors)
    if num_features >= 1:
        feats_raa_img = np.array(feats_raa_img).transpose(1, 0, 2).tolist()

        # Compute cosine similarity
        for trans in feats_raa_img:
            correct = 0
            for i in range(len(features)):
                feat1 = features[i]
                feat2 = trans[i]
                cos = np.dot(feat1, feat2) / (np.linalg.norm(feat1) * np.linalg.norm(feat2))
                print(cos)
                if cos <= 0.4:
                    correct += 1
                # Output similarity and corresponding labels
                
            print(f'Untarget Attack ASR: {correct / len(features)}')
            with open('face_results_eval.txt', 'a') as f:
                f.write(f'Untarget Attack ASR: {correct / len(features)} \n')

if __name__ == '__main__':
    main()