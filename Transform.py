from PIL import Image
import torchvision
import torchvision.transforms as T
from torchvision.transforms import functional as F
import numpy as np
import math
import torch
from io import BytesIO


def rotate_image(image, angle_degrees):
    """
    Rotate the image.

    Args:
        image (tensor): The input image.
        angle_degrees (int): The angle in degrees, clockwise rotation.

    Returns:
        tensor: The rotated image.
    """
    return F.rotate(image, angle_degrees)


def adjust_contrast(image, factor):
    """
    Adjust the contrast of the image.

    Args:
        image (tensor): The input image.
        factor (float): Non-negative float, 0 gives a solid gray image, 1 gives the original image, and increasing it increases the contrast.

    Returns:
        tensor: The image with adjusted contrast.
    """
    return F.adjust_contrast(image, factor)


def adjust_brightness(image, delta):
    """
    Adjust the brightness of the image.

    Args:
        image (tensor): The input image.
        delta (float): Non-negative float, 0 gives a solid black image, 1 gives the original image, and increasing it increases the brightness.

    Returns:
        tensor: The image with adjusted brightness.
    """
    return F.adjust_brightness(image, delta)


def adjust_saturation(image, saturation_factor):
    """
    Adjust the saturation of the image.

    Args:
        image (tensor): The input image.
        saturation_factor (float): Non-negative float, 0 gives a grayscale image, 1 gives the original image, and increasing it increases the saturation.

    Returns:
        tensor: The image with adjusted saturation.
    """
    return F.adjust_saturation(image, saturation_factor)


def resize_image(image, target_size_scale):
    """
    Resize the image.

    Args:
        image (tensor): The input image.
        target_size_scale (float): Non-negative float, scales the image size, <1 for downsizing and >1 for upsizing.

    Returns:
        tensor: The resized image.
    """
    img_shape = image.shape
    new_shape = [int(img_shape[-2] * target_size_scale), int(img_shape[-1] * target_size_scale)]
    resize = torchvision.transforms.Resize(new_shape)
    return resize(image)


def centercrop_image(image, target_size_scale):
    """
    Center crop the image.

    Args:
        image (tensor): The input image.
        target_size_scale (float): A float between 0 and 1, representing the proportion of the cropped image size.

    Returns:
        tensor: The center-cropped image.
    """
    img_shape = image.shape
    new_shape = [int(img_shape[-2] * target_size_scale), int(img_shape[-1] * target_size_scale)]
    return F.center_crop(image, new_shape)


def random_color(image, setting=[0.0, 0.0, 0.0, 0.0]):
    """
    Randomly adjust the brightness, contrast, saturation, and hue of the image within the given ranges.

    Args:
        image (tensor): The input image.
        brightness (float, optional): Non-negative, specifies the range of brightness changes. Defaults to 0.0.
        contrast (float, optional): Non-negative, specifies the range of contrast changes. Defaults to 0.0.
        saturation (float, optional): Non-negative, specifies the range of saturation changes. Defaults to 0.0.
        hue (float, optional): Specifies the range of hue changes. Defaults to 0.0.
    """
    args = setting
    jitter = T.ColorJitter(brightness=args[0], contrast=args[1], saturation=args[2], hue=args[3])
    return jitter(image)


def Gaussian_blur(image, setting=[5, 1.0]):
    """
    Apply Gaussian blur to the image.

    Args:
        image (tensor): The input image.
        kernel_size (int or sequence, optional): The size of the Gaussian kernel. Defaults to 5.
        sigma (float or sequence, optional): The standard deviation of the Gaussian kernel. Defaults to 1.0.
    """
    args = setting
    blurrer = T.GaussianBlur(kernel_size=args[0], sigma=args[1])
    return blurrer(image)


def Random_perspective(image, setting=[0.5, 0.5]):
    """
    Apply random perspective transformation to the image.

    Args:
        image (tensor): The input image.
        distortion_scale (float, optional): The degree of perspective transformation. Defaults to 0.5.
        p (float, optional): The probability of applying the transformation. Defaults to 0.5.
    """
    args = setting
    perspective_transformer = T.RandomPerspective(distortion_scale=args[0], p=args[1])
    return perspective_transformer(image)


def Random_Affine(image, setting=[0, (0, 0), (1, 1)]):
    """
    Apply random affine transformation to the image.

    Args:
        image (tensor): The input image.
        degrees (int or float or sequence, optional): The range of rotation angles. Defaults to 0.
        translate (tuple, optional): The range of translation distances. Defaults to (0, 0).
        scale (tuple, optional): The range of scaling factors. Defaults to (1, 1).
    """
    args = setting
    affine_transformer = T.RandomAffine(degrees=args[0], translate=args[1], scale=args[2])
    return affine_transformer(image)


def JPEG_transform(image, quality=100):
    """
    Apply JPEG compression to the image.

    Args:
        image (tensor): The input image.
        quality (sequence or number, optional): The compression quality, ranging from 1 to 100. Defaults to 100.
    """
    # Convert Tensor to PIL Image
    image = image.cpu()
    image = image.permute(0, 2, 3, 1).detach().numpy()  # Convert to (batch_size, H, W, C)

    # If quality is a tuple, randomly select a value
    if isinstance(quality, tuple):
        min_quality, max_quality = quality
        quality = np.random.randint(min_quality, max_quality + 1)

    # Initialize an empty list to store the compressed images
    compressed_images = []

    for img_array in image:
        # Convert values from [0, 1] to [0, 255]
        img_array = (img_array * 255).astype(np.uint8)
        img = Image.fromarray(img_array)

        # Save as JPEG format and reload
        bytes_io = BytesIO()
        img.save(bytes_io, format='JPEG', quality=quality)
        img = Image.open(bytes_io)

        # Convert PIL Image back to Tensor
        img_array = np.array(img)
        img_array = img_array.transpose((2, 0, 1))  # Convert to (C, H, W)
        img_tensor = torch.from_numpy(img_array).float() / 255.0  # Convert back to [0, 1] range

        compressed_images.append(img_tensor)

    # Convert the list of compressed images to a Tensor
    compressed_images = torch.stack(compressed_images)

    return compressed_images.to("cuda:0")