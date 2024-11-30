import torch
import torchvision.transforms as transforms
import random
import glob
import os
from PIL import Image
from torchvision.transforms.functional import to_pil_image
# import cv2
import time
from models import sequence_models, contrast_learning, spatio_temporal_model
import numpy as np
import io
import consts

# def load_model(epo_offset, model_save_folder, device, only_encoder=False, is_conv1_2d=False):
#     encoder_config_path =  find_first_json_with_prefix(model_save_folder, "SeqModel")

#     if is_conv1_2d:
#         encoder = spatio_temporal_model.ResNet_R2Plus1D_LN.load_from_config(encoder_config_path).to(device)
#     else:
#         encoder = sequence_models.EfficientSequenceModel.load_from_config(encoder_config_path).to(device)

#     encoder_path = os.path.join(model_save_folder, f"{encoder.model_name}_{epo_offset}.pth") 
#     encoder.load_state_dict(torch.load(encoder_path, map_location=device))

#     # print(encoder_path)

#     if only_encoder: return encoder

#     projector_config_path = find_first_json_with_prefix(model_save_folder, "Projector")
#     projector = contrast_learning.MLPs.load_from_config(projector_config_path).to(device)

#     projector_path = os.path.join(model_save_folder, f"{projector.model_name}_{epo_offset}.pth") 

#     projector.load_state_dict(torch.load(projector_path, map_location=device))

#     # print(projector_path)

#     return encoder, projector


def load_model(epo_offset, model_save_folder, device, only_encoder=False, is_conv1_2d=False):
    encoder_config_path =  find_first_json_with_prefix(model_save_folder, "SeqModel")

    if is_conv1_2d:
        encoder = spatio_temporal_model.ResNet_R2Plus1D_LN.load_from_config(encoder_config_path).to(device)
    else:
        encoder = sequence_models.EfficientSequenceModel.load_from_config(encoder_config_path).to(device)

    encoder_path = os.path.join(model_save_folder, f"{encoder.model_name}_{epo_offset}.pth") 
    encoder.load_state_dict(torch.load(encoder_path, map_location=device))

    print(encoder_path)

    if only_encoder: return encoder

    projector_config_path = find_first_json_with_prefix(model_save_folder, "Projector")
    projector = contrast_learning.MLPs.load_from_config(projector_config_path).to(device)

    projector_path = os.path.join(model_save_folder, f"{projector.model_name}_{epo_offset}.pth") 

    projector.load_state_dict(torch.load(projector_path, map_location=device))

    print(projector_path)

    return encoder, projector




def init_log(loss_keys):
    loss_log = {}

    ds_keys = list(consts.dataset_names.values()) + ["Average"]
    best_epo = {"epo_n": 0, "tr_time": 0, "tr": {key: float("inf") for key in loss_keys}, "tt": {ds: {} for ds in ds_keys}}

    for ds in ds_keys:
        best_epo["tt"][ds] = {key: consts.eva_metrics[key] for key in consts.eva_metrics}


    loss_log["best_epo"] = best_epo


    for lk in loss_keys:
        loss_log[lk] = []

    return loss_log


def clear_log_after(loss_log, epo_offset, loss_keys):
    for lk in loss_keys:
        loss_log[lk] = loss_log[lk][:epo_offset+1]
    return loss_log



def mini_fmt_loss_equ(loss_parts):
  # selected_keys = ["mse", "mae", "loss"]

  selected_keys = list(loss_parts.keys())


  parts = [f"{v:>7.4f}~{k}" for k, v in loss_parts.items() if k in selected_keys]
  ret = " + ".join(parts)
  # ret = ret.replace(f"{selected_keys[-2]} +", f"{selected_keys[-2]} =")
  # ret = ret.replace(f"{selected_keys[5]} +", f"{selected_keys[5]} +\n")
  return ret




def find_first_json_with_prefix(folder_path, prefix):
    """
    Find the first JSON file in a folder that starts with a given prefix.

    Args:
    - folder_path (str): Path to the folder to search in.
    - prefix (str): The prefix of the file name.

    Returns:
    - str: The path to the first matching JSON file, or None if no match is found.
    """
    search_pattern = os.path.join(folder_path, f"{prefix}*.json")
    matching_files = glob.glob(search_pattern)

    if matching_files:
        return matching_files[0]  # Return the first matching file
    else:
        return None


def save_concatenated_image_sequences(sequence1, sequence2, folder, filename_prefix='concat_seq_'):
    """
    Concatenate two sequences of images side by side and save them to a specified folder.

    Args:
    - sequence1 (Tensor): Image sequence tensor of shape [num_images, channels, height, width].
    - sequence2 (Tensor): Image sequence tensor of shape [num_images, channels, height, width].
    - folder (str): Path to the folder where concatenated images will be saved.
    - filename_prefix (str): Prefix for filenames of concatenated images.
    """
    if not os.path.exists(folder):
        os.makedirs(folder)

    for i in range(len(sequence1)):
        # Convert tensors to PIL images
        img1 = to_pil_image(sequence1[i])
        img2 = to_pil_image(sequence2[i])

        # Get dimensions for the new image (widths added together, height is the max height of the two)
        total_width = img1.width + img2.width
        max_height = max(img1.height, img2.height)

        # Create a new image with the appropriate height and width
        new_img = Image.new('RGB', (total_width, max_height))

        # Paste the two images side by side
        new_img.paste(img1, (0, 0))
        new_img.paste(img2, (img1.width, 0))

        # Save the concatenated image
        new_img.save(os.path.join(folder, f'{filename_prefix}{i:05d}.png'))

        
def numpy_image_to_tensor(image):
    """
    Convert a numpy array of an image to a PyTorch tensor and normalize it.

    Parameters:
    - image: A numpy array representing an image with pixel values in 0-255.

    Returns:
    - A PyTorch tensor of the image normalized to 0-1.
    """
    # Check if the image has three dimensions (H, W, C)
    # if image.ndim == 3:
    # Change the dimension order from HWC to CHW
    image = np.transpose(image, (2, 0, 1))
    image = image.astype(np.float32) / 255.0 
    # print(image)
    # image_tensor = image
    image_tensor = torch.from_numpy(image).float()  # Convert to float tensor
    # image_tensor = image_tensor.float()
    # image_tensor /= 255.0  # Normalize to the range [0, 1]

    # print(image_tensor.shape)

    return image_tensor


def random_transform(image, crop_offset, crop_size, h_flip, v_flip, rotate):
    """
    Apply specified transformations to a PyTorch tensor image.
    """
    # Crop
    image = transforms.functional.crop(image, *crop_offset, *crop_size)

    # Horizontal flip
    if h_flip:
        image = transforms.functional.hflip(image)

    # Vertical flip
    if v_flip:
        image = transforms.functional.vflip(image)

    # Rotate
    if rotate:
        image = transforms.functional.rotate(image, 90)

    return image

def generate_random_params(image_size, crop_size):
    """
    Generate random parameters for transformations.
    """
    height, width  = image_size
    crop_height, crop_width = crop_size
    # print(height, width, crop_height, crop_width)
    x = random.randint(0, width - crop_width)
    y = random.randint(0, height - crop_height)
    crop_offset = (y, x)
    h_flip = random.choice([True, False])
    v_flip = random.choice([True, False])
    # rotate = random.choice([True, False])
    return crop_offset, crop_size, h_flip, v_flip#, rotate

def read_aug_image(img_path, transform_params, noise_key, img_mode, device):
    """
    Read and augment an image using PyTorch.
    """
    img = Image.open(img_path)#.convert('RGB')

    img = apply_random_noise(img, noise_key, img_mode)

    # img_tensor = transforms.ToTensor()(img).to(device)
    img_tensor = numpy_image_to_tensor(img).to(device)

    # print(img_tensor)
    img_tensor = random_transform(img_tensor, *transform_params)
    return img_tensor

def count_image_files(folder_path):
    """
    Count the number of image files (.png or .jpg) in a given folder.

    Args:
    - folder_path (str): The path to the folder.

    Returns:
    - int: The number of image files (.png or .jpg) in the folder.
    """
    # Initialize the count of image files
    image_count = 0

    # List all files in the given folder
    for filename in os.listdir(folder_path):
        # Check if the file is an image (.png or .jpg)
        if filename.endswith('.png') or filename.endswith('.jpg'):
            image_count += 1

    return image_count


def load_sequence_segment(load_folder, method1, method2, segment_length, transform_params, noise_keys, img_mode, replace_ratio, device):
    """
    Load a sequence segment of images.
    """
    
    images_png = glob.glob(os.path.join(load_folder, '*.png'))
    images_jpg = glob.glob(os.path.join(load_folder, '*.jpg'))
    all_images = sorted(images_png+images_jpg)

    # print(all_images)


    start_idx = random.randint(0, len(all_images) - segment_length)
    segment_images_1 = all_images[start_idx:start_idx + segment_length]
    segment_images_2 = [img_path.replace(method1, method2) for img_path in segment_images_1]


    i_to_replace = random.sample(range(segment_length), int(replace_ratio*segment_length))


    seg_1, seg_2 = [], []
    img_i = 0
    for img_path_1, img_path_2 in zip(segment_images_1, segment_images_2):
        img_1 = read_aug_image(img_path_1, transform_params,noise_keys[0], img_mode, device)

        if img_i in i_to_replace:
            img_2 = read_aug_image(img_path_2, transform_params, noise_keys[1], img_mode, device)
        
        else:
            img_2 = img_1#.clone()

        seg_1.append(img_1)
        seg_2.append(img_2)

    # print(load_folder)
    scene_method = load_folder.split("/")[-2].replace(method2, method1) + "_" + method2
    # print(scene_method)
    
    # save_concatenated_image_sequences(seg_1, seg_2, f"seg_seqs/{scene_method}")
    return seg_1, seg_2


def img_to_lum_np(img):
    lum = img.convert('L')
    np_img = np.array(lum)
    np_img = np.expand_dims(np_img, axis=-1)
    return np_img



def mask_random_black_border(image, max_border_size, random_border):
    """
    Masks a black border with random sizes directly onto the original image without changing its size.

    Parameters:
    - image: A numpy array of shape (H, W, C) representing the image.
    - max_border_size: The maximum thickness for the black border.

    Returns:
    - The original image numpy array with a black border masked onto it.
    """
    if random_border == True:

        height, width, _ = image.shape

        # Generate random border sizes for each side
        top_border = np.random.randint(0, min(max_border_size, height // 2))
        bottom_border = np.random.randint(0, min(max_border_size, height // 2))
        left_border = np.random.randint(0, min(max_border_size, width // 2))
        right_border = np.random.randint(0, min(max_border_size, width // 2))
        
        # Mask top and bottom borders
        if top_border > 0:
            image[:top_border, :, :] = 0
        if bottom_border > 0:
            image[-bottom_border:, :, :] = 0

        # Mask left and right borders
        if left_border > 0:
            image[:, :left_border, :] = 0
        if right_border > 0:
            image[:, -right_border:, :] = 0
    
    else:
        # Ensure border_size is not larger than the image size
        border_size = min(border_size, image.shape[0] // 2, image.shape[1] // 2)
        
        # Mask top and bottom borders
        image[:border_size, :, :] = 0
        image[-border_size:, :, :] = 0

        # Mask left and right borders
        image[:, :border_size, :] = 0
        image[:, -border_size:, :] = 0


    return image



def apply_random_noise(img, key, img_mode):
    """
    Apply random noise patterns to an image based on a given key.
    
    Parameters:
    - img: PIL.Image object.
    - key: An integer key to determine the noise pattern.
    """


    if img_mode.upper() == 'LUM':
        # Convert the image to YCbCr color space and extract the Y (luminance) channel
        # lum = img.convert('YCbCr').split()[0]
        np_img = img_to_lum_np(img)
    else:

        np_img = np.array(img)

    # print(np_img.shape)
    
    # Normalize the key to get a range of options
    option = key #% 5
    
    if option == 0:
        # Gaussian Noise
        mean = 0
        var = 0.1
        sigma = var ** 0.5
        gauss = np.random.normal(mean, sigma, np_img.shape)
        noisy = np_img + gauss
    elif option == 1:
        # Salt and Pepper Noise
        s_vs_p = 0.5
        amount = 0.04
        out = np.copy(np_img)
        # Salt mode
        num_salt = np.ceil(amount * np_img.size * s_vs_p)
        # print(num_salt, np_img.shape)
        coords = [np.random.randint(0, i, int(num_salt))
                  for i in np_img.shape]
        out[tuple(coords)] = 1

        # Pepper mode
        num_pepper = np.ceil(amount * np_img.size * (1. - s_vs_p))
        coords = [np.random.randint(0, i, int(num_pepper))
                  for i in np_img.shape]
        out[tuple(coords)] = 0
        noisy = out
    elif option == 2:
        # Speckle Noise
        gauss = np.random.randn(*np_img.shape)
        noisy = np_img + np_img * gauss
    elif option == 3:
        # Poisson Noise
        vals = len(np.unique(np_img))
        vals = 2 ** np.ceil(np.log2(vals))
        noisy = np.random.poisson(np_img * vals) / float(vals)
    elif option == 4:
        # Multiplicative Noise (from previous example)
        noise = np.random.normal(1.0, 0.05, np_img.shape)
        noisy = np_img * noise
    
    elif option == 5:
        # Simulating JPEG Compression Artifacts
        # Compress the image with a random quality factor
        quality_factor = random.randint(10, 75)  # Lower quality factor -> more artifacts
        buffer = io.BytesIO()
        img.save(buffer, format="JPEG", quality=quality_factor)
        buffer.seek(0)
        noisy_img = Image.open(buffer)
        if img_mode.upper() == 'LUM':
            noisy_img = img_to_lum_np(noisy_img)
        return noisy_img  # Directly return the compressed image as PIL image

    # elif option <= 8:
    #     noisy = mask_random_black_border(np_img, max_border_size=50, random_border=True)

    else:
        noisy = np_img  # Fallback case, should not happen

    # Ensure the noisy image is within proper bounds
    noisy = np.clip(noisy, 0, 255).astype(np.uint8)
    
    # Convert back to PIL image
    # noisy_img = Image.fromarray(noisy)
    noisy_img = noisy
    
    return noisy_img


