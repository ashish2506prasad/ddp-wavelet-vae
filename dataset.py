import os
import glob
import numpy as np
import PIL
import torchvision.transforms as T
from torch.utils.data import Dataset
from torchvision.datasets import ImageFolder
from torchvision.datasets.folder import default_loader
from torchvision import transforms
import torch
from PIL import Image
from glob import glob
# from pytorch_wavelets import DWTForward, DWTInverse  # Importing DWT modules    

def center_crop_arr(pil_image, image_size):
    """
    Center cropping implementation from ADM.
    https://github.com/openai/guided-diffusion/blob/8fb3ad9197f16bbc40620447b2742e13458d2831/guided_diffusion/image_datasets.py#L126
    """
    while min(*pil_image.size) >= 2 * image_size:
        pil_image = pil_image.resize(
            tuple(x // 2 for x in pil_image.size), resample=Image.BOX
        )

    scale = image_size / min(*pil_image.size)
    pil_image = pil_image.resize(
        tuple(round(x * scale) for x in pil_image.size), resample=Image.BICUBIC
    )

    arr = np.array(pil_image)
    crop_y = (arr.shape[0] - image_size) // 2
    crop_x = (arr.shape[1] - image_size) // 2
    return Image.fromarray(arr[crop_y: crop_y + image_size, crop_x: crop_x + image_size])

def compute_global_stats(parent_dir, split='train', feature_type='image', num_dwt_levels=1):
    image_paths = sorted(glob(f"{parent_dir}/{split}/{feature_type}_{num_dwt_levels}_dwt_LL/*.npy"))
    all_pixels = []
    
    for path in image_paths:
        image = np.load(path)
        all_pixels.append(image.flatten())
    
    all_pixels = np.concatenate(all_pixels)
    global_mean = all_pixels.mean()
    global_std = all_pixels.std()
    
    print(f"Global mean: {global_mean}, Global std: {global_std}")
    return global_mean, global_std

class CustomDataset(Dataset):
    """
    Custom dataset to load images from a directory structure.
    Expects images to be in subdirectories of the parent directory.
    Args:
        parent_dir (str): Path to the parent directory containing image subdirectories.
    Returns
        torch.Tensor: Transformed image tensor.
        does not return class as they are not needed for feature extraction
    """
    
    def __init__(self, parent_dir, test_data=None, split='train', feature_type ='image', num_dwt_levels = 1, feature_size=128, global_mean=None, global_std=None):
        self.parent_dir = parent_dir
        # Fixed: Use parent_dir parameter instead of hardcoded path
        self.num_dwt_levels = num_dwt_levels
        self.feature_size = feature_size
        if split in ['train', 'val']:
            self.image_paths = sorted(glob(f"{parent_dir}/{split}/{feature_type}_{num_dwt_levels}_dwt_LL/*.npy"))
            print(f"Found {len(self.image_paths)} images in {split} set")
        else:
            # Use test_data if provided, otherwise use parent_dir
            print(test_data)
            test_path = test_data if test_data is not None else parent_dir
            classes = os.listdir(test_path)
            self.image_paths = []
            for class_ in classes:
                class_path = os.path.join(test_path, class_)
                sorted_images = sorted(glob(f"{class_path}/*/*.[Jj][Pp][Ee][Gg]"))  # matches .JPEG/.jpeg
                print(f"Found {len(sorted_images)} images in class {class_}")
                self.image_paths += sorted_images[int(len(sorted_images)*0.8):]  # Use 20% of images for testing
            print(f"Found {len(self.image_paths)} images in {split} set ")

        self.global_mean = global_mean
        self.global_std = global_std

    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        # label_path = self.label_paths[idx] if hasattr(self, 'label_paths') else None
        
        # Check if this is a test split with image files or train/val split with .npy files
        if img_path.lower().endswith(('.jpg', '.jpeg', '.png')):
            # Load image file for test data
            image = Image.open(img_path).convert('RGB')
            # Convert to tensor and normalize
            transform = transforms.Compose([
                transforms.Resize((self.feature_size*(2**self.num_dwt_levels), self.feature_size*(2**self.num_dwt_levels))),  # Adjust size as needed
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])  # Normalize to [-1, 1]
            ])
            image = transform(image)
        else:
            # Load .npy file for train/val data
            image = np.load(img_path)
            image = torch.from_numpy(image).float()
            
            # Use global normalization instead of per-image
            if self.global_mean is not None and self.global_std is not None:
                image = (image - self.global_mean) / (self.global_std + 1e-8)
            else:
                # Fallback to per-image normalization
                mean = image.mean()
                std = image.std() + 1e-8
                image = (image - mean) / std
            
            image = image.squeeze(0)
        
        return image, self.num_dwt_levels

if __name__ == "__main__":
    # Example usage
    # change parent directory as needed
    global_mean, global_std = compute_global_stats(parent_dir='D:\DDP_amit_sethi\workspace-wavelet-VAE\imagenet256', split='val', feature_type='image', num_dwt_levels=1)
    print(f"Global Mean: {global_mean}, Global Std: {global_std}")