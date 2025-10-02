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
    
    def __init__(self, parent_dir, test_data=None, split='train', image_size = 256, feature_size=128):
        self.parent_dir = parent_dir
        # Fixed: Use parent_dir parameter instead of hardcoded path
        assert image_size % feature_size == 0, "Image size must be divisible by feature size."
        assert split in ['train', 'val', 'test'], "Split must be one of 'train', 'val', or 'test'."
        self.num_dwt_levels = int(np.log2(image_size // feature_size))
        self.feature_size = feature_size
        if split in ['train', 'val']:
            self.image_paths = sorted(glob(f"{parent_dir}/imagenet{image_size}_{self.num_dwt_levels}_dwt_features/{split}/low_freq/*.npy"))
            print(f"Found {len(self.image_paths)} images in {split} set")
        else:
            # Use test_data if provided, otherwise use parent_dir
            print(test_data)
            test_path = os.path.join(parent_dir, 'test') if test_data is None else test_data
            sorted_images = sorted(glob(f"{test_path}/*/*.[Jj][Pp][Ee][Gg]"))  # matches .JPEG/.jpeg
            self.image_paths = sorted_images
            print(f"Found {len(sorted_images)} images in {split} set")

    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        if img_path.lower().endswith(('.jpg', '.jpeg', '.png')):
            # Load image file for test data
            image = Image.open(img_path).convert('RGB')
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
            image = image/4   # Scale from [-4, 4] to [-1, 1] wavelet tranform of [-1,1] scales it's range to [-4,4]  
            image = image.squeeze(0) if image.dim() == 4 else image  
        
        return image, self.num_dwt_levels