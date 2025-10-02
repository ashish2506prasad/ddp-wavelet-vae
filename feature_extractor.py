# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""
A minimal training script for DiT using PyTorch DDP.
"""
import torch
# the first flag below was False when we tested this script but True makes A100 training a lot faster:
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from torchvision.datasets import ImageFolder
from torchvision import transforms
import numpy as np
from collections import OrderedDict
from PIL import Image
from copy import deepcopy
from glob import glob
from time import time
import argparse
import logging
import os
from pytorch_wavelets import DWTForward, DWTInverse
import os
from torch.utils.data import Dataset
from PIL import Image


#################################################################################
#                             Training Helper Functions                         #
#################################################################################

@torch.no_grad()
def update_ema(ema_model, model, decay=0.9999):
    """
    Step the EMA model towards the current model.
    """
    ema_params = OrderedDict(ema_model.named_parameters())
    model_params = OrderedDict(model.named_parameters())
    
    for name, param in model_params.items():
        # TODO: Consider applying only to params that require_grad to avoid small numerical changes of pos_embed
        ema_params[name].mul_(decay).add_(param.data, alpha=1 - decay)

def requires_grad(model, flag=True):
    """
    Set requires_grad flag for all parameters in a model.
    """
    for p in model.parameters():
        p.requires_grad = flag

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

def extract_dwt_features(latent, num_dwt_levels=1, device='cpu'):
    dwt = DWTForward(J=num_dwt_levels, wave='haar', mode='zero').to(device)
    ll, h = dwt(latent)
    return ll, h
#################################################################################
#                                  Custom Dataset                               #
#################################################################################
class CustomImageFolder(Dataset):
    def __init__(self, root_dir, transform, split_name = 'val'):
        self.root_dir = root_dir
        self.transform = transform
        # Look for images in images/ subdirectory
        images_dir = os.path.join(root_dir, 'images')
        if os.path.exists(images_dir):
            self.image_paths = sorted(glob(os.path.join(images_dir, '*.JPEG')) + 
                                    glob(os.path.join(images_dir, '*.jpeg')))
        else:
            # Fallback: look directly in root_dir
            self.image_paths = sorted(glob(os.path.join(root_dir, '*.JPEG')) + 
                                    glob(os.path.join(root_dir, '*.jpeg')))
        print(f"Found {len(self.image_paths)} images for {split_name} split")
    
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        image = Image.open(img_path).convert('RGB')
        image = self.transform(image)
        # Return dummy label since we don't have class structure
        return image, -1  # -1 indicates no label


#################################################################################
#                                  Training Loop                                #
#################################################################################

# Replace the distributed setup section with this simple fix:

def main(args):
    assert torch.cuda.is_available(), "Training currently requires at least one GPU."

    # Force single GPU mode - no distributed processing
    distributed = False
    world_size = 1
    rank = 0
    device = 0  # Use GPU 0
    seed = args.global_seed
    num_dwt_levels = args.num_dwt_levels

    torch.manual_seed(seed)
    torch.cuda.set_device(device)
    print(f"Starting single GPU mode: rank={rank}, seed={seed}, world_size={world_size}.")

    for split in ['train', 'val']:
        # Setup feature folder
        args.split = split
        if split == 'train':
            split_data_path = os.path.join(args.data_path, 'train')
        else:  # val
            split_data_path = os.path.join(args.data_path, 'val')
        print(f"Processing {split} set...")
        os.makedirs(args.features_path, exist_ok=True)
        os.makedirs(os.path.join(args.features_path, f'imagenet{args.image_size}_{num_dwt_levels}_dwt_features/{args.split}/low_freq'), exist_ok=True)
        os.makedirs(os.path.join(args.features_path, f'imagenet{args.image_size}_{num_dwt_levels}_dwt_features/{args.split}/high_freq'), exist_ok=True)
        os.makedirs(os.path.join(args.features_path, f'imagenet{args.image_size}_{num_dwt_levels}_dwt_labels/{args.split}'), exist_ok=True)

        # Setup data - no distributed sampler
        transform = transforms.Compose([
            transforms.Lambda(lambda pil_image: center_crop_arr(pil_image, args.image_size)),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5]*3, std=[0.5]*3, inplace=True)
        ])
        if split == 'train':
            dataset = ImageFolder(split_data_path, transform=transform)
        if split == 'val':
            dataset = CustomImageFolder(split_data_path, transform=transform)
        # Simple DataLoader without distributed sampling
        loader = DataLoader(
            dataset,
            batch_size=args.global_batch_size,  # Use full batch size since no distribution
            shuffle=True,  # Enable shuffle for better class distribution
            num_workers=args.num_workers,
            pin_memory=True,
            drop_last=True
        )

        # Rest of your training loop remains the same
        train_steps = 0
        for x, y in loader:
            x = x.to(device)
            y = y.to(device)
            x, h = extract_dwt_features(x, num_dwt_levels=num_dwt_levels, device=device)
            x = x.detach().cpu().numpy()
            y = y.detach().cpu().numpy()

            # Convert list of tensors to list of numpy arrays
            h_numpy = [level.detach().cpu().numpy() for level in h]
            h_numpy = h_numpy[-1]  # retain the last feature

            for i in range(x.shape[0]):
                np.save(f'{args.features_path}/imagenet{args.image_size}_{num_dwt_levels}_dwt_features/{split}/low_freq/{train_steps}.npy', x[i:i+1])
                # Save as a list of arrays (one array per DWT level)
                # h_sample = [level[i:i+1] for level in h_numpy]
                np.save(f'{args.features_path}/imagenet{args.image_size}_{num_dwt_levels}_dwt_features/{split}/high_freq/{train_steps}.npy', h_numpy[i:i+1])
                np.save(f'{args.features_path}/imagenet{args.image_size}_{num_dwt_levels}_dwt_labels/{split}/{train_steps}.npy', y[i:i+1])
                train_steps += 1
                        
            if train_steps % 100 == 0:
                print(f"Processed {train_steps} samples")
        
        import zipfile
        with zipfile.ZipFile(f'{args.features_path}/imagenet{args.image_size}_{num_dwt_levels}_dwt_features_{args.split}.zip', 'w') as zipf:
            for file in glob(f'{args.features_path}/imagenet{args.image_size}_{num_dwt_levels}_dwt_features/{args.split}/*/*.npy'):
                zipf.write(file, os.path.relpath(file, args.features_path))
        
        # Only zip labels if they exist (train split)
        label_files = glob(f'{args.features_path}/imagenet{args.image_size}_{num_dwt_levels}_dwt_labels/{args.split}/*.npy')
        if label_files:
            with zipfile.ZipFile(f'{args.features_path}/imagenet{args.image_size}_{num_dwt_levels}_dwt_labels_{args.split}.zip', 'w') as zipf:
                for file in label_files:
                    zipf.write(file, os.path.relpath(file, args.features_path))

    print(f"Finished processing {train_steps} samples.")
    


if __name__ == "__main__":
    # Default args here will train DiT-XL/2 with the hyperparameters we used in our paper (except training iters).
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-path", type=str, required=True)
    parser.add_argument("--features-path", type=str, default="features")
    parser.add_argument("--results-dir", type=str, default="results")
    parser.add_argument("--image-size", type=int, choices=[256, 512, 1024], default=256)
    parser.add_argument("--global-batch-size", type=int, default=10)
    parser.add_argument("--global-seed", type=int, default=0)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--log-every", type=int, default=100)
    parser.add_argument("--num-dwt-levels", type=int, default=1, help="Number of DWT levels to use for feature extraction.")
    args = parser.parse_args()
    main(args)

    # Expected input directory structures:
    # - train: tiny-image-net-200/train/<class>/images/*.JPEG
    # - val:   tiny-image-net-200/val/images/*.JPEG  
    # - test:  tiny-image-net-200/test/images/*.JPEG
    
    # Output structure:
    # - features: features_path/imagenet{size}_{levels}_dwt_features/{split}/*.npy
    # - labels:   features_path/imagenet{size}_{levels}_dwt_labels/{split}/*.npy (train only)
    # - zips:     features_path/imagenet{size}_{levels}_dwt_features_{split}.zip
    #             features_path/imagenet{size}_{levels}_dwt_labels_{split}.zip (train only)