import torch
import torch.nn.functional as F
from contextlib import contextmanager
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
import math
import torch
import torch.nn as nn
import numpy as np
# from einops import rearrange
from util import instantiate_from_config
import torch
import numpy as np
from autoencoder import AutoencoderKL
import yaml
import os
import argparse
from dataset import CustomDataset
from torch.utils.data import DataLoader
from pytorch_wavelets import DWTForward, DWTInverse
from torchvision.utils import save_image
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
import glob
from dataset import compute_global_stats

def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='Train AutoencoderKL')
    
    # Model architecture arguments
    parser.add_argument('--config-file', type=str, default=None, help='Path to config file (YAML)')
    parser.add_argument('--embed-dim', type=int, default=4, help='Embedding dimension for latent space')
    parser.add_argument('--double-z', type=bool, default=True, help='Use double z channels')
    parser.add_argument('--z-channels', type=int, default=4, help='Number of latent channels')
    parser.add_argument('--resolution', type=int, default=128, help='Image resolution')
    parser.add_argument('--in-channels', type=int, default=3, help='Number of input channels')
    parser.add_argument('--out-ch', type=int, default=3, help='Number of output channels')
    parser.add_argument('--ch', type=int, default=128, help='Base number of channels')
    parser.add_argument('--ch-mult', nargs='+', type=int, default=[1, 2, 4, 4], help='Channel multipliers for each resolution level')
    parser.add_argument('--num-res-blocks', type=int, default=2, help='Number of residual blocks')
    parser.add_argument('--attn-resolutions', nargs='+', type=int, default=[16], help='Resolutions to apply attention')
    parser.add_argument('--dropout', type=float, default=0.0, help='Dropout rate')
    
    # Loss configuration
    parser.add_argument('--loss-type', type=str, default='LPIPSWithDiscriminator', choices=['LPIPSWithDiscriminator', 'MSE', 'L1'], help='Type of loss function')
    parser.add_argument('--disc-conditional', type=bool, default=False, help='Use conditional discriminator')
    parser.add_argument('--disc-in-channels', type=int, default=3, help='Discriminator input channels')
    parser.add_argument('--disc-start', type=int, default=50001, help='Step to start discriminator training')
    parser.add_argument('--disc-weight', type=float, default=0.5, help='Discriminator loss weight')
    parser.add_argument('--codebook-weight', type=float, default=1.0, help='Codebook loss weight')
    parser.add_argument('--pixelloss-weight', type=float, default=1.0, help='Pixel reconstruction loss weight')
    parser.add_argument('--perceptual-weight', type=float, default=1.0, help='Perceptual loss weight')
    parser.add_argument('--kl-weight', type=float, default=0.000001, help='KL divergence loss weight')
    
    # Training arguments
    parser.add_argument('--learning-rate', type=float, default=4.5e-6, help='Learning rate')
    parser.add_argument('--batch-size', type=int, default=8, help='Batch size')
    parser.add_argument('--num-epochs', type=int, default=100, help='Number of training epochs')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu', help='Device to use for training')
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--where", type=str, choices=["local", "cluster"], default="cluster", help="Run locally or on cluster")
    # parser.add_argument('--num-workers', type=int, default=4, help='Number of data loading workers')
    # parser.add_argument('--global-batch-size', type=int, default=8, help='Global batch size for distributed training')
    parser.add_argument('--val-every', type=int, default=1, help='Validation frequency (epochs)')
    parser.add_argument('--num-dwt-levels', type=int, default=1, help='Number of DWT levels for feature extraction')
    parser.add_argument('--results-dir', type=str, default='./results', help='Directory to save results')
    
    # Data arguments
    parser.add_argument('--image-key', type=str, default='image', help='Key for image data in batch')
    parser.add_argument('--train-data-path', type=str, required=True, help='Path to training data')
    parser.add_argument('--val-data-path', type=str, default=None, help='Path to validation data')
    parser.add_argument('--test-data-path', type=str, required=True, help='Path to test data')
    
    # Checkpoint arguments
    parser.add_argument('--ckpt-path', type=str, default='./checkpoints', help='Path to checkpoint to resume from')
    # parser.add_argument('--checkpoint-dir', type=str, default='./checkpoints', help='Directory to save checkpoints')
    parser.add_argument('--save-freq', type=int, default=10, help='Checkpoint save frequency (epochs)')
    
    # Monitoring arguments
    parser.add_argument('--monitor', type=str, default='val/rec_loss', help='Metric to monitor for checkpointing')
    parser.add_argument('--colorize-nlabels', type=int, default=None, help='Number of labels for colorization')
    
    # Logging arguments
    parser.add_argument('--log-freq', type=int, default=100, help='Logging frequency (steps)')
    parser.add_argument('--log-images-freq', type=int, default=1000, help='Image logging frequency (steps)')

    # global mean and std for LL normalization
    parser.add_argument('--global_mean', type=float, default=None, help='Global mean for LL normalization')
    parser.add_argument('--global_std', type=float, default=None, help='Global std for LL normalization')

    return parser.parse_args()

def create_configs_from_args(args):
    """Create ddconfig and lossconfig from parsed arguments"""
    if args.config_file is not None:
        with open(args.config, 'r') as f:
            config = yaml.safe_load(f)
        ddconfig = config['model']['params']['ddconfig']
        lossconfig = config['model']['params']['lossconfig']
        return ddconfig, lossconfig
    
    ddconfig = {
        "double_z": args.double_z,
        "z_channels": args.z_channels,
        "resolution": args.resolution,
        "in_channels": args.in_channels,
        "out_ch": args.out_ch,
        "ch": args.ch,
        "ch_mult": args.ch_mult,
        "num_res_blocks": args.num_res_blocks,
        "attn_resolutions": args.attn_resolutions,
        "dropout": args.dropout
    }
    
    # Create loss config based on loss type
    if args.loss_type == 'LPIPSWithDiscriminator':
        lossconfig = {
            "target": "losses.LPIPSWithDiscriminator",  # Adjust import path as needed
            "params": {
                "disc_conditional": args.disc_conditional,
                "disc_in_channels": args.disc_in_channels,
                "disc_start": args.disc_start,
                "disc_weight": args.disc_weight,
                # "codebook_weight": args.codebook_weight,
                "pixelloss_weight": args.pixelloss_weight,
                "perceptual_weight": args.perceptual_weight,
                "kl_weight": args.kl_weight
            }
        }
    else:
        # Simple MSE or L1 loss config
        lossconfig = {
            "target": f"losses.{args.loss_type}Loss",
            "params": {
                "weight": args.pixelloss_weight
            }
        }
    
    return ddconfig, lossconfig

# Main training function
def train(args):
    feature_type = 'image'
    ddconfig, lossconfig = create_configs_from_args(args)
    num_dwt_levels = args.num_dwt_levels

    if args.where == "local":
        # Local single GPU setup
        distributed = False
        rank = 0
        world_size = 1
        device = 0
        torch.cuda.set_device(device)
        print("Running in local single GPU mode")
    else:
        # Distributed training setup
        dist.init_process_group(backend='nccl')
        rank = dist.get_rank()
        world_size = dist.get_world_size()
        device = rank
        torch.cuda.set_device(device)
        distributed = True
        print(f"Running in distributed mode: rank={rank}, world_size={world_size}")
    
    # Create model with args
    model = AutoencoderKL(ddconfig=ddconfig,
                          lossconfig=lossconfig,
                          embed_dim=args.embed_dim,
                          args=args,
                          ckpt_path=None,
                          ignore_keys=[],
                          colorize_nlabels=args.colorize_nlabels,
                          monitor=args.monitor,
                          device=args.device)
    
    model.count_parameters()
    model.print_parameter_summary()

    if args.where == "cluster":
        model = model.to(device)
        model = DDP(model, device_ids=[device])
        base_model = model.module
    else:
        model = model.to(device)
        base_model = model

    train_dataset = CustomDataset(args.train_data_path, split='train', feature_type=feature_type, num_dwt_levels=num_dwt_levels, global_mean=args.global_mean, global_std=args.global_std) 
    val_path = args.val_data_path if args.val_data_path else args.train_data_path
    val_dataset = CustomDataset(val_path, split='val', feature_type=feature_type, num_dwt_levels=num_dwt_levels, global_mean=args.global_mean, global_std=args.global_std)
    shuffle = True if args.where == "local" else None

    train_sampler = DistributedSampler(train_dataset, num_replicas=world_size, rank=rank) if args.where == "cluster" else None
    val_sampler = DistributedSampler(val_dataset, num_replicas=world_size, rank=rank) if args.where == "cluster" else None
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,  # Use full batch size since no distribution
        shuffle=shuffle,  # Enable shuffle for better class distribution
        sampler=train_sampler,
        num_workers=args.num_workers,  
        pin_memory=True,
        drop_last=True
    )

    val_dataloader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,  # Use full batch size since no distribution
        shuffle=shuffle,  # Enable shuffle for better class distribution
        sampler=val_sampler,
        num_workers=args.num_workers,  
        pin_memory=True,
        drop_last=False
    )
    model.to(args.device)

    # Training loop
    import os
    if rank == 0:
        os.makedirs(args.ckpt_path, exist_ok=True)
    if args.where == "cluster":
        dist.barrier()
    
    for epoch in range(args.num_epochs):
        if args.where == "cluster":
            train_sampler.set_epoch(epoch)
        if rank == 0 and (epoch % 1 == 0 or epoch == args.num_epochs - 1):
            print(f"\nEpoch {epoch+1}/{args.num_epochs}")
        
        # Training
        avg_ae_loss, avg_disc_loss = base_model.train_epoch(train_dataloader, epoch)
        if rank == 0:
            print(f"Average Training - AE Loss: {avg_ae_loss:.4f}, Disc Loss: {avg_disc_loss:.4f}")
        
        # Validation
        if epoch % args.val_every == 0 or epoch == args.num_epochs - 1:
            # Only set epoch for distributed training
            if args.where == "cluster" and val_sampler is not None:
                val_sampler.set_epoch(epoch)
            
            if val_dataloader:
                val_losses = base_model.validate_epoch(val_dataloader)
                if rank == 0:
                    print(f"Validation Losses: {val_losses}")
            else:
                val_losses = {}
        
        # Save checkpoint
        if (epoch + 1) % args.save_freq == 0:
            if rank == 0:
                checkpoint_path = os.path.join(args.ckpt_path, f"checkpoint_epoch_{epoch+1}.pt")
                # Add global statistics to checkpoint metadata
                checkpoint_metadata = {
                    "val_losses": val_losses,
                    "global_mean": args.global_mean,
                    "global_std": args.global_std,
                    "num_dwt_levels": args.num_dwt_levels
                }
                base_model.save_checkpoint(checkpoint_path, epoch+1, checkpoint_metadata)

    # Save final checkpoint
    if rank == 0:
        print("Training complete. Saving final checkpoint.")
        final_ckpt_path = os.path.join(args.ckpt_path, f"final_checkpoint_epoch_{args.num_epochs}.pt")
        final_metadata = {
            "val_losses": val_losses if 'val_losses' in locals() else {},
            "global_mean": args.global_mean,
            "global_std": args.global_std,
            "num_dwt_levels": args.num_dwt_levels
        }
        base_model.save_checkpoint(final_ckpt_path, args.num_epochs, final_metadata)

    # if rank == 0:
    #     print("Training complete. Saving final checkpoint.")
    #     final_ckpt_path = os.path.join(args.ckpt_path, f"final_checkpoint_epoch_{args.num_epochs}.pt")
    #     if args.where == "cluster":
    #         base_model.save_checkpoint(final_ckpt_path, args.num_epochs, {"val_losses": val_losses if 'val_losses' in locals() else {}})
    #     else:
    #         base_model.save_checkpoint(final_ckpt_path, args.num_epochs, {"val_losses": val_losses if 'val_losses' in locals() else {}})

def extract_dwt_features(latent, num_dwt_levels=1, device='cpu'):
    dwt = DWTForward(J=num_dwt_levels, wave='haar', mode='zero').to(device)
    ll, x = dwt(latent)
    # x.shape: (B, n, 3, C, H/2^n, W/2^n)  # 3 corresponds to (LH, HL, HH)
    return ll, x

def calculate_psnr(img1, img2):
    mse = F.mse_loss(img1, img2)
    if mse == 0:
        return float('inf')
    return 20 * torch.log10(1.0 / torch.sqrt(mse))

def calculate_ssim(img1, img2, max_val=1.0, eps=1e-8):
    C1 = (0.01 * max_val) ** 2
    C2 = (0.03 * max_val) ** 2

    mu1 = F.avg_pool2d(img1, 3, 1, 1)
    mu2 = F.avg_pool2d(img2, 3, 1, 1)

    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1 * mu2

    sigma1_sq = F.avg_pool2d(img1 * img1, 3, 1, 1) - mu1_sq
    sigma2_sq = F.avg_pool2d(img2 * img2, 3, 1, 1) - mu2_sq
    sigma12 = F.avg_pool2d(img1 * img2, 3, 1, 1) - mu1_mu2

    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2) + eps)
    return ssim_map.mean()

def test(args):
    num_dwt_levels = args.num_dwt_levels
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Setup dataset and model
    test_dataset = CustomDataset(parent_dir=args.test_data_path, split='test', 
                                test_data=args.test_data_path, feature_type='image', 
                                num_dwt_levels=num_dwt_levels)
    loader = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=0)

    ddconfig, lossconfig = create_configs_from_args(args)

    ae_model = AutoencoderKL(ddconfig=ddconfig, lossconfig=lossconfig, embed_dim=args.embed_dim, args=args,
                            ckpt_path=None, ignore_keys=[],
                            colorize_nlabels=args.colorize_nlabels,
                            monitor=args.monitor, device=device)
    
    # Load 
    if args.ckpt_path.endswith(".pt"):
        print(f"Loading checkpoint: {args.ckpt_path}")
        # Load the full checkpoint to extract metadata
        checkpoint = torch.load(args.ckpt_path, map_location=device, weights_only=False)
        ae_model.load_checkpoint(args.ckpt_path)
        
        # Extract global statistics from checkpoint metadata
        if 'global_mean' in checkpoint and 'global_std' in checkpoint:
            global_mean = checkpoint['global_mean']
            global_std = checkpoint['global_std']
            print(f"Loaded global statistics - Mean: {global_mean}, Std: {global_std}")
        else:
            print("Warning: Global statistics not found in checkpoint!")
            
    else:
        checkpoint_files = glob.glob(os.path.join(args.ckpt_path, "*.pt"))
        if checkpoint_files:
            latest_ckpt = max(checkpoint_files, key=os.path.getctime)
            print(f"Loading checkpoint: {latest_ckpt}")
            
            # Load the full checkpoint to extract metadata
            checkpoint = torch.load(latest_ckpt, map_location=device, weights_only=False)
            ae_model.load_checkpoint(latest_ckpt)
            
            # Extract global statistics
            if 'global_mean' in checkpoint and 'global_std' in checkpoint:
                global_mean = checkpoint['global_mean']
                global_std = checkpoint['global_std']
                print(f"Loaded global statistics - Mean: {global_mean}, Std: {global_std}")
            else:
                print("Warning: Global statistics not found in checkpoint!")
        else:
            raise FileNotFoundError(f"No checkpoint files found in {args.ckpt_path}")

    ae_model.to(device)
    ae_model.eval()

    # Create output directory
    output_dir = os.path.join(args.results_dir, f'image/eval_results_{args.num_dwt_levels}_dwt')
    os.makedirs(output_dir, exist_ok=True)

    # Initialize DWT inverse
    dwt_inverse = DWTInverse(wave='haar', mode='zero').to(device)

    # Metrics storage
    mse_list, psnr_list, ssim_list = [], [], []
    mse_ll_list, psnr_ll_list, ssim_ll_list = [], [], []
    
    # Additional metrics for debugging
    reconstruction_ranges = []

    with torch.no_grad():
        for idx, (image, _) in enumerate(loader):
            original = image.to(device)
            
            # Extract DWT features
            ll, high_freq_features = extract_dwt_features(original, 
                                                         num_dwt_levels=args.num_dwt_levels, 
                                                         device=device)
            
            # Store original LL for comparison
            ll_original = ll.clone()
            
            # Normalize LL the same way as training data (consistent with CustomDataset)
            # ll_mean = ll.mean()
            # ll_std = ll.std() + 1e-8  # Add small epsilon to avoid division by zero
            ll_normalized = (ll - global_mean) / (global_std + 1e-8)
            
            # Model prediction on normalized LL
            num_dwt_tensor = torch.ones(1, 1, device=device, dtype=torch.float32) * args.num_dwt_levels
            predicted_ll_norm, _ = ae_model(ll_normalized, num_dwt_tensor)
            
            # Denormalize predicted LL back to original DWT coefficient range
            predicted_ll = predicted_ll_norm * global_std + global_mean
            
            # Calculate LL reconstruction metrics (comparing predicted vs original LL)
            mse_val_ll = F.mse_loss(predicted_ll, ll_original).item()
            psnr_val_ll = calculate_psnr(predicted_ll, ll_original).item()
            ssim_val_ll = calculate_ssim(predicted_ll, ll_original).item()
            
            # Reconstruct image using predicted LL and original high-frequency components
            reconstructed = predicted_ll
            
            # Reverse reconstruct through DWT levels
            hf_components = list(reversed(high_freq_features))  # Reverse for proper order
            for i in range(args.num_dwt_levels):
                reconstructed = dwt_inverse((reconstructed, [hf_components[i]]))
            
            # Store reconstruction range for debugging
            recon_min, recon_max = reconstructed.min().item(), reconstructed.max().item()
            reconstruction_ranges.append((recon_min, recon_max))
            
            # Debug print for first few images
            if idx < 3:
                print(f"Image {idx}:")
                print(f"  Original range: [{original.min():.3f}, {original.max():.3f}]")
                print(f"  LL original range: [{ll_original.min():.3f}, {ll_original.max():.3f}]")
                print(f"  LL normalized range: [{ll_normalized.min():.3f}, {ll_normalized.max():.3f}]")
                print(f"  Predicted LL (norm) range: [{predicted_ll_norm.min():.3f}, {predicted_ll_norm.max():.3f}]")
                print(f"  Predicted LL (denorm) range: [{predicted_ll.min():.3f}, {predicted_ll.max():.3f}]")
                print(f"  Reconstructed range: [{recon_min:.3f}, {recon_max:.3f}]")
            
            # Denormalize reconstructed image for metrics and saving
            # Since original images were normalized with mean=0.5, std=0.5 (range [-1,1] -> [0,1])
            # The reconstruction should be in [-1,1] range, so we convert back to [0,1]
            if recon_min >= -1.1 and recon_max <= 1.1:  # Reconstructed is likely in [-1,1] range
                reconstructed_denorm = (reconstructed + 1) / 2  # Convert [-1,1] to [0,1]
                original_denorm = (original + 1) / 2  # Convert original to [0,1] for fair comparison
            else:
                # If reconstruction is not in expected range, clamp to [0,1]
                reconstructed_denorm = reconstructed.clamp(0, 1)
                original_denorm = (original + 1) / 2
            
            # Calculate full reconstruction metrics
            mse_val = F.mse_loss(reconstructed_denorm, original_denorm).item()
            psnr_val = calculate_psnr(reconstructed_denorm, original_denorm).item()
            ssim_val = calculate_ssim(reconstructed_denorm, original_denorm).item()
            
            # Store metrics
            mse_list.append(mse_val)
            psnr_list.append(psnr_val)
            ssim_list.append(ssim_val)
            mse_ll_list.append(mse_val_ll)
            psnr_ll_list.append(psnr_val_ll)
            ssim_ll_list.append(ssim_val_ll)
            
            # Save reconstructed image (ensure it's in [0,1] range for saving)
            if idx < 100:  # Save first 100 images
                save_image(reconstructed_denorm.clamp(0, 1), os.path.join(output_dir, f'recon_{idx}.png'))
                
                # Also save original for comparison
                if idx < 10:  # Save original for first 10 images only
                    save_image(original_denorm.clamp(0, 1), os.path.join(output_dir, f'original_{idx}.png'))
            
            if idx % 10 == 0:  # More frequent logging for debugging
                print(f"Processed {idx} images - Full MSE: {mse_val:.6f}, PSNR: {psnr_val:.2f}, SSIM: {ssim_val:.4f}")
                print(f"                    - LL MSE: {mse_val_ll:.6f}, PSNR: {psnr_val_ll:.2f}, SSIM: {ssim_val_ll:.4f}")

    # Print reconstruction range statistics
    recon_mins, recon_maxs = zip(*reconstruction_ranges)
    print(f"\nReconstruction Range Statistics:")
    print(f"Min values: [{min(recon_mins):.3f}, {max(recon_mins):.3f}] (min-max across all images)")
    print(f"Max values: [{min(recon_maxs):.3f}, {max(recon_maxs):.3f}] (min-max across all images)")

    # Print final metrics
    print(f"\n=== Final Results (DWT Levels: {args.num_dwt_levels}) ===")
    print(f"Full Reconstruction Metrics:")
    print(f"  Average MSE: {sum(mse_list)/len(mse_list):.6f}")
    print(f"  Average PSNR: {sum(psnr_list)/len(psnr_list):.2f} dB")
    print(f"  Average SSIM: {sum(ssim_list)/len(ssim_list):.4f}")
    print(f"\nLL Component Reconstruction Metrics:")
    print(f"  Average MSE: {sum(mse_ll_list)/len(mse_ll_list):.6f}")
    print(f"  Average PSNR: {sum(psnr_ll_list)/len(psnr_ll_list):.2f} dB")
    print(f"  Average SSIM: {sum(ssim_ll_list)/len(ssim_ll_list):.4f}")
    
    # Save metrics to file
    import pandas as pd
    results_df = pd.DataFrame({
        'MSE_full': mse_list, 
        'PSNR_full': psnr_list, 
        'SSIM_full': ssim_list,
        'MSE_LL': mse_ll_list,
        'PSNR_LL': psnr_ll_list, 
        'SSIM_LL': ssim_ll_list
    })
    results_df.to_csv(os.path.join(output_dir, 'metrics.csv'), index=False)
    print(f"\nMetrics saved to: {os.path.join(output_dir, 'metrics.csv')}")
    print(f"Images saved to: {output_dir}")


if __name__ == "__main__":
    args = parse_args()
    global_mean, global_std = compute_global_stats(args.train_data_path, split='train', feature_type='image', num_dwt_levels=args.num_dwt_levels)
    args.global_mean = global_mean
    args.global_std = global_std
    print(f"Using global mean: {global_mean}, global std: {global_std}")
    # Determine rank for distributed training
    if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
        rank = int(os.environ['RANK'])
        is_distributed = True
    else:
        rank = 0
        is_distributed = False
    
    # Run training
    train(args)
    
    # Wait for all processes to finish training (if distributed)
    if is_distributed:
        import torch.distributed as dist
        if dist.is_initialized():
            dist.barrier()
    
    # Only run test and cleanup on rank 0
    # Reset global stats to None to avoid accidental usage in test
    args.global_mean = None
    args.global_std = None
    if rank == 0:
        test(args)
        
        # Create zip archive only if results directory exists
        if os.path.exists(args.results_dir):
            import shutil
            zip_filename = f"{args.results_dir}_results.zip"
            shutil.make_archive(zip_filename.replace('.zip', ''), 'zip', args.results_dir)
            print(f"Results zipped to: {zip_filename}")
        else:
            print(f"Results directory {args.results_dir} does not exist, skipping zip creation.")
    
    # Cleanup distributed training
    if is_distributed:
        import torch.distributed as dist
        if dist.is_initialized():
            dist.destroy_process_group()