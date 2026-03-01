import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import math
import csv
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm
import random
from skimage.metrics import peak_signal_noise_ratio as compare_psnr
from skimage.metrics import structural_similarity as compare_ssim
from typing import List, Tuple
import lpips
import pywt
from image_fusion import process_and_save_with_fusion
import time
import hashlib

os.environ['PYTORCH_NVML_BASED_CUDA_CHECK'] = '0'
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:512,expandable_segments:True'

from pytorch_fid import fid_score
import shutil
from PIL import Image

from stage_1 import ResidualPredictionNet
from data_setup import set_seed, create_dataloader, device
from diff_refiner import Stage2Model, NoiseScheduler 

class DDIMSampler:
    """
    DDIM sampler optimized for  inference.
    """
    def __init__(self, noise_scheduler, num_train_timesteps, num_inference_steps,max_noise_timestep):
        self.noise_scheduler = noise_scheduler
        self.num_train_timesteps = num_train_timesteps
        self.num_inference_steps = num_inference_steps
        self.max_noise_timestep = max_noise_timestep
        
        # Create the inference timestep schedule (DDIM style)
        step_ratio = self.max_noise_timestep // self.num_inference_steps
        self.timesteps = (np.arange(1, num_inference_steps + 1) * step_ratio).round()[::-1].copy().astype(np.int64)
        self.timesteps = torch.from_numpy(self.timesteps)
        
        trajectory = self.timesteps.tolist() + [0]
        print(f"DDIM Timestep Trajectory ({self.num_inference_steps} steps): {' → '.join(map(str, trajectory))}")
    
    def _get_previous_timestep(self, timestep: int) -> int:
        step_ratio = self.max_noise_timestep // self.num_inference_steps  
        prev_t = timestep - step_ratio
        return prev_t

    def sample_residual(self, model, downsampled_image, device, generator=None):
        
        batch_size = downsampled_image.shape[0]
        
        # Step 1: Get Stage 1 prediction (coarse HR)
        upsampled_image = F.interpolate(downsampled_image, 
                                      size=(512, 512), 
                                      mode='bicubic', align_corners=False)
        
        with torch.no_grad():
            if next(model.stage1_model.parameters()).dtype == torch.float16:
                stage1_residual = model.stage1_model(upsampled_image.half()).float()
            else:
                stage1_residual = model.stage1_model(upsampled_image)
            coarse_hr = torch.clamp(upsampled_image + stage1_residual, -1.0, 1.0)
        
        # Step 2: Add noise to coarse hr  using noise scheduler (same as training)
        initial_timestep = torch.full((batch_size,), self.max_noise_timestep, device=device, dtype=torch.long) # Start with maximum noise (199 for 200-step scheduler)
        pure_noise = torch.randn(coarse_hr.shape, generator=generator, device=device)
    
        initial_noisy_hr = self.noise_scheduler.add_noise(coarse_hr, pure_noise, initial_timestep)
        current_sample = initial_noisy_hr.clone()
        
        
        model.eval()
        with torch.no_grad():
            for step_idx, t in enumerate(tqdm(self.timesteps, desc="DDIM Sampling HR Image")):
                t_tensor = torch.full((batch_size,), t.item(), device=device, dtype=torch.long)
                
               
                predicted_noise = model(
                    noisy_hr_image=current_sample,  
                    downsampled_image=downsampled_image,
                    time_steps=t_tensor,
                    residual_map=stage1_residual
                )
                
                print(f"Step {step_idx}: t={t.item()}, noise range=[{predicted_noise.min():.4f}, {predicted_noise.max():.4f}]")
                
                # DDIM step 
                current_sample = self._ddim_step(
                    t.item(), current_sample, predicted_noise, 
                    eta=0.0,  # Deterministic DDIM - best for super-resolution
                    is_final_step=(step_idx == len(self.timesteps) - 1)
                )
        
        # Final reconstructed HR 
        final_hr = torch.clamp(current_sample, -1.0, 1.0)
        
        return final_hr, coarse_hr, stage1_residual, current_sample, initial_noisy_hr

    def _ddim_step(self, t: int, residual_t: torch.Tensor, model_output: torch.Tensor, eta: float = 0.0, is_final_step: bool = False):
        """
        Correct DDIM denoising step with proper variance computation.
        Uses float64 for internal calculations to improve numerical stability.
        """
        # --- 0. Store original dtype and upcast inputs to float64 ---
        original_dtype = residual_t.dtype
        residual_t_64 = residual_t.to(torch.float64)
        model_output_64 = model_output.to(torch.float64)

        prev_t = self._get_previous_timestep(t)
        
        # --- Get scheduler parameters and cast them to float64 as well ---
        alpha_prod_t = self.noise_scheduler.alphas_cumprod[t].to(torch.float64)
        alpha_prod_t_prev = self.noise_scheduler.alphas_cumprod[prev_t].to(torch.float64) if prev_t >= 0 else torch.tensor(1.0, device=residual_t.device, dtype=torch.float64)
        
        beta_prod_t = 1 - alpha_prod_t
        beta_prod_t_prev = 1 - alpha_prod_t_prev
        
        # --- 1. Predict original residual (x_0) using float64 variables ---
        pred_residual_0 = (residual_t_64 - beta_prod_t.sqrt() * model_output_64) / alpha_prod_t.sqrt()
        
        # 2. Clamp predicted residual for stability
        pred_residual_0 = torch.clamp(pred_residual_0, -1.0, 1.0)
        
        if is_final_step:
            # --- Downcast the final result before returning ---
            return pred_residual_0.to(original_dtype)
        
        # --- 3. Compute DDIM variance in float64 ---
        alpha_ratio = alpha_prod_t / alpha_prod_t_prev
        # Add a small epsilon to prevent division by zero or sqrt of negative
        variance_term = (1 - alpha_ratio)
        ddim_variance_base = torch.sqrt(beta_prod_t_prev / beta_prod_t) * torch.sqrt(torch.clamp(variance_term, min=1e-20))
        ddim_variance = eta * ddim_variance_base
        
        # --- 4. Compute the direction pointing to x_t in float64 ---
        direction_variance_term = beta_prod_t_prev - ddim_variance.pow(2)
        pred_dir = torch.sqrt(torch.clamp(direction_variance_term, min=1e-20)) * model_output_64
        
        # --- 5. DDIM step in float64 ---
        prev_residual = torch.sqrt(alpha_prod_t_prev) * pred_residual_0 + pred_dir
        
        # 6. Add stochastic noise if needed
        if eta > 0 and not is_final_step:
            # Generate noise in the original dtype and then cast
            noise = torch.randn_like(residual_t, dtype=original_dtype).to(torch.float64)
            prev_residual = prev_residual + ddim_variance * noise
        
        # --- 7. Downcast the final result before returning ---
        return prev_residual.to(original_dtype)
    
def calculate_fid_batch(original_images, predicted_images, output_dir, device):
    """Calculate FID for collected images"""
    orig_dir = os.path.join(output_dir, "fid_temp/original")
    pred_dir = os.path.join(output_dir, "fid_temp/predicted")
    os.makedirs(orig_dir, exist_ok=True)
    os.makedirs(pred_dir, exist_ok=True)
    
    # Save images
    for i, (orig, pred) in enumerate(zip(original_images, predicted_images)):
        # Convert from [-1, 1] to [0, 255]
        orig_norm = ((orig.squeeze() + 1) / 2 * 255).clamp(0, 255).byte().cpu().numpy()
        pred_norm = ((pred.squeeze() + 1) / 2 * 255).clamp(0, 255).byte().cpu().numpy()
        
        # Handle grayscale properly
        if orig_norm.ndim == 2:  # Single channel
            Image.fromarray(orig_norm).save(f"{orig_dir}/{i:04d}.png")
            Image.fromarray(pred_norm).save(f"{pred_dir}/{i:04d}.png")
        else:  # Multi-channel
            Image.fromarray(orig_norm).save(f"{orig_dir}/{i:04d}.png")
            Image.fromarray(pred_norm).save(f"{pred_dir}/{i:04d}.png")
    
    try:
        # Calculate FID
        fid_value = fid_score.calculate_fid_given_paths(
            [orig_dir, pred_dir], 
            batch_size=50, 
            device=device, 
            dims=2048
        )
        return fid_value
    finally:
        # Cleanup
        shutil.rmtree(os.path.join(output_dir, "fid_temp"))

def calculate_metrics_batch(original_batch, reconstructed_batch):
    """Calculate PSNR, SSIM, and NMSE for a batch of images."""
    metrics = {"psnr": [], "ssim": [], "nmse": []}
    batch_size = original_batch.shape[0]
    
    for i in range(batch_size):
        original = original_batch[i].cpu().detach().numpy().squeeze()
        reconstructed = reconstructed_batch[i].cpu().detach().numpy().squeeze()
        
        # De-normalize from [-1,1] to [0,1] for metric calculation
        original = np.clip((original + 1) / 2, 0, 1)
        reconstructed = np.clip((reconstructed + 1) / 2, 0, 1)
        
        try:
            psnr = compare_psnr(original, reconstructed, data_range=1.0)
            ssim = compare_ssim(original, reconstructed, 
                          data_range=1.0, 
                          channel_axis=None, 
                          gaussian_weights=True,  # <--- CRITICAL FIX
                          sigma=1.5,              # <--- CRITICAL FIX
                          use_sample_covariance=False)
            mse = np.mean((original - reconstructed) ** 2)
            signal_power = np.mean(original ** 2)
            nmse = mse / (signal_power + 1e-12)
            
            metrics["psnr"].append(psnr)
            metrics["ssim"].append(ssim)
            metrics["nmse"].append(nmse)
        except Exception as e:
            print(f"Error calculating metrics for sample {i}: {str(e)}")
            metrics["psnr"].append(np.nan)
            metrics["ssim"].append(np.nan)
            metrics["nmse"].append(np.nan)
    
    return metrics



def load_trained_model(model_path, config, device):
    """Load the trained Stage 2 model from checkpoint (no-reports version)."""
    print(f"Loading trained model from: {model_path}")
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model checkpoint not found: {model_path}")
    
    try:
        # Step 1: Instantiate Stage2Model (this loads Stage 1 from stage1_model_path)
        model = Stage2Model(
            time_emb_dim=config['time_emb_dim'],
            out_emb_dim=config['out_emb_dim'],
            device=device,
            stage1_model_path=config['stage1_model_path'],
        ).to(device)
        
        print(f"\n[1/3] Stage 1 initially loaded from: {config['stage1_model_path']}")
        
        # Step 2: Load Stage 2 checkpoint BUT filter out Stage 1 keys
        checkpoint = torch.load(model_path, map_location=device)
        if any(k.startswith('_orig_mod.') for k in checkpoint.keys()):
            checkpoint = {k.replace('_orig_mod.', ''): v for k, v in checkpoint.items()}
        
        # **CRITICAL FIX**: Remove stage1_model keys to prevent overwriting
        stage2_only_checkpoint = {k: v for k, v in checkpoint.items() 
                                  if not k.startswith('stage1_model.')}
        
        print(f"\n[2/3] Loading Stage 2 weights (filtered out {len(checkpoint) - len(stage2_only_checkpoint)} Stage 1 keys)")
        loaded = model.load_state_dict(stage2_only_checkpoint, strict=False)
        print(f"Missing keys: {loaded.missing_keys}")
        print(f"Unexpected keys: {loaded.unexpected_keys}")
        # Step 3: **CRITICAL** - Reload Stage 1 to ensure correct weights
        print(f"\n[3/3] Reloading Stage 1 model from: {config['stage1_model_path']}")
        
        stage1_checkpoint = torch.load(config['stage1_model_path'], map_location=device)
        
        if any(k.startswith('_orig_mod.') for k in stage1_checkpoint.keys()):
            stage1_checkpoint = {k.replace('_orig_mod.', ''): v for k, v in stage1_checkpoint.items()}
        
        model.stage1_model.load_state_dict(stage1_checkpoint, strict=True)
        
        # Force float32 and eval mode
        model.stage1_model.float()
        model.stage1_model.eval()
        
        # Freeze Stage 1
        for param in model.stage1_model.parameters():
            param.requires_grad = False
        
        # Verification test
        test_input = torch.randn(1, 1, 512, 512, device=device, dtype=torch.float32)
        with torch.no_grad():
            test_residual = model.stage1_model(test_input)
        
        print(f"✓ Stage 1 verification:")
        print(f"  - Dtype: {next(model.stage1_model.parameters()).dtype}")
        print(f"  - Test output stats: mean={test_residual.mean():.6f}, std={test_residual.std():.6f}")
        print(f"  - Test output range: [{test_residual.min():.4f}, {test_residual.max():.4f}]")
        
        # Compute weight hash for verification

        stage1_hash = hashlib.md5(
            torch.cat([p.flatten() for p in model.stage1_model.parameters()]).cpu().numpy().tobytes()
        ).hexdigest()[:8]
        print(f"  - Weight hash: {stage1_hash}")
        
        model.eval()
        for param in model.parameters():
            param.requires_grad = False
            
        print("\n✓ Model loaded successfully!")
        return model
        
    except Exception as e:  
        print(f"ERROR loading model: {str(e)}")
        raise e

def create_evaluation_visualizations(original_batch, reconstructed_batch, input_batch, 
                                   coarse_hr_batch, stage1_residual_batch, current_sample_batch,
                                   initial_noisy_hr_batch, batch_metrics, save_path, batch_idx):
    """Create comprehensive visualizations showing the complete residual-based pipeline."""
    batch_size = min(original_batch.shape[0], 2)
    fig, axes = plt.subplots(batch_size, 7, figsize=(40, 5 * batch_size + 1))  # 8 columns now
    if batch_size == 1: 
        axes = axes.reshape(1, -1)
    
    for i in range(batch_size):
        # Convert to numpy for plotting
        original_np = ((original_batch[i, 0].cpu() + 1) / 2).clamp(0, 1).numpy()
        reconstructed_np = ((reconstructed_batch[i, 0].cpu() + 1) / 2).clamp(0, 1).numpy()
        input_np = ((input_batch[i, 0].cpu() + 1) / 2).clamp(0, 1).numpy()
        coarse_hr_np = ((coarse_hr_batch[i, 0].cpu() + 1) / 2).clamp(0, 1).numpy()
        stage1_residual_np = ((stage1_residual_batch[i, 0].cpu() + 1) / 2).clamp(0, 1).numpy()
        current_sample_np = ((current_sample_batch[i, 0].cpu() + 1) / 2).clamp(0, 1).numpy()
        initial_noisy_hr_np = ((initial_noisy_hr_batch[i, 0].cpu() + 1) / 2).clamp(0, 1).numpy()
        diff_map = np.abs(original_np - reconstructed_np)
        
        # Plot the complete pipeline 
        axes[i, 0].imshow(input_np, cmap='gray'); axes[i, 0].set_title("Input (Upsampled LR)"); axes[i, 0].axis('off')
        axes[i, 1].imshow(stage1_residual_np, cmap='gray'); axes[i, 1].set_title("Stage 1 Residual (Clean)"); axes[i, 1].axis('off')
        axes[i, 2].imshow(initial_noisy_hr_np, cmap='gray'); axes[i, 2].set_title("Initial Noisy hr "); axes[i, 2].axis('off')
        axes[i, 3].imshow(coarse_hr_np, cmap='gray'); axes[i, 3].set_title("Coarse HR (Stage 1)"); axes[i, 3].axis('off')
        axes[i, 4].imshow(reconstructed_np, cmap='gray'); axes[i, 4].set_title("Final HR"); axes[i, 4].axis('off')
        axes[i, 5].imshow(original_np, cmap='gray'); axes[i, 5].set_title("Ground Truth HR"); axes[i, 5].axis('off')
        axes[i, 6].imshow(diff_map, cmap='hot'); axes[i, 6].set_title("Difference Map"); axes[i, 6].axis('off')
        
        # Add metrics text
        if i < len(batch_metrics['psnr']):
            metrics_text = (f"PSNR: {batch_metrics['psnr'][i]:.2f} dB | "
                          f"SSIM: {batch_metrics['ssim'][i]:.4f} | "
                          f"NMSE: {batch_metrics['nmse'][i]:.4f} | "
                          f"LPIPS: {batch_metrics['lpips'][i]:.4f}")
            axes[i, 5].text(0.5, -0.15, metrics_text, ha='center', va='center',
                            transform=axes[i, 5].transAxes,
                            bbox=dict(boxstyle="round,pad=0.3", facecolor="lightblue", alpha=0.8))

    plt.suptitle(f'Complete  Pipeline - Batch {batch_idx + 1}', fontsize=16)
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()


def create_metrics_plots(all_metrics, output_dir):
    """Create distribution plots for all metrics."""
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    axes = axes.flatten()
    
    metrics_info = [
        ('psnr', 'PSNR (dB)', 'Higher is better'),
        ('ssim', 'SSIM', 'Higher is better'),
        ('nmse', 'NMSE', 'Lower is better'),
        ('lpips', 'LPIPS', 'Lower is better')
    ]
    
    for idx, (metric_name, title, note) in enumerate(metrics_info):
        if idx < len(axes):
            values = [v for v in all_metrics[metric_name] if not np.isnan(v)]
            if values:
                axes[idx].hist(values, bins=30, alpha=0.7, edgecolor='black')
                axes[idx].set_title(f'{title} Distribution\n({note})')
                axes[idx].set_xlabel(title)
                axes[idx].set_ylabel('Frequency')
                axes[idx].grid(True, alpha=0.3)
                
                # Add statistics text
                mean_val = np.mean(values)
                std_val = np.std(values)
                axes[idx].axvline(mean_val, color='red', linestyle='--', 
                                label=f'Mean: {mean_val:.3f}±{std_val:.3f}')
                axes[idx].legend()
    
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plot_path = os.path.join(output_dir, "metrics_distributions.png")
    plt.savefig(plot_path, dpi=300)
    plt.close()
    print(f"Metrics distribution plots saved to: {plot_path}")


def evaluate_stage2_model(config):
    """
    Main evaluation function for Stage 2 model  inference.
    """
    print("=" * 60)
    print("STAGE 2 MODEL EVALUATION - INFERENCE")
    print("=" * 60)
    
    set_seed(config['evaluation_seed'])
    print(f"Using device: {device}")
    
    eval_dir = config['evaluation_output_dir']
    vis_dir = os.path.join(eval_dir, "visualizations")
    os.makedirs(vis_dir, exist_ok=True)

    images_output_dir = os.path.join(eval_dir, 'saved_images')
    os.makedirs(images_output_dir, exist_ok=True)
    print(f"Images will be saved to: {images_output_dir}")
    
    print("\n[STEP 1/6] Loading test dataset...")
    test_dataloader = create_dataloader(
        csv_file=config['test_csv_path'],
        max_items=config.get('max_test_items', None),
        batch_size=config['eval_batch_size'],
        shuffle=False,
        drop_last=False
    )
    

    if len(test_dataloader) == 0:
        print(f"\n[ERROR] Dataloader is empty. Check path: {config['test_csv_path']}")
        return
    print(f"--> Test dataset loaded: {len(test_dataloader)} batches.")
    
    print("\n[STEP 2/6] Loading trained model...")
    model = load_trained_model(config['model_path'], config, device)

    print("\n" + "="*60)
    print("STAGE 1 WEIGHT VERIFICATION")
    print("="*60)
    weight_bytes = torch.cat([p.flatten() for p in model.stage1_model.parameters()]).cpu().numpy().tobytes()
    weight_hash = hashlib.md5(weight_bytes).hexdigest()[:16]
    print(f"Stage 1 weight hash: {weight_hash}")
    print(f"Save this hash to compare between different Stage 1 models!")
    print("="*60 + "\n")

    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    print("\n[STEP 3/6] Initializing  sampler...")
    noise_scheduler = NoiseScheduler(num_timesteps=config['num_train_timesteps'], device=device)
    sampler = DDIMSampler(noise_scheduler, num_train_timesteps=config['num_train_timesteps'], num_inference_steps=config['num_inference_steps'],max_noise_timestep=config['max_noise_timestep'] )
    print(sampler.timesteps.tolist())
    print(sampler._get_previous_timestep(132))
    print(sampler._get_previous_timestep(88))
    print(sampler._get_previous_timestep(44))
    lpips_loss_fn = lpips.LPIPS(net='alex').to(device)
    
    all_metrics = {
    "psnr": [], "ssim": [], "nmse": [], "lpips": [],
    "coarse_psnr": [], "coarse_ssim": [], "coarse_nmse": [], "coarse_lpips": [],
    "fused_psnr": [], "fused_ssim": [], "fused_nmse": [], "fused_lpips": []
    }
    random_indices = random.sample(range(len(test_dataloader)), min(config['num_visualizations'], len(test_dataloader)))
    
    cumulative_idx = 0

    print(f"\n[STEP 4/6] Starting evaluation ...")
    print("-" * 60)
    
    collected_original = []
    collected_final_hr = []
    collected_coarse_hr = []
    collected_fused_hr = []
    
    inference_times = []
    warmup_batches = 2

    with torch.no_grad():
        for batch_idx, (original_hr, downsampled_lr) in enumerate(test_dataloader):
            print(f"Processing batch {batch_idx+1}/{len(test_dataloader)}...")
            
            original_hr = original_hr.float().to(device)
            downsampled_lr = downsampled_lr.float().to(device)
            
            # Ensure data is in [-1, 1] range
            original_hr = torch.clamp(original_hr, -1.0, 1.0)
            downsampled_lr = torch.clamp(downsampled_lr, -1.0, 1.0)
            
            if original_hr.shape[0] != downsampled_lr.shape[0]:
                print(f"Batch {batch_idx} has mismatched sizes: HR {original_hr.shape}, LR {downsampled_lr.shape}")
                continue
            
            generator = torch.Generator(device=device).manual_seed(config['evaluation_seed'] + batch_idx)
            
            upsampled_lr =  F.interpolate(downsampled_lr, 
                                      size=(512, 512), 
                                      mode='bicubic', align_corners=False)
            
            torch.cuda.synchronize()
            start_time = time.time()  

            final_hr, coarse_hr, stage1_residual, current_sample, initial_noisy_hr = sampler.sample_residual(
                model, downsampled_lr, device, generator
            )
            
            # Calculate metrics
            cumulative_idx, fused_hr = process_and_save_with_fusion(
                original_hr=original_hr,
                coarse_hr=coarse_hr,
                final_hr=final_hr,
                upsampled_lr=upsampled_lr,
                images_output_dir=images_output_dir,
                batch_idx=batch_idx,
                cumulative_idx=cumulative_idx,
                wavelet='db2',  # You can also try 'db1', 'db2', 'sym2', 'coif1'
                save_dwt_viz=False  # Set to True to visualize DWT decomposition
            )
            torch.cuda.synchronize()
            end_time = time.time()
            # --- END END-TO-END TIMER ---

            if batch_idx >= warmup_batches:
                # Time per image in batch
                inference_times.append((end_time - start_time) / original_hr.size(0))

            fused_hr = fused_hr.to(device)
            
            batch_metrics = calculate_metrics_batch(original_hr, final_hr)
            fused_batch_metrics = calculate_metrics_batch(original_hr, fused_hr)
            if original_hr.shape[1] == 1:
                original_hr_rgb = original_hr.repeat(1, 3, 1, 1)
                fused_hr_rgb = fused_hr.repeat(1, 3, 1, 1)
                fused_lpips_scores = lpips_loss_fn(original_hr_rgb, fused_hr_rgb).squeeze().cpu().tolist()
            else:
                fused_lpips_scores = lpips_loss_fn(original_hr, fused_hr).squeeze().cpu().tolist()

            if not isinstance(fused_lpips_scores, list):
                fused_lpips_scores = [fused_lpips_scores]
            fused_batch_metrics["lpips"] = fused_lpips_scores
            
            # Calculate LPIPS
            if original_hr.shape[1] == 1:  # Grayscale images
                original_hr_rgb = original_hr.repeat(1, 3, 1, 1)  # [B, 1, H, W] → [B, 3, H, W]
                final_hr_rgb = final_hr.repeat(1, 3, 1, 1)
                lpips_scores = lpips_loss_fn(original_hr_rgb, final_hr_rgb).squeeze().cpu().tolist()
            else:  # Already RGB
                lpips_scores = lpips_loss_fn(original_hr, final_hr).squeeze().cpu().tolist()

            if not isinstance(lpips_scores, list): 
                lpips_scores = [lpips_scores]
            batch_metrics["lpips"] = lpips_scores
            
            coarse_batch_metrics = calculate_metrics_batch(original_hr, coarse_hr)

            if original_hr.shape[1] == 1:  # Grayscale images
                original_hr_rgb = original_hr.repeat(1, 3, 1, 1)
                coarse_hr_rgb = coarse_hr.repeat(1, 3, 1, 1)
                coarse_lpips_scores = lpips_loss_fn(original_hr_rgb, coarse_hr_rgb).squeeze().cpu().tolist()
            else:  # Already RGB
                coarse_lpips_scores = lpips_loss_fn(original_hr, coarse_hr).squeeze().cpu().tolist()

            if not isinstance(coarse_lpips_scores, list):
                coarse_lpips_scores = [coarse_lpips_scores]
            coarse_batch_metrics["lpips"] = coarse_lpips_scores

            # Accumulate all metrics in one go
            all_metrics["psnr"].extend(batch_metrics["psnr"])
            all_metrics["ssim"].extend(batch_metrics["ssim"])
            all_metrics["nmse"].extend(batch_metrics["nmse"])
            all_metrics["lpips"].extend(batch_metrics["lpips"])
            all_metrics["coarse_psnr"].extend(coarse_batch_metrics["psnr"])
            all_metrics["coarse_ssim"].extend(coarse_batch_metrics["ssim"])
            all_metrics["coarse_nmse"].extend(coarse_batch_metrics["nmse"])
            all_metrics["coarse_lpips"].extend(coarse_batch_metrics["lpips"])
            all_metrics["fused_psnr"].extend(fused_batch_metrics["psnr"])
            all_metrics["fused_ssim"].extend(fused_batch_metrics["ssim"])
            all_metrics["fused_nmse"].extend(fused_batch_metrics["nmse"])
            all_metrics["fused_lpips"].extend(fused_batch_metrics["lpips"])
           
            
            # Generate visualizations for selected batches
            collected_original.extend([img.cpu() for img in original_hr])
            collected_final_hr.extend([img.cpu() for img in final_hr])
            collected_coarse_hr.extend([img.cpu() for img in coarse_hr])
            collected_fused_hr.extend([img.cpu() for img in fused_hr])

            if batch_idx in random_indices:
                upsampled_input = F.interpolate(downsampled_lr, size=original_hr.shape[-2:], mode='bicubic', align_corners=False)
                vis_path = os.path.join(vis_dir, f"evaluation_batch_{batch_idx:03d}.png")
                
                create_evaluation_visualizations(
                    original_hr, final_hr, upsampled_input, 
                    coarse_hr, stage1_residual, current_sample, initial_noisy_hr,
                    batch_metrics, vis_path, batch_idx
                )
                print(f"  -> Visualization saved: {vis_path}")
    
    print("\n[STEP 5/6] Calculating FID scores...")
    fid_final_hr = calculate_fid_batch(collected_original, collected_final_hr, eval_dir, device)
    fid_coarse_hr = calculate_fid_batch(collected_original, collected_coarse_hr, eval_dir, device)
    fid_fused_hr = calculate_fid_batch(collected_original, collected_fused_hr, eval_dir, device)


    print("\n[STEP 6/6] Calculating and saving final results...")
    
    # Calculate final statistics
    final_stats = {}
    for metric_name, values in all_metrics.items():
        clean_values = [v for v in values if not np.isnan(v)]
        if clean_values:
            final_stats[metric_name] = {
                'mean': np.mean(clean_values),
                'std': np.std(clean_values),
                'min': np.min(clean_values),
                'max': np.max(clean_values),
                'count': len(clean_values)
            }
        else:
            final_stats[metric_name] = {
                'mean': np.nan, 'std': np.nan, 'min': np.nan, 'max': np.nan, 'count': 0
            }

    # Print Final HR results
    print("\n" + "=" * 60)
    print("FINAL HR (Stage 2) vs GROUND TRUTH - RESULTS")
    print("=" * 60)
    for metric_name in ["psnr", "ssim", "nmse", "lpips"]:
        stats = final_stats[metric_name]
        print(f"\n{metric_name.upper()}:")
        if stats['count'] > 0:
            print(f"  Mean: {stats['mean']:.4f} ± {stats['std']:.4f}")
            print(f"  Range: [{stats['min']:.4f}, {stats['max']:.4f}]")

    # Print Coarse HR results
    print("\n" + "=" * 60)
    print("COARSE HR (Stage 1) vs GROUND TRUTH - RESULTS")
    print("=" * 60)
    for metric_name in ["coarse_psnr", "coarse_ssim", "coarse_nmse", "coarse_lpips"]:
        stats = final_stats[metric_name]
        display_name = metric_name.replace("coarse_", "").upper()
        print(f"\n{display_name}:")
        if stats['count'] > 0:
            print(f"  Mean: {stats['mean']:.4f} ± {stats['std']:.4f}")
            print(f"  Range: [{stats['min']:.4f}, {stats['max']:.4f}]")
    
    avg_inf_time = np.mean(inference_times) if inference_times else 0

    print(f"\nAverage Inference Time per Image: {avg_inf_time:.4f}s")
    
    print("\n" + "=" * 60)
    print("FID SCORES (Lower is better)")
    print("=" * 60)
    print(f"\nFinal HR FID: {fid_final_hr}")
    print(f"Coarse HR FID: {fid_coarse_hr}")
    print(f"Fused HR FID: {fid_fused_hr}")
    
    # Save results
    results_csv = os.path.join(eval_dir, "detailed_evaluation_results.csv")
    pd.DataFrame(all_metrics).to_csv(results_csv, index=False)
    print(f"\nDetailed results saved to: {results_csv}")
    
    summary_data = [{'metric': k, **v} for k, v in final_stats.items()]
    summary_csv = os.path.join(eval_dir, "evaluation_summary.csv")
    pd.DataFrame(summary_data).to_csv(summary_csv, index=False)
    print(f"Summary statistics saved to: {summary_csv}")
    
    create_metrics_plots(all_metrics, eval_dir)
    print(f"\nEvaluation finished successfully! Results are in: {eval_dir}")
    
    return final_stats


if __name__ == '__main__':
    evaluation_config = {
        'model_path': 'path/to/your/trained/diffusion_refiner_model.pth',
        'stage1_model_path': 'path/to/your/trained/stage_1_model.pth',
        'test_csv_path': 'path/to/your/infer_data.csv',
        'time_emb_dim': 128,
        'out_emb_dim': 512,
        'eval_batch_size': 8,
        'num_train_timesteps': 1000,  # Full scheduler range
        'num_inference_steps': 2,     # Number of denoising steps (adjust as needed)
        'max_noise_timestep': 98,  # Can be adjusted (5, 10, 15, 20, etc.)
        'max_test_items': None,  # Set to None for full evaluation
        'evaluation_seed': 42,
        'num_visualizations': 10,
        'evaluation_output_dir': 'output_dir' 
    }
    
    evaluate_stage2_model(evaluation_config)
