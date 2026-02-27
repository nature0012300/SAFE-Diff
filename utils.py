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
from pytorch_fid import fid_score
import shutil
from PIL import Image


from data_setup_final import set_seed, create_dataloader, device

def calculate_metrics(original, reconstructed):
    """
    Calculate PSNR, SSIM, and NMSE for CT image data.
    Performs validation checks specific to CT image intensity ranges.
    
    Args:
        original: Original CT image tensor (normalized to [-1,1])
        reconstructed: Reconstructed CT image tensor (normalized to [-1,1])
        
    Returns:
        tuple: (psnr, ssim, nmse) metrics or (None, None, None) if validation fails
    """
    original = original.cpu().detach().numpy()
    reconstructed = reconstructed.cpu().detach().numpy()
    original = original.squeeze()
    reconstructed = reconstructed.squeeze()
    
    # Basic shape and NaN validation
    if original.shape != reconstructed.shape:
        raise ValueError(f"Shape mismatch: original {original.shape} vs reconstructed {reconstructed.shape}")
    
    if np.isnan(original).any() or np.isnan(reconstructed).any():
        raise ValueError("NaN values found in input images")
        
    # **Check the correct intensity range**
    orig_min, orig_max = np.min(original), np.max(original)
    recon_min, recon_max = np.min(reconstructed), np.max(reconstructed)

    # if orig_max > 1.0 or orig_min < -1.0:
    #     print(f"Warning: Original image values outside expected range [-1,1]: min={orig_min}, max={orig_max}")
    # if recon_max > 1.0 or recon_min < -1.0:
    #     print(f"Warning: Reconstructed image values outside expected range [-1,1]: min={recon_min}, max={recon_max}")

    #  Convert back from [-1,1] → [0,1] before computing PSNR/SSIM
    original = (original + 1) / 2
    reconstructed = (reconstructed + 1) / 2
    
    # Clip to ensure valid PSNR/SSIM calculation
    original = np.clip(original, 0, 1)
    reconstructed = np.clip(reconstructed, 0, 1)
    
    try:
        # Calculate PSNR
        psnr = compare_psnr(original, reconstructed, data_range=1.0)
        
        # Calculate SSIM
        ssim = compare_ssim(original, reconstructed, 
                          data_range=1.0, 
                          multichannel=False,
                          gaussian_weights=True, 
                          sigma=1.5,
                          use_sample_covariance=False)
        
        # Calculate NMSE (Normalized Mean Square Error)
        mse = np.mean((original - reconstructed) ** 2)
        signal_power = np.mean(original ** 2)
        nmse = mse / (signal_power + 1e-12)  # Avoid division by zero
        
        # **Validation checks**
        if not (0 <= ssim <= 1):
            raise ValueError(f"Invalid SSIM value: {ssim}")
        if psnr < 0:
            raise ValueError(f"Invalid PSNR value: {psnr}")
        if nmse < 0:
            raise ValueError(f"Invalid NMSE value: {nmse}")
            
        return psnr, ssim, nmse
        
    except Exception as e:
        print(f"Error calculating metrics: {str(e)}")
        return None, None, None

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