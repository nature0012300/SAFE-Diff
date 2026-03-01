import torch
import os
from PIL import Image
import numpy as np
from tqdm import tqdm
import pywt
import numpy as np
from PIL import Image

def save_images_to_directories(original_hr, coarse_hr, final_hr,upsampled_lr, output_base_dir, batch_idx, start_idx=0):
    """
    Save original, stage_1 (coarse_hr), and final_hr images to organized directories.
    
    Args:
        original_hr: Tensor of original HR images [B, C, H, W]
        coarse_hr: Tensor of Stage 1 output images [B, C, H, W]
        final_hr: Tensor of Stage 2 output images [B, C, H, W]
        output_base_dir: Base directory for saving images
        batch_idx: Current batch index
        start_idx: Starting index for naming files (cumulative across batches)
    
    Returns:
        next_start_idx: Updated index for next batch
    """
    # Create directories if they don't exist
    dirs = {
        'original': os.path.join(output_base_dir, 'original'),
        'stage_1': os.path.join(output_base_dir, 'stage_1'),
        'final_hr': os.path.join(output_base_dir, 'final_hr'),
        'upsampled_lr': os.path.join(output_base_dir, 'upsampled_lr')
    }
    
    for dir_path in dirs.values():
        os.makedirs(dir_path, exist_ok=True)
    
    batch_size = original_hr.shape[0]
    
    # Save each image in the batch
    for i in range(batch_size):
        img_idx = start_idx + i
        
        # Convert tensors to numpy and normalize from [-1, 1] to [0, 255]
        original_np = ((original_hr[i].cpu().detach() + 1) / 2 * 255).clamp(0, 255).byte().numpy()
        coarse_np = ((coarse_hr[i].cpu().detach() + 1) / 2 * 255).clamp(0, 255).byte().numpy()
        final_np = ((final_hr[i].cpu().detach() + 1) / 2 * 255).clamp(0, 255).byte().numpy()
        upsampled_np = ((upsampled_lr[i].cpu().detach() + 1) / 2 * 255).clamp(0, 255).byte().numpy()
        
        # Handle grayscale (C=1) or RGB (C=3)
        if original_np.shape[0] == 1:  # Grayscale
            original_np = original_np.squeeze(0)
            coarse_np = coarse_np.squeeze(0)
            final_np = final_np.squeeze(0)
        else:  # RGB - transpose from CHW to HWC
            original_np = np.transpose(original_np, (1, 2, 0))
            coarse_np = np.transpose(coarse_np, (1, 2, 0))
            final_np = np.transpose(final_np, (1, 2, 0))
        if upsampled_np.shape[0] == 1:
            upsampled_np = upsampled_np.squeeze(0)
        else:
            upsampled_np = np.transpose(upsampled_np, (1, 2, 0))

                
        # Save images with consistent naming
        filename = f"image_{img_idx:05d}.png"
        Image.fromarray(original_np).save(os.path.join(dirs['original'], filename))
        Image.fromarray(coarse_np).save(os.path.join(dirs['stage_1'], filename))
        Image.fromarray(final_np).save(os.path.join(dirs['final_hr'], filename))
        Image.fromarray(upsampled_np).save(os.path.join(dirs['upsampled_lr'], filename))
    
    return start_idx + batch_size


def perform_swt_fusion(stage1_output, final_hr, wavelet='db2'):
    """
    Surgical 2-level SWT Fusion:
    - LL2 (Coarsest Approximation) -> From stage1_output (Anatomy)
    - Everything else -> From final_hr (Texture/Refined details)
    
    Args:
        stage1_output : Tensor [B, C, 512, 512]
        final_hr      : Tensor [B, C, 512, 512]
        wavelet       : Wavelet type ('db2', 'sym2', etc.)
    """
    batch_size, channels, height, width = stage1_output.shape
    fused_hr = torch.zeros_like(stage1_output)
    
    for b in range(batch_size):
        for c in range(channels):
            # Convert to NumPy (CPU processing)
            s1_np = stage1_output[b, c].cpu().detach().float().numpy()
            f2_np = final_hr[b, c].cpu().detach().float().numpy()
            
            # SWT2 returns: [ (cA2, (cH2, cV2, cD2)), (cA1, (cH1, cV1, cD1)) ]
            coeffs_s1 = pywt.swt2(s1_np, wavelet, level=2)
            coeffs_f2 = pywt.swt2(f2_np, wavelet, level=2)
            
            # --- Unpack Level 2 (Coarsest) ---
            cA2_s1, (cH2_s1, cV2_s1, cD2_s1) = coeffs_s1[0]
            cA2_f2, (cH2_f2, cV2_f2, cD2_f2) = coeffs_f2[0]
            
            # --- Unpack Level 1 (Finest) ---
            cA1_s1, (cH1_s1, cV1_s1, cD1_s1) = coeffs_s1[1]
            cA1_f2, (cH1_f2, cV1_f2, cD1_f2) = coeffs_f2[1]
            
            # --- FUSION STRATEGY ---
            # 1. Level 2: Take ONLY LL2 from Stage 1. Take High-Freqs (H, V, D) from Stage 2.
            fused_L2 = (cA2_s1, (cH2_f2, cV2_f2, cD2_f2))
            
            # 2. Level 1: Take EVERYTHING from Stage 2.
            # (cA1 from stage 2 is the "mid-range" approximation that fits stage 2 textures)
            fused_L1 = (cA1_f2, (cH1_f2, cV1_f2, cD1_f2))
            
            # Combine into the list format required for ISWT
            fused_coeffs = [fused_L2, fused_L1]
            
            # Reconstruction
            fused_np = pywt.iswt2(fused_coeffs, wavelet)
            
            # Back to Tensor
            fused_hr[b, c] = torch.from_numpy(fused_np).to(
                device=stage1_output.device, 
                dtype=stage1_output.dtype
            )
            
    return fused_hr


def save_fused_images(fused_hr, output_base_dir, batch_idx, start_idx=0):
    """
    Save fused HR images to directory.
    
    Args:
        fused_hr: Tensor of fused HR images [B, C, H, W]
        output_base_dir: Base directory for saving images
        batch_idx: Current batch index
        start_idx: Starting index for naming files
    
    Returns:
        next_start_idx: Updated index for next batch
    """
    # Create fused_hr directory
    fused_dir = os.path.join(output_base_dir, 'fused_hr')
    os.makedirs(fused_dir, exist_ok=True)
    
    batch_size = fused_hr.shape[0]
    
    # Save each image in the batch
    for i in range(batch_size):
        img_idx = start_idx + i
        
        # Convert tensor to numpy and normalize from [-1, 1] to [0, 255]
        fused_np = ((fused_hr[i].cpu().detach() + 1) / 2 * 255).clamp(0, 255).byte().numpy()
        
        # Handle grayscale (C=1) or RGB (C=3)
        if fused_np.shape[0] == 1:  # Grayscale
            fused_np = fused_np.squeeze(0)
        else:  # RGB - transpose from CHW to HWC
            fused_np = np.transpose(fused_np, (1, 2, 0))
        
        # Save image with consistent naming
        filename = f"image_{img_idx:05d}.png"
        Image.fromarray(fused_np).save(os.path.join(fused_dir, filename))
    
    return start_idx + batch_size


def visualize_dwt_decomposition(image_tensor, save_path, title="DWT Decomposition"):
    """
    Visualize DWT decomposition (LL, LH, HL, HH bands) for debugging.
    
    Args:
        image_tensor: Single image tensor [1, C, H, W] or [C, H, W]
        save_path: Path to save visualization
        title: Title for the plot
    """
    import matplotlib.pyplot as plt
    
    # Handle batch dimension
    if image_tensor.dim() == 4:
        image_tensor = image_tensor[0]
    
    # Take first channel if multi-channel
    img_np = image_tensor[0].cpu().detach().numpy()
    
    # Perform DWT
    coeffs = pywt.dwt2(img_np, 'db2')
    LL, (LH, HL, HH) = coeffs
    
    # Create visualization
    fig, axes = plt.subplots(2, 2, figsize=(10, 10))
    
    axes[0, 0].imshow(LL, cmap='gray')
    axes[0, 0].set_title('LL (Low-Low) - Approximation')
    axes[0, 0].axis('off')
    
    axes[0, 1].imshow(LH, cmap='gray')
    axes[0, 1].set_title('LH (Low-High) - Horizontal Details')
    axes[0, 1].axis('off')
    
    axes[1, 0].imshow(HL, cmap='gray')
    axes[1, 0].set_title('HL (High-Low) - Vertical Details')
    axes[1, 0].axis('off')
    
    axes[1, 1].imshow(HH, cmap='gray')
    axes[1, 1].set_title('HH (High-High) - Diagonal Details')
    axes[1, 1].axis('off')
    
    plt.suptitle(title)
    plt.tight_layout()
    plt.savefig(save_path, dpi=600, bbox_inches='tight')
    plt.close()

def process_and_save_with_fusion(original_hr, coarse_hr, final_hr,upsampled_lr, 
                                  images_output_dir, batch_idx, cumulative_idx,
                                  wavelet='db2', save_dwt_viz=False):
    """
    Complete pipeline: Save original/stage_1/final_hr + perform DWT fusion + save fused_hr
    
    Args:
        original_hr: Ground truth HR images
        coarse_hr: Stage 1 output (coarse HR)
        final_hr: Stage 2 output (final HR)
        images_output_dir: Base output directory
        batch_idx: Current batch index
        cumulative_idx: Cumulative image index
        wavelet: Wavelet type for DWT (default: 'haar')
        save_dwt_viz: Whether to save DWT decomposition visualizations
    
    Returns:
        Updated cumulative_idx, fused_hr tensor
    """
    
    # Step 1: Save original, stage_1, and final_hr (existing functionality)
    cumulative_idx = save_images_to_directories(
        original_hr=original_hr,
        coarse_hr=coarse_hr,
        final_hr=final_hr,
        upsampled_lr=upsampled_lr,
        output_base_dir=images_output_dir,
        batch_idx=batch_idx,
        start_idx=cumulative_idx
    )
    
    # Step 2: Perform DWT fusion
    print(f"  -> Performing DWT fusion (wavelet={wavelet})...")
    fused_hr = perform_swt_fusion(coarse_hr, final_hr, wavelet=wavelet)
    
    # Step 3: Save fused images
    save_fused_images(
        fused_hr=fused_hr,
        output_base_dir=images_output_dir,
        batch_idx=batch_idx,
        start_idx=cumulative_idx - original_hr.shape[0]  # Use same indices
    )
    
    # Save SWT decomposition visualizations if want to visulaize the decomposition
    if save_dwt_viz and batch_idx == 0:
        viz_dir = os.path.join(images_output_dir, 'dwt_visualizations')
        os.makedirs(viz_dir, exist_ok=True)
        
        visualize_dwt_decomposition(
            coarse_hr[0:1], 
            os.path.join(viz_dir, 'stage1_dwt.png'),
            'Stage 1 DWT Decomposition'
        )
        visualize_dwt_decomposition(
            final_hr[0:1],
            os.path.join(viz_dir, 'final_hr_dwt.png'),
            'Final HR DWT Decomposition'
        )
        visualize_dwt_decomposition(
            fused_hr[0:1],
            os.path.join(viz_dir, 'fused_hr_dwt.png'),
            'Fused HR DWT Decomposition'
        )
        print(f"  -> DWT visualizations saved to {viz_dir}")
    
    return cumulative_idx, fused_hr
