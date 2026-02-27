import torch
import os
from PIL import Image
import numpy as np
from tqdm import tqdm

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


import torch
import pywt
import numpy as np
from PIL import Image
import os

# def perform_dwt_fusion(stage1_output, final_hr, wavelet='haar'):
#     """
#     Perform DWT-based fusion:
#     - Use LL (low frequency) band from stage_1 output
#     - Use LH, HL, HH (high frequency) bands from final_hr
    
#     Args:
#         stage1_output: Tensor [B, C, H, W] - Stage 1 coarse HR output
#         final_hr: Tensor [B, C, H, W] - Stage 2 final HR output
#         wavelet: Wavelet type (default: 'haar', can use 'db1', 'db2', 'sym2', etc.)
    
#     Returns:
#         fused_hr: Tensor [B, C, H, W] - Fused HR image
#     """
#     batch_size, channels, height, width = stage1_output.shape
#     batch_size = 2
#     # Initialize output tensor
#     fused_hr = torch.zeros_like(stage1_output)
    
#     # Process each image in batch
#     for b in range(batch_size):
#         # Process each channel separately
#         for c in range(channels):
#             # Convert to numpy for pywt processing
#             stage1_np = stage1_output[b, c].cpu().detach().numpy()
#             final_np = final_hr[b, c].cpu().detach().numpy()
            
#             # Perform 2D DWT on both images
#             # Returns (LL, (LH, HL, HH))
#             coeffs_stage1 = pywt.dwt2(stage1_np, wavelet)
#             coeffs_final = pywt.dwt2(final_np, wavelet)
            
#             # Extract components
#             LL_stage1, (LH_stage1, HL_stage1, HH_stage1) = coeffs_stage1
#             LL_final, (LH_final, HL_final, HH_final) = coeffs_final
            
#             # Fusion strategy:
#             # - LL from stage_1 (low frequency content)
#             # - LH, HL, HH from final_hr (high frequency details)
#             fused_coeffs = (LL_stage1, (LH_final, HL_final, HH_final))
            
#             # Perform inverse DWT to reconstruct fused image
#             fused_np = pywt.idwt2(fused_coeffs, wavelet)
            
#             # Handle size mismatch due to DWT/IDWT (crop or pad if needed)
#             if fused_np.shape != stage1_np.shape:
#                 # Crop to original size if larger
#                 fused_np = fused_np[:height, :width]
            
#             # Convert back to tensor
#             fused_hr[b, c] = torch.from_numpy(fused_np).to(stage1_output.device)
    
#     return fused_hr


def perform_dwt_fusion(stage1_output, final_hr, wavelet='haar'):
    """
    Perform 2-level DWT-based fusion:
    - Decompose both images using 2-level DWT
    - Replace only the 2nd level LL band from stage1_output
    - Keep all other bands (1st level LH/HL/HH and 2nd level LH/HL/HH) from final_hr
    
    Args:
        stage1_output: Tensor [B, C, H, W] - Stage 1 coarse HR output
        final_hr: Tensor [B, C, H, W] - Stage 2 final HR output
        wavelet: Wavelet type (default: 'haar', can use 'db1', 'db2', 'sym2', etc.)
        
    Returns:
        fused_hr: Tensor [B, C, H, W] - Fused HR image
    """
    import pywt
    import torch
    
    batch_size, channels, height, width = stage1_output.shape
    
    # Initialize output tensor
    fused_hr = torch.zeros_like(stage1_output)
    
    # Process each image in batch
    for b in range(batch_size):
        # Process each channel separately
        for c in range(channels):
            # Convert to numpy for pywt processing
            stage1_np = stage1_output[b, c].cpu().detach().numpy()
            final_np = final_hr[b, c].cpu().detach().numpy()
            
            # Perform 2-level DWT on both images
            # Level 1 decomposition
            coeffs_stage1_L1 = pywt.dwt2(stage1_np, wavelet)
            coeffs_final_L1 = pywt.dwt2(final_np, wavelet)
            
            # Extract Level 1 components
            LL_stage1_L1, (LH_stage1_L1, HL_stage1_L1, HH_stage1_L1) = coeffs_stage1_L1
            LL_final_L1, (LH_final_L1, HL_final_L1, HH_final_L1) = coeffs_final_L1
            
            # Level 2 decomposition (decompose the LL band from Level 1)
            coeffs_stage1_L2 = pywt.dwt2(LL_stage1_L1, wavelet)
            coeffs_final_L2 = pywt.dwt2(LL_final_L1, wavelet)
            
            # Extract Level 2 components
            LL_stage1_L2, (LH_stage1_L2, HL_stage1_L2, HH_stage1_L2) = coeffs_stage1_L2
            LL_final_L2, (LH_final_L2, HL_final_L2, HH_final_L2) = coeffs_final_L2
            
            # Fusion strategy:
            # - Level 2 LL from stage1_output (deepest low frequency)
            # - Level 2 LH, HL, HH from final_hr (high frequency at level 2)
            fused_coeffs_L2 = (LL_stage1_L2, (LH_final_L2, HL_final_L2, HH_final_L2))
            
            # Reconstruct Level 1 LL band from fused Level 2 coefficients
            fused_LL_L1 = pywt.idwt2(fused_coeffs_L2, wavelet)
            
            # Handle size mismatch for Level 1 LL reconstruction
            if fused_LL_L1.shape != LL_final_L1.shape:
                # Crop to match the expected size
                h_L1, w_L1 = LL_final_L1.shape
                fused_LL_L1 = fused_LL_L1[:h_L1, :w_L1]
            
            # Now create Level 1 fused coefficients:
            # - Fused LL from above (which contains stage1's L2 LL)
            # - LH, HL, HH from final_hr's Level 1
            fused_coeffs_L1 = (fused_LL_L1, (LH_final_L1, HL_final_L1, HH_final_L1))
            
            # Perform inverse DWT to reconstruct final fused image
            fused_np = pywt.idwt2(fused_coeffs_L1, wavelet)
            
            # Handle size mismatch due to DWT/IDWT (crop to original size if needed)
            if fused_np.shape != stage1_np.shape:
                fused_np = fused_np[:height, :width]
            
            # Convert back to tensor
            fused_hr[b, c] = torch.from_numpy(fused_np).to(stage1_output.device)
    
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
    coeffs = pywt.dwt2(img_np, 'haar')
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


# Integration function for your evaluation loop
def process_and_save_with_fusion(original_hr, coarse_hr, final_hr,upsampled_lr, 
                                  images_output_dir, batch_idx, cumulative_idx,
                                  wavelet='haar', save_dwt_viz=False):
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
    fused_hr = perform_dwt_fusion(coarse_hr, final_hr, wavelet=wavelet)
    
    # Step 3: Save fused images
    save_fused_images(
        fused_hr=fused_hr,
        output_base_dir=images_output_dir,
        batch_idx=batch_idx,
        start_idx=cumulative_idx - original_hr.shape[0]  # Use same indices
    )
    
    # Optional: Save DWT decomposition visualizations for first batch
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


# Example usage in your evaluation loop:
"""
Replace this line in your code:

    cumulative_idx = save_images_to_directories(
        original_hr=original_hr,
        coarse_hr=coarse_hr,
        final_hr=final_hr,
        output_base_dir=images_output_dir,
        batch_idx=batch_idx,
        start_idx=cumulative_idx
    )

With:

    cumulative_idx, fused_hr = process_and_save_with_fusion(
        original_hr=original_hr,
        coarse_hr=coarse_hr,
        final_hr=final_hr,
        images_output_dir=images_output_dir,
        batch_idx=batch_idx,
        cumulative_idx=cumulative_idx,
        wavelet='haar',  # Can also try 'db1', 'db2', 'sym2', 'coif1'
        save_dwt_viz=True  # Set to True to see DWT decomposition
    )
    
    # Optional: Calculate metrics for fused_hr too
    fused_metrics = calculate_metrics_batch(original_hr, fused_hr)
"""
