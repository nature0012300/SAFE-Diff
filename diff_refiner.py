import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import math
import csv
import numpy as np
import pandas as pd
import glob
import random
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from tqdm import tqdm
import matplotlib.pyplot as plt
from utils import calculate_metrics

from torch.optim.lr_scheduler import SequentialLR, LinearLR, ConstantLR, CosineAnnealingLR
from skimage.metrics import peak_signal_noise_ratio as compare_psnr
from skimage.metrics import structural_similarity as compare_ssim

import lpips
from stage_1 import ResidualPredictionNet,de_normalize
from data_setup import set_seed, create_dataloader, device

torch.cuda.empty_cache()


import os 
os.environ["TOKENIZERS_PARALLELISM"] = "false"

class SelfAttentionLayer(nn.Module):
    """A standard Self-Attention Layer using PyTorch's MultiheadAttention."""
    def __init__(self, num_heads, emb_dim):
        super().__init__()
        self.attention = nn.MultiheadAttention(emb_dim, num_heads, batch_first=True)

    def forward(self, x):
        """
        Input:
            x: (B, Seq_Len, Dim) - The input sequence.
        Output:
            (B, Seq_Len, Dim) - The sequence after self-attention.
        """
        # print(f"SelfAttentionLayer input shape: {x.shape}")
        attn_output, _ = self.attention(x, x, x)
        return attn_output

class CrossAttentionLayer(nn.Module):
    """A standard Cross-Attention Layer using PyTorch's MultiheadAttention."""
    def __init__(self, num_heads, emb_dim, d_cross):
        super().__init__()
        # kdim and vdim are set to the dimension of the context (d_cross).
        self.attention = nn.MultiheadAttention(emb_dim, num_heads, kdim=d_cross, vdim=d_cross, batch_first=True)

    def forward(self, x, context):
        """
        Input:
            x: (B, Seq_Len_Q, Dim_Q) - The query sequence (e.g., image features).
            context: (B, Seq_Len_KV, Dim_KV) - The key/value sequence (e.g., guidance).
        Output:
            (B, Seq_Len_Q, Dim_Q) - The sequence 'x' after attending to 'context'.
        """
        attn_output, _ = self.attention(x, context, context)
        return attn_output

class UNetSelfAttentionBlock(nn.Module):
    """A U-Net attention block that ONLY performs self-attention, WITHOUT a large MLP."""
    def __init__(self, num_heads, emb_dim):
        super().__init__()
        self.self_attn = SelfAttentionLayer(num_heads, emb_dim)
        self.norm = nn.LayerNorm(emb_dim)
        self.dropout = nn.Dropout(0.1)

    def forward(self, x): # Note: the 'context' argument is now gone
        # A simpler, more memory-efficient structure
        x = x + self.dropout(self.self_attn(self.norm(x)))
        return x

class UNetResNetBlock(nn.Module):
    """
    A ResNet block for the U-Net WITHOUT gating mechanism.
    Standard ResNet block with time embedding injection.
    """
    def __init__(self, in_channels: int, out_channels: int, time_emb_dim: int):
        super().__init__()
        
        # --- Time Embedding Projection ---
        self.time_mlp = nn.Sequential(
            nn.SiLU(),
            nn.Linear(time_emb_dim, out_channels)
        )
        
        # --- Convolutional Layers ---
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        
        # --- Normalization Layers ---
        self.norm1 = nn.GroupNorm(max(1, min(8, in_channels)), in_channels)
        self.norm2 = nn.GroupNorm(max(1, min(8, out_channels)), out_channels)

        # --- Activation Function ---
        self.activation = nn.SiLU()

        # --- Residual Connection ---
        if in_channels != out_channels:
            self.residual_conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        else:
            self.residual_conv = nn.Identity()

    def _forward_impl(self, x: torch.Tensor, time_emb: torch.Tensor) -> torch.Tensor:
        """
        Forward pass implementation.
        
        Args:
            x (torch.Tensor): Input tensor of shape [B, in_channels, H, W].
            time_emb (torch.Tensor): Time embedding tensor of shape [B, time_emb_dim].
            
        Returns:
            torch.Tensor: Output tensor of shape [B, out_channels, H, W].
        """
        
        # 1. First convolution block
        h = self.activation(self.norm1(x))
        h = self.conv1(h)
        
        # 2. Inject time embedding
        time_emb_proj = self.time_mlp(time_emb).unsqueeze(-1).unsqueeze(-1)
        h = h + time_emb_proj
        
        # 3. Second convolution (no gating)
        h = self.activation(self.norm2(h))
        h = self.conv2(h)
        
        # 4. Add the residual connection
        return h + self.residual_conv(x)

    def forward(self, x: torch.Tensor, time_emb: torch.Tensor) -> torch.Tensor:
        """Defines the forward pass."""
        return self._forward_impl(x, time_emb)
# 4. THE MAIN  U-NET
# ==============================================================================

class WindowAttention(nn.Module):
    """Window-based multi-head self attention with relative position bias."""
    def __init__(self, dim, window_size, num_heads):
        super().__init__()
        self.dim = dim
        self.window_size = window_size  # (Wh, Ww)
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5

        # Define relative position bias table
        self.relative_position_bias_table = nn.Parameter(
            torch.zeros((2 * window_size[0] - 1) * (2 * window_size[1] - 1), num_heads)
        )
        
        # Get pair-wise relative position index
        coords_h = torch.arange(self.window_size[0])
        coords_w = torch.arange(self.window_size[1])
        coords = torch.stack(torch.meshgrid(coords_h, coords_w, indexing='ij'))
        coords_flatten = torch.flatten(coords, 1)
        relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]
        relative_coords = relative_coords.permute(1, 2, 0).contiguous()
        relative_coords[:, :, 0] += self.window_size[0] - 1
        relative_coords[:, :, 1] += self.window_size[1] - 1
        relative_coords[:, :, 0] *= 2 * self.window_size[1] - 1
        relative_position_index = relative_coords.sum(-1)
        self.register_buffer("relative_position_index", relative_position_index)

        self.qkv = nn.Linear(dim, dim * 3,bias = True)
        self.proj = nn.Linear(dim, dim)
        self.softmax = nn.Softmax(dim=-1)
        
        nn.init.trunc_normal_(self.relative_position_bias_table, std=0.02)

    def forward(self, x, mask=None):
        """
        Args:
            x: (B*num_windows, N, C) where N = window_size[0] * window_size[1]
            mask: (num_windows, N, N) or None
        """
        B_, N, C = x.shape
        qkv = self.qkv(x).reshape(B_, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        q = q * self.scale
        attn = (q @ k.transpose(-2, -1))

        # Add relative position bias
        relative_position_bias = self.relative_position_bias_table[
            self.relative_position_index.view(-1)
        ].view(self.window_size[0] * self.window_size[1], 
               self.window_size[0] * self.window_size[1], -1)
        relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()
        attn = attn + relative_position_bias.unsqueeze(0)

        if mask is not None:
            nW = mask.shape[0]
            attn = attn.view(B_ // nW, nW, self.num_heads, N, N) + mask.unsqueeze(1).unsqueeze(0)
            attn = attn.view(-1, self.num_heads, N, N)
       
        attn = attn - attn.max(dim=-1, keepdim=True)[0]  # Prevent overflow
        attn = self.softmax(attn)

        x = (attn @ v).transpose(1, 2).reshape(B_, N, C)
        x = self.proj(x)
        return x


def window_partition(x, window_size):
    """
    Args:
        x: (B, H, W, C)
        window_size: (int, int)
    Returns:
        windows: (B*num_windows, window_size[0]*window_size[1], C)
    """
    B, H, W, C = x.shape
    x = x.view(B, H // window_size[0], window_size[0], W // window_size[1], window_size[1], C)
    windows = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, window_size[0] * window_size[1], C)
    return windows


def window_reverse(windows, window_size, H, W):
    """
    Args:
        windows: (B*num_windows, window_size[0]*window_size[1], C)
        window_size: (int, int)
        H: Height of image
        W: Width of image
    Returns:
        x: (B, H, W, C)
    """
    B = int(windows.shape[0] / (H * W / window_size[0] / window_size[1]))
    x = windows.view(B, H // window_size[0], W // window_size[1], window_size[0], window_size[1], -1)
    x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, H, W, -1)
    return x
def create_mask(H, W, window_size, shift_size, device):
    """Create attention mask for SW-MSA (torch.compile compatible)"""
    img_mask = torch.zeros((1, H, W, 1), device=device)
    h_slices = (slice(0, -window_size),
                slice(-window_size, -shift_size),
                slice(-shift_size, None))
    w_slices = (slice(0, -window_size),
                slice(-window_size, -shift_size),
                slice(-shift_size, None))
    cnt = 0
    for h in h_slices:
        for w in w_slices:
            img_mask[:, h, w, :] = cnt
            cnt += 1

    mask_windows = window_partition(img_mask, (window_size, window_size))
    mask_windows = mask_windows.view(-1, window_size * window_size)
    
    # FIX: Use torch.where instead of masked_fill to avoid in-place operations
    attn_mask = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2)
    
    # Create mask in a torch.compile-friendly way (no in-place ops)
    attn_mask = torch.where(
        attn_mask != 0,
        torch.full_like(attn_mask, -10.0),
        torch.full_like(attn_mask, 0.0)
    )
    
    return attn_mask

class SwinTransformerBlock(nn.Module):
    """Swin Transformer Block with Post-Norm and proper padding."""
    def __init__(self, dim, num_heads, window_size=8, shift_size=0, mlp_ratio=4.):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.window_size = window_size
        self.shift_size = shift_size
        self.mlp_ratio = mlp_ratio
 
        if self.shift_size >= self.window_size:
          self.shift_size = 0

        self.attn = WindowAttention(dim, (window_size, window_size), num_heads)
        
        # Post-Norm: LayerNorm after attention
        self.norm1 = nn.LayerNorm(dim)
        
        # MLP
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(dim, mlp_hidden_dim),
            nn.GELU(),
            nn.Dropout(0.1),  # <-- ADD THIS
            nn.Linear(mlp_hidden_dim, dim),
            nn.Dropout(0.1) 
        )
        
        # Post-Norm: LayerNorm after MLP
        self.norm2 = nn.LayerNorm(dim)
        self.register_buffer("attn_mask", None)

    def forward(self, x):
        """
        Args:
            x: (B, C, H, W)
        """
        B, C, H, W = x.shape
        
        # Pad if needed
        pad_h = (self.window_size - H % self.window_size) % self.window_size
        pad_w = (self.window_size - W % self.window_size) % self.window_size
        
        if pad_h > 0 or pad_w > 0:
            x = F.pad(x, (0, pad_w, 0, pad_h))
            _, _, H_pad, W_pad = x.shape
        else:
            H_pad, W_pad = H, W
        
        # CHANGE THIS SECTION - Always recreate mask, don't cache
        if self.shift_size > 0:
            # Don't check self.attn_mask - always create fresh
            attn_mask = create_mask(H_pad, W_pad, self.window_size, 
                                self.shift_size, x.device)
        else:
            attn_mask = None
        
        # Rest of the code stays the same...
        x = x.permute(0, 2, 3, 1).contiguous()
        shortcut = x

        x = self.norm1(x)
        
        if self.shift_size > 0:
            shifted_x = torch.roll(x, shifts=(-self.shift_size, -self.shift_size), dims=(1, 2))
        else:
            shifted_x = x

        x_windows = window_partition(shifted_x, (self.window_size, self.window_size))
        attn_windows = self.attn(x_windows, mask=attn_mask)
        shifted_x = window_reverse(attn_windows, (self.window_size, self.window_size), H_pad, W_pad)
        
        if self.shift_size > 0:
            x_out = torch.roll(shifted_x, shifts=(self.shift_size, self.shift_size), dims=(1, 2))
        else:
            x_out = shifted_x
        
        x = shortcut + x_out

        x = x + self.mlp(self.norm2(x))
           
        x = x.permute(0, 3, 1, 2).contiguous()
        
        if pad_h > 0 or pad_w > 0:
            x = x[:, :, :H, :W]
        
        return x


class SwinTransformerStage(nn.Module):
    """Multiple Swin Transformer blocks - THIS WAS MISSING!"""
    def __init__(self, dim, depth, num_heads, window_size=8, mlp_ratio=4.):
        super().__init__()
        self.blocks = nn.ModuleList([
            SwinTransformerBlock(
                dim=dim,
                num_heads=num_heads,
                window_size=window_size,
                shift_size=0 if (i % 2 == 0) else window_size // 2,
                mlp_ratio=mlp_ratio
            )
            for i in range(depth)
        ])

    def forward(self, x):
        for block in self.blocks:
            x = block(x)
        return x


class DenoisingUNet(nn.Module):
    """
    Refined U-Net: 
    1. Injects time embeddings at every level.
    2. Scales num_heads logically with channel depth.
    3. Uses your UNetResNetBlock for time-aware feature extraction.
    """
    def __init__(self, in_channels, out_channels, time_emb_dim):
        super().__init__()
        
        # --- Encoder ---
        # Level 1: 64 channels (No Swin here to keep high-res features local)
        self.enc1 = UNetResNetBlock(in_channels, 64, time_emb_dim)
        
        # Level 2: 128 channels, 4 heads
        self.down2 = nn.Conv2d(64, 128, 3, stride=2, padding=1)
        self.enc2_res = UNetResNetBlock(128, 128, time_emb_dim)
        self.swin_enc2 = SwinTransformerStage(128, depth=1, num_heads=4, window_size=8)
        
        # Level 3: 256 channels, 8 heads
        self.down3 = nn.Conv2d(128, 256, 3, stride=2, padding=1)
        self.enc3_res = UNetResNetBlock(256, 256, time_emb_dim)
        self.swin_enc3 = SwinTransformerStage(256, depth=2, num_heads=8, window_size=8)
        
        # --- Bottleneck ---
        # Level 4: 512 channels, 16 heads
        self.down4 = nn.Conv2d(256, 512, 3, stride=2, padding=1)
        self.bottleneck_res = UNetResNetBlock(512, 512, time_emb_dim)
        self.swin_bottleneck = SwinTransformerStage(512, depth=2, num_heads=16, window_size=8)
        
        # --- Decoder ---
        # Up 3
        self.up3 = nn.ConvTranspose2d(512, 256, 2, stride=2)
        self.dec3_res = UNetResNetBlock(512, 256, time_emb_dim) # 256 (up) + 256 (skip)
        self.swin_dec3 = SwinTransformerStage(256, depth=2, num_heads=8, window_size=8)
        
        # Up 2
        self.up2 = nn.ConvTranspose2d(256, 128, 2, stride=2)
        self.dec2_res = UNetResNetBlock(256, 128, time_emb_dim) # 128 (up) + 128 (skip)
        self.swin_dec2 = SwinTransformerStage(128, depth=1, num_heads=4, window_size=8)
        
        # Up 1
        self.up1 = nn.ConvTranspose2d(128, 64, 2, stride=2)
        self.dec1_res = UNetResNetBlock(128, 64, time_emb_dim) # 64 (up) + 64 (skip)
        
        self.final = nn.Conv2d(64, out_channels, 1)

    def forward(self, x, time_emb):
        # Encoder
        e1 = self.enc1(x, time_emb)
        
        e2 = self.down2(e1)
        e2 = self.enc2_res(e2, time_emb)
        e2 = self.swin_enc2(e2)
        
        e3 = self.down3(e2)
        e3 = self.enc3_res(e3, time_emb)
        e3 = self.swin_enc3(e3)
        
        # Bottleneck
        b = self.down4(e3)
        b = self.bottleneck_res(b, time_emb)
        b = self.swin_bottleneck(b)
        
        # Decoder
        d3 = self.up3(b)
        d3 = torch.cat([d3, e3], dim=1) # Skip connection
        d3 = self.dec3_res(d3, time_emb)
        d3 = self.swin_dec3(d3)
        
        d2 = self.up2(d3)
        d2 = torch.cat([d2, e2], dim=1) # Skip connection
        d2 = self.dec2_res(d2, time_emb)
        d2 = self.swin_dec2(d2)
        
        d1 = self.up1(d2)
        d1 = torch.cat([d1, e1], dim=1) # Skip connection
        d1 = self.dec1_res(d1, time_emb)
        
        return self.final(d1)


class Stage2Model(nn.Module):
    """The main model that orchestrates all components for Stage 2."""
    def __init__(self, time_emb_dim: int, out_emb_dim: int, device, stage1_model_path):
        super().__init__()
        # Define model dimensions
        self.time_emb_dim = time_emb_dim
        self.out_emb_dim = out_emb_dim
        
        # Load Stage 1 model for residual prediction
        self.stage1_model = load_stage1_model(stage1_model_path, device)
        
        # Time Embedding MLP
        self.time_mlp = nn.Sequential(
            nn.Linear(self.time_emb_dim, self.time_emb_dim * 4),
            nn.SiLU(),
            nn.Linear(self.time_emb_dim * 4, self.time_emb_dim)
        )
        
        self.unet = DenoisingUNet(
            in_channels=2,
            out_channels=1,
            time_emb_dim=self.time_emb_dim
        )

    def forward(self, noisy_hr_image, downsampled_image, time_steps, residual_map, epoch: int = 0):
        """
        The main forward pass that orchestrates the entire Stage 2 process.
        Input:
            noisy_hr_image: (B, 1, H, W) - The noisy HR image at time t.
            downsampled_image: (B, 1, H_lr, W_lr) - The LR image with artifacts.
            time_steps: (B,) - The current timestep t for each image in the batch.
            residual_map: (B, 1, H_res, W_res) - The residual map from Stage 1.
            epoch: Current training epoch.
        Output:
            (B, 1, H, W) - The predicted noise to be removed from the image.
        """
        # 1. Get time embedding
        time_emb = self.get_time_embedding(time_steps)  # -> (B, 128)
        time_emb = self.time_mlp(time_emb)  # -> (B, 128)
        
        # 2. Upsample LR image to HR size
        upsampled_image = F.interpolate(
            downsampled_image,
            size=noisy_hr_image.shape[-2:],
            mode='bicubic',
            align_corners=False
        )  # -> (B, 1, H, W)
        
        # 3. Get residual map from Stage 1 if not provided
        if residual_map is None:
            with torch.no_grad():
                residual_map = self.stage1_model(upsampled_image)
        
        # 4. Create coarse HR image
        coarse_hr = torch.clamp(upsampled_image + residual_map, -1.0, 1.0)
        
        # 5. Concatenate inputs for U-Net
        unet_input = torch.cat([noisy_hr_image, coarse_hr], dim=1)
        
        # 6. Denoise using the U-Net with Swin Transformer attention
        predicted_noise = self.unet(unet_input, time_emb)
        
        return predicted_noise

    def get_time_embedding(self, timesteps):
        """Sinusoidal time embedding."""
        half_dim = self.time_emb_dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=timesteps.device, dtype=torch.float32) * -emb)
        emb = timesteps.float()[:, None] * emb[None, :]
        time_emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=1)
        return time_emb

# 6. TRAINING SCRIPT 

class NoiseScheduler:
    """A cosine noise scheduler for smoother transitions."""
    def __init__(self, num_timesteps=1000, beta_start=0.0001, beta_end=0.015, device=device):
        self.num_timesteps = num_timesteps
        self.device = device
        
        # Generate cosine schedule
        self.betas = self.cosine_beta_schedule(num_timesteps, beta_start, beta_end).to(device)
        self.alphas = 1.0 - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, dim=0).to(device)
    
    def cosine_beta_schedule(self, timesteps, beta_start, beta_end, s=0.008):
        """
        Creates a cosine schedule for beta values.
        s: small offset to prevent beta from being 0 at the beginning.
        """
        steps = timesteps + 1
        x = torch.linspace(0, timesteps, steps, dtype=torch.float32)
        
        # Cosine schedule for alphas_cumprod
        alphas_cumprod = torch.cos(((x / timesteps) + s) / (1 + s) * torch.pi * 0.5) ** 2
        alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
        
        # Calculate betas from alphas_cumprod
        betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
        
        # Clip betas to reasonable range and respect beta_start/beta_end bounds
        betas = torch.clip(betas, beta_start, beta_end)
        
        return betas
    
    # Correcting the add_noise function from your script as well
    def add_noise(self, x_0, noise, t):
        """Same noise addition function as your original."""
        sqrt_alpha_cumprod_t = torch.sqrt(self.alphas_cumprod[t]).view(-1, 1, 1, 1).to(x_0.device)
        sqrt_one_minus_alpha_cumprod_t = torch.sqrt(1.0 - self.alphas_cumprod[t]).view(-1, 1, 1, 1).to(x_0.device)
        return sqrt_alpha_cumprod_t * x_0 + sqrt_one_minus_alpha_cumprod_t * noise


def load_stage1_model(checkpoint_path, device):
    """Loads the pre-trained Stage 1 model and sets it to evaluation mode."""
    print(f"Loading pre-trained Stage 1 model from: {checkpoint_path}")
    stage1_params = {"in_channels": 1, "out_channels": 1, "feature_channels": 64, "n_res_blocks": 20, "upscale_factor": 4}
    model = ResidualPredictionNet(**stage1_params).to(device)

    ## loading the stage_1 model 
    saved_dict = torch.load(checkpoint_path, map_location=device)
    model_dict = model.state_dict()

    # Create a new dict with only matching keys
    filtered_dict = {}
    for key in model_dict.keys():
        if key in saved_dict:
            if model_dict[key].shape == saved_dict[key].shape:
                filtered_dict[key] = saved_dict[key]
            else:
                print(f"Shape mismatch for {key}: model {model_dict[key].shape} vs saved {saved_dict[key].shape}")
        else:
            print(f"Key {key} not found in saved model")

    model_dict.update(filtered_dict)
    model.load_state_dict(model_dict)
    print(f"Loaded {len(filtered_dict)} parameters from saved model")

    model.eval()
    for param in model.parameters():
        param.requires_grad = False
    print("Stage 1 model loaded successfully and frozen.")
    return model


class MetricsLogger:
    """
    A utility class to log training and validation metrics to a single CSV file.
    """
    def __init__(self, log_path='Directory_to_save_outputs/metrics_log.csv'):
        """
        Initializes the logger.
        Args:
            log_path (str): The full path to the log file.
        """
        self.log_path = log_path
        self.log_dir = os.path.dirname(log_path)
        
        # Create the directory if it doesn't exist
        os.makedirs(self.log_dir, exist_ok=True)
        
        # Write the header only if the file is new
        if not os.path.exists(self.log_path):
            with open(self.log_path, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow([
                    'epoch', 'phase', 'total_loss', 'diffusion_loss', "recon_loss",
                    'lpips_loss', 'psnr', 'ssim', 'nmse'
                ])

    def log(self, epoch, phase, metrics):
        """
        Appends a new row of metrics to the log file.
        Args:
            epoch (int): The current epoch number.
            phase (str): The phase ('train', 'val', or 'test').
            metrics (dict): A dictionary containing the metric names and their values.
        """
        with open(self.log_path, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([
                epoch,
                phase,
                metrics.get('total_loss', float('nan')),
                metrics.get('diffusion_loss', float('nan')),
                metrics.get('lpips_loss', float('nan')),
                metrics.get('recon_loss', float('nan')),
                metrics.get('psnr', float('nan')),
                metrics.get('ssim', float('nan')),
                metrics.get('nmse', float('nan'))
            ])

def visualize_batch_s2(model, batch, epoch, device, noise_scheduler,output_dir):
    """
    Generates and saves a visualization for a given batch of images from the Stage 2 model.
    It performs a single-step reconstruction at a fixed intermediate timestep to show progress.
    """
    model.eval()  # Ensure model is in eval mode for inference
    vis_dir = os.path.join(output_dir, "visualizations_s2")
    os.makedirs(vis_dir, exist_ok=True)
    
    # Unpack the batch and move tensors to the device
    original_hr, downsampled_lr = batch
    original_hr = original_hr.float().to(device)
    downsampled_lr = downsampled_lr.float().to(device)
    upsampled_lr = F.interpolate(downsampled_lr, size=original_hr.shape[-2:], mode='bicubic', align_corners=False).to(device)
    
    with torch.no_grad(), torch.amp.autocast('cuda'): 
        # --- 1. Perform a single-step reconstruction for visualization ---
        # t = torch.randint(0, noise_scheduler.num_timesteps, (original_hr.shape[0],), device=device)
        t = torch.randint(0,300, (original_hr.shape[0],), device=device)
        residual_map = model.stage1_model(upsampled_lr).float().to(device)
        coarse_hr = torch.clamp(upsampled_lr + residual_map, -1.0, 1.0)
        # Generate the noisy image that the model will denoise
        
        actual_noise = torch.randn_like(original_hr)

        noisy_image = noise_scheduler.add_noise(original_hr, actual_noise, t)
        noisy_input_plot = (noisy_image.cpu() + 1) / 2.0

        # Get the model's prediction for the noise
        predicted_noise = model(noisy_image, downsampled_lr, t,residual_map,epoch= epoch)
        
        # Use the DDPM formula to get the predicted x_0 (the sr image)
        sqrt_alpha_t = torch.sqrt(noise_scheduler.alphas_cumprod[t]).view(-1, 1, 1, 1)
        sqrt_one_minus_alpha_t = torch.sqrt(1.0 - noise_scheduler.alphas_cumprod[t]).view(-1, 1, 1, 1)
        sr_image = (noisy_image - sqrt_one_minus_alpha_t * predicted_noise) / sqrt_alpha_t
        sr_image = torch.clamp(sr_image, -1.0, 1.0)
    # --- 2. Prepare images for plotting (move to CPU and de-normalize) ---
    # De-normalize from [-1, 1] to [0, 1] for display
    original_hr_plot = (original_hr.cpu() + 1) / 2.0
    sr_plot = (sr_image.cpu() + 1) / 2.0
    residual_map_plot = (residual_map.cpu() + 1) / 2.0
    coarse_hr_plot = (coarse_hr.cpu() + 1) / 2.0

    # Upsample the LR image for a side-by-side visual comparison
    upsampled_lr_plot = (upsampled_lr.cpu() + 1) / 2.0
    
    
    # --- 3. Create and save the plot ---
    num_images_to_show = min(original_hr.shape[0], 1)  # Show up to 4 images
    fig, axes = plt.subplots(num_images_to_show, 7, figsize=(25, 6 * num_images_to_show))
    if num_images_to_show == 1:
        axes = axes.reshape(1, -1) # Adjust shape for single-row subplots

    for i in range(num_images_to_show):
        axes[i, 0].imshow(torch.clamp(upsampled_lr_plot[i, 0], 0, 1), cmap='gray')
        axes[i, 0].set_title("Input (Upsampled LR)")
        axes[i, 0].axis('off')

        axes[i, 1].imshow(torch.clamp(residual_map_plot[i, 0], 0, 1), cmap='gray')
        axes[i, 1].set_title("Predicted Residual Map (Stage 1)")
        axes[i, 1].axis('off')

        axes[i, 2].imshow(torch.clamp(noisy_input_plot[i, 0], 0, 1), cmap='gray')
        axes[i, 2].set_title(f"Noisy Input (t={t[i].item()})")
        axes[i, 2].axis('off')
        
        axes[i, 3].imshow(torch.clamp(coarse_hr_plot[i, 0], 0, 1), cmap='gray')
        axes[i, 3].set_title("Coarse HR Image (Stage 1 Output)")
        axes[i, 3].axis('off')
        
        axes[i, 4].imshow(torch.clamp(sr_plot[i, 0], 0, 1), cmap='gray')
        axes[i, 4].set_title("SuperResolved Image (1-Step)")
        axes[i, 4].axis('off')
        
        axes[i, 5].imshow(torch.clamp(original_hr_plot[i, 0], 0, 1), cmap='gray')
        axes[i, 5].set_title("Original Ground Truth")
        axes[i, 5].axis('off')

        # Display the timestep used for this reconstruction
        t_val_current = t[i].item() 
        axes[i, 6].text(0.5, 0.5, f"t = {t_val_current}", ha='center', va='center', fontsize=20, color='blue')
        axes[i, 6].axis('off')
        
    fig.suptitle(f'Validation Visualization - Epoch {epoch+1}', fontsize=24)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    save_path = os.path.join(vis_dir, f"epoch_{epoch+1:03d}_visualization.png")
    plt.savefig(save_path)
    plt.close(fig)

    model.train() # Switch back to training mode after visualization


class CharbonnierLoss(nn.Module):
    """Robust L1 Loss equivalent."""
    def __init__(self, eps=1e-3):
        super(CharbonnierLoss, self).__init__()
        self.eps = eps

    def forward(self, x, y):
        diff = x - y
        loss = torch.sqrt(diff * diff + self.eps * self.eps)
        return torch.mean(loss)

class FrequencyLoss(nn.Module):
    """Simple Spectral Amplitude Loss."""
    def __init__(self):
        super(FrequencyLoss, self).__init__()
        self.l1_loss = nn.L1Loss()

    def forward(self, pred, target):
        # 1. Compute 2D FFT
        # RFFT2 is faster for real-valued inputs
        pred_fft = torch.fft.rfft2(pred, norm='ortho')
        target_fft = torch.fft.rfft2(target, norm='ortho')
        
        # 2. Extract Amplitude (Magnitude)
        pred_amp = torch.abs(pred_fft)
        target_amp = torch.abs(target_fft)
        
        # 3. Calculate distance in Frequency Domain
        return self.l1_loss(pred_amp, target_amp)


def train_one_epoch_s2(model, dataloader, optimizer, noise_scheduler, criterion_l1, lpips_loss_fn,criterion_char,
                     w_char,device, epoch, scaler):
    """
    Handles the training logic for one complete epoch with full reproducibility.
    """
    lpips_weight = 0.35
    model.train()
    model.to(device)
    train_metrics = {
        "total_loss": [], "diffusion_loss": [], "lpips_loss": [], 'recon_loss': [] ,
        "psnr": [], "ssim": [], "nmse": []
    }
    base_seed = 42  # Base seed for reproducibility
    num_batches = len(dataloader)
    loop = tqdm(enumerate(dataloader), total=num_batches, desc=f"Training Epoch {epoch+1}",mininterval=1.0)

    for batch_idx, (original, downsampled_image) in loop:
        
        # --- REPRODUCIBILITY STEP ---
        # Set a unique, deterministic seed for this specific training step.
        step_seed = base_seed + epoch * num_batches + batch_idx
        set_seed(step_seed)
        
        original = original.float().to(device)

        downsampled_image = downsampled_image.float().to(device) 
        upsampled_image = F.interpolate(downsampled_image, 
                                      size=original.shape[-2:],
                                       mode='bicubic', align_corners=False) # -> (B, 1, H, W)
        upsampled_image = upsampled_image.to(device)
        with torch.no_grad(), torch.amp.autocast('cuda'):  # ← ADD autocast here
            residual_map = model.stage1_model(upsampled_image).float()  # ← ADD .float()
        residual_map = residual_map.to(device)
        coarse_hr = torch.clamp(upsampled_image + residual_map, -1.0, 1.0)
  
        # accumulation_steps = 4
        with torch.amp.autocast('cuda'):
            # --- 1. Diffusion Forward Pass ---
        
            t = torch.randint(0,300, (original.shape[0],), device=device)
               
            actual_noise = torch.randn_like(original)
            
            noisy_original = noise_scheduler.add_noise(original, actual_noise, t)
    

            # --- 2. Model Prediction ---
            predicted_noise = model(noisy_hr_image=noisy_original, downsampled_image=downsampled_image,
                            time_steps=t,residual_map=residual_map,epoch=epoch)

            # --- 3. Loss Calculation ---
            diffusion_loss = criterion_l1(predicted_noise, actual_noise)
            
            sqrt_alpha_t = torch.sqrt(noise_scheduler.alphas_cumprod[t]).view(-1, 1, 1, 1)
            sqrt_one_minus_alpha_t = torch.sqrt(1.0 - noise_scheduler.alphas_cumprod[t]).view(-1, 1, 1, 1)
            predicted_hr = (noisy_original - sqrt_one_minus_alpha_t * predicted_noise) / sqrt_alpha_t
            predicted_hr = torch.clamp(predicted_hr, -1.0, 1.0)
            
            del noisy_original

            lpips_loss = lpips_loss_fn(predicted_hr, original).mean()
            char_loss = criterion_char(predicted_hr, original)

            total_loss = (1- lpips_weight - w_char) * (diffusion_loss) + (lpips_weight * lpips_loss) + (w_char * char_loss)

        if torch.isnan(total_loss) or torch.isinf(total_loss):
            print(f"NaN/Inf loss at batch {batch_idx}, epoch {epoch+1}, skipping batch")
            optimizer.zero_grad()
            continue

        # Check for NaN in predicted_hr before metrics
        if torch.isnan(predicted_hr).any() or torch.isinf(predicted_hr).any():
            print(f"⚠️  NaN/Inf in predicted_hr at batch {batch_idx}, epoch {epoch+1}, skipping batch")
            optimizer.zero_grad()
            continue
            
        # --- 4. Backpropagation ---
        scaler.scale(total_loss).backward()
        scaler.unscale_(optimizer)
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad()
        
        if (batch_idx + 1) % (100) == 0: 
            print(f"Gradient norm at step {batch_idx+1}: {grad_norm:.4f}")
            # -----------------------------------
            
        # --- 5. Update Metrics ---
        train_metrics["total_loss"].append(total_loss.item())
        train_metrics["lpips_loss"].append(lpips_loss.item())
        train_metrics["diffusion_loss"].append(diffusion_loss.item())
        train_metrics["recon_loss"].append(char_loss.item())

        del predicted_noise, actual_noise
        
        predicted_hr_detached = predicted_hr.detach()
        for i in range(predicted_hr_detached.shape[0]):
            psnr, ssim, nmse = calculate_metrics(original[i], predicted_hr_detached[i])
            if psnr is not None:
                train_metrics["psnr"].append(psnr)
                train_metrics["ssim"].append(ssim)
                train_metrics["nmse"].append(nmse)
        
                
    return {key: np.mean(val) for key, val in train_metrics.items() if val}

def validate_one_epoch_s2(model, dataloader, noise_scheduler, criterion_l1, lpips_loss_fn,criterion_char,
                       w_char, device, epoch, base_seed,output_dir):
    """
    Handles the validation logic for one complete epoch.
    """
    lpips_weight = 0.35
    model.eval() 
    val_metrics = {
        "total_loss": [],
        "diffusion_loss": [],
        "lpips_loss": [],
        'recon_loss':[],
        "psnr": [],
        "ssim": [],
        "nmse": []
    }

    num_batches = len(dataloader)
    loop = tqdm(enumerate(dataloader), total=num_batches, desc=f"Validating Epoch {epoch+1}",mininterval=1.0)

    with torch.no_grad(), torch.amp.autocast('cuda'): 
        for batch_idx, (original,downsampled_image) in loop:

            original = original.float().to(device) 
            downsampled_image = downsampled_image.float().to(device)
            upsampled_image = F.interpolate(downsampled_image, 
                                      size=original.shape[-2:],
                                       mode='bicubic', align_corners=False)
            upsampled_image = upsampled_image.float().to(device)
            with torch.amp.autocast('cuda'):  
                residual_map = model.stage1_model(upsampled_image).float()  
            residual_map = residual_map.to(device)
            coarse_hr = torch.clamp(upsampled_image + residual_map, -1.0, 1.0)
            
            # --- 1. Diffusion Forward Pass ---
            t = torch.randint(0, 300, (original.shape[0],), device=device)
            actual_noise = torch.randn_like(original)
       
            noisy_original = noise_scheduler.add_noise(original, actual_noise, t)

            # --- 2. Model Prediction ---
            predicted_noise = model(noisy_hr_image=noisy_original, downsampled_image=downsampled_image,
                            time_steps=t,residual_map=residual_map,epoch=epoch)

            # --- 3. Loss Calculation
            diffusion_loss = criterion_l1(predicted_noise, actual_noise)

            # Derive the reconstructed image for LPIPS and image quality metrics
            sqrt_alpha_t = torch.sqrt(noise_scheduler.alphas_cumprod[t]).view(-1, 1, 1, 1)
            sqrt_one_minus_alpha_t = torch.sqrt(1.0 - noise_scheduler.alphas_cumprod[t]).view(-1, 1, 1, 1)
            predicted_hr = (noisy_original - sqrt_one_minus_alpha_t * predicted_noise) / sqrt_alpha_t
            predicted_hr = torch.clamp(predicted_hr, -1.0, 1.0)

            del noisy_original

            lpips_loss = lpips_loss_fn(predicted_hr, original).mean()
            char_loss = criterion_char(predicted_hr, original)

            total_loss = (1- lpips_weight - w_char) * (diffusion_loss) + (lpips_weight * lpips_loss) + (w_char * char_loss) 
            # --- 4. Update Metrics ---
            val_metrics["total_loss"].append(total_loss.item())
            val_metrics["lpips_loss"].append(lpips_loss.item())
            val_metrics["diffusion_loss"].append(diffusion_loss.item())
            val_metrics["recon_loss"].append(char_loss.item())

            del predicted_noise, actual_noise

            # Calculate image quality metrics
            for i in range(predicted_hr.shape[0]):
                psnr, ssim, nmse = calculate_metrics(original[i], predicted_hr[i])
                if psnr is not None:
                    val_metrics["psnr"].append(psnr)
                    val_metrics["ssim"].append(ssim)
                    val_metrics["nmse"].append(nmse)

            if batch_idx % 20 == 0:
                torch.cuda.empty_cache()
    
    try:
        vis_batch = next(iter(dataloader))
        visualize_batch_s2(
            model=model,
            batch=vis_batch,
            epoch=epoch,
            device=device,
            noise_scheduler=noise_scheduler,
            output_dir=output_dir
        )
    except StopIteration:
        print("Warning: Could not get a batch from val_dataloader for visualization.")
    
    return {key: np.mean(val) for key, val in val_metrics.items() if val}

def train_and_validate(config, model, train_dataloader, val_dataloader,scheduler, optimizer, noise_scheduler, 
                      criterion_l1, lpips_loss_fn,criterion_char,w_char,stage1_model_path,device):
    """
    The main orchestrator for the training and validation process, adapted for the
    diffusion model pipeline.
    """
    # Initialize the tools for logging and checkpointing
    model.noise_scheduler = noise_scheduler
    logger = MetricsLogger(log_path=config['log_path'])
    best_val_loss = float('inf')  # Initialize with infinity
    best_epoch = 0
    # These lists will store the average metrics for each epoch for a final summary
    results_summary = {
        "train_total_loss": [], "val_total_loss": [],
        "train_psnr": [], "val_psnr": [],
        "train_ssim": [], "val_ssim": [],
        "train_nmse": [], "val_nmse": []
    }
    scaler = torch.amp.GradScaler('cuda')
    
    print("\n--- Starting Training and Validation ---")
    
    # The main loop over epochs
    for epoch in range(config['num_epochs']):
        current_lr = optimizer.param_groups[0]['lr']

        print(f"\n===== Epoch {epoch+1}/{config['num_epochs']} =====")
        print(f"Learning Rate: {current_lr:.6f}")  
    

        # --- Training Phase ---
        train_metrics = train_one_epoch_s2(
            model=model,
            dataloader=train_dataloader,
            optimizer=optimizer,
            noise_scheduler=noise_scheduler,
            criterion_l1=criterion_l1,
            lpips_loss_fn=lpips_loss_fn,
            criterion_char = criterion_char,
            w_char= w_char,
            device=device,
            epoch=epoch,
            scaler=scaler 
        )
        # Log and store the averaged training metrics for the epoch
        logger.log(epoch + 1, 'train', train_metrics)
        results_summary["train_total_loss"].append(train_metrics['total_loss'])
        results_summary["train_psnr"].append(train_metrics['psnr'])
        results_summary["train_ssim"].append(train_metrics['ssim'])
        results_summary["train_nmse"].append(train_metrics['nmse'])
        print(f"\nTrain Metrics: Loss={train_metrics['total_loss']:.4f}, PSNR={train_metrics['psnr']:.2f}, SSIM={train_metrics['ssim']:.4f}, NMSE={train_metrics['nmse']:.4f}")

        # --- Validation Phase ---
        val_metrics = validate_one_epoch_s2(
            model=model,
            dataloader=val_dataloader,
            noise_scheduler=noise_scheduler,
            criterion_l1=criterion_l1,
            lpips_loss_fn=lpips_loss_fn,
            criterion_char = criterion_char,
            w_char=w_char,
            device=device,
            epoch=epoch,
            base_seed=config['base_seed'],
            output_dir=config['output_dir']
        )
        scheduler.step()
        # Log and store the averaged validation metrics for the epoch
        logger.log(epoch + 1, 'val', val_metrics)
        results_summary["val_total_loss"].append(val_metrics['total_loss'])
        results_summary["val_psnr"].append(val_metrics['psnr'])
        results_summary["val_ssim"].append(val_metrics['ssim'])
        results_summary["val_nmse"].append(val_metrics['nmse'])
        print(f"\nVal Metrics:  Loss={val_metrics['total_loss']:.4f}, PSNR={val_metrics['psnr']:.2f}, SSIM={val_metrics['ssim']:.4f}, NMSE={val_metrics['nmse']:.4f}")

       
        avg_val_loss = val_metrics['total_loss']
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            best_epoch = epoch + 1
            # Save the best model
            best_model_path = os.path.join(config['output_dir'], 'best_model.pth')
            torch.save(model.state_dict(), best_model_path)
            print(f" New best model saved! Epoch {best_epoch}, Val Loss: {best_val_loss:.4f}")
        
        if (epoch + 1) % 10 == 0:  # Save every 10 epochs
            checkpoint_path = os.path.join(config['output_dir'], f'model_epoch_{epoch+1}.pth')
            torch.save(model.state_dict(), checkpoint_path)
            print(f"Model saved at epoch {epoch+1}: {checkpoint_path}")   

    print("\n--- Training Finished ---")
   
    print(f"Best model (Epoch {best_epoch}, Val Loss: {best_val_loss:.4f}) saved to: {os.path.join(config['output_dir'], 'best_model.pth')}")
    print(f"All epoch metrics logged to: {config['log_path']}")
    
    # Return the summary of all epochs
    return results_summary

def plot_metrics_from_csv(csv_path, output_dir):
    """
    Reads training and validation metrics from a CSV log file and generates plots.
    
    Args:
        csv_path (str): The path to the metrics CSV log file.
        output_dir (str): The directory where the plot images will be saved.
    """
    print(f"Reading metrics from {csv_path}...")
    column_names = [
        'epoch', 'phase', 'total_loss', 'diffusion_loss','recon_loss', 
        'lpips_loss', 'psnr', 'ssim', 'nmse']
    try:
        # Tell pandas to use the names you provided and that there is no header row in the file.
        df = pd.read_csv(csv_path)
    except FileNotFoundError:
        print(f"Error: Log file not found at {csv_path}. Skipping plotting.")
        return
    except pd.errors.EmptyDataError:
        print(f"Error: Log file is empty at {csv_path}. Skipping plotting.")
        return

    # Separate the data into training and validation sets
    train_df = df[df['phase'] == 'train'].sort_values(by='epoch')
    val_df = df[df['phase'] == 'val'].sort_values(by='epoch')

    # A helper dictionary to define what to plot and how to label it.
    # The key is the plot title/Y-axis label, the value is the column name in the CSV.
    metrics_to_plot = {
        "Total Loss": "total_loss",
        "Diffusion Loss": "diffusion_loss", 
        "LPIPS Loss": "lpips_loss",         
        "Reconstruction Loss": "recon_loss",     
        "PSNR (dB)": "psnr",
        "SSIM": "ssim",
        "NMSE": "nmse"
    }
    
    for title, column_name in metrics_to_plot.items():
        plt.figure(figsize=(10, 6))
        
        # Plot training data
        plt.plot(train_df['epoch'], train_df[column_name], 'b-o', label=f'Train {title}')
        
        # Plot validation data
        plt.plot(val_df['epoch'], val_df[column_name], 'r-o', label=f'Validation {title}')
        
        plt.title(f'{title} vs. Epochs')
        plt.xlabel("Epoch")
        plt.ylabel(title)
        plt.legend()
        plt.grid(True)
        
        # Generate a clean filename for the plot
        filename = title.lower().replace(' (db)', '').replace(' ', '_') + "_final_plot.png"
        save_path = os.path.join(output_dir, filename)
        
        plt.savefig(save_path)
        print(f"Saved {title} plot to {save_path}")
        plt.close()



if __name__ == '__main__':
    # --- 2. Configuration ---

    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    # torch.backends.cudnn.benchmark = True

    config = {
        "num_epochs": 80,
        "batch_size": 8,
        "base_seed": 42,
        "learning_rate": 1.2e-4,
        "w_char": 0.4,
        "time_emb_dim": 128, 
        "emb_dim": 512,       # Must match ReportEncoder output
        "out_emb_dim": 512,  # Dimension for cross-attention
        "stage1_model_path": "path/to/trained/stage_1.pth", 
        "output_dir": "Directory_to_save_outputs",  
        "log_path": "Directory_to_save_outputs/metrics_log_s2.csv",
        "save_path": "Directory_to_save_outputs/stage2_model_final.pth"
    }
    os.makedirs(config['output_dir'], exist_ok=True)

    # --- 3. Setup Device, Seed, and Dataloaders ---
    set_seed(config['base_seed'])
    print(f"Using device: {device}")

    train_dataloader = create_dataloader(csv_file="path/to/your/train_data.csv",
                                        max_items = None, batch_size=config['batch_size'], shuffle=True)
    val_dataloader = create_dataloader(csv_file="path/to/your/val_data.csv",
                                        max_items = None, batch_size=config['batch_size'], shuffle=False)
        
    # --- 4. Initialize Models, Optimizer, and Loss Functions ---
    print("--- Initializing models and training components ---")
    model = Stage2Model(
        time_emb_dim=config['time_emb_dim'],
        out_emb_dim=config['out_emb_dim'],
        device=device,
        stage1_model_path = config['stage1_model_path']
    ).to(device)

    torch.cuda.empty_cache()
    
    if hasattr(torch, 'compile'):
        print(" Compiling model with torch.compile()...")
        model = torch.compile(model, mode='reduce-overhead')
        print("✓ Model compiled successfully! First epoch will be slower due to compilation.")
    else:
        print(" torch.compile() not available (requires PyTorch 2.0+). Skipping compilation.")

    optimizer = torch.optim.AdamW(model.parameters(), lr=config['learning_rate'],weight_decay=0.01)

    warmup_scheduler = LinearLR(
        optimizer,
        start_factor=0.05,
        end_factor=1.0,
        total_iters=5
    )

    decay_scheduler = CosineAnnealingLR(
        optimizer,
        T_max=config['num_epochs'] - 5,     
        eta_min=8e-6  
    )

    scheduler = SequentialLR(
        optimizer,
        schedulers=[warmup_scheduler, decay_scheduler],
        milestones=[5]
    )   
  
    noise_scheduler = NoiseScheduler(device = device)
    
    criterion_l1 = nn.MSELoss().to(device) 

    # 2. Perceptual Loss (Keep LPIPS)
    lpips_loss_fn = lpips.LPIPS(net='alex').to(device)

    # 3. New Refiner Losses
    criterion_char = CharbonnierLoss().to(device)

    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Trainable parameters: {trainable:,}")
    # --- 5. Run Training and Validation ---
    training_results = train_and_validate(
        config=config,
        model=model,
        train_dataloader=train_dataloader,
        val_dataloader=val_dataloader,
        scheduler=scheduler,
        optimizer=optimizer,
        noise_scheduler=noise_scheduler,
        criterion_l1=criterion_l1,
        lpips_loss_fn=lpips_loss_fn,
        criterion_char=criterion_char,
        w_char=config['w_char'],
        stage1_model_path=config['stage1_model_path'],
        device=device
    )
    torch.cuda.empty_cache() 
    # --- 6. Plot Final Metrics ---
    print("\n--- Plotting final training metrics ---")
    plot_metrics_from_csv(
        csv_path=config['log_path'], 
        output_dir=config['output_dir']
    )

    print("\n--- Pipeline Complete ---")

