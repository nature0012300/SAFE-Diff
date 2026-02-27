import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import os 
from tqdm import tqdm
from pathlib import Path
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from utils import calculate_metrics
from data_setup_final import (set_seed,LitsSliceDatasetCSV, create_dataloader, train_dir, val_dir, test_dir, root_dir, device)  
from torch.cuda.amp import GradScaler
from torch.amp import autocast
# --- Architectural Components ---

class SEBlock(nn.Module):
    """
    Squeeze-and-Excitation Block (a form of Channel Attention).
    This block learns to re-weight channel features, allowing the model to
    focus on the most important information.
    """
    def __init__(self, channels, reduction_ratio=16):
        super(SEBlock, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channels, channels // reduction_ratio, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channels // reduction_ratio, channels, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)

# class CBAMBlock(nn.Module):
#     """
#     Convolutional Block Attention Module (CBAM).
#     Combines channel attention and spatial attention to refine features
#     both across channels and spatial locations.
#     """
#     def __init__(self, channels, reduction_ratio=16):
#         super(CBAMBlock, self).__init__()
        
#         # Channel Attention Module
#         self.avg_pool = nn.AdaptiveAvgPool2d(1)
#         self.max_pool = nn.AdaptiveMaxPool2d(1)
        
#         # Shared MLP for channel attention
#         self.mlp = nn.Sequential(
#             nn.Linear(channels, channels // reduction_ratio, bias=False),
#             nn.ReLU(inplace=True),
#             nn.Linear(channels // reduction_ratio, channels, bias=False)
#         )
        
#         # Spatial Attention Module
#         self.conv_spatial = nn.Conv2d(2, 1, kernel_size=7, padding=3, bias=False)
        
#         self.sigmoid = nn.Sigmoid()

#     def forward(self, x):
#         # Channel Attention
#         b, c, _, _ = x.size()
        
#         # Average and max pooling
#         avg_pool = self.avg_pool(x).view(b, c)
#         max_pool = self.max_pool(x).view(b, c)
        
#         # Pass through shared MLP and sum
#         avg_out = self.mlp(avg_pool)
#         max_out = self.mlp(max_pool)
#         channel_att = self.sigmoid(avg_out + max_out).view(b, c, 1, 1)
        
#         # Apply channel attention
#         x = x * channel_att
        
#         # Spatial Attention
#         # Channel-wise average and max pooling
#         avg_spatial = torch.mean(x, dim=1, keepdim=True)  # [B, 1, H, W]
#         max_spatial, _ = torch.max(x, dim=1, keepdim=True)  # [B, 1, H, W]
        
#         # Concatenate along channel dimension
#         spatial_input = torch.cat([avg_spatial, max_spatial], dim=1)  # [B, 2, H, W]
        
#         # Apply convolution and sigmoid
#         spatial_att = self.sigmoid(self.conv_spatial(spatial_input))  # [B, 1, H, W]
        
#         # Apply spatial attention
#         x = x * spatial_att
        
#         return x

class ResidualBlock(nn.Module):
    """
    The core building block of the network. It allows the network to learn
    modifications (residuals) to the identity feature map, which stabilizes training
    for very deep networks.
    """
    def __init__(self, channels):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.se = SEBlock(channels) # Add attention here

    def forward(self, x):
        out = self.relu(self.conv1(x))
        out = self.conv2(out)
        out = self.se(out)
        return out + x

class ResidualPredictionNet(nn.Module):
    """
    The main model that takes an LR image and predicts the residual map.
    """
    def __init__(self, in_channels:int, out_channels:int, feature_channels:int, n_res_blocks:int, upscale_factor:int):
        super(ResidualPredictionNet, self).__init__()
        self.upscale_factor = upscale_factor

        # 1. Initial Convolution Layer
        # We will perform upsampling in the training loop/forward pass,
        # but the first conv layer acts on the upsampled image.
        self.head = nn.Conv2d(in_channels, feature_channels, kernel_size=3, padding=1)

        # 2. Main Body with Residual Blocks
        body = [ResidualBlock(feature_channels) for _ in range(n_res_blocks)]
        self.body = nn.Sequential(*body)

        # 3. Reconstruction Tail
        self.tail = nn.Conv2d(feature_channels, out_channels, kernel_size=3, padding=1)

    def forward(self, lr_image):       
        x_head = self.head(lr_image)
        x_body = self.body(x_head)
        
        predicted_residual = self.tail(x_body)
        
        return predicted_residual

#  ========== Training Setup ========== 
class EAGLELoss(nn.Module):
    def __init__(self, patch_size=8, epsilon=1e-8, cutoff=4.0):
        super(EAGLELoss, self).__init__()
        self.patch_size = patch_size
        self.epsilon = epsilon
        self.cutoff = cutoff  # Frequency cutoff for high-pass Gaussian
        self.force_float32 = True  # Force float32 for FFT computations
        # Scharr kernels for x and y gradients
        self.scharr_x = torch.tensor([[3., 0., -3.],
                                      [10., 0., -10.],
                                      [3., 0., -3.]], dtype=torch.float32).unsqueeze(0).unsqueeze(0)/16.0
        self.scharr_y = torch.tensor([[3., 10., 3.],
                                      [0., 0., 0.],
                                      [-3., -10., -3.]], dtype=torch.float32).unsqueeze(0).unsqueeze(0)/16.0
    
    def gradient_maps(self, img):
        # img shape: [B, 1, H, W]
        gx = F.conv2d(img, self.scharr_x.to(img.device), padding=1)
        gy = F.conv2d(img, self.scharr_y.to(img.device), padding=1)
        return gx, gy
    
    def patch_variance(self, gm):
        # Divide gm into non-overlapping patches and compute variance for each patch
        B, C, H, W = gm.shape
        patch_size = self.patch_size
        patches = gm.unfold(2, patch_size, patch_size).unfold(3, patch_size, patch_size)
        # patches: [B, C, nrows, ncols, patch_size, patch_size]
        var_map = patches.var(dim=-1, keepdim=False).var(dim=-2, keepdim=False)
        # var_map: [B, C, nrows, ncols]
        return var_map
    
    def high_pass_mask(self, shape, cutoff, device):
        """Create a Gaussian high-pass filter mask."""
        rows, cols = shape
        center_x, center_y = rows//2, cols//2
        xv, yv = torch.meshgrid(torch.arange(rows), torch.arange(cols), indexing='ij')
        freq = torch.sqrt((xv - center_x)**2 + (yv - center_y)**2).float().to(device)
        mask = 1 - torch.exp(-((freq - cutoff)**2)/2)
        return mask
    
    def forward(self, output, target):
        # Ensure images are [B, 1, H, W]
        if output.dim() == 3: output = output.unsqueeze(1)
        if target.dim() == 3: target = target.unsqueeze(1)
        # Gradient maps
        gx_out, gy_out = self.gradient_maps(output)
        gx_tar, gy_tar = self.gradient_maps(target)
        # Patch variances
        vx_out = self.patch_variance(gx_out)
        vy_out = self.patch_variance(gy_out)
        vx_tar = self.patch_variance(gx_tar)
        vy_tar = self.patch_variance(gy_tar)
        # Apply DFT and take magnitude
        if self.force_float32:
            vx_out_fft = torch.fft.fftshift(torch.fft.fft2(vx_out.float(), norm='ortho'))
            vx_tar_fft = torch.fft.fftshift(torch.fft.fft2(vx_tar.float(), norm='ortho'))
            vy_out_fft = torch.fft.fftshift(torch.fft.fft2(vy_out.float(), norm='ortho'))
            vy_tar_fft = torch.fft.fftshift(torch.fft.fft2(vy_tar.float(), norm='ortho'))
        else:
            vx_out_fft = torch.fft.fftshift(torch.fft.fft2(vx_out, norm='ortho'))
            vx_tar_fft = torch.fft.fftshift(torch.fft.fft2(vx_tar, norm='ortho'))
            vy_out_fft = torch.fft.fftshift(torch.fft.fft2(vy_out, norm='ortho'))
            vy_tar_fft = torch.fft.fftshift(torch.fft.fft2(vy_tar, norm='ortho'))
        mx_out = torch.abs(vx_out_fft)
        mx_tar = torch.abs(vx_tar_fft)
        my_out = torch.abs(vy_out_fft)
        my_tar = torch.abs(vy_tar_fft)
        # High-pass mask
        HP_mask = self.high_pass_mask(mx_out.shape[-2:], self.cutoff, output.device) # shape: [nrows, ncols]
        HP_mask = HP_mask.unsqueeze(0).unsqueeze(0)  # [1,1,nrows,ncols]
        # Apply mask and compute L1 loss
        loss_x = F.l1_loss(mx_out * HP_mask, mx_tar * HP_mask)
        loss_y = F.l1_loss(my_out * HP_mask, my_tar * HP_mask)
        loss = (loss_x + loss_y) / 2
        return loss

def de_normalize(tensor):
    """Convert tensor from [-1, 1] to [0, 1] for visualization."""
    return (tensor + 1) / 2

def visualize_batch(model, batch, epoch, device, output_dir):
    """Generates and saves a plot for a given batch of images."""
    model.eval() # Ensure model is in eval mode
    vis_dir = os.path.join(output_dir, "visualizations")
    os.makedirs(vis_dir, exist_ok=True)
    
    # Unpack the batch and move to device
    original,downsampled,_,_= batch
    original = original.float().to(device)
    downsampled = downsampled.float().to(device)

    with torch.no_grad():
        # Pre-process for model
        upsampled = F.interpolate(downsampled, scale_factor=model.upscale_factor, mode='bicubic', align_corners=False)
        
        # Get prediction
        predicted_residual_norm = model(upsampled)
    
    # Post-process for plotting (move to CPU)
    original, upsampled = original.cpu(), upsampled.cpu()
    predicted_residual = predicted_residual_norm.cpu()
    
    # De-normalize images to [0,1] and calculate residuals
    original = de_normalize(original)
    upsampled = de_normalize(upsampled)
    ground_truth_residual = original - upsampled
    coarse_hr = torch.clamp(upsampled + predicted_residual,-1,1)
    coarse_hr = de_normalize(coarse_hr)
    print(f"Coarse HR min/max: {coarse_hr.min().item():.4f}/{coarse_hr.max().item():.4f}")
    # --- Plotting ---
    img_idx = 0
    fig, axes = plt.subplots(1, 5, figsize=(25, 5))
    
    axes[0].imshow(upsampled[img_idx, 0].numpy(), cmap='gray')
    axes[0].set_title('Input (Upsampled LR)')
    axes[0].axis('off')

    axes[1].imshow(ground_truth_residual[img_idx, 0].numpy(), cmap='gray')
    axes[1].set_title('Ground Truth Residual')
    axes[1].axis('off')

    axes[2].imshow(predicted_residual[img_idx, 0].numpy(), cmap='gray')
    axes[2].set_title('Predicted Residual')
    axes[2].axis('off')

    axes[3].imshow(coarse_hr[img_idx, 0].numpy(), cmap='gray')
    axes[3].set_title('Coarse HR Image')
    axes[3].axis('off')

    
    axes[4].imshow(original[img_idx, 0].numpy(), cmap='gray')
    axes[4].set_title('Original Image')
    axes[4].axis('off')

    plt.suptitle(f'Fixed Image Visualization - Epoch {epoch+1}', fontsize=16)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig(os.path.join(vis_dir, f"epoch_{epoch+1:03d}_result.png"))
    plt.close(fig)
    model.train()


def train_one_epoch_s1(model, train_dataloader,loss_fn,eagle_loss_fn,optimizer,scaler, device,epoch, eagle_weight,num_epochs):
    model.train()

    train_loss = 0
    mse_loss_train =0
    eagle_loss_train =0
    total_psnr = 0.0
    total_ssim = 0.0
    total_samples = 0

    for batch_idx, (original,downsampled_image,_,_) in enumerate(tqdm(train_dataloader, desc="Training")):
        
        original = original.float().to(device)
        downsampled_image = downsampled_image.float().to(device)
        
        # upsample the image to the original size
        upsampled_image = F.interpolate(downsampled_image, 
                   scale_factor=model.upscale_factor, mode='bicubic', align_corners=False  )
        upsampled_image = upsampled_image.float().to(device)

        optimizer.zero_grad()
        with autocast('cuda'):
            predicted_residual = model(upsampled_image) #pass parameters
            ground_truth_residual = original - upsampled_image
            coarse_hr = torch.clamp(upsampled_image + predicted_residual,-1,1)

            epoch_progress = epoch / num_epochs
            mse_loss = loss_fn(predicted_residual, ground_truth_residual) 
            eagle_loss = eagle_loss_fn(predicted_residual, ground_truth_residual) 

            loss =  mse_loss + eagle_weight * eagle_loss
        mse_loss_train += mse_loss.item()
        eagle_loss_train += eagle_loss.item()
        train_loss += loss.item()  

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        for i in range(coarse_hr.shape[0]):
                try:
                    train_psnr, train_ssim, train_nmse = calculate_metrics(original[i], coarse_hr[i])
                    total_psnr += train_psnr
                    total_ssim += train_ssim
                    total_samples += 1
                except Exception as e:
                    print(f"Metric error: {e}")
        # Update metrics
    train_loss/=len(train_dataloader)   
    mse_loss_train /= len(train_dataloader) 
    eagle_loss_train /= len(train_dataloader)
    avg_psnr = total_psnr / total_samples if total_samples > 0 else 0
    avg_ssim = total_ssim / total_samples if total_samples > 0 else 0

    return train_loss ,mse_loss_train,eagle_loss_train,avg_psnr, avg_ssim, train_nmse

def validate_one_epoch_s1(model, val_dataloader, loss_fn,eagle_loss_fn,device,epoch,eagle_weight, num_epochs,output_dir):
    
    val_loss =0
    mse_loss_val=0
    eagle_loss_val=0
    total_psnr = 0.0
    total_ssim = 0.0
    total_samples = 0

    model.eval()        
    with torch.no_grad():
        for original,downsampled_image,_,_ in tqdm(val_dataloader, desc="Validation"):
            epoch_progress = epoch / num_epochs
            original = original.float().to(device)
            downsampled_image = downsampled_image.float().to(device)

            upsampled_image = F.interpolate(downsampled_image, 
                   scale_factor=model.upscale_factor, mode='bicubic', align_corners=False  )
            upsampled_image = upsampled_image.float().to(device)
            
            predicted_residual = model(upsampled_image) #pass parameters to model
            ground_truth_residual = original - upsampled_image
            
            mse_loss = loss_fn(predicted_residual, ground_truth_residual) 

            eagle_loss = eagle_loss_fn(predicted_residual, ground_truth_residual) 

            loss =  mse_loss + eagle_weight * eagle_loss
            val_loss += loss.item()  
            mse_loss_val += mse_loss.item()
            eagle_loss_val += eagle_loss.item()

            coarse_hr = torch.clamp(upsampled_image + predicted_residual,-1,1)
            for i in range(coarse_hr.shape[0]):
                    try:
                        val_psnr, val_ssim, val_nmse = calculate_metrics(original[i], coarse_hr[i])
                        total_psnr += val_psnr
                        total_ssim += val_ssim
                        total_samples += 1
                    except Exception as e:
                        print(f"Metric error: {e}")

    # Set the seed to make the dataloader's shuffle predictable
    # set_seed(42)
    # Get the *first batch* from a new iterator over the same dataloader
    fixed_vis_batch = next(iter(val_dataloader))
    
    # Call the dedicated visualization function
    visualize_batch(model, fixed_vis_batch, epoch, device, output_dir)
    
        
    val_loss /= len(val_dataloader)
    mse_loss_val /= len(val_dataloader) 
    eagle_loss_val /= len(val_dataloader)
    val_psnr = total_psnr / total_samples if total_samples > 0 else 0
    val_ssim = total_ssim / total_samples if total_samples > 0 else 0

    return val_loss ,mse_loss_val,eagle_loss_val ,val_psnr, val_ssim, val_nmse

def train_and_validate(model,hyperparams, train_dataloader, val_dataloader,loss_fn,eagle_loss_fn,optimizer,scaler,num_epochs,eagle_weight, device,output_dir):
    results = []
    os.makedirs(output_dir, exist_ok=True)
    save_interval = 7  # Save model every 10 epochs
    
    best_val_loss = float('inf')  # Initialize with infinity
    best_epoch = 0
    best_model_path = os.path.join(output_dir, "stage1_best_model.pth")

    print(f"----Starting Training------ ")
    print(f"Hyperparameters : {hyperparams}")
        
        # Training metrics storage
    train_losses, val_losses = [], []

    for epoch in range(num_epochs):
        print(f"\nEpoch {epoch+1}/{num_epochs}")

        # Training with importance map visualization
        train_loss ,mse_loss_train,eagle_loss_train,train_psnr,train_ssim,train_nmse = train_one_epoch_s1(model, train_dataloader,loss_fn,eagle_loss_fn, optimizer,scaler, device,epoch,eagle_weight, num_epochs)
        train_losses.append(train_loss)
        print(f"Training - Loss: {train_loss} || MSE Loss: {mse_loss_train} || EAGLE Loss: {eagle_loss_train}")
        print(f"Train PSNR: {train_psnr:.6f}, SSIM: {train_ssim:.6f}, NMSE: {train_nmse:.6f}")
        # Validation
        val_loss,mse_loss_val,eagle_loss_val ,val_psnr, val_ssim, val_nmse= validate_one_epoch_s1(model, val_dataloader, loss_fn,eagle_loss_fn,device,epoch, eagle_weight,num_epochs,output_dir)
        val_losses.append(val_loss)

        print(f"Validation - Loss: {val_loss} || MSE Loss: {mse_loss_val} || EAGLE Loss: {eagle_loss_val}")
        print(f"Val PSNR: {val_psnr:.6f}, SSIM: {val_ssim:.6f}, NMSE: {val_nmse:.6f}")
        print(f"Saved visualization for epoch {epoch+1}.")
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_epoch = epoch + 1
            torch.save(model.state_dict(), best_model_path)
            print(f"🌟 New best model saved! Epoch {best_epoch}, Val Loss: {best_val_loss:.6f}")

        if (epoch + 1) % save_interval == 0:
            save_path = os.path.join(output_dir,f"stage1_model_epoch_{epoch+1}.pth")
            torch.save(model.state_dict(),save_path)
            print(f"Checkpoint Saved at {save_path}")

    final_path = os.path.join(output_dir,"stage1_model_final.pth")   
    torch.save(model.state_dict(),final_path) 

    print(f"\n Best model: Epoch {best_epoch} with validation loss: {best_val_loss:.6f}")
    print(f"Best model saved at: {best_model_path}")

    print(f"\n -----Training Completed  -----")   
    print(f"Final model saved at {final_path}")
    # Store results
    results.append({
        "params": hyperparams,
        "train_losses": train_losses,
        "val_losses": val_losses,
        "final_model_path": final_path})
       
    return results

if __name__ == '__main__':
    # --- Configuration ---
    BATCH_SIZE = 8
    NUM_EPOCHS = 95
    LEARNING_RATE = 1e-4
    OUTPUT_DIR = "training_output_stage1_kits"

    # --- Hyperparameters for the Model ---
    hyperparams_s1 = {
        "in_channels": 1, 
        "out_channels": 1, 
        "feature_channels": 64, 
        "n_res_blocks": 20, 
        "upscale_factor": 4
    }
    
    train_dataloader = create_dataloader(csv_file="/home/m24ma2010/Kits/train_data.csv",
                                        max_items = None, batch_size=BATCH_SIZE , shuffle=True)
    val_dataloader = create_dataloader(csv_file="/home/m24ma2010/Kits/validation_data.csv",
                                        max_items = None, batch_size=BATCH_SIZE , shuffle=False)
    # test_dataloader = create_dataloader(csv_file="/home/m24ma2010/my_model/liver_ct_reports_test.csv",

    # --- Initialize Model and Optimizer ---
    model_s1 = ResidualPredictionNet(**hyperparams_s1).to(device)
    optimizer_s1 = optim.Adam(model_s1.parameters(), lr=LEARNING_RATE)
    scaler = GradScaler()
    loss_fn_s1 = nn.L1Loss()

    eagle_loss_fn = EAGLELoss(patch_size=4, cutoff=2.011769321).to(device)

    print("--- Model initialized ---")
    print(f"Model parameters: {sum(p.numel() for p in model_s1.parameters()):,}")

    # --- Start Training ---
    results = train_and_validate(
        model=model_s1,
        hyperparams=hyperparams_s1,
        train_dataloader=train_dataloader,
        val_dataloader=val_dataloader,
        loss_fn=loss_fn_s1,
        eagle_loss_fn=eagle_loss_fn,
        optimizer=optimizer_s1,
        scaler=scaler,
        eagle_weight=0,
        num_epochs=NUM_EPOCHS,
        device=device,
        output_dir=OUTPUT_DIR,
        
    )
    
    print("Training completed successfully!")

# CUDA_VISIBLE_DEVICES=0 nohup python stage_1.py > train_8.txt 2>&1 &

# out_stage_1_3tr.log -   eagle_weight = 0 , batch = 8 
# train_1_kits.log  -   eagle_weight = 0 , batch = 8 