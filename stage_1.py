import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt

import os 
from tqdm import tqdm
import torch.optim as optim
from utils import calculate_metrics
from data_setup import (set_seed,LitsSliceDatasetCSV, create_dataloader, device)  
from torch.cuda.amp import GradScaler
from torch.amp import autocast

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
        self.se = SEBlock(channels) 

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

#  ========== Training Setup =======

def de_normalize(tensor):
    """Convert tensor from [-1, 1] to [0, 1] for visualization."""
    return (tensor + 1) / 2

def visualize_batch(model, batch, epoch, device, output_dir):
    """Generates and saves a plot for a given batch of images."""
    model.eval() # Ensure model is in eval mode
    vis_dir = os.path.join(output_dir, "visualizations")
    os.makedirs(vis_dir, exist_ok=True)
    
    # Unpack the batch and move to device
    original,downsampled = batch
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


def train_one_epoch_s1(model, train_dataloader,loss_fn,optimizer,scaler, device,epoch, num_epochs):
    model.train()

    train_loss = 0
    mae_loss_train =0
    total_psnr = 0.0
    total_ssim = 0.0
    total_samples = 0

    for batch_idx, (original,downsampled_image) in enumerate(tqdm(train_dataloader, desc="Training")):
        
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

            mae_loss = loss_fn(predicted_residual, ground_truth_residual) 

            loss =  mae_loss 
        mae_loss_train += mae_loss.item()
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
    mae_loss_train /= len(train_dataloader) 

    avg_psnr = total_psnr / total_samples if total_samples > 0 else 0
    avg_ssim = total_ssim / total_samples if total_samples > 0 else 0

    return train_loss ,mae_loss_train,avg_psnr, avg_ssim, train_nmse

def validate_one_epoch_s1(model, val_dataloader, loss_fn,device,epoch, num_epochs,output_dir):
    
    val_loss =0
    mae_loss_val=0
    total_psnr = 0.0
    total_ssim = 0.0
    total_samples = 0

    model.eval()        
    with torch.no_grad():
        for original,downsampled_image in tqdm(val_dataloader, desc="Validation"):
            original = original.float().to(device)
            downsampled_image = downsampled_image.float().to(device)

            upsampled_image = F.interpolate(downsampled_image, 
                   scale_factor=model.upscale_factor, mode='bicubic', align_corners=False  )
            upsampled_image = upsampled_image.float().to(device)
            
            predicted_residual = model(upsampled_image) #pass parameters to model
            ground_truth_residual = original - upsampled_image
            
            mae_loss = loss_fn(predicted_residual, ground_truth_residual) 

            loss =  mae_loss 
            val_loss += loss.item()  
            mae_loss_val += mae_loss.item()

            coarse_hr = torch.clamp(upsampled_image + predicted_residual,-1,1)
            for i in range(coarse_hr.shape[0]):
                    try:
                        val_psnr, val_ssim, val_nmse = calculate_metrics(original[i], coarse_hr[i])
                        total_psnr += val_psnr
                        total_ssim += val_ssim
                        total_samples += 1
                    except Exception as e:
                        print(f"Metric error: {e}")

    # Get the *first batch* from a new iterator over the same dataloader
    fixed_vis_batch = next(iter(val_dataloader))
    
    # Call the dedicated visualization function
    visualize_batch(model, fixed_vis_batch, epoch, device, output_dir)
    
        
    val_loss /= len(val_dataloader)
    mae_loss_val /= len(val_dataloader) 
    val_psnr = total_psnr / total_samples if total_samples > 0 else 0
    val_ssim = total_ssim / total_samples if total_samples > 0 else 0

    return val_loss ,mae_loss_val ,val_psnr, val_ssim, val_nmse

def train_and_validate(model,hyperparams, train_dataloader, val_dataloader,loss_fn,optimizer,scaler,num_epochs, device,output_dir):
    results = []
    os.makedirs(output_dir, exist_ok=True)
    save_interval = 7  # Save model every 7 epochs
    
    best_val_loss = float('inf')  # Initialize with infinity
    best_epoch = 0
    best_model_path = os.path.join(output_dir, "stage1_best_model.pth")

    print(f"----Starting Training------ ")
    print(f"Hyperparameters : {hyperparams}")
        
        # Training metrics storage
    train_losses, val_losses = [], []

    for epoch in range(num_epochs):
        print(f"\nEpoch {epoch+1}/{num_epochs}")

        # Training
        train_loss ,mae_loss_train,train_psnr,train_ssim,train_nmse = train_one_epoch_s1(model, train_dataloader,loss_fn,optimizer,scaler, device,epoch,num_epochs)
        train_losses.append(train_loss)
        print(f"Training - Loss: {train_loss} || MAE Loss: {mae_loss_train} ")
        print(f"Train PSNR: {train_psnr:.6f}, SSIM: {train_ssim:.6f}, NMSE: {train_nmse:.6f}")
        # Validation
        val_loss,mae_loss_val,val_psnr, val_ssim, val_nmse= validate_one_epoch_s1(model, val_dataloader, loss_fn,device,epoch,num_epochs,output_dir)
        val_losses.append(val_loss)

        print(f"Validation - Loss: {val_loss} || MAE Loss: {mae_loss_val} ")
        print(f"Val PSNR: {val_psnr:.6f}, SSIM: {val_ssim:.6f}, NMSE: {val_nmse:.6f}")
        print(f"Saved visualization for epoch {epoch+1}.")
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_epoch = epoch + 1
            torch.save(model.state_dict(), best_model_path)
            print(f" New best model saved! Epoch {best_epoch}, Val Loss: {best_val_loss:.6f}")

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
    OUTPUT_DIR = "path/to/output_dir"

    # --- Hyperparameters for the Model ---
    hyperparams_s1 = {
        "in_channels": 1, 
        "out_channels": 1, 
        "feature_channels": 64, 
        "n_res_blocks": 20, 
        "upscale_factor": 4
    }
    
    train_dataloader = create_dataloader(csv_file="path/to/your/train_data.csv",
                                        max_items = None, batch_size=BATCH_SIZE , shuffle=True)
    val_dataloader = create_dataloader(csv_file="path/to/your/val_data.csv",
                                        max_items = None, batch_size=BATCH_SIZE , shuffle=False)

    # --- Initialize Model and Optimizer ---
    model_s1 = ResidualPredictionNet(**hyperparams_s1).to(device)
    optimizer_s1 = optim.Adam(model_s1.parameters(), lr=LEARNING_RATE)
    scaler = GradScaler()
    loss_fn_s1 = nn.L1Loss()


    print("--- Model initialized ---")
    print(f"Model parameters: {sum(p.numel() for p in model_s1.parameters()):,}")

    # --- Start Training ---
    results = train_and_validate(
        model=model_s1,
        hyperparams=hyperparams_s1,
        train_dataloader=train_dataloader,
        val_dataloader=val_dataloader,
        loss_fn=loss_fn_s1,
        optimizer=optimizer_s1,
        scaler=scaler,
        num_epochs=NUM_EPOCHS,
        device=device,
        output_dir=OUTPUT_DIR,
        
    )
    
    print("Training completed successfully!")
