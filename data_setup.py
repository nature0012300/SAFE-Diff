import os
import random
import numpy as np
import torch
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, Subset
import torchvision.transforms as transforms
from torchvision.transforms import InterpolationMode
from torchvision.transforms import functional as TF
from PIL import Image
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt

torch.cuda.empty_cache()

train_dir = '/DATA2/lits/train_lits'
val_dir = '/DATA2/lits/val_lits'
test_dir = '/DATA2/lits/test_lits'
root_dir = 'Root_Dir'

# Set Device
device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
# device = "cpu"
print(f"Device is: {device}")

# Configuration
CONFIG = {
    'hr_size': (512, 512),
    'lr_size': (128, 128),
    'lr_size_mid': (256, 256),
    'seed': 42
}

def set_seed(seed: int = 42):
    """Set seed for reproducibility across all random number generators."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    # For deterministic behavior in PyTorch operations
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def save_sample_images(images, prefix, image_type, save_dir='sample_image_display'):
    """Save sample images for visualization and debugging."""
    os.makedirs(save_dir, exist_ok=True)
    
    for i, image in enumerate(images):
        if image is None:
            continue
            
        image_path = os.path.join(save_dir, f"{prefix}_{image_type}_image{i+1}.png")
        
        # Handle tensor input
        if torch.is_tensor(image):
            image_to_save = image.squeeze().cpu().numpy()
        else:
            image_to_save = image.squeeze() if hasattr(image, 'squeeze') else image
            
        # Ensure proper format for saving
        if image_to_save.ndim == 2:
            plt.imsave(image_path, image_to_save, cmap='gray')
        else:
            print(f"Warning: Unexpected image dimensions for saving: {image_to_save.shape}")

class RandomResize(object):
    """
    A custom transform that resizes an image using a randomly selected interpolation method.
    """
    def __init__(self, size):
        self.size = size
        # Define the list of interpolation methods to choose from
        self.methods = [InterpolationMode.BILINEAR, InterpolationMode.BICUBIC, InterpolationMode.BOX]

    def __call__(self, img):
        # Randomly select one interpolation method for each image
        method = random.choice(self.methods)
        # Use the functional API for resizing, which is more flexible
        return TF.resize(img, self.size, interpolation=method)

class LitsSliceDatasetCSV(Dataset):
    """
    APPROACH 1 with CSV: Slice-based Dataset that loads image paths from CSV file
    
    Loads image paths 
    Each __getitem__ call returns (HR_image, LR_clean) for a single slice.
    """
    
    def __init__(self, csv_file, hr_size=(512, 512), lr_size=(128,128),lr_size_mid=(256,256),
                 image_path_column='image_path',root_dir=None):
        """
        Initialize the dataset.
        
        Args:
            csv_file: Path to CSV file containing image paths
            hr_size: Target size for high-resolution images
            lr_size: Target size for low-resolution images
            image_path_column: Name of column containing image paths in CSV
            root_dir: Optional root directory to prepend to relative paths in CSV
        """
        self.csv_file = csv_file
        self.hr_size = hr_size
        self.lr_size = lr_size
        self.lr_size_mid = lr_size_mid
        self.image_path_column = image_path_column
        self.root_dir = Path(root_dir) if root_dir else None
        
        # Load CSV data
        self.data_df = self._load_csv_data()
        
        # Define image transforms
        self.hr_transform = transforms.Compose([
            transforms.Resize(self.hr_size, interpolation=InterpolationMode.BICUBIC),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5], std=[0.5])
        ])
        
        self.lr_transform = transforms.Compose([
            # RandomResize(self.lr_size_mid),
            # transforms.GaussianBlur(kernel_size=(5, 5), sigma=(1.0, 2.0)), 
            # RandomResize(self.lr_size),
            transforms.Resize(self.lr_size, interpolation=InterpolationMode.BICUBIC),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5], std=[0.5])
        ])
        
        print(f"Dataset initialized with {len(self.data_df)} samples from CSV")

    def _load_csv_data(self):
        """Load and validate CSV data."""
        try:
            print(f"Loading data from CSV file: {self.csv_file}")
            df = pd.read_csv(self.csv_file)
            print(f"CSV loaded successfully with {len(df)} rows and columns: {list(df.columns)}")
            
            # Validate required columns
            required_columns = [self.image_path_column]
            missing_columns = [col for col in required_columns if col not in df.columns]
            
            if missing_columns:
                raise ValueError(f"Missing required columns in CSV: {missing_columns}")
            
            # Remove rows with missing values in required columns
            initial_len = len(df)
            df = df.dropna(subset=required_columns)
            if len(df) < initial_len:
                print(f"Removed {initial_len - len(df)} rows with missing values")
            
            # Validate image paths exist
            valid_rows = []
            for idx, row in df.iterrows():
                image_path = self._get_full_image_path(row[self.image_path_column])
                if os.path.exists(image_path):
                    valid_rows.append(idx)
                else:
                    print(f"Warning: Image not found: {image_path}")
            
            if len(valid_rows) < len(df):
                print(f"Found {len(df) - len(valid_rows)} missing image files")
                df = df.loc[valid_rows].reset_index(drop=True)
            
            print(f"Final dataset contains {len(df)} valid samples")
            return df
            
        except Exception as e:
            print(f"Error loading CSV file: {e}")
            raise

    def _get_full_image_path(self, image_path):
        """Get full path to image, handling relative paths."""
        image_path = Path(image_path)
        
        # If path is relative and root_dir is provided, prepend root_dir
        if not image_path.is_absolute() and self.root_dir is not None:
            image_path = self.root_dir / image_path
            
        return str(image_path)

    def __len__(self):
        return len(self.data_df)

    def __getitem__(self, idx):
        """
        Get a single slice with HR and clean LR
        
        Returns:
            tuple: (hr_tensor, lr_clean_tensor)
                   - Image tensors have shape [1, H, W] where H, W depend on hr_size/lr_size
        """
        if idx >= len(self.data_df):
            raise IndexError(f"Index {idx} out of range for dataset of size {len(self.data_df)}")
        
        # Get row data
        row = self.data_df.iloc[idx]
        image_path = self._get_full_image_path(row[self.image_path_column])
        
        try:
            # Load and properly normalize image
            with Image.open(image_path) as img:
                np_image = np.array(img).astype(np.float32)
                
                if img.mode == 'I;16':  # If 16-bit CT image
                    np_image = self.normalize_ct(np_image, min_bound=-1000, max_bound=1000)
                elif img.mode != 'L':
                    img = img.convert('L')
                    np_image = np.array(img).astype(np.float32) / 255.0
                else:
                    np_image = np_image.astype(np.float32) / 255.0
                
                # Convert back to PIL for transforms
                pil_image = Image.fromarray((np_image * 255).astype(np.uint8))
                
            # Generate HR version
            hr_tensor = self.hr_transform(pil_image)  # Shape: [1, hr_size[0], hr_size[1]]
            
            # Generate clean LR version
            lr_clean_tensor = self.lr_transform(pil_image)  # Shape: [1, lr_size[0], lr_size[1]]
            
            return hr_tensor, lr_clean_tensor
            
        except Exception as e:
            print(f"Error loading image {image_path}: {e}")
            return None, None
    
    def normalize_ct(self, image, min_bound=-1000, max_bound=1000):
        """Normalize CT image intensities from HU to [0,1] range"""
        image = np.clip(image, min_bound, max_bound)
        return (image - min_bound) / (max_bound - min_bound)

    def get_sample_info(self, idx):
        """Get additional information about a sample (useful for debugging)."""
        row = self.data_df.iloc[idx]
        return {
            'image_path': self._get_full_image_path(row[self.image_path_column]),
            'csv_index': idx
        }

def custom_collate_fn(batch):
    """
    Custom collate function that handles None values from failed image loads.
    
    Args:
        batch: List of tuples from __getitem__
        
    Returns:
        Batched tensors (None, None, None) if no valid samples
    """
    # Filter out None values
    valid_batch = [item for item in batch if item[0] is not None]
    
    if len(valid_batch) == 0:
        print("Warning: No valid samples in batch")
        return None, None
    
    hr_batch = [item[0] for item in valid_batch]
    lr_clean_batch = [item[1] for item in valid_batch]
    
    # Use default collate on image tensors
    hr_tensor = torch.stack(hr_batch)
    lr_clean_tensor = torch.stack(lr_clean_batch)

    return hr_tensor, lr_clean_tensor

def create_dataloader(csv_file, batch_size=4, max_items=None, 
                         seed=42, shuffle=True, num_workers=4, pin_memory=True,persistent_workers=True, 
                         hr_size=(512, 512), lr_size=(128,128),lr_size_mid=(256,256),
                         image_path_column='image_path', drop_last=True,
                         root_dir=None):
    """
    Create a DataLoader for the LITS slice dataset using CSV file.
    
    Args:
        csv_file: Path to CSV file containing image paths
        batch_size: Batch size for training
        max_items: Maximum number of items to use (None for all)
        seed: Random seed for reproducibility
        shuffle: Whether to shuffle the dataset
        num_workers: Number of worker processes for data loading
        pin_memory: Whether to pin memory for faster GPU transfer
        hr_size: Size for high-resolution images
        lr_size: Size for low-resolution images
        image_path_column: Name of column containing image paths in CSV
        root_dir: Optional root directory to prepend to relative paths in CSV
        
    Returns:
        DataLoader instance
    """
    set_seed(seed)
    
    # Create dataset
    dataset = LitsSliceDatasetCSV(
        csv_file=csv_file,
        hr_size=hr_size,
        lr_size=lr_size,
        lr_size_mid=lr_size_mid,
        image_path_column=image_path_column,
        root_dir=root_dir
    )
    
    # Create subset if max_items specified
    if max_items is not None and max_items < len(dataset):
        subset_indices = list(range(min(max_items, len(dataset))))
        dataset = Subset(dataset, subset_indices)
        print(f"Using subset of {len(dataset)} items")
    
    # Create DataLoader
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory and torch.cuda.is_available(),
        collate_fn=custom_collate_fn,
        drop_last=drop_last,  # Drop incomplete batches for consistent training
        persistent_workers=num_workers > 0  # Keep workers alive between epochs
    )
    
    return dataloader
