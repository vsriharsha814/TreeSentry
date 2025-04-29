import os
import numpy as np
import torch
from torch.utils.data import Dataset
import rasterio
from src.utils import get_logger

class DeforestDataset(Dataset):
    def __init__(self, tiles_dir, transform=None):
        """
        Dataset for deforestation detection from satellite imagery.
        
        Args:
            tiles_dir: Directory containing input tiles and corresponding labels
            transform: Optional transforms to apply to images
        """
        self.logger = get_logger(self.__class__.__name__)
        self.tiles_dir = tiles_dir
        self.transform = transform
        
        # Expecting directory structure:
        # tiles_dir/
        #   inputs/  <- multi-band satellite images
        #   labels/  <- binary deforestation masks
        
        self.input_dir = os.path.join(tiles_dir, 'inputs')
        self.labels_dir = os.path.join(tiles_dir, 'labels')
        
        if not os.path.exists(self.input_dir) or not os.path.exists(self.labels_dir):
            self.logger.error(f"Input or labels directory not found in {tiles_dir}")
            raise FileNotFoundError(f"Missing data directories in {tiles_dir}")
        
        # Get all input tile filenames
        self.tile_files = [f for f in os.listdir(self.input_dir) 
                          if f.endswith('.tif') or f.endswith('.npy')]
        
        self.logger.info(f"Found {len(self.tile_files)} tile samples")
        
    def __len__(self):
        return len(self.tile_files)
    
    def __getitem__(self, idx):
        """Load a single (input, label) pair for training"""
        tile_name = self.tile_files[idx]
        base_name = os.path.splitext(tile_name)[0]
        
        # Load input tile - handle both NumPy arrays and GeoTIFFs
        input_path = os.path.join(self.input_dir, tile_name)
        if tile_name.endswith('.npy'):
            # Load NumPy array directly
            input_array = np.load(input_path)
            # Convert to float32 and normalize if needed
            input_tensor = torch.from_numpy(input_array).float()
            
            # If array is [H, W, C], transpose to [C, H, W] for PyTorch
            if input_tensor.ndim == 3 and input_tensor.shape[2] < input_tensor.shape[0]:
                input_tensor = input_tensor.permute(2, 0, 1)
                
        else:  # GeoTIFF
            with rasterio.open(input_path) as src:
                input_array = src.read()  # Already in [C, H, W] format
                input_tensor = torch.from_numpy(input_array).float()
        
        # Normalize input to [0, 1] range if not already
        if input_tensor.max() > 1.0:
            input_tensor = input_tensor / input_tensor.max()
        
        # Load corresponding label mask
        label_path = os.path.join(self.labels_dir, f"{base_name}_label.tif")
        if not os.path.exists(label_path):
            # Try alternate extensions if tif not found
            alt_paths = [
                os.path.join(self.labels_dir, f"{base_name}_label.npy"),
                os.path.join(self.labels_dir, f"{base_name}.tif"),
                os.path.join(self.labels_dir, f"{base_name}.npy")
            ]
            
            for path in alt_paths:
                if os.path.exists(path):
                    label_path = path
                    break
            else:
                raise FileNotFoundError(f"No label found for {tile_name}")
        
        # Load label based on file type
        if label_path.endswith('.npy'):
            label_array = np.load(label_path)
            label_tensor = torch.from_numpy(label_array).float()
        else:  # GeoTIFF
            with rasterio.open(label_path) as src:
                label_array = src.read(1)  # Assume single band for mask
                label_tensor = torch.from_numpy(label_array).float()
        
        # Ensure label is binary and properly shaped [1, H, W]
        if label_tensor.ndim == 2:
            label_tensor = label_tensor.unsqueeze(0)
        
        # Apply any transforms if provided
        if self.transform:
            input_tensor, label_tensor = self.transform(input_tensor, label_tensor)
            
        return input_tensor, label_tensor