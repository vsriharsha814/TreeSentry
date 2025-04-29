import os
import numpy as np
import torch
from torch.utils.data import Dataset
import glob
from src.utils import get_logger

class DeforestDataset(Dataset):
    def __init__(self, processed_dir, transform=None):
        """
        Dataset for deforestation detection from satellite imagery.
        
        Args:
            processed_dir: Directory containing processed input tiles and labels
            transform: Optional transforms to apply to images
        """
        self.logger = get_logger(self.__class__.__name__)
        self.processed_dir = processed_dir
        self.transform = transform
        
        # Expecting directory structure:
        # processed_dir/
        #   inputs/  <- multi-band satellite images
        #   labels/  <- binary deforestation masks
        
        self.input_dir = os.path.join(processed_dir, 'inputs')
        self.labels_dir = os.path.join(processed_dir, 'labels')
        
        # Check if directories exist
        if not os.path.exists(self.input_dir) or not os.path.exists(self.labels_dir):
            self.logger.error(f"Input or labels directory not found in {processed_dir}")
            raise FileNotFoundError(f"Missing data directories in {processed_dir}")
        
        # Get all input tile filenames (.npy files)
        self.input_files = sorted(glob.glob(os.path.join(self.input_dir, "*.npy")))
        
        # Make sure we have matching label files
        valid_inputs = []
        self.label_files = []
        
        for input_file in self.input_files:
            basename = os.path.basename(input_file).replace(".npy", "")
            label_file = os.path.join(self.labels_dir, f"{basename}_label.npy")
            
            if os.path.exists(label_file):
                valid_inputs.append(input_file)
                self.label_files.append(label_file)
        
        self.input_files = valid_inputs
        self.logger.info(f"Found {len(self.input_files)} valid samples")
        
    def __len__(self):
        return len(self.input_files)
    
    def __getitem__(self, idx):
        """Load a single (input, label) pair for training"""
        # Load input and label arrays
        input_array = np.load(self.input_files[idx])
        label_array = np.load(self.label_files[idx])
        
        # Convert to PyTorch tensors
        input_tensor = torch.from_numpy(input_array).float()
        label_tensor = torch.from_numpy(label_array).float()
        
        # Make sure input is in correct format [C, H, W]
        if input_tensor.ndim == 3 and input_tensor.shape[0] < input_tensor.shape[1]:
            input_tensor = input_tensor.permute(2, 0, 1)
        
        # Make sure label has channel dimension [1, H, W]
        if label_tensor.ndim == 2:
            label_tensor = label_tensor.unsqueeze(0)
        
        # Apply any transforms if provided
        if self.transform:
            input_tensor, label_tensor = self.transform(input_tensor, label_tensor)
        
        return input_tensor, label_tensor


class TemporalDeforestDataset(Dataset):
    def __init__(self, processed_dir, time_steps=4, transform=None):
        """
        Dataset for temporal deforestation detection, loading sequences of satellite images.
        
        Args:
            processed_dir: Directory containing processed input tiles and labels
            time_steps: Number of time steps to include in each sequence
            transform: Optional transforms to apply to images
        """
        self.logger = get_logger(self.__class__.__name__)
        self.processed_dir = processed_dir
        self.time_steps = time_steps
        self.transform = transform
        
        # Directories for inputs and labels
        self.input_dir = os.path.join(processed_dir, 'inputs')
        self.labels_dir = os.path.join(processed_dir, 'labels')
        
        # Check if directories exist
        if not os.path.exists(self.input_dir) or not os.path.exists(self.labels_dir):
            self.logger.error(f"Input or labels directory not found in {processed_dir}")
            raise FileNotFoundError(f"Missing data directories in {processed_dir}")
        
        # Get all input files and group by location (removing year)
        all_files = glob.glob(os.path.join(self.input_dir, "*.npy"))
        location_groups = {}
        
        for file_path in all_files:
            filename = os.path.basename(file_path)
            # Extract location identifier (tiles are named like: tile_YEAR_Y_X_IDX.npy)
            parts = filename.split('_')
            if len(parts) >= 5:  # Make sure filename has enough parts
                location_key = f"{parts[0]}_{parts[2]}_{parts[3]}_{parts[4].split('.')[0]}"
                year = parts[1]
                
                if location_key not in location_groups:
                    location_groups[location_key] = {}
                
                location_groups[location_key][year] = file_path
        
        # Find locations with enough time steps
        self.valid_sequences = []
        
        for location, years_dict in location_groups.items():
            if len(years_dict) >= time_steps:
                # Sort by year
                sorted_years = sorted(years_dict.keys())
                
                # Get the latest year for label
                latest_year = sorted_years[-1]
                
                # Check if we have a matching label for the latest year
                latest_file = years_dict[latest_year]
                label_file = latest_file.replace('inputs', 'labels').replace('.npy', '_label.npy')
                
                if os.path.exists(label_file):
                    # Use the last time_steps years
                    years_to_use = sorted_years[-time_steps:]
                    input_files = [years_dict[year] for year in years_to_use]
                    
                    # Store sequence info
                    self.valid_sequences.append({
                        'inputs': input_files,
                        'label': label_file
                    })
        
        self.logger.info(f"Found {len(self.valid_sequences)} valid temporal sequences")
    
    def __len__(self):
        return len(self.valid_sequences)
    
    def __getitem__(self, idx):
        """Load a sequence of inputs and corresponding label"""
        sequence = self.valid_sequences[idx]
        
        # Load all input arrays in the sequence
        input_arrays = [np.load(file) for file in sequence['inputs']]
        
        # Stack arrays along a new first dimension (time)
        stacked_inputs = np.stack(input_arrays, axis=0)
        
        # Load label
        label_array = np.load(sequence['label'])
        
        # Convert to PyTorch tensors
        input_tensor = torch.from_numpy(stacked_inputs).float()  # [T, C, H, W]
        label_tensor = torch.from_numpy(label_array).float()
        
        # Make sure label has channel dimension [1, H, W]
        if label_tensor.ndim == 2:
            label_tensor = label_tensor.unsqueeze(0)
        
        # Apply any transforms if provided
        if self.transform:
            # Note: transform would need to handle temporal data
            input_tensor, label_tensor = self.transform(input_tensor, label_tensor)
        
        # Reshape to [C, T, H, W] format for 3D convolutions
        input_tensor = input_tensor.permute(1, 0, 2, 3)
        
        return input_tensor, label_tensor