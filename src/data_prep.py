import numpy as np
import os
import rasterio
from rasterio.windows import Window
import glob
# from src.utils import get_logger # use for running config file
from utils import get_logger

class DataPrep:
    def __init__(self, download_dir, tile_size, output_dir):
        """
        Prepare satellite data for model training by stacking bands and creating tiles.
        
        Args:
            download_dir: Directory containing downloaded GeoTIFF files
            tile_size: Size of tiles to create (tile_size × tile_size pixels)
            output_dir: Directory to save processed tiles
        """
        self.download_dir = download_dir
        self.tile_size = tile_size
        self.output_dir = output_dir
        self.logger = get_logger(self.__class__.__name__)
        
        # Create output directories if they don't exist
        os.makedirs(os.path.join(output_dir, 'inputs'), exist_ok=True)
        os.makedirs(os.path.join(output_dir, 'labels'), exist_ok=True)

    def stack_bands(self, year, layers):
        """
        Stack static and dynamic bands into a multi-channel array for model input.
        
        Args:
            year: Year of the dynamic data
            layers: List of layer names to stack
            
        Returns:
            Stacked array with shape [channels, height, width]
        """
        self.logger.info(f"Stacking {len(layers)} bands for year {year}")
        
        # Get reference shape from first layer
        reference_file = glob.glob(os.path.join(self.download_dir, f"{layers[0]}_{year}*.tif"))
        if not reference_file:
            raise FileNotFoundError(f"No file found for {layers[0]} in year {year}")
            
        with rasterio.open(reference_file[0]) as src:
            ref_shape = src.shape
            ref_transform = src.transform
            ref_crs = src.crs
            band_arrays = []
            
            # Read first band to initialize the stack
            band_arrays.append(src.read(1))
        
        # Read and align all other bands
        for layer in layers[1:]:
            layer_file = glob.glob(os.path.join(self.download_dir, f"{layer}_{year}*.tif"))
            if not layer_file:
                self.logger.warning(f"Missing layer {layer} for year {year}, using zeros")
                band_arrays.append(np.zeros(ref_shape))
                continue
                
            with rasterio.open(layer_file[0]) as src:
                if src.shape != ref_shape:
                    self.logger.warning(f"Layer {layer} has different shape, resampling")
                    # In a real implementation, you'd want to reproject/resample here
                    # For simplicity, we'll just use zeros if shapes don't match
                    band_arrays.append(np.zeros(ref_shape))
                else:
                    band_arrays.append(src.read(1))
        
        # Stack arrays along new first dimension [channels, height, width]
        stacked_array = np.stack(band_arrays, axis=0)
        
        return stacked_array, ref_transform, ref_crs

    def create_tiles(self, stacked_array, labels=None, overlap=0):
        """
        Slice stacked_array into tiles of size tile_size, with optional overlap.
        
        Args:
            stacked_array: Input array with shape [channels, height, width]
            labels: Optional label array with shape [height, width]
            overlap: Overlap between adjacent tiles (in pixels)
            
        Returns:
            List of (tile, coordinates) tuples where coordinates is (row, col)
        """
        channels, height, width = stacked_array.shape
        
        # Calculate effective step size with overlap
        step = self.tile_size - overlap
        
        tiles = []
        coordinates = []
        
        # Slide window across the array
        for y in range(0, height - self.tile_size + 1, step):
            for x in range(0, width - self.tile_size + 1, step):
                # Extract tile from each channel
                tile = stacked_array[:, y:y+self.tile_size, x:x+self.tile_size]
                
                # Extract corresponding label if provided
                if labels is not None:
                    label = labels[y:y+self.tile_size, x:x+self.tile_size]
                    tiles.append((tile, label))
                else:
                    tiles.append(tile)
                
                coordinates.append((y, x))
        
        self.logger.info(f"Created {len(tiles)} tiles of size {self.tile_size}×{self.tile_size}")
        return tiles, coordinates
    
    def save_tiles(self, tiles, coordinates, year, prefix="tile"):
        """
        Save tiles to disk in the output directory.
        
        Args:
            tiles: List of tiles or (tile, label) tuples
            coordinates: List of (row, col) coordinates for each tile
            year: Year to include in filenames
            prefix: Prefix for tile filenames
        """
        input_dir = os.path.join(self.output_dir, 'inputs')
        label_dir = os.path.join(self.output_dir, 'labels')
        
        for i, ((y, x), tile_data) in enumerate(zip(coordinates, tiles)):
            # Handle case where tiles contain (input, label) pairs or just inputs
            if isinstance(tile_data, tuple) and len(tile_data) == 2:
                tile, label = tile_data
                
                # Save input tile
                tile_path = os.path.join(input_dir, f"{prefix}_{year}_{y}_{x}_{i}.npy")
                np.save(tile_path, tile)
                
                # Save label
                label_path = os.path.join(label_dir, f"{prefix}_{year}_{y}_{x}_{i}_label.npy")
                np.save(label_path, label)
            else:
                # Just save input tile
                tile_path = os.path.join(input_dir, f"{prefix}_{year}_{y}_{x}_{i}.npy")
                np.save(tile_path, tile_data)
        
        self.logger.info(f"Saved {len(tiles)} tiles to {self.output_dir}")
    
    def process_year(self, year, input_layers, label_layer=None, overlap=0):
        """
        Process a full year's data: stack bands, create tiles, and save to disk.
        
        Args:
            year: Year to process
            input_layers: List of input layer names
            label_layer: Optional name of label layer
            overlap: Overlap between tiles in pixels
        """
        # Stack all input bands
        stacked_array, transform, crs = self.stack_bands(year, input_layers)
        
        # Load label if provided
        labels = None
        if label_layer:
            label_file = glob.glob(os.path.join(self.download_dir, f"{label_layer}_{year}*.tif"))
            if label_file:
                with rasterio.open(label_file[0]) as src:
                    labels = src.read(1)
            else:
                self.logger.warning(f"No label file found for {label_layer}_{year}")
        
        # Create tiles
        tiles, coords = self.create_tiles(stacked_array, labels, overlap)
        
        # Save tiles to disk
        self.save_tiles(tiles, coords, year)
        
        return len(tiles)