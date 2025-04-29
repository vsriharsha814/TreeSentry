import numpy as np
import os
from rasterio.windows import Window

class DataPrep:
    def __init__(self, download_dir, tile_size, output_dir):
        self.download_dir = download_dir
        self.tile_size = tile_size
        self.output_dir = output_dir

    def stack_bands(self, year, layers):
        """
        Stack static and dynamic bands into a multi-channel array for model input.
        """
        # TODO: read each layer GeoTIFF, align shapes, stack
        pass

    def create_tiles(self, stacked_array):
        """
        Slice stacked_array into tiles of size tile_size, save to disk or return list.
        """
        # TODO: implement sliding-window cropping
        pass
