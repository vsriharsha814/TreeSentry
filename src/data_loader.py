import ee
import rasterio
import geopandas as gpd

class DataLoader:
    def __init__(self, boundary_shp, tiles_shp, download_dir):
        ee.Initialize()
        self.boundary = gpd.read_file(boundary_shp)
        self.tiles = gpd.read_file(tiles_shp)
        self.download_dir = download_dir

    def download(self, years, static_layers, dynamic_layers):
        """
        Download specified layers for each year and save as GeoTIFFs in download_dir.
        """
        # TODO: implement using Earth Engine Python API, mimicking download_data.py
        pass

    def load_geotiff(self, path):
        with rasterio.open(path) as src:
            return src.read(), src.transform
