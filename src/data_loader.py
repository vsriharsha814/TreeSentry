import ee
import os
import rasterio
import geopandas as gpd
import time
from utils import get_logger

class DataLoader:
    def __init__(self, boundary_shp, tiles_shp=None, download_dir=None):
        """
        Download satellite imagery from Google Earth Engine.
        
        Args:
            boundary_shp: Path to shapefile defining the area of interest
            tiles_shp: Optional path to shapefile defining tiles for parallel processing
            download_dir: Directory to save downloaded data
        """
        try:
            ee.Initialize()
        except Exception as e:
            print("Error initializing Earth Engine. Make sure you're authenticated.")
            print("Run 'earthengine authenticate' on the command line.")
            raise e
            
        self.logger = get_logger(self.__class__.__name__)
        self.boundary = gpd.read_file(boundary_shp)
        
        if tiles_shp:
            self.tiles = gpd.read_file(tiles_shp)
        else:
            self.tiles = None
            
        self.download_dir = download_dir or os.path.join(os.getcwd(), 'downloads')
        os.makedirs(self.download_dir, exist_ok=True)
        
    def _get_ee_geometry(self, geom):
        """Convert a GeoPandas geometry to an Earth Engine geometry"""
        # For polygons/multipolygons
        if geom.geom_type == 'Polygon':
            coords = [list(zip(*geom.exterior.coords.xy))]
            return ee.Geometry.Polygon(coords)
        elif geom.geom_type == 'MultiPolygon':
            all_coords = []
            for poly in geom.geoms:
                coords = list(zip(*poly.exterior.coords.xy))
                all_coords.append(coords)
            return ee.Geometry.MultiPolygon(all_coords)
        else:
            raise ValueError(f"Unsupported geometry type: {geom.geom_type}")
            
    def _get_satellite_collection(self, year, satellite='sentinel2'):
        """Get Earth Engine image collection for specified satellite and year"""
        start_date = f'{year}-01-01'
        end_date = f'{year}-12-31'
        
        if satellite.lower() == 'sentinel2':
            collection = (ee.ImageCollection('COPERNICUS/S2_SR')
                         .filterDate(start_date, end_date)
                         .filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', 20)))
        elif satellite.lower() == 'landsat8':
            collection = (ee.ImageCollection('LANDSAT/LC08/C02/T1_L2')
                         .filterDate(start_date, end_date)
                         .filter(ee.Filter.lt('CLOUD_COVER', 20)))
        elif satellite.lower() == 'landsat9':
            collection = (ee.ImageCollection('LANDSAT/LC09/C02/T1_L2')
                         .filterDate(start_date, end_date)
                         .filter(ee.Filter.lt('CLOUD_COVER', 20)))
        else:
            raise ValueError(f"Unsupported satellite: {satellite}")
            
        return collection
        
    def _get_indices_image(self, collection, bands_dict, indices=['NDVI']):
        """Calculate vegetation indices and create a composite image"""
        # First create a median composite
        composite = collection.median()
        
        # Calculate requested indices
        for idx in indices:
            if idx == 'NDVI':
                # Add NDVI band
                nir_band = bands_dict.get('nir', 'B8')
                red_band = bands_dict.get('red', 'B4')
                ndvi = composite.normalizedDifference([nir_band, red_band]).rename('NDVI')
                composite = composite.addBands(ndvi)
            elif idx == 'EVI':
                # Enhanced Vegetation Index
                nir_band = bands_dict.get('nir', 'B8')
                red_band = bands_dict.get('red', 'B4')
                blue_band = bands_dict.get('blue', 'B2')
                
                evi = composite.expression(
                    '2.5 * ((NIR - RED) / (NIR + 6 * RED - 7.5 * BLUE + 1))', {
                        'NIR': composite.select(nir_band),
                        'RED': composite.select(red_band),
                        'BLUE': composite.select(blue_band)
                    }).rename('EVI')
                composite = composite.addBands(evi)
            elif idx == 'NDWI':
                # Normalized Difference Water Index
                nir_band = bands_dict.get('nir', 'B8')
                swir_band = bands_dict.get('swir', 'B11')
                ndwi = composite.normalizedDifference([nir_band, swir_band]).rename('NDWI')
                composite = composite.addBands(ndwi)
                
        return composite
    
    def download(self, years, satellite='sentinel2', indices=['NDVI'], static_layers=None):
        """
        Download satellite imagery and derived indices for specified years.
        
        Args:
            years: List of years to download data for
            satellite: Satellite source ('sentinel2', 'landsat8', 'landsat9')
            indices: List of indices to calculate ('NDVI', 'EVI', 'NDWI')
            static_layers: Optional dictionary of static layers to download once
        """
        # Set bands based on satellite
        if satellite.lower() == 'sentinel2':
            bands_dict = {
                'blue': 'B2',
                'green': 'B3',
                'red': 'B4',
                'nir': 'B8',
                'swir1': 'B11',
                'swir2': 'B12'
            }
            bands_to_download = ['B2', 'B3', 'B4', 'B8', 'B11', 'B12']
            scale = 10  # meters per pixel
        elif satellite.lower() in ['landsat8', 'landsat9']:
            bands_dict = {
                'blue': 'SR_B2',
                'green': 'SR_B3',
                'red': 'SR_B4',
                'nir': 'SR_B5',
                'swir1': 'SR_B6',
                'swir2': 'SR_B7'
            }
            bands_to_download = ['SR_B2', 'SR_B3', 'SR_B4', 'SR_B5', 'SR_B6', 'SR_B7']
            scale = 30  # meters per pixel
        
        # Process for each year
        for year in years:
            self.logger.info(f"Processing year {year}")
            
            # Get image collection for the year
            collection = self._get_satellite_collection(year, satellite)
            
            # Create a composite image with indices
            composite = self._get_indices_image(collection, bands_dict, indices)
            
            # Process each tile or the whole boundary
            geometries = []
            if self.tiles is not None:
                for idx, row in self.tiles.iterrows():
                    geometries.append((f"tile_{idx}", self._get_ee_geometry(row.geometry)))
            else:
                # Use the full boundary
                geometries = [("full_area", self._get_ee_geometry(self.boundary.iloc[0].geometry))]
            
            # Download for each geometry
            for name, geometry in geometries:
                # Download original bands
                for band in bands_to_download:
                    filename = f"{band}_{year}_{name}.tif"
                    output_path = os.path.join(self.download_dir, filename)
                    
                    if os.path.exists(output_path):
                        self.logger.info(f"File {filename} already exists, skipping")
                        continue
                    
                    self.logger.info(f"Downloading {filename}")
                    
                    try:
                        # Export the band
                        task = ee.batch.Export.image.toDrive(
                            image=composite.select(band),
                            description=f"{band}_{year}_{name}",
                            folder="EarthEngineExports",
                            fileNamePrefix=f"{band}_{year}_{name}",
                            region=geometry,
                            scale=scale,
                            maxPixels=1e9
                        )
                        task.start()
                        
                        # Poll for completion
                        while task.status()['state'] in ['READY', 'RUNNING']:
                            self.logger.info(f"Task status: {task.status()['state']}")
                            time.sleep(30)
                        
                        if task.status()['state'] == 'COMPLETED':
                            self.logger.info(f"Download complete: {filename}")
                        else:
                            self.logger.error(f"Download failed: {task.status()}")
                            
                    except Exception as e:
                        self.logger.error(f"Error downloading {filename}: {e}")
                
                # Download indices
                for idx in indices:
                    filename = f"{idx}_{year}_{name}.tif"
                    output_path = os.path.join(self.download_dir, filename)
                    
                    if os.path.exists(output_path):
                        self.logger.info(f"File {filename} already exists, skipping")
                        continue
                    
                    self.logger.info(f"Downloading {filename}")
                    
                    try:
                        # Export the index
                        task = ee.batch.Export.image.toDrive(
                            image=composite.select(idx),
                            description=f"{idx}_{year}_{name}",
                            folder="EarthEngineExports",
                            fileNamePrefix=f"{idx}_{year}_{name}",
                            region=geometry,
                            scale=scale,
                            maxPixels=1e9
                        )
                        task.start()
                        
                        # Poll for completion
                        while task.status()['state'] in ['READY', 'RUNNING']:
                            self.logger.info(f"Task status: {task.status()['state']}")
                            time.sleep(30)
                        
                        if task.status()['state'] == 'COMPLETED':
                            self.logger.info(f"Download complete: {filename}")
                        else:
                            self.logger.error(f"Download failed: {task.status()}")
                            
                    except Exception as e:
                        self.logger.error(f"Error downloading {filename}: {e}")
            
        # Download static layers if specified
        if static_layers:
            self.download_static_layers(static_layers, geometries)
                    
    def download_static_layers(self, static_layers, geometries):
        """Download static (non-temporal) layers"""
        for layer_name, layer_id in static_layers.items():
            # Get the Earth Engine dataset
            dataset = ee.Image(layer_id)
            
            for name, geometry in geometries:
                filename = f"{layer_name}_{name}.tif"
                output_path = os.path.join(self.download_dir, filename)
                
                if os.path.exists(output_path):
                    self.logger.info(f"File {filename} already exists, skipping")
                    continue
                
                self.logger.info(f"Downloading static layer {filename}")
                
                try:
                    # Export the layer
                    task = ee.batch.Export.image.toDrive(
                        image=dataset,
                        description=f"{layer_name}_{name}",
                        folder="EarthEngineExports",
                        fileNamePrefix=f"{layer_name}_{name}",
                        region=geometry,
                        scale=30,  # Default to 30m resolution for static layers
                        maxPixels=1e9
                    )
                    task.start()
                    
                    # Poll for completion
                    while task.status()['state'] in ['READY', 'RUNNING']:
                        self.logger.info(f"Task status: {task.status()['state']}")
                        time.sleep(30)
                    
                    if task.status()['state'] == 'COMPLETED':
                        self.logger.info(f"Download complete: {filename}")
                    else:
                        self.logger.error(f"Download failed: {task.status()}")
                        
                except Exception as e:
                    self.logger.error(f"Error downloading {filename}: {e}")
    
    def load_geotiff(self, path):
        """
        Load a GeoTIFF file.
        
        Args:
            path: Path to GeoTIFF file
            
        Returns:
            data: Array of raster data
            transform: Affine transform
        """
        with rasterio.open(path) as src:
            return src.read(), src.transform