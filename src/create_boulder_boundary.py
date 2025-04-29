import geopandas as gpd
from shapely.geometry import Polygon
import os

# Create data directory if it doesn't exist
os.makedirs('data', exist_ok=True)

# Approximate coordinates of Boulder, CO (in WGS84)
# These form a rough rectangle around the city
boulder_coords = [
    (-105.301, 39.964),  # Northwest corner
    (-105.301, 39.964),  # Northeast corner
    (-105.178, 39.964),  # Northeast corner
    (-105.178, 39.941),  # Southeast corner
    (-105.30, 39.941),   # Southwest corner
    (-105.301, 39.964)   # Back to start
]

# Create a polygon from the coordinates
polygon = Polygon(boulder_coords)

# Create a GeoDataFrame
gdf = gpd.GeoDataFrame(index=[0], crs='EPSG:4326', geometry=[polygon])

# Add attributes
gdf['name'] = 'Boulder'
gdf['area_km2'] = gdf.geometry.area * 111 * 111  # Approximate conversion to km²

# Save to shapefile
output_path = 'data/boulder_boundary.shp'
gdf.to_file(output_path)

print(f"Boulder boundary shapefile created at {output_path}")