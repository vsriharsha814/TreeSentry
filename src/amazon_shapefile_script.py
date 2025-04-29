"""
Amazon Rainforest Shapefile Generator

This script creates a simplified Amazon rainforest boundary shapefile for use in
the deforestation detection project. It provides two methods:
1. Download official boundaries from available sources
2. Create a simplified boundary based on coordinates

Requirements:
- geopandas
- requests
- shapely
- matplotlib

Install dependencies with:
pip install geopandas requests shapely matplotlib
"""

import os
import sys
import requests
import zipfile
import io
import geopandas as gpd
from shapely.geometry import Polygon
import matplotlib.pyplot as plt


def download_brazilian_biomes_shapefile():
    """
    Download the official Brazilian biomes shapefile which includes the Amazon biome
    from Global Forest Watch.
    """
    # Create data directory if it doesn't exist
    os.makedirs('data', exist_ok=True)
    
    # Check if file already exists
    if os.path.exists('data/brazil_biomes.zip'):
        print("Brazil biomes shapefile already downloaded.")
        return 'data/brazil_biomes.zip'
    
    # URL for Brazilian biomes from Global Forest Watch
    # This URL contains the official IBGE biomes of Brazil
    url = "https://data.globalforestwatch.org/api/download/v1/dataset/54ec099791644be4b273d9d8a853d452_4/vector?&prefix=gfw_brazil_biomes"
    
    try:
        print("Downloading Brazil biomes shapefile...")
        response = requests.get(url)
        response.raise_for_status()  # Raise an exception for HTTP errors
        
        # Save the zip file
        with open('data/brazil_biomes.zip', 'wb') as f:
            f.write(response.content)
        
        print("Download complete!")
        return 'data/brazil_biomes.zip'
    
    except requests.exceptions.RequestException as e:
        print(f"Error downloading shapefile: {e}")
        return None


def extract_amazon_biome(zip_path):
    """
    Extract the Amazon biome from the Brazilian biomes shapefile
    """
    if not zip_path or not os.path.exists(zip_path):
        print("Zip file not found.")
        return None
    
    try:
        # Read the shapefile from the zip
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            # List all files to find the shapefile
            shapefile_name = None
            for name in zip_ref.namelist():
                if name.endswith('.shp'):
                    shapefile_name = name
                    break
            
            if not shapefile_name:
                print("No shapefile found in the zip file.")
                return None
            
            # Extract all related files to data directory
            zip_ref.extractall('data')
        
        # Read the shapefile
        shapefile_path = os.path.join('data', shapefile_name)
        gdf = gpd.read_file(shapefile_path)
        
        # Filter to get only the Amazon biome
        amazon_biome = gdf[gdf['name'].str.contains('Amazônia', case=False) | 
                           gdf['name'].str.contains('Amazonia', case=False)]
        
        if amazon_biome.empty:
            print("Amazon biome not found in the shapefile.")
            # Check what columns and values are available
            print("Available columns:", gdf.columns.tolist())
            print("Available values in the name column:", gdf['name'].unique())
            return None
        
        # Save the Amazon biome as a separate shapefile
        amazon_output_path = 'data/amazon_biome.shp'
        amazon_biome.to_file(amazon_output_path)
        print(f"Amazon biome shapefile saved to {amazon_output_path}")
        
        return amazon_output_path
    
    except Exception as e:
        print(f"Error extracting Amazon biome: {e}")
        return None


def create_simplified_amazon_shapefile():
    """
    Create a simplified Amazon rainforest boundary polygon based on approximate coordinates.
    This is a fallback if the download method fails.
    """
    try:
        # Approximate coordinates of Amazon rainforest boundary
        # These are simplified points that roughly outline the Amazon biome
        coordinates = [
            (-73.0, -13.0),  # Peru
            (-70.0, -15.0),  # Bolivia
            (-65.0, -15.0),  # Bolivia
            (-60.0, -13.0),  # Brazil/Bolivia border
            (-55.0, -10.0),  # Central Brazil
            (-50.0, -5.0),   # Eastern Amazon
            (-48.0, -2.0),   # Northeastern Brazil
            (-49.0, 2.0),    # Mouth of Amazon
            (-53.0, 5.0),    # Northern Brazil
            (-58.0, 6.0),    # Guyana
            (-62.0, 8.0),    # Venezuela
            (-67.0, 6.0),    # Colombia
            (-72.0, 0.0),    # Colombia/Peru border
            (-76.0, -5.0),   # Peru
            (-78.0, -8.0),   # Peru
            (-73.0, -13.0),  # Back to starting point
        ]
        
        # Create a polygon from the coordinates
        polygon = Polygon(coordinates)
        
        # Create a GeoDataFrame
        gdf = gpd.GeoDataFrame(index=[0], crs='EPSG:4326', geometry=[polygon])
        
        # Add attributes
        gdf['name'] = 'Amazon Rainforest'
        gdf['area_km2'] = gdf.geometry.area * 111 * 111  # Approximate conversion to km²
        
        # Save to shapefile
        output_path = 'data/amazon_rainforest_simplified.shp'
        gdf.to_file(output_path)
        
        print(f"Simplified Amazon rainforest shapefile created at {output_path}")
        return output_path
    
    except Exception as e:
        print(f"Error creating simplified Amazon shapefile: {e}")
        return None


def visualize_shapefile(shapefile_path):
    """
    Visualize the shapefile on a map
    """
    if not shapefile_path or not os.path.exists(shapefile_path):
        print("Shapefile not found for visualization.")
        return
    
    try:
        # Read the shapefile
        gdf = gpd.read_file(shapefile_path)
        
        # Create a figure
        fig, ax = plt.subplots(figsize=(10, 10))
        
        # Plot the shapefile
        gdf.plot(ax=ax, alpha=0.5, edgecolor='black')
        
        # Add title and labels
        ax.set_title('Amazon Rainforest Boundary')
        ax.set_xlabel('Longitude')
        ax.set_ylabel('Latitude')
        
        # Save the figure
        plt.savefig('data/amazon_rainforest_map.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        print("Map visualization saved to data/amazon_rainforest_map.png")
    
    except Exception as e:
        print(f"Error visualizing shapefile: {e}")


def main():
    print("Amazon Rainforest Shapefile Generator")
    print("--------------------------------------")
    
    # First try to download the official shapefile
    zip_path = download_brazilian_biomes_shapefile()
    
    if zip_path:
        # Extract the Amazon biome
        amazon_path = extract_amazon_biome(zip_path)
        
        if amazon_path:
            # Visualize the shapefile
            visualize_shapefile(amazon_path)
            print("\nSuccessfully created Amazon rainforest shapefile from official sources!")
            print(f"The shapefile is located at: {os.path.abspath(amazon_path)}")
            print("You can now use this shapefile for your deforestation detection project.")
            return
    
    # If official method fails, create a simplified version
    print("\nFalling back to simplified boundary creation...")
    simplified_path = create_simplified_amazon_shapefile()
    
    if simplified_path:
        # Visualize the shapefile
        visualize_shapefile(simplified_path)
        print("\nSuccessfully created a simplified Amazon rainforest shapefile!")
        print(f"The shapefile is located at: {os.path.abspath(simplified_path)}")
        print("You can now use this shapefile for your deforestation detection project.")
        print("\nNote: This is a simplified approximation. For more accurate boundaries,")
        print("consider obtaining official boundaries from IBGE or other sources.")
    else:
        print("\nFailed to create Amazon rainforest shapefile.")
        print("Please try again or obtain the shapefile manually from:")
        print("- IBGE (Brazilian Institute of Geography and Statistics)")
        print("- TerraBrasilis (http://terrabrasilis.dpi.inpe.br/)")
        print("- Global Forest Watch (https://data.globalforestwatch.org/)")


if __name__ == "__main__":
    main()
