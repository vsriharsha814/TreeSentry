# Setting Up Amazon Rainforest Data for Deforestation Detection

This guide explains how to set up the Amazon rainforest boundary data needed for your deforestation detection project.

## Prerequisites

Before you begin, make sure you have:

1. Python 3.7+ installed
2. pip package manager
3. Git (optional, for cloning the repository)

## Step 1: Install Required Dependencies

The shapefile generator script requires several Python packages. Install them using pip:

```bash
pip install geopandas requests shapely matplotlib
```

If you encounter issues installing GeoPandas, you might need to install some system dependencies first. On Ubuntu/Debian:

```bash
sudo apt-get install python3-dev libgdal-dev
```

On macOS with Homebrew:

```bash
brew install gdal
```

## Step 2: Generate the Amazon Rainforest Shapefile

Run the provided script to generate the Amazon rainforest shapefile:

```bash
python amazon_shapefile_script.py
```

The script will:
1. Attempt to download official Brazilian biomes data from Global Forest Watch
2. Extract the Amazon biome from this data
3. If that fails, create a simplified approximation of the Amazon rainforest boundary
4. Save the shapefile to the `data/` directory
5. Generate a visualization of the boundary

## Step 3: Integrate with Your Deforestation Detection Project

After generating the shapefile, you'll need to integrate it with your deforestation detection project:

1. Copy or move the generated shapefile files (`.shp`, `.shx`, `.dbf`, `.prj`) to your project's `data/` directory
2. Rename the files to `boundary.shp` (and corresponding extensions) as required by the project

```bash
# Navigate to your project directory
cd path/to/deforestation-detection

# Create data directory if it doesn't exist
mkdir -p data

# Copy the shapefile files (replace with actual path if different)
cp path/to/amazon_shapefile_script/data/amazon_biome.* data/

# Rename to boundary.shp as expected by the project
mv data/amazon_biome.shp data/boundary.shp
mv data/amazon_biome.shx data/boundary.shx
mv data/amazon_biome.dbf data/boundary.dbf
mv data/amazon_biome.prj data/boundary.prj
```

## Step 4: Run the Deforestation Detection Pipeline

Now you can run your deforestation detection pipeline as described in the project documentation:

```bash
# Run the complete pipeline
python main.py --config config.yaml --stage all

# Or run individual stages
python main.py --config config.yaml --stage download
python main.py --config config.yaml --stage preprocess
python main.py --config config.yaml --stage train
```

## Alternative Data Sources

If you need more detailed or specific Amazon rainforest boundary data, consider these sources:

1. **IBGE (Brazilian Institute of Geography and Statistics)** - Official Brazilian biomes data
   - Website: https://www.ibge.gov.br/en/geosciences/maps/brazil-environmental-information/18341-biomes.html

2. **TerraBrasilis** - Platform by INPE for environmental monitoring data
   - Website: http://terrabrasilis.dpi.inpe.br/en/download-2/

3. **MapBiomas** - Annual land cover and use maps of Brazil
   - Website: https://brasil.mapbiomas.org/en/mapas-de-referencia/?cama_set_language=en

4. **Global Forest Watch** - Forest monitoring data
   - Website: https://data.globalforestwatch.org/datasets/54ec099791644be4b273d9d8a853d452_4

## Customizing the Boundary

If you need to focus on a specific region of the Amazon rainforest:

1. Open the shapefile in a GIS software like QGIS (free and open-source)
2. Edit the boundary to focus on your area of interest
3. Save the modified shapefile and use it in your project

## Troubleshooting

If you encounter issues:

- **Missing GDAL/OGR libraries**: Install the system GDAL libraries as mentioned in Step 1
- **Download errors**: Check your internet connection or try the simplified boundary creation
- **Memory errors**: For very large areas, consider creating smaller, more focused boundary tiles

## Next Steps

After setting up your Amazon rainforest boundary data, you can:

1. Download satellite imagery using the project's data acquisition tools
2. Process the imagery into training samples
3. Train one of the available neural network models
4. Generate deforestation predictions
5. Analyze changes over time

Refer to the project's main documentation for detailed instructions on each step.
