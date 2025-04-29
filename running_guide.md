# Step-by-Step Guide to Run the Deforestation Detection Project

## 1. Set Up Your Environment

First, create a Python environment and install the dependencies:

```bash
# Create a virtual environment (optional but recommended)
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

## 2. Authenticate with Google Earth Engine

This project requires access to Google Earth Engine for downloading satellite imagery:

```bash
# Install the Earth Engine command line tool if not already done
pip install earthengine-api

# Authenticate with your Google account
earthengine authenticate
```

Follow the authentication flow in your browser to grant access.

## 3. Prepare Your Boundary Data

You'll need a shapefile defining your area of interest:

1. Place your shapefile in the `data/` directory
2. The shapefile should be named `boundary.shp` (or update the path in `config.yaml`)
3. Optionally, you can provide a `tiles.shp` if you want to split the area into processing tiles

> Note: If you don't have a shapefile, you can create one using QGIS or similar GIS software.

## 4. Organize Your Project Structure

Ensure your project structure looks like this:

```
deforestation-detection/
├── data/
│   ├── boundary.shp  (and associated .dbf, .shx files)
│   └── tiles.shp     (optional)
├── src/
│   ├── data_loader.py
│   ├── data_prep.py
│   ├── dataset.py    (newly created)
│   ├── models.py
│   ├── utils.py
│   ├── train.py
│   ├── enhanced_train.py
│   └── evaluate.py
├── config.yaml
├── main.py           (newly created)
├── requirements.txt  (newly created)
└── README.md         (newly created)
```

## 5. Configure the Project

Review and edit the `config.yaml` file to match your specific needs:

- Update the data paths if needed
- Adjust the years of satellite imagery you want to process
- Select the type of satellite data (sentinel2, landsat8, or landsat9)
- Choose the model architecture you want to use
- Tune training parameters

## 6. Run the Complete Pipeline

To run the entire pipeline from data download to model training:

```bash
python main.py --config config.yaml --stage all
```

## 7. Run Individual Stages (if needed)

If you prefer to run the stages separately or need to restart from a specific point:

### a. Download satellite data:
```bash
python main.py --config config.yaml --stage download
```

### b. Preprocess the data:
```bash
python main.py --config config.yaml --stage preprocess
```

### c. Train the model:
```bash
python main.py --config config.yaml --stage train
```

## 8. Monitor Training Progress

During training, you can:
- Check the console output for progress and metrics
- View log files in the `logs/` directory
- See visualizations in `results/visualizations/` (created every 5 epochs)

## 9. Use the Trained Model

After training completes, the best model will be saved in the `models/` directory. You can use this model to:
- Make predictions on new satellite imagery
- Create deforestation maps
- Monitor changes over time

## Troubleshooting

### Common Issues:

1. **Earth Engine Authentication Errors**:
   - Make sure you've completed the authentication process
   - Try running `earthengine authenticate --quiet` again

2. **Missing Dependencies**:
   - Ensure all packages in requirements.txt are installed
   - Some packages may require additional system libraries

3. **Memory Issues During Training**:
   - Reduce batch size in config.yaml
   - Process smaller areas or create smaller tiles

4. **Missing or Incomplete Data**:
   - Check the data download stage logs
   - Ensure your boundary shapefile is correctly formatted
   - Check for sufficient cloud-free imagery in your region of interest

5. **GPU Issues**:
   - If using GPU, ensure CUDA is properly installed
   - Set device to 'cpu' in the code if GPU is unavailable
