# Deforestation Detection Using Deep Learning

This project uses satellite imagery and deep learning to detect and monitor deforestation. The system processes temporal satellite data from sources like Sentinel-2 and Landsat, and trains various neural network architectures to identify areas where deforestation has occurred.

## Features

- Data acquisition from Google Earth Engine
- Processing of satellite imagery into training samples
- Multiple neural network architectures:
  - UNet for semantic segmentation
  - 3D CNN for spatio-temporal analysis
  - ConvLSTM for time series analysis
- Configuration-based pipeline
- Visualization and evaluation tools

## Project Structure

```
deforestation-detection/
├── data/
│   ├── raw/          # For downloaded satellite data
│   ├── processed/    # For processed tiles
│   ├── boundary.shp  # You need to provide this
│   └── tiles.shp     # Optional
├── models/           # Saved models
├── logs/             # Training logs
├── results/
│   └── visualizations/ # Visualization outputs
├── src/              # Source code
├── config.yaml       # Configuration
├── main.py           # Entry point
└── README.md         # This file
```

## Prerequisites

- Python 3.7+
- Google Earth Engine account and authentication
- Required Python packages (see requirements.txt)
- Shapefile defining your area of interest

## Installation

1. Clone the repository:
   ```
   git clone https://github.com/yourusername/deforestation-detection.git
   cd deforestation-detection
   ```

2. Install requirements:
   ```
   pip install -r requirements.txt
   ```

3. Authenticate with Google Earth Engine:
   ```
   earthengine authenticate
   ```

## How to Use

### 1. Prepare Your Data

Place your boundary shapefile (defining your area of interest) in the `data/` directory.

### 2. Configure the Project

Edit `config.yaml` to specify your parameters:
- Data paths
- Satellite imagery sources and years
- Preprocessing parameters
- Model architecture and training parameters

### 3. Run the Pipeline

To run the complete pipeline:
```
python main.py --config config.yaml --stage all
```

Or run individual stages:
```
python main.py --config config.yaml --stage download    # Download data
python main.py --config config.yaml --stage preprocess  # Process data
python main.py --config config.yaml --stage train       # Train models
```

### 4. Evaluate Results

The trained models will be stored in the `models/` directory. Visualizations will be saved in `results/visualizations/`.

## Models

This project implements several neural network architectures for deforestation detection:

1. **UNet**: A convolutional network architecture for semantic segmentation.
2. **Simple3DCNN**: A 3D convolutional neural network for spatio-temporal analysis.
3. **ConvLSTM**: A convolutional LSTM for capturing temporal patterns in satellite imagery.

## License

[Insert your license information here]

## Acknowledgments

- [Add any acknowledgments or references here]

## Contact

[Your contact information]