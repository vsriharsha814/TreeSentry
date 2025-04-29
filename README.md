# Deforestation Detection Using Deep Learning

This project implements a comprehensive pipeline for detecting and monitoring deforestation using satellite imagery and deep learning. The system processes temporal satellite data from sources like Sentinel-2 and Landsat, and trains various neural network architectures to identify areas where deforestation has occurred.

<p align="center">
  <img src="results/visualizations/deforestation_prediction_example.png" alt="Deforestation Prediction Example" width="600"/>
  <br>
  <em>Example of deforestation detection visualization (will be generated when you run the project)</em>
</p>

## Features

- **Data acquisition** from Google Earth Engine
- **Processing of satellite imagery** into training samples
- **Multiple neural network architectures**:
  - UNet for semantic segmentation
  - 3D CNN for spatio-temporal analysis
  - ConvLSTM for time series analysis
  - Simple2DCNN for single-image analysis
- **Configuration-based pipeline**
- **Comprehensive evaluation metrics and visualizations**
- **Temporal change analysis tools**
- **Automated report generation**

## Prerequisites

- Python 3.7+
- Google Earth Engine account and authentication
- Required Python packages (see requirements.txt)
- Shapefile defining your area of interest

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/deforestation-detection.git
   cd deforestation-detection
   ```

2. Create a virtual environment and install requirements:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   pip install -r requirements.txt
   ```

3. Authenticate with Google Earth Engine:
   ```bash
   earthengine authenticate
   ```

## Quick Start

1. **Prepare your data**:
   - Place your boundary shapefile in the `data/` directory
   - The shapefile should define your area of interest

2. **Configure the project**:
   - Edit `config.yaml` to specify your parameters
   - Set satellite imagery sources, years, and model parameters

3. **Run the complete pipeline**:
   ```bash
   python main.py --config config.yaml --stage all
   ```

4. **Generate a report**:
   ```bash
   python src/generate_report.py --config config.yaml \
                                --predictions_dir results/predictions \
                                --evaluation_dir results/evaluation \
                                --changes_dir results/change_analysis \
                                --output_dir results/report
   ```

5. **View results**:
   - Open `results/report/deforestation_report.html` in a web browser
   - Check `results/visualizations/` for detailed visualizations

## Project Structure

```bash
deforestation-detection/
├── data/                     # Data directory
│   ├── raw/                  # For downloaded satellite data
│   ├── processed/            # For processed tiles
│   ├── boundary.shp          # You need to provide this
│   └── tiles.shp             # Optional
├── models/                   # Saved models
├── logs/                     # Training logs
├── results/                  # Results directory
│   ├── evaluation/           # Evaluation outputs
│   ├── predictions/          # Prediction outputs
│   ├── change_analysis/      # Change analysis outputs
│   ├── report/               # Generated reports
│   └── visualizations/       # Visualization outputs
├── src/                      # Source code
│   ├── data_loader.py        # Download satellite imagery
│   ├── data_prep.py          # Process and tile imagery
│   ├── dataset.py            # Dataset classes for PyTorch
│   ├── enhanced_train.py     # Enhanced training script
│   ├── evaluate_model.py     # Model evaluation
│   ├── models.py             # Neural network models
│   ├── predict.py            # Generate predictions
│   ├── train.py              # Basic training code
│   ├── utils.py              # Utility functions
│   ├── visualize_changes.py  # Visualize temporal changes
│   └── generate_report.py    # Generate HTML report
├── config.yaml               # Configuration
├── main.py                   # Entry point
├── requirements.txt          # Project dependencies
└── running_guide.md          # Detailed guide for running the project
```

## Detailed Documentation

For more detailed instructions, see `running_guide.md` for step-by-step guidance on running each component of the pipeline.

## Models

This project implements several neural network architectures for deforestation detection:

1. **UNet**: A convolutional network architecture for semantic segmentation with skip connections to preserve spatial details.

2. **Simple2DCNN**: A basic convolutional neural network for processing single satellite images.

3. **Simple3DCNN**: A 3D convolutional neural network for spatio-temporal analysis, processing a sequence of satellite images over time.

4. **ConvLSTM**: A convolutional LSTM for capturing temporal patterns in satellite imagery with the ability to learn long-term dependencies.

## Visualization Examples

The project generates various visualizations:

- Deforestation probability maps
- Binary deforestation masks
- Year-to-year change maps
- Temporal progression visualizations
- Evaluation metric charts

## License

[Insert your license information here]

## Acknowledgments

- Google Earth Engine for satellite imagery access
- PyTorch for deep learning framework
- Rasterio and GeoPandas for geospatial processing

## Contact

[Your contact information]
