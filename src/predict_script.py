#!/usr/bin/env python3
"""
Deforestation Detection Project
Script for making predictions on new satellite imagery
"""
import os
import argparse
import yaml
import numpy as np
import torch
import rasterio
from rasterio.windows import Window
from tqdm import tqdm
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import glob

# Import local modules
from models import Simple2DCNN, Simple3DCNN, ConvLSTM, UNet
from data_prep import DataPrep
from utils import get_logger

logger = get_logger("Prediction")

def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description="Generate Deforestation Predictions")
    parser.add_argument('--config', type=str, default='config.yaml', help='Path to config file')
    parser.add_argument('--model_path', type=str, required=True, help='Path to trained model weights')
    parser.add_argument('--input_dir', type=str, required=True, help='Directory containing input satellite imagery')
    parser.add_argument('--output_dir', type=str, default='results/predictions', help='Directory to save prediction results')
    parser.add_argument('--tile_size', type=int, default=256, help='Size of tiles for prediction')
    parser.add_argument('--overlap', type=int, default=32, help='Overlap between adjacent tiles')
    parser.add_argument('--threshold', type=float, default=0.5, help='Threshold for binary predictions')
    return parser.parse_args()

def load_model(model_type, model_path, in_channels, time_steps=None):
    """Load a trained model from disk"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Initialize model based on type
    if model_type == 'simple2dcnn':
        model = Simple2DCNN(in_channels=in_channels)
    elif model_type == 'simple3dcnn':
        model = Simple3DCNN(in_channels=in_channels, time_steps=time_steps)
    elif model_type == 'convlstm':
        model = ConvLSTM(input_channels=in_channels, hidden_channels=64, num_layers=2)
    elif model_type == 'unet':
        model = UNet(in_channels=in_channels)
    else:
        raise ValueError(f"Unknown model type: {model_type}")
    
    # Load weights
    model.load_state_dict(torch.load(model_path, map_location=device))
    model = model.to(device)
    model.eval()
    
    return model, device

def load_satellite_data(input_dir, config):
    """
    Load satellite data for prediction.
    Returns stacked arrays for each year in the format required by the model.
    """
    # Determine which layers to include based on config
    satellite = config['preprocessing']['satellite'].lower()
    if satellite == 'sentinel2':
        bands = ['B2', 'B3', 'B4', 'B8', 'B11', 'B12']
    elif satellite in ['landsat8', 'landsat9']:
        bands = ['SR_B2', 'SR_B3', 'SR_B4', 'SR_B5', 'SR_B6', 'SR_B7']
    else:
        bands = []
        logger.warning(f"Unknown satellite {satellite}, using default bands")
    
    # Add indices
    indices = config['preprocessing']['indices']
    
    # Add static layers if enabled
    static_layers = []
    if config['preprocessing'].get('use_static_layers', False):
        static_layers = list(config['preprocessing'].get('static_layers', {}).keys())
    
    # Define all input layers
    input_layers = bands + indices + static_layers
    
    # Get years from config
    years = config['preprocessing']['years']
    
    # Create a data prep instance for stacking bands
    data_prep = DataPrep(
        download_dir=input_dir,
        tile_size=args.tile_size,
        output_dir=os.path.join(args.output_dir, 'tmp')
    )
    
    # Stack data for each year
    stacked_data = {}
    
    for year in years:
        try:
            # Stack bands for this year
            stacked_array, transform, crs = data_prep.stack_bands(year, input_layers)
            stacked_data[year] = {
                'data': stacked_array,
                'transform': transform,
                'crs': crs
            }
        except Exception as e:
            logger.error(f"Error stacking bands for year {year}: {e}")
    
    return stacked_data

def predict_tiles(model, stacked_data, tile_size, overlap, threshold, device, model_type, time_steps=None):
    """
    Generate predictions by tiling the input images and running the model on each tile.
    """
    # For temporal models, we need data from multiple years
    is_temporal = model_type in ['simple3dcnn', 'convlstm']
    
    if is_temporal:
        # Make sure we have enough years of data
        years = sorted(stacked_data.keys())
        if len(years) < time_steps:
            logger.error(f"Not enough years of data for temporal model (need {time_steps}, got {len(years)})")
            return None, None, None, None
        
        # Use the latest years for prediction
        years_to_use = years[-time_steps:]
        prediction_year = years[-1]  # Use the latest year for output
        
        logger.info(f"Using years {years_to_use} for temporal prediction with target year {prediction_year}")
        
        # Get reference data from the latest year
        reference_data = stacked_data[prediction_year]
        data_shape = reference_data['data'].shape
        height, width = data_shape[1], data_shape[2]
        
        # Create a data preparation instance for tiling
        data_prep = DataPrep(
            download_dir="",  # Not used
            tile_size=tile_size,
            output_dir=""     # Not used
        )
        
        # Create a temporal stack [T, C, H, W]
        temporal_stack = np.stack([stacked_data[year]['data'] for year in years_to_use], axis=0)
        
        # Reshape to [C, T, H, W] for model
        temporal_stack = np.transpose(temporal_stack, (1, 0, 2, 3))
        
        # Create output array for predictions
        prediction_map = np.zeros((height, width), dtype=np.float32)
        count_map = np.zeros((height, width), dtype=np.int32)
        
        # Calculate effective step size
        step = tile_size - overlap
        
        # For each tile position
        for y in range(0, height - tile_size + 1, step):
            for x in range(0, width - tile_size + 1, step):
                # Extract tile
                tile = temporal_stack[:, :, y:y+tile_size, x:x+tile_size]
                
                # Convert to torch tensor
                tile_tensor = torch.from_numpy(tile).float().unsqueeze(0).to(device)
                
                # Run inference
                with torch.no_grad():
                    output = model(tile_tensor)
                
                # Add to prediction map
                prediction_map[y:y+tile_size, x:x+tile_size] += output[0, 0].cpu().numpy()
                count_map[y:y+tile_size, x:x+tile_size] += 1
        
        # Average overlapping predictions
        prediction_map = np.divide(prediction_map, count_map, out=np.zeros_like(prediction_map), where=count_map != 0)
        
        # Apply threshold for binary map
        binary_map = (prediction_map > threshold).astype(np.uint8)
        
        return prediction_map, binary_map, reference_data['transform'], reference_data['crs']
    
    else:
        # For non-temporal models, just use the most recent year
        years = sorted(stacked_data.keys())
        prediction_year = years[-1]
        
        logger.info(f"Using year {prediction_year} for non-temporal prediction")
        
        # Get data for this year
        data = stacked_data[prediction_year]['data']
        height, width = data.shape[1], data.shape[2]
        
        # Create a data preparation instance for tiling
        data_prep = DataPrep(
            download_dir="",  # Not used
            tile_size=tile_size,
            output_dir=""     # Not used
        )
        
        # Create output array for predictions
        prediction_map = np.zeros((height, width), dtype=np.float32)
        count_map = np.zeros((height, width), dtype=np.int32)
        
        # Calculate effective step size
        step = tile_size - overlap
        
        # For each tile position
        for y in range(0, height - tile_size + 1, step):
            for x in range(0, width - tile_size + 1, step):
                # Extract tile
                tile = data[:, y:y+tile_size, x:x+tile_size]
                
                # Convert to torch tensor
                tile_tensor = torch.from_numpy(tile).float().unsqueeze(0).to(device)
                
                # Run inference
                with torch.no_grad():
                    output = model(tile_tensor)
                
                # Add to prediction map
                prediction_map[y:y+tile_size, x:x+tile_size] += output[0, 0].cpu().numpy()
                count_map[y:y+tile_size, x:x+tile_size] += 1
        
        # Average overlapping predictions
        prediction_map = np.divide(prediction_map, count_map, out=np.zeros_like(prediction_map), where=count_map != 0)
        
        # Apply threshold for binary map
        binary_map = (prediction_map > threshold).astype(np.uint8)
        
        return prediction_map, binary_map, stacked_data[prediction_year]['transform'], stacked_data[prediction_year]['crs']

def save_prediction_maps(prediction_map, binary_map, transform, crs, output_dir, year=None):
    """Save prediction maps as GeoTIFF files"""
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Add year to filename if provided
    year_str = f"_{year}" if year else ""
    
    # Save probability map
    prob_path = os.path.join(output_dir, f"deforestation_probability{year_str}.tif")
    with rasterio.open(
        prob_path,
        'w',
        driver='GTiff',
        height=prediction_map.shape[0],
        width=prediction_map.shape[1],
        count=1,
        dtype=prediction_map.dtype,
        crs=crs,
        transform=transform
    ) as dst:
        dst.write(prediction_map, 1)
    
    # Save binary map
    binary_path = os.path.join(output_dir, f"deforestation_binary{year_str}.tif")
    with rasterio.open(
        binary_path,
        'w',
        driver='GTiff',
        height=binary_map.shape[0],
        width=binary_map.shape[1],
        count=1,
        dtype=binary_map.dtype,
        crs=crs,
        transform=transform
    ) as dst:
        dst.write(binary_map, 1)
    
    logger.info(f"Saved prediction maps to {output_dir}")
    return prob_path, binary_path

def create_visualization(input_data, prediction_map, binary_map, output_dir, year=None):
    """Create and save visualization of the predictions"""
    # Create output directory for visualizations
    viz_dir = os.path.join(output_dir, 'visualizations')
    os.makedirs(viz_dir, exist_ok=True)
    
    # Add year to filename if provided - ensure year is part of the filename
    year_str = f"_{year}" if year else ""
    
    # Extract RGB bands from input data (assuming first 3 channels are RGB)
    if input_data.shape[0] >= 3:
        rgb = input_data[:3]
        # Normalize for visualization
        for i in range(3):
            if rgb[i].max() > 0:
                rgb[i] = (rgb[i] - rgb[i].min()) / (rgb[i].max() - rgb[i].min())
        
        rgb = np.transpose(rgb, (1, 2, 0))
    else:
        # Create a grayscale image if RGB not available
        rgb = np.zeros((input_data.shape[1], input_data.shape[2], 3))
        if input_data.shape[0] > 0:
            gray = (input_data[0] - input_data[0].min()) / (input_data[0].max() - input_data[0].min() + 1e-8)
            rgb[:, :, 0] = gray
            rgb[:, :, 1] = gray
            rgb[:, :, 2] = gray
    
    # Create RGB visualization with prediction overlay
    plt.figure(figsize=(15, 15))
    
    # Plot RGB image
    plt.imshow(rgb)
    
    # Create a red mask for deforested areas (binary map)
    red_mask = np.zeros((binary_map.shape[0], binary_map.shape[1], 4))
    red_mask[:, :, 0] = 1.0  # Red
    red_mask[:, :, 3] = binary_map * 0.5  # Alpha channel (50% transparent)
    
    # Overlay the mask
    plt.imshow(red_mask)
    
    plt.title(f"Deforestation Prediction{' for ' + str(year) if year else ''}")
    plt.axis('off')
    
    # Save the visualization - ensure unique filename with year
    viz_path = os.path.join(viz_dir, f"deforestation_visualization{year_str}.png")
    plt.savefig(viz_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    # Create a heatmap visualization of probability
    plt.figure(figsize=(10, 10))
    
    # Create a colormap for the probability map
    cmap = plt.cm.RdYlGn_r  # Red (high) to green (low) colormap, reversed
    
    plt.imshow(prediction_map, cmap=cmap, vmin=0, vmax=1)
    plt.colorbar(label='Deforestation Probability')
    plt.title(f"Deforestation Probability{' for ' + str(year) if year else ''}")
    plt.axis('off')
    
    # Save the visualization - ensure unique filename with year
    prob_viz_path = os.path.join(viz_dir, f"deforestation_probability{year_str}.png")
    plt.savefig(prob_viz_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    logger.info(f"Saved visualizations to {viz_dir}")
    return viz_path, prob_viz_path

def main():
    """Main prediction function"""
    # Parse arguments
    global args
    args = parse_args()
    
    # Load configuration
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Extract model parameters from config
    model_type = config['training']['model_type']
    in_channels = config['training']['in_channels']
    time_steps = config.get('training', {}).get('time_steps', 4)
    
    # Load model
    logger.info(f"Loading {model_type} model from {args.model_path}")
    model, device = load_model(
        model_type=model_type,
        model_path=args.model_path,
        in_channels=in_channels,
        time_steps=time_steps
    )
    
    # Load satellite data
    logger.info(f"Loading satellite data from {args.input_dir}")
    stacked_data = load_satellite_data(args.input_dir, config)
    
    if not stacked_data:
        logger.error("No valid data found. Exiting.")
        return
    
    # Process each year separately
    for year, data in stacked_data.items():
        logger.info(f"Processing year {year}")
        
        # For temporal models, we need to ensure we have enough prior years
        if model_type in ['simple3dcnn', 'convlstm']:
            years = sorted(stacked_data.keys())
            current_year_idx = years.index(year)
            
            # Skip if we don't have enough prior years for temporal analysis
            if current_year_idx < time_steps - 1:
                logger.info(f"Skipping year {year} - not enough prior years for temporal analysis")
                continue
            
            # Create a temporal subset with current year and required prior years
            temporal_subset = {y: stacked_data[y] for y in years[current_year_idx-(time_steps-1):current_year_idx+1]}
            
            # Generate predictions for this temporal window
            prediction_map, binary_map, transform, crs = predict_tiles(
                model=model,
                stacked_data=temporal_subset,
                tile_size=args.tile_size,
                overlap=args.overlap,
                threshold=args.threshold,
                device=device,
                model_type=model_type,
                time_steps=time_steps
            )
        else:
            # For non-temporal models, just process the current year
            single_year_data = {year: data}
            prediction_map, binary_map, transform, crs = predict_tiles(
                model=model,
                stacked_data=single_year_data,
                tile_size=args.tile_size,
                overlap=args.overlap,
                threshold=args.threshold,
                device=device,
                model_type=model_type
            )
        
        if prediction_map is None:
            logger.error(f"Failed to generate predictions for year {year}. Skipping.")
            continue
        
        # Save prediction maps with proper year
        logger.info(f"Saving prediction maps for year {year}")
        prob_path, binary_path = save_prediction_maps(
            prediction_map=prediction_map,
            binary_map=binary_map,
            transform=transform,
            crs=crs,
            output_dir=args.output_dir,
            year=year
        )
        
        # Create visualizations with proper year
        logger.info(f"Creating visualizations for year {year}")
        create_visualization(
            input_data=data['data'],
            prediction_map=prediction_map,
            binary_map=binary_map,
            output_dir=args.output_dir,
            year=year
        )
    
    logger.info(f"Prediction complete! Results saved to {args.output_dir}")

if __name__ == "__main__":
    main()