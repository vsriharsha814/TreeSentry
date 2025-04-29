#!/usr/bin/env python3
"""
Deforestation Detection Project
Main entry point for running the pipeline
"""
import os
import argparse
import yaml
from tqdm import tqdm

# Import local modules
from src.data_loader import DataLoader
from src.data_prep import DataPrep
from src.enhanced_train import Trainer
from src.utils import get_logger

logger = get_logger("Main")

def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description="Deforestation Detection Pipeline")
    parser.add_argument('--config', type=str, default='config.yaml', 
                        help='Path to config file')
    parser.add_argument('--stage', type=str, choices=['download', 'preprocess', 'train', 'all'],
                        default='all', help='Pipeline stage to run')
    return parser.parse_args()

def download_stage(config):
    """Run the data download stage"""
    logger.info("Starting data download stage")
    
    # Ensure directories exist
    os.makedirs(config['data']['download_dir'], exist_ok=True)
    
    # Create data loader
    data_loader = DataLoader(
        boundary_shp=config['data']['boundary_shp'],
        tiles_shp=config.get('data', {}).get('tiles_shp'),
        download_dir=config['data']['download_dir']
    )
    
    # Get static layers if enabled
    static_layers = None
    if config['preprocessing'].get('use_static_layers', False):
        static_layers = config['preprocessing'].get('static_layers', {})
    
    # Download data for each year
    data_loader.download(
        years=config['preprocessing']['years'],
        satellite=config['preprocessing']['satellite'],
        indices=config['preprocessing']['indices'],
        static_layers=static_layers
    )
    
    logger.info("Data download stage complete")

def preprocess_stage(config):
    """Run the data preprocessing stage"""
    logger.info("Starting data preprocessing stage")
    
    # Ensure directories exist
    os.makedirs(config['data']['processed_dir'], exist_ok=True)
    
    # Create data preprocessor
    data_prep = DataPrep(
        download_dir=config['data']['download_dir'],
        tile_size=config['preprocessing']['tile_size'],
        output_dir=config['data']['processed_dir']
    )
    
    # Process each year
    for year in tqdm(config['preprocessing']['years'], desc="Processing years"):
        # Define input layers (satellite bands + indices)
        sat = config['preprocessing']['satellite'].lower()
        
        # Define bands based on satellite
        if sat == 'sentinel2':
            bands = ['B2', 'B3', 'B4', 'B8', 'B11', 'B12']
        elif sat in ['landsat8', 'landsat9']:
            bands = ['SR_B2', 'SR_B3', 'SR_B4', 'SR_B5', 'SR_B6', 'SR_B7']
        else:
            bands = []
            logger.warning(f"Unknown satellite {sat}, no bands defined")
        
        # Add indices
        indices = config['preprocessing']['indices']
        
        # Add static layers if enabled (no year suffix for static layers)
        static_layers = []
        if config['preprocessing'].get('use_static_layers', False):
            static_layers = list(config['preprocessing'].get('static_layers', {}).keys())
        
        # Define all input layers
        input_layers = bands + indices + static_layers
        
        # Use NDVI as label for now (can be changed to actual deforestation labels)
        label_layer = 'NDVI'
        if 'NDVI' not in indices:
            label_layer = indices[0] if indices else None
        
        # Process the data for this year
        data_prep.process_year(
            year=year,
            input_layers=input_layers,
            label_layer=label_layer,
            overlap=config['preprocessing']['overlap']
        )
    
    logger.info("Data preprocessing stage complete")

def train_stage(config):
    """Run the model training stage"""
    logger.info("Starting model training stage")

    args = parse_args()

    # Print input channels for debugging
    logger.info(f"Model configured for {config['training']['in_channels']} input channels")
    
    # Create and run trainer
    trainer = Trainer(config_path=args.config)
    trainer.train()
    
    logger.info("Model training stage complete")

def main():
    """Main entry point for the pipeline"""
    # Parse arguments
    args = parse_args()
    
    # Load configuration
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    
    # Create directories
    os.makedirs(config['data']['download_dir'], exist_ok=True)
    os.makedirs(config['data']['processed_dir'], exist_ok=True)
    os.makedirs(config['training']['save_dir'], exist_ok=True)
    os.makedirs(config['evaluation']['visualization']['save_dir'], exist_ok=True)
    os.makedirs(config['logging']['save_dir'], exist_ok=True)
    
    # Run the requested pipeline stage(s)
    if args.stage in ['download', 'all']:
        download_stage(config)
    
    if args.stage in ['preprocess', 'all']:
        preprocess_stage(config)
    
    if args.stage in ['train', 'all']:
        train_stage(config)
    
    logger.info("Pipeline completed successfully")

if __name__ == "__main__":
    main()