#!/usr/bin/env python3
"""
Deforestation Detection Project
Script for visualizing deforestation changes over time
"""
import os
import argparse
import glob
import numpy as np
import rasterio
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import matplotlib.patches as mpatches
from utils import get_logger

logger = get_logger("ChangeVisualization")

def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description="Visualize Deforestation Changes Over Time")
    parser.add_argument('--input_dir', type=str, required=True, help='Directory containing prediction maps for multiple years')
    parser.add_argument('--output_dir', type=str, default='results/change_analysis', help='Directory to save change analysis results')
    parser.add_argument('--probability', action='store_true', help='Use probability maps instead of binary')
    return parser.parse_args()

def load_prediction_maps(input_dir, use_probability=False):
    """Load prediction maps for all available years"""
    # Find all relevant GeoTIFF files
    if use_probability:
        pattern = "deforestation_probability_*.tif"
    else:
        pattern = "deforestation_binary_*.tif"
    
    file_paths = sorted(glob.glob(os.path.join(input_dir, pattern)))
    
    if not file_paths:
        logger.error(f"No prediction maps found matching pattern {pattern} in {input_dir}")
        return None
    
    # Load each file and extract year from filename
    prediction_maps = {}
    
    for path in file_paths:
        # Extract year from filename
        filename = os.path.basename(path)
        year_str = filename.split('_')[-1].split('.')[0]
        
        try:
            year = int(year_str)
        except ValueError:
            logger.warning(f"Could not extract year from filename {filename}, skipping")
            continue
        
        # Load the file
        with rasterio.open(path) as src:
            prediction_maps[year] = {
                'data': src.read(1),
                'transform': src.transform,
                'crs': src.crs
            }
    
    logger.info(f"Loaded {len(prediction_maps)} prediction maps for years: {sorted(prediction_maps.keys())}")
    return prediction_maps

def calculate_change_maps(prediction_maps):
    """Calculate year-to-year change maps and overall change statistics"""
    years = sorted(prediction_maps.keys())
    
    if len(years) < 2:
        logger.error("Need at least two years to calculate changes")
        return None, None
    
    # Get reference shape from first year
    first_year = years[0]
    shape = prediction_maps[first_year]['data'].shape
    
    # Create change maps dictionary
    change_maps = {}
    
    # Calculate year-to-year changes
    for i in range(len(years) - 1):
        year1 = years[i]
        year2 = years[i + 1]
        
        # Calculate change (positive values indicate new deforestation)
        change = prediction_maps[year2]['data'] - prediction_maps[year1]['data']
        
        change_maps[(year1, year2)] = {
            'data': change,
            'transform': prediction_maps[year1]['transform'],
            'crs': prediction_maps[year1]['crs']
        }
    
    # Calculate overall change from first to last year
    overall_change = prediction_maps[years[-1]]['data'] - prediction_maps[years[0]]['data']
    
    change_maps[('overall', years[0], years[-1])] = {
        'data': overall_change,
        'transform': prediction_maps[years[0]]['transform'],
        'crs': prediction_maps[years[0]]['crs']
    }
    
    # Calculate statistics for each period
    statistics = {}
    
    for period, change_data in change_maps.items():
        change = change_data['data']
        
        # Calculate statistics
        new_deforestation = np.sum(change > 0)
        reforestation = np.sum(change < 0)
        unchanged = np.sum(change == 0)
        total_pixels = change.size
        
        # Convert to percentages
        statistics[period] = {
            'new_deforestation': new_deforestation,
            'new_deforestation_pct': (new_deforestation / total_pixels) * 100,
            'reforestation': reforestation,
            'reforestation_pct': (reforestation / total_pixels) * 100,
            'unchanged': unchanged,
            'unchanged_pct': (unchanged / total_pixels) * 100,
        }
    
    return change_maps, statistics

def create_change_visualizations(prediction_maps, change_maps, statistics, output_dir):
    """Create and save visualizations of deforestation changes"""
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Create a colormap for change maps
    # Red: new deforestation, Green: reforestation, Beige/Tan: unchanged
    colors = ['green', 'beige', 'red']
    change_cmap = ListedColormap(colors)
    
    # Create legend patches
    legend_patches = [
        mpatches.Patch(color='red', label='New Deforestation'),
        mpatches.Patch(color='beige', label='Unchanged'),
        mpatches.Patch(color='green', label='Reforestation')
    ]
    
    # Create visualizations for year-to-year changes
    for period, change_data in change_maps.items():
        if period[0] == 'overall':
            title = f"Overall Change ({period[1]} to {period[2]})"
            filename = f"change_overall_{period[1]}_to_{period[2]}.png"
        else:
            title = f"Change from {period[0]} to {period[1]}"
            filename = f"change_{period[0]}_to_{period[1]}.png"
        
        # Create a tri-color change map (red/beige/green)
        change = change_data['data']
        change_tri = np.zeros_like(change, dtype=np.int8)
        change_tri[change > 0] = 2  # New deforestation (red)
        change_tri[change < 0] = 0  # Reforestation (green)
        change_tri[change == 0] = 1  # Unchanged (beige)
        
        # Create figure
        plt.figure(figsize=(12, 10))
        
        # Plot change map
        plt.imshow(change_tri, cmap=change_cmap)
        
        # Add title and legend
        plt.title(title)
        plt.legend(handles=legend_patches, loc='lower right')
        plt.axis('off')
        
        # Add statistics to the plot
        stats = statistics[period]
        stats_text = (
            f"New Deforestation: {stats['new_deforestation_pct']:.2f}%\n"
            f"Reforestation: {stats['reforestation_pct']:.2f}%\n"
            f"Unchanged: {stats['unchanged_pct']:.2f}%"
        )
        
        plt.figtext(0.02, 0.02, stats_text, backgroundcolor='white', alpha=0.8)
        
        # Save the figure
        plt.savefig(os.path.join(output_dir, filename), dpi=300, bbox_inches='tight')
        plt.close()
    
    # Create a multi-year visualization showing progression
    years = sorted(prediction_maps.keys())
    
    # Only create this visualization if we have 3 or more years
    if len(years) >= 3:
        # Create a figure with one subplot for each year
        fig, axes = plt.subplots(1, len(years), figsize=(16, 5))
        
        if len(years) == 1:
            axes = [axes]  # Make it iterable if only one subplot
        
        # Plot each year
        for i, year in enumerate(years):
            prediction = prediction_maps[year]['data']
            axes[i].imshow(prediction, cmap='RdYlGn_r')
            axes[i].set_title(f"Year {year}")
            axes[i].axis('off')
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, "multi_year_progression.png"), dpi=300, bbox_inches='tight')
        plt.close()
    
    # Generate a deforestation timeline plot
    plt.figure(figsize=(10, 6))
    
    # Calculate total deforestation for each year
    deforestation_by_year = {year: np.mean(data['data']) * 100 for year, data in prediction_maps.items()}
    
    # Plot timeline
    plt.plot(list(deforestation_by_year.keys()), list(deforestation_by_year.values()), 
             marker='o', linestyle='-', linewidth=2)
    
    plt.xlabel('Year')
    plt.ylabel('Deforestation Coverage (%)')
    plt.title('Deforestation Timeline')
    plt.grid(True)
    plt.xticks(list(deforestation_by_year.keys()))
    
    # Save the figure
    plt.savefig(os.path.join(output_dir, "deforestation_timeline.png"), dpi=300, bbox_inches='tight')
    plt.close()
    
    # Save statistics to a text file
    stats_path = os.path.join(output_dir, "change_statistics.txt")
    with open(stats_path, 'w') as f:
        f.write("Deforestation Change Statistics\n")
        f.write("==============================\n\n")
        
        # Year-to-year statistics
        for period, stats in statistics.items():
            if period[0] == 'overall':
                f.write(f"OVERALL CHANGE ({period[1]} to {period[2]}):\n")
            else:
                f.write(f"CHANGE FROM {period[0]} TO {period[1]}:\n")
            
            f.write(f"  New Deforestation: {stats['new_deforestation']} pixels ({stats['new_deforestation_pct']:.2f}%)\n")
            f.write(f"  Reforestation: {stats['reforestation']} pixels ({stats['reforestation_pct']:.2f}%)\n")
            f.write(f"  Unchanged: {stats['unchanged']} pixels ({stats['unchanged_pct']:.2f}%)\n\n")
    
    logger.info(f"Saved change visualizations and statistics to {output_dir}")

def main():
    """Main function for change visualization"""
    # Parse arguments
    args = parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Load prediction maps
    logger.info(f"Loading prediction maps from {args.input_dir}")
    prediction_maps = load_prediction_maps(args.input_dir, use_probability=args.probability)
    
    if not prediction_maps:
        logger.error("No valid prediction maps found. Exiting.")
        return
    
    # Calculate change maps and statistics
    logger.info("Calculating change maps and statistics")
    change_maps, statistics = calculate_change_maps(prediction_maps)
    
    if not change_maps:
        logger.error("Failed to calculate change maps. Exiting.")
        return
    
    # Create visualizations
    logger.info("Creating change visualizations")
    create_change_visualizations(
        prediction_maps=prediction_maps,
        change_maps=change_maps,
        statistics=statistics,
        output_dir=args.output_dir
    )
    
    logger.info(f"Change analysis complete! Results saved to {args.output_dir}")

if __name__ == "__main__":
    main()