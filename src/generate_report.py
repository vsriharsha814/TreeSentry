#!/usr/bin/env python3
"""
Deforestation Detection Project
Script to generate a comprehensive report with results and analysis
"""
import os
import argparse
import glob
import yaml
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
import rasterio
from src.utils import get_logger

logger = get_logger("ReportGenerator")

def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description="Generate Deforestation Analysis Report")
    parser.add_argument('--config', type=str, default='config.yaml', help='Path to config file')
    parser.add_argument('--predictions_dir', type=str, required=True, help='Directory containing prediction results')
    parser.add_argument('--evaluation_dir', type=str, required=True, help='Directory containing evaluation results')
    parser.add_argument('--changes_dir', type=str, required=True, help='Directory containing change analysis results')
    parser.add_argument('--output_dir', type=str, default='results/report', help='Directory to save report')
    parser.add_argument('--title', type=str, default='Deforestation Detection Analysis Report', help='Report title')
    return parser.parse_args()

def load_config(config_path):
    """Load and return project configuration"""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

def load_evaluation_metrics(evaluation_dir):
    """Load model evaluation metrics from file"""
    metrics_path = os.path.join(evaluation_dir, 'metrics.txt')
    
    if not os.path.exists(metrics_path):
        logger.error(f"Metrics file not found at {metrics_path}")
        return None
    
    metrics = {}
    
    with open(metrics_path, 'r') as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('Evaluation Results') or line.startswith('Confusion Matrix'):
                continue
            
            if ':' in line:
                key, value = line.split(':', 1)
                try:
                    metrics[key.strip()] = float(value.strip())
                except ValueError:
                    metrics[key.strip()] = value.strip()
    
    return metrics

def generate_html_report(config, predictions_dir, evaluation_dir, changes_dir, output_dir, title):
    """Generate an HTML report with all results and visualizations"""
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Load evaluation metrics
    metrics = load_evaluation_metrics(evaluation_dir)
    
    # Find all visualization images
    prediction_images = glob.glob(os.path.join(predictions_dir, 'visualizations', '*.png'))
    evaluation_images = glob.glob(os.path.join(evaluation_dir, '*.png'))
    change_images = glob.glob(os.path.join(changes_dir, '*.png'))
    
    # Get current date and time
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    # Create HTML content
    html_content = f"""
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>{title}</title>
        <style>
            body {{
                font-family: Arial, sans-serif;
                line-height: 1.6;
                margin: 0;
                padding: 0;
                color: #333;
            }}
            .container {{
                width: 90%;
                max-width: 1200px;
                margin: 0 auto;
                padding: 20px;
            }}
            h1, h2, h3 {{
                color: #2c3e50;
            }}
            h1 {{
                text-align: center;
                padding-bottom: 20px;
                border-bottom: 2px solid #eee;
            }}
            .section {{
                margin: 40px 0;
                padding-bottom: 20px;
                border-bottom: 1px solid #eee;
            }}
            .metrics-table {{
                width: 100%;
                border-collapse: collapse;
                margin: 20px 0;
            }}
            .metrics-table th, .metrics-table td {{
                padding: 12px 15px;
                text-align: left;
                border-bottom: 1px solid #ddd;
            }}
            .metrics-table th {{
                background-color: #f8f9fa;
            }}
            .gallery {{
                display: flex;
                flex-wrap: wrap;
                gap: 20px;
                margin: 20px 0;
            }}
            .gallery-item {{
                flex: 1 0 45%;
                max-width: 600px;
                margin-bottom: 20px;
            }}
            .gallery-item img {{
                width: 100%;
                border: 1px solid #ddd;
                border-radius: 4px;
            }}
            .gallery-item figcaption {{
                text-align: center;
                font-style: italic;
                margin-top: 10px;
            }}
            .footer {{
                text-align: center;
                margin-top: 40px;
                font-size: 0.8em;
                color: #777;
            }}
        </style>
    </head>
    <body>
        <div class="container">
            <h1>{title}</h1>
            <p><em>Generated on {now}</em></p>
            
            <div class="section">
                <h2>Project Configuration</h2>
                <h3>Data Sources</h3>
                <ul>
                    <li><strong>Satellite:</strong> {config['preprocessing']['satellite']}</li>
                    <li><strong>Years:</strong> {', '.join(map(str, config['preprocessing']['years']))}</li>
                    <li><strong>Indices:</strong> {', '.join(config['preprocessing']['indices'])}</li>
                </ul>
                
                <h3>Model Configuration</h3>
                <ul>
                    <li><strong>Model Type:</strong> {config['training']['model_type']}</li>
                    <li><strong>Input Channels:</strong> {config['training']['in_channels']}</li>
                    <li><strong>Time Steps:</strong> {config['training'].get('time_steps', 'N/A')}</li>
                </ul>
            </div>
    """
    
    # Add model evaluation section if metrics are available
    if metrics:
        html_content += f"""
            <div class="section">
                <h2>Model Evaluation</h2>
                <table class="metrics-table">
                    <tr>
                        <th>Metric</th>
                        <th>Value</th>
                    </tr>
        """
        
        # Add each metric to the table
        for key, value in metrics.items():
            if isinstance(value, float):
                html_content += f"""
                    <tr>
                        <td>{key}</td>
                        <td>{value:.4f}</td>
                    </tr>
                """
            else:
                html_content += f"""
                    <tr>
                        <td>{key}</td>
                        <td>{value}</td>
                    </tr>
                """
        
        html_content += """
                </table>
        """
        
        # Add evaluation visualizations
        if evaluation_images:
            html_content += """
                <h3>Evaluation Visualizations</h3>
                <div class="gallery">
            """
            
            for img_path in evaluation_images:
                img_name = os.path.basename(img_path)
                img_title = img_name.replace('.png', '').replace('_', ' ').title()
                
                # Copy image to report directory
                img_dest = os.path.join(output_dir, img_name)
                os.system(f'cp "{img_path}" "{img_dest}"')
                
                html_content += f"""
                    <figure class="gallery-item">
                        <img src="{img_name}" alt="{img_title}">
                        <figcaption>{img_title}</figcaption>
                    </figure>
                """
            
            html_content += """
                </div>
            """
        
        html_content += """
            </div>
        """
    
    # Add prediction results section
    if prediction_images:
        html_content += """
            <div class="section">
                <h2>Deforestation Predictions</h2>
                <div class="gallery">
        """
        
        for img_path in prediction_images:
            img_name = os.path.basename(img_path)
            img_title = img_name.replace('.png', '').replace('_', ' ').title()
            
            # Copy image to report directory
            img_dest = os.path.join(output_dir, img_name)
            os.system(f'cp "{img_path}" "{img_dest}"')
            
            html_content += f"""
                <figure class="gallery-item">
                    <img src="{img_name}" alt="{img_title}">
                    <figcaption>{img_title}</figcaption>
                </figure>
            """
        
        html_content += """
                </div>
            </div>
        """
    
    # Add change analysis section
    if change_images:
        html_content += """
            <div class="section">
                <h2>Deforestation Change Analysis</h2>
                <div class="gallery">
        """
        
        for img_path in change_images:
            img_name = os.path.basename(img_path)
            img_title = img_name.replace('.png', '').replace('_', ' ').title()
            
            # Copy image to report directory
            img_dest = os.path.join(output_dir, img_name)
            os.system(f'cp "{img_path}" "{img_dest}"')
            
            html_content += f"""
                <figure class="gallery-item">
                    <img src="{img_name}" alt="{img_title}">
                    <figcaption>{img_title}</figcaption>
                </figure>
            """
        
        html_content += """
                </div>
            </div>
        """
    
    # Add conclusions and recommendations
    html_content += """
            <div class="section">
                <h2>Conclusions and Recommendations</h2>
                <p>
                    This report presents the results of a deforestation detection analysis using 
                    deep learning techniques applied to satellite imagery. The model was trained to 
                    identify areas of deforestation from spectral signatures and temporal patterns 
                    in the data.
                </p>
                <p>
                    Based on the results, areas with high deforestation probability should be 
                    prioritized for field verification and potential conservation interventions. 
                    The change analysis highlights regions experiencing rapid forest loss, which 
                    may indicate illegal logging or land conversion activities.
                </p>
                <p>
                    For improved accuracy in future analyses, we recommend:
                </p>
                <ul>
                    <li>Increasing the temporal resolution by including more frequent satellite observations</li>
                    <li>Incorporating additional data sources such as radar imagery for cloud-penetrating capabilities</li>
                    <li>Validating results with ground-truth data from field surveys</li>
                    <li>Experimenting with ensemble methods that combine multiple model predictions</li>
                </ul>
            </div>
            
            <div class="footer">
                <p>Deforestation Detection Project &copy; 2023</p>
            </div>
        </div>
    </body>
    </html>
    """
    
    # Write HTML report to file
    report_path = os.path.join(output_dir, 'deforestation_report.html')
    with open(report_path, 'w') as f:
        f.write(html_content)
    
    logger.info(f"Generated HTML report at {report_path}")
    return report_path

def main():
    """Main function to generate the report"""
    # Parse arguments
    args = parse_args()
    
    # Load configuration
    config = load_config(args.config)
    
    # Generate report
    logger.info(f"Generating deforestation detection report")
    report_path = generate_html_report(
        config=config,
        predictions_dir=args.predictions_dir,
        evaluation_dir=args.evaluation_dir,
        changes_dir=args.changes_dir,
        output_dir=args.output_dir,
        title=args.title
    )
    
    logger.info(f"Report generation complete! Report saved to {report_path}")

if __name__ == "__main__":
    main()