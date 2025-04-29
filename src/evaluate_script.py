#!/usr/bin/env python3
"""
Deforestation Detection Project
Script for evaluating trained models
"""
import os
import argparse
import yaml
import numpy as np
import torch
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from tqdm import tqdm
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, 
    roc_auc_score, confusion_matrix
)

# Import local modules
from models import Simple2DCNN, Simple3DCNN, ConvLSTM, UNet
from dataset import DeforestDataset, TemporalDeforestDataset
from utils import get_logger

logger = get_logger("Evaluation")

def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description="Evaluate Deforestation Detection Model")
    parser.add_argument('--config', type=str, default='config.yaml', help='Path to config file')
    parser.add_argument('--model_path', type=str, required=True, help='Path to trained model weights')
    parser.add_argument('--test_dir', type=str, required=True, help='Path to test data directory')
    parser.add_argument('--output_dir', type=str, default='results/evaluation', help='Directory to save evaluation results')
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

def create_dataloader(test_dir, model_type, batch_size=16, time_steps=None):
    """Create data loader for test data"""
    # For temporal models, use TemporalDeforestDataset
    if model_type in ['simple3dcnn', 'convlstm'] and time_steps:
        dataset = TemporalDeforestDataset(test_dir, time_steps=time_steps)
    else:
        dataset = DeforestDataset(test_dir)
    
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4
    )
    
    return dataloader

def evaluate(model, dataloader, device):
    """Evaluate the model on test data"""
    all_targets = []
    all_outputs = []
    
    with torch.no_grad():
        for inputs, targets in tqdm(dataloader, desc="Evaluating"):
            # Move data to device
            inputs = inputs.to(device)
            targets = targets.to(device)
            
            # Forward pass
            outputs = model(inputs)
            
            # Collect predictions and targets for metrics
            all_outputs.append(outputs.cpu().numpy())
            all_targets.append(targets.cpu().numpy())
    
    # Concatenate all predictions and targets
    all_outputs = np.concatenate(all_outputs, axis=0)
    all_targets = np.concatenate(all_targets, axis=0)
    
    return all_targets, all_outputs

def calculate_metrics(targets, outputs, threshold=0.5):
    """Calculate and return evaluation metrics"""
    # Flatten arrays for metric calculation
    targets_flat = targets.reshape(-1).astype(float)
    outputs_flat = outputs.reshape(-1)
    
    # Threshold predictions
    preds_binary = (outputs_flat > threshold).astype(float)
    
    # Calculate metrics
    metrics = {
        'accuracy': accuracy_score(targets_flat, preds_binary),
        'precision': precision_score(targets_flat, preds_binary, zero_division=0),
        'recall': recall_score(targets_flat, preds_binary, zero_division=0),
        'f1': f1_score(targets_flat, preds_binary, zero_division=0),
        'auc': roc_auc_score(targets_flat, outputs_flat)
    }
    
    # Calculate confusion matrix for more detailed analysis
    cm = confusion_matrix(targets_flat, preds_binary)
    
    return metrics, cm

def create_visualizations(targets, outputs, config, output_dir):
    """Create and save evaluation visualizations"""
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Select a subset of samples for visualization
    num_samples = min(10, targets.shape[0])
    sample_indices = np.random.choice(targets.shape[0], num_samples, replace=False)
    
    # Plot sample predictions vs ground truth
    fig, axes = plt.subplots(num_samples, 2, figsize=(10, 5 * num_samples))
    
    for i, idx in enumerate(sample_indices):
        # Ground truth
        axes[i, 0].imshow(targets[idx, 0], cmap='gray')
        axes[i, 0].set_title('Ground Truth')
        axes[i, 0].axis('off')
        
        # Prediction
        axes[i, 1].imshow(outputs[idx, 0], cmap='gray')
        axes[i, 1].set_title('Prediction')
        axes[i, 1].axis('off')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'sample_predictions.png'))
    plt.close(fig)
    
    # Calculate precision-recall curve for different thresholds
    thresholds = np.linspace(0, 1, 100)
    precisions = []
    recalls = []
    
    targets_flat = targets.reshape(-1).astype(float)
    outputs_flat = outputs.reshape(-1)
    
    for threshold in thresholds:
        preds = (outputs_flat > threshold).astype(float)
        precision = precision_score(targets_flat, preds, zero_division=0)
        recall = recall_score(targets_flat, preds, zero_division=0)
        precisions.append(precision)
        recalls.append(recall)
    
    # Plot precision-recall curve
    plt.figure(figsize=(10, 6))
    plt.plot(recalls, precisions)
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curve')
    plt.grid(True)
    plt.savefig(os.path.join(output_dir, 'precision_recall_curve.png'))
    plt.close()
    
    # Plot metric histogram for different thresholds
    metrics_data = {}
    
    for threshold in np.linspace(0.3, 0.7, 5):  # Focus on typical threshold range
        preds = (outputs_flat > threshold).astype(float)
        accuracy = accuracy_score(targets_flat, preds)
        precision = precision_score(targets_flat, preds, zero_division=0)
        recall = recall_score(targets_flat, preds, zero_division=0)
        f1 = f1_score(targets_flat, preds, zero_division=0)
        
        metrics_data[f'Threshold {threshold:.1f}'] = {
            'Accuracy': accuracy,
            'Precision': precision,
            'Recall': recall,
            'F1 Score': f1
        }
    
    # Plot metrics as grouped bar chart
    labels = list(metrics_data.keys())
    metrics_to_plot = ['Accuracy', 'Precision', 'Recall', 'F1 Score']
    
    x = np.arange(len(labels))
    width = 0.2
    
    fig, ax = plt.subplots(figsize=(12, 6))
    
    for i, metric in enumerate(metrics_to_plot):
        values = [metrics_data[label][metric] for label in labels]
        ax.bar(x + (i - 1.5) * width, values, width, label=metric)
    
    ax.set_xlabel('Threshold')
    ax.set_ylabel('Score')
    ax.set_title('Metrics at Different Thresholds')
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.legend()
    plt.grid(True, axis='y')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'metrics_by_threshold.png'))
    plt.close()

def main():
    """Main evaluation function"""
    # Parse arguments
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
    
    # Create data loader
    logger.info(f"Loading test data from {args.test_dir}")
    dataloader = create_dataloader(
        test_dir=args.test_dir,
        model_type=model_type,
        batch_size=config['training']['batch_size'],
        time_steps=time_steps
    )
    
    # Evaluate model
    logger.info("Evaluating model...")
    targets, outputs = evaluate(model, dataloader, device)
    
    # Calculate metrics
    logger.info("Calculating metrics...")
    metrics, confusion_matrix = calculate_metrics(targets, outputs)
    
    # Log metrics
    logger.info("Evaluation Results:")
    for metric, value in metrics.items():
        logger.info(f"  {metric.capitalize()}: {value:.4f}")
    
    # Log confusion matrix
    logger.info(f"Confusion Matrix:")
    logger.info(f"  True Positives: {confusion_matrix[1, 1]}")
    logger.info(f"  False Positives: {confusion_matrix[0, 1]}")
    logger.info(f"  True Negatives: {confusion_matrix[0, 0]}")
    logger.info(f"  False Negatives: {confusion_matrix[1, 0]}")
    
    # Save metrics to file
    metrics_path = os.path.join(args.output_dir, 'metrics.txt')
    with open(metrics_path, 'w') as f:
        f.write("Evaluation Results:\n")
        for metric, value in metrics.items():
            f.write(f"{metric.capitalize()}: {value:.4f}\n")
        
        f.write("\nConfusion Matrix:\n")
        f.write(f"True Positives: {confusion_matrix[1, 1]}\n")
        f.write(f"False Positives: {confusion_matrix[0, 1]}\n")
        f.write(f"True Negatives: {confusion_matrix[0, 0]}\n")
        f.write(f"False Negatives: {confusion_matrix[1, 0]}\n")
    
    # Create visualizations
    logger.info("Creating visualizations...")
    create_visualizations(targets, outputs, config, args.output_dir)
    
    logger.info(f"Evaluation complete! Results saved to {args.output_dir}")

if __name__ == "__main__":
    main()