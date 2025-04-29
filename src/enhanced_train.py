import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset, random_split
import numpy as np
import os
import yaml
import argparse
from tqdm import tqdm
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

# Import our modules
from src.models import Simple2DCNN, Simple3DCNN, ConvLSTM, UNet
from src.dataset import DeforestDataset
from src.utils import get_logger

class Trainer:
    def __init__(self, config_path):
        """
        Initialize the trainer with the given configuration
        
        Args:
            config_path: Path to the YAML configuration file
        """
        # Load configuration
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        # Set up logger
        self.logger = get_logger('DeforestTrainer')
        
        # Create output directories
        os.makedirs(self.config['training']['save_dir'], exist_ok=True)
        os.makedirs(self.config['evaluation']['visualization']['save_dir'], exist_ok=True)
        os.makedirs(self.config['logging']['save_dir'], exist_ok=True)
        
        # Set device
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.logger.info(f"Using device: {self.device}")
        
        # Set random seeds for reproducibility
        torch.manual_seed(42)
        np.random.seed(42)
        
    def prepare_data(self):
        """Prepare datasets and dataloaders"""
        self.logger.info("Preparing datasets...")
        
        # For temporal models, we need a different dataset
        is_temporal = self.config['training']['model_type'] in ['simple3dcnn', 'convlstm']
        
        # Create dataset based on model type
        if is_temporal:
            # We need to create a temporal dataset that loads sequences
            from dataset import TemporalDeforestDataset
            self.dataset = TemporalDeforestDataset(
                self.config['data']['processed_dir'],
                time_steps=self.config['training']['time_steps']
            )
        else:
            # Standard dataset for non-temporal models
            self.dataset = DeforestDataset(self.config['data']['processed_dir'])
        
        # Split dataset into training and validation
        val_size = int(len(self.dataset) * self.config['training']['validation_split'])
        train_size = len(self.dataset) - val_size
        
        self.train_dataset, self.val_dataset = random_split(
            self.dataset, [train_size, val_size]
        )
        
        self.logger.info(f"Training set size: {train_size}")
        self.logger.info(f"Validation set size: {val_size}")
        
        # Create data loaders
        self.train_loader = DataLoader(
            self.train_dataset,
            batch_size=self.config['training']['batch_size'],
            shuffle=True,
            num_workers=4
        )
        
        self.val_loader = DataLoader(
            self.val_dataset,
            batch_size=self.config['training']['batch_size'],
            shuffle=False,
            num_workers=4
        )
    
    def build_model(self):
        """Build the model based on configuration"""
        self.logger.info(f"Building {self.config['training']['model_type']} model...")
        
        in_channels = self.config['training']['in_channels']
        
        # Create model based on specified type
        if self.config['training']['model_type'] == 'simple2dcnn':
            self.model = Simple2DCNN(in_channels=in_channels)
        elif self.config['training']['model_type'] == 'simple3dcnn':
            self.model = Simple3DCNN(
                in_channels=in_channels, 
                time_steps=self.config['training']['time_steps']
            )
        elif self.config['training']['model_type'] == 'convlstm':
            self.model = ConvLSTM(
                input_channels=in_channels,
                hidden_channels=64,
                num_layers=2
            )
        elif self.config['training']['model_type'] == 'unet':
            self.model = UNet(in_channels=in_channels)
        else:
            raise ValueError(f"Unknown model type: {self.config['training']['model_type']}")
        
        # Move model to device
        self.model = self.model.to(self.device)
        
        # Print model summary
        self.logger.info(f"Model parameters: {sum(p.numel() for p in self.model.parameters())}")
        
        # Define loss function
        self.criterion = nn.BCELoss()
        
        # Define optimizer
        self.optimizer = optim.Adam(
            self.model.parameters(),
            lr=self.config['training']['lr'],
            weight_decay=self.config['training']['weight_decay']
        )
        
        # Learning rate scheduler
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, 
            mode='min', 
            factor=0.5, 
            patience=5
        )
    
    def train_epoch(self, epoch):
        """Train for one epoch"""
        self.model.train()
        epoch_loss = 0
        pbar = tqdm(self.train_loader, desc=f"Epoch {epoch+1}/{self.config['training']['epochs']}")
        
        for batch_idx, (inputs, targets) in enumerate(pbar):
            print(f"Targets min: {targets.min().item()}, max: {targets.max().item()}")

            # Move data to device
            inputs = inputs.to(self.device)
            targets = targets.to(self.device)
            
            # Zero the parameter gradients
            self.optimizer.zero_grad()
            
            # Forward pass
            outputs = self.model(inputs)
            loss = self.criterion(outputs, targets)
            
            # Backward pass and optimize
            loss.backward()
            self.optimizer.step()
            
            # Update statistics
            epoch_loss += loss.item()
            pbar.set_postfix({'loss': loss.item()})
        
        # Average loss for the epoch
        epoch_loss /= len(self.train_loader)
        return epoch_loss
    
    def validate(self):
        """Validate the model"""
        self.model.eval()
        val_loss = 0
        all_targets = []
        all_outputs = []
        
        with torch.no_grad():
            for inputs, targets in self.val_loader:
                # Move data to device
                inputs = inputs.to(self.device)
                targets = targets.to(self.device)
                
                # Forward pass
                outputs = self.model(inputs)
                loss = self.criterion(outputs, targets)
                
                # Update statistics
                val_loss += loss.item()
                
                # Collect predictions and targets for metrics
                all_outputs.append(outputs.cpu().numpy())
                all_targets.append(targets.cpu().numpy())
        
        # Average loss
        val_loss /= len(self.val_loader)
        
        # Concatenate all predictions and targets
        all_outputs = np.concatenate(all_outputs, axis=0)
        all_targets = np.concatenate(all_targets, axis=0)
        
        # Calculate metrics
        metrics = self.calculate_metrics(all_targets, all_outputs)
        metrics['loss'] = val_loss
        
        return metrics
    
    def calculate_metrics(self, targets, outputs):
        """Calculate evaluation metrics"""
        # Flatten arrays
        targets_flat = targets.reshape(-1).astype(float)
        outputs_flat = outputs.reshape(-1)
        
        # Binary predictions (threshold at 0.5)
        preds_binary = (outputs_flat > 0.5).astype(float)
        targets_binary = (targets_flat > 0.5).astype(float)
        
        metrics = {
        'accuracy': accuracy_score(targets_binary, preds_binary),
        'precision': precision_score(targets_binary, preds_binary, zero_division=0),
        'recall': recall_score(targets_binary, preds_binary, zero_division=0),
        'f1': f1_score(targets_binary, preds_binary, zero_division=0),
        }

        # Only calculate AUC if both classes are present
        unique_classes = np.unique(targets_binary)
        if len(unique_classes) > 1:
            metrics['auc'] = roc_auc_score(targets_binary, outputs_flat)
        else:
            metrics['auc'] = 0.5  # Default AUC for single-class data

        
        return metrics
    
    def visualize_predictions(self, epoch):
        """Visualize model predictions on validation data"""
        self.model.eval()
        
        # Get a batch from validation set
        inputs, targets = next(iter(self.val_loader))
        inputs = inputs.to(self.device)
        
        # Get predictions
        with torch.no_grad():
            outputs = self.model(inputs)
        
        # Move to CPU and convert to numpy
        inputs = inputs.cpu().numpy()
        targets = targets.cpu().numpy()
        outputs = outputs.cpu().numpy()
        
        # Number of samples to visualize
        n_samples = min(4, inputs.shape[0])
        
        # Create figure
        fig, axes = plt.subplots(n_samples, 3, figsize=(15, 4 * n_samples))
        
        for i in range(n_samples):
            # Input image (first 3 channels as RGB)
            if inputs.ndim == 5:  # For temporal models, take the last time step
                rgb = inputs[i, -1, :3]
            else:
                rgb = inputs[i, :3]
                
            # Normalize for visualization
            rgb = (rgb - rgb.min()) / (rgb.max() - rgb.min())
            
            # Transpose if needed
            if rgb.shape[0] == 3:
                rgb = np.transpose(rgb, (1, 2, 0))
            
            # Target and output
            target = targets[i, 0]
            output = outputs[i, 0]
            
            # Plot
            axes[i, 0].imshow(rgb)
            axes[i, 0].set_title('Input Image')
            axes[i, 0].axis('off')
            
            axes[i, 1].imshow(target, cmap='gray')
            axes[i, 1].set_title('Ground Truth')
            axes[i, 1].axis('off')
            
            axes[i, 2].imshow(output, cmap='gray')
            axes[i, 2].set_title('Prediction')
            axes[i, 2].axis('off')
        
        plt.tight_layout()
        
        # Save figure
        save_path = os.path.join(
            self.config['evaluation']['visualization']['save_dir'],
            f"epoch_{epoch+1}.png"
        )
        plt.savefig(save_path)
        plt.close(fig)
    
    def train(self):
        """Train the model"""
        self.logger.info("Starting training...")
        
        # Prepare data
        self.prepare_data()
        
        # Build model
        self.build_model()
        
        # Training loop
        best_val_loss = float('inf')
        patience_counter = 0
        
        for epoch in range(self.config['training']['epochs']):
            # Train for one epoch
            train_loss = self.train_epoch(epoch)
            
            # Validate
            val_metrics = self.validate()
            val_loss = val_metrics['loss']
            
            # Learning rate scheduler step
            self.scheduler.step(val_loss)
            
            # Log metrics
            self.logger.info(f"Epoch {epoch+1}/{self.config['training']['epochs']}")
            self.logger.info(f"  Train Loss: {train_loss:.4f}")
            self.logger.info(f"  Val Loss: {val_loss:.4f}")
            for metric, value in val_metrics.items():
                if metric != 'loss':
                    self.logger.info(f"  Val {metric.capitalize()}: {value:.4f}")
            
            # Visualize predictions
            if (epoch + 1) % 5 == 0:
                self.visualize_predictions(epoch)
            
            # Save best model
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                
                # Save model
                model_path = os.path.join(
                    self.config['training']['save_dir'],
                    f"{self.config['training']['model_type']}_best.pth"
                )
                torch.save(self.model.state_dict(), model_path)
                self.logger.info(f"  Saved best model to {model_path}")
            else:
                patience_counter += 1
            
            # Early stopping
            if patience_counter >= self.config['training']['early_stopping_patience']:
                self.logger.info(f"Early stopping after {epoch+1} epochs")
                break
        
        self.logger.info("Training complete!")

def main():
    parser = argparse.ArgumentParser(description='Train deforestation detection model')
    parser.add_argument('--config', type=str, default='config.yaml', help='Path to config file')
    args = parser.parse_args()
    
    trainer = Trainer(args.config)
    trainer.train()

if __name__ == '__main__':
    main()