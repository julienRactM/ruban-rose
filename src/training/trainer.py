"""
Training infrastructure for breast cancer detection models
Handles training, validation, and medical-specific metrics
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from typing import Dict, List, Optional, Tuple
import time
import os
from pathlib import Path
import yaml
import logging


class MedicalMetrics:
    """Calculate medical-specific metrics"""
    
    @staticmethod
    def calculate_metrics(y_true: np.ndarray, y_pred: np.ndarray, y_proba: np.ndarray) -> Dict[str, float]:
        """Calculate comprehensive medical metrics"""
        
        # Basic metrics
        accuracy = accuracy_score(y_true, y_pred)
        
        # Medical-specific metrics (cancer = class 0, healthy = class 1)
        sensitivity = recall_score(y_true, y_pred, pos_label=0, zero_division=0)  # True positive rate for cancer
        specificity = recall_score(y_true, y_pred, pos_label=1, zero_division=0)  # True positive rate for healthy
        precision_cancer = precision_score(y_true, y_pred, pos_label=0, zero_division=0)
        precision_healthy = precision_score(y_true, y_pred, pos_label=1, zero_division=0)
        f1_cancer = f1_score(y_true, y_pred, pos_label=0, zero_division=0)
        f1_healthy = f1_score(y_true, y_pred, pos_label=1, zero_division=0)
        
        # Overall F1 (macro average)
        f1_macro = f1_score(y_true, y_pred, average='macro', zero_division=0)
        
        # AUC-ROC
        try:
            auc_roc = roc_auc_score(y_true, y_proba[:, 0]) if y_proba is not None else 0.0
        except:
            auc_roc = 0.0
        
        return {
            'accuracy': accuracy,
            'sensitivity': sensitivity,  # Cancer detection rate
            'specificity': specificity,  # Healthy detection rate
            'precision_cancer': precision_cancer,
            'precision_healthy': precision_healthy,
            'f1_score': f1_cancer,  # Primary F1 for cancer detection
            'f1_macro': f1_macro,
            'auc_roc': auc_roc
        }
    
    @staticmethod
    def print_metrics(metrics: Dict[str, float], prefix: str = ""):
        """Print metrics in medical format"""
        print(f"\n{prefix} Medical Metrics:")
        print(f"  Sensitivity (Cancer Detection): {metrics['sensitivity']:.4f}")
        print(f"  Specificity (Healthy Detection): {metrics['specificity']:.4f}")  
        print(f"  F1-Score (Cancer): {metrics['f1_score']:.4f}")
        print(f"  Accuracy: {metrics['accuracy']:.4f}")
        print(f"  AUC-ROC: {metrics['auc_roc']:.4f}")


class BreastCancerTrainer:
    """
    Trainer class for breast cancer detection models with medical focus
    """
    
    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        config: Dict,
        device: torch.device,
        save_dir: str = "models/checkpoints"
    ):
        """
        Initialize trainer
        
        Args:
            model: Model to train
            train_loader: Training data loader
            val_loader: Validation data loader
            config: Configuration dictionary
            device: Training device
            save_dir: Directory to save model checkpoints
        """
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.config = config
        self.device = device
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        
        # Training configuration
        training_config = config.get('training', {})
        self.epochs = config.get('models', {}).get('cnn', {}).get('epochs', 50)  # Default fallback
        self.patience = training_config.get('patience', 10)
        self.monitor_metric = training_config.get('monitor_metric', 'val_f1_score')
        self.save_best_only = training_config.get('save_best_only', True)
        
        # Setup optimizer and loss
        self._setup_optimizer()
        self._setup_loss_function()
        
        # Training tracking
        self.best_metric = 0.0
        self.patience_counter = 0
        self.training_history = {
            'train_loss': [], 'val_loss': [],
            'train_metrics': [], 'val_metrics': []
        }
        
        # Setup logging
        self._setup_logging()
    
    def _setup_optimizer(self):
        """Setup optimizer based on model type"""
        model_configs = self.config.get('models', {})
        
        # Determine model type and get learning rate
        if hasattr(self.model, 'backbone') and 'resnet' in str(type(self.model)).lower():
            lr = model_configs.get('resnet', {}).get('learning_rate', 0.0001)
        elif 'vit' in str(type(self.model)).lower() or 'transformer' in str(type(self.model)).lower():
            lr = model_configs.get('vision_transformer', {}).get('learning_rate', 0.00001)
        else:
            lr = model_configs.get('cnn', {}).get('learning_rate', 0.001)
        
        self.optimizer = optim.AdamW(
            self.model.parameters(),
            lr=lr,
            weight_decay=1e-5
        )
        
        # Learning rate scheduler
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer,
            mode='max',
            factor=0.5,
            patience=5
        )
    
    def _setup_loss_function(self):
        """Setup loss function with class weights"""
        # Get class weights from training dataset
        if hasattr(self.train_loader.dataset, 'get_class_weights'):
            class_weights = self.train_loader.dataset.get_class_weights().to(self.device)
        else:
            class_weights = torch.tensor([1.0, 1.0]).to(self.device)
        
        self.criterion = nn.CrossEntropyLoss(weight=class_weights)
    
    def _setup_logging(self):
        """Setup logging and tensorboard"""
        log_dir = self.config.get('logging', {}).get('log_file', 'logs/training.log')
        os.makedirs(os.path.dirname(log_dir), exist_ok=True)
        
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_dir),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)
        
        # Tensorboard
        if self.config.get('logging', {}).get('tensorboard', True):
            self.writer = SummaryWriter(log_dir='logs/tensorboard')
        else:
            self.writer = None
    
    def train_epoch(self) -> Tuple[float, Dict[str, float]]:
        """Train for one epoch"""
        self.model.train()
        running_loss = 0.0
        all_predictions = []
        all_labels = []
        all_probas = []
        
        for batch_idx, (data, target) in enumerate(self.train_loader):
            data, target = data.to(self.device), target.to(self.device)
            
            self.optimizer.zero_grad()
            output = self.model(data)
            loss = self.criterion(output, target)
            loss.backward()
            
            # Gradient clipping for stable training
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            
            self.optimizer.step()
            
            running_loss += loss.item()
            
            # Collect predictions for metrics
            probas = torch.softmax(output, dim=1)
            predictions = torch.argmax(output, dim=1)
            
            all_predictions.extend(predictions.detach().cpu().numpy())
            all_labels.extend(target.detach().cpu().numpy())
            all_probas.extend(probas.detach().cpu().numpy())
        
        # Calculate metrics
        avg_loss = running_loss / len(self.train_loader)
        metrics = MedicalMetrics.calculate_metrics(
            np.array(all_labels),
            np.array(all_predictions),
            np.array(all_probas)
        )
        
        return avg_loss, metrics
    
    def validate(self) -> Tuple[float, Dict[str, float]]:
        """Validate model"""
        self.model.eval()
        running_loss = 0.0
        all_predictions = []
        all_labels = []
        all_probas = []
        
        with torch.no_grad():
            for data, target in self.val_loader:
                data, target = data.to(self.device), target.to(self.device)
                
                output = self.model(data)
                loss = self.criterion(output, target)
                running_loss += loss.item()
                
                # Collect predictions
                probas = torch.softmax(output, dim=1)
                predictions = torch.argmax(output, dim=1)
                
                all_predictions.extend(predictions.detach().cpu().numpy())
                all_labels.extend(target.detach().cpu().numpy())
                all_probas.extend(probas.detach().cpu().numpy())
        
        # Calculate metrics
        avg_loss = running_loss / len(self.val_loader)
        metrics = MedicalMetrics.calculate_metrics(
            np.array(all_labels),
            np.array(all_predictions),
            np.array(all_probas)
        )
        
        return avg_loss, metrics
    
    def train(self, status_callback=None) -> Dict:
        """
        Full training loop
        
        Args:
            status_callback: Optional callback function to update training status
        
        Returns:
            Training history dictionary
        """
        self.logger.info(f"Starting training for {self.epochs} epochs")
        self.logger.info(f"Monitoring metric: {self.monitor_metric}")
        
        start_time = time.time()
        
        for epoch in range(self.epochs):
            epoch_start = time.time()
            
            # Training
            train_loss, train_metrics = self.train_epoch()
            
            # Validation
            val_loss, val_metrics = self.validate()
            
            # Update learning rate
            monitor_value = val_metrics.get('f1_score', 0)
            self.scheduler.step(monitor_value)
            
            # Update status callback if provided
            if status_callback:
                status_callback(epoch + 1, self.epochs, train_loss, val_loss, val_metrics)
            
            # Log metrics
            epoch_time = time.time() - epoch_start
            self.logger.info(
                f"Epoch {epoch+1}/{self.epochs} ({epoch_time:.2f}s) - "
                f"Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}"
            )
            
            MedicalMetrics.print_metrics(train_metrics, "Train")
            MedicalMetrics.print_metrics(val_metrics, "Validation")
            
            # Tensorboard logging
            if self.writer:
                self.writer.add_scalar('Loss/Train', train_loss, epoch)
                self.writer.add_scalar('Loss/Val', val_loss, epoch)
                for metric_name, value in val_metrics.items():
                    self.writer.add_scalar(f'Metrics/Val_{metric_name}', value, epoch)
            
            # Save checkpoint
            current_metric = val_metrics.get('f1_score', 0)
            is_best = current_metric > self.best_metric
            
            if is_best:
                self.best_metric = current_metric
                self.patience_counter = 0
                self._save_checkpoint(epoch, val_metrics, is_best=True)
            else:
                self.patience_counter += 1
            
            if not self.save_best_only:
                self._save_checkpoint(epoch, val_metrics, is_best=False)
            
            # Early stopping
            if self.patience_counter >= self.patience:
                self.logger.info(f"Early stopping after {epoch+1} epochs")
                break
            
            # Store history
            self.training_history['train_loss'].append(train_loss)
            self.training_history['val_loss'].append(val_loss)
            self.training_history['train_metrics'].append(train_metrics)
            self.training_history['val_metrics'].append(val_metrics)
        
        total_time = time.time() - start_time
        self.logger.info(f"Training completed in {total_time:.2f} seconds")
        self.logger.info(f"Best {self.monitor_metric}: {self.best_metric:.4f}")
        
        if self.writer:
            self.writer.close()
        
        return self.training_history
    
    def _save_checkpoint(self, epoch: int, metrics: Dict[str, float], is_best: bool = False):
        """Save model checkpoint"""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'metrics': metrics,
            'config': self.config
        }
        
        # Save checkpoint
        if is_best:
            checkpoint_path = self.save_dir / 'best_model.pth'
            torch.save(checkpoint, checkpoint_path)
            self.logger.info(f"Best model saved to {checkpoint_path}")
        else:
            checkpoint_path = self.save_dir / f'checkpoint_epoch_{epoch+1}.pth'
            torch.save(checkpoint, checkpoint_path)


def load_model_checkpoint(checkpoint_path: str, model: nn.Module, device: torch.device) -> Dict:
    """
    Load model from checkpoint
    
    Args:
        checkpoint_path: Path to checkpoint file
        model: Model instance
        device: Device to load model on
        
    Returns:
        Checkpoint information
    """
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    
    return {
        'epoch': checkpoint['epoch'],
        'metrics': checkpoint['metrics'],
        'config': checkpoint.get('config', {})
    }