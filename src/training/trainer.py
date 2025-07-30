"""
Training infrastructure for breast cancer detection models
Handles training, validation, and medical-specific metrics
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import torch.backends.mps
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, matthews_corrcoef, confusion_matrix
from typing import Dict, List, Optional, Tuple
import time
import os
from pathlib import Path
import yaml
import logging
import platform
import gc
import pickle
import json


class MedicalMetrics:
    """Calculate medical-specific metrics"""
    
    @staticmethod
    def calculate_metrics(y_true: np.ndarray, y_pred: np.ndarray, y_proba: np.ndarray) -> Dict[str, float]:
        """Calculate comprehensive medical metrics"""
        
        # Ensure proper data types for sklearn compatibility
        y_true = np.array(y_true, dtype=np.int32)
        y_pred = np.array(y_pred, dtype=np.int32)
        if y_proba is not None:
            y_proba = np.array(y_proba, dtype=np.float32)
        
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
        
        # Matthews Correlation Coefficient (MCC) - excellent for medical imaging
        mcc = matthews_corrcoef(y_true, y_pred)
        
        # Confusion Matrix for detailed analysis
        cm = confusion_matrix(y_true, y_pred)
        
        return {
            'accuracy': accuracy,
            'sensitivity': sensitivity,  # Cancer detection rate
            'recall': sensitivity,  # Same as sensitivity - cancer detection rate
            'specificity': specificity,  # Healthy detection rate
            'precision_cancer': precision_cancer,
            'precision_healthy': precision_healthy,
            'f1_score': f1_cancer,  # Primary F1 for cancer detection
            'f1_macro': f1_macro,
            'auc_roc': auc_roc,
            'mcc': mcc,  # Matthews Correlation Coefficient
            'confusion_matrix': cm.tolist()  # Convert to list for JSON serialization
        }
    
    @staticmethod
    def print_metrics(metrics: Dict[str, float], prefix: str = ""):
        """Print metrics in medical format"""
        print(f"\n{prefix} Medical Metrics:")
        print(f"  Recall (Cancer Detection): {metrics['recall']:.4f}")
        print(f"  Sensitivity (Cancer Detection): {metrics['sensitivity']:.4f}")
        print(f"  Specificity (Healthy Detection): {metrics['specificity']:.4f}")  
        print(f"  F1-Score (Cancer): {metrics['f1_score']:.4f}")
        print(f"  Matthews Correlation Coefficient: {metrics['mcc']:.4f}")
        print(f"  Accuracy: {metrics['accuracy']:.4f}")
        print(f"  AUC-ROC: {metrics['auc_roc']:.4f}")
        
        # Print confusion matrix in a readable format
        if 'confusion_matrix' in metrics:
            cm = metrics['confusion_matrix']
            print(f"  Confusion Matrix:")
            print(f"    Predicted:  [Cancer] [Healthy]")
            print(f"    Cancer:     [{cm[0][0]:6d}] [{cm[0][1]:7d}]")
            print(f"    Healthy:    [{cm[1][0]:6d}] [{cm[1][1]:7d}]")


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
        
        # Fix: Determine epochs based on actual model type, not just CNN
        # Also check if epochs is directly provided in the config (from Optuna optimization)
        model_configs = config.get('models', {})
        
        # First priority: directly specified epochs in model config (from Optuna)
        model_type_name = str(type(model)).lower()
        if ('resnet' in model_type_name) or hasattr(model, 'backbone'):
            # ResNet model (both MedicalResNet and LightweightResNet)
            resnet_config = model_configs.get('resnet', {})
            self.epochs = resnet_config.get('epochs', 30)
        elif 'vit' in model_type_name or 'transformer' in model_type_name:
            # Vision Transformer model
            vit_config = model_configs.get('vision_transformer', {})
            self.epochs = vit_config.get('epochs', 20)
        else:
            # CNN or other models
            cnn_config = model_configs.get('cnn', {})
            self.epochs = cnn_config.get('epochs', 50)
        
        # Log epochs configuration for debugging
        print(f"Training epochs set to: {self.epochs} for model type: {type(model).__name__}")
        
        self.patience = training_config.get('patience', 10)
        self.monitor_metric = training_config.get('monitor_metric', 'val_f1_score')
        self.save_best_only = training_config.get('save_best_only', True)
        
        # Setup logging first (needed by other methods)
        self._setup_logging()
        
        # M4 Pro specific optimizations
        self.is_m4_pro = self._detect_m4_pro()
        # Disable mixed precision on MPS due to float64 conversion issues in PyTorch
        self.use_mixed_precision = training_config.get('mixed_precision', True) and self.is_m4_pro and device.type != 'mps'
        self.scaler = torch.GradScaler('cuda') if self.use_mixed_precision and device.type == 'cuda' else None
        
        # Setup M4 Pro optimizations
        if self.is_m4_pro:
            self._setup_m4_pro_optimizations()
        
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
        
        if self.is_m4_pro:
            if device.type == 'mps':
                self.logger.info(f"M4 Pro optimizations enabled: Mixed Precision disabled on MPS (PyTorch limitation)")
            else:
                self.logger.info(f"M4 Pro optimizations enabled: Mixed Precision={self.use_mixed_precision}")
    
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
        
        # M4 Pro specific loss optimizations
        if self.is_m4_pro and self.device.type == 'mps':
            # Ensure loss computation is optimized for MPS
            self.criterion = self.criterion.to(self.device)
    
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
    
    def _detect_m4_pro(self) -> bool:
        """Detect if running on Apple M4 Pro"""
        try:
            if platform.system() == 'Darwin':
                import subprocess
                result = subprocess.run(['sysctl', '-n', 'machdep.cpu.brand_string'], 
                                      capture_output=True, text=True)
                cpu_brand = result.stdout.strip()
                return 'Apple M4 Pro' in cpu_brand or 'M4 Pro' in cpu_brand
        except:
            pass
        return torch.backends.mps.is_available()  # Fallback to MPS availability
    
    def _setup_m4_pro_optimizations(self):
        """Setup M4 Pro specific optimizations"""
        if self.device.type == 'mps':
            # Enable MPS optimizations
            try:
                # Set optimal memory allocation
                torch.mps.set_per_process_memory_fraction(0.85)
                
                # Enable graph mode for better performance
                if hasattr(torch.backends.mps, 'enable_graph_mode'):
                    torch.backends.mps.enable_graph_mode(True)
                
                if hasattr(self, 'logger') and self.logger:
                    self.logger.info("M4 Pro MPS optimizations enabled")
            except Exception as e:
                if hasattr(self, 'logger') and self.logger:
                    self.logger.warning(f"Could not enable all MPS optimizations: {e}")
                else:
                    print(f"Warning: Could not enable all MPS optimizations: {e}")
    
    def _cleanup_memory(self):
        """Cleanup memory for M4 Pro"""
        gc.collect()
        if self.device.type == 'mps':
            try:
                torch.mps.empty_cache()
            except:
                pass
    
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
            
            # Mixed precision training (disabled on MPS due to PyTorch limitations)
            if self.use_mixed_precision and self.scaler is not None and self.device.type == 'cuda':
                with torch.autocast(device_type='cuda', dtype=torch.float16):
                    output = self.model(data)
                    loss = self.criterion(output, target)
                
                self.scaler.scale(loss).backward()
                
                # Gradient clipping with scaler
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                # Standard training (used on MPS and when mixed precision is disabled)
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
            
            # Convert to CPU and ensure proper dtypes for sklearn compatibility
            all_predictions.extend(predictions.detach().cpu().numpy().astype(np.int32))
            all_labels.extend(target.detach().cpu().numpy().astype(np.int32))
            all_probas.extend(probas.detach().cpu().numpy().astype(np.float32))
        
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
                
                # Use mixed precision for validation (disabled on MPS due to PyTorch limitations)
                if self.use_mixed_precision and self.device.type == 'cuda':
                    with torch.autocast(device_type='cuda', dtype=torch.float16):
                        output = self.model(data)
                        loss = self.criterion(output, target)
                else:
                    output = self.model(data)
                    loss = self.criterion(output, target)
                
                running_loss += loss.item()
                
                # Collect predictions
                probas = torch.softmax(output, dim=1)
                predictions = torch.argmax(output, dim=1)
                
                # Convert to CPU and ensure proper dtypes for sklearn compatibility
                all_predictions.extend(predictions.detach().cpu().numpy().astype(np.int32))
                all_labels.extend(target.detach().cpu().numpy().astype(np.int32))
                all_probas.extend(probas.detach().cpu().numpy().astype(np.float32))
        
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
            
            # M4 Pro memory cleanup between epochs
            if self.is_m4_pro and epoch % 5 == 0:  # Cleanup every 5 epochs
                self._cleanup_memory()
        
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
            
            # Save detailed predictions and analysis for best model
            self._save_best_model_analysis(metrics)
        else:
            checkpoint_path = self.save_dir / f'checkpoint_epoch_{epoch+1}.pth'
            torch.save(checkpoint, checkpoint_path)
    
    def _save_best_model_analysis(self, metrics: Dict[str, float]):
        """Save detailed prediction analysis for the best model"""
        try:
            # Generate predictions on validation set for detailed analysis
            predictions_data = self._generate_detailed_predictions()
            
            # Create analysis directory
            analysis_dir = self.save_dir / 'best_model_analysis'
            analysis_dir.mkdir(exist_ok=True)
            
            # Save prediction matrix and probabilities
            prediction_matrix_path = analysis_dir / 'prediction_matrix.pkl'
            with open(prediction_matrix_path, 'wb') as f:
                pickle.dump(predictions_data, f)
            
            # Save metrics and confusion matrix as JSON
            metrics_path = analysis_dir / 'detailed_metrics.json'
            # Convert numpy arrays to lists for JSON serialization
            json_metrics = {}
            for key, value in metrics.items():
                if isinstance(value, np.ndarray):
                    json_metrics[key] = value.tolist()
                else:
                    json_metrics[key] = float(value) if isinstance(value, (np.float32, np.float64)) else value
            
            with open(metrics_path, 'w') as f:
                json.dump(json_metrics, f, indent=2)
            
            # Save confusion matrix visualization data
            cm_data = {
                'confusion_matrix': metrics['confusion_matrix'],
                'class_names': ['Cancer (0)', 'Healthy (1)'],
                'model_type': str(type(self.model).__name__),
                'timestamp': time.strftime('%Y-%m-%d %H:%M:%S')
            }
            
            cm_path = analysis_dir / 'confusion_matrix_data.json'
            with open(cm_path, 'w') as f:
                json.dump(cm_data, f, indent=2)
            
            self.logger.info(f"Best model analysis saved to {analysis_dir}")
            
        except Exception as e:
            self.logger.error(f"Failed to save best model analysis: {str(e)}")
    
    def _generate_detailed_predictions(self) -> Dict:
        """Generate detailed predictions on validation set"""
        self.model.eval()
        all_predictions = []
        all_labels = []
        all_probabilities = []
        all_image_paths = []
        
        with torch.no_grad():
            for batch_idx, (data, target) in enumerate(self.val_loader):
                data, target = data.to(self.device), target.to(self.device)
                
                # Get model output
                output = self.model(data)
                probabilities = torch.softmax(output, dim=1)
                predictions = torch.argmax(output, dim=1)
                
                # Convert to CPU and store
                batch_predictions = predictions.cpu().numpy()
                batch_labels = target.cpu().numpy()
                batch_probabilities = probabilities.cpu().numpy()
                
                all_predictions.extend(batch_predictions)
                all_labels.extend(batch_labels)
                all_probabilities.extend(batch_probabilities)
                
                # Try to get image paths if available in dataset
                if hasattr(self.val_loader.dataset, 'get_sample_info'):
                    for i in range(len(batch_predictions)):
                        sample_idx = batch_idx * self.val_loader.batch_size + i
                        if sample_idx < len(self.val_loader.dataset):
                            info = self.val_loader.dataset.get_sample_info(sample_idx)
                            all_image_paths.append(info.get('path', f'sample_{sample_idx}'))
                        else:
                            all_image_paths.append(f'sample_{sample_idx}')
                else:
                    # Fallback to indices
                    for i in range(len(batch_predictions)):
                        sample_idx = batch_idx * self.val_loader.batch_size + i
                        all_image_paths.append(f'sample_{sample_idx}')
        
        return {
            'predictions': np.array(all_predictions),
            'true_labels': np.array(all_labels),
            'probabilities': np.array(all_probabilities),
            'image_paths': all_image_paths,
            'model_type': str(type(self.model).__name__),
            'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
            'validation_samples': len(all_predictions)
        }


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