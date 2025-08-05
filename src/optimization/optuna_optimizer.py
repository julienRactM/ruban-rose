"""
Optuna Hyperparameter Optimization Engine for Breast Cancer Detection Models
Provides automated hyperparameter tuning with medical-focused objectives
"""

import optuna
import torch
import torch.backends.mps
import numpy as np
import json
import logging
import gc
import platform
from typing import Dict, Any, Callable, Optional, List
from pathlib import Path
from datetime import datetime
import sqlite3
import psutil

try:
    # Try relative imports first (when used as module)
    from ..models.cnn_model import create_cnn_model
    from ..models.resnet_model import create_resnet_model  
    from ..models.densenet_model import create_densenet_model
    from ..models.faster_rcnn_model import create_faster_rcnn_model
    from ..models.vision_transformer import create_vit_model
    from ..training.trainer import BreastCancerTrainer
    from ..data_loaders.breast_cancer_dataloader import create_dataloaders
except ImportError:
    # Fall back to direct imports (when run from src directory)
    from models.cnn_model import create_cnn_model
    from models.resnet_model import create_resnet_model  
    from models.densenet_model import create_densenet_model
    from models.faster_rcnn_model import create_faster_rcnn_model
    from models.vision_transformer import create_vit_model
    from training.trainer import BreastCancerTrainer
    from data_loaders.breast_cancer_dataloader import create_dataloaders


class OptunaOptimizer:
    """
    Core optimization engine for hyperparameter tuning of medical imaging models
    Optimized for Apple M4 Pro with MPS backend and unified memory architecture
    """
    
    def __init__(
        self,
        model_type: str,
        train_loader,
        val_loader,
        base_config: Dict[str, Any],
        device: torch.device,
        study_name: Optional[str] = None,
        storage_path: Optional[str] = None
    ):
        """
        Initialize Optuna optimizer
        
        Args:
            model_type: Type of model ('cnn', 'resnet', 'vision_transformer')
            train_loader: Training data loader
            val_loader: Validation data loader
            base_config: Base configuration dictionary
            device: Training device
            study_name: Name for the optimization study
            storage_path: Path to store optimization results
        """
        self.model_type = model_type
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.base_config = base_config.copy()
        self.device = device
        
        # Study configuration
        self.study_name = study_name or f"{model_type}_optimization_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        self.storage_path = storage_path or f"optimization_studies/{self.study_name}.db"
        
        # Create storage directory
        Path(self.storage_path).parent.mkdir(parents=True, exist_ok=True)
        
        # Setup logging
        self.logger = logging.getLogger(f"OptunaOptimizer_{model_type}")
        
        # Optimization tracking
        self.current_trial = 0
        self.total_trials = 0
        self.best_value = 0.0
        self.best_params = {}
        self.optimization_status = "ready"
        
        # Determine optimization metric from config
        optuna_config = base_config.get('optuna', {})
        self.optimize_metric = optuna_config.get('optimize_metric', 'mcc')  # Default to MCC for medical
        
        # Validate and map optimization metric
        valid_metrics = ['mcc', 'recall', 'sensitivity', 'specificity', 'auc_roc', 'medical_composite', 'f1_score']
        if self.optimize_metric not in valid_metrics:
            self.logger.warning(f"Invalid optimize_metric '{self.optimize_metric}'. Using 'mcc' instead.")
            self.optimize_metric = 'mcc'
        
        # Map metric names for consistency
        if self.optimize_metric == 'recall':
            self.optimize_metric = 'sensitivity'  # Same metric, different name
            
        self.logger.info(f"Optuna optimization will maximize: {self.optimize_metric}")
        
        # Callbacks for real-time updates
        self.progress_callback: Optional[Callable] = None
        self.trial_callback: Optional[Callable] = None
        
        # M4 Pro specific optimizations
        self.is_m4_pro = self._detect_m4_pro()
        self.unified_memory_gb = self._get_unified_memory_size()
        self.max_batch_size = self._calculate_optimal_batch_size()
        
        # Memory management for trials
        self._setup_memory_management()
        
        if self.is_m4_pro:
            self.logger.info(f"M4 Pro detected with {self.unified_memory_gb:.1f}GB unified memory")
            self.logger.info(f"Optimized max batch size: {self.max_batch_size}")
        
    def set_progress_callback(self, callback: Callable):
        """Set callback function for real-time progress updates"""
        self.progress_callback = callback
        
    def set_trial_callback(self, callback: Callable):
        """Set callback function for trial completion updates"""
        self.trial_callback = callback
    
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
    
    def _get_unified_memory_size(self) -> float:
        """Get unified memory size in GB"""
        try:
            total_memory = psutil.virtual_memory().total
            return total_memory / (1024**3)  # Convert to GB
        except:
            return 32.0  # Conservative fallback
    
    def _calculate_optimal_batch_size(self) -> int:
        """Calculate optimal batch size for M4 Pro based on unified memory"""
        if not self.is_m4_pro:
            return 64  # Standard fallback
        
        # M4 Pro specific calculations
        # Account for model size, intermediate activations, and gradients
        if self.unified_memory_gb >= 36:  # 38GB M4 Pro
            if self.model_type == 'vision_transformer':
                return 128  # ViT can handle larger batches efficiently
            elif self.model_type == 'resnet':
                return 256  # ResNet is memory efficient
            elif self.model_type == 'densenet':
                return 192  # DenseNet moderate batch size (memory efficient)
            elif self.model_type == 'faster_rcnn':
                return 32   # Faster R-CNN smaller batches (complex model)
            else:  # CNN
                return 192  # Custom CNN moderate batch size
        else:
            # Fallback for other configurations
            if self.model_type == 'faster_rcnn':
                return 16  # Even smaller for complex model on limited memory
            else:
                return 96
    
    def _setup_memory_management(self):
        """Setup memory management strategies for M4 Pro"""
        if self.is_m4_pro and torch.backends.mps.is_available():
            # Enable optimized memory management
            torch.backends.mps.empty_cache = lambda: gc.collect()
    
    def _cleanup_trial_memory(self):
        """Cleanup memory between trials for M4 Pro"""
        gc.collect()
        if self.is_m4_pro and torch.backends.mps.is_available():
            try:
                torch.mps.empty_cache()
            except:
                pass  # Fallback silently
    
    def _suggest_parameters(self, trial: optuna.Trial) -> Dict[str, Any]:
        """
        Suggest hyperparameters based on model type
        
        Args:
            trial: Optuna trial object
            
        Returns:
            Dictionary of suggested parameters
        """
        params = {}
        
        if self.model_type == 'cnn':
            # M4 Pro optimized batch sizes
            batch_options = self._get_optimized_batch_sizes('cnn')
            params.update({
                'learning_rate': trial.suggest_float('learning_rate', 1e-5, 1e-2, log=True),
                'dropout': trial.suggest_float('dropout', 0.1, 0.7),
                'epochs': trial.suggest_int('epochs', 8, 20),  # Optimized range for faster trials
                'architecture': trial.suggest_categorical('architecture', ['custom', 'compact']),
                'batch_size': trial.suggest_categorical('batch_size', batch_options)
            })
            
        elif self.model_type == 'resnet':
            # M4 Pro optimized batch sizes for ResNet
            batch_options = self._get_optimized_batch_sizes('resnet')
            params.update({
                'learning_rate': trial.suggest_float('learning_rate', 1e-6, 1e-3, log=True),
                'epochs': trial.suggest_int('epochs', 10, 30),  # Optimized for M4 Pro training speed
                'architecture': trial.suggest_categorical('architecture', ['resnet18', 'resnet34', 'resnet50', 'lightweight']),
                'pretrained': trial.suggest_categorical('pretrained', [True, False]),
                'fine_tune_layers': trial.suggest_int('fine_tune_layers', -1, 10),
                'batch_size': trial.suggest_categorical('batch_size', batch_options)
            })
            
        elif self.model_type == 'densenet':
            # M4 Pro optimized batch sizes for DenseNet
            batch_options = self._get_optimized_batch_sizes('densenet')
            params.update({
                'learning_rate': trial.suggest_float('learning_rate', 1e-6, 1e-3, log=True),
                'epochs': trial.suggest_int('epochs', 10, 25),  # Optimized for M4 Pro
                'architecture': trial.suggest_categorical('architecture', ['densenet121', 'densenet169', 'densenet201', 'compact']),
                'pretrained': trial.suggest_categorical('pretrained', [True, False]),
                'growth_rate': trial.suggest_categorical('growth_rate', [16, 24, 32, 48]),
                'compression': trial.suggest_float('compression', 0.3, 0.7),
                'dropout_rate': trial.suggest_float('dropout_rate', 0.1, 0.5),
                'batch_size': trial.suggest_categorical('batch_size', batch_options)
            })
            
        elif self.model_type == 'faster_rcnn':
            # M4 Pro optimized batch sizes for Faster R-CNN (smaller due to complexity)
            batch_options = self._get_optimized_batch_sizes('faster_rcnn')
            params.update({
                'learning_rate': trial.suggest_float('learning_rate', 1e-7, 1e-3, log=True),
                'epochs': trial.suggest_int('epochs', 8, 20),  # Optimized for complex model
                'backbone': trial.suggest_categorical('backbone', ['resnet50', 'resnet101', 'compact']),
                'pretrained': trial.suggest_categorical('pretrained', [True, False]),
                'anchor_scales': trial.suggest_categorical('anchor_scales', [[4,8,16], [8,16,32], [16,32,64]]),
                'roi_pool_size': trial.suggest_categorical('roi_pool_size', [5, 7, 9]),
                'dropout_rate': trial.suggest_float('dropout_rate', 0.2, 0.6),
                'batch_size': trial.suggest_categorical('batch_size', batch_options)
            })
            
        elif self.model_type == 'vision_transformer':
            # M4 Pro optimized batch sizes for ViT
            batch_options = self._get_optimized_batch_sizes('vision_transformer')
            params.update({
                'learning_rate': trial.suggest_float('learning_rate', 1e-7, 1e-4, log=True),
                'epochs': trial.suggest_int('epochs', 8, 25),  # Optimized for M4 Pro
                'model_name': trial.suggest_categorical('model_name', ['vit_tiny_patch16_224', 'vit_small_patch16_224', 'compact']),
                'dropout_rate': trial.suggest_float('dropout_rate', 0.0, 0.3),
                'patch_size': trial.suggest_int('patch_size', 4, 8),
                'batch_size': trial.suggest_categorical('batch_size', batch_options)
            })
            
        return params
    
    def _get_optimized_batch_sizes(self, model_type: str) -> List[int]:
        """Get M4 Pro optimized batch sizes for each model type"""
        if not self.is_m4_pro:
            # Standard batch sizes for non-M4 Pro systems
            if model_type == 'vision_transformer':
                return [8, 16, 32]
            elif model_type == 'resnet':
                return [16, 32, 64]
            elif model_type == 'densenet':
                return [16, 32, 48]
            elif model_type == 'faster_rcnn':
                return [4, 8, 16]  # Smaller batches for complex model
            else:  # CNN
                return [16, 32, 64]
        
        # M4 Pro optimized batch sizes based on unified memory
        # Reduced maximum batch sizes to prevent memory issues during optimization
        if model_type == 'vision_transformer':
            # ViT benefits from larger batches on M4 Pro, but keep reasonable limits
            return [32, 64, 128]
        elif model_type == 'resnet':
            # ResNet can handle large batches efficiently, but cap at 256 for optimization stability
            if self.unified_memory_gb >= 36:  # 38GB M4 Pro
                return [64, 128, 192, 256]
            else:
                return [32, 64, 128]
        elif model_type == 'densenet':
            # DenseNet moderate batch sizes (memory efficient)
            if self.unified_memory_gb >= 36:  # 38GB M4 Pro
                return [48, 96, 128, 192]
            else:
                return [32, 64, 96]
        elif model_type == 'faster_rcnn':
            # Faster R-CNN smaller batch sizes (complex model)
            if self.unified_memory_gb >= 36:  # 38GB M4 Pro
                return [8, 16, 24, 32]
            else:
                return [4, 8, 16]
        else:  # CNN
            # Custom CNN moderate batch sizes
            return [48, 96, 128, 192]
    
    def _create_model_with_params(self, params: Dict[str, Any]):
        """Create model with suggested parameters"""
        try:
            self.logger.info(f"Creating {self.model_type} model with parameters: {params}")
            
            # Update config with suggested parameters
            config = self.base_config.copy()
            
            # Update model-specific config
            if self.model_type not in config.get('models', {}):
                config['models'] = config.get('models', {})
                config['models'][self.model_type] = {}
                
            config['models'][self.model_type].update(params)
            
            # Update data config if batch_size is suggested
            if 'batch_size' in params:
                config['data']['batch_size'] = params['batch_size']
                self.logger.debug(f"Updated batch_size to {params['batch_size']}")
            
            # Log the final config for this model
            self.logger.debug(f"Final config for {self.model_type}: {config['models'][self.model_type]}")
            
            # Create model with detailed error catching
            model = None
            if self.model_type == 'cnn':
                self.logger.debug("Creating CNN model")
                model = create_cnn_model(config)
            elif self.model_type == 'resnet':
                self.logger.debug("Creating ResNet model")
                # Additional validation for ResNet parameters
                resnet_config = config['models']['resnet']
                architecture = resnet_config.get('architecture', 'resnet18')
                pretrained = resnet_config.get('pretrained', True)
                fine_tune_layers = resnet_config.get('fine_tune_layers', -1)
                
                self.logger.debug(f"ResNet parameters - architecture: {architecture}, pretrained: {pretrained}, fine_tune_layers: {fine_tune_layers}")
                
                # Validate architecture
                valid_architectures = ['resnet18', 'resnet34', 'resnet50', 'resnet101', 'lightweight']
                if architecture not in valid_architectures:
                    raise ValueError(f"Invalid ResNet architecture '{architecture}'. Valid options: {valid_architectures}")
                
                # Validate fine_tune_layers
                if not isinstance(fine_tune_layers, int) or fine_tune_layers < -1 or fine_tune_layers > 10:
                    raise ValueError(f"Invalid fine_tune_layers '{fine_tune_layers}'. Must be integer between -1 and 10")
                
                model = create_resnet_model(config)
                self.logger.debug("ResNet model created successfully")
                
            elif self.model_type == 'densenet':
                self.logger.debug("Creating DenseNet model")
                model = create_densenet_model(config)
                
            elif self.model_type == 'faster_rcnn':
                self.logger.debug("Creating Faster R-CNN model")
                model = create_faster_rcnn_model(config)
                
            elif self.model_type == 'vision_transformer':
                self.logger.debug("Creating Vision Transformer model")
                model = create_vit_model(config)
            else:
                raise ValueError(f"Unknown model type: {self.model_type}")
            
            if model is None:
                raise RuntimeError(f"Model creation failed - returned None for {self.model_type}")
            
            # Move model to device
            model = model.to(self.device)
            self.logger.debug(f"Model moved to device: {self.device}")
            
            # Validate model can process expected input
            # Set model to eval mode to avoid BatchNorm issues with batch_size=1
            model.eval()
            test_input = torch.randn(1, 3, 50, 50).to(self.device)
            with torch.no_grad():
                test_output = model(test_input)
                if test_output.shape[1] != 2:
                    raise RuntimeError(f"Model output shape incorrect: {test_output.shape}, expected (1, 2)")
            
            self.logger.info(f"{self.model_type} model created and validated successfully")
            return model, config
            
        except Exception as e:
            self.logger.error(f"Failed to create {self.model_type} model with parameters {params}")
            self.logger.error(f"Error: {str(e)}")
            import traceback
            self.logger.error(f"Stack trace: {traceback.format_exc()}")
            raise
    
    def _objective(self, trial: optuna.Trial) -> float:
        """
        Objective function for optimization
        
        Args:
            trial: Optuna trial object
            
        Returns:
            Objective value to maximize (selected medical metric)
        """
        try:
            # Cleanup memory from previous trial
            self._cleanup_trial_memory()
            
            # Update trial counter
            self.current_trial = trial.number + 1
            
            # Suggest parameters
            params = self._suggest_parameters(trial)
            
            # Apply M4 Pro specific optimizations to parameters
            params = self._apply_m4_pro_optimizations(params)
            
            # Create model with suggested parameters
            model, config = self._create_model_with_params(params)
            
            # Update progress
            if self.progress_callback:
                self.progress_callback(
                    self.current_trial, 
                    self.total_trials, 
                    f"Trial {self.current_trial}: Testing {params}"
                )
            
            # Create trainer
            trainer = BreastCancerTrainer(
                model=model,
                train_loader=self.train_loader,
                val_loader=self.val_loader,
                config=config,
                device=self.device,
                save_dir=f"optimization_temp/{self.study_name}/trial_{trial.number}"
            )
            
            # Train model
            history = trainer.train()
            
            # Get best validation metrics based on selected optimization metric
            if history['val_metrics']:
                # Find best epoch based on selected optimization metric
                best_metrics = max(history['val_metrics'], key=lambda x: x.get(self.optimize_metric, 0.0))
                
                # Extract key metrics
                optimization_value = best_metrics.get(self.optimize_metric, 0.0)
                sensitivity = best_metrics['sensitivity']
                specificity = best_metrics['specificity']
                f1_score = best_metrics.get('f1_score', 0.0)
                auc_roc = best_metrics.get('auc_roc', 0.0)
                mcc = best_metrics.get('mcc', 0.0)
                accuracy = best_metrics.get('accuracy', 0.0)
                
                # Log all metrics for analysis
                trial.set_user_attr('sensitivity', sensitivity)
                trial.set_user_attr('specificity', specificity)
                trial.set_user_attr('f1_score', f1_score)
                trial.set_user_attr('accuracy', accuracy)
                trial.set_user_attr('auc_roc', auc_roc)
                trial.set_user_attr('mcc', mcc)
                trial.set_user_attr('optimization_metric', self.optimize_metric)
                trial.set_user_attr('optimization_value', optimization_value)
                trial.set_user_attr('epochs_completed', len(history['val_metrics']))
                
                # Update best tracking based on selected metric
                if optimization_value > self.best_value:
                    self.best_value = optimization_value
                    self.best_params = params.copy()
                
                # Trial callback with selected metric as primary value
                if self.trial_callback:
                    try:
                        # Try new callback signature with AUC-ROC and MCC
                        self.trial_callback(trial.number + 1, params, optimization_value, sensitivity, specificity, 
                                          accuracy, mcc, auc_roc)
                    except TypeError:
                        # Fallback to older signature
                        self.trial_callback(trial.number + 1, params, optimization_value, sensitivity, specificity, 
                                          accuracy, mcc)
                
                # Log trial results with selected metric highlighted
                self.logger.info(f"Trial {trial.number}: {self.optimize_metric.upper()}={optimization_value:.4f}, "
                               f"Sensitivity={sensitivity:.4f}, Specificity={specificity:.4f}, "
                               f"AUC-ROC={auc_roc:.4f}, MCC={mcc:.4f}")
                
                return optimization_value
            else:
                self.logger.warning(f"Trial {trial.number}: No validation metrics available")
                return 0.0
                
        except Exception as e:
            error_msg = f"Trial {trial.number} failed with error: {str(e)}"
            self.logger.error(error_msg)
            self.logger.error(f"Trial {trial.number} parameters that caused failure: {params}")
            
            # Log detailed stack trace for debugging
            import traceback
            stack_trace = traceback.format_exc()
            self.logger.error(f"Trial {trial.number} full stack trace:\n{stack_trace}")
            
            # Set user attributes for analysis
            trial.set_user_attr('error_type', type(e).__name__)
            trial.set_user_attr('error_message', str(e))
            trial.set_user_attr('failed_stage', 'unknown')
            
            # Try to determine which stage failed
            if 'model creation' in str(e).lower() or 'create' in str(e).lower():
                trial.set_user_attr('failed_stage', 'model_creation')
            elif 'train' in str(e).lower() or 'training' in str(e).lower():
                trial.set_user_attr('failed_stage', 'training')
            elif 'validation' in str(e).lower() or 'val' in str(e).lower():
                trial.set_user_attr('failed_stage', 'validation')
            elif 'memory' in str(e).lower() or 'cuda' in str(e).lower() or 'mps' in str(e).lower():
                trial.set_user_attr('failed_stage', 'memory_device')
            
            # Cleanup memory on failure
            self._cleanup_trial_memory()
            
            # Trial callback with error information (if callback supports error parameter)
            if self.trial_callback:
                try:
                    # Try to call with error parameter (including auc_roc=0.0)
                    self.trial_callback(trial.number + 1, params, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, error_msg)
                except TypeError:
                    try:
                        # Fallback to standard callback signature with accuracy, mcc, auc_roc
                        self.trial_callback(trial.number + 1, params, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
                    except TypeError:
                        # Final fallback to older signature
                        self.trial_callback(trial.number + 1, params, 0.0, 0.0, 0.0, 0.0, 0.0)
            
            # Update progress callback with error
            if self.progress_callback:
                self.progress_callback(
                    trial.number + 1, 
                    self.total_trials, 
                    f"Trial {trial.number + 1} FAILED: {str(e)[:100]}..."
                )
            
            return 0.0
        finally:
            # Always cleanup memory after trial
            self._cleanup_trial_memory()
    
    def _apply_m4_pro_optimizations(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Apply M4 Pro specific optimizations to trial parameters"""
        if not self.is_m4_pro:
            return params
        
        # Ensure batch size doesn't exceed optimal limits
        if 'batch_size' in params:
            params['batch_size'] = min(params['batch_size'], self.max_batch_size)
        
        # Adjust epochs for faster trials on M4 Pro
        if 'epochs' in params and self.unified_memory_gb >= 36:
            # M4 Pro can train faster, so slightly reduce epochs for exploration
            params['epochs'] = max(5, int(params['epochs'] * 0.8))
        
        return params
    
    def optimize(
        self, 
        n_trials: int = 50,
        timeout: Optional[int] = None,
        pruner: Optional[optuna.pruners.BasePruner] = None
    ) -> optuna.Study:
        """
        Run hyperparameter optimization
        
        Args:
            n_trials: Number of optimization trials
            timeout: Maximum optimization time in seconds
            pruner: Optuna pruner for early stopping
            
        Returns:
            Completed Optuna study
        """
        self.total_trials = n_trials
        self.optimization_status = "running"
        
        # Setup storage
        storage = f"sqlite:///{self.storage_path}"
        
        # Setup pruner
        if pruner is None:
            pruner = optuna.pruners.MedianPruner(
                n_startup_trials=5,
                n_warmup_steps=10,
                interval_steps=1
            )
        
        try:
            # Create or load study
            study = optuna.create_study(
                study_name=self.study_name,
                storage=storage,
                direction='maximize',  # Maximize selected medical metric
                pruner=pruner,
                load_if_exists=True
            )
            
            self.logger.info(f"Starting optimization for {self.model_type} with {n_trials} trials")
            
            # Run optimization
            study.optimize(
                self._objective,
                n_trials=n_trials,
                timeout=timeout,
                callbacks=[self._optimization_callback]
            )
            
            self.optimization_status = "completed"
            
            # Log results
            self.logger.info(f"Optimization completed. Best {self.optimize_metric.upper()}: {study.best_value:.4f}")
            self.logger.info(f"Best parameters: {study.best_params}")
            
            return study
            
        except Exception as e:
            self.optimization_status = "failed"
            self.logger.error(f"Optimization failed: {str(e)}")
            raise
    
    def _optimization_callback(self, study: optuna.Study, trial: optuna.Trial):
        """Callback called after each trial"""
        if self.progress_callback:
            progress = (trial.number + 1) / self.total_trials * 100
            self.progress_callback(
                trial.number + 1,
                self.total_trials,
                f"Trial {trial.number + 1} completed. Best F1: {study.best_value:.4f}"
            )
    
    def get_best_parameters(self) -> Dict[str, Any]:
        """Get the best parameters found during optimization"""
        return self.best_params.copy()
    
    def get_optimization_status(self) -> Dict[str, Any]:
        """Get current optimization status"""
        return {
            'status': self.optimization_status,
            'current_trial': self.current_trial,
            'total_trials': self.total_trials,
            'best_value': self.best_value,
            'best_params': self.best_params.copy()
        }
    
    def save_results(self, filepath: str):
        """Save optimization results to file"""
        results = {
            'study_name': self.study_name,
            'model_type': self.model_type,
            'best_value': self.best_value,
            'best_params': self.best_params,
            'optimization_status': self.optimization_status,
            'timestamp': datetime.now().isoformat()
        }
        
        with open(filepath, 'w') as f:
            json.dump(results, f, indent=2)
    
    @staticmethod
    def load_study(study_name: str, storage_path: str) -> optuna.Study:
        """Load existing optimization study"""
        storage = f"sqlite:///{storage_path}"
        return optuna.load_study(study_name=study_name, storage=storage)


class OptimizationPresets:
    """
    Predefined optimization presets for different scenarios
    """
    
    @staticmethod
    def get_fast_preset() -> Dict[str, Any]:
        """Fast optimization preset (fewer trials, smaller search space)"""
        return {
            'n_trials': 3,
            'timeout': 1800,  # 30 minutes
            'pruner': optuna.pruners.MedianPruner(n_startup_trials=1, n_warmup_steps=2)
        }
    
    @staticmethod
    def get_thorough_preset() -> Dict[str, Any]:
        """Thorough optimization preset (more trials, larger search space)"""
        return {
            'n_trials': 100,
            'timeout': 14400,  # 4 hours
            'pruner': optuna.pruners.MedianPruner(n_startup_trials=10, n_warmup_steps=15)
        }
    
    @staticmethod
    def get_medical_preset() -> Dict[str, Any]:
        """Medical-focused optimization preset (balanced sensitivity/specificity)"""
        return {
            'n_trials': 50,
            'timeout': 7200,  # 2 hours
            'pruner': optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=10)
        }