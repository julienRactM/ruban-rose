"""
Parameter Search Spaces for Hyperparameter Optimization
Defines search spaces optimized for medical imaging tasks
"""

import optuna
from typing import Dict, Any, List, Tuple
from dataclasses import dataclass


@dataclass
class ParameterRange:
    """Define parameter range with bounds and suggestions"""
    min_val: float
    max_val: float
    log: bool = False
    step: float = None
    categories: List[Any] = None


class MedicalParameterSpaces:
    """
    Parameter search spaces optimized for medical imaging classification
    Considers the unique requirements of breast cancer detection
    """
    
    @staticmethod
    def get_cnn_search_space() -> Dict[str, ParameterRange]:
        """
        CNN parameter search space optimized for 50x50 medical images
        """
        return {
            # Learning rate: Conservative range for medical data
            'learning_rate': ParameterRange(
                min_val=1e-5, 
                max_val=1e-2, 
                log=True
            ),
            
            # Dropout: Higher values for medical regularization
            'dropout': ParameterRange(
                min_val=0.1, 
                max_val=0.7
            ),
            
            # Epochs: Sufficient for convergence but not excessive
            'epochs': ParameterRange(
                min_val=20, 
                max_val=100,
                step=5
            ),
            
            # Architecture variants
            'architecture': ParameterRange(
                categories=['custom', 'compact']
            ),
            
            # Batch size: Memory-efficient options
            'batch_size': ParameterRange(
                categories=[16, 32, 64]
            ),
            
            # CNN-specific: Number of filters in first layer
            'initial_filters': ParameterRange(
                min_val=16,
                max_val=64,
                step=16
            ),
            
            # CNN-specific: Filter multiplier
            'filter_multiplier': ParameterRange(
                min_val=1.5,
                max_val=3.0,
                step=0.5
            )
        }
    
    @staticmethod
    def get_resnet_search_space() -> Dict[str, ParameterRange]:
        """
        ResNet parameter search space for transfer learning
        """
        return {
            # Learning rate: Lower for pre-trained models
            'learning_rate': ParameterRange(
                min_val=1e-6, 
                max_val=1e-3, 
                log=True
            ),
            
            # Epochs: Fewer needed for transfer learning
            'epochs': ParameterRange(
                min_val=15, 
                max_val=60,
                step=5
            ),
            
            # Architecture variants
            'architecture': ParameterRange(
                categories=['resnet18', 'resnet34', 'resnet50', 'lightweight']
            ),
            
            # Pre-trained weights
            'pretrained': ParameterRange(
                categories=[True, False]
            ),
            
            # Fine-tuning strategy
            'fine_tune_layers': ParameterRange(
                min_val=-1,  # -1 means all layers
                max_val=10,
                step=1
            ),
            
            # Batch size: Larger models need smaller batches
            'batch_size': ParameterRange(
                categories=[16, 32, 64]
            ),
            
            # Weight decay for regularization
            'weight_decay': ParameterRange(
                min_val=1e-6,
                max_val=1e-3,
                log=True
            )
        }
    
    @staticmethod
    def get_vit_search_space() -> Dict[str, ParameterRange]:
        """
        Vision Transformer parameter search space
        """
        return {
            # Learning rate: Much lower for transformers
            'learning_rate': ParameterRange(
                min_val=1e-7, 
                max_val=1e-4, 
                log=True
            ),
            
            # Epochs: Transformers often need fewer epochs
            'epochs': ParameterRange(
                min_val=10, 
                max_val=40,
                step=5
            ),
            
            # Model variants
            'model_name': ParameterRange(
                categories=['vit_tiny_patch16_224', 'vit_small_patch16_224', 'compact']
            ),
            
            # Dropout rate: Lower for transformers
            'dropout_rate': ParameterRange(
                min_val=0.0, 
                max_val=0.3
            ),
            
            # Patch size for compact ViT (50x50 images)
            'patch_size': ParameterRange(
                min_val=4, 
                max_val=8,
                step=1
            ),
            
            # Batch size: Smaller for memory efficiency
            'batch_size': ParameterRange(
                categories=[8, 16, 32]
            ),
            
            # Attention dropout
            'attention_dropout': ParameterRange(
                min_val=0.0,
                max_val=0.2
            ),
            
            # Layer dropout (stochastic depth)
            'layer_dropout': ParameterRange(
                min_val=0.0,
                max_val=0.1
            )
        }
    
    @staticmethod
    def suggest_parameter(trial: optuna.Trial, param_name: str, param_range: ParameterRange) -> Any:
        """
        Suggest parameter value based on parameter range definition
        
        Args:
            trial: Optuna trial
            param_name: Name of parameter
            param_range: Parameter range definition
            
        Returns:
            Suggested parameter value
        """
        if param_range.categories is not None:
            return trial.suggest_categorical(param_name, param_range.categories)
        elif param_range.step is not None:
            if isinstance(param_range.step, int):
                return trial.suggest_int(
                    param_name, 
                    int(param_range.min_val), 
                    int(param_range.max_val),
                    step=int(param_range.step)
                )
            else:
                return trial.suggest_float(
                    param_name,
                    param_range.min_val,
                    param_range.max_val,
                    step=param_range.step,
                    log=param_range.log
                )
        else:
            return trial.suggest_float(
                param_name,
                param_range.min_val,
                param_range.max_val,
                log=param_range.log
            )


class OptimizationStrategies:
    """
    Different optimization strategies for various scenarios
    """
    
    @staticmethod
    def get_sensitivity_focused_objective() -> Dict[str, Any]:
        """
        Objective function prioritizing sensitivity (cancer detection)
        Critical for medical applications to minimize false negatives
        """
        return {
            'primary_metric': 'sensitivity',
            'secondary_metric': 'f1_score',
            'weights': {'sensitivity': 0.6, 'f1_score': 0.4},
            'min_sensitivity_threshold': 0.85  # Minimum acceptable sensitivity
        }
    
    @staticmethod
    def get_balanced_objective() -> Dict[str, Any]:
        """
        Balanced objective for sensitivity and specificity
        """
        return {
            'primary_metric': 'f1_score',
            'secondary_metric': 'balanced_accuracy',
            'weights': {'f1_score': 0.7, 'balanced_accuracy': 0.3},
            'min_sensitivity_threshold': 0.75
        }
    
    @staticmethod
    def get_specificity_focused_objective() -> Dict[str, Any]:
        """
        Objective function prioritizing specificity (reducing false positives)
        """
        return {
            'primary_metric': 'specificity',
            'secondary_metric': 'f1_score',
            'weights': {'specificity': 0.6, 'f1_score': 0.4},
            'min_specificity_threshold': 0.85
        }


class MedicalConstraints:
    """
    Medical-specific constraints for hyperparameter optimization
    """
    
    @staticmethod
    def validate_medical_parameters(params: Dict[str, Any], model_type: str) -> bool:
        """
        Validate parameters meet medical imaging requirements
        
        Args:
            params: Suggested parameters
            model_type: Type of model
            
        Returns:
            True if parameters are valid for medical use
        """
        # Minimum epochs for reliable medical model training
        if params.get('epochs', 0) < 15:
            return False
        
        # Learning rate constraints for medical stability
        lr = params.get('learning_rate', 0)
        if model_type == 'cnn' and (lr < 1e-5 or lr > 1e-2):
            return False
        elif model_type in ['resnet', 'vision_transformer'] and lr > 1e-3:
            return False
        
        # Dropout constraints to prevent underfitting on medical data
        dropout = params.get('dropout', 0)
        if dropout > 0.8:  # Too much dropout can hurt medical performance
            return False
        
        return True
    
    @staticmethod
    def get_medical_pruning_strategy() -> optuna.pruners.BasePruner:
        """
        Pruning strategy optimized for medical model training
        More conservative to ensure model convergence
        """
        return optuna.pruners.MedianPruner(
            n_startup_trials=8,  # More startup trials for stability
            n_warmup_steps=15,   # More warmup for medical convergence
            interval_steps=1,
            n_min_trials=5       # Minimum trials before pruning
        )


class DatasetAdaptiveSpaces:
    """
    Adaptive parameter spaces based on dataset characteristics
    """
    
    @staticmethod
    def adapt_for_dataset_size(base_space: Dict[str, ParameterRange], dataset_size: int) -> Dict[str, ParameterRange]:
        """
        Adapt parameter space based on dataset size
        
        Args:
            base_space: Base parameter space
            dataset_size: Number of training samples
            
        Returns:
            Adapted parameter space
        """
        adapted_space = base_space.copy()
        
        # Smaller datasets need more regularization
        if dataset_size < 1000:
            # Increase dropout range for small datasets
            if 'dropout' in adapted_space:
                adapted_space['dropout'] = ParameterRange(
                    min_val=0.3, 
                    max_val=0.8
                )
            
            # Reduce batch size options for small datasets
            if 'batch_size' in adapted_space:
                adapted_space['batch_size'] = ParameterRange(
                    categories=[8, 16, 32]
                )
        
        # Larger datasets can handle lower regularization
        elif dataset_size > 10000:
            if 'dropout' in adapted_space:
                adapted_space['dropout'] = ParameterRange(
                    min_val=0.0, 
                    max_val=0.5
                )
            
            # Larger batch sizes for big datasets
            if 'batch_size' in adapted_space:
                adapted_space['batch_size'] = ParameterRange(
                    categories=[32, 64, 128]
                )
        
        return adapted_space
    
    @staticmethod
    def adapt_for_class_imbalance(base_space: Dict[str, ParameterRange], imbalance_ratio: float) -> Dict[str, ParameterRange]:
        """
        Adapt parameter space for class imbalanced datasets
        
        Args:
            base_space: Base parameter space
            imbalance_ratio: Ratio of minority to majority class
            
        Returns:
            Adapted parameter space
        """
        adapted_space = base_space.copy()
        
        # Highly imbalanced datasets (< 0.3 ratio)
        if imbalance_ratio < 0.3:
            # More conservative learning rates
            if 'learning_rate' in adapted_space:
                lr_range = adapted_space['learning_rate']
                adapted_space['learning_rate'] = ParameterRange(
                    min_val=lr_range.min_val,
                    max_val=lr_range.max_val * 0.5,  # Reduce max LR
                    log=lr_range.log
                )
            
            # More epochs for imbalanced learning
            if 'epochs' in adapted_space:
                epochs_range = adapted_space['epochs']
                adapted_space['epochs'] = ParameterRange(
                    min_val=max(epochs_range.min_val, 30),  # Minimum 30 epochs
                    max_val=epochs_range.max_val + 20,      # Add 20 more epochs
                    step=epochs_range.step
                )
        
        return adapted_space