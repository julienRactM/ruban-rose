"""
M4 Pro GPU Optimization Utilities for Breast Cancer Detection Models

This module provides comprehensive optimizations specifically for Apple M4 Pro:
- MPS device detection and configuration
- Memory management for 38GB unified memory
- Mixed precision training optimization
- Thermal throttling awareness
- Batch size optimization for Apple Silicon
"""

import torch
import psutil
import time
import logging
from typing import Dict, Any, Optional, Tuple, List
from pathlib import Path
import subprocess
import json


class M4ProOptimizer:
    """
    Apple M4 Pro specific optimizations for ML training
    """
    
    def __init__(self, logger: Optional[logging.Logger] = None):
        self.logger = logger or logging.getLogger(__name__)
        self.device_info = self._get_device_info()
        self.memory_info = self._get_memory_info()
        
    def _get_device_info(self) -> Dict[str, Any]:
        """Get comprehensive M4 Pro device information"""
        info = {
            'mps_available': torch.backends.mps.is_available(),
            'mps_built': torch.backends.mps.is_built(),
            'device_name': 'Unknown',
            'memory_gb': 0,
            'is_m4_pro': False,
            'recommended_device': 'cpu'
        }
        
        try:
            # Get system info using system_profiler (macOS specific)
            result = subprocess.run(
                ['system_profiler', 'SPHardwareDataType', '-json'],
                capture_output=True, text=True, timeout=10
            )
            if result.returncode == 0:
                data = json.loads(result.stdout)
                hardware = data.get('SPHardwareDataType', [{}])[0]
                chip_name = hardware.get('chip_type', '')
                memory_str = hardware.get('physical_memory', '0 GB')
                
                info['device_name'] = chip_name
                info['is_m4_pro'] = 'M4 Pro' in chip_name or 'M4' in chip_name
                
                # Extract memory size
                if 'GB' in memory_str:
                    info['memory_gb'] = int(memory_str.split(' ')[0])
                
        except Exception as e:
            self.logger.warning(f"Could not get system info: {e}")
        
        # Determine recommended device
        if info['mps_available'] and info['is_m4_pro']:
            info['recommended_device'] = 'mps'
        elif info['mps_available']:
            info['recommended_device'] = 'mps'
        else:
            info['recommended_device'] = 'cpu'
            
        return info
    
    def _get_memory_info(self) -> Dict[str, Any]:
        """Get current memory usage information"""
        memory = psutil.virtual_memory()
        return {
            'total_gb': memory.total / (1024**3),
            'available_gb': memory.available / (1024**3),
            'used_gb': memory.used / (1024**3),
            'percent_used': memory.percent
        }
    
    def get_optimal_device(self) -> torch.device:
        """Get the optimal PyTorch device for M4 Pro"""
        if self.device_info['mps_available']:
            device = torch.device('mps')
            self.logger.info(f"Using MPS device on {self.device_info['device_name']}")
            
            # Enable MPS optimizations if available
            try:
                # Set MPS allocator settings for better memory management
                torch.mps.set_per_process_memory_fraction(0.8)  # Use 80% of available memory
                self.logger.info("Configured MPS memory fraction to 0.8")
            except Exception as e:
                self.logger.warning(f"Could not configure MPS memory settings: {e}")
                
            return device
        else:
            self.logger.warning("MPS not available, falling back to CPU")
            return torch.device('cpu')
    
    def get_optimal_batch_size(self, model_size: str = 'medium', image_size: Tuple[int, int] = (50, 50)) -> int:
        """
        Calculate optimal batch size for M4 Pro based on model size and available memory
        
        Args:
            model_size: 'small', 'medium', 'large'
            image_size: Input image dimensions
            
        Returns:
            Optimal batch size
        """
        if not self.device_info['mps_available']:
            return 16  # Conservative CPU batch size
        
        # M4 Pro specific optimizations
        available_memory_gb = self.memory_info['available_gb']
        
        # Base batch sizes for different model types on M4 Pro
        batch_size_map = {
            'small': {
                'base': 128,
                'memory_factor': 0.5  # Small models use less memory per sample
            },
            'medium': {
                'base': 64,
                'memory_factor': 1.0  # Reference size
            },
            'large': {
                'base': 32,
                'memory_factor': 2.0  # Large models use more memory per sample
            }
        }
        
        config = batch_size_map.get(model_size, batch_size_map['medium'])
        
        # Adjust based on available memory (M4 Pro has 36-128GB unified memory)
        if available_memory_gb > 30:  # High memory system
            memory_multiplier = 1.5
        elif available_memory_gb > 20:  # Medium memory system
            memory_multiplier = 1.2
        elif available_memory_gb > 10:  # Low memory system
            memory_multiplier = 1.0
        else:  # Very low memory
            memory_multiplier = 0.5
        
        # Adjust based on image size
        image_pixels = image_size[0] * image_size[1]
        size_factor = max(0.5, min(2.0, 2500 / image_pixels))  # 50x50 = 2500 pixels baseline
        
        optimal_batch_size = int(config['base'] * memory_multiplier * size_factor)
        
        # Ensure batch size is within reasonable bounds
        optimal_batch_size = max(4, min(512, optimal_batch_size))
        
        # Prefer powers of 2 for better GPU utilization
        powers_of_2 = [4, 8, 16, 32, 64, 128, 256, 512]
        optimal_batch_size = min(powers_of_2, key=lambda x: abs(x - optimal_batch_size))
        
        self.logger.info(f"Optimal batch size for {model_size} model: {optimal_batch_size}")
        return optimal_batch_size
    
    def configure_mixed_precision(self) -> Dict[str, Any]:
        """
        Configure mixed precision training for M4 Pro
        
        Returns:
            Mixed precision configuration
        """
        config = {
            'enabled': False,
            'dtype': torch.float32,
            'scaler_enabled': False,
            'reason': 'MPS not available'
        }
        
        if self.device_info['mps_available']:
            # M4 Pro supports float16 operations which can significantly speed up training
            config.update({
                'enabled': True,
                'dtype': torch.float16,
                'scaler_enabled': True,  # Use GradScaler to prevent underflow
                'reason': 'M4 Pro MPS supports mixed precision'
            })
            
            self.logger.info("Enabled mixed precision training with float16 for M4 Pro")
        
        return config
    
    def get_optimal_num_workers(self) -> int:
        """
        Get optimal number of workers for data loading on M4 Pro
        
        Apple Silicon works better with threading vs multiprocessing for certain operations
        """
        if self.device_info['mps_available']:
            # M4 Pro has efficient cores, but data loading benefits from threading
            # due to unified memory architecture
            cpu_count = psutil.cpu_count(logical=False)  # Physical cores
            
            # For M4 Pro (12 cores: 8 performance + 4 efficiency)
            # Use fewer workers to avoid overhead and memory contention
            if cpu_count >= 12:  # M4 Pro
                optimal_workers = 4
            elif cpu_count >= 8:  # M4 or similar
                optimal_workers = 3
            else:
                optimal_workers = 2
                
            self.logger.info(f"Optimal data loader workers for M4 Pro: {optimal_workers}")
            return optimal_workers
        else:
            return min(4, psutil.cpu_count(logical=False))
    
    def optimize_dataloader_config(self, base_config: Dict[str, Any]) -> Dict[str, Any]:
        """
        Optimize data loader configuration for M4 Pro
        
        Args:
            base_config: Base data configuration
            
        Returns:
            Optimized configuration
        """
        optimized_config = base_config.copy()
        
        # Set optimal workers
        optimized_config['num_workers'] = self.get_optimal_num_workers()
        
        # Disable pin_memory for MPS (can cause issues)
        if self.device_info['mps_available']:
            optimized_config['pin_memory'] = False
            optimized_config['persistent_workers'] = True  # Reuse workers for efficiency
        
        # Optimize batch size if not already set optimally
        current_batch_size = optimized_config.get('batch_size', 32)
        optimal_batch_size = self.get_optimal_batch_size('medium')
        
        if abs(current_batch_size - optimal_batch_size) > 16:
            optimized_config['batch_size'] = optimal_batch_size
            self.logger.info(f"Adjusted batch size from {current_batch_size} to {optimal_batch_size}")
        
        return optimized_config
    
    def get_thermal_info(self) -> Dict[str, Any]:
        """
        Get thermal information (basic CPU temperature monitoring)
        M4 Pro has good thermal management, but monitoring helps with optimization
        """
        thermal_info = {
            'cpu_temp': None,
            'thermal_state': 'unknown',
            'throttling_detected': False
        }
        
        try:
            # Try to get thermal info on macOS
            result = subprocess.run(
                ['sudo', 'powermetrics', '--samplers', 'smc', '-n', '1', '--show-initial-usage'],
                capture_output=True, text=True, timeout=10
            )
            
            if result.returncode == 0:
                # Parse temperature from powermetrics output
                for line in result.stdout.split('\n'):
                    if 'CPU die temperature' in line:
                        temp_str = line.split(':')[-1].strip()
                        if '°C' in temp_str:
                            thermal_info['cpu_temp'] = float(temp_str.split('°C')[0])
                            
                            # Assess thermal state
                            if thermal_info['cpu_temp'] > 85:
                                thermal_info['thermal_state'] = 'hot'
                                thermal_info['throttling_detected'] = True
                            elif thermal_info['cpu_temp'] > 70:
                                thermal_info['thermal_state'] = 'warm'
                            else:
                                thermal_info['thermal_state'] = 'normal'
                        break
                        
        except Exception as e:
            self.logger.debug(f"Could not get thermal info: {e}")
        
        return thermal_info
    
    def optimize_training_config(self, base_config: Dict[str, Any]) -> Dict[str, Any]:
        """
        Optimize training configuration for M4 Pro
        
        Args:
            base_config: Base training configuration
            
        Returns:
            M4 Pro optimized configuration
        """
        optimized_config = base_config.copy()
        
        # Set optimal device
        optimized_config['device'] = 'mps' if self.device_info['mps_available'] else 'cpu'
        
        # Configure mixed precision
        mixed_precision_config = self.configure_mixed_precision()
        optimized_config['mixed_precision'] = mixed_precision_config['enabled']
        optimized_config['precision_dtype'] = mixed_precision_config['dtype']
        optimized_config['use_scaler'] = mixed_precision_config['scaler_enabled']
        
        # Optimize gradient clipping for M4 Pro
        optimized_config['gradient_clipping'] = 1.0  # Conservative clipping for stability
        
        # Memory management
        optimized_config['empty_cache_frequency'] = 10  # Clear cache every 10 batches
        
        # Checkpoint frequency (save memory by not keeping too many checkpoints)
        optimized_config['checkpoint_frequency'] = 5  # Every 5 epochs
        
        self.logger.info("Applied M4 Pro specific training optimizations")
        return optimized_config
    
    def optimize_optuna_config(self, base_trials: int = 50) -> Dict[str, Any]:
        """
        Optimize Optuna hyperparameter search for M4 Pro
        
        Args:
            base_trials: Base number of trials
            
        Returns:
            Optimized Optuna configuration
        """
        config = {
            'n_trials': base_trials,
            'memory_management': True,
            'early_stopping': True,
            'parallel_trials': 1,  # M4 Pro: Sequential trials to avoid memory issues
            'pruning_enabled': True,
            'checkpoint_interval': 5  # Save every 5 trials
        }
        
        # Adjust trials based on available memory and thermal state
        thermal_info = self.get_thermal_info()
        memory_gb = self.memory_info['available_gb']
        
        if memory_gb > 30:  # High memory
            config['n_trials'] = min(base_trials * 2, 100)
        elif memory_gb < 10:  # Low memory
            config['n_trials'] = max(base_trials // 2, 10)
        
        # Reduce trials if thermal throttling detected
        if thermal_info.get('throttling_detected', False):
            config['n_trials'] = max(config['n_trials'] // 2, 10)
            config['cooling_delay'] = 30  # 30 second delay between trials
            self.logger.warning("Thermal throttling detected, reducing optimization intensity")
        
        return config
    
    def get_optimal_search_spaces(self, model_type: str) -> Dict[str, Any]:
        """
        Get M4 Pro optimized hyperparameter search spaces
        
        Args:
            model_type: Type of model ('cnn', 'resnet', 'vision_transformer')
            
        Returns:
            Optimized search spaces
        """
        base_spaces = {
            'cnn': {
                'learning_rate': (1e-5, 1e-2),
                'batch_size': [16, 32, 64, 128],  # M4 Pro can handle larger batches
                'dropout': (0.1, 0.7),
                'epochs': (10, 30),  # Shorter epochs for faster iteration
            },
            'resnet': {
                'learning_rate': (1e-6, 1e-3),
                'batch_size': [16, 32, 64],  # ResNet uses more memory
                'epochs': (15, 45),
                'architecture': ['resnet18', 'resnet34'],  # Focus on smaller ResNets for speed
            },
            'vision_transformer': {
                'learning_rate': (1e-7, 1e-4),
                'batch_size': [8, 16, 32],  # ViT uses most memory
                'epochs': (10, 30),
                'model_name': ['vit_tiny_patch16_224', 'vit_small_patch16_224'],  # Smaller models
                'dropout_rate': (0.0, 0.3),
            }
        }
        
        return base_spaces.get(model_type, base_spaces['cnn'])
    
    def monitor_resource_usage(self) -> Dict[str, Any]:
        """
        Monitor current resource usage during training/optimization
        
        Returns:
            Resource usage information
        """
        memory = psutil.virtual_memory()
        cpu_percent = psutil.cpu_percent(interval=1)
        
        usage_info = {
            'memory_used_gb': memory.used / (1024**3),
            'memory_available_gb': memory.available / (1024**3),
            'memory_percent': memory.percent,
            'cpu_percent': cpu_percent,
            'timestamp': time.time()
        }
        
        # Add thermal info if available
        thermal_info = self.get_thermal_info()
        usage_info.update(thermal_info)
        
        return usage_info
    
    def should_throttle_training(self) -> Tuple[bool, str]:
        """
        Determine if training should be throttled based on system resources
        
        Returns:
            (should_throttle, reason)
        """
        memory_info = self._get_memory_info()
        thermal_info = self.get_thermal_info()
        
        # Check memory pressure
        if memory_info['percent_used'] > 90:
            return True, "High memory usage (>90%)"
        
        # Check thermal state
        if thermal_info.get('throttling_detected', False):
            return True, "Thermal throttling detected"
        
        # Check CPU usage (if consistently high, might indicate system stress)
        cpu_percent = psutil.cpu_percent(interval=1)
        if cpu_percent > 95:
            return True, "High CPU usage (>95%)"
        
        return False, "System resources normal"
    
    def cleanup_memory(self):
        """
        Clean up GPU/system memory for M4 Pro optimization
        """
        if self.device_info['mps_available']:
            try:
                # Clear MPS cache
                torch.mps.empty_cache()
                self.logger.debug("Cleared MPS cache")
            except Exception as e:
                self.logger.warning(f"Could not clear MPS cache: {e}")
        
        # Force garbage collection
        import gc
        gc.collect()
        
    def get_optimization_summary(self) -> str:
        """
        Get a summary of M4 Pro optimizations applied
        
        Returns:
            Summary string
        """
        device_name = self.device_info.get('device_name', 'Unknown')
        memory_gb = self.device_info.get('memory_gb', 0)
        mps_available = self.device_info.get('mps_available', False)
        
        summary = f"""
M4 Pro Optimization Summary:
============================
Device: {device_name}
Memory: {memory_gb} GB unified memory
MPS Available: {mps_available}
Recommended Device: {self.device_info['recommended_device']}

Optimizations Applied:
- Optimal batch sizes for unified memory architecture
- Mixed precision training ({torch.float16 if mps_available else 'disabled'})
- Thermal-aware optimization scheduling
- Memory-efficient data loading
- MPS-specific memory management

Resource Status:
- Available Memory: {self.memory_info['available_gb']:.1f} GB
- Memory Usage: {self.memory_info['percent_used']:.1f}%
"""
        
        return summary


def create_m4_pro_optimizer(logger: Optional[logging.Logger] = None) -> M4ProOptimizer:
    """
    Factory function to create M4ProOptimizer instance
    
    Args:
        logger: Optional logger instance
        
    Returns:
        M4ProOptimizer instance
    """
    return M4ProOptimizer(logger)


def test_m4_pro_optimizer():
    """Test the M4 Pro optimizer"""
    print("Testing M4 Pro Optimizer...")
    
    optimizer = create_m4_pro_optimizer()
    
    print(optimizer.get_optimization_summary())
    
    # Test optimal configurations
    print("\nOptimal Configurations:")
    print(f"Device: {optimizer.get_optimal_device()}")
    print(f"Batch size (CNN): {optimizer.get_optimal_batch_size('medium')}")
    print(f"Batch size (ViT): {optimizer.get_optimal_batch_size('large')}")
    print(f"Num workers: {optimizer.get_optimal_num_workers()}")
    
    # Test mixed precision
    mp_config = optimizer.configure_mixed_precision()
    print(f"Mixed precision: {mp_config}")
    
    # Test resource monitoring
    usage = optimizer.monitor_resource_usage()
    print(f"Resource usage: {usage}")
    
    # Test throttling check
    should_throttle, reason = optimizer.should_throttle_training()
    print(f"Should throttle: {should_throttle} ({reason})")
    
    print("✅ M4 Pro optimizer test completed!")


if __name__ == "__main__":
    test_m4_pro_optimizer()