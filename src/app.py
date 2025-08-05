"""
Web Application for Breast Cancer Detection Model Training
Provides interface to select and train different models
"""

from flask import Flask, render_template, request, jsonify, redirect, url_for
from flask_cors import CORS
import yaml
import json
import os
import threading
import time
from pathlib import Path
import torch
import numpy as np
from datetime import datetime

# Import model factories
from models.cnn_model import create_cnn_model
from models.resnet_model import create_resnet_model
from models.densenet_model import create_densenet_model
from models.faster_rcnn_model import create_faster_rcnn_model
from models.vision_transformer import create_vit_model
from data_loaders.breast_cancer_dataloader import create_dataloaders
from training.trainer import BreastCancerTrainer

# Import optimization components
from optimization.optuna_optimizer import OptunaOptimizer, OptimizationPresets
from optimization.tracking import OptimizationTracker, OptimizationResult, get_global_tracker

app = Flask(__name__, template_folder='src/templates')
CORS(app)

# Global variables for training status
training_status = {
    'is_training': False,
    'current_model': None,
    'epoch': 0,
    'total_epochs': 0,
    'metrics': {},
    'logs': [],
    'start_time': None,
    'model_results': {}
}

# Global variables for optimization status
optimization_status = {
    'is_optimizing': False,
    'current_model': None,
    'current_trial': 0,
    'total_trials': 0,
    'best_medical_score': 0.0,
    'best_params': {},
    'optimization_mode': None,
    'logs': [],
    'start_time': None,
    'study_name': None,
    'trial_results': []
}

# Global optimization tracker
optimization_tracker = get_global_tracker()

# Load configuration
def load_config():
    """Load configuration from YAML file"""
    try:
        with open('config.yaml', 'r') as f:
            return yaml.safe_load(f)
    except FileNotFoundError:
        return {}

def sanitize_metrics_for_json(metrics):
    """Convert numpy arrays and types to JSON-serializable format"""
    if not isinstance(metrics, dict):
        return metrics
        
    sanitized = {}
    for key, value in metrics.items():
        if isinstance(value, np.ndarray):
            sanitized[key] = value.tolist()
        elif isinstance(value, (np.float32, np.float64, np.int32, np.int64)):
            sanitized[key] = float(value)
        else:
            sanitized[key] = value
    return sanitized

def save_config(config):
    """Save configuration to YAML file"""
    with open('config.yaml', 'w') as f:
        yaml.dump(config, f, default_flow_style=False, indent=2)

@app.route('/')
def index():
    """Main page with model selection and training interface"""
    config = load_config()
    return render_template('index.html', 
                         config=config, 
                         training_status=training_status)

@app.route('/api/config', methods=['GET', 'POST'])
def api_config():
    """API endpoint for configuration management"""
    if request.method == 'GET':
        config = load_config()
        return jsonify(config)
    
    elif request.method == 'POST':
        try:
            new_config = request.json
            save_config(new_config)
            return jsonify({'success': True, 'message': 'Configuration saved successfully'})
        except Exception as e:
            return jsonify({'success': False, 'error': str(e)}), 400

@app.route('/api/start_training', methods=['POST'])
def start_training():
    """Start training for selected models"""
    if training_status['is_training']:
        return jsonify({'success': False, 'error': 'Training already in progress'}), 400
    
    try:
        selected_models = request.json.get('models', [])
        model_parameters = request.json.get('parameters', {})
        data_config = request.json.get('data_config', {})
        
        if not selected_models:
            return jsonify({'success': False, 'error': 'No models selected'}), 400
        
        # Start training in background thread
        training_thread = threading.Thread(
            target=train_models_background,
            args=(selected_models, model_parameters, data_config)
        )
        training_thread.daemon = True
        training_thread.start()
        
        return jsonify({'success': True, 'message': 'Training started'})
    
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/api/training_status')
def get_training_status():
    """Get current training status"""
    return jsonify(training_status)

@app.route('/api/stop_training', methods=['POST'])
def stop_training():
    """Stop current training"""
    # Note: This is a simple implementation
    # In production, you'd want more sophisticated training control
    training_status['is_training'] = False
    training_status['current_model'] = None
    return jsonify({'success': True, 'message': 'Training stopped'})

@app.route('/api/model_results')
def get_model_results():
    """Get training results for all models"""
    return jsonify(training_status['model_results'])

@app.route('/api/start_optimization', methods=['POST'])
def start_optimization():
    """Start hyperparameter optimization for selected model"""
    if training_status['is_training'] or optimization_status['is_optimizing']:
        return jsonify({'success': False, 'error': 'Training or optimization already in progress'}), 400
    
    try:
        data = request.json
        model_type = data.get('model_type')
        optimization_mode = data.get('optimization_mode', 'fast')
        data_config = data.get('data_config', {})
        
        if not model_type:
            return jsonify({'success': False, 'error': 'No model selected for optimization'}), 400
        
        # Start optimization in background thread
        optimization_thread = threading.Thread(
            target=run_optimization_background,
            args=(model_type, optimization_mode, data_config)
        )
        optimization_thread.daemon = True
        optimization_thread.start()
        
        return jsonify({'success': True, 'message': 'Optimization started'})
    
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/api/optimization_status')
def get_optimization_status():
    """Get current optimization status"""
    return jsonify(optimization_status)

@app.route('/api/stop_optimization', methods=['POST'])
def stop_optimization():
    """Stop current optimization"""
    optimization_status['is_optimizing'] = False
    optimization_status['current_model'] = None
    return jsonify({'success': True, 'message': 'Optimization stopped'})

@app.route('/api/optimization_results/<study_name>')
def get_optimization_results(study_name):
    """Get detailed optimization results for a study"""
    try:
        study_data = optimization_tracker.get_study_progress(study_name)
        if study_data:
            return jsonify(study_data)
        else:
            return jsonify({'error': 'Study not found'}), 404
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/dataset_info')
def get_dataset_info():
    """Get dataset information"""
    try:
        config = load_config()
        data_path = config.get('data', {}).get('data_path', 'data/BHI')
        
        if not os.path.exists(data_path):
            return jsonify({'error': 'Dataset not found', 'path': data_path})
        
        # Quick dataset scan
        dataset_info = scan_dataset(data_path)
        return jsonify(dataset_info)
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

def scan_dataset(data_path):
    """Scan dataset for basic information"""
    info = {
        'total_patients': 0,
        'total_images': 0,
        'class_distribution': {'0': 0, '1': 0},
        'sample_patients': []
    }
    
    data_dir = Path(data_path)
    
    for patient_folder in data_dir.iterdir():
        if patient_folder.is_dir() and patient_folder.name != "IDC_regular_ps50_idx5":
            info['total_patients'] += 1
            
            if len(info['sample_patients']) < 5:
                info['sample_patients'].append(patient_folder.name)
            
            for class_folder in patient_folder.iterdir():
                if class_folder.is_dir() and class_folder.name in ['0', '1']:
                    class_images = len([f for f in class_folder.iterdir() if f.suffix.lower() == '.png'])
                    info['class_distribution'][class_folder.name] += class_images
                    info['total_images'] += class_images
    
    return info

def run_optimization_background(model_type, optimization_mode, data_config=None):
    """Background optimization function"""
    global optimization_status
    
    try:
        # Initialize optimization status
        study_name = f"{model_type}_optimization_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        optimization_status.update({
            'is_optimizing': True,
            'current_model': model_type,
            'start_time': datetime.now().isoformat(),
            'logs': [],
            'study_name': study_name,
            'optimization_mode': optimization_mode,
            'current_trial': 0,
            'total_trials': 0,
            'best_medical_score': 0.0,
            'best_params': {},
            'trial_results': []
        })
        
        config = load_config()
        
        # Apply data configuration
        if data_config:
            if 'data_percentage' in data_config:
                config['data']['data_percentage'] = data_config['data_percentage']
                add_optimization_log(f"Using {data_config['data_percentage']:.1%} of dataset")
            if 'batch_size' in data_config:
                config['data']['batch_size'] = data_config['batch_size']
                add_optimization_log(f"Batch size: {data_config['batch_size']}")
            if 'class_balance_enabled' in data_config:
                config['class_balance']['enabled'] = data_config['class_balance_enabled']
                if 'class_balance_ratio' in data_config:
                    config['class_balance']['target_ratio'] = data_config['class_balance_ratio']
        
        # Device selection with M4 Pro optimizations
        training_config = config.get('training', {})
        device_config = training_config.get('device', 'auto')
        
        if device_config == 'mps' and torch.backends.mps.is_available():
            device = torch.device('mps')
            # Apply M4 Pro specific optimizations
            _apply_m4_pro_optimizations(config)
        elif device_config == 'cuda' and torch.cuda.is_available():
            device = torch.device('cuda')
        elif device_config == 'cpu':
            device = torch.device('cpu')
        else:  # auto
            if torch.backends.mps.is_available():
                device = torch.device('mps')
                _apply_m4_pro_optimizations(config)
            elif torch.cuda.is_available():
                device = torch.device('cuda')
            else:
                device = torch.device('cpu')
        
        add_optimization_log(f"Starting optimization on device: {device}")
        add_optimization_log(f"Model: {model_type}, Mode: {optimization_mode}")
        
        # Create data loaders
        add_optimization_log("Loading dataset...")
        train_loader, val_loader = create_dataloaders(config)
        add_optimization_log(f"Dataset loaded - Train: {len(train_loader.dataset)}, Val: {len(val_loader.dataset)}")
        
        # Get optimization preset
        if optimization_mode == 'fast':
            preset = OptimizationPresets.get_fast_preset()
        elif optimization_mode == 'thorough':
            preset = OptimizationPresets.get_thorough_preset()
        elif optimization_mode == 'medical':
            preset = OptimizationPresets.get_medical_preset()
        else:
            preset = OptimizationPresets.get_fast_preset()
        
        optimization_status['total_trials'] = preset['n_trials']
        add_optimization_log(f"Optimization preset: {optimization_mode} ({preset['n_trials']} trials, {preset['timeout']}s timeout)")
        
        # Start tracking
        optimization_tracker.start_study(study_name, model_type)
        
        # Create optimizer
        optimizer = OptunaOptimizer(
            model_type=model_type,
            train_loader=train_loader,
            val_loader=val_loader,
            base_config=config,
            device=device,
            study_name=study_name
        )
        
        # Set up callbacks for real-time updates
        def optimization_progress_callback(current_trial, total_trials, message):
            optimization_status.update({
                'current_trial': current_trial,
                'total_trials': total_trials
            })
            add_optimization_log(message)
        
        def trial_completion_callback(trial_id, params, optimization_value, sensitivity, specificity, accuracy=0.0, mcc=0.0, auc_roc=0.0, error_msg=None):
            # Handle both new and old callback signatures
            f1_score = optimization_value  # For backward compatibility with OptimizationResult
            
            result = OptimizationResult(
                trial_id=trial_id,
                model_type=model_type,
                parameters=params,
                f1_score=f1_score,  # Store optimization_value as f1_score for compatibility
                sensitivity=sensitivity,  # Recall is same as sensitivity in medical context
                specificity=specificity,
                accuracy=accuracy,
                mcc=mcc,  # Matthews Correlation Coefficient
                auc_roc=auc_roc,  # AUC-ROC score
                training_time=0.0,  # Will be updated if available
                epochs_completed=params.get('epochs', 0),
                timestamp=datetime.now().isoformat(),
                study_name=study_name
            )
            
            optimization_tracker.record_trial(result)
            optimization_status['trial_results'].append(result.to_dict())
            
            # Update best results using optimization_value (the selected metric)
            if optimization_value > optimization_status['best_medical_score']:
                optimization_status.update({
                    'best_medical_score': optimization_value,
                    'best_params': params.copy()
                })
                # Get the current optimization metric name from optimizer
                metric_name = getattr(optimizer, 'optimize_metric', 'score')
                add_optimization_log(f"New best {metric_name}: {optimization_value:.4f}")
        
        optimizer.set_progress_callback(optimization_progress_callback)
        optimizer.set_trial_callback(trial_completion_callback)
        
        # Run optimization
        add_optimization_log("Starting hyperparameter optimization...")
        study = optimizer.optimize(
            n_trials=preset['n_trials'],
            timeout=preset['timeout'],
            pruner=preset['pruner']
        )
        
        # Finish tracking
        optimization_tracker.finish_study(preset['n_trials'], preset['timeout'])
        
        # Store final results
        optimization_status.update({
            'is_optimizing': False,
            'current_model': None,
            'best_medical_score': study.best_value,
            'best_params': study.best_params
        })
        
        add_optimization_log("\n=== Optimization Completed! ===")
        metric_name = getattr(optimizer, 'optimize_metric', 'score').upper()
        add_optimization_log(f"Best {metric_name}: {study.best_value:.4f}")
        add_optimization_log(f"Best Parameters: {study.best_params}")
        add_optimization_log(f"Total Trials: {len(study.trials)}")
        
    except Exception as e:
        optimization_status['is_optimizing'] = False
        optimization_status['current_model'] = None
        add_optimization_log(f"Optimization failed: {str(e)}")
        print(f"Optimization error: {e}")

def train_models_background(selected_models, model_parameters=None, data_config=None):
    """Background training function"""
    global training_status
    
    try:
        # Initialize training status
        training_status.update({
            'is_training': True,
            'start_time': datetime.now().isoformat(),
            'logs': [],
            'model_results': {},
            'epoch': 0,
            'total_epochs': 0,
            'metrics': {}
        })
        
        config = load_config()
        
        # Apply data configuration
        if data_config:
            if 'data_percentage' in data_config:
                config['data']['data_percentage'] = data_config['data_percentage']
                add_log(f"Using {data_config['data_percentage']:.1%} of dataset")
            if 'batch_size' in data_config:
                config['data']['batch_size'] = data_config['batch_size']
                add_log(f"Batch size: {data_config['batch_size']}")
            if 'class_balance_enabled' in data_config:
                config['class_balance']['enabled'] = data_config['class_balance_enabled']
                if 'class_balance_ratio' in data_config:
                    config['class_balance']['target_ratio'] = data_config['class_balance_ratio']
                    cancer_pct = int(data_config['class_balance_ratio'] * 100)
                    healthy_pct = 100 - cancer_pct
                    add_log(f"Class balance: {cancer_pct}% cancer, {healthy_pct}% healthy")
                else:
                    add_log(f"Class balancing: {'enabled' if data_config['class_balance_enabled'] else 'disabled'}")
            if 'early_stopping_patience' in data_config:
                config['training']['patience'] = data_config['early_stopping_patience']
                add_log(f"Early stopping patience: {data_config['early_stopping_patience']} epochs")
        
        # Apply custom parameters to config
        if model_parameters:
            for model_name, params in model_parameters.items():
                if model_name in config.get('models', {}):
                    config['models'][model_name].update(params)
                    add_log(f"Applied custom parameters for {model_name}: {params}")
        
        # Device selection based on config with M4 Pro optimizations
        training_config = config.get('training', {})
        device_config = training_config.get('device', 'auto')
        
        if device_config == 'mps' and torch.backends.mps.is_available():
            device = torch.device('mps')
            _apply_m4_pro_optimizations(config)
        elif device_config == 'cuda' and torch.cuda.is_available():
            device = torch.device('cuda')
        elif device_config == 'cpu':
            device = torch.device('cpu')
        else:  # auto
            if torch.backends.mps.is_available():
                device = torch.device('mps')
                _apply_m4_pro_optimizations(config)
            elif torch.cuda.is_available():
                device = torch.device('cuda')
            else:
                device = torch.device('cpu')
        
        # Add log
        add_log(f"Starting training on device: {device}")
        add_log(f"Selected models: {', '.join(selected_models)}")
        
        # Create data loaders
        add_log("Loading dataset...")
        train_loader, val_loader = create_dataloaders(config)
        add_log(f"Dataset loaded - Train: {len(train_loader.dataset)}, Val: {len(val_loader.dataset)}")
        
        # Train each selected model
        for model_name in selected_models:
            if not training_status['is_training']:  # Check if training was stopped
                break
            
            try:
                add_log(f"\n=== Training {model_name.upper()} ===")
                training_status['current_model'] = model_name
                
                # Create model
                model = create_model(model_name, config)
                add_log(f"Created {model_name} model")
                
                # Get epochs for this model
                model_config = config.get('models', {}).get(model_name, {})
                epochs = model_config.get('epochs', 20)
                training_status['total_epochs'] = epochs
                
                # Update training config with model-specific monitoring metric
                model_config = config.get('models', {}).get(model_name, {})
                training_config = config.get('training', {})
                if 'monitor_metric' in model_config:
                    training_config['monitor_metric'] = model_config['monitor_metric']
                    config['training'] = training_config
                    add_log(f"Using {model_config['monitor_metric']} as monitoring metric for {model_name}")
                
                # Create trainer
                trainer = BreastCancerTrainer(
                    model=model,
                    train_loader=train_loader,
                    val_loader=val_loader,
                    config=config,
                    device=device,
                    save_dir=f"models/checkpoints/{model_name}"
                )
                
                # Override trainer's logger to capture logs
                original_logger = trainer.logger
                trainer.logger = WebLogger(original_logger, add_log)
                
                # Create status callback to update progress
                def update_training_status(epoch, total_epochs, train_loss, val_loss, metrics):
                    # Convert tensor values to Python scalars to avoid tensor errors
                    def safe_get_metric(key, default=0):
                        value = metrics.get(key, default)
                        if hasattr(value, 'item'):  # PyTorch tensor
                            return float(value.item())
                        elif isinstance(value, (np.floating, np.integer)):  # NumPy types
                            return float(value)
                        else:
                            return float(value) if value is not None else default
                    
                    training_status.update({
                        'epoch': epoch,
                        'total_epochs': total_epochs,
                        'metrics': {
                            'train_loss': float(train_loss) if hasattr(train_loss, 'item') else train_loss,
                            'val_loss': float(val_loss) if hasattr(val_loss, 'item') else val_loss,
                            'recall': safe_get_metric('recall'),
                            'sensitivity': safe_get_metric('sensitivity'),
                            'specificity': safe_get_metric('specificity'),
                            'auc_roc': safe_get_metric('auc_roc'),
                            'mcc': safe_get_metric('mcc'),
                            'medical_composite': safe_get_metric('medical_composite')
                        }
                    })
                
                # Train model with status callback
                history = trainer.train(status_callback=update_training_status)
                
                # Store results
                if history['val_metrics']:
                    best_metrics = max(history['val_metrics'], key=lambda x: x['medical_composite'])
                    # Sanitize metrics for JSON serialization
                    sanitized_metrics = sanitize_metrics_for_json(best_metrics)
                    training_status['model_results'][model_name] = {
                        'completed': True,
                        'best_metrics': sanitized_metrics,
                        'training_time': time.time(),
                        'epochs_completed': len(history['val_metrics'])
                    }
                    
                    add_log(f"{model_name} training completed!")
                    add_log(f"Best Medical Composite Score: {best_metrics['medical_composite']:.4f}")
                    add_log(f"Best Recall: {best_metrics['recall']:.4f}")
                    add_log(f"Best Specificity: {best_metrics['specificity']:.4f}")
                    add_log(f"Best AUC-ROC: {best_metrics['auc_roc']:.4f}")
                    add_log(f"Best MCC: {best_metrics['mcc']:.4f}")
                
            except Exception as e:
                error_msg = f"Error training {model_name}: {str(e)}"
                add_log(error_msg)
                training_status['model_results'][model_name] = {
                    'completed': False,
                    'error': str(e)
                }
        
        # Training completed
        training_status.update({
            'is_training': False,
            'current_model': None,
            'epoch': 0,
            'total_epochs': 0
        })
        add_log("\n=== All Training Completed! ===")
        
        # Summary
        completed_models = [name for name, result in training_status['model_results'].items() 
                           if result.get('completed', False)]
        add_log(f"Successfully trained: {', '.join(completed_models)}")
        
    except Exception as e:
        training_status['is_training'] = False
        training_status['current_model'] = None
        add_log(f"Training failed: {str(e)}")

def create_model(model_name, config):
    """Create model based on name"""
    if model_name == 'cnn':
        return create_cnn_model(config)
    elif model_name == 'resnet':
        return create_resnet_model(config)
    elif model_name == 'densenet':
        return create_densenet_model(config) 
    elif model_name == 'faster_rcnn':
        return create_faster_rcnn_model(config)
    elif model_name == 'vision_transformer':
        return create_vit_model(config)
    else:
        raise ValueError(f"Unknown model: {model_name}")

def add_log(message):
    """Add log message to training status"""
    timestamp = datetime.now().strftime("%H:%M:%S")
    log_entry = f"[{timestamp}] {message}"
    training_status['logs'].append(log_entry)
    
    # Keep only last 100 log entries
    if len(training_status['logs']) > 100:
        training_status['logs'] = training_status['logs'][-100:]
    
    print(log_entry)  # Also print to console

def add_optimization_log(message):
    """Add log message to optimization status"""
    timestamp = datetime.now().strftime("%H:%M:%S")
    log_entry = f"[{timestamp}] {message}"
    optimization_status['logs'].append(log_entry)
    
    # Keep only last 100 log entries
    if len(optimization_status['logs']) > 100:
        optimization_status['logs'] = optimization_status['logs'][-100:]
    
    print(log_entry)  # Also print to console

class WebLogger:
    """Custom logger that captures logs for web interface"""
    
    def __init__(self, original_logger, log_function):
        self.original_logger = original_logger
        self.log_function = log_function
    
    def info(self, message):
        self.original_logger.info(message)
        self.log_function(message)
    
    def error(self, message):
        self.original_logger.error(message)
        self.log_function(f"ERROR: {message}")
    
    def warning(self, message):
        self.original_logger.warning(message)
        self.log_function(f"WARNING: {message}")


def _apply_m4_pro_optimizations(config):
    """Apply M4 Pro specific optimizations to configuration"""
    try:
        import platform
        import subprocess
        
        # Detect M4 Pro
        is_m4_pro = False
        if platform.system() == 'Darwin':
            try:
                result = subprocess.run(['sysctl', '-n', 'machdep.cpu.brand_string'], 
                                      capture_output=True, text=True)
                cpu_brand = result.stdout.strip()
                is_m4_pro = 'Apple M4 Pro' in cpu_brand or 'M4 Pro' in cpu_brand
            except:
                pass
        
        if not is_m4_pro:
            is_m4_pro = torch.backends.mps.is_available()
        
        if is_m4_pro:
            # Apply M4 Pro memory optimizations
            m4_pro_config = config.get('training', {}).get('m4_pro_optimizations', {})
            
            # Set memory fraction
            memory_fraction = m4_pro_config.get('memory_fraction', 0.85)
            try:
                if hasattr(torch.mps, 'set_per_process_memory_fraction'):
                    torch.mps.set_per_process_memory_fraction(memory_fraction)
            except:
                pass
            
            # Enable graph mode if available
            if m4_pro_config.get('enable_graph_mode', True):
                try:
                    if hasattr(torch.backends.mps, 'enable_graph_mode'):
                        torch.backends.mps.enable_graph_mode(True)
                except:
                    pass
            
            print(f"Applied M4 Pro optimizations: memory_fraction={memory_fraction}")
    except Exception as e:
        print(f"Warning: Could not apply M4 Pro optimizations: {e}")

# Create templates directory and basic HTML template
def create_templates():
    """Create HTML templates for the web interface"""
    # Get the directory where app.py is located
    app_dir = Path(__file__).parent
    templates_dir = app_dir / "templates"
    templates_dir.mkdir(exist_ok=True)
    create_templates_at_path(templates_dir)

def create_templates_at_path(templates_dir):
    """Create HTML templates at specified path"""
    # Create main template
    html_template = '''
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Breast Cancer Detection - Model Training</title>
    <style>
        * { margin: 0; padding: 0; box-sizing: border-box; }
        body { font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; background: #f5f5f5; }
        .header { background: #2c3e50; color: white; padding: 1rem; text-align: center; }
        .container { max-width: 1200px; margin: 0 auto; padding: 2rem; }
        .card { background: white; border-radius: 8px; padding: 1.5rem; margin-bottom: 1rem; box-shadow: 0 2px 4px rgba(0,0,0,0.1); }
        .models-grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(300px, 1fr)); gap: 1rem; }
        .model-card { border: 2px solid #e0e0e0; border-radius: 8px; padding: 1rem; transition: all 0.3s; }
        .model-card:hover { border-color: #3498db; }
        .model-card.selected { border-color: #e74c3c; background: #ffeaea; }
        .model-card h3 { color: #2c3e50; margin-bottom: 0.5rem; }
        .model-params { display: none; margin-top: 1rem; padding: 1rem; background: #f8f9fa; border-radius: 4px; border-left: 4px solid #e74c3c; }
        .model-card.selected .model-params { display: block; }
        .param-group { margin-bottom: 1rem; }
        .param-group label { display: block; margin-bottom: 0.25rem; font-weight: 600; color: #2c3e50; }
        .param-group input, .param-group select { width: 100%; padding: 0.5rem; border: 1px solid #ddd; border-radius: 4px; }
        .param-group small { color: #666; font-size: 0.85rem; }
        .btn { background: #3498db; color: white; border: none; padding: 0.75rem 1.5rem; border-radius: 4px; cursor: pointer; font-size: 1rem; }
        .btn:hover { background: #2980b9; }
        .btn:disabled { background: #bdc3c7; cursor: not-allowed; }
        .btn-danger { background: #e74c3c; }
        .btn-danger:hover { background: #c0392b; }
        .status-panel { background: #2c3e50; color: white; padding: 1rem; border-radius: 8px; }
        .logs { height: 300px; overflow-y: auto; background: #f8f9fa; padding: 1rem; border-radius: 4px; font-family: monospace; font-size: 0.9rem; }
        .metrics-table { width: 100%; border-collapse: collapse; margin-top: 1rem; }
        .metrics-table th, .metrics-table td { padding: 0.5rem; text-align: left; border-bottom: 1px solid #ddd; }
        .metrics-table th { background: #f8f9fa; color: #2c3e50; font-weight: 600; }
        .metrics-table td { color: #2c3e50; }
        .metrics-table tbody tr:hover { background-color: #f1f3f4; }
        .metrics-table tbody tr:hover td { color: #1a252f; }
        .progress { background: #ecf0f1; border-radius: 10px; height: 20px; margin: 0.5rem 0; }
        .progress-bar { background: #3498db; height: 100%; border-radius: 10px; transition: width 0.3s; }
        .dataset-info { display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 1rem; }
        .stat-box { text-align: center; padding: 1rem; background: #ecf0f1; border-radius: 8px; }
        .stat-number { font-size: 2rem; font-weight: bold; color: #2c3e50; }
        .stat-label { color: #7f8c8d; margin-top: 0.5rem; }
        
        /* Training Mode Selection Styles */
        .training-mode-label:hover { border-color: #3498db !important; transform: translateY(-2px); box-shadow: 0 4px 8px rgba(0,0,0,0.1); }
        .training-mode-label.selected { border-color: #e74c3c !important; background: #ffeaea !important; }
        .training-mode-label.selected:hover { border-color: #c0392b !important; }
        
        /* Responsive design for training mode */
        @media (max-width: 768px) {
            .training-mode-options { flex-direction: column; align-items: center; }
            .training-mode-option { max-width: 100%; }
        }
    </style>
</head>
<body>
    <div class="header">
        <h1>🔬 Breast Cancer Detection - Model Training Interface</h1>
        <p>Train and compare CNN, ResNet, and Vision Transformer models</p>
    </div>
    
    <div class="container">

        <!-- Dataset Information -->
        <div class="card">
            <h2>📊 Dataset Information</h2>
            <div id="dataset-info" class="dataset-info">
                <div class="stat-box">
                    <div class="stat-number" id="total-patients">-</div>
                    <div class="stat-label">Patients</div>
                </div>
                <div class="stat-box">
                    <div class="stat-number" id="total-images">-</div>
                    <div class="stat-label">Images</div>
                </div>
                <div class="stat-box">
                    <div class="stat-number" id="cancer-images">-</div>
                    <div class="stat-label">Cancer Images</div>
                </div>
                <div class="stat-box">
                    <div class="stat-number" id="healthy-images">-</div>
                    <div class="stat-label">Healthy Images</div>
                </div>
            </div>
        </div>
        
        <!-- Data Configuration -->
        <div class="card">
            <h2>⚙️ Data Configuration</h2>
            <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(250px, 1fr)); gap: 1rem;">
                <div class="param-group">
                    <label for="data-percentage">Dataset Percentage</label>
                    <input type="number" id="data-percentage" value="5" min="0.1" max="100" step="0.1">
                    <small>Percentage of dataset to use (0.1% = ~180 images, 5% = ~9000 images)</small>
                </div>
                <div class="param-group">
                    <label for="batch-size">Batch Size</label>
                    <input type="number" id="batch-size" value="32" min="4" max="128" step="4">
                    <small>Training batch size</small>
                </div>
                <div class="param-group">
                    <label for="class-balance">Class Balance Ratio</label>
                    <input type="range" id="class-balance" min="0" max="100" value="50" oninput="updateClassBalanceDisplay(this.value)">
                    <div style="display: flex; justify-content: space-between; font-size: 0.85rem; color: #666; margin-top: 0.25rem;">
                        <span>Cancer: <span id="cancer-percentage">50</span>%</span>
                        <span>Healthy: <span id="healthy-percentage">50</span>%</span>
                    </div>
                    <small>Adjust class distribution (50/50 recommended for medical training)</small>
                </div>
                <div class="param-group">
                    <label for="early-stopping-patience">Early Stopping Patience</label>
                    <input type="number" id="early-stopping-patience" value="20" min="1" max="100" step="1">
                    <small>Number of epochs to wait before stopping if no improvement (applies to all models)</small>
                </div>
            </div>
        </div>
        
        <!-- Model Selection -->
        <div class="card">
            <h2>🤖 Model Selection</h2>
            <div class="models-grid">
                <div class="model-card" data-model="cnn" onclick="toggleModel('cnn')">
                    <h3>Custom CNN</h3>
                    <p>Custom convolutional neural network optimized for 50x50 medical images</p>
                    <p><strong>Training time:</strong> ~20-30 minutes</p>
                    <p><strong>Parameters:</strong> ~500K</p>
                    
                    <div class="model-params" onclick="event.stopPropagation()">
                        <h4>CNN Parameters</h4>
                        <div class="param-group">
                            <label for="cnn-epochs">Epochs</label>
                            <input type="number" id="cnn-epochs" value="50" min="1" max="200">
                            <small>Number of training epochs</small>
                        </div>
                        <div class="param-group">
                            <label for="cnn-lr">Learning Rate</label>
                            <input type="number" id="cnn-lr" value="0.001" step="0.0001" min="0.0001" max="0.1">
                            <small>Learning rate for optimizer</small>
                        </div>
                        <div class="param-group">
                            <label for="cnn-dropout">Dropout Rate</label>
                            <input type="number" id="cnn-dropout" value="0.5" step="0.1" min="0.0" max="0.9">
                            <small>Dropout probability for regularization</small>
                        </div>
                        <div class="param-group">
                            <label for="cnn-architecture">Architecture</label>
                            <select id="cnn-architecture">
                                <option value="custom">Custom CNN</option>
                                <option value="compact">Compact CNN</option>
                            </select>
                            <small>CNN architecture variant</small>
                        </div>
                        <div class="param-group">
                            <label for="cnn-monitor-metric">Monitoring Metric</label>
                            <select id="cnn-monitor-metric">
                                <option value="mcc">Matthews Correlation Coefficient (MCC)</option>
                                <option value="recall">Recall (Cancer Detection)</option>
                                <option value="auc_roc">AUC-ROC (Overall Performance)</option>
                            </select>
                            <small>Metric used for early stopping and best model selection</small>
                        </div>
                    </div>
                </div>
                
                <div class="model-card" data-model="resnet" onclick="toggleModel('resnet')">
                    <h3>ResNet</h3>
                    <p>Pre-trained ResNet with medical adaptations and transfer learning</p>
                    <p><strong>Training time:</strong> ~30-45 minutes</p>
                    <p><strong>Parameters:</strong> ~11M</p>
                    
                    <div class="model-params" onclick="event.stopPropagation()">
                        <h4>ResNet Parameters</h4>
                        <div class="param-group">
                            <label for="resnet-epochs">Epochs</label>
                            <input type="number" id="resnet-epochs" value="30" min="1" max="200">
                            <small>Number of training epochs</small>
                        </div>
                        <div class="param-group">
                            <label for="resnet-lr">Learning Rate</label>
                            <input type="number" id="resnet-lr" value="0.0001" step="0.00001" min="0.00001" max="0.01">
                            <small>Learning rate for fine-tuning</small>
                        </div>
                        <div class="param-group">
                            <label for="resnet-architecture">Architecture</label>
                            <select id="resnet-architecture">
                                <option value="resnet18">ResNet-18</option>
                                <option value="resnet34">ResNet-34</option>
                                <option value="resnet50">ResNet-50</option>
                                <option value="lightweight">Lightweight ResNet</option>
                            </select>
                            <small>ResNet architecture variant</small>
                        </div>
                        <div class="param-group">
                            <label for="resnet-pretrained">Pre-trained</label>
                            <select id="resnet-pretrained">
                                <option value="true">Yes (ImageNet)</option>
                                <option value="false">No (Random init)</option>
                            </select>
                            <small>Use ImageNet pre-trained weights</small>
                        </div>
                        <div class="param-group">
                            <label for="resnet-fine-tune">Fine-tune Layers</label>
                            <input type="number" id="resnet-fine-tune" value="-1" min="-1" max="10">
                            <small>Number of layers to fine-tune (-1 for all)</small>
                        </div>
                        <div class="param-group">
                            <label for="resnet-monitor-metric">Monitoring Metric</label>
                            <select id="resnet-monitor-metric">
                                <option value="mcc">Matthews Correlation Coefficient (MCC)</option>
                                <option value="recall">Recall (Cancer Detection)</option>
                                <option value="auc_roc">AUC-ROC (Overall Performance)</option>
                            </select>
                            <small>Metric used for early stopping and best model selection</small>
                        </div>
                    </div>
                </div>
                
                <div class="model-card" data-model="densenet" onclick="toggleModel('densenet')">
                    <h3>DenseNet</h3>
                    <p>Dense Convolutional Network with efficient feature reuse and medical adaptations</p>
                    <p><strong>Training time:</strong> ~25-35 minutes</p>
                    <p><strong>Parameters:</strong> ~8M (DenseNet-121)</p>
                    
                    <div class="model-params" onclick="event.stopPropagation()">
                        <h4>DenseNet Parameters</h4>
                        <div class="param-group">
                            <label for="densenet-epochs">Epochs</label>
                            <input type="number" id="densenet-epochs" value="25" min="1" max="200">
                            <small>Number of training epochs</small>
                        </div>
                        <div class="param-group">
                            <label for="densenet-lr">Learning Rate</label>
                            <input type="number" id="densenet-lr" value="0.0001" step="0.00001" min="0.00001" max="0.01">
                            <small>Learning rate for fine-tuning</small>
                        </div>
                        <div class="param-group">
                            <label for="densenet-architecture">Architecture</label>
                            <select id="densenet-architecture">
                                <option value="densenet121">DenseNet-121</option>
                                <option value="densenet169">DenseNet-169</option>
                                <option value="densenet201">DenseNet-201</option>
                                <option value="compact">Compact DenseNet</option>
                            </select>
                            <small>DenseNet architecture variant</small>
                        </div>
                        <div class="param-group">
                            <label for="densenet-pretrained">Pre-trained</label>
                            <select id="densenet-pretrained">
                                <option value="true">Yes (ImageNet)</option>
                                <option value="false">No (Random init)</option>
                            </select>
                            <small>Use ImageNet pre-trained weights</small>
                        </div>
                        <div class="param-group">
                            <label for="densenet-growth-rate">Growth Rate</label>
                            <input type="number" id="densenet-growth-rate" value="32" min="8" max="64" step="4">
                            <small>Growth rate for dense layers</small>
                        </div>
                        <div class="param-group">
                            <label for="densenet-compression">Compression</label>
                            <input type="number" id="densenet-compression" value="0.5" min="0.1" max="1.0" step="0.1">
                            <small>Compression factor for transition layers</small>
                        </div>
                        <div class="param-group">
                            <label for="densenet-dropout">Dropout Rate</label>
                            <input type="number" id="densenet-dropout" value="0.3" step="0.05" min="0.0" max="0.8">
                            <small>Dropout rate in classification head</small>
                        </div>
                        <div class="param-group">
                            <label for="densenet-monitor-metric">Monitoring Metric</label>
                            <select id="densenet-monitor-metric">
                                <option value="mcc">Matthews Correlation Coefficient (MCC)</option>
                                <option value="recall">Recall (Cancer Detection)</option>
                                <option value="auc_roc">AUC-ROC (Overall Performance)</option>
                            </select>
                            <small>Metric used for early stopping and best model selection</small>
                        </div>
                    </div>
                </div>
                
                <div class="model-card" data-model="faster_rcnn" onclick="toggleModel('faster_rcnn')">
                    <h3>Faster R-CNN</h3>
                    <p>Region-based CNN adapted for classification with attention mechanisms</p>
                    <p><strong>Training time:</strong> ~35-50 minutes</p>
                    <p><strong>Parameters:</strong> ~28M (ResNet-50 backbone)</p>
                    
                    <div class="model-params" onclick="event.stopPropagation()">
                        <h4>Faster R-CNN Parameters</h4>
                        <div class="param-group">
                            <label for="faster-rcnn-epochs">Epochs</label>
                            <input type="number" id="faster-rcnn-epochs" value="20" min="1" max="100">
                            <small>Number of training epochs</small>
                        </div>
                        <div class="param-group">
                            <label for="faster-rcnn-lr">Learning Rate</label>
                            <input type="number" id="faster-rcnn-lr" value="0.00005" step="0.000001" min="0.000001" max="0.001">
                            <small>Learning rate for complex architecture</small>
                        </div>
                        <div class="param-group">
                            <label for="faster-rcnn-backbone">Backbone</label>
                            <select id="faster-rcnn-backbone">
                                <option value="resnet50">ResNet-50</option>
                                <option value="resnet101">ResNet-101</option>
                                <option value="compact">Compact R-CNN</option>
                            </select>
                            <small>Backbone architecture for feature extraction</small>
                        </div>
                        <div class="param-group">
                            <label for="faster-rcnn-pretrained">Pre-trained</label>
                            <select id="faster-rcnn-pretrained">
                                <option value="true">Yes (COCO)</option>
                                <option value="false">No (Random init)</option>
                            </select>
                            <small>Use COCO pre-trained weights</small>
                        </div>
                        <div class="param-group">
                            <label for="faster-rcnn-anchor-scales">Anchor Scales</label>
                            <select id="faster-rcnn-anchor-scales">
                                <option value="8,16,32">Small (8,16,32)</option>
                                <option value="4,8,16">Very Small (4,8,16)</option>
                                <option value="16,32,64">Large (16,32,64)</option>
                            </select>
                            <small>Anchor scales for region proposals</small>
                        </div>
                        <div class="param-group">
                            <label for="faster-rcnn-roi-pool-size">ROI Pool Size</label>
                            <input type="number" id="faster-rcnn-roi-pool-size" value="7" min="3" max="14" step="2">
                            <small>ROI pooling output size (odd numbers preferred)</small>
                        </div>
                        <div class="param-group">
                            <label for="faster-rcnn-dropout">Dropout Rate</label>
                            <input type="number" id="faster-rcnn-dropout" value="0.5" step="0.05" min="0.0" max="0.8">
                            <small>Dropout rate in classification heads</small>
                        </div>
                        <div class="param-group">
                            <label for="faster-rcnn-monitor-metric">Monitoring Metric</label>
                            <select id="faster-rcnn-monitor-metric">
                                <option value="mcc">Matthews Correlation Coefficient (MCC)</option>
                                <option value="recall">Recall (Cancer Detection)</option>
                                <option value="auc_roc">AUC-ROC (Overall Performance)</option>
                            </select>
                            <small>Metric used for early stopping and best model selection</small>
                        </div>
                    </div>
                </div>
                
                <div class="model-card" data-model="vision_transformer" onclick="toggleModel('vision_transformer')">
                    <h3>Vision Transformer</h3>
                    <p>Transformer-based architecture with attention mechanisms</p>
                    <p><strong>Training time:</strong> ~45-60 minutes</p>
                    <p><strong>Parameters:</strong> ~22M</p>
                    
                    <div class="model-params" onclick="event.stopPropagation()">
                        <h4>Vision Transformer Parameters</h4>
                        <div class="param-group">
                            <label for="vit-epochs">Epochs</label>
                            <input type="number" id="vit-epochs" value="20" min="1" max="100">
                            <small>Number of training epochs</small>
                        </div>
                        <div class="param-group">
                            <label for="vit-lr">Learning Rate</label>
                            <input type="number" id="vit-lr" value="0.00001" step="0.000001" min="0.000001" max="0.001">
                            <small>Learning rate for transformer</small>
                        </div>
                        <div class="param-group">
                            <label for="vit-model">Model Variant</label>
                            <select id="vit-model">
                                <option value="vit_tiny_patch16_224">ViT-Tiny</option>
                                <option value="vit_small_patch16_224">ViT-Small</option>
                                <option value="vit_base_patch16_224">ViT-Base</option>
                                <option value="compact">Compact ViT</option>
                            </select>
                            <small>Vision Transformer model size</small>
                        </div>
                        <div class="param-group">
                            <label for="vit-dropout">Dropout Rate</label>
                            <input type="number" id="vit-dropout" value="0.1" step="0.05" min="0.0" max="0.5">
                            <small>Dropout rate in classification head</small>
                        </div>
                        <div class="param-group">
                            <label for="vit-patch-size">Patch Size (for Compact)</label>
                            <input type="number" id="vit-patch-size" value="5" min="2" max="10">
                            <small>Patch size for compact ViT (50px images)</small>
                        </div>
                        <div class="param-group">
                            <label for="vit-monitor-metric">Monitoring Metric</label>
                            <select id="vit-monitor-metric">
                                <option value="mcc">Matthews Correlation Coefficient (MCC)</option>
                                <option value="recall">Recall (Cancer Detection)</option>
                                <option value="auc_roc">AUC-ROC (Overall Performance)</option>
                            </select>
                            <small>Metric used for early stopping and best model selection</small>
                        </div>
                    </div>
                </div>
            </div>
            
            <div style="margin-top: 2rem;">
                <!-- Training Mode Selection -->
                <div class="training-mode-section" style="margin-bottom: 2rem; padding: 1.5rem; background: #f8f9fa; border-radius: 8px; border-left: 4px solid #3498db;">
                    <h3 style="text-align: center; margin-bottom: 1.5rem; color: #2c3e50; font-size: 1.2rem;">Select Training Mode</h3>
                    <div class="training-mode-options" style="display: flex; flex-wrap: wrap; gap: 1.5rem; justify-content: center;">
                        <div class="training-mode-option" style="flex: 1; min-width: 250px; max-width: 300px;">
                            <label class="training-mode-label" style="display: block; padding: 1rem; border: 2px solid #e0e0e0; border-radius: 8px; cursor: pointer; transition: all 0.3s; background: white;" onclick="selectTrainingMode('manual')">
                                <div style="display: flex; align-items: center; gap: 0.75rem; margin-bottom: 0.5rem;">
                                    <input type="radio" name="training-mode" value="manual" checked onchange="toggleTrainingMode()" style="margin: 0;">
                                    <strong style="font-size: 1.1rem; color: #2c3e50;">Manual Training</strong>
                                </div>
                                <p style="margin: 0; font-size: 0.9rem; color: #666; line-height: 1.4;">
                                    Train selected models with custom parameters. Full control over hyperparameters and training settings.
                                </p>
                            </label>
                        </div>
                        <div class="training-mode-option" style="flex: 1; min-width: 250px; max-width: 300px;">
                            <label class="training-mode-label" style="display: block; padding: 1rem; border: 2px solid #e0e0e0; border-radius: 8px; cursor: pointer; transition: all 0.3s; background: white;" onclick="selectTrainingMode('optimization')">
                                <div style="display: flex; align-items: center; gap: 0.75rem; margin-bottom: 0.5rem;">
                                    <input type="radio" name="training-mode" value="optimization" onchange="toggleTrainingMode()" style="margin: 0;">
                                    <strong style="font-size: 1.1rem; color: #2c3e50;">Hyperparameter</strong>
                                </div>
                                <p style="margin: 0; font-size: 0.9rem; color: #666; line-height: 1.4;">
                                    Automatically find optimal hyperparameters using Optuna. Best for maximizing model performance.
                                </p>
                            </label>
                        </div>
                    </div>
                </div>

                <!-- Manual Training Controls -->
                <div id="manual-training-controls" style="text-align: center; padding: 1.5rem; background: #f8f9fa; border-radius: 8px; border-left: 4px solid #27ae60;">
                    <h4 style="margin-bottom: 1rem; color: #2c3e50;">Manual Training Controls</h4>
                    <p style="margin-bottom: 1.5rem; color: #666; font-size: 0.95rem;">
                        Start training with your selected models and custom parameters
                    </p>
                    <div style="display: flex; gap: 1rem; justify-content: center; flex-wrap: wrap;">
                        <button class="btn" id="start-training-btn" onclick="startTraining()" style="min-width: 200px;">
                            🚀 Start Training Selected Models
                        </button>
                        <button class="btn btn-danger" id="stop-training-btn" onclick="stopTraining()" style="display: none; min-width: 150px;">
                            ⏹️ Stop Training
                        </button>
                    </div>
                </div>

                <!-- Optimization Controls -->
                <div id="optimization-controls" style="display: none; padding: 1.5rem; background: #f8f9fa; border-radius: 8px; border-left: 4px solid #9b59b6;">
                    <h4 style="margin-bottom: 1rem; color: #2c3e50; text-align: center;">Hyperparameter Optimization Controls</h4>
                    <p style="margin-bottom: 1.5rem; color: #666; font-size: 0.95rem; text-align: center;">
                        Automatically find the best hyperparameters for optimal model performance
                    </p>
                    <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 1rem; margin-bottom: 1.5rem; text-align: left;">
                        <div class="param-group">
                            <label for="optimization-model">Model to Optimize</label>
                            <select id="optimization-model">
                                <option value="cnn">CNN</option>
                                <option value="resnet">ResNet</option>
                                <option value="densenet">DenseNet</option>
                                <option value="faster_rcnn">Faster R-CNN</option>
                                <option value="vision_transformer">Vision Transformer</option>
                            </select>
                            <small>Select one model for optimization</small>
                        </div>
                        <div class="param-group">
                            <label for="optimization-preset">Optimization Preset</label>
                            <select id="optimization-preset">
                                <option value="fast">Fast (20 trials, 1 hour)</option>
                                <option value="medical" selected>Medical (50 trials, 2 hours)</option>
                                <option value="thorough">Thorough (100 trials, 4 hours)</option>
                            </select>
                            <small>Choose optimization strategy</small>
                        </div>
                        <div class="param-group">
                            <label for="optimization-trials">Number of Trials</label>
                            <input type="number" id="optimization-trials" value="50" min="5" max="200">
                            <small>Override preset trial count</small>
                        </div>
                    </div>
                    <div style="text-align: center;">
                        <div style="display: flex; gap: 1rem; justify-content: center; flex-wrap: wrap;">
                            <button class="btn" id="start-optimization-btn" onclick="startOptimization()" style="min-width: 220px;">
                                🔍 Start Hyperparameter Optimization
                            </button>
                            <button class="btn btn-danger" id="stop-optimization-btn" onclick="stopOptimization()" style="display: none; min-width: 150px;">
                                ⏹️ Stop Optimization
                            </button>
                        </div>
                    </div>
                </div>
            </div>
        </div>
        
        <!-- Snake Game (appears during training) -->
        <div class="card" id="snake-game-card" style="display: none;">
            <h2>🐍 Snake Game - Play while training!</h2>
            <div style="text-align: center;">
                <canvas id="snakeCanvas" width="400" height="300" style="border: 2px solid #2c3e50; background: #ecf0f1;"></canvas>
                <div style="margin-top: 1rem;">
                    <div>Score: <span id="snake-score">0</span></div>
                    <div style="margin-top: 0.5rem; font-size: 0.9rem; color: #666;">
                        Use arrow keys to control the snake
                    </div>
                    <button id="see-results-btn" class="btn" onclick="scrollToResults()" style="display: none; margin-top: 1rem;">
                        🎯 See Training Results
                    </button>
                </div>
            </div>
        </div>

        <!-- Training/Optimization Status -->
        <div class="card">
            <h2 id="status-header">📈 Training Status</h2>
            <div class="status-panel">
                <!-- Training Status -->
                <div id="training-status-panel">
                    <div id="training-status">Ready to start training...</div>
                    <div id="progress-container" style="display: none;">
                        <div>Current Model: <span id="current-model">-</span></div>
                        <div>Epoch: <span id="current-epoch">0</span>/<span id="total-epochs">0</span></div>
                        <div class="progress">
                            <div class="progress-bar" id="progress-bar" style="width: 0%;"></div>
                        </div>
                    </div>
                </div>
                
                <!-- Optimization Status -->
                <div id="optimization-status-panel" style="display: none;">
                    <div id="optimization-status">Ready to start optimization...</div>
                    <div id="optimization-progress-container" style="display: none;">
                        <div>Model: <span id="optimization-current-model">-</span></div>
                        <div>Trial: <span id="optimization-current-trial">0</span>/<span id="optimization-total-trials">0</span></div>
                        <div>Best Recall: <span id="optimization-best-recall">0.0000</span></div>
                        <div>Best Medical Score: <span id="optimization-best-medical">0.0000</span></div>
                        <div class="progress">
                            <div class="progress-bar" id="optimization-progress-bar" style="width: 0%;"></div>
                        </div>
                        <div style="margin-top: 0.5rem;">
                            <details>
                                <summary style="cursor: pointer; color: #3498db;">Best Parameters Found</summary>
                                <div id="optimization-best-params" style="margin-top: 0.5rem; font-size: 0.9rem; background: #34495e; padding: 0.5rem; border-radius: 4px; font-family: monospace;">
                                    No parameters found yet...
                                </div>
                            </details>
                        </div>
                    </div>
                    
                    <!-- Optuna Trial Results -->
                    <div id="optuna-results-container" style="margin-top: 1rem; display: none;">
                        <h4>Trial Results</h4>
                        <div id="optuna-results-table" style="max-height: 300px; overflow-y: auto;">
                            <p>Trial results will appear here during optimization...</p>
                        </div>
                    </div>
                </div>
            </div>
            
            <!-- Training Logs -->
            <div style="margin-top: 1rem;">
                <h3>Training Logs</h3>
                <div id="training-logs" class="logs">No logs yet...</div>
            </div>
        </div>
        
        <!-- Results -->
        <div class="card">
            <h2>📊 Training Results</h2>
            <div id="results-container">
                <p>Results will appear here after training completion...</p>
            </div>
        </div>
    </div>

    <script>
        let selectedModels = [];
        let updateInterval;
        let currentTrainingMode = 'manual';
        
        // Snake game variables
        let snake = [{x: 200, y: 150}];
        let dx = 20, dy = 0;
        let food = {x: 0, y: 0};
        let score = 0;
        let gameRunning = false;
        let gameLoop;

        // Load dataset info on page load
        window.onload = function() {
            loadDatasetInfo();
            loadConfigDefaults();
            startStatusUpdates();
            initSnakeGame();
            initializeTrainingMode();
        };
        
        function initializeTrainingMode() {
            // Set initial visual state for manual training mode (default)
            const manualLabel = document.querySelector('input[name="training-mode"][value="manual"]').closest('.training-mode-label');
            manualLabel.classList.add('selected');
            toggleTrainingMode();
        }

        function updateClassBalanceDisplay(value) {
            const cancerPercentage = parseInt(value);
            const healthyPercentage = 100 - cancerPercentage;
            document.getElementById('cancer-percentage').textContent = cancerPercentage;
            document.getElementById('healthy-percentage').textContent = healthyPercentage;
        }

        function loadDatasetInfo() {
            fetch('/api/dataset_info')
                .then(response => response.json())
                .then(data => {
                    if (data.error) {
                        console.error('Dataset error:', data.error);
                        return;
                    }
                    document.getElementById('total-patients').textContent = data.total_patients;
                    document.getElementById('total-images').textContent = data.total_images;
                    document.getElementById('cancer-images').textContent = data.class_distribution['0'];
                    document.getElementById('healthy-images').textContent = data.class_distribution['1'];
                })
                .catch(error => console.error('Error loading dataset info:', error));
        }

        function loadConfigDefaults() {
            fetch('/api/config')
                .then(response => response.json())
                .then(config => {
                    // Set early stopping patience from config
                    const patience = config.training?.patience || 20;
                    document.getElementById('early-stopping-patience').value = patience;
                    
                    // Set other defaults from config
                    const dataPercentage = (config.data?.data_percentage || 0.05) * 100;
                    document.getElementById('data-percentage').value = dataPercentage;
                    
                    const batchSize = config.data?.batch_size || 32;
                    document.getElementById('batch-size').value = batchSize;
                })
                .catch(error => console.error('Error loading config defaults:', error));
        }

        function toggleModel(modelName) {
            const card = document.querySelector(`[data-model="${modelName}"]`);
            const index = selectedModels.indexOf(modelName);
            
            if (index > -1) {
                selectedModels.splice(index, 1);
                card.classList.remove('selected');
            } else {
                selectedModels.push(modelName);
                card.classList.add('selected');
            }
            
            updateStartButton();
        }

        function updateStartButton() {
            const startBtn = document.getElementById('start-training-btn');
            startBtn.disabled = selectedModels.length === 0;
        }

        function startTraining() {
            if (selectedModels.length === 0) {
                alert('Please select at least one model to train.');
                return;
            }

            // Collect data configuration
            const classBalanceValue = parseInt(document.getElementById('class-balance').value);
            const dataConfig = {
                data_percentage: parseFloat(document.getElementById('data-percentage').value) / 100, // Convert % to decimal
                batch_size: parseInt(document.getElementById('batch-size').value),
                class_balance_ratio: classBalanceValue / 100, // Convert to 0.0-1.0 range
                class_balance_enabled: classBalanceValue !== 50, // Enable balancing if not 50/50
                early_stopping_patience: parseInt(document.getElementById('early-stopping-patience').value)
            };

            // Collect parameters for each selected model
            const modelParams = {};
            
            selectedModels.forEach(modelName => {
                if (modelName === 'cnn') {
                    modelParams.cnn = {
                        epochs: parseInt(document.getElementById('cnn-epochs').value),
                        learning_rate: parseFloat(document.getElementById('cnn-lr').value),
                        dropout: parseFloat(document.getElementById('cnn-dropout').value),
                        architecture: document.getElementById('cnn-architecture').value,
                        monitor_metric: document.getElementById('cnn-monitor-metric').value
                    };
                } else if (modelName === 'resnet') {
                    modelParams.resnet = {
                        epochs: parseInt(document.getElementById('resnet-epochs').value),
                        learning_rate: parseFloat(document.getElementById('resnet-lr').value),
                        architecture: document.getElementById('resnet-architecture').value,
                        pretrained: document.getElementById('resnet-pretrained').value === 'true',
                        fine_tune_layers: parseInt(document.getElementById('resnet-fine-tune').value),
                        monitor_metric: document.getElementById('resnet-monitor-metric').value
                    };
                } else if (modelName === 'densenet') {
                    modelParams.densenet = {
                        epochs: parseInt(document.getElementById('densenet-epochs').value),
                        learning_rate: parseFloat(document.getElementById('densenet-lr').value),
                        architecture: document.getElementById('densenet-architecture').value,
                        pretrained: document.getElementById('densenet-pretrained').value === 'true',
                        growth_rate: parseInt(document.getElementById('densenet-growth-rate').value),
                        compression: parseFloat(document.getElementById('densenet-compression').value),
                        dropout_rate: parseFloat(document.getElementById('densenet-dropout').value),
                        monitor_metric: document.getElementById('densenet-monitor-metric').value
                    };
                } else if (modelName === 'faster_rcnn') {
                    const anchorScalesStr = document.getElementById('faster-rcnn-anchor-scales').value;
                    const anchorScales = anchorScalesStr.split(',').map(s => parseInt(s.trim()));
                    
                    modelParams.faster_rcnn = {
                        epochs: parseInt(document.getElementById('faster-rcnn-epochs').value),
                        learning_rate: parseFloat(document.getElementById('faster-rcnn-lr').value),
                        backbone: document.getElementById('faster-rcnn-backbone').value,
                        pretrained: document.getElementById('faster-rcnn-pretrained').value === 'true',
                        anchor_scales: anchorScales,
                        roi_pool_size: parseInt(document.getElementById('faster-rcnn-roi-pool-size').value),
                        dropout_rate: parseFloat(document.getElementById('faster-rcnn-dropout').value),
                        monitor_metric: document.getElementById('faster-rcnn-monitor-metric').value
                    };
                } else if (modelName === 'vision_transformer') {
                    modelParams.vision_transformer = {
                        epochs: parseInt(document.getElementById('vit-epochs').value),
                        learning_rate: parseFloat(document.getElementById('vit-lr').value),
                        model_name: document.getElementById('vit-model').value,
                        dropout_rate: parseFloat(document.getElementById('vit-dropout').value),
                        patch_size: parseInt(document.getElementById('vit-patch-size').value),
                        monitor_metric: document.getElementById('vit-monitor-metric').value
                    };
                }
            });

            fetch('/api/start_training', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ 
                    models: selectedModels,
                    parameters: modelParams,
                    data_config: dataConfig
                })
            })
            .then(response => response.json())
            .then(data => {
                if (data.success) {
                    document.getElementById('start-training-btn').style.display = 'none';
                    document.getElementById('stop-training-btn').style.display = 'inline-block';
                    document.getElementById('progress-container').style.display = 'block';
                } else {
                    alert('Error starting training: ' + data.error);
                }
            });
        }

        function stopTraining() {
            fetch('/api/stop_training', { method: 'POST' })
            .then(response => response.json())
            .then(data => {
                document.getElementById('start-training-btn').style.display = 'inline-block';
                document.getElementById('stop-training-btn').style.display = 'none';
                document.getElementById('progress-container').style.display = 'none';
            });
        }

        function startStatusUpdates() {
            updateInterval = setInterval(updateStatus, 2000);
        }

        function updateStatus() {
            fetch('/api/training_status')
            .then(response => response.json())
            .then(status => {
                // Update status display
                const statusElement = document.getElementById('training-status');
                const snakeGameCard = document.getElementById('snake-game-card');
                const seeResultsBtn = document.getElementById('see-results-btn');
                
                if (status.is_training) {
                    statusElement.textContent = `Training in progress... (Started: ${new Date(status.start_time).toLocaleTimeString()})`;
                    document.getElementById('current-model').textContent = status.current_model || '-';
                    document.getElementById('current-epoch').textContent = status.epoch || 0;
                    document.getElementById('total-epochs').textContent = status.total_epochs || 0;
                    
                    const progress = status.total_epochs > 0 ? (status.epoch / status.total_epochs) * 100 : 0;
                    document.getElementById('progress-bar').style.width = progress + '%';
                    
                    // Show snake game during training
                    snakeGameCard.style.display = 'block';
                    if (!gameRunning) startSnakeGame();
                    seeResultsBtn.style.display = 'none';
                } else {
                    statusElement.textContent = 'Ready to start training...';
                    document.getElementById('start-training-btn').style.display = 'inline-block';
                    document.getElementById('stop-training-btn').style.display = 'none';
                    document.getElementById('progress-container').style.display = 'none';
                    
                    // Show results button if training completed with results
                    if (Object.keys(status.model_results).length > 0) {
                        seeResultsBtn.style.display = 'inline-block';
                        stopSnakeGame();
                    } else {
                        snakeGameCard.style.display = 'none';
                        stopSnakeGame();
                    }
                }

                // Update logs
                const logsElement = document.getElementById('training-logs');
                if (status.logs && status.logs.length > 0) {
                    logsElement.innerHTML = status.logs.join('<br>');
                    logsElement.scrollTop = logsElement.scrollHeight;
                }

                // Update results
                updateResults(status.model_results);
            });
        }

        function updateResults(results) {
            const container = document.getElementById('results-container');
            
            if (Object.keys(results).length === 0) {
                container.innerHTML = '<p>Results will appear here after training completion...</p>';
                return;
            }

            let html = '<table class="metrics-table"><thead><tr><th>Model</th><th>Status</th><th>Recall</th><th>Specificity</th><th>AUC-ROC</th><th>MCC</th></tr></thead><tbody>';
            
            for (const [modelName, result] of Object.entries(results)) {
                const status = result.completed ? '✅ Completed' : '❌ Failed';
                const metrics = result.best_metrics || {};
                
                html += `<tr>
                    <td><strong>${modelName.toUpperCase()}</strong></td>
                    <td>${status}</td>
                    <td>${(metrics.recall || metrics.sensitivity || 0).toFixed(4)}</td>
                    <td>${(metrics.specificity || 0).toFixed(4)}</td>
                    <td>${(metrics.auc_roc || 0).toFixed(4)}</td>
                    <td>${(metrics.mcc || 0).toFixed(4)}</td>
                </tr>`;
            }
            
            html += '</tbody></table>';
            container.innerHTML = html;
        }

        function updateOptunaResults(trialResults) {
            const container = document.getElementById('optuna-results-table');
            const resultsContainer = document.getElementById('optuna-results-container');
            
            if (!trialResults || trialResults.length === 0) {
                container.innerHTML = '<p>Trial results will appear here during optimization...</p>';
                resultsContainer.style.display = 'none';
                return;
            }

            // Show the results container
            resultsContainer.style.display = 'block';
            
            let html = '<table class="metrics-table"><thead><tr><th>Trial</th><th>Recall</th><th>Specificity</th><th>AUC-ROC</th><th>MCC</th><th>Parameters</th></tr></thead><tbody>';
            
            // Sort trials by recall descending to show best results first (medical priority)
            const sortedTrials = [...trialResults].sort((a, b) => (b.sensitivity || 0) - (a.sensitivity || 0));
            
            for (const trial of sortedTrials) {
                const recall = trial.sensitivity || 0; // Recall is same as sensitivity in medical context
                const isTopResult = trial.sensitivity === Math.max(...trialResults.map(t => t.sensitivity || 0));
                const rowClass = isTopResult ? 'style="background-color: #d4edda; color: #155724;"' : 'style="background-color: #f8f9fa;"';
                
                // Summarize parameters for display
                const paramSummary = trial.parameters ? 
                    Object.entries(trial.parameters)
                        .slice(0, 3) // Show first 3 parameters
                        .map(([key, value]) => `${key}: ${value}`)
                        .join(', ') + (Object.keys(trial.parameters).length > 3 ? '...' : '')
                    : 'N/A';
                
                html += `<tr ${rowClass}>
                    <td><strong>#${trial.trial_id}</strong></td>
                    <td>${recall.toFixed(4)}</td>
                    <td>${(trial.specificity || 0).toFixed(4)}</td>
                    <td>${(trial.auc_roc || 0).toFixed(4)}</td>
                    <td>${(trial.mcc || 0).toFixed(4)}</td>
                    <td title="${JSON.stringify(trial.parameters || {}, null, 2)}" style="max-width: 200px; overflow: hidden; text-overflow: ellipsis; white-space: nowrap; cursor: help;">${paramSummary}</td>
                </tr>`;
            }
            
            html += '</tbody></table>';
            container.innerHTML = html;
        }

        // Snake Game Functions
        function initSnakeGame() {
            const canvas = document.getElementById('snakeCanvas');
            if (!canvas) return;
            
            const ctx = canvas.getContext('2d');
            generateFood();
            
            // Initial draw
            drawSnakeGame(ctx, canvas);
            
            // Add keyboard controls
            document.addEventListener('keydown', changeDirection);
        }

        function startSnakeGame() {
            if (gameRunning) return;
            
            // Reset game state
            snake = [{x: 200, y: 150}];
            dx = 20; dy = 0;
            score = 0;
            gameRunning = true;
            document.getElementById('snake-score').textContent = score;
            
            generateFood();
            gameLoop = setInterval(updateSnakeGame, 150);
        }

        function stopSnakeGame() {
            gameRunning = false;
            if (gameLoop) clearInterval(gameLoop);
        }

        function updateSnakeGame() {
            if (!gameRunning) return;
            
            const canvas = document.getElementById('snakeCanvas');
            const ctx = canvas.getContext('2d');
            
            // Move snake
            let head = {x: snake[0].x + dx, y: snake[0].y + dy};
            
            // Wall wrapping instead of collision
            if (head.x < 0) head.x = canvas.width - 20;
            if (head.x >= canvas.width) head.x = 0;
            if (head.y < 0) head.y = canvas.height - 20;
            if (head.y >= canvas.height) head.y = 0;
            
            // Check self collision
            for (let segment of snake) {
                if (head.x === segment.x && head.y === segment.y) {
                    resetSnakeGame();
                    return;
                }
            }
            
            snake.unshift(head);
            
            // Check food collision with tolerance for any coordinate issues
            const distance = Math.abs(head.x - food.x) + Math.abs(head.y - food.y);
            if (distance < 20) { // If head is within one grid cell of food
                score += 10;
                document.getElementById('snake-score').textContent = score;
                generateFood();
                // Don't remove tail when food is eaten (snake grows)
            } else {
                snake.pop();
            }
            
            drawSnakeGame(ctx, canvas);
        }

        function drawSnakeGame(ctx, canvas) {
            // Clear canvas
            ctx.fillStyle = '#ecf0f1';
            ctx.fillRect(0, 0, canvas.width, canvas.height);
            
            // Draw snake head (different color)
            if (snake.length > 0) {
                ctx.fillStyle = '#34495e';
                ctx.fillRect(snake[0].x, snake[0].y, 18, 18);
                
                // Draw snake body
                ctx.fillStyle = '#2c3e50';
                for (let i = 1; i < snake.length; i++) {
                    ctx.fillRect(snake[i].x, snake[i].y, 18, 18);
                }
            }
            
            // Draw food as a circle for better visibility
            ctx.fillStyle = '#e74c3c';
            ctx.beginPath();
            ctx.arc(food.x + 9, food.y + 9, 8, 0, 2 * Math.PI);
            ctx.fill();
        }

        function generateFood() {
            const canvas = document.getElementById('snakeCanvas');
            if (!canvas) return;
            
            let newFood;
            let attempts = 0;
            
            // Ensure food doesn't spawn on snake
            do {
                newFood = {
                    x: Math.floor(Math.random() * (canvas.width / 20)) * 20,
                    y: Math.floor(Math.random() * (canvas.height / 20)) * 20
                };
                attempts++;
            } while (attempts < 100 && snake.some(segment => segment.x === newFood.x && segment.y === newFood.y));
            
            food.x = newFood.x;
            food.y = newFood.y;
        }

        function changeDirection(event) {
            if (!gameRunning) return;
            
            const LEFT_KEY = 37, RIGHT_KEY = 39, UP_KEY = 38, DOWN_KEY = 40;
            
            // Prevent page scrolling when using arrow keys during game
            if ([LEFT_KEY, RIGHT_KEY, UP_KEY, DOWN_KEY].includes(event.keyCode)) {
                event.preventDefault();
            }
            
            switch(event.keyCode) {
                case LEFT_KEY:
                    if (dx === 0) { dx = -20; dy = 0; }
                    break;
                case UP_KEY:
                    if (dy === 0) { dx = 0; dy = -20; }
                    break;
                case RIGHT_KEY:
                    if (dx === 0) { dx = 20; dy = 0; }
                    break;
                case DOWN_KEY:
                    if (dy === 0) { dx = 0; dy = 20; }
                    break;
            }
        }

        function resetSnakeGame() {
            stopSnakeGame();
            setTimeout(startSnakeGame, 1000); // Restart after 1 second
        }

        function scrollToResults() {
            document.getElementById('results-container').scrollIntoView({ behavior: 'smooth' });
        }

        // Training Mode Functions
        function selectTrainingMode(mode) {
            // Update radio buttons
            document.querySelector('input[name="training-mode"][value="manual"]').checked = (mode === 'manual');
            document.querySelector('input[name="training-mode"][value="optimization"]').checked = (mode === 'optimization');
            
            // Update visual selection
            document.querySelectorAll('.training-mode-label').forEach(label => {
                label.classList.remove('selected');
            });
            
            const selectedLabel = document.querySelector(`input[name="training-mode"][value="${mode}"]`).closest('.training-mode-label');
            selectedLabel.classList.add('selected');
            
            // Trigger the mode toggle
            toggleTrainingMode();
        }
        
        function toggleTrainingMode() {
            const manualMode = document.querySelector('input[name="training-mode"][value="manual"]').checked;
            const optimizationMode = document.querySelector('input[name="training-mode"][value="optimization"]').checked;
            
            currentTrainingMode = manualMode ? 'manual' : 'optimization';
            
            // Update visual selection
            document.querySelectorAll('.training-mode-label').forEach(label => {
                label.classList.remove('selected');
            });
            
            if (manualMode) {
                document.querySelector('input[name="training-mode"][value="manual"]').closest('.training-mode-label').classList.add('selected');
            } else {
                document.querySelector('input[name="training-mode"][value="optimization"]').closest('.training-mode-label').classList.add('selected');
            }
            
            // Toggle control visibility
            document.getElementById('manual-training-controls').style.display = manualMode ? 'block' : 'none';
            document.getElementById('optimization-controls').style.display = optimizationMode ? 'block' : 'none';
            
            // Toggle status panel visibility
            document.getElementById('training-status-panel').style.display = manualMode ? 'block' : 'none';
            document.getElementById('optimization-status-panel').style.display = optimizationMode ? 'block' : 'none';
            
            // Update header
            document.getElementById('status-header').textContent = manualMode ? '📈 Training Status' : '🔍 Optimization Status';
        }

        function startOptimization() {
            const model = document.getElementById('optimization-model').value;
            const preset = document.getElementById('optimization-preset').value;
            const trials = parseInt(document.getElementById('optimization-trials').value);
            
            // Collect data configuration
            const classBalanceValue = parseInt(document.getElementById('class-balance').value);
            const dataConfig = {
                data_percentage: parseFloat(document.getElementById('data-percentage').value) / 100,
                batch_size: parseInt(document.getElementById('batch-size').value),
                class_balance_ratio: classBalanceValue / 100,
                class_balance_enabled: classBalanceValue !== 50
            };

            fetch('/api/start_optimization', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ 
                    model_type: model,
                    optimization_mode: preset,
                    n_trials: trials,
                    data_config: dataConfig
                })
            })
            .then(response => response.json())
            .then(data => {
                if (data.success) {
                    document.getElementById('start-optimization-btn').style.display = 'none';
                    document.getElementById('stop-optimization-btn').style.display = 'inline-block';
                    document.getElementById('optimization-progress-container').style.display = 'block';
                } else {
                    alert('Error starting optimization: ' + data.error);
                }
            })
            .catch(error => {
                console.error('Error:', error);
                alert('Error starting optimization: ' + error.message);
            });
        }

        function stopOptimization() {
            fetch('/api/stop_optimization', { method: 'POST' })
            .then(response => response.json())
            .then(data => {
                document.getElementById('start-optimization-btn').style.display = 'inline-block';
                document.getElementById('stop-optimization-btn').style.display = 'none';
                document.getElementById('optimization-progress-container').style.display = 'none';
            })
            .catch(error => {
                console.error('Error:', error);
            });
        }

        // Update the existing updateStatus function to handle optimization
        function updateStatus() {
            if (currentTrainingMode === 'manual') {
                updateTrainingStatus();
            } else {
                updateOptimizationStatus();
            }
        }

        function updateTrainingStatus() {
            fetch('/api/training_status')
            .then(response => response.json())
            .then(status => {
                // Update status display
                const statusElement = document.getElementById('training-status');
                const snakeGameCard = document.getElementById('snake-game-card');
                const seeResultsBtn = document.getElementById('see-results-btn');
                
                if (status.is_training) {
                    statusElement.textContent = `Training in progress... (Started: ${new Date(status.start_time).toLocaleTimeString()})`;
                    document.getElementById('current-model').textContent = status.current_model || '-';
                    document.getElementById('current-epoch').textContent = status.epoch || 0;
                    document.getElementById('total-epochs').textContent = status.total_epochs || 0;
                    
                    const progress = status.total_epochs > 0 ? (status.epoch / status.total_epochs) * 100 : 0;
                    document.getElementById('progress-bar').style.width = progress + '%';
                    
                    // Show snake game during training
                    snakeGameCard.style.display = 'block';
                    if (!gameRunning) startSnakeGame();
                    seeResultsBtn.style.display = 'none';
                } else {
                    statusElement.textContent = 'Ready to start training...';
                    document.getElementById('start-training-btn').style.display = 'inline-block';
                    document.getElementById('stop-training-btn').style.display = 'none';
                    document.getElementById('progress-container').style.display = 'none';
                    
                    // Show results button if training completed with results
                    if (Object.keys(status.model_results).length > 0) {
                        seeResultsBtn.style.display = 'inline-block';
                        stopSnakeGame();
                    } else {
                        snakeGameCard.style.display = 'none';
                        stopSnakeGame();
                    }
                }

                // Update logs
                const logsElement = document.getElementById('training-logs');
                if (status.logs && status.logs.length > 0) {
                    logsElement.innerHTML = status.logs.join('<br>');
                    logsElement.scrollTop = logsElement.scrollHeight;
                }

                // Update results
                updateResults(status.model_results);
            })
            .catch(error => console.error('Error fetching training status:', error));
        }

        function updateOptimizationStatus() {
            fetch('/api/optimization_status')
            .then(response => response.json())
            .then(status => {
                const statusElement = document.getElementById('optimization-status');
                const snakeGameCard = document.getElementById('snake-game-card');
                const seeResultsBtn = document.getElementById('see-results-btn');
                
                if (status.is_optimizing) {
                    statusElement.textContent = `Optimization in progress... (Started: ${new Date(status.start_time).toLocaleTimeString()})`;
                    document.getElementById('optimization-current-model').textContent = status.current_model || '-';
                    document.getElementById('optimization-current-trial').textContent = status.current_trial || 0;
                    document.getElementById('optimization-total-trials').textContent = status.total_trials || 0;
                    
                    // Calculate best recall from trial results
                    const bestRecall = status.trial_results && status.trial_results.length > 0 ? 
                        Math.max(...status.trial_results.map(t => t.sensitivity || 0)) : 0;
                    document.getElementById('optimization-best-recall').textContent = bestRecall.toFixed(4);
                    document.getElementById('optimization-best-medical').textContent = (status.best_medical_score || 0).toFixed(4);
                    
                    const progress = status.total_trials > 0 ? (status.current_trial / status.total_trials) * 100 : 0;
                    document.getElementById('optimization-progress-bar').style.width = progress + '%';
                    
                    // Show best parameters
                    if (status.best_params && Object.keys(status.best_params).length > 0) {
                        document.getElementById('optimization-best-params').textContent = JSON.stringify(status.best_params, null, 2);
                    }
                    
                    // Update trial results table
                    updateOptunaResults(status.trial_results || []);
                    
                    // Show snake game during optimization
                    snakeGameCard.style.display = 'block';
                    if (!gameRunning) startSnakeGame();
                    seeResultsBtn.style.display = 'none';
                } else {
                    statusElement.textContent = 'Ready to start optimization...';
                    document.getElementById('start-optimization-btn').style.display = 'inline-block';
                    document.getElementById('stop-optimization-btn').style.display = 'none';
                    document.getElementById('optimization-progress-container').style.display = 'none';
                    
                    // Show results if optimization completed
                    if (status.best_medical_score > 0) {
                        seeResultsBtn.style.display = 'inline-block';
                        stopSnakeGame();
                    } else {
                        snakeGameCard.style.display = 'none';
                        stopSnakeGame();
                    }
                }

                // Update logs
                const logsElement = document.getElementById('training-logs');
                if (status.logs && status.logs.length > 0) {
                    logsElement.innerHTML = status.logs.join('<br>');
                    logsElement.scrollTop = logsElement.scrollHeight;
                }
            })
            .catch(error => console.error('Error fetching optimization status:', error));
        }
    </script>
</body>
</html>
    '''
    
    with open(os.path.join(templates_dir, "index.html"), "w") as f:
        f.write(html_template)

if __name__ == '__main__':
    # Get paths relative to project root
    import os
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(script_dir)  # Parent of src directory
    
    # Create templates directory in src (before changing working directory)
    templates_dir = os.path.join(script_dir, 'templates')
    os.makedirs(templates_dir, exist_ok=True)
    create_templates_at_path(templates_dir)
    
    # Update Flask app template folder to absolute path
    app.template_folder = templates_dir
    
    # Set working directory to project root for data access
    os.chdir(project_root)
    
    # Load configuration
    config_path = 'config.yaml'  # Now relative to project root
    try:
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
    except FileNotFoundError:
        config = {}
    
    web_config = config.get('web', {})
    
    print("🔬 Breast Cancer Detection Training Interface")
    print("=" * 50)
    print("Starting web application...")
    print(f"Open your browser to: http://localhost:{web_config.get('port', 8080)}")
    print("\nFeatures:")
    print("  - Model selection (CNN, ResNet, Vision Transformer)")
    print("  - Real-time training monitoring")
    print("  - Medical metrics visualization")
    print("  - Dataset information display")
    print("\nPress Ctrl+C to stop the server")
    print("=" * 50)
    
    app.run(
        host=web_config.get('host', '0.0.0.0'),
        port=web_config.get('port', 8080),
        debug=web_config.get('debug', False)
    )