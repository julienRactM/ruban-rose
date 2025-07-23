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
from datetime import datetime

# Import model factories
from models.cnn_model import create_cnn_model
from models.resnet_model import create_resnet_model
from models.vision_transformer import create_vit_model
from data_loaders.breast_cancer_dataloader import create_dataloaders
from training.trainer import BreastCancerTrainer

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

# Load configuration
def load_config():
    """Load configuration from YAML file"""
    try:
        with open('config.yaml', 'r') as f:
            return yaml.safe_load(f)
    except FileNotFoundError:
        return {}

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
        
        # Apply custom parameters to config
        if model_parameters:
            for model_name, params in model_parameters.items():
                if model_name in config.get('models', {}):
                    config['models'][model_name].update(params)
                    add_log(f"Applied custom parameters for {model_name}: {params}")
        
        # Use MPS for M4 Pro, fallback to CPU
        if torch.backends.mps.is_available():
            device = torch.device('mps')
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
                    training_status.update({
                        'epoch': epoch,
                        'total_epochs': total_epochs,
                        'metrics': {
                            'train_loss': train_loss,
                            'val_loss': val_loss,
                            'f1_score': metrics.get('f1_score', 0),
                            'sensitivity': metrics.get('sensitivity', 0),
                            'specificity': metrics.get('specificity', 0),
                            'accuracy': metrics.get('accuracy', 0)
                        }
                    })
                
                # Train model with status callback
                history = trainer.train(status_callback=update_training_status)
                
                # Store results
                if history['val_metrics']:
                    best_metrics = max(history['val_metrics'], key=lambda x: x['f1_score'])
                    training_status['model_results'][model_name] = {
                        'completed': True,
                        'best_metrics': best_metrics,
                        'training_time': time.time(),
                        'epochs_completed': len(history['val_metrics'])
                    }
                    
                    add_log(f"{model_name} training completed!")
                    add_log(f"Best F1-Score: {best_metrics['f1_score']:.4f}")
                    add_log(f"Best Sensitivity: {best_metrics['sensitivity']:.4f}")
                
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
        .metrics-table th { background: #f8f9fa; }
        .progress { background: #ecf0f1; border-radius: 10px; height: 20px; margin: 0.5rem 0; }
        .progress-bar { background: #3498db; height: 100%; border-radius: 10px; transition: width 0.3s; }
        .dataset-info { display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 1rem; }
        .stat-box { text-align: center; padding: 1rem; background: #ecf0f1; border-radius: 8px; }
        .stat-number { font-size: 2rem; font-weight: bold; color: #2c3e50; }
        .stat-label { color: #7f8c8d; margin-top: 0.5rem; }
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
                    </div>
                </div>
            </div>
            
            <div style="margin-top: 1rem; text-align: center;">
                <button class="btn" id="start-training-btn" onclick="startTraining()">
                    🚀 Start Training Selected Models
                </button>
                <button class="btn btn-danger" id="stop-training-btn" onclick="stopTraining()" style="display: none;">
                    ⏹️ Stop Training
                </button>
            </div>
        </div>
        
        <!-- Training Status -->
        <div class="card">
            <h2>📈 Training Status</h2>
            <div class="status-panel">
                <div id="training-status">Ready to start training...</div>
                <div id="progress-container" style="display: none;">
                    <div>Current Model: <span id="current-model">-</span></div>
                    <div>Epoch: <span id="current-epoch">0</span>/<span id="total-epochs">0</span></div>
                    <div class="progress">
                        <div class="progress-bar" id="progress-bar" style="width: 0%;"></div>
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

        // Load dataset info on page load
        window.onload = function() {
            loadDatasetInfo();
            startStatusUpdates();
        };

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
                class_balance_enabled: classBalanceValue !== 50 // Enable balancing if not 50/50
            };

            // Collect parameters for each selected model
            const modelParams = {};
            
            selectedModels.forEach(modelName => {
                if (modelName === 'cnn') {
                    modelParams.cnn = {
                        epochs: parseInt(document.getElementById('cnn-epochs').value),
                        learning_rate: parseFloat(document.getElementById('cnn-lr').value),
                        dropout: parseFloat(document.getElementById('cnn-dropout').value),
                        architecture: document.getElementById('cnn-architecture').value
                    };
                } else if (modelName === 'resnet') {
                    modelParams.resnet = {
                        epochs: parseInt(document.getElementById('resnet-epochs').value),
                        learning_rate: parseFloat(document.getElementById('resnet-lr').value),
                        architecture: document.getElementById('resnet-architecture').value,
                        pretrained: document.getElementById('resnet-pretrained').value === 'true',
                        fine_tune_layers: parseInt(document.getElementById('resnet-fine-tune').value)
                    };
                } else if (modelName === 'vision_transformer') {
                    modelParams.vision_transformer = {
                        epochs: parseInt(document.getElementById('vit-epochs').value),
                        learning_rate: parseFloat(document.getElementById('vit-lr').value),
                        model_name: document.getElementById('vit-model').value,
                        dropout_rate: parseFloat(document.getElementById('vit-dropout').value),
                        patch_size: parseInt(document.getElementById('vit-patch-size').value)
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
                if (status.is_training) {
                    statusElement.textContent = `Training in progress... (Started: ${new Date(status.start_time).toLocaleTimeString()})`;
                    document.getElementById('current-model').textContent = status.current_model || '-';
                    document.getElementById('current-epoch').textContent = status.epoch || 0;
                    document.getElementById('total-epochs').textContent = status.total_epochs || 0;
                    
                    const progress = status.total_epochs > 0 ? (status.epoch / status.total_epochs) * 100 : 0;
                    document.getElementById('progress-bar').style.width = progress + '%';
                } else {
                    statusElement.textContent = 'Ready to start training...';
                    document.getElementById('start-training-btn').style.display = 'inline-block';
                    document.getElementById('stop-training-btn').style.display = 'none';
                    document.getElementById('progress-container').style.display = 'none';
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

            let html = '<table class="metrics-table"><thead><tr><th>Model</th><th>Status</th><th>F1-Score</th><th>Sensitivity</th><th>Specificity</th><th>Accuracy</th></tr></thead><tbody>';
            
            for (const [modelName, result] of Object.entries(results)) {
                const status = result.completed ? '✅ Completed' : '❌ Failed';
                const metrics = result.best_metrics || {};
                
                html += `<tr>
                    <td><strong>${modelName.toUpperCase()}</strong></td>
                    <td>${status}</td>
                    <td>${(metrics.f1_score || 0).toFixed(4)}</td>
                    <td>${(metrics.sensitivity || 0).toFixed(4)}</td>
                    <td>${(metrics.specificity || 0).toFixed(4)}</td>
                    <td>${(metrics.accuracy || 0).toFixed(4)}</td>
                </tr>`;
            }
            
            html += '</tbody></table>';
            container.innerHTML = html;
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