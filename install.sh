#!/bin/bash

# Breast Cancer Detection Project Setup Script
# This script installs dependencies, sets up the environment, and prepares the project structure

set -e  # Exit on any error

echo "=================================="
echo "Breast Cancer Detection Setup"
echo "=================================="

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Function to print colored output
print_status() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

print_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"  
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Check Python version
print_status "Checking Python version..."
python_version=$(python3 -c 'import sys; print(".".join(map(str, sys.version_info[:2])))')
required_version="3.8"

if python3 -c "import sys; exit(0 if sys.version_info >= (3, 8) else 1)"; then
    print_success "Python $python_version is compatible"
else
    print_error "Python $python_version is too old. Please install Python 3.8+"
    exit 1
fi

# Create virtual environment if it doesn't exist
if [ ! -d "venv" ]; then
    print_status "Creating virtual environment..."
    python3 -m venv venv
    print_success "Virtual environment created"
else
    print_status "Virtual environment already exists"
fi

# Activate virtual environment
print_status "Activating virtual environment..."
source venv/bin/activate

# Upgrade pip
print_status "Upgrading pip..."
pip install --upgrade pip

# Install PyTorch first (with CUDA support if available)
print_status "Installing PyTorch..."
if command -v nvidia-smi &> /dev/null; then
    print_status "NVIDIA GPU detected, installing PyTorch with CUDA support..."
    pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
else
    print_status "No GPU detected, installing CPU-only PyTorch..."
    pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
fi

# Install other requirements
print_status "Installing Python dependencies..."
pip install -r requirements.txt

# Install Detectron2 separately (optional for Faster R-CNN)
print_status "Installing Detectron2 for Faster R-CNN support..."
if python3 -c "import torch; print('CUDA available:', torch.cuda.is_available())" 2>/dev/null | grep -q "True"; then
    print_status "Installing Detectron2 with CUDA support..."
    pip install 'git+https://github.com/facebookresearch/detectron2.git' || print_status "⚠️ Detectron2 installation failed - Faster R-CNN will be disabled"
else
    print_status "Installing Detectron2 for CPU..."
    pip install 'git+https://github.com/facebookresearch/detectron2.git' || print_status "⚠️ Detectron2 installation failed - Faster R-CNN will be disabled"
fi

# Create project directories if they don't exist
print_status "Setting up project structure..."
mkdir -p src/{models,data_loaders,utils,training}
mkdir -p models/{saved_models,checkpoints}
mkdir -p notebooks/{eda,modelling,evaluation}
mkdir -p logs
mkdir -p results/{plots,metrics,reports}

print_success "Project directories created"

# Create __init__.py files for Python modules
touch src/__init__.py
touch src/models/__init__.py
touch src/data_loaders/__init__.py
touch src/utils/__init__.py
touch src/training/__init__.py

# Set executable permissions for Python scripts
chmod +x src/preprocess_images.py

# Create basic config file if it doesn't exist
if [ ! -f "config.yaml" ]; then
    print_status "Creating default configuration file..."
    cat > config.yaml << EOF
# Breast Cancer Detection Configuration

# Data Configuration
data:
  data_path: "data/BHI"
  image_size: [50, 50]
  batch_size: 32
  data_percentage: 1.0  # Use full dataset (0.1 for 10% during development)
  num_workers: 4
  
# Class Balancing
class_balance:
  enabled: true
  target_ratio: 0.5  # 50-50 split
  augmentation_ratio: 0.5  # 50% augmentation, 50% reduction
  
# Data Augmentation
augmentation:
  enabled: true
  rotation_range: 15
  brightness_range: [0.8, 1.2] 
  contrast_range: [0.8, 1.2]
  horizontal_flip: true
  vertical_flip: false  # Not anatomically appropriate
  gaussian_noise: 0.01

# Model Configuration
models:
  cnn:
    enabled: true
    evaluate: 1
    architecture: "custom"
    learning_rate: 0.001
    epochs: 50
    
  resnet:
    enabled: true  
    evaluate: 1
    architecture: "resnet18"
    pretrained: true
    learning_rate: 0.0001
    epochs: 30
    
  faster_rcnn:
    enabled: true
    evaluate: 1
    backbone: "resnet50"
    learning_rate: 0.0001
    epochs: 25
    
  vision_transformer:
    enabled: true
    evaluate: 1
    model_name: "vit_base_patch16_224"
    learning_rate: 0.00001
    epochs: 20

# Training Configuration  
training:
  device: "auto"  # auto, cuda, cpu
  mixed_precision: true
  gradient_clipping: 1.0
  patience: 15  # Early stopping patience
  
# Evaluation Metrics (Medical Focus)
metrics:
  primary: ["sensitivity", "specificity", "f1_score"]
  secondary: ["accuracy", "precision", "recall", "auc_roc"]
  
# Hyperparameter Optimization
optuna:
  enabled: false  # Enable after selecting best model
  n_trials: 100
  optimize_metric: "f1_score"
  
# Logging and Output
logging:
  level: "INFO"
  save_logs: true
  log_file: "logs/training.log"
  
output:
  save_models: true
  save_plots: true
  results_dir: "results"
EOF
    print_success "Default configuration created"
fi

# Run automatic image preprocessing
print_status "Running automatic image preprocessing..."
if [ -d "data/BHI" ]; then
    python3 -c "
import sys
sys.path.append('src')
from preprocess_images import check_image_dimensions, find_non_standard_images, resize_non_standard_images
import os

data_path = 'data/BHI'
target_size = (50, 50)

print('Checking image dimensions...')
dimension_counts = check_image_dimensions(data_path, sample_size=20)

print('Finding non-standard images...')
non_standard_images = find_non_standard_images(data_path, target_size)

if non_standard_images:
    print(f'Auto-resizing {len(non_standard_images)} images to 50x50...')
    resize_non_standard_images(non_standard_images, target_size, method='resize')
    
    # Verify results
    remaining_non_standard = find_non_standard_images(data_path, target_size)
    if not remaining_non_standard:
        print('✅ All images are now 50x50 pixels!')
        
        # Document the preprocessing in README
        with open('PREPROCESSING_LOG.md', 'w') as f:
            f.write(f'''# Image Preprocessing Log

## Summary
- Total non-standard images found: {len(non_standard_images)}
- All images automatically resized to 50x50 pixels during installation
- Method used: Simple resize (may alter aspect ratio slightly)
- Original dimension distribution: {dict(dimension_counts)}

## Preprocessing Details
- Target size: 50x50 pixels  
- Resize method: LANCZOS resampling
- All images successfully converted
- No manual intervention required

## Impact on Model Training
- Consistent input dimensions for all models
- Preprocessing maintains image content while standardizing size
- Medical features preserved through high-quality resampling

Generated during installation on: $(date)
''')
        print('📝 Preprocessing log saved to PREPROCESSING_LOG.md')
    else:
        print(f'⚠️ {len(remaining_non_standard)} images still need attention.')
else:
    print('✅ All images are already 50x50 pixels!')
" || print_status "Preprocessing completed with warnings - check manually if needed"
else
    print_status "BHI dataset not found in data/ directory"  
    print_status "Place dataset in data/BHI/ and run: python3 src/preprocess_images.py"
fi

# Check GPU availability
print_status "Checking GPU availability..."
python3 -c "
import torch
if torch.cuda.is_available():
    print(f'GPU available: {torch.cuda.get_device_name(0)}')
    print(f'CUDA version: {torch.version.cuda}')
else:
    print('No GPU available, using CPU')
"

# Create startup scripts
print_status "Creating utility scripts..."

# Create run_preprocessing.sh
cat > run_preprocessing.sh << 'EOF'
#!/bin/bash
source venv/bin/activate
python3 src/preprocess_images.py
EOF
chmod +x run_preprocessing.sh

# Create run_training.sh  
cat > run_training.sh << 'EOF'
#!/bin/bash
source venv/bin/activate
python3 src/app.py
EOF
chmod +x run_training.sh

# Create jupyter_start.sh
cat > jupyter_start.sh << 'EOF'
#!/bin/bash
source venv/bin/activate
jupyter notebook --ip=0.0.0.0 --port=8888 --no-browser
EOF
chmod +x jupyter_start.sh

print_success "Utility scripts created"

# Final setup verification
print_status "Verifying installation..."
python3 -c "
import torch
import torchvision
import sklearn
import PIL
import cv2
print('✅ Core libraries imported successfully')

try:
    import transformers
    print('✅ Transformers available for Vision Transformer')
except ImportError:
    print('⚠️ Transformers not available')

try:
    import optuna
    print('✅ Optuna available for hyperparameter optimization')  
except ImportError:
    print('⚠️ Optuna not available')

try:
    import detectron2
    print('✅ Detectron2 available for Faster R-CNN')
except ImportError:
    print('⚠️ Detectron2 not available - Faster R-CNN will be disabled')
"

echo ""
echo "=================================="
echo "Setup Complete!"
echo "=================================="
print_success "Environment successfully configured"

echo ""
echo "Next steps:"
echo "1. Activate environment: source venv/bin/activate"
echo "2. Place BHI dataset in data/BHI/"
echo "3. Run preprocessing: ./run_preprocessing.sh"
echo "4. Start training interface: ./run_training.sh"
echo "5. Or start Jupyter: ./jupyter_start.sh"
echo ""
echo "Configuration file: config.yaml"
echo "Logs directory: logs/"
echo "Results directory: results/"
echo ""

print_status "Setup script completed successfully!"