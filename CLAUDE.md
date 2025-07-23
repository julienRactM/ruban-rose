# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview
Breast cancer detection system using deep learning models on histopathological images. The project implements and compares multiple architectures (CNN, ResNet, Faster R-CNN, Vision Transformer) for binary classification of breast tissue images.

## Data Structure
- `data/BHI/`: Breast Histopathology Images dataset with patient folders containing binary classification subdirectories
  - Each patient folder (e.g., `10253/`) contains:
    - `0/`: Cancerous tissue images (class 0)  
    - `1/`: Healthy tissue images (class 1)
  - Images are PNG format with naming pattern: `{patient_id}_idx5_x{x_coord}_y{y_coord}_class{label}.png`
  - Target image size: 50x50 pixels

## Architecture
```
├── data/BHI/                    # Dataset (binary classification images)
├── src/                         # Source code modules
├── models/                      # Trained model artifacts and definitions
├── notebooks/                   # Jupyter notebooks for EDA and experiments
│   ├── eda/                    # Exploratory data analysis
│   └── modelling/              # Model development notebooks  
├── config.yaml                 # Configuration parameters
├── requirements.txt            # Python dependencies
└── install.sh                 # Setup and installation script
```

## Key Development Commands
- **Setup Environment**: `./install.sh` - Installs dependencies and prepares environment
- **Data Preprocessing**: `python src/preprocess_images.py` - Validates and resizes images to 50x50
- **Start Training Interface**: `python src/app.py` - Launches web interface for model training
- **Run Model Training**: Configure via web interface or `config.yaml` parameters
- **Model Evaluation**: Automated evaluation runs based on config `evaluate` flags

## Configuration Parameters
The `config.yaml` file controls:
- `data_percentage`: Percentage of dataset to load (for development/testing)
- `batch_size`: Training batch size
- `models`: Individual model configurations with `evaluate: 0/1` flags
- `class_balance`: Ensures 50/50 class distribution through augmentation/reduction

## Models Implemented
1. **CNN**: Custom convolutional neural network
2. **ResNet**: Residual network architecture  
3. **Faster R-CNN**: Region-based CNN for object detection approach
4. **Vision Transformer**: Transformer-based vision model

## Medical Evaluation Metrics
The project uses three specialized metrics for medical imaging (detailed in `metrics.md`):
- Sensitivity/Recall (minimize false negatives)
- Specificity (minimize false positives) 
- F1-Score (balanced precision-recall for imbalanced medical data)

## Data Loading Strategy
- Custom PyTorch DataLoader handles class imbalancing
- 50% data augmentation for minority class
- 50% data reduction for majority class
- Configurable dataset percentage for development

## Hyperparameter Optimization
- Optuna integration for automated hyperparameter tuning
- Optimizes best-performing model architecture
- Medical-focused objective functions

## Development Workflow
1. Run `./install.sh` to set up environment
2. Use `src/preprocess_images.py` to validate image dimensions
3. Configure training parameters in `config.yaml`
4. Launch training via web interface: `python src/app.py`
5. Monitor training and evaluation through configured metrics
6. Best model selection based on medical evaluation criteria

## Important Notes
- All images must be exactly 50x50 pixels for model compatibility
- Class labels: 0 = cancerous, 1 = healthy tissue
- Medical imaging requires careful attention to false negative rates
- Data augmentation strategies account for medical image characteristics