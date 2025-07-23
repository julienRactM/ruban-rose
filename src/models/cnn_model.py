"""
Custom CNN Model for Breast Cancer Detection
Optimized for 50x50 histopathology images
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional


class BreastCancerCNN(nn.Module):
    """
    Custom CNN for breast cancer classification from histopathology images
    
    Architecture designed for 50x50 input images with medical imaging considerations:
    - Progressive feature extraction with appropriate pooling
    - Dropout for regularization to prevent overfitting on medical data
    - Batch normalization for stable training
    - Medical-appropriate activation functions
    """
    
    def __init__(
        self,
        input_channels: int = 3,
        num_classes: int = 2,
        dropout_rate: float = 0.5,
        config: Optional[Dict] = None
    ):
        """
        Initialize CNN model
        
        Args:
            input_channels: Number of input channels (3 for RGB)
            num_classes: Number of output classes (2 for binary)
            dropout_rate: Dropout probability
            config: Additional configuration parameters
        """
        super(BreastCancerCNN, self).__init__()
        
        self.config = config or {}
        self.num_classes = num_classes
        self.dropout_rate = dropout_rate
        
        # Convolutional layers - Progressive feature extraction
        # Input: 3 x 50 x 50
        
        # First block - Basic feature detection
        self.conv1 = nn.Conv2d(input_channels, 32, kernel_size=3, padding=1)  # 32 x 50 x 50
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 32, kernel_size=3, padding=1)  # 32 x 50 x 50
        self.bn2 = nn.BatchNorm2d(32)
        self.pool1 = nn.MaxPool2d(2, 2)  # 32 x 25 x 25
        
        # Second block - Pattern recognition
        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, padding=1)  # 64 x 25 x 25
        self.bn3 = nn.BatchNorm2d(64)
        self.conv4 = nn.Conv2d(64, 64, kernel_size=3, padding=1)  # 64 x 25 x 25
        self.bn4 = nn.BatchNorm2d(64)
        self.pool2 = nn.MaxPool2d(2, 2)  # 64 x 12 x 12
        
        # Third block - Complex features
        self.conv5 = nn.Conv2d(64, 128, kernel_size=3, padding=1)  # 128 x 12 x 12
        self.bn5 = nn.BatchNorm2d(128)
        self.conv6 = nn.Conv2d(128, 128, kernel_size=3, padding=1)  # 128 x 12 x 12
        self.bn6 = nn.BatchNorm2d(128)
        self.pool3 = nn.MaxPool2d(2, 2)  # 128 x 6 x 6
        
        # Fourth block - High-level features
        self.conv7 = nn.Conv2d(128, 256, kernel_size=3, padding=1)  # 256 x 6 x 6
        self.bn7 = nn.BatchNorm2d(256)
        self.conv8 = nn.Conv2d(256, 256, kernel_size=3, padding=1)  # 256 x 6 x 6
        self.bn8 = nn.BatchNorm2d(256)
        self.pool4 = nn.MaxPool2d(2, 2)  # 256 x 3 x 3
        
        # Global Average Pooling to reduce parameters
        self.global_avg_pool = nn.AdaptiveAvgPool2d(1)  # 256 x 1 x 1
        
        # Classifier head
        self.dropout = nn.Dropout(dropout_rate)
        self.fc1 = nn.Linear(256, 128)
        self.bn_fc1 = nn.BatchNorm1d(128)
        self.dropout2 = nn.Dropout(dropout_rate * 0.5)
        self.fc2 = nn.Linear(128, 64)
        self.bn_fc2 = nn.BatchNorm1d(64)
        self.fc_out = nn.Linear(64, num_classes)
        
        # Initialize weights
        self._initialize_weights()
    
    def _initialize_weights(self):
        """Initialize weights using medical imaging best practices"""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass
        
        Args:
            x: Input tensor of shape (batch_size, 3, 50, 50)
            
        Returns:
            Output logits of shape (batch_size, num_classes)
        """
        # First block
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = self.pool1(x)
        
        # Second block  
        x = F.relu(self.bn3(self.conv3(x)))
        x = F.relu(self.bn4(self.conv4(x)))
        x = self.pool2(x)
        
        # Third block
        x = F.relu(self.bn5(self.conv5(x)))
        x = F.relu(self.bn6(self.conv6(x)))
        x = self.pool3(x)
        
        # Fourth block
        x = F.relu(self.bn7(self.conv7(x)))
        x = F.relu(self.bn8(self.conv8(x)))
        x = self.pool4(x)
        
        # Global average pooling
        x = self.global_avg_pool(x)
        x = x.view(x.size(0), -1)  # Flatten
        
        # Classifier
        x = self.dropout(x)
        x = F.relu(self.bn_fc1(self.fc1(x)))
        x = self.dropout2(x)
        x = F.relu(self.bn_fc2(self.fc2(x)))
        x = self.fc_out(x)
        
        return x
    
    def get_feature_maps(self, x: torch.Tensor, layer_name: str = 'conv8') -> torch.Tensor:
        """
        Extract feature maps from specified layer for visualization
        
        Args:
            x: Input tensor
            layer_name: Name of layer to extract features from
            
        Returns:
            Feature maps from specified layer
        """
        # Forward pass until specified layer
        x = F.relu(self.bn1(self.conv1(x)))
        if layer_name == 'conv1': return x
        
        x = F.relu(self.bn2(self.conv2(x)))
        if layer_name == 'conv2': return x
        x = self.pool1(x)
        
        x = F.relu(self.bn3(self.conv3(x)))
        if layer_name == 'conv3': return x
        
        x = F.relu(self.bn4(self.conv4(x)))
        if layer_name == 'conv4': return x
        x = self.pool2(x)
        
        x = F.relu(self.bn5(self.conv5(x)))
        if layer_name == 'conv5': return x
        
        x = F.relu(self.bn6(self.conv6(x)))
        if layer_name == 'conv6': return x
        x = self.pool3(x)
        
        x = F.relu(self.bn7(self.conv7(x)))
        if layer_name == 'conv7': return x
        
        x = F.relu(self.bn8(self.conv8(x)))
        if layer_name == 'conv8': return x
        
        return x
    
    def count_parameters(self) -> int:
        """Count trainable parameters"""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
    
    def get_model_info(self) -> Dict:
        """Get model information"""
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        
        return {
            'model_name': 'BreastCancerCNN',
            'total_parameters': total_params,
            'trainable_parameters': trainable_params,
            'model_size_mb': total_params * 4 / (1024 ** 2),  # Assuming float32
            'input_shape': (3, 50, 50),
            'output_classes': self.num_classes,
            'dropout_rate': self.dropout_rate
        }


class CompactCNN(nn.Module):
    """
    Compact CNN variant for faster training and inference
    Reduced parameters while maintaining performance
    """
    
    def __init__(
        self,
        input_channels: int = 3,
        num_classes: int = 2,
        dropout_rate: float = 0.3
    ):
        super(CompactCNN, self).__init__()
        
        # Streamlined architecture
        self.conv1 = nn.Conv2d(input_channels, 32, 3, padding=1)  # 32 x 50 x 50
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1, stride=2)   # 64 x 25 x 25
        self.conv3 = nn.Conv2d(64, 128, 3, padding=1, stride=2)  # 128 x 12 x 12
        self.conv4 = nn.Conv2d(128, 256, 3, padding=1, stride=2) # 256 x 6 x 6
        
        self.bn1 = nn.BatchNorm2d(32)
        self.bn2 = nn.BatchNorm2d(64)
        self.bn3 = nn.BatchNorm2d(128)
        self.bn4 = nn.BatchNorm2d(256)
        
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        self.dropout = nn.Dropout(dropout_rate)
        self.classifier = nn.Linear(256, num_classes)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = F.relu(self.bn4(self.conv4(x)))
        
        x = self.global_pool(x)
        x = x.view(x.size(0), -1)
        x = self.dropout(x)
        x = self.classifier(x)
        
        return x


def create_cnn_model(config: Dict) -> nn.Module:
    """
    Factory function to create CNN model based on configuration
    
    Args:
        config: Configuration dictionary
        
    Returns:
        CNN model instance
    """
    cnn_config = config.get('models', {}).get('cnn', {})
    
    architecture = cnn_config.get('architecture', 'custom')
    dropout = cnn_config.get('dropout', 0.5)
    
    if architecture == 'custom':
        model = BreastCancerCNN(
            input_channels=3,
            num_classes=2,
            dropout_rate=dropout,
            config=cnn_config
        )
    elif architecture == 'compact':
        model = CompactCNN(
            input_channels=3,
            num_classes=2,
            dropout_rate=dropout
        )
    else:
        raise ValueError(f"Unknown CNN architecture: {architecture}")
    
    return model


def test_cnn_model():
    """Test the CNN model implementation"""
    print("Testing CNN Model...")
    
    # Create model
    model = BreastCancerCNN()
    print(f"Created model: {model.__class__.__name__}")
    
    # Test forward pass
    batch_size = 4
    x = torch.randn(batch_size, 3, 50, 50)
    
    model.eval()
    with torch.no_grad():
        output = model(x)
    
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {output.shape}")
    print(f"Output range: [{output.min():.3f}, {output.max():.3f}]")
    
    # Test model info
    info = model.get_model_info()
    print("\nModel Information:")
    for key, value in info.items():
        print(f"  {key}: {value}")
    
    # Test feature extraction
    feature_maps = model.get_feature_maps(x, 'conv4')
    print(f"\nFeature maps shape (conv4): {feature_maps.shape}")
    
    # Test compact model
    compact_model = CompactCNN()
    compact_output = compact_model(x)
    print(f"\nCompact model output shape: {compact_output.shape}")
    print(f"Compact model parameters: {sum(p.numel() for p in compact_model.parameters()):,}")
    
    print("✅ CNN model test completed successfully!")


if __name__ == "__main__":
    test_cnn_model()