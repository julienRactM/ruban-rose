"""
DenseNet Model Implementation for Breast Cancer Detection
Uses pre-trained DenseNet with medical imaging adaptations
"""

import torch
import torch.nn as nn
import torchvision.models as models
from typing import Dict, Optional
import torch.nn.functional as F


class MedicalDenseNet(nn.Module):
    """
    DenseNet adapted for medical imaging with breast cancer-specific modifications
    """
    
    def __init__(
        self,
        architecture: str = 'densenet121',
        num_classes: int = 2,
        pretrained: bool = True,
        growth_rate: int = 32,
        compression: float = 0.5,
        dropout_rate: float = 0.3,
        config: Optional[Dict] = None
    ):
        """
        Initialize Medical DenseNet
        
        Args:
            architecture: DenseNet variant ('densenet121', 'densenet169', 'densenet201')
            num_classes: Number of output classes
            pretrained: Use ImageNet pre-trained weights
            growth_rate: Growth rate for DenseNet layers
            compression: Compression factor for transition layers
            dropout_rate: Dropout rate before final classifier
            config: Additional configuration
        """
        super(MedicalDenseNet, self).__init__()
        
        self.config = config or {}
        self.architecture = architecture
        self.num_classes = num_classes
        self.growth_rate = growth_rate
        self.compression = compression
        
        # Load base DenseNet model with error handling
        try:
            if architecture == 'densenet121':
                self.backbone = models.densenet121(pretrained=pretrained)
                feature_dim = 1024
            elif architecture == 'densenet169':
                self.backbone = models.densenet169(pretrained=pretrained)
                feature_dim = 1664
            elif architecture == 'densenet201':
                self.backbone = models.densenet201(pretrained=pretrained)
                feature_dim = 1920
            elif architecture == 'densenet161':
                self.backbone = models.densenet161(pretrained=pretrained)
                feature_dim = 2208
            else:
                valid_archs = ['densenet121', 'densenet169', 'densenet201', 'densenet161']
                raise ValueError(f"Unsupported architecture '{architecture}'. Valid options: {valid_archs}")
        except Exception as e:
            print(f"Error loading DenseNet backbone '{architecture}' with pretrained={pretrained}: {str(e)}")
            raise
        
        # Modify input layer for 50x50 images if needed
        # Standard ImageNet input is 224x224, we have 50x50
        # Keep original conv0 but adapt stride and padding
        original_conv0 = self.backbone.features.conv0
        self.backbone.features.conv0 = nn.Conv2d(
            3, 64, kernel_size=7, stride=1, padding=3, bias=False
        )
        
        # Initialize new conv0 with pre-trained weights if available
        if pretrained:
            with torch.no_grad():
                self.backbone.features.conv0.weight.copy_(original_conv0.weight)
        
        # Replace classifier with medical-specific head
        self.backbone.classifier = nn.Identity()  # Remove original classifier
        
        # Medical-specific classifier head with progressive dimensionality reduction
        self.classifier = nn.Sequential(
            nn.BatchNorm1d(feature_dim),
            nn.Dropout(dropout_rate),
            nn.Linear(feature_dim, feature_dim // 2),
            nn.BatchNorm1d(feature_dim // 2),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate * 0.5),
            nn.Linear(feature_dim // 2, feature_dim // 4),
            nn.BatchNorm1d(feature_dim // 4),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate * 0.25),
            nn.Linear(feature_dim // 4, num_classes)
        )
        
        # Initialize classifier weights
        self._initialize_classifier()
    
    def _initialize_classifier(self):
        """Initialize classifier weights using medical imaging best practices"""
        for module in self.classifier.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_normal_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
            elif isinstance(module, nn.BatchNorm1d):
                nn.init.constant_(module.weight, 1)
                nn.init.constant_(module.bias, 0)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass
        
        Args:
            x: Input tensor of shape (batch_size, 3, 50, 50)
            
        Returns:
            Output logits of shape (batch_size, num_classes)
        """
        # Extract features using DenseNet backbone
        features = self.backbone.features(x)
        
        # Global average pooling
        features = F.relu(features, inplace=True)
        features = F.adaptive_avg_pool2d(features, (1, 1))
        features = torch.flatten(features, 1)
        
        # Medical classifier
        output = self.classifier(features)
        
        return output
    
    def get_feature_maps(self, x: torch.Tensor, layer_name: str = 'denseblock4') -> torch.Tensor:
        """
        Extract feature maps from specified layer for visualization
        
        Args:
            x: Input tensor
            layer_name: Name of layer to extract features from
            
        Returns:
            Feature maps from specified layer
        """
        # Forward pass through backbone features
        features = x
        for name, layer in self.backbone.features.named_children():
            features = layer(features)
            if name == layer_name:
                return features
        
        return features
    
    def count_parameters(self) -> Dict[str, int]:
        """Count model parameters"""
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        
        backbone_params = sum(p.numel() for p in self.backbone.parameters())
        classifier_params = sum(p.numel() for p in self.classifier.parameters())
        
        return {
            'total_parameters': total_params,
            'trainable_parameters': trainable_params,
            'backbone_parameters': backbone_params,
            'classifier_parameters': classifier_params
        }
    
    def get_model_info(self) -> Dict:
        """Get comprehensive model information"""
        param_counts = self.count_parameters()
        
        return {
            'model_name': f'Medical{self.architecture.upper()}',
            'architecture': self.architecture,
            'input_shape': (3, 50, 50),
            'output_classes': self.num_classes,
            'growth_rate': self.growth_rate,
            'compression': self.compression,
            **param_counts,
            'model_size_mb': param_counts['total_parameters'] * 4 / (1024 ** 2)
        }


class CompactDenseNet(nn.Module):
    """
    Compact DenseNet for faster training and inference
    Custom implementation optimized for 50x50 images
    """
    
    def __init__(
        self,
        num_classes: int = 2,
        growth_rate: int = 16,
        block_config: tuple = (6, 12, 8),
        num_init_features: int = 32,
        dropout_rate: float = 0.2
    ):
        super(CompactDenseNet, self).__init__()
        
        self.num_classes = num_classes
        self.growth_rate = growth_rate
        
        # Initial convolution - adapted for small images
        self.features = nn.Sequential()
        self.features.add_module('conv0', nn.Conv2d(3, num_init_features, kernel_size=3, stride=1, padding=1, bias=False))
        self.features.add_module('norm0', nn.BatchNorm2d(num_init_features))
        self.features.add_module('relu0', nn.ReLU(inplace=True))
        
        # Dense blocks
        num_features = num_init_features
        for i, num_layers in enumerate(block_config):
            block = self._make_dense_block(num_layers, num_features, growth_rate, dropout_rate)
            self.features.add_module(f'denseblock{i+1}', block)
            num_features += num_layers * growth_rate
            
            # Transition layers (except after the last block)
            if i != len(block_config) - 1:
                trans = self._make_transition(num_features, num_features // 2)
                self.features.add_module(f'transition{i+1}', trans)
                num_features = num_features // 2
        
        # Final batch norm
        self.features.add_module('norm_final', nn.BatchNorm2d(num_features))
        
        # Global average pooling and classifier
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.dropout = nn.Dropout(dropout_rate)
        self.classifier = nn.Linear(num_features, num_classes)
        
        # Initialize weights
        self._initialize_weights()
    
    def _make_dense_layer(self, input_features: int, growth_rate: int, dropout_rate: float):
        """Create a single dense layer"""
        return nn.Sequential(
            nn.BatchNorm2d(input_features),
            nn.ReLU(inplace=True),
            nn.Conv2d(input_features, 4 * growth_rate, kernel_size=1, stride=1, bias=False),
            nn.BatchNorm2d(4 * growth_rate),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate),
            nn.Conv2d(4 * growth_rate, growth_rate, kernel_size=3, stride=1, padding=1, bias=False)
        )
    
    def _make_dense_block(self, num_layers: int, input_features: int, growth_rate: int, dropout_rate: float):
        """Create a dense block"""
        layers = []
        for i in range(num_layers):
            layer = self._make_dense_layer(input_features + i * growth_rate, growth_rate, dropout_rate)
            layers.append(layer)
        
        return DenseBlock(layers)
    
    def _make_transition(self, input_features: int, output_features: int):
        """Create a transition layer"""
        return nn.Sequential(
            nn.BatchNorm2d(input_features),
            nn.ReLU(inplace=True),
            nn.Conv2d(input_features, output_features, kernel_size=1, stride=1, bias=False),
            nn.AvgPool2d(kernel_size=2, stride=2)
        )
    
    def _initialize_weights(self):
        """Initialize model weights"""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.features(x)
        features = F.relu(features, inplace=True)
        features = self.avgpool(features)
        features = torch.flatten(features, 1)
        features = self.dropout(features)
        output = self.classifier(features)
        return output


class DenseBlock(nn.Module):
    """Dense block implementation for CompactDenseNet"""
    
    def __init__(self, layers):
        super(DenseBlock, self).__init__()
        self.layers = nn.ModuleList(layers)
    
    def forward(self, x):
        features = [x]
        for layer in self.layers:
            new_feature = layer(torch.cat(features, 1))
            features.append(new_feature)
        return torch.cat(features, 1)


def create_densenet_model(config: Dict) -> nn.Module:
    """
    Factory function to create DenseNet model based on configuration
    
    Args:
        config: Configuration dictionary
        
    Returns:
        DenseNet model instance
    """
    try:
        densenet_config = config.get('models', {}).get('densenet', {})
        
        architecture = densenet_config.get('architecture', 'densenet121')
        pretrained = densenet_config.get('pretrained', True)
        growth_rate = densenet_config.get('growth_rate', 32)
        compression = densenet_config.get('compression', 0.5)
        dropout_rate = densenet_config.get('dropout_rate', 0.3)
        
        # Validate parameters
        valid_architectures = ['densenet121', 'densenet169', 'densenet201', 'densenet161', 'compact']
        if architecture not in valid_architectures:
            raise ValueError(f"Invalid DenseNet architecture '{architecture}'. Valid options: {valid_architectures}")
        
        if not isinstance(pretrained, bool):
            raise ValueError(f"pretrained must be boolean, got {type(pretrained)}: {pretrained}")
        
        if not isinstance(growth_rate, int) or growth_rate < 8 or growth_rate > 64:
            raise ValueError(f"growth_rate must be integer between 8 and 64, got: {growth_rate}")
        
        if not isinstance(compression, (int, float)) or compression < 0.1 or compression > 1.0:
            raise ValueError(f"compression must be float between 0.1 and 1.0, got: {compression}")
        
        if not isinstance(dropout_rate, (int, float)) or dropout_rate < 0 or dropout_rate > 1:
            raise ValueError(f"dropout_rate must be float between 0 and 1, got: {dropout_rate}")
        
        print(f"Creating DenseNet model - Architecture: {architecture}, Pretrained: {pretrained}")
        print(f"Parameters - Growth rate: {growth_rate}, Compression: {compression}, Dropout: {dropout_rate}")
        
        if architecture == 'compact':
            # Create compact custom implementation
            model = CompactDenseNet(
                num_classes=2,
                growth_rate=growth_rate,
                dropout_rate=dropout_rate
            )
        else:
            # Create standard DenseNet with medical adaptations
            model = MedicalDenseNet(
                architecture=architecture,
                num_classes=2,
                pretrained=pretrained,
                growth_rate=growth_rate,
                compression=compression,
                dropout_rate=dropout_rate,
                config=densenet_config
            )
        
        if model is None:
            raise RuntimeError(f"Failed to create DenseNet model with architecture '{architecture}'")
        
        print(f"✅ Successfully created {architecture} model")
        return model
        
    except Exception as e:
        print(f"❌ Error creating DenseNet model: {str(e)}")
        raise


def test_densenet_model():
    """Test the DenseNet model implementation"""
    print("Testing DenseNet Model...")
    
    # Detect device
    if torch.backends.mps.is_available():
        device = torch.device("mps")
        print("✅ Using GPU via MPS (Apple Silicon)")
    else:
        device = torch.device("cpu")
        print("❌ MPS not available. Using CPU.")

    # Test configurations
    test_configs = [
        {'architecture': 'densenet121', 'pretrained': True},
        {'architecture': 'densenet169', 'pretrained': False},
        {'architecture': 'compact', 'growth_rate': 16}
    ]
    
    for i, test_config in enumerate(test_configs):
        print(f"\n--- Test {i+1}: {test_config} ---")
        
        try:
            # Create config
            config = {'models': {'densenet': test_config}}
            
            # Create and move model to device
            model = create_densenet_model(config).to(device)
            print(f"Created model: {model.__class__.__name__} on {device}")
            
            # Prepare dummy input and move to device
            batch_size = 4
            x = torch.randn(batch_size, 3, 50, 50).to(device)
            
            model.eval()
            with torch.no_grad():
                output = model(x)
            
            print(f"Input shape: {x.shape}")
            print(f"Output shape: {output.shape}")
            print(f"Output range: [{output.min():.3f}, {output.max():.3f}]")
            
            # Test model info
            if hasattr(model, 'get_model_info'):
                info = model.get_model_info()
                print(f"Model parameters: {info['total_parameters']:,}")
                print(f"Model size: {info['model_size_mb']:.2f} MB")
            
            print(f"✅ Test {i+1} completed successfully!")
            
        except Exception as e:
            print(f"❌ Test {i+1} failed: {str(e)}")
    
    print("\n✅ DenseNet model test completed!")


if __name__ == "__main__":
    test_densenet_model()