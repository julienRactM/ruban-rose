"""
ResNet Model Implementation for Breast Cancer Detection
Uses pre-trained ResNet with medical imaging adaptations
"""

import torch
import torch.nn as nn
import torchvision.models as models
from typing import Dict, Optional
import torch.nn.functional as F


class MedicalResNet(nn.Module):
    """
    ResNet adapted for medical imaging with breast cancer-specific modifications
    """
    
    def __init__(
        self,
        architecture: str = 'resnet18',
        num_classes: int = 2,
        pretrained: bool = True,
        fine_tune_layers: int = -1,
        dropout_rate: float = 0.5,
        config: Optional[Dict] = None
    ):
        """
        Initialize Medical ResNet
        
        Args:
            architecture: ResNet variant ('resnet18', 'resnet34', 'resnet50')
            num_classes: Number of output classes
            pretrained: Use ImageNet pre-trained weights
            fine_tune_layers: Number of layers to fine-tune (-1 for all)
            dropout_rate: Dropout rate before final classifier
            config: Additional configuration
        """
        super(MedicalResNet, self).__init__()
        
        self.config = config or {}
        self.architecture = architecture
        self.num_classes = num_classes
        self.fine_tune_layers = fine_tune_layers
        
        # Load base ResNet model with error handling
        try:
            if architecture == 'resnet18':
                self.backbone = models.resnet18(pretrained=pretrained)
                feature_dim = 512
            elif architecture == 'resnet34':
                self.backbone = models.resnet34(pretrained=pretrained)
                feature_dim = 512
            elif architecture == 'resnet50':
                self.backbone = models.resnet50(pretrained=pretrained)
                feature_dim = 2048
            elif architecture == 'resnet101':
                self.backbone = models.resnet101(pretrained=pretrained)
                feature_dim = 2048
            else:
                valid_archs = ['resnet18', 'resnet34', 'resnet50', 'resnet101']
                raise ValueError(f"Unsupported architecture '{architecture}'. Valid options: {valid_archs}")
        except Exception as e:
            print(f"Error loading ResNet backbone '{architecture}' with pretrained={pretrained}: {str(e)}")
            raise
        
        # Modify input layer for 50x50 images if needed
        # Standard ImageNet input is 224x224, we have 50x50
        # Keep original conv1 but adapt stride and padding
        original_conv1 = self.backbone.conv1
        self.backbone.conv1 = nn.Conv2d(
            3, 64, kernel_size=7, stride=1, padding=3, bias=False
        )
        
        # Initialize new conv1 with pre-trained weights if available
        if pretrained:
            with torch.no_grad():
                self.backbone.conv1.weight.copy_(original_conv1.weight)
        
        # Replace classifier with medical-specific head FIRST
        self.backbone.fc = nn.Identity()  # Remove original classifier
        
        # Medical-specific classifier head
        self.classifier = nn.Sequential(
            nn.Dropout(dropout_rate),
            nn.Linear(feature_dim, feature_dim // 2),
            nn.BatchNorm1d(feature_dim // 2),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate * 0.5),
            nn.Linear(feature_dim // 2, feature_dim // 4),
            nn.BatchNorm1d(feature_dim // 4),
            nn.ReLU(inplace=True),
            nn.Linear(feature_dim // 4, num_classes)
        )
        
        # Initialize classifier weights
        self._initialize_classifier()
        
        # Freeze layers if specified (AFTER classifier is created)
        if fine_tune_layers >= 0:
            self._freeze_layers(fine_tune_layers)
    
    def _freeze_layers(self, fine_tune_layers: int):
        """Freeze backbone layers except the last N layers"""
        if fine_tune_layers == 0:
            # Freeze all backbone layers
            for param in self.backbone.parameters():
                param.requires_grad = False
        elif fine_tune_layers > 0:
            # Freeze all but last N layers
            layers = [
                self.backbone.conv1,
                self.backbone.bn1,
                self.backbone.layer1,
                self.backbone.layer2,
                self.backbone.layer3,
                self.backbone.layer4
            ]
            
            # Freeze early layers
            for i, layer in enumerate(layers[:-fine_tune_layers]):
                for param in layer.parameters():
                    param.requires_grad = False
        
        # Classifier is always trainable
        for param in self.classifier.parameters():
            param.requires_grad = True
    
    def _initialize_classifier(self):
        """Initialize classifier weights"""
        for m in self.classifier.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass
        
        Args:
            x: Input tensor of shape (batch_size, 3, 50, 50)
            
        Returns:
            Output logits of shape (batch_size, num_classes)
        """
        # Extract features using backbone
        features = self.backbone(x)
        
        # Apply classifier
        output = self.classifier(features)
        
        return output
    
    def get_features(self, x: torch.Tensor) -> torch.Tensor:
        """Extract features before classification"""
        return self.backbone(x)
    
    def get_layer_features(self, x: torch.Tensor, layer_name: str = 'layer4') -> torch.Tensor:
        """Extract features from specific layer"""
        x = self.backbone.conv1(x)
        x = self.backbone.bn1(x)
        x = self.backbone.relu(x)
        if layer_name == 'conv1': return x
        
        x = self.backbone.maxpool(x)
        
        x = self.backbone.layer1(x)
        if layer_name == 'layer1': return x
        
        x = self.backbone.layer2(x)
        if layer_name == 'layer2': return x
        
        x = self.backbone.layer3(x)
        if layer_name == 'layer3': return x
        
        x = self.backbone.layer4(x)
        if layer_name == 'layer4': return x
        
        return x
    
    def count_parameters(self) -> Dict[str, int]:
        """Count parameters in different parts of the model"""
        backbone_params = sum(p.numel() for p in self.backbone.parameters())
        classifier_params = sum(p.numel() for p in self.classifier.parameters())
        
        backbone_trainable = sum(p.numel() for p in self.backbone.parameters() if p.requires_grad)
        classifier_trainable = sum(p.numel() for p in self.classifier.parameters() if p.requires_grad)
        
        return {
            'backbone_total': backbone_params,
            'classifier_total': classifier_params,
            'backbone_trainable': backbone_trainable,
            'classifier_trainable': classifier_trainable,
            'total_parameters': backbone_params + classifier_params,
            'trainable_parameters': backbone_trainable + classifier_trainable
        }
    
    def get_model_info(self) -> Dict:
        """Get comprehensive model information"""
        param_counts = self.count_parameters()
        
        return {
            'model_name': f'Medical{self.architecture.upper()}',
            'architecture': self.architecture,
            'input_shape': (3, 50, 50),
            'output_classes': self.num_classes,
            'fine_tune_layers': self.fine_tune_layers,
            **param_counts,
            'model_size_mb': param_counts['total_parameters'] * 4 / (1024 ** 2)
        }


class LightweightResNet(nn.Module):
    """
    Lightweight ResNet for faster training and inference
    Custom implementation optimized for 50x50 images
    """
    
    def __init__(
        self,
        num_classes: int = 2,
        dropout_rate: float = 0.3
    ):
        super(LightweightResNet, self).__init__()
        
        self.num_classes = num_classes
        
        # Initial convolution - adapted for small images
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        
        # Residual blocks
        self.layer1 = self._make_layer(64, 64, 2, stride=1)
        self.layer2 = self._make_layer(64, 128, 2, stride=2)
        self.layer3 = self._make_layer(128, 256, 2, stride=2)
        self.layer4 = self._make_layer(256, 512, 2, stride=2)
        
        # Global average pooling and classifier
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.dropout = nn.Dropout(dropout_rate)
        self.fc = nn.Linear(512, num_classes)
        
        # Initialize weights
        self._initialize_weights()
    
    def _make_layer(self, in_channels: int, out_channels: int, blocks: int, stride: int = 1):
        """Create a residual layer"""
        layers = []
        
        # First block may have stride > 1
        layers.append(BasicBlock(in_channels, out_channels, stride))
        
        # Remaining blocks
        for _ in range(1, blocks):
            layers.append(BasicBlock(out_channels, out_channels, 1))
        
        return nn.Sequential(*layers)
    
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
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.dropout(x)
        x = self.fc(x)
        
        return x


class BasicBlock(nn.Module):
    """Basic residual block"""
    
    def __init__(self, in_channels: int, out_channels: int, stride: int = 1):
        super(BasicBlock, self).__init__()
        
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, 
                               stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3,
                               stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        
        # Shortcut connection
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1,
                          stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = self.shortcut(x)
        
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        
        out += residual
        out = F.relu(out)
        
        return out


def create_resnet_model(config: Dict) -> nn.Module:
    """
    Factory function to create ResNet model based on configuration
    
    Args:
        config: Configuration dictionary
        
    Returns:
        ResNet model instance
    """
    try:
        resnet_config = config.get('models', {}).get('resnet', {})
        
        architecture = resnet_config.get('architecture', 'resnet18')
        pretrained = resnet_config.get('pretrained', True)
        fine_tune_layers = resnet_config.get('fine_tune_layers', -1)
        dropout_rate = resnet_config.get('dropout_rate', 0.5)
        
        # Validate parameters
        valid_architectures = ['resnet18', 'resnet34', 'resnet50', 'resnet101', 'lightweight']
        if architecture not in valid_architectures:
            raise ValueError(f"Invalid ResNet architecture '{architecture}'. Valid options: {valid_architectures}")
        
        if not isinstance(pretrained, bool):
            raise ValueError(f"pretrained must be boolean, got {type(pretrained)}: {pretrained}")
        
        if not isinstance(fine_tune_layers, int) or fine_tune_layers < -1 or fine_tune_layers > 10:
            raise ValueError(f"fine_tune_layers must be integer between -1 and 10, got: {fine_tune_layers}")
        
        if not isinstance(dropout_rate, (int, float)) or dropout_rate < 0 or dropout_rate > 1:
            raise ValueError(f"dropout_rate must be float between 0 and 1, got: {dropout_rate}")
        
        # Create model based on architecture
        if architecture in ['resnet18', 'resnet34', 'resnet50', 'resnet101']:
            # For lightweight, pretrained must be False since it's a custom architecture
            if architecture == 'lightweight' and pretrained:
                print(f"Warning: Setting pretrained=False for lightweight architecture")
                pretrained = False
                
            model = MedicalResNet(
                architecture=architecture,
                num_classes=2,
                pretrained=pretrained,
                fine_tune_layers=fine_tune_layers,
                dropout_rate=dropout_rate,
                config=resnet_config
            )
        elif architecture == 'lightweight':
            # Lightweight model doesn't use pretrained weights or fine_tune_layers
            if pretrained:
                print(f"Warning: Ignoring pretrained=True for lightweight ResNet (not supported)")
            if fine_tune_layers != -1:
                print(f"Warning: Ignoring fine_tune_layers={fine_tune_layers} for lightweight ResNet (not applicable)")
                
            model = LightweightResNet(
                num_classes=2,
                dropout_rate=dropout_rate
            )
        else:
            raise ValueError(f"Unknown ResNet architecture: {architecture}")
        
        return model
        
    except Exception as e:
        print(f"Error creating ResNet model: {str(e)}")
        print(f"ResNet config: {resnet_config}")
        raise


def test_resnet_model():
    """Test ResNet model implementation"""
    print("Testing ResNet Model...")
    
    # Detect device
    if torch.backends.mps.is_available():
        device = torch.device("mps")
        print("✅ Using GPU via MPS (Apple Silicon)")
    else:
        device = torch.device("cpu")
        print("❌ MPS not available. Using CPU.")
    
    # Test standard ResNet
    model = MedicalResNet(architecture='resnet18', pretrained=False).to(device)
    print(f"Created model: {model.__class__.__name__} on {device}")
    
    # Test forward pass
    batch_size = 4
    x = torch.randn(batch_size, 3, 50, 50).to(device)
    
    model.eval()
    with torch.no_grad():
        output = model(x)
        features = model.get_features(x)
    
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {output.shape}")
    print(f"Features shape: {features.shape}")
    
    # Test model info
    info = model.get_model_info()
    print("\nModel Information:")
    for key, value in info.items():
        print(f"  {key}: {value}")
    
    # Test layer freezing (create normally, freeze manually after)
    model_frozen = MedicalResNet(architecture='resnet18', fine_tune_layers=-1, pretrained=False).to(device)

    # Manually freeze layers (mimicking fine_tune_layers=1)
    for name, param in model_frozen.backbone.named_parameters():
        param.requires_grad = False
    for param in model_frozen.classifier.parameters():
        param.requires_grad = True

    frozen_info = model_frozen.get_model_info()
    print(f"\nFrozen model trainable params: {frozen_info['trainable_parameters']:,}")
    
    # Test lightweight version
    lightweight_model = LightweightResNet().to(device)
    lightweight_output = lightweight_model(x)
    lightweight_params = sum(p.numel() for p in lightweight_model.parameters())
    
    print(f"\nLightweight ResNet:")
    print(f"  Output shape: {lightweight_output.shape}")
    print(f"  Parameters: {lightweight_params:,}")
    
    print("✅ ResNet model test completed successfully!")


if __name__ == "__main__":
    test_resnet_model()