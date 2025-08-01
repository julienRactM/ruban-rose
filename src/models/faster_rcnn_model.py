"""
Faster R-CNN Model Implementation for Breast Cancer Detection
Adapted from object detection to classification for 50x50 medical patches
"""

import torch
import torch.nn as nn
import torchvision.models as models
from torchvision.models.detection import FasterRCNN
from torchvision.models.detection.backbone_utils import resnet_fpn_backbone
from typing import Dict, Optional, List
import torch.nn.functional as F


class MedicalFasterRCNN(nn.Module):
    """
    Faster R-CNN adapted for medical patch classification
    Converts detection architecture to classification for 50x50 images
    """
    
    def __init__(
        self,
        backbone: str = 'resnet50',
        num_classes: int = 2,
        pretrained: bool = True,
        anchor_scales: List[int] = [8, 16, 32],
        roi_pool_size: int = 7,
        dropout_rate: float = 0.5,
        config: Optional[Dict] = None
    ):
        """
        Initialize Medical Faster R-CNN
        
        Args:
            backbone: Backbone architecture ('resnet50', 'resnet101')
            num_classes: Number of output classes
            pretrained: Use COCO pre-trained weights
            anchor_scales: Anchor scales for RPN
            roi_pool_size: ROI pooling output size
            dropout_rate: Dropout rate in classification head
            config: Additional configuration
        """
        super(MedicalFasterRCNN, self).__init__()
        
        self.config = config or {}
        self.backbone_name = backbone
        self.num_classes = num_classes
        self.anchor_scales = anchor_scales
        self.roi_pool_size = roi_pool_size
        
        # Create backbone with FPN
        try:
            if backbone == 'resnet50':
                self.backbone = resnet_fpn_backbone('resnet50', pretrained=pretrained)
                backbone_out_channels = 256
            elif backbone == 'resnet101':
                self.backbone = resnet_fpn_backbone('resnet101', pretrained=pretrained)
                backbone_out_channels = 256
            else:
                valid_backbones = ['resnet50', 'resnet101']
                raise ValueError(f"Unsupported backbone '{backbone}'. Valid options: {valid_backbones}")
        except Exception as e:
            print(f"Error loading Faster R-CNN backbone '{backbone}' with pretrained={pretrained}: {str(e)}")
            raise
        
        # Region Proposal Network (RPN) - simplified for classification
        self.rpn = SimplifiedRPN(backbone_out_channels, len(anchor_scales))
        
        # ROI Head - adapted for classification
        self.roi_head = ClassificationROIHead(
            backbone_out_channels,
            num_classes,
            roi_pool_size,
            dropout_rate
        )
        
        # Global classification head (fallback) - MPS compatible
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.global_classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(dropout_rate),
            nn.Linear(backbone_out_channels, backbone_out_channels // 2),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate * 0.5),
            nn.Linear(backbone_out_channels // 2, num_classes)
        )
        
        # Initialize weights
        self._initialize_weights()
    
    def _initialize_weights(self):
        """Initialize weights for custom layers"""
        for module in [self.rpn, self.roi_head, self.global_classifier]:
            for m in module.modules():
                if isinstance(m, nn.Conv2d):
                    nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                    if m.bias is not None:
                        nn.init.constant_(m.bias, 0)
                elif isinstance(m, nn.Linear):
                    nn.init.xavier_normal_(m.weight)
                    if m.bias is not None:
                        nn.init.constant_(m.bias, 0)
                elif isinstance(m, nn.BatchNorm2d):
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
        # Extract features using FPN backbone
        features = self.backbone(x)
        
        # Use the finest scale feature map for small images
        if '0' in features:  # FPN level 0 (finest)
            main_features = features['0']
        elif 'pool' in features:  # Pooled features
            main_features = features['pool']
        else:
            # Use the first available feature map
            main_features = list(features.values())[0]
        
        # Generate region proposals (simplified for classification)
        rpn_features = self.rpn(main_features)
        
        # ROI-based classification
        roi_output = self.roi_head(main_features, rpn_features)
        
        # Global classification as ensemble - MPS compatible pooling
        if main_features.device.type == 'mps':
            # Manual global average pooling for MPS compatibility
            b, c, h, w = main_features.size()
            pooled_global = F.avg_pool2d(main_features, (h, w), stride=1).view(b, c, 1, 1)
        else:
            pooled_global = self.global_pool(main_features)
        
        global_output = self.global_classifier(pooled_global)
        
        # Combine ROI and global predictions (weighted average)
        combined_output = 0.7 * roi_output + 0.3 * global_output
        
        return combined_output
    
    def get_feature_maps(self, x: torch.Tensor, level: str = '0') -> torch.Tensor:
        """
        Extract feature maps from specified FPN level
        
        Args:
            x: Input tensor
            level: FPN level ('0', '1', '2', '3')
            
        Returns:
            Feature maps from specified level
        """
        features = self.backbone(x)
        return features.get(level, list(features.values())[0])
    
    def count_parameters(self) -> Dict[str, int]:
        """Count model parameters"""
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        
        backbone_params = sum(p.numel() for p in self.backbone.parameters())
        rpn_params = sum(p.numel() for p in self.rpn.parameters())
        roi_params = sum(p.numel() for p in self.roi_head.parameters())
        global_params = sum(p.numel() for p in self.global_classifier.parameters())
        
        return {
            'total_parameters': total_params,
            'trainable_parameters': trainable_params,
            'backbone_parameters': backbone_params,
            'rpn_parameters': rpn_params,
            'roi_parameters': roi_params,
            'global_classifier_parameters': global_params
        }
    
    def get_model_info(self) -> Dict:
        """Get comprehensive model information"""
        param_counts = self.count_parameters()
        
        return {
            'model_name': f'MedicalFasterRCNN_{self.backbone_name.upper()}',
            'backbone': self.backbone_name,
            'input_shape': (3, 50, 50),
            'output_classes': self.num_classes,
            'anchor_scales': self.anchor_scales,
            'roi_pool_size': self.roi_pool_size,
            **param_counts,
            'model_size_mb': param_counts['total_parameters'] * 4 / (1024 ** 2)
        }


class SimplifiedRPN(nn.Module):
    """
    Simplified Region Proposal Network for classification
    Generates attention-like features rather than actual proposals
    """
    
    def __init__(self, in_channels: int, num_anchors: int):
        super(SimplifiedRPN, self).__init__()
        
        # Convolutional layer for feature processing
        self.conv = nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1)
        
        # Attention mechanism for region importance
        self.attention = nn.Sequential(
            nn.Conv2d(in_channels, in_channels // 4, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels // 4, 1, kernel_size=1),
            nn.Sigmoid()
        )
        
        # Feature enhancement
        self.feature_enhance = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size=1),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True)
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Process features
        features = F.relu(self.conv(x))
        
        # Generate attention weights
        attention_weights = self.attention(features)
        
        # Apply attention
        attended_features = features * attention_weights
        
        # Enhance features
        enhanced_features = self.feature_enhance(attended_features)
        
        return enhanced_features


class ClassificationROIHead(nn.Module):
    """
    ROI Head adapted for classification instead of detection
    """
    
    def __init__(self, in_channels: int, num_classes: int, roi_pool_size: int, dropout_rate: float):
        super(ClassificationROIHead, self).__init__()
        
        self.roi_pool_size = roi_pool_size
        
        # ROI pooling layer - MPS compatible implementation
        self.roi_pool = nn.AdaptiveAvgPool2d((roi_pool_size, roi_pool_size))
        self.roi_pool_size = roi_pool_size
        
        # Classification head - dynamically calculate feature dimension
        # For 50x50 input images and roi_pool_size, estimate the feature dimension
        # This will be corrected during first forward pass if needed
        estimated_feature_dim = in_channels * roi_pool_size * roi_pool_size
        
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(estimated_feature_dim, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate),
            nn.Linear(1024, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate * 0.5),
            nn.Linear(512, num_classes)
        )
        
        # Flag to track if we need to reinitialize the first layer
        self._first_forward = True
    
    def forward(self, features: torch.Tensor, rpn_features: torch.Tensor) -> torch.Tensor:
        # Use enhanced features from RPN
        # MPS compatible pooling - handle non-divisible sizes
        if rpn_features.device.type == 'mps':
            # Manual implementation for MPS compatibility
            b, c, h, w = rpn_features.size()
            # Use average pooling with kernel size that fits the feature map
            kernel_h = max(1, h // self.roi_pool_size)
            kernel_w = max(1, w // self.roi_pool_size)
            pooled_features = F.avg_pool2d(rpn_features, (kernel_h, kernel_w), stride=(kernel_h, kernel_w))
        else:
            pooled_features = self.roi_pool(rpn_features)
        
        # Dynamic first layer resizing on first forward pass
        if self._first_forward:
            flattened = torch.flatten(pooled_features, 1)
            actual_feature_dim = flattened.shape[1]
            expected_feature_dim = self.classifier[1].in_features
            
            if actual_feature_dim != expected_feature_dim:
                # Recreate the first linear layer with correct dimensions
                device = rpn_features.device
                new_first_layer = nn.Linear(actual_feature_dim, 1024).to(device)
                # Copy the sequential module and replace the first linear layer
                new_classifier = nn.Sequential(
                    nn.Flatten(),
                    new_first_layer,
                    *list(self.classifier.children())[2:]  # Keep the rest
                ).to(device)
                self.classifier = new_classifier
            
            self._first_forward = False
        
        # Classify
        classification_output = self.classifier(pooled_features)
        
        return classification_output


class CompactFasterRCNN(nn.Module):
    """
    Compact Faster R-CNN for faster training and inference
    Simplified architecture optimized for 50x50 images
    """
    
    def __init__(
        self,
        num_classes: int = 2,
        backbone_channels: int = 256,
        dropout_rate: float = 0.3
    ):
        super(CompactFasterRCNN, self).__init__()
        
        self.num_classes = num_classes
        
        # Simple backbone for small images
        self.backbone = nn.Sequential(
            # Initial conv block
            nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
            
            # Residual-like blocks
            self._make_layer(64, 128, 2, stride=1),
            self._make_layer(128, 256, 2, stride=2),
            self._make_layer(256, backbone_channels, 2, stride=2),
        )
        
        # Simplified RPN
        self.rpn = SimplifiedRPN(backbone_channels, 3)
        
        # ROI head
        self.roi_head = ClassificationROIHead(backbone_channels, num_classes, 7, dropout_rate)
        
        # Global classifier - MPS compatible
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.global_classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(dropout_rate),
            nn.Linear(backbone_channels, num_classes)
        )
        
        # Initialize weights
        self._initialize_weights()
    
    def _make_layer(self, in_channels: int, out_channels: int, blocks: int, stride: int = 1):
        """Create a simple residual layer"""
        layers = []
        
        # First block with potential stride
        layers.append(nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False))
        layers.append(nn.BatchNorm2d(out_channels))
        layers.append(nn.ReLU(inplace=True))
        
        # Additional blocks
        for _ in range(1, blocks):
            layers.append(nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False))
            layers.append(nn.BatchNorm2d(out_channels))
            layers.append(nn.ReLU(inplace=True))
        
        return nn.Sequential(*layers)
    
    def _initialize_weights(self):
        """Initialize model weights"""
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
        # Extract features
        features = self.backbone(x)
        
        # RPN processing
        rpn_features = self.rpn(features)
        
        # ROI classification
        roi_output = self.roi_head(features, rpn_features)
        
        # Global classification - MPS compatible pooling
        if features.device.type == 'mps':
            # Manual global average pooling for MPS compatibility
            b, c, h, w = features.size()
            pooled_features = F.avg_pool2d(features, (h, w), stride=1).view(b, c, 1, 1)
        else:
            pooled_features = self.global_pool(features)
        
        global_output = self.global_classifier(pooled_features)
        
        # Combine predictions
        combined_output = 0.6 * roi_output + 0.4 * global_output
        
        return combined_output


def create_faster_rcnn_model(config: Dict) -> nn.Module:
    """
    Factory function to create Faster R-CNN model based on configuration
    
    Args:
        config: Configuration dictionary
        
    Returns:
        Faster R-CNN model instance
    """
    try:
        faster_rcnn_config = config.get('models', {}).get('faster_rcnn', {})
        
        backbone = faster_rcnn_config.get('backbone', 'resnet50')
        pretrained = faster_rcnn_config.get('pretrained', True)
        anchor_scales = faster_rcnn_config.get('anchor_scales', [8, 16, 32])
        roi_pool_size = faster_rcnn_config.get('roi_pool_size', 7)
        dropout_rate = faster_rcnn_config.get('dropout_rate', 0.5)
        
        # Validate parameters
        valid_backbones = ['resnet50', 'resnet101', 'compact']
        if backbone not in valid_backbones:
            raise ValueError(f"Invalid Faster R-CNN backbone '{backbone}'. Valid options: {valid_backbones}")
        
        if not isinstance(pretrained, bool):
            raise ValueError(f"pretrained must be boolean, got {type(pretrained)}: {pretrained}")
        
        if not isinstance(anchor_scales, list) or len(anchor_scales) == 0:
            raise ValueError(f"anchor_scales must be non-empty list, got: {anchor_scales}")
        
        if not isinstance(roi_pool_size, int) or roi_pool_size < 3 or roi_pool_size > 14:
            raise ValueError(f"roi_pool_size must be integer between 3 and 14, got: {roi_pool_size}")
        
        if not isinstance(dropout_rate, (int, float)) or dropout_rate < 0 or dropout_rate > 1:
            raise ValueError(f"dropout_rate must be float between 0 and 1, got: {dropout_rate}")
        
        print(f"Creating Faster R-CNN model - Backbone: {backbone}, Pretrained: {pretrained}")
        print(f"Parameters - Anchor scales: {anchor_scales}, ROI pool size: {roi_pool_size}, Dropout: {dropout_rate}")
        
        if backbone == 'compact':
            # Create compact custom implementation
            model = CompactFasterRCNN(
                num_classes=2,
                dropout_rate=dropout_rate
            )
        else:
            # Create standard Faster R-CNN with medical adaptations
            model = MedicalFasterRCNN(
                backbone=backbone,
                num_classes=2,
                pretrained=pretrained,
                anchor_scales=anchor_scales,
                roi_pool_size=roi_pool_size,
                dropout_rate=dropout_rate,
                config=faster_rcnn_config
            )
        
        if model is None:
            raise RuntimeError(f"Failed to create Faster R-CNN model with backbone '{backbone}'")
        
        print(f"✅ Successfully created Faster R-CNN model with {backbone} backbone")
        return model
        
    except Exception as e:
        print(f"❌ Error creating Faster R-CNN model: {str(e)}")
        raise


def test_faster_rcnn_model():
    """Test the Faster R-CNN model implementation"""
    print("Testing Faster R-CNN Model...")
    
    # Detect device
    if torch.backends.mps.is_available():
        device = torch.device("mps")
        print("✅ Using GPU via MPS (Apple Silicon)")
    else:
        device = torch.device("cpu")
        print("❌ MPS not available. Using CPU.")

    # Test configurations
    test_configs = [
        {'backbone': 'resnet50', 'pretrained': True},
        {'backbone': 'resnet101', 'pretrained': False},
        {'backbone': 'compact', 'dropout_rate': 0.2}
    ]
    
    for i, test_config in enumerate(test_configs):
        print(f"\n--- Test {i+1}: {test_config} ---")
        
        try:
            # Create config
            config = {'models': {'faster_rcnn': test_config}}
            
            # Create and move model to device
            model = create_faster_rcnn_model(config).to(device)
            print(f"Created model: {model.__class__.__name__} on {device}")
            
            # Prepare dummy input and move to device
            batch_size = 2  # Smaller batch size for complex model
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
    
    print("\n✅ Faster R-CNN model test completed!")


if __name__ == "__main__":
    test_faster_rcnn_model()