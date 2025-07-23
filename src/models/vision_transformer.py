"""
Vision Transformer (ViT) Implementation for Breast Cancer Detection
Adapted for 50x50 histopathology images with medical imaging considerations
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional, Tuple
import timm
import math


class MedicalViT(nn.Module):
    """
    Vision Transformer adapted for medical breast cancer imaging
    
    Key adaptations:
    - Input size handling for 50x50 images (upscaled to ViT requirements)
    - Medical-specific classification head
    - Attention visualization for interpretability
    - Dropout and regularization for medical data
    """
    
    def __init__(
        self,
        model_name: str = 'vit_base_patch16_224',
        num_classes: int = 2,
        input_size: Tuple[int, int] = (50, 50),
        pretrained: bool = True,
        dropout_rate: float = 0.1,
        config: Optional[Dict] = None
    ):
        """
        Initialize Medical ViT
        
        Args:
            model_name: ViT model variant from timm
            num_classes: Number of output classes
            input_size: Original input image size (will be upscaled for ViT)
            pretrained: Use ImageNet pre-trained weights
            dropout_rate: Dropout rate in classification head
            config: Additional configuration
        """
        super(MedicalViT, self).__init__()
        
        self.config = config or {}
        self.model_name = model_name
        self.num_classes = num_classes
        self.input_size = input_size
        self.dropout_rate = dropout_rate
        
        # ViT requires larger input sizes (typically 224x224)
        # We'll upscale 50x50 to 224x224
        self.target_size = 224
        
        # Load pre-trained ViT model
        self.backbone = timm.create_model(
            model_name,
            pretrained=pretrained,
            num_classes=0,  # Remove original classifier
            img_size=self.target_size
        )
        
        # Get feature dimension
        self.feature_dim = self.backbone.num_features
        
        # Input upscaling layer
        self.upscale = nn.Upsample(
            size=(self.target_size, self.target_size),
            mode='bilinear',
            align_corners=False
        )
        
        # Medical-specific classification head
        self.classifier = self._build_classifier()
        
        # Initialize classifier weights
        self._initialize_classifier()
    
    def _build_classifier(self) -> nn.Module:
        """Build medical-specific classification head"""
        return nn.Sequential(
            nn.LayerNorm(self.feature_dim),
            nn.Dropout(self.dropout_rate),
            nn.Linear(self.feature_dim, self.feature_dim // 2),
            nn.GELU(),
            nn.Dropout(self.dropout_rate * 0.5),
            nn.Linear(self.feature_dim // 2, self.feature_dim // 4),
            nn.GELU(),
            nn.Linear(self.feature_dim // 4, self.num_classes)
        )
    
    def _initialize_classifier(self):
        """Initialize classifier weights"""
        for m in self.classifier.modules():
            if isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass
        
        Args:
            x: Input tensor of shape (batch_size, 3, 50, 50)
            
        Returns:
            Output logits of shape (batch_size, num_classes)
        """
        # Upscale input to ViT requirements
        x_upscaled = self.upscale(x)
        
        # Extract features using ViT backbone
        features = self.backbone(x_upscaled)
        
        # Apply classification head
        output = self.classifier(features)
        
        return output
    
    def get_features(self, x: torch.Tensor) -> torch.Tensor:
        """Extract features before classification"""
        x_upscaled = self.upscale(x)
        return self.backbone(x_upscaled)
    
    def get_attention_maps(self, x: torch.Tensor, layer_idx: int = -1) -> torch.Tensor:
        """
        Extract attention maps for visualization
        
        Args:
            x: Input tensor
            layer_idx: Transformer layer index (-1 for last layer)
            
        Returns:
            Attention maps tensor
        """
        x_upscaled = self.upscale(x)
        
        # Forward pass with attention extraction
        attention_maps = []
        
        def attention_hook(module, input, output):
            if hasattr(output, 'size') and len(output.size()) == 4:
                attention_maps.append(output)
        
        # Register hook on transformer blocks
        transformer_blocks = self.backbone.blocks
        if layer_idx == -1:
            layer_idx = len(transformer_blocks) - 1
        
        hook = transformer_blocks[layer_idx].attn.register_forward_hook(attention_hook)
        
        try:
            _ = self.backbone(x_upscaled)
            if attention_maps:
                return attention_maps[0]
        finally:
            hook.remove()
        
        return None
    
    def count_parameters(self) -> Dict[str, int]:
        """Count parameters in different parts of the model"""
        backbone_params = sum(p.numel() for p in self.backbone.parameters())
        classifier_params = sum(p.numel() for p in self.classifier.parameters())
        
        return {
            'backbone_total': backbone_params,
            'classifier_total': classifier_params,
            'total_parameters': backbone_params + classifier_params,
            'trainable_parameters': sum(p.numel() for p in self.parameters() if p.requires_grad)
        }
    
    def get_model_info(self) -> Dict:
        """Get comprehensive model information"""
        param_counts = self.count_parameters()
        
        return {
            'model_name': f'Medical{self.model_name.upper()}',
            'base_model': self.model_name,
            'input_size': self.input_size,
            'target_size': (self.target_size, self.target_size),
            'output_classes': self.num_classes,
            'feature_dim': self.feature_dim,
            'dropout_rate': self.dropout_rate,
            **param_counts,
            'model_size_mb': param_counts['total_parameters'] * 4 / (1024 ** 2)
        }


class CompactViT(nn.Module):
    """
    Compact Vision Transformer for faster training and inference
    Custom implementation optimized for 50x50 medical images
    """
    
    def __init__(
        self,
        img_size: int = 50,
        patch_size: int = 5,
        num_classes: int = 2,
        embed_dim: int = 192,
        depth: int = 6,
        num_heads: int = 6,
        dropout_rate: float = 0.1
    ):
        """
        Initialize Compact ViT
        
        Args:
            img_size: Input image size
            patch_size: Patch size for tokenization
            num_classes: Number of output classes
            embed_dim: Embedding dimension
            depth: Number of transformer layers
            num_heads: Number of attention heads
            dropout_rate: Dropout rate
        """
        super(CompactViT, self).__init__()
        
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_classes = num_classes
        self.embed_dim = embed_dim
        self.num_patches = (img_size // patch_size) ** 2
        
        # Patch embedding
        self.patch_embed = PatchEmbed(
            img_size=img_size,
            patch_size=patch_size,
            embed_dim=embed_dim
        )
        
        # Class token and position embedding
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, self.num_patches + 1, embed_dim))
        
        # Transformer blocks
        self.blocks = nn.ModuleList([
            TransformerBlock(
                dim=embed_dim,
                num_heads=num_heads,
                dropout_rate=dropout_rate
            ) for _ in range(depth)
        ])
        
        # Layer norm and classifier
        self.norm = nn.LayerNorm(embed_dim)
        self.head = nn.Linear(embed_dim, num_classes)
        
        # Initialize weights
        self._initialize_weights()
    
    def _initialize_weights(self):
        """Initialize model weights"""
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.LayerNorm):
                nn.init.constant_(m.weight, 1.0)
                nn.init.constant_(m.bias, 0)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass"""
        B = x.shape[0]
        
        # Patch embedding
        x = self.patch_embed(x)
        
        # Add class token
        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat([cls_tokens, x], dim=1)
        
        # Add position embedding
        x = x + self.pos_embed
        
        # Transformer blocks
        for block in self.blocks:
            x = block(x)
        
        # Layer norm and classification
        x = self.norm(x)
        cls_token_final = x[:, 0]
        output = self.head(cls_token_final)
        
        return output


class PatchEmbed(nn.Module):
    """Patch embedding layer"""
    
    def __init__(self, img_size: int = 50, patch_size: int = 5, embed_dim: int = 192):
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = (img_size // patch_size) ** 2
        
        self.proj = nn.Conv2d(3, embed_dim, kernel_size=patch_size, stride=patch_size)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.proj(x)  # (B, embed_dim, H//patch_size, W//patch_size)
        x = x.flatten(2).transpose(1, 2)  # (B, num_patches, embed_dim)
        return x


class TransformerBlock(nn.Module):
    """Transformer block with multi-head attention"""
    
    def __init__(self, dim: int, num_heads: int, dropout_rate: float = 0.0):
        super().__init__()
        
        self.norm1 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(
            embed_dim=dim,
            num_heads=num_heads,
            dropout=dropout_rate,
            batch_first=True
        )
        
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, dim * 4),
            nn.GELU(),
            nn.Dropout(dropout_rate),
            nn.Linear(dim * 4, dim),
            nn.Dropout(dropout_rate)
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Multi-head attention with residual connection
        x_norm = self.norm1(x)
        attn_out, _ = self.attn(x_norm, x_norm, x_norm)
        x = x + attn_out
        
        # MLP with residual connection
        x = x + self.mlp(self.norm2(x))
        
        return x


def create_vit_model(config: Dict) -> nn.Module:
    """
    Factory function to create ViT model based on configuration
    
    Args:
        config: Configuration dictionary
        
    Returns:
        ViT model instance
    """
    vit_config = config.get('models', {}).get('vision_transformer', {})
    
    model_name = vit_config.get('model_name', 'vit_base_patch16_224')
    dropout_rate = vit_config.get('dropout_rate', 0.1)
    input_size = tuple(config.get('data', {}).get('image_size', [50, 50]))
    
    if model_name == 'compact':
        # Use compact custom implementation
        model = CompactViT(
            img_size=input_size[0],
            patch_size=vit_config.get('patch_size', 5),
            num_classes=2,
            embed_dim=vit_config.get('embed_dim', 192),
            depth=vit_config.get('depth', 6),
            num_heads=vit_config.get('num_heads', 6),
            dropout_rate=dropout_rate
        )
    else:
        # Use timm pre-trained ViT
        model = MedicalViT(
            model_name=model_name,
            num_classes=2,
            input_size=input_size,
            pretrained=True,
            dropout_rate=dropout_rate,
            config=vit_config
        )
    
    return model


def test_vit_model():
    """Test Vision Transformer model implementation"""
    print("Testing Vision Transformer Model...")
    
    # Test Medical ViT
    model = MedicalViT(model_name='vit_tiny_patch16_224', pretrained=False)
    print(f"Created model: {model.__class__.__name__}")
    
    # Test forward pass
    batch_size = 4
    x = torch.randn(batch_size, 3, 50, 50)
    
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
    
    # Test compact ViT
    compact_model = CompactViT(
        img_size=50,
        patch_size=5,
        embed_dim=96,
        depth=4,
        num_heads=4
    )
    
    with torch.no_grad():
        compact_output = compact_model(x)
    
    compact_params = sum(p.numel() for p in compact_model.parameters())
    
    print(f"\nCompact ViT:")
    print(f"  Output shape: {compact_output.shape}")
    print(f"  Parameters: {compact_params:,}")
    print(f"  Patches: {compact_model.num_patches}")
    
    # Test attention visualization (if available)
    try:
        attention = model.get_attention_maps(x)
        if attention is not None:
            print(f"  Attention maps shape: {attention.shape}")
    except Exception as e:
        print(f"  Attention extraction not available: {e}")
    
    print("✅ Vision Transformer model test completed successfully!")


if __name__ == "__main__":
    test_vit_model()