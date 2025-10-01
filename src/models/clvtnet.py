import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional, Dict, List
import math
import numpy as np

from src.config import Config

class PositionalEncoding(nn.Module):
    """Positional encoding for transformer layers."""
    
    def __init__(self, d_model: int, max_len: int = 5000):
        """Initialize positional encoding.
        
        Args:
            d_model: Model dimension
            max_len: Maximum sequence length
        """
        super().__init__()
        
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * 
                           (-math.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        
        self.register_buffer('pe', pe)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply positional encoding to input."""
        return x + self.pe[:x.size(0), :]

class ChannelAttention(nn.Module):
    """Channel attention mechanism."""
    
    def __init__(self, in_channels: int, reduction: int = 16):
        """Initialize channel attention.
        
        Args:
            in_channels: Number of input channels
            reduction: Channel reduction ratio
        """
        super().__init__()
        
        self.global_avg_pool = nn.AdaptiveAvgPool2d(1)
        self.global_max_pool = nn.AdaptiveMaxPool2d(1)
        
        self.fc = nn.Sequential(
            nn.Linear(in_channels, in_channels // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(in_channels // reduction, in_channels, bias=False)
        )
        
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply channel attention to input."""
        b, c, _, _ = x.size()
        
        # Global average pooling
        avg_pool = self.global_avg_pool(x).view(b, c)
        avg_out = self.fc(avg_pool)
        
        # Global max pooling
        max_pool = self.global_max_pool(x).view(b, c)
        max_out = self.fc(max_pool)
        
        # Combine and apply sigmoid
        attention = self.sigmoid(avg_out + max_out).view(b, c, 1, 1)
        
        return x * attention

class SpatialAttention(nn.Module):
    """Spatial attention mechanism."""
    
    def __init__(self, kernel_size: int = 7):
        """Initialize spatial attention.
        
        Args:
            kernel_size: Convolution kernel size
        """
        super().__init__()
        
        self.conv = nn.Conv2d(2, 1, kernel_size=kernel_size, 
                             padding=kernel_size // 2, bias=False)
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply spatial attention to input."""
        # Channel-wise statistics
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        
        # Concatenate and convolve
        attention_input = torch.cat([avg_out, max_out], dim=1)
        attention = self.sigmoid(self.conv(attention_input))
        
        return x * attention

class ConvolutionalStem(nn.Module):
    """Convolutional stem for initial feature extraction."""
    
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int = 3):
        """Initialize convolutional stem.
        
        Args:
            in_channels: Number of input channels
            out_channels: Number of output channels
            kernel_size: Convolution kernel size
        """
        super().__init__()
        
        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, out_channels // 2, kernel_size=kernel_size, 
                     stride=2, padding=kernel_size // 2, bias=False),
            nn.BatchNorm2d(out_channels // 2),
            nn.GELU(),
            
            nn.Conv2d(out_channels // 2, out_channels // 2, kernel_size=kernel_size,
                     stride=1, padding=kernel_size // 2, bias=False),
            nn.BatchNorm2d(out_channels // 2),
            nn.GELU(),
            
            nn.Conv2d(out_channels // 2, out_channels, kernel_size=kernel_size,
                     stride=2, padding=kernel_size // 2, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.GELU(),
            
            nn.Conv2d(out_channels, out_channels, kernel_size=1, stride=1, bias=False),
            nn.BatchNorm2d(out_channels)
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply convolutional stem to input."""
        return self.stem(x)

class LocalBranch(nn.Module):
    """Local feature extraction branch using depthwise convolutions and attention."""
    
    def __init__(self, dim: int, kernel_size: int = 3, reduction: int = 16):
        """Initialize local branch.
        
        Args:
            dim: Input dimension
            kernel_size: Convolution kernel size
            reduction: Channel reduction ratio for attention
        """
        super().__init__()
        
        # Depthwise separable convolution
        self.depthwise_conv = nn.Conv2d(dim, dim, kernel_size=kernel_size, 
                                       padding=kernel_size // 2, groups=dim, bias=False)
        self.pointwise_conv = nn.Conv2d(dim, dim, kernel_size=1, bias=False)
        
        # Batch normalization and activation
        self.bn1 = nn.BatchNorm2d(dim)
        self.bn2 = nn.BatchNorm2d(dim)
        self.activation = nn.GELU()
        
        # Attention mechanisms
        self.channel_attention = ChannelAttention(dim, reduction)
        self.spatial_attention = SpatialAttention()
        
        # Dropout
        self.dropout = nn.Dropout2d(0.1)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply local branch processing to input."""
        # Depthwise separable convolution
        out = self.depthwise_conv(x)
        out = self.bn1(out)
        out = self.activation(out)
        
        out = self.pointwise_conv(out)
        out = self.bn2(out)
        out = self.activation(out)
        
        # Apply attention mechanisms
        out = self.channel_attention(out)
        out = self.spatial_attention(out)
        
        # Residual connection
        out = out + x
        
        return self.dropout(out)

class GlobalBranch(nn.Module):
    """Global feature extraction branch using multi-head self-attention."""
    
    def __init__(self, dim: int, num_heads: int, reduction: int = 2):
        """Initialize global branch.
        
        Args:
            dim: Input dimension
            num_heads: Number of attention heads
            reduction: Spatial reduction factor for efficiency
        """
        super().__init__()
        
        self.dim = dim
        self.num_heads = num_heads
        self.reduction = reduction
        
        # Spatial reduction convolution
        if reduction > 1:
            self.spatial_reduction = nn.Conv2d(
                dim, dim, kernel_size=reduction, stride=reduction, groups=dim
            )
        else:
            self.spatial_reduction = nn.Identity()
        
        # Multi-head self-attention
        self.multihead_attention = nn.MultiheadAttention(
            embed_dim=dim, 
            num_heads=num_heads, 
            batch_first=False
        )
        
        # Layer normalization
        self.layer_norm = nn.LayerNorm(dim)
        
        # Feed-forward network
        self.ffn = nn.Sequential(
            nn.Linear(dim, dim * 4),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(dim * 4, dim),
            nn.Dropout(0.1)
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply global branch processing to input."""
        B, C, H, W = x.shape
        
        # Spatial reduction for efficiency
        x_reduced = self.spatial_reduction(x)  # Shape: (B, C, H', W')
        _, _, H_red, W_red = x_reduced.shape
        
        # Flatten spatial dimensions for attention
        x_flat = x.flatten(2).permute(2, 0, 1)  # Shape: (H*W, B, C)
        x_reduced_flat = x_reduced.flatten(2).permute(2, 0, 1)  # Shape: (H'*W', B, C)
        
        # Apply multi-head self-attention
        attn_out, _ = self.multihead_attention(
            query=x_flat,
            key=x_reduced_flat,
            value=x_reduced_flat
        )
        
        # Residual connection and layer norm
        attn_out = self.layer_norm(attn_out + x_flat)
        
        # Feed-forward network
        ffn_out = self.ffn(attn_out)
        output = self.layer_norm(ffn_out + attn_out)
        
        # Reshape back to spatial format
        output = output.permute(1, 2, 0).reshape(B, C, H, W)
        
        return output

class FeatureFusion(nn.Module):
    """Feature fusion module for combining local and global features."""
    
    def __init__(self, dim: int, reduction: int = 4):
        """Initialize feature fusion module.
        
        Args:
            dim: Input dimension
            reduction: Dimension reduction factor
        """
        super().__init__()
        
        self.global_avg_pool = nn.AdaptiveAvgPool2d(1)
        
        self.fusion_fc = nn.Sequential(
            nn.Linear(dim * 2, dim * 2 // reduction),
            nn.GELU(),
            nn.Linear(dim * 2 // reduction, dim * 2),
            nn.Sigmoid()
        )
        
        self.combine_conv = nn.Sequential(
            nn.Conv2d(dim * 2, dim, kernel_size=1, bias=False),
            nn.BatchNorm2d(dim),
            nn.GELU()
        )
    
    def forward(self, local_features: torch.Tensor, global_features: torch.Tensor) -> torch.Tensor:
        """Fuse local and global features.
        
        Args:
            local_features: Features from local branch
            global_features: Features from global branch
            
        Returns:
            Fused features
        """
        # Concatenate features
        combined = torch.cat([local_features, global_features], dim=1)
        
        # Global context for fusion weights
        global_context = self.global_avg_pool(combined).flatten(1)
        fusion_weights = self.fusion_fc(global_context).unsqueeze(-1).unsqueeze(-1)
        
        # Apply fusion weights
        weighted_combined = combined * fusion_weights
        
        # Final combination
        output = self.combine_conv(weighted_combined)
        
        return output

class SPENBlock(nn.Module):
    """Spatial-Positional Encoding Network block."""
    
    def __init__(self, dim: int, num_heads: int, kernel_size: int = 3):
        """Initialize SPEN block.
        
        Args:
            dim: Feature dimension
            num_heads: Number of attention heads
            kernel_size: Convolution kernel size
        """
        super().__init__()
        
        # Convolutional positional encoding
        self.conv_pos_encoding = nn.Conv2d(dim, dim, kernel_size=kernel_size, 
                                          padding=kernel_size // 2, groups=dim)
        
        # Local and global branches
        self.local_branch = LocalBranch(dim, kernel_size)
        self.global_branch = GlobalBranch(dim, num_heads)
        
        # Feature fusion
        self.feature_fusion = FeatureFusion(dim)
        
        # Layer normalization
        self.layer_norm = nn.LayerNorm(dim)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply SPEN block processing to input."""
        # Convolutional positional encoding
        pos_encoded = x + self.conv_pos_encoding(x)
        
        # Process through local and global branches
        local_out = self.local_branch(pos_encoded)
        global_out = self.global_branch(pos_encoded)
        
        # Fuse features
        fused = self.feature_fusion(local_out, global_out)
        
        # Layer normalization (channel-wise)
        B, C, H, W = fused.shape
        fused_flat = fused.permute(0, 2, 3, 1).reshape(-1, C)
        normalized = self.layer_norm(fused_flat)
        output = normalized.reshape(B, H, W, C).permute(0, 3, 1, 2)
        
        return output

class CLVTNet(nn.Module):
    """Convolutional-LSTM Vision Transformer Network for EEG Classification."""
    
    def __init__(self, config: Config):
        """Initialize CLVTNet model.
        
        Args:
            config: Configuration object containing model parameters
        """
        super().__init__()
        
        self.config = config
        model_config = config.model
        
        # Input processing
        self.conv_stem = ConvolutionalStem(
            model_config.input_channels, 
            model_config.embed_dims[0],
            model_config.filter_sizes[0]
        )
        
        # Multi-stage processing
        self.stages = nn.ModuleList()
        
        for i, (dim, num_heads, num_blocks, filter_size) in enumerate(
            zip(model_config.embed_dims, model_config.num_heads, 
                model_config.num_blocks, model_config.filter_sizes)
        ):
            
            # Downsampling layer (except for first stage)
            if i > 0:
                downsample = nn.Sequential(
                    nn.Conv2d(model_config.embed_dims[i-1], dim, 
                             kernel_size=2, stride=2, bias=False),
                    nn.BatchNorm2d(dim)
                )
            else:
                downsample = nn.Identity()
            
            # Create stage with multiple SPEN blocks
            stage_blocks = nn.ModuleList([
                SPENBlock(dim, num_heads, filter_size) 
                for _ in range(num_blocks)
            ])
            
            stage = nn.Sequential(downsample, *stage_blocks)
            self.stages.append(stage)
        
        # Global average pooling
        self.global_avg_pool = nn.AdaptiveAvgPool2d(1)
        
        # Classification head
        final_dim = model_config.embed_dims[-1]
        self.classifier = nn.Sequential(
            nn.Dropout(model_config.dropout_rate),
            nn.Linear(final_dim, final_dim // 2),
            nn.GELU(),
            nn.Dropout(model_config.dropout_rate / 2),
            nn.Linear(final_dim // 2, model_config.num_classes)
        )
        
        # Initialize weights
        self.apply(self._init_weights)
    
    def _init_weights(self, module):
        """Initialize model weights."""
        if isinstance(module, nn.Linear):
            nn.init.trunc_normal_(module.weight, std=0.02)
            if module.bias is not None:
                nn.init.constant_(module.bias, 0)
        elif isinstance(module, nn.Conv2d):
            nn.init.kaiming_normal_(module.weight, mode='fan_out', nonlinearity='relu')
            if module.bias is not None:
                nn.init.constant_(module.bias, 0)
        elif isinstance(module, (nn.BatchNorm2d, nn.LayerNorm)):
            nn.init.constant_(module.bias, 0)
            nn.init.constant_(module.weight, 1.0)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through CLVTNet.
        
        Args:
            x: Input tensor (batch_size, channels, height, width)
            
        Returns:
            Classification logits (batch_size, num_classes)
        """
        # Initial feature extraction
        features = self.conv_stem(x)
        
        # Multi-stage processing
        for stage in self.stages:
            features = stage(features)
        
        # Global average pooling
        pooled = self.global_avg_pool(features)
        pooled = pooled.flatten(1)
        
        # Classification
        output = self.classifier(pooled)
        
        return output
    
    def extract_features(self, x: torch.Tensor, layer_idx: Optional[int] = None) -> torch.Tensor:
        """Extract features from intermediate layers.
        
        Args:
            x: Input tensor
            layer_idx: Index of layer to extract features from (None for final features)
            
        Returns:
            Extracted features
        """
        features = self.conv_stem(x)
        
        for i, stage in enumerate(self.stages):
            features = stage(features)
            if layer_idx is not None and i == layer_idx:
                return features
        
        return features
    
    def get_attention_maps(self, x: torch.Tensor) -> List[torch.Tensor]:
        """Extract attention maps from all stages.
        
        Args:
            x: Input tensor
            
        Returns:
            List of attention maps from each stage
        """
        attention_maps = []
        features = self.conv_stem(x)
        
        for stage in self.stages:
            # Process through stage and collect attention
            if hasattr(stage, '__iter__'):
                for block in stage:
                    if isinstance(block, SPENBlock):
                        # Extract attention from global branch
                        global_branch = block.global_branch
                        if hasattr(global_branch, 'multihead_attention'):
                            # This would require modifying the forward pass to return attention
                            # For now, we'll return the features as a proxy
                            attention_maps.append(features.mean(dim=1, keepdim=True))
            features = stage(features)
        
        return attention_maps
    
    def compute_model_complexity(self) -> Dict[str, int]:
        """Compute model complexity metrics."""
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        
        # Count FLOPs for a typical input
        input_size = (1, self.config.model.input_channels, 
                     self.config.model.image_size, self.config.model.image_size)
        
        # Approximate FLOP calculation
        flops = 0
        current_size = input_size
        
        # Conv stem FLOPs
        stem_params = sum(p.numel() for p in self.conv_stem.parameters())
        flops += stem_params * current_size[2] * current_size[3]
        
        # Update size after stem (assuming 4x downsampling)
        current_size = (current_size[0], self.config.model.embed_dims[0], 
                       current_size[2] // 4, current_size[3] // 4)
        
        # Stage FLOPs
        for i, (dim, num_blocks) in enumerate(
            zip(self.config.model.embed_dims, self.config.model.num_blocks)
        ):
            if i > 0:
                current_size = (current_size[0], dim, 
                               current_size[2] // 2, current_size[3] // 2)
            
            # Approximate FLOPs for each block
            spatial_size = current_size[2] * current_size[3]
            block_flops = dim * spatial_size * 10  # Rough approximation
            flops += block_flops * num_blocks
        
        # Classifier FLOPs
        classifier_params = sum(p.numel() for p in self.classifier.parameters())
        flops += classifier_params
        
        return {
            'total_parameters': total_params,
            'trainable_parameters': trainable_params,
            'approximate_flops': flops,
            'model_size_mb': total_params * 4 / (1024 * 1024)  # Assuming float32
        }

class ModelFactory:
    """Factory class for creating different model variants."""
    
    @staticmethod
    def create_tiny_model(num_classes: int = 2, input_channels: int = 1, 
                         image_size: int = 64) -> CLVTNet:
        """Create a tiny model variant for fast experimentation."""
        from src.config import Config, ModelConfig
        
        model_config = ModelConfig(
            input_channels=input_channels,
            num_classes=num_classes,
            image_size=image_size,
            embed_dims=[32, 64, 128, 256],
            num_heads=[2, 4, 8, 16],
            filter_sizes=[3, 3, 5, 5],
            num_blocks=[1, 1, 2, 1],
            dropout_rate=0.1
        )
        
        config = Config(model=model_config)
        return CLVTNet(config)
    
    @staticmethod
    def create_small_model(num_classes: int = 2, input_channels: int = 1,
                          image_size: int = 64) -> CLVTNet:
        """Create a small model variant."""
        from src.config import Config, ModelConfig
        
        model_config = ModelConfig(
            input_channels=input_channels,
            num_classes=num_classes,
            image_size=image_size,
            embed_dims=[64, 128, 256, 512],
            num_heads=[4, 8, 16, 32],
            filter_sizes=[3, 5, 7, 9],
            num_blocks=[2, 2, 4, 2],
            dropout_rate=0.1
        )
        
        config = Config(model=model_config)
        return CLVTNet(config)
    
    @staticmethod
    def create_base_model(num_classes: int = 2, input_channels: int = 1,
                         image_size: int = 64) -> CLVTNet:
        """Create the base model variant (default configuration)."""
        from src.config import Config
        
        config = Config()
        config.model.num_classes = num_classes
        config.model.input_channels = input_channels
        config.model.image_size = image_size
        
        return CLVTNet(config)
    
    @staticmethod
    def create_large_model(num_classes: int = 2, input_channels: int = 1,
                          image_size: int = 64) -> CLVTNet:
        """Create a large model variant for best performance."""
        from src.config import Config, ModelConfig
        
        model_config = ModelConfig(
            input_channels=input_channels,
            num_classes=num_classes,
            image_size=image_size,
            embed_dims=[96, 192, 384, 768],
            num_heads=[6, 12, 24, 48],
            filter_sizes=[3, 5, 7, 9],
            num_blocks=[2, 4, 8, 4],
            dropout_rate=0.1
        )
        
        config = Config(model=model_config)
        return CLVTNet(config)

def test_model_creation():
    """Test function for model creation and forward pass."""
    import torch
    from src.config import Config
    
    # Create model with default configuration
    config = Config()
    model = CLVTNet(config)
    
    # Test forward pass
    batch_size = 4
    input_tensor = torch.randn(batch_size, config.model.input_channels, 
                              config.model.image_size, config.model.image_size)
    
    with torch.no_grad():
        output = model(input_tensor)
    
    print(f"Model created successfully!")
    print(f"Input shape: {input_tensor.shape}")
    print(f"Output shape: {output.shape}")
    
    # Print model complexity
    complexity = model.compute_model_complexity()
    print(f"Model complexity: {complexity}")
    
    return model

if __name__ == "__main__":
    test_model_creation()