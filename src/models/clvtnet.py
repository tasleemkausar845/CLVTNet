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
    """Squeeze-and-Excitation (SE) channel attention mechanism."""
    
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
    """Convolutional stem for initial feature extraction (3×3 conv layers, stride pattern per text)."""
    
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int = 3):
        """
        Args:
            in_channels: Number of input channels
            out_channels: Number of output channels
            kernel_size: Convolution kernel size
        """
        super().__init__()
        
        self.stem = nn.Sequential(
    nn.Conv2d(in_channels, out_channels // 2, kernel_size=3, stride=2, padding=1, bias=False),  # 1st
    nn.BatchNorm2d(out_channels // 2),
    nn.GELU(),

    nn.Conv2d(out_channels // 2, out_channels // 2, kernel_size=3, stride=2, padding=1, bias=False),  # 2nd
    nn.BatchNorm2d(out_channels // 2),
    nn.GELU(),

    nn.Conv2d(out_channels // 2, out_channels // 2, kernel_size=3, stride=1, padding=1, bias=False),  # 3rd
    nn.BatchNorm2d(out_channels // 2),
    nn.GELU(),

    nn.Conv2d(out_channels // 2, out_channels // 2, kernel_size=3, stride=1, padding=1, bias=False),  # 4th
    nn.BatchNorm2d(out_channels // 2),
    nn.GELU(),

    nn.Conv2d(out_channels // 2, out_channels, kernel_size=1, stride=1, bias=False),  # final 1x1
    nn.BatchNorm2d(out_channels)
)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply convolutional stem."""
        return self.stem(x)


class ConvModule(nn.Module):
    """Convolutional module with different variants for ablation studies."""
    
    def __init__(self, dim: int, kernel_size: int = 3, variant: str = 'lite'):
        """Initialize convolutional module.
        
        Args:
            dim: Input/output dimension
            kernel_size: Convolution kernel size
            variant: Module variant ('standard', 'lite', 'heavy')
        """
        super().__init__()
        
        self.variant = variant
        
        if variant == 'standard':
            # Standard 3x3 conv
            self.conv = nn.Sequential(
                nn.Conv2d(dim, dim, kernel_size=3, padding=1, bias=False),
                nn.BatchNorm2d(dim),
                nn.GELU()
            )
        elif variant == 'lite':
            # Depthwise separable conv (lighter)
            self.conv = nn.Sequential(
                nn.Conv2d(dim, dim, kernel_size=3, padding=1, groups=dim, bias=False),
                nn.BatchNorm2d(dim),
                nn.Conv2d(dim, dim, kernel_size=1, bias=False),
                nn.BatchNorm2d(dim),
                nn.GELU()
            )
        elif variant == 'heavy':
            # Heavier conv with more parameters
            self.conv = nn.Sequential(
                nn.Conv2d(dim, dim * 2, kernel_size=3, padding=1, bias=False),
                nn.BatchNorm2d(dim * 2),
                nn.GELU(),
                nn.Conv2d(dim * 2, dim, kernel_size=1, bias=False),
                nn.BatchNorm2d(dim),
                nn.GELU()
            )
        else:
            raise ValueError(f"Unknown conv module variant: {variant}")
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply convolutional module."""
        return self.conv(x)


class LocalBranch(nn.Module):
    """Local feature extraction branch using depthwise convolutions and attention."""
    
    def __init__(self, dim: int, kernel_size: int = 3, reduction: int = 16, 
                 use_se: bool = False, use_conv: bool = True, conv_variant: str = 'lite'):
        """Initialize local branch.
        
        Args:
            dim: Input dimension
            kernel_size: Convolution kernel size
            reduction: Channel reduction ratio for attention
            use_se: Whether to use SE attention in local branch
            use_conv: Whether to use convolutional module
            conv_variant: Variant of convolutional module
        """
        super().__init__()
        
        self.use_se = use_se
        self.use_conv = use_conv
        
        # Depthwise separable convolution
        self.depthwise_conv = nn.Conv2d(dim, dim, kernel_size=kernel_size, 
                                       padding=kernel_size // 2, groups=dim, bias=False)
        self.pointwise_conv = nn.Conv2d(dim, dim, kernel_size=1, bias=False)
        
        # Batch normalization and activation
        self.bn1 = nn.BatchNorm2d(dim)
        self.bn2 = nn.BatchNorm2d(dim)
        self.activation = nn.GELU()
        
        # Optional SE attention in local branch
        if use_se:
            self.channel_attention = ChannelAttention(dim, reduction)
        else:
            self.channel_attention = nn.Identity()
        
        # Spatial attention
        self.spatial_attention = SpatialAttention()
        
        # Optional convolutional module
        if use_conv:
            self.conv_module = ConvModule(dim, kernel_size=3, variant=conv_variant)
        else:
            self.conv_module = nn.Identity()
        
        # Dropout
        self.dropout = nn.Dropout2d(0.1)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply local branch processing to input."""
        identity = x
        
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
        
        # Apply conv module if enabled
        if self.use_conv:
            out = self.conv_module(out)
        
        # Residual connection
        out = out + identity
        
        return self.dropout(out)


class TokenReduction(nn.Module):
    """Token reduction module with different methods for ablation studies."""
    
    def __init__(self, dim: int, reduction_factor: int = 2, method: str = 'ctr'):
        """Initialize token reduction.
        
        Args:
            dim: Input dimension
            reduction_factor: Reduction factor (e.g., 2 means reduce by half)
            method: Reduction method ('ctr', 'uniform', 'mean')
        """
        super().__init__()
        
        self.reduction_factor = reduction_factor
        self.method = method
        
        if method == 'ctr':
            # Content-aware token reduction with learned projection
	     self.score_net = nn.Sequential(
    		nn.Conv2d(dim, dim // 4, kernel_size=3, 
              		stride=reduction_factor, padding=1, bias=False),
   		nn.BatchNorm2d(dim // 4),
    		nn.GELU(),
    		nn.Conv2d(dim // 4, 1, kernel_size=1, bias=False)
	      )
          
        elif method == 'uniform':
            # Uniform sampling (no learnable parameters)
            pass
        elif method == 'mean':
            # Average pooling
            self.pool = nn.AvgPool2d(kernel_size=reduction_factor, stride=reduction_factor)
        else:
            raise ValueError(f"Unknown token reduction method: {method}")
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply token reduction."""
        if self.reduction_factor == 1:
            return x
        
        if self.method == 'ctr':
            # Content-aware token reduction
            B, C, H, W = x.shape
            scores = self.score_net(x)  # B, 1, H, W
            scores = scores.view(B, 1, -1)  # B, 1, H*W
            scores = F.softmax(scores, dim=-1)
            
            # Keep top-k tokens based on scores
            k = (H * W) // (self.reduction_factor ** 2)
            topk_indices = torch.topk(scores, k, dim=-1)[1]  # B, 1, k
            
            # Reshape and gather
            x_flat = x.view(B, C, -1)  # B, C, H*W
            topk_indices_expanded = topk_indices.expand(B, C, k)
            reduced = torch.gather(x_flat, 2, topk_indices_expanded)  # B, C, k
            
            # Reshape to spatial format
            new_h = new_w = int(math.sqrt(k))
            reduced = reduced.view(B, C, new_h, new_w)
            return reduced
            
        elif self.method == 'uniform':
            # Uniform spatial downsampling
            return F.avg_pool2d(x, kernel_size=self.reduction_factor, stride=self.reduction_factor)
            
        elif self.method == 'mean':
            # Average pooling
            return self.pool(x)


class GlobalBranch(nn.Module):
    """Global feature extraction branch using multi-head self-attention."""
    
    def __init__(self, dim: int, num_heads: int, reduction_factor: int = 2, 
                 token_reduction_method: str = 'ctr'):
        """Initialize global branch.
        
        Args:
            dim: Input dimension
            num_heads: Number of attention heads
            reduction_factor: Token reduction factor for efficiency
            token_reduction_method: Method for token reduction
        """
        super().__init__()
        
        self.dim = dim
        self.num_heads = num_heads
        
        # Token reduction for efficiency
        self.token_reduction = TokenReduction(dim, reduction_factor, token_reduction_method)
        
        # Multi-head self-attention
        self.multihead_attention = nn.MultiheadAttention(
            embed_dim=dim, 
            num_heads=num_heads, 
            batch_first=False
        )
        
        # Layer normalization
        self.layer_norm1 = nn.LayerNorm(dim)
        self.layer_norm2 = nn.LayerNorm(dim)
        
        # Feed-forward network
        self.ffn = nn.Sequential(
            nn.Linear(dim, dim * 4),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(dim * 4, dim),
            nn.Dropout(0.1)
        )
        
        # Dropout
        self.dropout = nn.Dropout(0.1)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply global branch processing to input."""
        B, C, H, W = x.shape
        
        # Apply token reduction
        x_reduced = self.token_reduction(x)
        _, _, H_r, W_r = x_reduced.shape
        
        # Reshape for attention: (H*W, B, C)
        x_seq = x_reduced.flatten(2).permute(2, 0, 1)
        
        # Apply layer normalization
        x_norm = self.layer_norm1(x_seq)
        
        # Multi-head self-attention
        attn_out, _ = self.multihead_attention(x_norm, x_norm, x_norm)
        attn_out = self.dropout(attn_out)
        
        # Residual connection
        x_seq = x_seq + attn_out
        
        # Feed-forward network with residual
        x_norm = self.layer_norm2(x_seq)
        ffn_out = self.ffn(x_norm)
        x_seq = x_seq + ffn_out
        
        # Reshape back to spatial format
        output = x_seq.permute(1, 2, 0).reshape(B, C, H_r, W_r)
        
        # Upsample back to original spatial size if reduced
        if H_r != H or W_r != W:
            output = F.interpolate(output, size=(H, W), mode='bilinear', align_corners=False)
        
        return output


class FeatureFusion(nn.Module):
    """Feature fusion module with optional SE attention."""
    
    def __init__(self, dim: int, use_se: bool = True, se_reduction: int = 4):
        """Initialize feature fusion.
        
        Args:
            dim: Input dimension
            use_se: Whether to use SE attention for fusion
            se_reduction: SE reduction ratio
        """
        super().__init__()
        
        self.use_se = use_se
        
        # Fusion convolution
        self.fusion_conv = nn.Sequential(
            nn.Conv2d(dim * 2, dim, kernel_size=1, bias=False),
            nn.BatchNorm2d(dim),
            nn.GELU()
        )
        
        # Optional SE attention for fusion
        if use_se:
            self.se_attention = ChannelAttention(dim, reduction=se_reduction)
        else:
            self.se_attention = nn.Identity()
    
    def forward(self, local_feat: torch.Tensor, global_feat: torch.Tensor) -> torch.Tensor:
        """Fuse local and global features.
        
        Args:
            local_feat: Local branch features
            global_feat: Global branch features
            
        Returns:
            Fused features
        """
        # Concatenate features
        fused = torch.cat([local_feat, global_feat], dim=1)
        
        # Apply fusion convolution
        fused = self.fusion_conv(fused)
        
        # Apply SE attention if enabled
        fused = self.se_attention(fused)
        
        return fused


class ConvPositionalEncoding(nn.Module):
    """Convolutional positional encoding."""
    
    def __init__(self, dim: int, kernel_size: int = 3):
        """Initialize convolutional positional encoding.
        
        Args:
            dim: Input dimension
            kernel_size: Convolution kernel size
        """
        super().__init__()
        
        self.conv = nn.Conv2d(dim, dim, kernel_size=kernel_size, 
                             padding=kernel_size // 2, groups=dim, bias=False)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply convolutional positional encoding."""
        return self.conv(x)


class SPENBlock(nn.Module):
    """Spatial-channel Parallel Enhancement Network block."""
    
    def __init__(self, dim: int, num_heads: int, filter_size: int, 
                 use_local: bool = True, use_global: bool = True,
                 use_se_in_fusion: bool = True, use_se_in_local: bool = False,
                 se_reduction_ratio: int = 4, use_conv_module: bool = True,
                 conv_module_variant: str = 'lite',
                 token_reduction_factor: int = 2,
                 token_reduction_method: str = 'ctr'):
        """Initialize SPEN block with ablation support.
        
        Args:
            dim: Input dimension
            num_heads: Number of attention heads
            filter_size: Filter size for convolutions
            use_local: Whether to use local branch
            use_global: Whether to use global branch
            use_se_in_fusion: Whether to use SE in fusion module
            use_se_in_local: Whether to use SE in local branch
            se_reduction_ratio: SE reduction ratio
            use_conv_module: Whether to use convolutional module
            conv_module_variant: Variant of convolutional module
            token_reduction_factor: Token reduction factor for global branch
            token_reduction_method: Token reduction method
        """
        super().__init__()
        
        self.use_local = use_local
        self.use_global = use_global
        
        if not use_local and not use_global:
            raise ValueError("At least one branch must be enabled")
        
        # Convolutional positional encoding
        self.conv_pos_encoding = ConvPositionalEncoding(dim, kernel_size=3)
        
        # Local branch
        if use_local:
            self.local_branch = LocalBranch(
                dim, filter_size, reduction=16, 
                use_se=use_se_in_local,
                use_conv=use_conv_module,
                conv_variant=conv_module_variant
            )
        else:
            self.local_branch = None
        
        # Global branch
        if use_global:
            self.global_branch = GlobalBranch(
                dim, num_heads,
                reduction_factor=token_reduction_factor,
                token_reduction_method=token_reduction_method
            )
        else:
            self.global_branch = None
        
        # Feature fusion
        if use_local and use_global:
            self.feature_fusion = FeatureFusion(dim, use_se=use_se_in_fusion, 
                                               se_reduction=se_reduction_ratio)
            self.layer_norm = nn.LayerNorm(dim)
        else:
            # If only one branch, no fusion needed
            self.feature_fusion = None
            self.layer_norm = nn.LayerNorm(dim)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through SPEN block.
        
        Args:
            x: Input tensor (B, C, H, W)
            
        Returns:
            Output tensor (B, C, H, W)
        """
        # Convolutional positional encoding
        pos_encoded = x + self.conv_pos_encoding(x)
        
        # Process through branches
        if self.use_local and self.use_global:
            # Both branches enabled
            local_out = self.local_branch(pos_encoded)
            global_out = self.global_branch(pos_encoded)
            
            # Fuse features
            fused = self.feature_fusion(local_out, global_out)
            
        elif self.use_local:
            # Only local branch
            fused = self.local_branch(pos_encoded)
            
        elif self.use_global:
            # Only global branch
            fused = self.global_branch(pos_encoded)
        
        # Layer normalization (channel-wise)
        B, C, H, W = fused.shape
        fused_flat = fused.permute(0, 2, 3, 1).reshape(-1, C)
        normalized = self.layer_norm(fused_flat)
        output = normalized.reshape(B, H, W, C).permute(0, 3, 1, 2)
        
        return output


class CLVTNet(nn.Module):
    """Convolutional-LSTM Vision Transformer Network for EEG Classification."""
    
    def __init__(self, config: Config, ablation_config=None):
        """Initialize CLVTNet model.
        
        Args:
            config: Configuration object containing model parameters
            ablation_config: Optional AblationConfig for ablation studies
        """
        super().__init__()
        
        self.config = config
        self.ablation_config = ablation_config
        model_config = config.model
        
        # Extract ablation parameters if provided
        if ablation_config is not None:
            use_local = ablation_config.use_local_branch
            use_global = ablation_config.use_global_branch
            token_reduction_method = ablation_config.token_reduction_method
            token_reduction_factor = ablation_config.token_reduction_factor
            use_se_in_fusion = ablation_config.use_se_in_fusion
            use_se_in_local = ablation_config.use_se_in_local
            se_reduction_ratio = ablation_config.se_reduction_ratio
            use_conv_module = ablation_config.use_conv_module
            conv_module_variant = ablation_config.conv_module_variant
        else:
            # Default configuration (full model)
            use_local = True
            use_global = True
            token_reduction_method = 'ctr'
            token_reduction_factor = 2
            use_se_in_fusion = True
            use_se_in_local = False
            se_reduction_ratio = 4
            use_conv_module = True
            conv_module_variant = 'lite'
        
        # Adjust input channels for multiband SSF if enabled
        input_channels = model_config.input_channels
        if hasattr(config.data, 'use_multiband_ssf') and config.data.use_multiband_ssf:
            input_channels = len(config.data.frequency_bands)
        
        # Input processing
        self.conv_stem = ConvolutionalStem(
            input_channels, 
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
                SPENBlock(
                    dim, num_heads, filter_size,
                    use_local=use_local,
                    use_global=use_global,
                    use_se_in_fusion=use_se_in_fusion,
                    use_se_in_local=use_se_in_local,
                    se_reduction_ratio=se_reduction_ratio,
                    use_conv_module=use_conv_module,
                    conv_module_variant=conv_module_variant,
                    token_reduction_factor=token_reduction_factor,
                    token_reduction_method=token_reduction_method
                ) 
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
                        if block.global_branch is not None:
                            # Return features as proxy for attention
                            attention_maps.append(features.mean(dim=1, keepdim=True))
            features = stage(features)
        
        return attention_maps
    
    def compute_model_complexity(self) -> Dict[str, float]:
        """Compute model complexity metrics.
        
        Returns:
            Dictionary with parameter counts, FLOPs, and model size
        """
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
        from config import Config, ModelConfig
        
        model_config = ModelConfig(
            input_channels=input_channels,
            num_classes=num_classes,
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
        from config import Config, ModelConfig
        
        model_config = ModelConfig(
            input_channels=input_channels,
            num_classes=num_classes,
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
        from config import Config
        
        config = Config()
        config.model.num_classes = num_classes
        config.model.input_channels = input_channels
        
        return CLVTNet(config)
    
    @staticmethod
    def create_large_model(num_classes: int = 2, input_channels: int = 1,
                          image_size: int = 64) -> CLVTNet:
        """Create a large model variant for best performance."""
        from config import Config, ModelConfig
        
        model_config = ModelConfig(
            input_channels=input_channels,
            num_classes=num_classes,
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
    from config import Config
    from ablation_configs import AblationConfig
    
    # Test 1: Create model with default configuration
    print("Test 1: Default configuration")
    config = Config()
    model = CLVTNet(config)
    
    # Test forward pass
    batch_size = 4
    input_tensor = torch.randn(batch_size, config.model.input_channels, 
                              64, 64)
    
    with torch.no_grad():
        output = model(input_tensor)
    
    print(f"Model created successfully!")
    print(f"Input shape: {input_tensor.shape}")
    print(f"Output shape: {output.shape}")
    
    # Print model complexity
    complexity = model.compute_model_complexity()
    print(f"Model complexity: {complexity}\n")
    
    # Test 2: Create model with ablation config (CLVT-G)
    print("Test 2: CLVT-G (Global only)")
    ablation_config = AblationConfig(
        use_local_branch=False,
        use_global_branch=True,
        use_se_in_fusion=False,
        use_conv_module=False
    )
    model_ablation = CLVTNet(config, ablation_config=ablation_config)
    
    with torch.no_grad():
        output_ablation = model_ablation(input_tensor)
    
    print(f"CLVT-G created successfully!")
    print(f"Output shape: {output_ablation.shape}")
    complexity_ablation = model_ablation.compute_model_complexity()
    print(f"Model complexity: {complexity_ablation}\n")
    
    # Test 3: Create model with multiband SSF
    print("Test 3: Multiband SSF")
    config_multiband = Config()
    config_multiband.data.use_multiband_ssf = True
    config_multiband.model.input_channels = 3  # alpha, beta, gamma
    
    model_multiband = CLVTNet(config_multiband)
    input_multiband = torch.randn(batch_size, 3, 64, 64)
    
    with torch.no_grad():
        output_multiband = model_multiband(input_multiband)
    
    print(f"Multiband model created successfully!")
    print(f"Input shape: {input_multiband.shape}")
    print(f"Output shape: {output_multiband.shape}\n")
    
    return model


if __name__ == "__main__":
    test_model_creation()