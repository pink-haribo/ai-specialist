"""
Attention Modules for GAIN-MTL

Implements various attention mechanisms:
- GAIN (Guided Attention Inference Network) style attention
- CBAM (Convolutional Block Attention Module)
- Channel and Spatial Attention

References:
- GAIN: "Tell Me Where to Look" (CVPR 2018)
- CBAM: "CBAM: Convolutional Block Attention Module" (ECCV 2018)
"""

from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class ChannelAttention(nn.Module):
    """
    Channel Attention Module from CBAM.

    Focuses on 'what' is meaningful in the feature maps by
    exploiting inter-channel relationships.

    Args:
        in_channels: Number of input channels
        reduction_ratio: Reduction ratio for the bottleneck
    """

    def __init__(self, in_channels: int, reduction_ratio: int = 16):
        super().__init__()

        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        # Shared MLP
        reduced_channels = max(in_channels // reduction_ratio, 8)
        self.mlp = nn.Sequential(
            nn.Conv2d(in_channels, reduced_channels, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(reduced_channels, in_channels, 1, bias=False),
        )

        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input tensor of shape (B, C, H, W)

        Returns:
            Channel attention weights of shape (B, C, 1, 1)
        """
        avg_out = self.mlp(self.avg_pool(x))
        max_out = self.mlp(self.max_pool(x))
        attention = self.sigmoid(avg_out + max_out)
        return attention


class SpatialAttention(nn.Module):
    """
    Spatial Attention Module from CBAM.

    Focuses on 'where' is an informative part by utilizing
    inter-spatial relationships.

    Args:
        kernel_size: Convolution kernel size
    """

    def __init__(self, kernel_size: int = 7):
        super().__init__()

        padding = kernel_size // 2
        self.conv = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input tensor of shape (B, C, H, W)

        Returns:
            Spatial attention weights of shape (B, 1, H, W)
        """
        # Channel-wise pooling
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)

        # Concatenate and convolve
        concat = torch.cat([avg_out, max_out], dim=1)
        attention = self.sigmoid(self.conv(concat))

        return attention


class CBAM(nn.Module):
    """
    Convolutional Block Attention Module.

    Combines channel and spatial attention sequentially.

    Args:
        in_channels: Number of input channels
        reduction_ratio: Reduction ratio for channel attention
        spatial_kernel_size: Kernel size for spatial attention
    """

    def __init__(
        self,
        in_channels: int,
        reduction_ratio: int = 16,
        spatial_kernel_size: int = 7,
    ):
        super().__init__()

        self.channel_attention = ChannelAttention(in_channels, reduction_ratio)
        self.spatial_attention = SpatialAttention(spatial_kernel_size)

    def forward(
        self,
        x: torch.Tensor,
        return_attention: bool = False
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Args:
            x: Input tensor of shape (B, C, H, W)
            return_attention: Whether to return attention maps

        Returns:
            Attended features and optionally attention maps
        """
        # Channel attention
        ca = self.channel_attention(x)
        x = x * ca

        # Spatial attention
        sa = self.spatial_attention(x)
        out = x * sa

        if return_attention:
            return out, sa
        return out, None


class GuidedAttentionModule(nn.Module):
    """
    GAIN-style Guided Attention Module.

    This module learns attention maps that can be supervised with
    external guidance (e.g., defect masks) to ensure the model
    focuses on the correct regions.

    Key features:
    - Learnable attention generation
    - Multi-scale feature processing
    - External supervision compatible

    Args:
        in_channels: Number of input channels
        attention_channels: Intermediate channels for attention computation
        use_cbam: Whether to include CBAM-style attention
    """

    def __init__(
        self,
        in_channels: int,
        attention_channels: int = 512,
        use_cbam: bool = True,
    ):
        super().__init__()

        self.use_cbam = use_cbam

        # CBAM for initial feature refinement
        if use_cbam:
            self.cbam = CBAM(in_channels)

        # Attention generation network
        self.attention_conv = nn.Sequential(
            nn.Conv2d(in_channels, attention_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(attention_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(attention_channels, attention_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(attention_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(attention_channels, 1, 1),
        )

        # Learnable temperature for attention sharpening
        self.temperature = nn.Parameter(torch.ones(1))

        self._attention_logits = None

    def forward(
        self,
        x: torch.Tensor,
        return_attention: bool = True
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Generate attention-weighted features.

        Args:
            x: Input tensor of shape (B, C, H, W)
            return_attention: Whether to return the attention map

        Returns:
            Tuple of (attended features, attention logits)
            Note: Returns logits (pre-sigmoid) for numerical stability with BCE loss.
                  Apply sigmoid externally when probability values are needed.
        """
        # Apply CBAM if enabled
        if self.use_cbam:
            x, _ = self.cbam(x)

        # Generate attention logits (pre-sigmoid for numerical stability)
        attention_logits = self.attention_conv(x) / self.temperature

        # Store for external access (e.g., visualization)
        self._attention_logits = attention_logits

        # Apply attention using sigmoid for feature weighting
        attention_map = torch.sigmoid(attention_logits)
        attended_features = x * attention_map

        # Return logits for loss computation (BCE with logits)
        return attended_features, attention_logits

    def get_attention_logits(self) -> Optional[torch.Tensor]:
        """Return the last computed attention logits (pre-sigmoid)."""
        return self._attention_logits

    def get_attention_map(self) -> Optional[torch.Tensor]:
        """Return the last computed attention map (with sigmoid applied)."""
        if self._attention_logits is not None:
            return torch.sigmoid(self._attention_logits)
        return None


class AttentionMiningHead(nn.Module):
    """
    GAIN Attention Mining Stream (S_am).

    This head generates attention maps that are trained to highlight
    discriminative regions for classification.

    From GAIN paper: "The attention mining stream is designed to
    discover the most discriminative regions."

    Args:
        in_channels: Number of input channels
        hidden_channels: Hidden layer channels
    """

    def __init__(self, in_channels: int, hidden_channels: int = 256):
        super().__init__()

        # Note: No sigmoid at the end - returns logits for numerical stability
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, hidden_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(hidden_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_channels, hidden_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(hidden_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_channels, 1, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Generate attention logits.

        Args:
            x: Input features (B, C, H, W)

        Returns:
            Attention logits (B, 1, H, W) - apply sigmoid externally for probability
        """
        return self.conv(x)


class MultiScaleAttention(nn.Module):
    """
    Multi-Scale Attention Module.

    Processes features at multiple scales and fuses attention maps
    for better localization of defects of varying sizes.

    Args:
        in_channels_list: List of input channels for each scale
        out_channels: Output channels after fusion
    """

    def __init__(
        self,
        in_channels_list: list,
        out_channels: int = 256,
    ):
        super().__init__()

        self.num_scales = len(in_channels_list)

        # Per-scale attention modules
        self.scale_attentions = nn.ModuleList([
            GuidedAttentionModule(in_ch, out_channels)
            for in_ch in in_channels_list
        ])

        # Feature adapters to unify channels
        self.adapters = nn.ModuleList([
            nn.Conv2d(in_ch, out_channels, 1)
            for in_ch in in_channels_list
        ])

        # Fusion weights (learnable)
        self.fusion_weights = nn.Parameter(torch.ones(self.num_scales))

    def forward(
        self,
        features_list: list,
        target_size: Optional[Tuple[int, int]] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Process multi-scale features and generate fused attention.

        Args:
            features_list: List of feature tensors at different scales
            target_size: Target spatial size for resizing

        Returns:
            Tuple of (fused features, fused attention map)
        """
        if target_size is None:
            target_size = features_list[0].shape[2:]

        attended_features = []
        attention_maps = []
        weights = F.softmax(self.fusion_weights, dim=0)

        for i, (feat, attn_module, adapter) in enumerate(
            zip(features_list, self.scale_attentions, self.adapters)
        ):
            # Generate attention
            attn_feat, attn_map = attn_module(feat)

            # Adapt channels
            adapted = adapter(attn_feat)

            # Resize to target size
            if adapted.shape[2:] != target_size:
                adapted = F.interpolate(
                    adapted, size=target_size, mode='bilinear', align_corners=False
                )
                attn_map = F.interpolate(
                    attn_map, size=target_size, mode='bilinear', align_corners=False
                )

            attended_features.append(adapted * weights[i])
            attention_maps.append(attn_map * weights[i])

        # Fuse features and attention maps
        fused_features = sum(attended_features)
        fused_attention = sum(attention_maps)

        return fused_features, fused_attention
