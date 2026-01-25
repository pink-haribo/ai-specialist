"""
Task-specific Heads for GAIN-MTL

Implements:
- Classification Head: Binary/multi-class defect classification
- Localization Head: Defect region segmentation
- Feature Pyramid Network: Multi-scale feature fusion

Reference:
- CXR-MultiTaskNet (Nature 2025) for joint architecture design
"""

from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class ClassificationHead(nn.Module):
    """
    Classification Head for defect/non-defect prediction.

    Features:
    - Global Average Pooling
    - Dropout for regularization
    - Multi-layer classifier with optional intermediate features

    Args:
        in_channels: Number of input channels
        num_classes: Number of output classes (2 for binary)
        hidden_dim: Hidden layer dimension
        dropout: Dropout probability
    """

    def __init__(
        self,
        in_channels: int,
        num_classes: int = 2,
        hidden_dim: int = 512,
        dropout: float = 0.5,
    ):
        super().__init__()

        self.gap = nn.AdaptiveAvgPool2d(1)

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(in_channels, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes),
        )

        # For feature extraction before final classification
        self.feature_extractor = nn.Sequential(
            nn.Flatten(),
            nn.Linear(in_channels, hidden_dim),
            nn.ReLU(inplace=True),
        )

        self.final_classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes),
        )

    def forward(
        self,
        x: torch.Tensor,
        return_features: bool = False
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Forward pass for classification.

        Args:
            x: Input features (B, C, H, W)
            return_features: Whether to return intermediate features

        Returns:
            Tuple of (logits, optional features)
        """
        pooled = self.gap(x)

        if return_features:
            features = self.feature_extractor(pooled)
            logits = self.final_classifier(features)
            return logits, features
        else:
            logits = self.classifier(pooled)
            return logits, None


class LocalizationHead(nn.Module):
    """
    Localization Head for defect region segmentation.

    Generates pixel-wise defect probability maps.
    Uses lightweight decoder for efficient inference.

    Args:
        in_channels: Number of input channels (from FPN or backbone)
        hidden_channels: Hidden layer channels
        num_classes: Number of output classes (1 for binary mask)
        upsample_factor: Final upsampling factor
    """

    def __init__(
        self,
        in_channels: int,
        hidden_channels: int = 256,
        num_classes: int = 1,
        upsample_factor: int = 4,
    ):
        super().__init__()

        self.upsample_factor = upsample_factor

        # Decoder blocks
        self.decoder = nn.Sequential(
            nn.Conv2d(in_channels, hidden_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(hidden_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_channels, hidden_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(hidden_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_channels, hidden_channels // 2, 3, padding=1, bias=False),
            nn.BatchNorm2d(hidden_channels // 2),
            nn.ReLU(inplace=True),
        )

        # Final prediction layer
        self.predictor = nn.Conv2d(hidden_channels // 2, num_classes, 1)

        # Activation
        self.sigmoid = nn.Sigmoid()

    def forward(
        self,
        x: torch.Tensor,
        target_size: Optional[Tuple[int, int]] = None
    ) -> torch.Tensor:
        """
        Generate defect localization map.

        Args:
            x: Input features (B, C, H, W)
            target_size: Optional target output size

        Returns:
            Defect probability map (B, 1, H', W')
        """
        out = self.decoder(x)
        out = self.predictor(out)

        # Upsample to target size
        if target_size is not None:
            out = F.interpolate(
                out, size=target_size, mode='bilinear', align_corners=False
            )
        elif self.upsample_factor > 1:
            out = F.interpolate(
                out, scale_factor=self.upsample_factor,
                mode='bilinear', align_corners=False
            )

        return self.sigmoid(out)


class FeaturePyramidNetwork(nn.Module):
    """
    Feature Pyramid Network for multi-scale feature fusion.

    Combines features from different backbone stages to create
    semantically rich features at multiple scales.

    Args:
        in_channels_list: List of input channels from backbone stages
        out_channels: Output channels for all pyramid levels
        extra_blocks: Whether to add extra pyramid levels
    """

    def __init__(
        self,
        in_channels_list: List[int],
        out_channels: int = 256,
        extra_blocks: bool = False,
    ):
        super().__init__()

        self.num_levels = len(in_channels_list)

        # Lateral connections (1x1 convs to reduce channels)
        self.lateral_convs = nn.ModuleList([
            nn.Conv2d(in_ch, out_channels, 1)
            for in_ch in in_channels_list
        ])

        # Output convolutions (3x3 convs to reduce aliasing)
        self.output_convs = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True),
            )
            for _ in in_channels_list
        ])

        # Extra blocks for more pyramid levels (optional)
        self.extra_blocks = extra_blocks
        if extra_blocks:
            self.extra_conv = nn.Conv2d(out_channels, out_channels, 3, stride=2, padding=1)

        self.out_channels = out_channels

    def forward(self, features: List[torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        Build feature pyramid from backbone features.

        Args:
            features: List of feature tensors from backbone (low to high level)

        Returns:
            Dictionary of pyramid features {p2, p3, p4, p5, ...}
        """
        assert len(features) == self.num_levels

        # Build laterals
        laterals = [
            lateral_conv(feat)
            for feat, lateral_conv in zip(features, self.lateral_convs)
        ]

        # Top-down pathway with lateral connections
        for i in range(self.num_levels - 2, -1, -1):
            upsampled = F.interpolate(
                laterals[i + 1],
                size=laterals[i].shape[2:],
                mode='bilinear',
                align_corners=False
            )
            laterals[i] = laterals[i] + upsampled

        # Apply output convolutions
        pyramid_features = [
            output_conv(lateral)
            for lateral, output_conv in zip(laterals, self.output_convs)
        ]

        # Create output dictionary
        outputs = {
            f'p{i+2}': feat for i, feat in enumerate(pyramid_features)
        }

        # Add extra levels if needed
        if self.extra_blocks and len(pyramid_features) > 0:
            outputs['p6'] = self.extra_conv(pyramid_features[-1])

        return outputs

    def get_fused_features(self, features: List[torch.Tensor]) -> torch.Tensor:
        """
        Get a single fused feature map from all pyramid levels.

        Args:
            features: List of feature tensors from backbone

        Returns:
            Fused feature tensor at the highest resolution
        """
        pyramid = self.forward(features)

        # Use the highest resolution feature map
        target_size = pyramid['p2'].shape[2:]

        # Resize and sum all levels
        fused = pyramid['p2']
        for key in ['p3', 'p4', 'p5']:
            if key in pyramid:
                resized = F.interpolate(
                    pyramid[key], size=target_size,
                    mode='bilinear', align_corners=False
                )
                fused = fused + resized

        return fused


class MultiTaskHead(nn.Module):
    """
    Combined Multi-Task Head for joint classification and localization.

    Provides a unified interface for both tasks while sharing features.

    Args:
        in_channels: Input feature channels
        num_classes: Number of classification classes
        fpn_channels: FPN output channels
        in_channels_list: List of input channels for FPN (if using multi-scale)
    """

    def __init__(
        self,
        in_channels: int,
        num_classes: int = 2,
        fpn_channels: int = 256,
        in_channels_list: Optional[List[int]] = None,
    ):
        super().__init__()

        # FPN for multi-scale features (optional)
        self.use_fpn = in_channels_list is not None
        if self.use_fpn:
            self.fpn = FeaturePyramidNetwork(in_channels_list, fpn_channels)
            loc_in_channels = fpn_channels
        else:
            loc_in_channels = in_channels

        # Classification head
        self.classification_head = ClassificationHead(
            in_channels=in_channels,
            num_classes=num_classes,
        )

        # Localization head
        self.localization_head = LocalizationHead(
            in_channels=loc_in_channels,
        )

    def forward(
        self,
        features: torch.Tensor,
        multi_scale_features: Optional[List[torch.Tensor]] = None,
        target_size: Optional[Tuple[int, int]] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Joint forward pass for classification and localization.

        Args:
            features: Main feature tensor for classification
            multi_scale_features: Optional multi-scale features for FPN
            target_size: Target size for localization output

        Returns:
            Dictionary with 'cls_logits' and 'loc_map'
        """
        # Classification
        cls_logits, cls_features = self.classification_head(features, return_features=True)

        # Localization
        if self.use_fpn and multi_scale_features is not None:
            fpn_features = self.fpn.get_fused_features(multi_scale_features)
            loc_map = self.localization_head(fpn_features, target_size)
        else:
            loc_map = self.localization_head(features, target_size)

        return {
            'cls_logits': cls_logits,
            'cls_features': cls_features,
            'loc_map': loc_map,
        }
