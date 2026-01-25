"""
EfficientNetV2 Backbone for GAIN-MTL

Provides multi-scale feature extraction using EfficientNetV2 variants.
Supports pretrained weights from timm library.
"""

from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import timm


class EfficientNetV2Backbone(nn.Module):
    """
    EfficientNetV2 backbone with multi-scale feature extraction.

    Extracts features from multiple stages for FPN-style multi-scale processing.

    Args:
        model_name: EfficientNetV2 variant ('efficientnetv2_s', 'efficientnetv2_m', 'efficientnetv2_l')
        pretrained: Whether to use ImageNet pretrained weights
        out_indices: Indices of stages to extract features from
        freeze_stages: Number of stages to freeze (0 = no freezing)
    """

    # Feature dimensions for each EfficientNetV2 variant at different stages
    FEATURE_DIMS = {
        'efficientnetv2_s': [48, 64, 160, 256],      # stages 1-4
        'efficientnetv2_m': [48, 80, 176, 304],
        'efficientnetv2_l': [64, 96, 224, 384],
        'efficientnetv2_rw_s': [48, 64, 160, 272],
        'tf_efficientnetv2_s': [48, 64, 160, 256],
        'tf_efficientnetv2_m': [48, 80, 176, 304],
        'tf_efficientnetv2_l': [64, 96, 224, 384],
    }

    def __init__(
        self,
        model_name: str = 'efficientnetv2_s',
        pretrained: bool = True,
        out_indices: Tuple[int, ...] = (1, 2, 3, 4),
        freeze_stages: int = 0,
    ):
        super().__init__()

        self.model_name = model_name
        self.out_indices = out_indices
        self.freeze_stages = freeze_stages

        # Create EfficientNetV2 model with feature extraction
        self.model = timm.create_model(
            model_name,
            pretrained=pretrained,
            features_only=True,
            out_indices=out_indices,
        )

        # Get feature dimensions
        if model_name in self.FEATURE_DIMS:
            self.feature_dims = [self.FEATURE_DIMS[model_name][i-1] for i in out_indices]
        else:
            # Get from model info for unknown variants
            self.feature_dims = self.model.feature_info.channels()

        # Freeze early stages if specified
        if freeze_stages > 0:
            self._freeze_stages(freeze_stages)

    def _freeze_stages(self, num_stages: int):
        """Freeze the first num_stages stages of the backbone."""
        # Freeze stem
        if hasattr(self.model, 'conv_stem'):
            for param in self.model.conv_stem.parameters():
                param.requires_grad = False
        if hasattr(self.model, 'bn1'):
            for param in self.model.bn1.parameters():
                param.requires_grad = False

        # Freeze blocks
        if hasattr(self.model, 'blocks'):
            for i, block in enumerate(self.model.blocks):
                if i < num_stages:
                    for param in block.parameters():
                        param.requires_grad = False

    def forward(self, x: torch.Tensor) -> List[torch.Tensor]:
        """
        Forward pass extracting multi-scale features.

        Args:
            x: Input tensor of shape (B, 3, H, W)

        Returns:
            List of feature tensors from each stage
        """
        features = self.model(x)
        return features

    def get_feature_dims(self) -> List[int]:
        """Return feature dimensions for each output stage."""
        return self.feature_dims

    @property
    def final_feature_dim(self) -> int:
        """Return the dimension of the final (deepest) feature map."""
        return self.feature_dims[-1]


def get_backbone(
    name: str = 'efficientnetv2_s',
    pretrained: bool = True,
    **kwargs
) -> EfficientNetV2Backbone:
    """
    Factory function to create backbone.

    Args:
        name: Backbone name
        pretrained: Whether to use pretrained weights
        **kwargs: Additional arguments for backbone

    Returns:
        Backbone module
    """
    return EfficientNetV2Backbone(
        model_name=name,
        pretrained=pretrained,
        **kwargs
    )


# Convenience classes for specific variants
class EfficientNetV2S(EfficientNetV2Backbone):
    """EfficientNetV2-S (Small) backbone."""
    def __init__(self, pretrained: bool = True, **kwargs):
        super().__init__(model_name='efficientnetv2_s', pretrained=pretrained, **kwargs)


class EfficientNetV2M(EfficientNetV2Backbone):
    """EfficientNetV2-M (Medium) backbone."""
    def __init__(self, pretrained: bool = True, **kwargs):
        super().__init__(model_name='efficientnetv2_m', pretrained=pretrained, **kwargs)


class EfficientNetV2L(EfficientNetV2Backbone):
    """EfficientNetV2-L (Large) backbone."""
    def __init__(self, pretrained: bool = True, **kwargs):
        super().__init__(model_name='efficientnetv2_l', pretrained=pretrained, **kwargs)
