"""
EfficientNetV2 Backbone for GAIN-MTL (mmpretrain 기반)

OpenMMLab의 mmpretrain 라이브러리를 사용하여 pretrained EfficientNetV2를 로드합니다.
다양한 pretrained weights와 쉬운 설정을 지원합니다.

mmpretrain 설치:
    pip install mmpretrain

사용 가능한 EfficientNetV2 모델:
    - efficientnetv2-s: EfficientNetV2-S (Small)
    - efficientnetv2-m: EfficientNetV2-M (Medium)
    - efficientnetv2-l: EfficientNetV2-L (Large)
"""

from typing import Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn

try:
    from mmpretrain.models.backbones import EfficientNetV2 as MMEfficientNetV2
    from mmpretrain.models.utils import make_divisible
    MMPRETRAIN_AVAILABLE = True
except ImportError:
    MMPRETRAIN_AVAILABLE = False
    print("Warning: mmpretrain not installed. Install with: pip install mmpretrain")


class EfficientNetV2Backbone(nn.Module):
    """
    EfficientNetV2 backbone using mmpretrain.

    Extracts multi-scale features for FPN-style processing.

    Args:
        arch: Architecture variant ('s', 'm', 'l', 'xl')
        out_indices: Indices of stages to output features from (0-indexed)
        frozen_stages: Number of stages to freeze (-1 = none)
        pretrained: Whether to use pretrained weights or path to checkpoint
        init_cfg: Initialization config (for mmpretrain compatibility)

    Example:
        >>> backbone = EfficientNetV2Backbone(arch='s', out_indices=(1, 2, 3, 4))
        >>> x = torch.randn(1, 3, 512, 512)
        >>> features = backbone(x)
        >>> for f in features:
        ...     print(f.shape)
    """

    # EfficientNetV2 architecture specifications
    # arch: (expand_ratio, channels, num_blocks, stride, block_type)
    ARCH_SETTINGS = {
        's': {
            'stage_channels': [24, 48, 64, 128, 160, 256],  # channels after each stage
            'out_channels': [24, 48, 64, 128, 160, 256],
        },
        'm': {
            'stage_channels': [24, 48, 80, 160, 176, 304, 512],
            'out_channels': [24, 48, 80, 160, 176, 304, 512],
        },
        'l': {
            'stage_channels': [32, 64, 96, 192, 224, 384, 640],
            'out_channels': [32, 64, 96, 192, 224, 384, 640],
        },
        'xl': {
            'stage_channels': [32, 64, 96, 192, 256, 512, 640],
            'out_channels': [32, 64, 96, 192, 256, 512, 640],
        },
    }

    # Pretrained checkpoint URLs (mmpretrain에서 제공)
    PRETRAINED_URLS = {
        's': 'https://download.openmmlab.com/mmclassification/v0/efficientnetv2/efficientnetv2-s_3rdparty_in1k_20221220-f0eaff9d.pth',
        'm': 'https://download.openmmlab.com/mmclassification/v0/efficientnetv2/efficientnetv2-m_3rdparty_in1k_20221220-9dc0c729.pth',
        'l': 'https://download.openmmlab.com/mmclassification/v0/efficientnetv2/efficientnetv2-l_3rdparty_in1k_20221220-5c3bac0f.pth',
        'xl': 'https://download.openmmlab.com/mmclassification/v0/efficientnetv2/efficientnetv2-xl_3rdparty_in21k_20221220-583ac18e.pth',
    }

    def __init__(
        self,
        arch: str = 's',
        out_indices: Tuple[int, ...] = (1, 2, 3, 4),
        frozen_stages: int = -1,
        pretrained: Union[bool, str] = True,
        drop_path_rate: float = 0.0,
        init_cfg: Optional[Dict] = None,
    ):
        super().__init__()

        if not MMPRETRAIN_AVAILABLE:
            raise ImportError(
                "mmpretrain is required for this backbone. "
                "Install with: pip install mmpretrain"
            )

        self.arch = arch.lower()
        self.out_indices = out_indices
        self.frozen_stages = frozen_stages

        # Validate arch
        if self.arch not in self.ARCH_SETTINGS:
            raise ValueError(
                f"Invalid arch '{arch}'. "
                f"Available: {list(self.ARCH_SETTINGS.keys())}"
            )

        # Build init_cfg for pretrained weights
        if init_cfg is None and pretrained:
            if isinstance(pretrained, str):
                # Custom checkpoint path
                init_cfg = dict(type='Pretrained', checkpoint=pretrained)
            else:
                # Use default pretrained weights
                init_cfg = dict(
                    type='Pretrained',
                    checkpoint=self.PRETRAINED_URLS.get(self.arch),
                    prefix='backbone.',
                )

        # Create mmpretrain EfficientNetV2 backbone
        # mmpretrain의 EfficientNetV2는 frozen_stages=-1을 허용하지 않음
        # -1은 "아무것도 freeze하지 않음"을 의미하므로 0으로 변환
        mm_frozen_stages = 0 if frozen_stages < 0 else frozen_stages

        self.backbone = MMEfficientNetV2(
            arch=self.arch,
            out_indices=out_indices,
            frozen_stages=mm_frozen_stages,
            drop_path_rate=drop_path_rate,
            init_cfg=init_cfg,
        )

        # Initialize weights
        self.backbone.init_weights()

        # Get output channel dimensions
        self._feature_dims = self._get_feature_dims()

    def _get_feature_dims(self) -> List[int]:
        """Get output feature dimensions for each stage."""
        arch_settings = self.ARCH_SETTINGS[self.arch]
        out_channels = arch_settings['out_channels']

        # out_indices에 해당하는 채널 수 반환
        return [out_channels[i] for i in self.out_indices]

    def forward(self, x: torch.Tensor) -> List[torch.Tensor]:
        """
        Forward pass extracting multi-scale features.

        Args:
            x: Input tensor of shape (B, 3, H, W)

        Returns:
            List of feature tensors from specified stages
        """
        return self.backbone(x)

    def get_feature_dims(self) -> List[int]:
        """Return feature dimensions for each output stage."""
        return self._feature_dims

    @property
    def final_feature_dim(self) -> int:
        """Return the dimension of the final (deepest) feature map."""
        return self._feature_dims[-1]

    def freeze_stages(self, num_stages: int):
        """
        Freeze the first num_stages stages.

        Args:
            num_stages: Number of stages to freeze
        """
        self.backbone.frozen_stages = num_stages
        self.backbone._freeze_stages()


class EfficientNetV2BackboneFallback(nn.Module):
    """
    Fallback EfficientNetV2 backbone using timm when mmpretrain is not available.

    This provides the same interface but uses timm as the backend.
    """

    ARCH_TO_TIMM = {
        's': 'tf_efficientnetv2_s',
        'm': 'tf_efficientnetv2_m',
        'l': 'tf_efficientnetv2_l',
        'xl': 'tf_efficientnetv2_xl',
    }

    def __init__(
        self,
        arch: str = 's',
        out_indices: Tuple[int, ...] = (1, 2, 3, 4),
        frozen_stages: int = -1,
        pretrained: Union[bool, str] = True,
        **kwargs,
    ):
        super().__init__()

        try:
            import timm
        except ImportError:
            raise ImportError("Either mmpretrain or timm is required. Install with: pip install timm")

        self.arch = arch.lower()
        self.out_indices = out_indices

        # Map arch to timm model name
        timm_model_name = self.ARCH_TO_TIMM.get(self.arch)
        if timm_model_name is None:
            raise ValueError(f"Invalid arch '{arch}'")

        # Create timm model
        self.model = timm.create_model(
            timm_model_name,
            pretrained=pretrained if isinstance(pretrained, bool) else False,
            features_only=True,
            out_indices=out_indices,
        )

        # Load custom checkpoint if provided
        if isinstance(pretrained, str):
            state_dict = torch.load(pretrained, map_location='cpu')
            self.model.load_state_dict(state_dict, strict=False)

        # Get feature dimensions
        self._feature_dims = self.model.feature_info.channels()

        # Freeze stages
        if frozen_stages >= 0:
            self._freeze_stages(frozen_stages)

    def _freeze_stages(self, num_stages: int):
        """Freeze early stages."""
        if hasattr(self.model, 'conv_stem'):
            for param in self.model.conv_stem.parameters():
                param.requires_grad = False

        if hasattr(self.model, 'blocks'):
            for i, block in enumerate(self.model.blocks):
                if i < num_stages:
                    for param in block.parameters():
                        param.requires_grad = False

    def forward(self, x: torch.Tensor) -> List[torch.Tensor]:
        """Forward pass."""
        return self.model(x)

    def get_feature_dims(self) -> List[int]:
        """Return feature dimensions."""
        return self._feature_dims

    @property
    def final_feature_dim(self) -> int:
        """Return final feature dimension."""
        return self._feature_dims[-1]


def get_backbone(
    arch: str = 's',
    pretrained: Union[bool, str] = True,
    out_indices: Tuple[int, ...] = (1, 2, 3, 4),
    frozen_stages: int = -1,
    use_mmpretrain: bool = True,
    **kwargs,
) -> nn.Module:
    """
    Factory function to create EfficientNetV2 backbone.

    Args:
        arch: Architecture variant ('s', 'm', 'l', 'xl')
        pretrained: Whether to use pretrained weights or path to checkpoint
        out_indices: Indices of stages to output features from
        frozen_stages: Number of stages to freeze (-1 = none)
        use_mmpretrain: Whether to prefer mmpretrain over timm
        **kwargs: Additional arguments

    Returns:
        EfficientNetV2 backbone module
    """
    if use_mmpretrain and MMPRETRAIN_AVAILABLE:
        return EfficientNetV2Backbone(
            arch=arch,
            out_indices=out_indices,
            frozen_stages=frozen_stages,
            pretrained=pretrained,
            **kwargs,
        )
    else:
        print("Using timm fallback for EfficientNetV2 backbone")
        return EfficientNetV2BackboneFallback(
            arch=arch,
            out_indices=out_indices,
            frozen_stages=frozen_stages,
            pretrained=pretrained,
            **kwargs,
        )


# Convenience classes
class EfficientNetV2S(EfficientNetV2Backbone if MMPRETRAIN_AVAILABLE else EfficientNetV2BackboneFallback):
    """EfficientNetV2-S (Small) backbone."""
    def __init__(self, pretrained: bool = True, **kwargs):
        super().__init__(arch='s', pretrained=pretrained, **kwargs)


class EfficientNetV2M(EfficientNetV2Backbone if MMPRETRAIN_AVAILABLE else EfficientNetV2BackboneFallback):
    """EfficientNetV2-M (Medium) backbone."""
    def __init__(self, pretrained: bool = True, **kwargs):
        super().__init__(arch='m', pretrained=pretrained, **kwargs)


class EfficientNetV2L(EfficientNetV2Backbone if MMPRETRAIN_AVAILABLE else EfficientNetV2BackboneFallback):
    """EfficientNetV2-L (Large) backbone."""
    def __init__(self, pretrained: bool = True, **kwargs):
        super().__init__(arch='l', pretrained=pretrained, **kwargs)


class EfficientNetV2XL(EfficientNetV2Backbone if MMPRETRAIN_AVAILABLE else EfficientNetV2BackboneFallback):
    """EfficientNetV2-XL (Extra Large) backbone."""
    def __init__(self, pretrained: bool = True, **kwargs):
        super().__init__(arch='xl', pretrained=pretrained, **kwargs)
