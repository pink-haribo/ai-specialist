"""
GAIN-MTL: Guided Attention Inference Multi-Task Learning Model

Main model integrating:
- EfficientNetV2 backbone
- GAIN-style guided attention
- Multi-task learning (classification + localization)
- Counterfactual reasoning

This is the core model for manufacturing defect detection with
interpretable attention maps.

References:
- GAIN: "Tell Me Where to Look" (CVPR 2018)
- Multi-Task Learning for Defect Detection (PMC 2019)
- CXR-MultiTaskNet (Nature 2025)
"""

from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from .backbone import EfficientNetV2Backbone, get_backbone
from .attention import GuidedAttentionModule, AttentionMiningHead, CBAM
from .heads import ClassificationHead, LocalizationHead, FeaturePyramidNetwork
from .counterfactual import CounterfactualModule


class GAINMTLModel(nn.Module):
    """
    GAIN-MTL (Guided Attention Inference Multi-Task Learning) Model.

    A comprehensive model for manufacturing defect detection that:
    1. Uses EfficientNetV2 for robust feature extraction
    2. Employs GAIN-style attention with external supervision
    3. Performs joint classification and localization
    4. Incorporates counterfactual reasoning for robust learning

    Architecture:
    ```
    Input Image
         │
         ▼
    ┌─────────────────┐
    │ EfficientNetV2  │ ──► Multi-scale features
    │    Backbone     │
    └─────────────────┘
              │
              ▼
    ┌─────────────────┐
    │      FPN        │ ──► Unified multi-scale features
    └─────────────────┘
              │
        ┌─────┴─────┐
        ▼           ▼
    ┌───────┐  ┌─────────────┐
    │ GAIN  │  │ Localization│
    │ Attn  │  │    Head     │
    └───────┘  └─────────────┘
        │           │
        ▼           ▼
    ┌───────┐  ┌───────┐
    │ Class │  │ Defect│
    │ Head  │  │  Map  │
    └───────┘  └───────┘
    ```

    Training vs Inference:
    ```
    Training:
    ├── cls_logits          → cls_loss (baseline, for comparison)
    ├── attended_cls_logits → am_loss (main classification output)
    ├── attention_map       → guide_loss (supervised to match GT mask)
    └── localization_map    → loc_loss (auxiliary task, improves backbone)

    Inference:
    ├── attended_cls_logits → Final classification output
    └── attention_map       → CAM for interpretability
    ```

    The localization head serves as an auxiliary task during training,
    helping the backbone learn better spatial features. At inference time,
    only attended_cls_logits (classification) and attention_map (CAM) are used.

    Args:
        backbone_name: Name of EfficientNetV2 variant
        num_classes: Number of classification classes (default: 2 for binary)
        pretrained: Whether to use pretrained backbone weights
        fpn_channels: Number of channels in FPN
        attention_channels: Number of channels in attention module
        use_counterfactual: Whether to include counterfactual module
        freeze_backbone_stages: Number of backbone stages to freeze
    """

    def __init__(
        self,
        backbone_arch: str = 's',
        num_classes: int = 2,
        pretrained: bool = True,
        fpn_channels: int = 256,
        attention_channels: int = 512,
        use_counterfactual: bool = True,
        freeze_backbone_stages: int = -1,
        out_indices: Tuple[int, ...] = (1, 2, 3, 4),
    ):
        super().__init__()

        self.num_classes = num_classes
        self.use_counterfactual = use_counterfactual

        # ============ Backbone (mmpretrain EfficientNetV2) ============
        self.backbone = get_backbone(
            arch=backbone_arch,
            pretrained=pretrained,
            frozen_stages=freeze_backbone_stages,
            out_indices=out_indices,
        )
        backbone_channels = self.backbone.get_feature_dims()
        final_channels = backbone_channels[-1]

        # ============ Feature Pyramid Network ============
        self.fpn = FeaturePyramidNetwork(
            in_channels_list=backbone_channels,
            out_channels=fpn_channels,
        )

        # ============ GAIN Attention Module ============
        # This is the core GAIN component - learns to attend to defect regions
        self.attention_module = GuidedAttentionModule(
            in_channels=final_channels,
            attention_channels=attention_channels,
            use_cbam=True,
        )

        # Attention Mining Head (GAIN S_am stream)
        self.attention_mining_head = AttentionMiningHead(
            in_channels=final_channels,
            hidden_channels=attention_channels // 2,
        )

        # ============ Classification Heads ============
        # Stream 1: Classification on full features (S_cl)
        self.classification_head = ClassificationHead(
            in_channels=final_channels,
            num_classes=num_classes,
            hidden_dim=512,
            dropout=0.5,
        )

        # Stream 2: Classification on attended features (GAIN)
        self.attended_classification_head = ClassificationHead(
            in_channels=final_channels,
            num_classes=num_classes,
            hidden_dim=512,
            dropout=0.5,
        )

        # ============ Localization Head ============
        self.localization_head = LocalizationHead(
            in_channels=fpn_channels,
            hidden_channels=fpn_channels // 2,
            num_classes=1,  # Binary defect mask
            upsample_factor=4,
        )

        # ============ Counterfactual Module ============
        if use_counterfactual:
            self.counterfactual_module = CounterfactualModule(
                feature_dim=final_channels,
                hidden_dim=512,
                num_classes=num_classes,
                suppression_mode='learned',
            )
        else:
            self.counterfactual_module = None

        # ============ Auxiliary Components ============
        # Feature adapter for attended features
        self.feature_adapter = nn.Sequential(
            nn.Conv2d(final_channels, final_channels, 1),
            nn.BatchNorm2d(final_channels),
            nn.ReLU(inplace=True),
        )

        # Initialize weights
        self._init_weights()

    def _init_weights(self):
        """Initialize weights for non-pretrained layers."""
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
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(
        self,
        x: torch.Tensor,
        defect_mask: Optional[torch.Tensor] = None,
        return_attention: bool = True,
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass through the GAIN-MTL model.

        Args:
            x: Input images (B, 3, H, W)
            defect_mask: Ground truth defect masks for counterfactual learning (B, 1, H, W)
            return_attention: Whether to return attention maps

        Returns:
            Dictionary containing:
                - cls_logits: Classification logits from main stream
                - attended_cls_logits: Classification logits from attended stream
                - attention_map: GAIN attention map
                - localization_map: Defect localization map
                - features: Final backbone features
                - attended_features: Attention-weighted features
                - cf_logits: Counterfactual classification logits (if enabled)
                - cf_features: Counterfactual features (if enabled)
        """
        batch_size = x.size(0)
        input_size = x.shape[2:]  # (H, W)

        # ============ Feature Extraction ============
        # Multi-scale features from backbone
        multi_scale_features = self.backbone(x)
        final_features = multi_scale_features[-1]  # Highest level features

        # ============ FPN Processing ============
        fpn_features = self.fpn(multi_scale_features)
        fpn_fused = self.fpn.get_fused_features(multi_scale_features)

        # ============ GAIN Attention ============
        # Generate attention map through guided attention module
        attended_features, attention_map = self.attention_module(
            final_features, return_attention=True
        )

        # Also generate attention through mining head (for comparison/ensemble)
        mined_attention = self.attention_mining_head(final_features)

        # Combine attention maps (optional: use learned weights)
        combined_attention = (attention_map + mined_attention) / 2

        # ============ Classification ============
        # Stream 1: Full features classification
        cls_logits, cls_features = self.classification_head(
            final_features, return_features=True
        )

        # Stream 2: Attended features classification (GAIN core)
        attended_adapted = self.feature_adapter(attended_features)
        attended_cls_logits, _ = self.attended_classification_head(
            attended_adapted, return_features=False
        )

        # ============ Localization ============
        localization_map = self.localization_head(
            fpn_fused, target_size=input_size
        )

        # ============ Build Output Dictionary ============
        # Training outputs:
        #   - cls_logits: baseline classification (without attention)
        #   - attended_cls_logits: main classification (with attention, use for inference)
        #   - attention_map: CAM supervised by GT mask (use for inference)
        #   - localization_map: auxiliary task output (training only)
        outputs = {
            # Main outputs (used at inference)
            'attended_cls_logits': attended_cls_logits,  # Main classification output
            'attention_map': combined_attention,          # CAM for interpretability
            # Baseline outputs (for comparison/ablation)
            'cls_logits': cls_logits,                     # Baseline without attention
            # Auxiliary outputs (training only)
            'localization_map': localization_map,         # Auxiliary task
            'attention_map_main': attention_map,
            'attention_map_mined': mined_attention,
            # Internal features (for advanced use)
            'features': final_features,
            'attended_features': attended_features,
            'cls_features': cls_features,
        }

        # ============ Counterfactual (Training Only) ============
        if self.use_counterfactual and defect_mask is not None:
            cf_outputs = self.counterfactual_module(
                final_features,
                defect_mask,
                combined_attention,
            )
            outputs['cf_logits'] = cf_outputs['cf_logits']
            outputs['cf_features'] = cf_outputs['cf_features']
            outputs['suppression_mask'] = cf_outputs['suppression_mask']
        else:
            outputs['cf_logits'] = None
            outputs['cf_features'] = None

        return outputs

    def get_attention_map(
        self,
        x: torch.Tensor,
        upsample: bool = True,
    ) -> torch.Tensor:
        """
        Get attention map for visualization.

        Args:
            x: Input images (B, 3, H, W)
            upsample: Whether to upsample to input resolution

        Returns:
            Attention map tensor
        """
        with torch.no_grad():
            outputs = self.forward(x, return_attention=True)
            attention = outputs['attention_map']

            if upsample:
                attention = F.interpolate(
                    attention,
                    size=x.shape[2:],
                    mode='bilinear',
                    align_corners=False,
                )

        return attention

    def predict(
        self,
        x: torch.Tensor,
        threshold: float = 0.5,
    ) -> Dict[str, torch.Tensor]:
        """
        Make predictions with confidence scores.

        Uses attended_cls_logits as the main classification output,
        which leverages attention guided by GT mask during training.

        Args:
            x: Input images (B, 3, H, W)
            threshold: Threshold for CAM-based defect detection

        Returns:
            Dictionary with predictions and confidence
        """
        with torch.no_grad():
            outputs = self.forward(x)

            # Classification prediction (using attended logits - main output)
            probs = F.softmax(outputs['attended_cls_logits'], dim=1)
            confidence, predictions = probs.max(dim=1)

            # CAM-based defect detection
            cam = outputs['attention_map']
            defect_detected = (cam > threshold).any(dim=[1, 2, 3])

        return {
            'predictions': predictions,
            'confidence': confidence,
            'probabilities': probs,
            'cam': cam,  # Attention map as CAM
            'defect_detected': defect_detected,
        }

    def get_explanation(
        self,
        x: torch.Tensor,
    ) -> Dict[str, Any]:
        """
        Generate comprehensive explanation for prediction.

        Uses attended_cls_logits as the main prediction output.
        Returns CAM (attention_map) for interpretability.

        Args:
            x: Input image (1, 3, H, W) - single image

        Returns:
            Dictionary with all explanation components
        """
        assert x.size(0) == 1, "Explanation works with single image"

        with torch.no_grad():
            outputs = self.forward(x)

            # Upsample CAM to input resolution
            input_size = x.shape[2:]
            cam_up = F.interpolate(
                outputs['attention_map'], size=input_size,
                mode='bilinear', align_corners=False
            )

            # Main classification (attended - uses attention guided by mask)
            probs = F.softmax(outputs['attended_cls_logits'], dim=1)
            pred_class = probs.argmax(dim=1).item()
            pred_conf = probs[0, pred_class].item()

            # Baseline classification (without attention, for comparison)
            baseline_probs = F.softmax(outputs['cls_logits'], dim=1)
            baseline_pred = baseline_probs.argmax(dim=1).item()

        return {
            'prediction': pred_class,
            'confidence': pred_conf,
            'probabilities': probs.squeeze().cpu().numpy(),
            'cam': cam_up.squeeze().cpu().numpy(),
            'baseline_prediction': baseline_pred,
            'baseline_probabilities': baseline_probs.squeeze().cpu().numpy(),
            'consistency': pred_class == baseline_pred,
        }


def build_gain_mtl_model(
    config: Dict[str, Any],
) -> GAINMTLModel:
    """
    Factory function to build GAIN-MTL model from config.

    Args:
        config: Configuration dictionary

    Returns:
        Configured GAINMTLModel instance
    """
    return GAINMTLModel(
        backbone_arch=config.get('backbone_arch', 's'),
        num_classes=config.get('num_classes', 2),
        pretrained=config.get('pretrained', True),
        fpn_channels=config.get('fpn_channels', 256),
        attention_channels=config.get('attention_channels', 512),
        use_counterfactual=config.get('use_counterfactual', True),
        freeze_backbone_stages=config.get('freeze_backbone_stages', -1),
        out_indices=tuple(config.get('out_indices', [1, 2, 3, 4])),
    )


# Convenience aliases for different model sizes
class GAINMTLB0(GAINMTLModel):
    """GAIN-MTL with EfficientNetV2-B0 backbone (mmpretrain)."""
    def __init__(self, num_classes: int = 2, **kwargs):
        super().__init__(
            backbone_arch='b0',
            num_classes=num_classes,
            fpn_channels=128,
            attention_channels=192,
            **kwargs
        )


class GAINMTLB1(GAINMTLModel):
    """GAIN-MTL with EfficientNetV2-B1 backbone (mmpretrain)."""
    def __init__(self, num_classes: int = 2, **kwargs):
        super().__init__(
            backbone_arch='b1',
            num_classes=num_classes,
            fpn_channels=128,
            attention_channels=208,
            **kwargs
        )


class GAINMTLSmall(GAINMTLModel):
    """GAIN-MTL with EfficientNetV2-S backbone (mmpretrain)."""
    def __init__(self, num_classes: int = 2, **kwargs):
        super().__init__(
            backbone_arch='s',
            num_classes=num_classes,
            fpn_channels=256,
            attention_channels=256,
            **kwargs
        )


class GAINMTLMedium(GAINMTLModel):
    """GAIN-MTL with EfficientNetV2-M backbone (mmpretrain)."""
    def __init__(self, num_classes: int = 2, **kwargs):
        super().__init__(
            backbone_arch='m',
            num_classes=num_classes,
            fpn_channels=256,
            attention_channels=384,
            **kwargs
        )


class GAINMTLLarge(GAINMTLModel):
    """GAIN-MTL with EfficientNetV2-L backbone (mmpretrain)."""
    def __init__(self, num_classes: int = 2, **kwargs):
        super().__init__(
            backbone_arch='l',
            num_classes=num_classes,
            fpn_channels=384,
            attention_channels=512,
            **kwargs
        )


class GAINMTLExtraLarge(GAINMTLModel):
    """GAIN-MTL with EfficientNetV2-XL backbone (mmpretrain)."""
    def __init__(self, num_classes: int = 2, **kwargs):
        super().__init__(
            backbone_arch='xl',
            num_classes=num_classes,
            fpn_channels=512,
            attention_channels=640,
            **kwargs
        )
