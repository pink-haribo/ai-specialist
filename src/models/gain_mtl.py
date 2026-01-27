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
    Strategy Guide:
    - Strategy 1: cls only
    - Strategy 2: cls + cam_guide (weight-based CAM supervision)
    - Strategy 3: cls + am + guide (attention mining)
    - Strategy 4: cls + am + loc + guide (attention + localization)
    - Strategy 5: Full (all losses including counterfactual)

    Training:
    ├── cls_logits          → cls_loss (baseline, all strategies)
    ├── attended_cls_logits → am_loss (Strategy 3+)
    ├── cam                 → cam_guide_loss (Strategy 2 only)
    ├── attention_map       → guide_loss (Strategy 3+)
    └── localization_map    → loc_loss (Strategy 4+)

    Inference:
    ├── attended_cls_logits → Final classification output
    ├── cam                 → Weight-based CAM (directly from classifier)
    └── attention_map       → Attention module output
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
        # Use the last 4 stages of the backbone for both FPN and classification.
        # This ensures FPN gets deep multi-scale features and classification
        # uses the deepest stage, matching standard detection/segmentation practice.
        #
        # EfficientNetV2 stage counts (stem + block stages):
        #   B0/B1/S: 7 (indices 0-6) → last 4 = (3,4,5,6)
        #   M/L/XL:  8 (indices 0-7) → last 4 = (4,5,6,7)
        arch_settings = EfficientNetV2Backbone.ARCH_SETTINGS.get(backbone_arch.lower())
        if arch_settings is not None:
            num_stages = len(arch_settings['out_channels'])
            last_stage_idx = num_stages - 1
            num_fpn_levels = min(4, num_stages - 1)  # Exclude stem (index 0)
            out_indices = tuple(range(last_stage_idx - num_fpn_levels + 1, last_stage_idx + 1))

        self.backbone = get_backbone(
            arch=backbone_arch,
            pretrained=pretrained,
            frozen_stages=freeze_backbone_stages,
            out_indices=out_indices,
        )
        backbone_channels = self.backbone.get_feature_dims()
        final_channels = backbone_channels[-1]

        # ============ Feature Pyramid Network ============
        # FPN uses all returned features (multi-scale);
        # classification uses final_features (the deepest stage, also in FPN).
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
        """Initialize weights for non-pretrained layers.

        Only initializes newly added heads/modules (FPN, attention, classification,
        localization, counterfactual, feature_adapter). The pretrained backbone
        weights are preserved.
        """
        modules_to_init = [
            self.fpn,
            self.attention_module,
            self.attention_mining_head,
            self.classification_head,
            self.attended_classification_head,
            self.localization_head,
            self.feature_adapter,
        ]
        if self.counterfactual_module is not None:
            modules_to_init.append(self.counterfactual_module)

        for module in modules_to_init:
            for m in module.modules():
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
                - cam: Weight-based CAM from classification head (no extra module)
                - attention_map: GAIN attention map from attention module
                - localization_map: Defect localization map
                - features: Final backbone features
                - attended_features: Attention-weighted features
                - cf_logits: Counterfactual classification logits (if enabled)
                - cf_features: Counterfactual features (if enabled)
        """
        batch_size = x.size(0)
        input_size = x.shape[2:]  # (H, W)

        # ============ Feature Extraction ============
        # Multi-scale features from backbone (last 4 stages)
        multi_scale_features = self.backbone(x)
        final_features = multi_scale_features[-1]  # Deepest features for classification

        # ============ FPN Processing ============
        fpn_features = self.fpn(multi_scale_features)
        fpn_fused = self.fpn.get_fused_features(multi_scale_features)

        # ============ GAIN Attention ============
        # Generate attention logits through guided attention module
        # Note: Returns logits (pre-sigmoid) for numerical stability with BCE loss
        attended_features, attention_logits = self.attention_module(
            final_features, return_attention=True
        )

        # Also generate attention logits through mining head (for comparison/ensemble)
        mined_attention_logits = self.attention_mining_head(final_features)

        # Combine attention logits (optional: use learned weights)
        combined_attention_logits = (attention_logits + mined_attention_logits) / 2

        # ============ Classification ============
        # Stream 1: Full features classification
        cls_logits, cls_features = self.classification_head(
            final_features, return_features=True
        )

        # Generate Weight-based CAM from classification head
        # This CAM is directly derived from the classifier weights (no extra module)
        cam = self.classification_head.get_cam(final_features, class_idx=1, normalize=True)

        # Stream 2: Attended features classification (GAIN core)
        attended_adapted = self.feature_adapter(attended_features)
        attended_cls_logits, _ = self.attended_classification_head(
            attended_adapted, return_features=False
        )

        # ============ Localization ============
        # Note: Returns logits (pre-sigmoid) for numerical stability with BCE loss
        localization_logits = self.localization_head(
            fpn_fused, target_size=input_size
        )

        # ============ Build Output Dictionary ============
        # Training outputs:
        #   - cls_logits: baseline classification (without attention)
        #   - attended_cls_logits: main classification (with attention, use for inference)
        #   - attention_map: attention logits (pre-sigmoid) for BCE with logits loss
        #   - cam: Weight-based CAM from classifier (already normalized [0,1])
        #   - localization_map: localization logits (pre-sigmoid) for BCE with logits loss
        outputs = {
            # Main outputs (used at inference)
            'attended_cls_logits': attended_cls_logits,       # Main classification output
            'attention_map': combined_attention_logits,       # Attention logits (apply sigmoid for visualization)
            # Weight-based CAM (directly from classifier, for Strategy 2)
            'cam': cam,                                       # CAM from classification head weights (already [0,1])
            # Baseline outputs (for comparison/ablation)
            'cls_logits': cls_logits,                         # Baseline without attention
            # Auxiliary outputs (training only - logits for BCE with logits)
            'localization_map': localization_logits,          # Localization logits (pre-sigmoid)
            'attention_map_main': attention_logits,           # Main attention logits
            'attention_map_mined': mined_attention_logits,    # Mined attention logits
            # Internal features (for advanced use)
            'features': final_features,
            'attended_features': attended_features,
            'cls_features': cls_features,
        }

        # ============ Counterfactual (Training Only) ============
        if self.use_counterfactual and defect_mask is not None:
            # Apply sigmoid to attention logits for counterfactual module
            combined_attention_probs = torch.sigmoid(combined_attention_logits)
            cf_outputs = self.counterfactual_module(
                final_features,
                defect_mask,
                combined_attention_probs,
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
            Attention map tensor (with sigmoid applied)
        """
        with torch.no_grad():
            outputs = self.forward(x, return_attention=True)
            # Apply sigmoid to convert logits to probabilities
            attention = torch.sigmoid(outputs['attention_map'])

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

            # CAM-based defect detection (apply sigmoid to attention logits)
            cam = torch.sigmoid(outputs['attention_map'])
            defect_detected = (cam > threshold).any(dim=[1, 2, 3])

        return {
            'predictions': predictions,
            'confidence': confidence,
            'probabilities': probs,
            'cam': cam,  # Attention map as CAM (sigmoid applied)
            'defect_detected': defect_detected,
        }

    def get_explanation(
        self,
        x: torch.Tensor,
    ) -> Dict[str, Any]:
        """
        Generate comprehensive explanation for prediction.

        Uses attended_cls_logits as the main prediction output.
        Returns CAM (attention_map with sigmoid) for interpretability.

        Args:
            x: Input image (1, 3, H, W) - single image

        Returns:
            Dictionary with all explanation components
        """
        assert x.size(0) == 1, "Explanation works with single image"

        with torch.no_grad():
            outputs = self.forward(x)

            # Apply sigmoid to attention logits and upsample CAM to input resolution
            input_size = x.shape[2:]
            cam = torch.sigmoid(outputs['attention_map'])
            cam_up = F.interpolate(
                cam, size=input_size,
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
