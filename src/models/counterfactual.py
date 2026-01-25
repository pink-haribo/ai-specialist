"""
Counterfactual Learning Module for GAIN-MTL

Implements counterfactual reasoning:
"If the defect region was removed, the sample should be classified as non-defective."

This forces the model to rely on the actual defect regions for classification,
preventing spurious correlations.

References:
- "Looking in the Mirror: A Faithful Counterfactual Explanation Method" (2025)
- "TraCE: Training calibration-based counterfactual explainers" (2022)
"""

from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class CounterfactualModule(nn.Module):
    """
    Counterfactual Learning Module.

    Simulates "what if the defect was not there?" by:
    1. Using the defect mask to identify defect features
    2. Suppressing or replacing those features
    3. Classifying the modified features (should predict non-defective)

    Args:
        feature_dim: Dimension of input features
        hidden_dim: Hidden layer dimension
        num_classes: Number of output classes
        suppression_mode: How to suppress defect features
            - 'zero': Set defect features to zero
            - 'mean': Replace with feature mean
            - 'learned': Learn a replacement pattern
    """

    def __init__(
        self,
        feature_dim: int,
        hidden_dim: int = 512,
        num_classes: int = 2,
        suppression_mode: str = 'learned',
    ):
        super().__init__()

        self.feature_dim = feature_dim
        self.suppression_mode = suppression_mode

        # Mask processor: converts defect mask to feature space
        self.mask_processor = nn.Sequential(
            nn.Conv2d(1, 64, 3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 128, 3, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, feature_dim, 1),
            nn.Sigmoid(),
        )

        # Learned feature replacement (for 'learned' mode)
        if suppression_mode == 'learned':
            self.feature_replacer = nn.Sequential(
                nn.Conv2d(feature_dim, feature_dim, 3, padding=1, bias=False),
                nn.BatchNorm2d(feature_dim),
                nn.ReLU(inplace=True),
                nn.Conv2d(feature_dim, feature_dim, 3, padding=1, bias=False),
                nn.BatchNorm2d(feature_dim),
            )

        # Feature refinement after modification
        self.feature_refiner = nn.Sequential(
            nn.Conv2d(feature_dim, feature_dim, 3, padding=1, bias=False),
            nn.BatchNorm2d(feature_dim),
            nn.ReLU(inplace=True),
            nn.Conv2d(feature_dim, feature_dim, 3, padding=1, bias=False),
            nn.BatchNorm2d(feature_dim),
        )

        # Classifier for counterfactual features
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(feature_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(hidden_dim, num_classes),
        )

    def forward(
        self,
        features: torch.Tensor,
        defect_mask: torch.Tensor,
        attention_map: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Generate counterfactual features and predictions.

        Args:
            features: Original features (B, C, H, W)
            defect_mask: Ground truth defect mask (B, 1, H', W')
            attention_map: Model's attention map (optional, for comparison)

        Returns:
            Dictionary containing:
                - cf_features: Counterfactual features
                - cf_logits: Classification logits for counterfactual
                - suppression_mask: The mask used for suppression
        """
        batch_size = features.size(0)

        # Resize defect mask to feature spatial size
        mask_resized = F.interpolate(
            defect_mask,
            size=features.shape[2:],
            mode='bilinear',
            align_corners=False
        )

        # Process mask to feature space (learnable weighting)
        suppression_mask = self.mask_processor(mask_resized)

        # Suppress defect features based on mode
        if self.suppression_mode == 'zero':
            # Simply zero out defect regions
            cf_features = features * (1 - suppression_mask)

        elif self.suppression_mode == 'mean':
            # Replace with feature mean
            feature_mean = features.mean(dim=[2, 3], keepdim=True)
            cf_features = features * (1 - suppression_mask) + feature_mean * suppression_mask

        elif self.suppression_mode == 'learned':
            # Learn replacement features
            replacement = self.feature_replacer(features)
            cf_features = features * (1 - suppression_mask) + replacement * suppression_mask

        else:
            raise ValueError(f"Unknown suppression mode: {self.suppression_mode}")

        # Refine counterfactual features
        cf_features = self.feature_refiner(cf_features) + cf_features  # Residual connection

        # Classify counterfactual
        cf_logits = self.classifier(cf_features)

        return {
            'cf_features': cf_features,
            'cf_logits': cf_logits,
            'suppression_mask': suppression_mask,
        }


class AttentionBasedCounterfactual(nn.Module):
    """
    Counterfactual module that uses model's attention instead of GT mask.

    Useful for:
    - Inference time explanations (no GT mask available)
    - Self-supervised counterfactual learning
    - Attention quality assessment

    Args:
        feature_dim: Dimension of input features
        hidden_dim: Hidden layer dimension
        num_classes: Number of output classes
        threshold: Attention threshold for defining "defect region"
    """

    def __init__(
        self,
        feature_dim: int,
        hidden_dim: int = 512,
        num_classes: int = 2,
        threshold: float = 0.5,
    ):
        super().__init__()

        self.threshold = threshold

        # Feature modification network
        self.modifier = nn.Sequential(
            nn.Conv2d(feature_dim, feature_dim, 3, padding=1, bias=False),
            nn.BatchNorm2d(feature_dim),
            nn.ReLU(inplace=True),
            nn.Conv2d(feature_dim, feature_dim, 3, padding=1, bias=False),
            nn.BatchNorm2d(feature_dim),
        )

        # Classifier
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(feature_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(hidden_dim, num_classes),
        )

    def forward(
        self,
        features: torch.Tensor,
        attention_map: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """
        Generate counterfactual using attention map.

        Args:
            features: Original features (B, C, H, W)
            attention_map: Model's attention map (B, 1, H, W)

        Returns:
            Dictionary with cf_features and cf_logits
        """
        # Threshold attention to get binary mask
        attention_binary = (attention_map > self.threshold).float()

        # Suppress attended regions
        suppressed = features * (1 - attention_binary)

        # Generate replacement features
        replacement = self.modifier(features)

        # Combine: keep non-attended regions, replace attended regions
        cf_features = suppressed + replacement * attention_binary

        # Classify
        cf_logits = self.classifier(cf_features)

        return {
            'cf_features': cf_features,
            'cf_logits': cf_logits,
            'attention_binary': attention_binary,
        }


class CounterfactualConsistencyLoss(nn.Module):
    """
    Loss function for counterfactual consistency.

    Ensures that:
    1. Defective samples with defect removed → Non-defective
    2. Non-defective samples stay non-defective when "removing" nothing
    3. Attention-based and GT-based counterfactuals are consistent

    Args:
        margin: Margin for contrastive loss
        consistency_weight: Weight for attention-GT consistency
    """

    def __init__(
        self,
        margin: float = 1.0,
        consistency_weight: float = 0.5,
    ):
        super().__init__()

        self.margin = margin
        self.consistency_weight = consistency_weight
        self.ce_loss = nn.CrossEntropyLoss()

    def forward(
        self,
        cf_logits: torch.Tensor,
        original_labels: torch.Tensor,
        has_defect: torch.Tensor,
        attention_cf_logits: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Compute counterfactual consistency loss.

        Args:
            cf_logits: Logits from counterfactual (defect removed)
            original_labels: Original labels (0=normal, 1=defective)
            has_defect: Boolean mask for samples with defects
            attention_cf_logits: Optional logits from attention-based CF

        Returns:
            Dictionary of loss components
        """
        losses = {}

        # For defective samples: CF should predict normal (class 0)
        if has_defect.sum() > 0:
            defect_cf_logits = cf_logits[has_defect]
            target_normal = torch.zeros(
                defect_cf_logits.size(0),
                dtype=torch.long,
                device=cf_logits.device
            )
            losses['cf_to_normal'] = self.ce_loss(defect_cf_logits, target_normal)
        else:
            losses['cf_to_normal'] = torch.tensor(0.0, device=cf_logits.device)

        # For normal samples: CF should still predict normal
        normal_mask = ~has_defect
        if normal_mask.sum() > 0:
            normal_cf_logits = cf_logits[normal_mask]
            target_normal = torch.zeros(
                normal_cf_logits.size(0),
                dtype=torch.long,
                device=cf_logits.device
            )
            losses['cf_stay_normal'] = self.ce_loss(normal_cf_logits, target_normal)
        else:
            losses['cf_stay_normal'] = torch.tensor(0.0, device=cf_logits.device)

        # Consistency between attention-based and GT-based counterfactuals
        if attention_cf_logits is not None and has_defect.sum() > 0:
            gt_cf_probs = F.softmax(cf_logits[has_defect], dim=1)
            attn_cf_probs = F.softmax(attention_cf_logits[has_defect], dim=1)
            losses['cf_consistency'] = F.kl_div(
                gt_cf_probs.log(),
                attn_cf_probs,
                reduction='batchmean'
            ) * self.consistency_weight
        else:
            losses['cf_consistency'] = torch.tensor(0.0, device=cf_logits.device)

        # Total loss
        losses['total'] = sum(losses.values())

        return losses
