"""
Comprehensive Loss Functions for GAIN-MTL

Implements all loss components:
1. Classification Loss (Focal Loss for class imbalance)
2. Attention Mining Loss (GAIN S_am)
3. Guided Attention Loss (attention supervision with GT mask)
4. Localization Loss (Dice + BCE for segmentation)
5. Counterfactual Loss (forcing correct reasoning)
6. Consistency Loss (attention-localization alignment)

Reference:
- GAIN: "Tell Me Where to Look" (CVPR 2018)
- Focal Loss: "Focal Loss for Dense Object Detection" (ICCV 2017)
"""

from typing import Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class FocalLoss(nn.Module):
    """
    Focal Loss for handling class imbalance.

    FL(p_t) = -alpha_t * (1 - p_t)^gamma * log(p_t)

    Args:
        alpha: Weighting factor for positive class
        gamma: Focusing parameter (higher = more focus on hard examples)
        reduction: Reduction method ('mean', 'sum', 'none')
    """

    def __init__(
        self,
        alpha: float = 0.25,
        gamma: float = 2.0,
        reduction: str = 'mean',
    ):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(
        self,
        inputs: torch.Tensor,
        targets: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute focal loss.

        Args:
            inputs: Predicted logits (B, C)
            targets: Ground truth labels (B,)

        Returns:
            Focal loss value
        """
        ce_loss = F.cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-ce_loss)

        # Apply focal term
        focal_term = (1 - pt) ** self.gamma

        # Apply alpha weighting
        if self.alpha is not None:
            alpha_t = self.alpha * targets + (1 - self.alpha) * (1 - targets)
            focal_loss = alpha_t * focal_term * ce_loss
        else:
            focal_loss = focal_term * ce_loss

        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        return focal_loss


class DiceLoss(nn.Module):
    """
    Dice Loss for segmentation.

    Dice = 2 * |A ∩ B| / (|A| + |B|)
    Loss = 1 - Dice

    Args:
        smooth: Smoothing factor to avoid division by zero
        reduction: Reduction method
    """

    def __init__(
        self,
        smooth: float = 1e-6,
        reduction: str = 'mean',
    ):
        super().__init__()
        self.smooth = smooth
        self.reduction = reduction

    def forward(
        self,
        inputs: torch.Tensor,
        targets: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute Dice loss.

        Args:
            inputs: Predicted masks (B, 1, H, W)
            targets: Ground truth masks (B, 1, H, W)

        Returns:
            Dice loss value
        """
        # Flatten
        inputs_flat = inputs.flatten(1)
        targets_flat = targets.flatten(1)

        # Compute intersection and union
        intersection = (inputs_flat * targets_flat).sum(dim=1)
        union = inputs_flat.sum(dim=1) + targets_flat.sum(dim=1)

        # Dice coefficient
        dice = (2.0 * intersection + self.smooth) / (union + self.smooth)

        # Dice loss
        loss = 1.0 - dice

        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        return loss


class IoULoss(nn.Module):
    """
    IoU (Intersection over Union) Loss.

    IoU = |A ∩ B| / |A ∪ B|
    Loss = 1 - IoU

    Args:
        smooth: Smoothing factor
        threshold: Threshold for binarization (optional)
    """

    def __init__(
        self,
        smooth: float = 1e-6,
        threshold: Optional[float] = None,
    ):
        super().__init__()
        self.smooth = smooth
        self.threshold = threshold

    def forward(
        self,
        inputs: torch.Tensor,
        targets: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute IoU loss.

        Args:
            inputs: Predicted masks (B, 1, H, W)
            targets: Ground truth masks (B, 1, H, W)

        Returns:
            IoU loss value
        """
        if self.threshold is not None:
            inputs = (inputs > self.threshold).float()

        # Flatten
        inputs_flat = inputs.flatten(1)
        targets_flat = targets.flatten(1)

        # Compute IoU
        intersection = (inputs_flat * targets_flat).sum(dim=1)
        union = inputs_flat.sum(dim=1) + targets_flat.sum(dim=1) - intersection

        iou = (intersection + self.smooth) / (union + self.smooth)

        return (1 - iou).mean()


class CAMGuidanceLoss(nn.Module):
    """
    Loss for guiding Class Activation Map (CAM) to align with GT mask.

    This is a simpler alternative to AttentionGuidanceLoss that works
    directly with weight-based CAM from the classification head,
    without requiring additional attention modules.

    Key differences from AttentionGuidanceLoss:
    - Works with CAM generated from classifier weights (no extra module)
    - Only supervises defective samples (normal samples don't need CAM guidance)
    - Simpler loss formulation: BCE + Dice

    For defective samples: CAM should highlight defect regions (match GT mask)
    For normal samples: No supervision (CAM naturally shows less activation)

    Args:
        alpha: Weight for BCE loss
        use_dice: Whether to include Dice loss
        use_iou: Whether to include IoU loss
    """

    def __init__(
        self,
        alpha: float = 1.0,
        use_dice: bool = True,
        use_iou: bool = False,
    ):
        super().__init__()
        self.alpha = alpha
        self.use_dice = use_dice
        self.use_iou = use_iou
        self.dice_loss = DiceLoss()
        self.iou_loss = IoULoss()

    def forward(
        self,
        cam: torch.Tensor,
        defect_mask: torch.Tensor,
        has_defect: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """
        Compute CAM guidance loss.

        Args:
            cam: Model's CAM (B, 1, H, W), values in [0, 1]
            defect_mask: Ground truth defect mask (B, 1, H, W)
            has_defect: Boolean mask indicating defective samples (B,)

        Returns:
            Dictionary of loss components
        """
        losses = {}
        device = cam.device

        # Resize CAM to match mask size if needed
        if cam.shape[2:] != defect_mask.shape[2:]:
            cam_resized = F.interpolate(
                cam,
                size=defect_mask.shape[2:],
                mode='bilinear',
                align_corners=False
            )
        else:
            cam_resized = cam

        # === Loss for defective samples only ===
        if has_defect.sum() > 0:
            defect_cam = cam_resized[has_defect]
            defect_gt = defect_mask[has_defect]

            # BCE loss for pixel-wise alignment
            bce_loss = F.binary_cross_entropy(
                defect_cam,
                defect_gt,
                reduction='mean'
            )
            losses['cam_bce'] = self.alpha * bce_loss

            # Dice loss for region overlap
            if self.use_dice:
                dice_loss = self.dice_loss(defect_cam, defect_gt)
                losses['cam_dice'] = self.alpha * dice_loss

            # IoU loss (optional)
            if self.use_iou:
                iou_loss = self.iou_loss(defect_cam, defect_gt)
                losses['cam_iou'] = self.alpha * iou_loss
        else:
            losses['cam_bce'] = torch.tensor(0.0, device=device)
            if self.use_dice:
                losses['cam_dice'] = torch.tensor(0.0, device=device)
            if self.use_iou:
                losses['cam_iou'] = torch.tensor(0.0, device=device)

        return losses


class AttentionGuidanceLoss(nn.Module):
    """
    Loss for guiding attention to correct regions.

    For defective samples: attention should align with defect mask
    For normal samples: attention should be distributed (high entropy)

    Args:
        alpha: Weight for alignment loss (defective)
        beta: Weight for entropy loss (normal)
        use_iou: Whether to include IoU loss
    """

    def __init__(
        self,
        alpha: float = 1.0,
        beta: float = 0.1,
        use_iou: bool = True,
    ):
        super().__init__()
        self.alpha = alpha
        self.beta = beta
        self.use_iou = use_iou
        self.dice_loss = DiceLoss()
        self.iou_loss = IoULoss()

    def forward(
        self,
        attention_map: torch.Tensor,
        defect_mask: torch.Tensor,
        has_defect: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """
        Compute attention guidance loss.

        Args:
            attention_map: Model's attention map (B, 1, H, W)
            defect_mask: Ground truth defect mask (B, 1, H, W)
            has_defect: Boolean mask indicating defective samples (B,)

        Returns:
            Dictionary of loss components
        """
        losses = {}
        device = attention_map.device

        # Resize attention to match mask size if needed
        if attention_map.shape[2:] != defect_mask.shape[2:]:
            attention_resized = F.interpolate(
                attention_map,
                size=defect_mask.shape[2:],
                mode='bilinear',
                align_corners=False
            )
        else:
            attention_resized = attention_map

        # === Loss for defective samples ===
        if has_defect.sum() > 0:
            defect_attention = attention_resized[has_defect]
            defect_gt = defect_mask[has_defect]

            # BCE loss for alignment
            bce_loss = F.binary_cross_entropy(
                defect_attention,
                defect_gt,
                reduction='mean'
            )
            losses['attention_bce'] = self.alpha * bce_loss

            # Dice loss for better overlap
            dice_loss = self.dice_loss(defect_attention, defect_gt)
            losses['attention_dice'] = self.alpha * dice_loss

            # IoU loss (optional)
            if self.use_iou:
                iou_loss = self.iou_loss(defect_attention, defect_gt)
                losses['attention_iou'] = self.alpha * iou_loss
        else:
            losses['attention_bce'] = torch.tensor(0.0, device=device)
            losses['attention_dice'] = torch.tensor(0.0, device=device)
            if self.use_iou:
                losses['attention_iou'] = torch.tensor(0.0, device=device)

        # === Loss for normal samples (entropy maximization) ===
        normal_mask = ~has_defect
        if normal_mask.sum() > 0:
            normal_attention = attention_resized[normal_mask]

            # Flatten spatial dimensions
            attn_flat = normal_attention.flatten(start_dim=2)  # (B, 1, H*W)

            # Normalize to probability distribution
            attn_prob = attn_flat / (attn_flat.sum(dim=2, keepdim=True) + 1e-6)

            # Compute negative entropy (we want to maximize entropy = minimize negative entropy)
            entropy = -(attn_prob * torch.log(attn_prob + 1e-6)).sum(dim=2)
            max_entropy = torch.log(torch.tensor(attn_flat.size(2), dtype=torch.float, device=device))

            # Normalized negative entropy loss
            entropy_loss = 1 - (entropy / max_entropy).mean()
            losses['attention_entropy'] = self.beta * entropy_loss
        else:
            losses['attention_entropy'] = torch.tensor(0.0, device=device)

        return losses


class GAINMTLLoss(nn.Module):
    """
    Comprehensive Loss Function for GAIN-MTL.

    Combines all loss components with configurable weights:
    - λ_cls: Classification loss weight
    - λ_am: Attention mining loss weight
    - λ_cam_guide: CAM guidance loss weight (Strategy 2: weight-based CAM)
    - λ_loc: Localization loss weight
    - λ_guide: Guided attention loss weight (attention module)
    - λ_cf: Counterfactual loss weight
    - λ_consist: Consistency loss weight

    Strategy Guide:
    - Strategy 1: cls only
    - Strategy 2: cls + cam_guide (weight-based CAM supervision)
    - Strategy 3: cls + am + guide (attention mining)
    - Strategy 4: cls + am + loc + guide (attention + localization)
    - Strategy 5: Full (all losses including counterfactual)

    Args:
        lambda_cls: Weight for classification loss
        lambda_am: Weight for attention mining loss
        lambda_cam_guide: Weight for CAM guidance loss (weight-based CAM)
        lambda_loc: Weight for localization loss
        lambda_guide: Weight for guided attention loss (attention module)
        lambda_cf: Weight for counterfactual loss
        lambda_consist: Weight for consistency loss
        focal_gamma: Gamma parameter for focal loss
        focal_alpha: Alpha parameter for focal loss
    """

    def __init__(
        self,
        lambda_cls: float = 1.0,
        lambda_am: float = 0.5,
        lambda_cam_guide: float = 0.0,
        lambda_loc: float = 0.3,
        lambda_guide: float = 0.5,
        lambda_cf: float = 0.3,
        lambda_consist: float = 0.2,
        focal_gamma: float = 2.0,
        focal_alpha: float = 0.25,
    ):
        super().__init__()

        # Loss weights
        self.lambda_cls = lambda_cls
        self.lambda_am = lambda_am
        self.lambda_cam_guide = lambda_cam_guide
        self.lambda_loc = lambda_loc
        self.lambda_guide = lambda_guide
        self.lambda_cf = lambda_cf
        self.lambda_consist = lambda_consist

        # Component losses
        self.focal_loss = FocalLoss(alpha=focal_alpha, gamma=focal_gamma)
        self.dice_loss = DiceLoss()
        self.cam_guidance_loss = CAMGuidanceLoss()
        self.attention_guidance_loss = AttentionGuidanceLoss()

    def forward(
        self,
        outputs: Dict[str, torch.Tensor],
        targets: Dict[str, torch.Tensor],
    ) -> Dict[str, torch.Tensor]:
        """
        Compute total loss and all components.

        Args:
            outputs: Model outputs dictionary containing:
                - cls_logits: Classification logits
                - attended_cls_logits: Attended classification logits
                - cam: Weight-based CAM from classifier (for Strategy 2)
                - attention_map: Attention map from attention module
                - localization_map: Localization map
                - cf_logits: Counterfactual logits (optional)

            targets: Target dictionary containing:
                - label: Classification labels (B,)
                - defect_mask: Defect masks (B, 1, H, W)
                - has_defect: Boolean mask for defective samples (B,)

        Returns:
            Dictionary of all loss components and total loss
        """
        losses = {}
        device = outputs['cls_logits'].device

        labels = targets['label']
        defect_mask = targets['defect_mask']
        has_defect = targets['has_defect']

        # ============ 1. Classification Loss ============
        cls_loss = self.focal_loss(outputs['cls_logits'], labels)
        losses['cls'] = self.lambda_cls * cls_loss

        # ============ 2. Attention Mining Loss (GAIN S_am) ============
        am_loss = self.focal_loss(outputs['attended_cls_logits'], labels)
        losses['am'] = self.lambda_am * am_loss

        # ============ 3. CAM Guidance Loss (Strategy 2) ============
        # Supervise weight-based CAM directly from classifier (no extra module)
        if self.lambda_cam_guide > 0 and 'cam' in outputs:
            cam_losses = self.cam_guidance_loss(
                outputs['cam'],
                defect_mask,
                has_defect,
            )
            for key, value in cam_losses.items():
                losses[f'cam_guide_{key}'] = self.lambda_cam_guide * value
        else:
            losses['cam_guide_cam_bce'] = torch.tensor(0.0, device=device)
            losses['cam_guide_cam_dice'] = torch.tensor(0.0, device=device)

        # ============ 4. Localization Loss ============
        if self.lambda_loc > 0 and has_defect.sum() > 0:
            loc_map = outputs['localization_map']

            # Resize defect mask if needed
            if loc_map.shape[2:] != defect_mask.shape[2:]:
                defect_mask_resized = F.interpolate(
                    defect_mask,
                    size=loc_map.shape[2:],
                    mode='bilinear',
                    align_corners=False
                )
            else:
                defect_mask_resized = defect_mask

            # Only compute for defective samples
            loc_pred = loc_map[has_defect]
            loc_gt = defect_mask_resized[has_defect]

            # Dice loss
            dice_loss = self.dice_loss(loc_pred, loc_gt)

            # BCE loss
            bce_loss = F.binary_cross_entropy(loc_pred, loc_gt)

            losses['loc'] = self.lambda_loc * (dice_loss + bce_loss)
        else:
            losses['loc'] = torch.tensor(0.0, device=device)

        # ============ 5. Guided Attention Loss (attention module) ============
        if self.lambda_guide > 0:
            attention_losses = self.attention_guidance_loss(
                outputs['attention_map'],
                defect_mask,
                has_defect,
            )
            for key, value in attention_losses.items():
                losses[f'guide_{key}'] = self.lambda_guide * value
        else:
            losses['guide_attention_bce'] = torch.tensor(0.0, device=device)
            losses['guide_attention_dice'] = torch.tensor(0.0, device=device)
            losses['guide_attention_iou'] = torch.tensor(0.0, device=device)
            losses['guide_attention_entropy'] = torch.tensor(0.0, device=device)

        # ============ 7. Counterfactual Loss ============
        if self.lambda_cf > 0 and outputs['cf_logits'] is not None and has_defect.sum() > 0:
            # Counterfactual should predict normal (class 0) for defective samples
            cf_logits = outputs['cf_logits']

            # Create target: all zeros (normal class)
            cf_target = torch.zeros(cf_logits.size(0), dtype=torch.long, device=device)

            cf_loss = F.cross_entropy(cf_logits, cf_target)
            losses['cf'] = self.lambda_cf * cf_loss
        else:
            losses['cf'] = torch.tensor(0.0, device=device)

        # ============ 8. Consistency Loss ============
        # Attention map and localization map should be consistent for defective samples
        if self.lambda_consist > 0 and has_defect.sum() > 0:
            attention_map = outputs['attention_map']
            loc_map = outputs['localization_map']

            # Resize to same size
            if attention_map.shape[2:] != loc_map.shape[2:]:
                attention_resized = F.interpolate(
                    attention_map,
                    size=loc_map.shape[2:],
                    mode='bilinear',
                    align_corners=False
                )
            else:
                attention_resized = attention_map

            # Only for defective samples
            attn_defect = attention_resized[has_defect]
            loc_defect = loc_map[has_defect]

            consistency_loss = F.mse_loss(attn_defect, loc_defect)
            losses['consist'] = self.lambda_consist * consistency_loss
        else:
            losses['consist'] = torch.tensor(0.0, device=device)

        # ============ Total Loss ============
        total_loss = sum(v for k, v in losses.items()
                        if not k.startswith('guide_') and not k.startswith('cam_guide_'))
        total_loss += sum(v for k, v in losses.items() if k.startswith('guide_'))
        total_loss += sum(v for k, v in losses.items() if k.startswith('cam_guide_'))
        losses['total'] = total_loss

        return losses

    def update_weights(
        self,
        lambda_cls: Optional[float] = None,
        lambda_am: Optional[float] = None,
        lambda_cam_guide: Optional[float] = None,
        lambda_loc: Optional[float] = None,
        lambda_guide: Optional[float] = None,
        lambda_cf: Optional[float] = None,
        lambda_consist: Optional[float] = None,
    ):
        """Update loss weights (useful for multi-stage training)."""
        if lambda_cls is not None:
            self.lambda_cls = lambda_cls
        if lambda_am is not None:
            self.lambda_am = lambda_am
        if lambda_cam_guide is not None:
            self.lambda_cam_guide = lambda_cam_guide
        if lambda_loc is not None:
            self.lambda_loc = lambda_loc
        if lambda_guide is not None:
            self.lambda_guide = lambda_guide
        if lambda_cf is not None:
            self.lambda_cf = lambda_cf
        if lambda_consist is not None:
            self.lambda_consist = lambda_consist


def build_loss(config: Dict) -> GAINMTLLoss:
    """Factory function to build loss from config."""
    return GAINMTLLoss(
        lambda_cls=config.get('lambda_cls', 1.0),
        lambda_am=config.get('lambda_am', 0.5),
        lambda_cam_guide=config.get('lambda_cam_guide', 0.0),
        lambda_loc=config.get('lambda_loc', 0.3),
        lambda_guide=config.get('lambda_guide', 0.5),
        lambda_cf=config.get('lambda_cf', 0.3),
        lambda_consist=config.get('lambda_consist', 0.2),
        focal_gamma=config.get('focal_gamma', 2.0),
        focal_alpha=config.get('focal_alpha', 0.25),
    )
