"""
Explanation and Visualization Module for GAIN-MTL

Provides:
- CAM/Attention visualization
- Counterfactual explanations
- Comprehensive prediction explanations
- Comparison visualizations
"""

from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.colors import LinearSegmentedColormap
import cv2


class DefectExplainer:
    """
    Explanation generator for GAIN-MTL model predictions.

    Generates comprehensive explanations including:
    - Attention map visualization
    - Localization prediction
    - Counterfactual analysis
    - Confidence scores
    """

    def __init__(
        self,
        model: torch.nn.Module,
        device: Optional[torch.device] = None,
        class_names: Optional[List[str]] = None,
        strategy: int = 3,
    ):
        """
        Initialize explainer.

        Args:
            model: Trained GAIN-MTL model
            device: Device to use (auto-detect if None)
            class_names: List of class names (default: ['Normal', 'Defective'])
            strategy: Training strategy (1-2: cam_prob, 3+: attention_map_prob)
        """
        self.model = model
        self.device = device or next(model.parameters()).device
        self.class_names = class_names or ['Normal', 'Defective']
        self.strategy = strategy
        self.cam_source = 'cam_prob' if strategy <= 2 else 'attention_map_prob'
        self.model.eval()

    @torch.no_grad()
    def explain(
        self,
        image: Union[torch.Tensor, np.ndarray],
        defect_mask: Optional[Union[torch.Tensor, np.ndarray]] = None,
    ) -> Dict[str, Any]:
        """
        Generate comprehensive explanation for a single image.

        Args:
            image: Input image (C, H, W) or (H, W, C)
            defect_mask: Optional ground truth mask for comparison

        Returns:
            Dictionary containing all explanation components
        """
        # Prepare image
        if isinstance(image, np.ndarray):
            if image.ndim == 3 and image.shape[2] == 3:
                image = torch.from_numpy(image.transpose(2, 0, 1))
            image = image.float()
            if image.max() > 1:
                image = image / 255.0

        # Add batch dimension
        if image.dim() == 3:
            image = image.unsqueeze(0)

        image = image.to(self.device)

        # Get model outputs (need all outputs for explanation)
        outputs = self.model(image, skip_unused=False)

        # Classification results
        cls_logits = outputs['cls_logits'][0]
        cls_probs = F.softmax(cls_logits, dim=0)
        pred_class = cls_probs.argmax().item()
        pred_conf = cls_probs[pred_class].item()

        # Attended classification
        attended_logits = outputs['attended_cls_logits'][0]
        attended_probs = F.softmax(attended_logits, dim=0)
        attended_pred = attended_probs.argmax().item()

        # CAM for defective class (strategy-dependent source)
        cam_map = outputs[self.cam_source][0, 0]
        cam_up = F.interpolate(
            outputs[self.cam_source],
            size=image.shape[2:],
            mode='bilinear',
            align_corners=False
        )[0, 0]

        # Localization map
        loc_map = outputs['localization_map_prob'][0, 0]

        # Build explanation
        explanation = {
            'prediction': pred_class,
            'prediction_name': self.class_names[pred_class],
            'confidence': pred_conf,
            'probabilities': cls_probs.cpu().numpy(),
            'attended_prediction': attended_pred,
            'attended_confidence': attended_probs[attended_pred].item(),
            'consistency': pred_class == attended_pred,
            'cam': cam_up.cpu().numpy(),
            'cam_raw': cam_map.cpu().numpy(),
            'localization_map': loc_map.cpu().numpy(),
            'features': outputs['features'][0].cpu().numpy(),
        }

        # Compare with ground truth if provided
        if defect_mask is not None:
            if isinstance(defect_mask, torch.Tensor):
                defect_mask = defect_mask.cpu().numpy()
            if defect_mask.ndim == 3:
                defect_mask = defect_mask[0]

            explanation['ground_truth_mask'] = defect_mask
            explanation['cam_iou'] = self._compute_iou(
                cam_up.cpu().numpy(),
                defect_mask
            )
            explanation['loc_iou'] = self._compute_iou(
                loc_map.cpu().numpy(),
                defect_mask
            )

        # Counterfactual analysis (if defective)
        if pred_class == 1 and outputs.get('cf_logits') is not None:
            cf_logits = outputs['cf_logits'][0]
            cf_probs = F.softmax(cf_logits, dim=0)
            explanation['counterfactual'] = {
                'prediction': cf_probs.argmax().item(),
                'probabilities': cf_probs.cpu().numpy(),
                'confidence_normal': cf_probs[0].item(),
                'message': f"If defect region removed, normal probability: {cf_probs[0].item():.2%}"
            }
        else:
            explanation['counterfactual'] = None

        return explanation

    def _compute_iou(
        self,
        pred: np.ndarray,
        target: np.ndarray,
        threshold: float = 0.5,
    ) -> float:
        """Compute IoU between prediction and target."""
        # Resize if needed
        if pred.shape != target.shape:
            pred = cv2.resize(pred, (target.shape[1], target.shape[0]))

        pred_binary = (pred > threshold).astype(np.float32)
        target_binary = (target > threshold).astype(np.float32)

        intersection = (pred_binary * target_binary).sum()
        union = pred_binary.sum() + target_binary.sum() - intersection

        return float((intersection + 1e-6) / (union + 1e-6))

    def visualize(
        self,
        image: Union[torch.Tensor, np.ndarray],
        explanation: Optional[Dict[str, Any]] = None,
        defect_mask: Optional[np.ndarray] = None,
        save_path: Optional[str] = None,
        figsize: Tuple[int, int] = (15, 10),
        image_path: Optional[str] = None,
        gt_label: Optional[int] = None,
    ) -> plt.Figure:
        """
        Visualize explanation.

        Args:
            image: Original image
            explanation: Pre-computed explanation (or will compute)
            defect_mask: Optional ground truth mask
            save_path: Path to save figure
            figsize: Figure size
            image_path: Optional source image path to display
            gt_label: Optional ground truth label (0=Normal, 1=Defective)

        Returns:
            Matplotlib figure
        """
        # Get explanation if not provided
        if explanation is None:
            explanation = self.explain(image, defect_mask)

        # Convert image for display
        if isinstance(image, torch.Tensor):
            image = image.cpu().numpy()
        if image.ndim == 4:
            image = image[0]
        if image.shape[0] == 3:
            image = image.transpose(1, 2, 0)

        # Denormalize ImageNet normalization if needed
        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])
        if image.dtype != np.uint8:
            if image.min() < 0 or image.max() <= 1:
                # ImageNet-normalized (range ~ [-2.1, 2.7]) or [0, 1] scaled
                image = image * std + mean
            image = np.clip(image * 255, 0, 255).astype(np.uint8)

        # Create figure
        # Top row: Image, GT Mask, CAM (defective class)
        # Bottom row: Localization, CAM vs GT, Counterfactual
        fig, axes = plt.subplots(2, 3, figsize=figsize)

        # Build suptitle with source path and GT label
        suptitle_parts = []
        if image_path:
            suptitle_parts.append(f'Source: {image_path}')
        if gt_label is not None:
            gt_label_name = self.class_names[gt_label]
            suptitle_parts.append(f'GT: {gt_label_name}')
        if suptitle_parts:
            fig.suptitle('  |  '.join(suptitle_parts), fontsize=10, y=0.995)

        # === Top Row ===

        # 1. Original image with prediction
        axes[0, 0].imshow(image)
        pred_name = explanation['prediction_name']
        conf = explanation['confidence']
        color = 'green' if explanation['prediction'] == 0 else 'red'
        axes[0, 0].set_title(f'Prediction: {pred_name}\nConfidence: {conf:.2%}', color=color)
        axes[0, 0].axis('off')

        # 2. Ground truth mask
        if 'ground_truth_mask' in explanation:
            axes[0, 1].imshow(image)
            gt_overlay = axes[0, 1].imshow(
                explanation['ground_truth_mask'],
                alpha=0.6,
                cmap='Reds',
                vmin=0,
                vmax=1,
            )
            axes[0, 1].set_title('Ground Truth Mask')
            axes[0, 1].axis('off')
            plt.colorbar(gt_overlay, ax=axes[0, 1], fraction=0.046)
        else:
            axes[0, 1].text(
                0.5, 0.5,
                'No Ground Truth\nProvided',
                ha='center', va='center',
                fontsize=12, transform=axes[0, 1].transAxes
            )
            axes[0, 1].axis('off')

        # 3. CAM (defective class)
        axes[0, 2].imshow(image)
        cam_overlay = axes[0, 2].imshow(
            explanation['cam'],
            alpha=0.6,
            cmap='jet',
            vmin=0,
            vmax=1,
        )
        iou_text = f"\nCAM-IoU: {explanation.get('cam_iou', 0):.3f}" if 'cam_iou' in explanation else ''
        axes[0, 2].set_title(f'CAM (Defective){iou_text}')
        axes[0, 2].axis('off')
        plt.colorbar(cam_overlay, ax=axes[0, 2], fraction=0.046)

        # === Bottom Row ===

        # 4. Localization map
        axes[1, 0].imshow(image)
        loc_overlay = axes[1, 0].imshow(
            explanation['localization_map'],
            alpha=0.6,
            cmap='hot',
            vmin=0,
            vmax=1,
        )
        axes[1, 0].set_title('Localization Map\n(Detected defect regions)')
        axes[1, 0].axis('off')
        plt.colorbar(loc_overlay, ax=axes[1, 0], fraction=0.046)

        # 5. CAM vs GT comparison
        if 'ground_truth_mask' in explanation:
            cam = explanation['cam']
            gt = explanation['ground_truth_mask']
            if cam.shape != gt.shape:
                cam = cv2.resize(cam, (gt.shape[1], gt.shape[0]))

            axes[1, 1].imshow(cam, cmap='jet')
            axes[1, 1].contour(gt, colors='red', linewidths=2, levels=[0.5])
            axes[1, 1].set_title('CAM vs GT\n(Red: GT boundary)')
            axes[1, 1].axis('off')
        else:
            axes[1, 1].imshow(explanation['cam'], cmap='jet')
            axes[1, 1].set_title('CAM')
            axes[1, 1].axis('off')

        # 6. Counterfactual explanation
        if explanation['counterfactual'] is not None:
            cf = explanation['counterfactual']
            text = f"Counterfactual Analysis:\n\n{cf['message']}"
            axes[1, 2].text(
                0.5, 0.5, text,
                ha='center', va='center',
                fontsize=11,
                transform=axes[1, 2].transAxes,
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5)
            )
            axes[1, 2].set_title('Counterfactual\n(Causal Analysis)')
        else:
            axes[1, 2].text(
                0.5, 0.5,
                'Counterfactual\nNot Available\n(Normal sample)',
                ha='center', va='center',
                fontsize=11,
                transform=axes[1, 2].transAxes
            )
        axes[1, 2].axis('off')

        plt.tight_layout(rect=[0, 0, 1, 0.97])

        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f'Saved explanation to {save_path}')

        return fig


def visualize_explanation(
    image: np.ndarray,
    attention_map: np.ndarray,
    localization_map: np.ndarray,
    prediction: int,
    confidence: float,
    defect_mask: Optional[np.ndarray] = None,
    class_names: List[str] = ['Normal', 'Defective'],
    save_path: Optional[str] = None,
) -> plt.Figure:
    """
    Standalone visualization function.

    Args:
        image: Original image (H, W, 3)
        attention_map: Attention map (H, W)
        localization_map: Localization map (H, W)
        prediction: Predicted class
        confidence: Prediction confidence
        defect_mask: Optional ground truth mask
        class_names: Class names
        save_path: Path to save figure

    Returns:
        Matplotlib figure
    """
    fig, axes = plt.subplots(1, 4 if defect_mask is not None else 3, figsize=(16, 4))

    # Original image
    axes[0].imshow(image)
    color = 'green' if prediction == 0 else 'red'
    axes[0].set_title(f'{class_names[prediction]}: {confidence:.2%}', color=color)
    axes[0].axis('off')

    # Attention map
    axes[1].imshow(image)
    axes[1].imshow(attention_map, alpha=0.6, cmap='jet')
    axes[1].set_title('Attention Map')
    axes[1].axis('off')

    # Localization map
    axes[2].imshow(image)
    axes[2].imshow(localization_map, alpha=0.6, cmap='hot')
    axes[2].set_title('Localization')
    axes[2].axis('off')

    # Ground truth (if provided)
    if defect_mask is not None:
        axes[3].imshow(image)
        axes[3].imshow(defect_mask, alpha=0.6, cmap='Reds')
        axes[3].set_title('Ground Truth')
        axes[3].axis('off')

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')

    return fig


def create_comparison_figure(
    images: List[np.ndarray],
    attention_maps: List[np.ndarray],
    predictions: List[int],
    confidences: List[float],
    defect_masks: Optional[List[np.ndarray]] = None,
    titles: Optional[List[str]] = None,
    class_names: List[str] = ['Normal', 'Defective'],
    save_path: Optional[str] = None,
    max_cols: int = 4,
) -> plt.Figure:
    """
    Create comparison figure for multiple samples.

    Args:
        images: List of images
        attention_maps: List of attention maps
        predictions: List of predictions
        confidences: List of confidences
        defect_masks: Optional list of ground truth masks
        titles: Optional titles for each sample
        class_names: Class names
        save_path: Path to save figure
        max_cols: Maximum columns

    Returns:
        Matplotlib figure
    """
    n_samples = len(images)
    n_cols = min(n_samples, max_cols)
    n_rows = (n_samples + n_cols - 1) // n_cols

    fig, axes = plt.subplots(n_rows * 2, n_cols, figsize=(4 * n_cols, 5 * n_rows))

    if n_rows == 1 and n_cols == 1:
        axes = np.array([[axes[0]], [axes[1]]])
    elif n_rows == 1:
        axes = axes.reshape(2, -1)
    elif n_cols == 1:
        axes = axes.reshape(-1, 2).T

    for i, (img, attn, pred, conf) in enumerate(
        zip(images, attention_maps, predictions, confidences)
    ):
        row = (i // n_cols) * 2
        col = i % n_cols

        # Image with prediction
        axes[row, col].imshow(img)
        color = 'green' if pred == 0 else 'red'
        title = titles[i] if titles else f'Sample {i+1}'
        axes[row, col].set_title(
            f'{title}\n{class_names[pred]}: {conf:.2%}',
            color=color
        )
        axes[row, col].axis('off')

        # Attention map
        axes[row + 1, col].imshow(img)
        axes[row + 1, col].imshow(attn, alpha=0.6, cmap='jet')
        if defect_masks and defect_masks[i] is not None:
            axes[row + 1, col].contour(
                defect_masks[i], colors='red', linewidths=1.5, levels=[0.5]
            )
        axes[row + 1, col].set_title('Attention')
        axes[row + 1, col].axis('off')

    # Hide empty subplots
    for i in range(n_samples, n_rows * n_cols):
        row = (i // n_cols) * 2
        col = i % n_cols
        axes[row, col].axis('off')
        axes[row + 1, col].axis('off')

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')

    return fig
