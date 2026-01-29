"""
Evaluation Metrics for GAIN-MTL

Implements comprehensive metrics:
- Classification: Accuracy, Precision, Recall, F1, AUC-ROC
- CAM Quality: CAM-IoU, PointGame, Energy-Inside
- Localization: IoU, Dice, Pixel Accuracy

Reference metrics from paper evaluation tables.
"""

from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
)


def compute_classification_metrics(
    predictions: np.ndarray,
    labels: np.ndarray,
    probabilities: Optional[np.ndarray] = None,
) -> Dict[str, float]:
    """
    Compute classification metrics.

    Args:
        predictions: Predicted class labels
        labels: Ground truth labels
        probabilities: Predicted probabilities (for AUC-ROC)

    Returns:
        Dictionary of metrics
    """
    metrics = {}

    # Basic metrics (convert to float for JSON serialization)
    metrics['accuracy'] = float(accuracy_score(labels, predictions))
    metrics['precision'] = float(precision_score(labels, predictions, zero_division=0))
    metrics['recall'] = float(recall_score(labels, predictions, zero_division=0))
    metrics['f1'] = float(f1_score(labels, predictions, zero_division=0))

    # AUC-ROC (if probabilities provided)
    if probabilities is not None and len(np.unique(labels)) > 1:
        try:
            # Use probability of positive class
            if probabilities.ndim > 1:
                probs = probabilities[:, 1]
            else:
                probs = probabilities
            metrics['auc_roc'] = float(roc_auc_score(labels, probs))
        except ValueError:
            metrics['auc_roc'] = 0.0
    else:
        metrics['auc_roc'] = 0.0

    # Confusion matrix derived metrics
    cm = confusion_matrix(labels, predictions)
    if cm.shape == (2, 2):
        tn, fp, fn, tp = cm.ravel()
        metrics['true_positive'] = int(tp)
        metrics['true_negative'] = int(tn)
        metrics['false_positive'] = int(fp)
        metrics['false_negative'] = int(fn)
        metrics['specificity'] = float(tn / (tn + fp)) if (tn + fp) > 0 else 0.0
        metrics['false_positive_rate'] = float(fp / (fp + tn)) if (fp + tn) > 0 else 0.0
        metrics['false_negative_rate'] = float(fn / (fn + tp)) if (fn + tp) > 0 else 0.0

    return metrics


def compute_cam_metrics(
    attention_maps: Union[torch.Tensor, np.ndarray],
    defect_masks: Union[torch.Tensor, np.ndarray],
    threshold: float = 0.5,
) -> Dict[str, float]:
    """
    Compute CAM quality metrics.

    Args:
        attention_maps: Model's attention maps (B, 1, H, W) or (B, H, W)
        defect_masks: Ground truth defect masks (B, 1, H, W) or (B, H, W)
        threshold: Threshold for binarization

    Returns:
        Dictionary of CAM metrics
    """
    # Convert to numpy if needed
    if isinstance(attention_maps, torch.Tensor):
        attention_maps = attention_maps.detach().cpu().numpy()
    if isinstance(defect_masks, torch.Tensor):
        defect_masks = defect_masks.detach().cpu().numpy()

    # Ensure 4D
    if attention_maps.ndim == 3:
        attention_maps = attention_maps[:, np.newaxis, :, :]
    if defect_masks.ndim == 3:
        defect_masks = defect_masks[:, np.newaxis, :, :]

    # Resize attention to mask size if needed
    if attention_maps.shape[2:] != defect_masks.shape[2:]:
        attention_maps = _resize_maps(attention_maps, defect_masks.shape[2:])

    batch_size = attention_maps.shape[0]
    ious = []
    point_games = []
    energy_inside = []

    for i in range(batch_size):
        attn = attention_maps[i, 0]
        mask = defect_masks[i, 0]

        # Skip if mask is empty (shouldn't happen for defective samples)
        if mask.sum() == 0:
            continue

        # 1. CAM-IoU
        attn_binary = (attn > threshold).astype(np.float32)
        mask_binary = (mask > threshold).astype(np.float32)

        intersection = (attn_binary * mask_binary).sum()
        union = attn_binary.sum() + mask_binary.sum() - intersection

        iou = (intersection + 1e-6) / (union + 1e-6)
        ious.append(iou)

        # 2. PointGame: Is the max activation inside the defect region?
        max_idx = np.unravel_index(np.argmax(attn), attn.shape)
        point_game = mask[max_idx] > threshold
        point_games.append(float(point_game))

        # 3. Energy-Inside: What fraction of attention energy is inside defect region?
        total_energy = attn.sum()
        inside_energy = (attn * mask_binary).sum()
        energy_ratio = inside_energy / (total_energy + 1e-6)
        energy_inside.append(energy_ratio)

    metrics = {
        'cam_iou': float(np.mean(ious)) if ious else 0.0,
        'point_game': float(np.mean(point_games)) if point_games else 0.0,
        'energy_inside': float(np.mean(energy_inside)) if energy_inside else 0.0,
    }

    return metrics


def compute_localization_metrics(
    predictions: Union[torch.Tensor, np.ndarray],
    targets: Union[torch.Tensor, np.ndarray],
    threshold: float = 0.5,
) -> Dict[str, float]:
    """
    Compute localization/segmentation metrics.

    Args:
        predictions: Predicted masks (B, 1, H, W)
        targets: Ground truth masks (B, 1, H, W)
        threshold: Threshold for binarization

    Returns:
        Dictionary of localization metrics
    """
    # Convert to numpy
    if isinstance(predictions, torch.Tensor):
        predictions = predictions.detach().cpu().numpy()
    if isinstance(targets, torch.Tensor):
        targets = targets.detach().cpu().numpy()

    # Binarize
    pred_binary = (predictions > threshold).astype(np.float32)
    target_binary = (targets > threshold).astype(np.float32)

    # Flatten for pixel-level metrics
    pred_flat = pred_binary.flatten()
    target_flat = target_binary.flatten()

    # IoU
    intersection = (pred_flat * target_flat).sum()
    union = pred_flat.sum() + target_flat.sum() - intersection
    iou = (intersection + 1e-6) / (union + 1e-6)

    # Dice
    dice = (2 * intersection + 1e-6) / (pred_flat.sum() + target_flat.sum() + 1e-6)

    # Pixel Accuracy
    pixel_accuracy = (pred_flat == target_flat).mean()

    # Precision and Recall for segmentation
    tp = (pred_flat * target_flat).sum()
    fp = (pred_flat * (1 - target_flat)).sum()
    fn = ((1 - pred_flat) * target_flat).sum()

    precision = tp / (tp + fp + 1e-6)
    recall = tp / (tp + fn + 1e-6)

    return {
        'loc_iou': float(iou),
        'loc_dice': float(dice),
        'loc_pixel_accuracy': float(pixel_accuracy),
        'loc_precision': float(precision),
        'loc_recall': float(recall),
    }


def _resize_maps(maps: np.ndarray, target_size: Tuple[int, int]) -> np.ndarray:
    """Resize maps to target size using bilinear interpolation."""
    import cv2
    batch_size = maps.shape[0]
    resized = np.zeros((batch_size, 1, target_size[0], target_size[1]))

    for i in range(batch_size):
        resized[i, 0] = cv2.resize(
            maps[i, 0],
            (target_size[1], target_size[0]),  # cv2 uses (W, H)
            interpolation=cv2.INTER_LINEAR
        )

    return resized


class MetricTracker:
    """
    Tracks and aggregates metrics during training/evaluation.

    Supports:
    - Running averages
    - Per-class metrics
    - Metric history
    """

    def __init__(self, metrics: List[str]):
        """
        Initialize metric tracker.

        Args:
            metrics: List of metric names to track
        """
        self.metrics = metrics
        self.reset()

    def reset(self):
        """Reset all tracked values."""
        self._values = {m: [] for m in self.metrics}
        self._counts = {m: 0 for m in self.metrics}

    def update(self, metric_dict: Dict[str, float], n: int = 1):
        """
        Update metrics with new values.

        Args:
            metric_dict: Dictionary of metric values
            n: Number of samples (for weighted average)
        """
        for key, value in metric_dict.items():
            if key in self._values:
                self._values[key].append(value)
                self._counts[key] += n

    def get_average(self, metric: str) -> float:
        """Get running average for a metric."""
        if metric in self._values and len(self._values[metric]) > 0:
            return np.mean(self._values[metric])
        return 0.0

    def get_all_averages(self) -> Dict[str, float]:
        """Get averages for all metrics."""
        return {m: self.get_average(m) for m in self.metrics}

    def get_history(self, metric: str) -> List[float]:
        """Get full history for a metric."""
        return self._values.get(metric, [])


def _compute_per_image_cam_metrics(
    attention_map: np.ndarray,
    defect_mask: np.ndarray,
    threshold: float = 0.5,
) -> Dict[str, float]:
    """
    Compute CAM quality metrics for a single image.

    Args:
        attention_map: Attention map (1, H, W) or (H, W)
        defect_mask: Ground truth mask (1, H, W) or (H, W)
        threshold: Binarization threshold

    Returns:
        Dictionary with cam_iou, point_game, energy_inside
    """
    attn = attention_map[0] if attention_map.ndim == 3 else attention_map
    mask = defect_mask[0] if defect_mask.ndim == 3 else defect_mask

    # Resize if needed
    if attn.shape != mask.shape:
        import cv2
        attn = cv2.resize(attn, (mask.shape[1], mask.shape[0]), interpolation=cv2.INTER_LINEAR)

    if mask.sum() == 0:
        return {'cam_iou': 0.0, 'point_game': 0.0, 'energy_inside': 0.0}

    attn_bin = (attn > threshold).astype(np.float32)
    mask_bin = (mask > threshold).astype(np.float32)

    intersection = (attn_bin * mask_bin).sum()
    union = attn_bin.sum() + mask_bin.sum() - intersection
    cam_iou = float((intersection + 1e-6) / (union + 1e-6))

    max_idx = np.unravel_index(np.argmax(attn), attn.shape)
    point_game = float(mask[max_idx] > threshold)

    total_energy = attn.sum()
    inside_energy = (attn * mask_bin).sum()
    energy_inside = float(inside_energy / (total_energy + 1e-6))

    return {'cam_iou': cam_iou, 'point_game': point_game, 'energy_inside': energy_inside}


def _compute_per_image_loc_metrics(
    prediction: np.ndarray,
    target: np.ndarray,
    threshold: float = 0.5,
) -> Dict[str, float]:
    """
    Compute localization metrics for a single image.

    Args:
        prediction: Predicted mask (1, H, W) or (H, W)
        target: Ground truth mask (1, H, W) or (H, W)
        threshold: Binarization threshold

    Returns:
        Dictionary with loc_iou, loc_dice
    """
    pred = prediction.flatten()
    tgt = target.flatten()

    pred_bin = (pred > threshold).astype(np.float32)
    tgt_bin = (tgt > threshold).astype(np.float32)

    intersection = (pred_bin * tgt_bin).sum()
    union = pred_bin.sum() + tgt_bin.sum() - intersection
    iou = float((intersection + 1e-6) / (union + 1e-6))
    dice = float((2 * intersection + 1e-6) / (pred_bin.sum() + tgt_bin.sum() + 1e-6))

    return {'loc_iou': iou, 'loc_dice': dice}


def evaluate_model(
    model: torch.nn.Module,
    dataloader: torch.utils.data.DataLoader,
    device: torch.device,
    threshold: float = 0.5,
    return_per_image: bool = False,
    cam_source: str = 'attention_map_prob',
) -> Union[Dict[str, float], Tuple[Dict[str, float], List[Dict[str, Any]]]]:
    """
    Comprehensive model evaluation.

    Args:
        model: Model to evaluate
        dataloader: Data loader
        device: Device to use
        threshold: Threshold for predictions
        return_per_image: If True, also return per-image results
        cam_source: Key for CAM probabilities in model outputs.
            'cam_prob' for Strategy 2 (weight-based CAM),
            'attention_map_prob' for Strategy 3+ (attention module).
            Both are already sigmoid-applied [0,1].

    Returns:
        Dictionary of all metrics, or tuple of (metrics, per_image_results) if
        return_per_image is True.
    """
    model.eval()

    all_predictions = []
    all_labels = []
    all_probabilities = []
    all_attention_maps = []
    all_defect_masks = []
    all_loc_maps = []
    all_has_defect = []
    all_image_paths = []

    with torch.no_grad():
        for batch in dataloader:
            images = batch['image'].to(device)
            labels = batch['label'].to(device)
            defect_masks = batch['defect_mask'].to(device)
            has_defect = batch['has_defect'].to(device)
            image_paths = batch.get('image_path', [None] * len(images))

            outputs = model(images)

            # Classification
            probs = F.softmax(outputs['cls_logits'], dim=1)
            preds = probs.argmax(dim=1)

            all_predictions.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_probabilities.extend(probs.cpu().numpy())

            # Attention and localization (for defective samples)
            # Use pre-computed probabilities (already sigmoid-applied)
            all_attention_maps.append(outputs[cam_source].cpu())
            all_defect_masks.append(defect_masks.cpu())
            all_loc_maps.append(outputs['localization_map_prob'].cpu())
            all_has_defect.extend(has_defect.cpu().numpy())

            if return_per_image:
                all_image_paths.extend(image_paths)

    # Classification metrics
    all_predictions = np.array(all_predictions)
    all_labels = np.array(all_labels)
    all_probabilities = np.array(all_probabilities)
    all_has_defect = np.array(all_has_defect)

    cls_metrics = compute_classification_metrics(
        all_predictions, all_labels, all_probabilities
    )

    # CAM and localization metrics (only for defective samples)
    all_attention_maps = torch.cat(all_attention_maps, dim=0)
    all_defect_masks = torch.cat(all_defect_masks, dim=0)
    all_loc_maps = torch.cat(all_loc_maps, dim=0)

    defect_attention = all_attention_maps[all_has_defect]
    defect_masks = all_defect_masks[all_has_defect]
    defect_loc = all_loc_maps[all_has_defect]

    if len(defect_attention) > 0:
        cam_metrics = compute_cam_metrics(defect_attention, defect_masks, threshold)
        loc_metrics = compute_localization_metrics(defect_loc, defect_masks, threshold)
    else:
        cam_metrics = {'cam_iou': 0.0, 'point_game': 0.0, 'energy_inside': 0.0}
        loc_metrics = {'loc_iou': 0.0, 'loc_dice': 0.0, 'loc_pixel_accuracy': 0.0}

    # Combine all metrics
    all_metrics = {**cls_metrics, **cam_metrics, **loc_metrics}

    if not return_per_image:
        return all_metrics

    # Build per-image results
    per_image_results = []
    for i in range(len(all_predictions)):
        pred = int(all_predictions[i])
        label = int(all_labels[i])
        probs_i = all_probabilities[i]
        defect_prob = float(probs_i[1]) if len(probs_i) > 1 else float(probs_i[0])

        result = {
            'image_path': all_image_paths[i] if all_image_paths else None,
            'ground_truth': label,
            'prediction': pred,
            'confidence': float(probs_i[pred]),
            'defect_probability': defect_prob,
            'correct': pred == label,
        }

        # Add CAM/localization metrics for defective samples (ground truth)
        if all_has_defect[i]:
            attn_np = all_attention_maps[i].numpy()
            mask_np = all_defect_masks[i].numpy()
            loc_np = all_loc_maps[i].numpy()

            cam_per = _compute_per_image_cam_metrics(attn_np, mask_np, threshold)
            loc_per = _compute_per_image_loc_metrics(loc_np, mask_np, threshold)
            result.update(cam_per)
            result.update(loc_per)

        per_image_results.append(result)

    return all_metrics, per_image_results
