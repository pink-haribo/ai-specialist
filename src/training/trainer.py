"""
Multi-Stage Trainer for GAIN-MTL

Implements the training strategy from GAIN paper:
- Stage 1: Warm-up (Classification only)
- Stage 2: Add Attention Mining
- Stage 3: Add Localization
- Stage 4: Full training with all losses

Also supports:
- Mixed precision training
- Gradient accumulation
- Learning rate scheduling
- Comprehensive logging
"""

import os
from typing import Any, Callable, Dict, List, Optional, Tuple
from dataclasses import dataclass, field

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.cuda.amp import GradScaler, autocast
from torch.optim import Optimizer
from torch.optim.lr_scheduler import _LRScheduler

import numpy as np
from tqdm import tqdm

from ..models import GAINMTLModel
from ..losses import GAINMTLLoss


@dataclass
class TrainingConfig:
    """Configuration for training."""
    # Basic training
    num_epochs: int = 100
    batch_size: int = 16
    learning_rate: float = 1e-4
    weight_decay: float = 1e-5

    # Multi-stage ratios (fraction of total epochs)
    stage1_ratio: float = 0.25  # Classification only
    stage2_ratio: float = 0.25  # + Attention mining
    stage3_ratio: float = 0.25  # + Localization
    stage4_ratio: float = 0.25  # Full training

    # Optimization
    gradient_accumulation_steps: int = 1
    max_grad_norm: float = 1.0
    use_amp: bool = True

    # Scheduler
    scheduler_type: str = 'cosine'  # 'cosine', 'step', 'plateau'
    warmup_epochs: int = 5

    # Logging & Checkpointing
    log_interval: int = 10
    eval_interval: int = 1
    save_interval: int = 5
    checkpoint_dir: str = 'checkpoints'

    # Device
    device: str = 'cuda'


class GAINMTLTrainer:
    """
    Trainer for GAIN-MTL model.

    Handles:
    - Forward/backward passes
    - Loss computation
    - Metric tracking
    - Checkpointing

    Args:
        model: GAIN-MTL model instance
        criterion: Loss function
        optimizer: Optimizer
        scheduler: Learning rate scheduler (optional)
        config: Training configuration
        logger: Logger for tracking metrics (optional)
    """

    def __init__(
        self,
        model: GAINMTLModel,
        criterion: GAINMTLLoss,
        optimizer: Optimizer,
        scheduler: Optional[_LRScheduler] = None,
        config: Optional[TrainingConfig] = None,
        logger: Optional[Any] = None,
    ):
        self.model = model
        self.criterion = criterion
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.config = config or TrainingConfig()
        self.logger = logger

        # Device setup
        self.device = torch.device(
            self.config.device if torch.cuda.is_available() else 'cpu'
        )
        self.model.to(self.device)

        # Mixed precision
        self.scaler = GradScaler() if self.config.use_amp else None

        # Tracking
        self.current_epoch = 0
        self.global_step = 0
        self.best_metric = 0.0

    def train_epoch(
        self,
        train_loader: DataLoader,
        epoch: int,
    ) -> Dict[str, float]:
        """
        Train for one epoch.

        Args:
            train_loader: Training data loader
            epoch: Current epoch number

        Returns:
            Dictionary of average losses for the epoch
        """
        self.model.train()
        self.current_epoch = epoch

        total_losses = {}
        num_batches = len(train_loader)
        accumulation_steps = self.config.gradient_accumulation_steps

        progress_bar = tqdm(
            train_loader,
            desc=f'Epoch {epoch}',
            leave=True,
        )

        for batch_idx, batch in enumerate(progress_bar):
            # Move to device
            images = batch['image'].to(self.device)
            labels = batch['label'].to(self.device)
            defect_masks = batch['defect_mask'].to(self.device)
            has_defect = batch['has_defect'].to(self.device)

            # Forward pass with optional AMP
            with autocast(enabled=self.config.use_amp):
                outputs = self.model(images, defect_mask=defect_masks)

                targets = {
                    'label': labels,
                    'defect_mask': defect_masks,
                    'has_defect': has_defect,
                }

                losses = self.criterion(outputs, targets)
                loss = losses['total'] / accumulation_steps

            # Backward pass
            if self.scaler is not None:
                self.scaler.scale(loss).backward()
            else:
                loss.backward()

            # Gradient accumulation
            if (batch_idx + 1) % accumulation_steps == 0:
                if self.scaler is not None:
                    self.scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(),
                        self.config.max_grad_norm
                    )
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                else:
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(),
                        self.config.max_grad_norm
                    )
                    self.optimizer.step()

                self.optimizer.zero_grad()
                self.global_step += 1

            # Accumulate losses
            for key, value in losses.items():
                if key not in total_losses:
                    total_losses[key] = []
                total_losses[key].append(value.item())

            # Update progress bar
            if batch_idx % self.config.log_interval == 0:
                progress_bar.set_postfix({
                    'loss': f"{losses['total'].item():.4f}",
                    'cls': f"{losses['cls'].item():.4f}",
                })

        # Compute averages
        avg_losses = {k: np.mean(v) for k, v in total_losses.items()}

        # Log to logger if available
        if self.logger is not None:
            for key, value in avg_losses.items():
                self.logger.log({f'train/{key}': value}, step=epoch)

        return avg_losses

    @torch.no_grad()
    def validate(
        self,
        val_loader: DataLoader,
        epoch: int,
    ) -> Dict[str, float]:
        """
        Validate the model.

        Args:
            val_loader: Validation data loader
            epoch: Current epoch number

        Returns:
            Dictionary of validation metrics
        """
        self.model.eval()

        total_losses = {}
        all_preds = []
        all_labels = []
        all_cam_ious = []

        for batch in tqdm(val_loader, desc='Validation', leave=False):
            images = batch['image'].to(self.device)
            labels = batch['label'].to(self.device)
            defect_masks = batch['defect_mask'].to(self.device)
            has_defect = batch['has_defect'].to(self.device)

            # Forward pass
            outputs = self.model(images)

            targets = {
                'label': labels,
                'defect_mask': defect_masks,
                'has_defect': has_defect,
            }

            losses = self.criterion(outputs, targets)

            # Accumulate losses
            for key, value in losses.items():
                if key not in total_losses:
                    total_losses[key] = []
                total_losses[key].append(value.item())

            # Classification predictions
            preds = outputs['cls_logits'].argmax(dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

            # CAM-IoU for defective samples
            if has_defect.sum() > 0:
                cam_iou = self._compute_cam_iou(
                    outputs['attention_map'][has_defect],
                    defect_masks[has_defect]
                )
                all_cam_ious.extend(cam_iou.cpu().numpy())

        # Compute metrics
        all_preds = np.array(all_preds)
        all_labels = np.array(all_labels)

        accuracy = (all_preds == all_labels).mean()
        avg_losses = {k: np.mean(v) for k, v in total_losses.items()}

        metrics = {
            'accuracy': accuracy,
            'cam_iou': np.mean(all_cam_ious) if all_cam_ious else 0.0,
            **{f'val_{k}': v for k, v in avg_losses.items()}
        }

        # Log to logger if available
        if self.logger is not None:
            for key, value in metrics.items():
                self.logger.log({f'val/{key}': value}, step=epoch)

        return metrics

    def _compute_cam_iou(
        self,
        attention_map: torch.Tensor,
        defect_mask: torch.Tensor,
        threshold: float = 0.5,
    ) -> torch.Tensor:
        """Compute IoU between attention map and defect mask."""
        # Resize attention to mask size
        if attention_map.shape[2:] != defect_mask.shape[2:]:
            attention_map = torch.nn.functional.interpolate(
                attention_map,
                size=defect_mask.shape[2:],
                mode='bilinear',
                align_corners=False
            )

        # Binarize
        attention_binary = (attention_map > threshold).float()
        mask_binary = (defect_mask > threshold).float()

        # Compute IoU
        intersection = (attention_binary * mask_binary).sum(dim=[1, 2, 3])
        union = attention_binary.sum(dim=[1, 2, 3]) + mask_binary.sum(dim=[1, 2, 3]) - intersection

        iou = (intersection + 1e-6) / (union + 1e-6)
        return iou

    def save_checkpoint(
        self,
        path: str,
        epoch: int,
        metrics: Dict[str, float],
    ):
        """Save model checkpoint."""
        os.makedirs(os.path.dirname(path), exist_ok=True)

        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict() if self.scheduler else None,
            'metrics': metrics,
            'config': self.config,
            'global_step': self.global_step,
            'best_metric': self.best_metric,
        }

        if self.scaler is not None:
            checkpoint['scaler_state_dict'] = self.scaler.state_dict()

        torch.save(checkpoint, path)
        print(f'Checkpoint saved to {path}')

    def load_checkpoint(self, path: str) -> int:
        """Load model checkpoint. Returns the epoch number."""
        checkpoint = torch.load(path, map_location=self.device)

        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

        if self.scheduler and checkpoint['scheduler_state_dict']:
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])

        if self.scaler is not None and 'scaler_state_dict' in checkpoint:
            self.scaler.load_state_dict(checkpoint['scaler_state_dict'])

        self.global_step = checkpoint.get('global_step', 0)
        self.best_metric = checkpoint.get('best_metric', 0.0)

        print(f'Checkpoint loaded from {path}')
        return checkpoint['epoch']


class MultiStageTrainer(GAINMTLTrainer):
    """
    Multi-Stage Training Strategy for GAIN-MTL.

    Implements progressive training:
    - Stage 1: Classification warm-up
    - Stage 2: Add attention mining
    - Stage 3: Add localization
    - Stage 4: Full training with counterfactual

    This strategy helps the model learn:
    1. Good features first (classification)
    2. Where to look (attention)
    3. Precise localization
    4. Correct reasoning (counterfactual)
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # Stage configurations
        self.stage_configs = self._create_stage_configs()
        self.current_stage = 1

    def _create_stage_configs(self) -> Dict[int, Dict[str, float]]:
        """Create loss weight configurations for each stage."""
        return {
            1: {  # Classification only
                'lambda_cls': 1.0,
                'lambda_am': 0.0,
                'lambda_loc': 0.0,
                'lambda_guide': 0.0,
                'lambda_cf': 0.0,
                'lambda_consist': 0.0,
            },
            2: {  # + Attention mining
                'lambda_cls': 1.0,
                'lambda_am': 0.5,
                'lambda_loc': 0.0,
                'lambda_guide': 0.3,
                'lambda_cf': 0.0,
                'lambda_consist': 0.0,
            },
            3: {  # + Localization
                'lambda_cls': 1.0,
                'lambda_am': 0.5,
                'lambda_loc': 0.3,
                'lambda_guide': 0.5,
                'lambda_cf': 0.0,
                'lambda_consist': 0.2,
            },
            4: {  # Full training
                'lambda_cls': 1.0,
                'lambda_am': 0.5,
                'lambda_loc': 0.3,
                'lambda_guide': 0.5,
                'lambda_cf': 0.3,
                'lambda_consist': 0.2,
            },
        }

    def _get_stage_for_epoch(self, epoch: int) -> int:
        """Determine the training stage for given epoch."""
        total_epochs = self.config.num_epochs
        stage1_end = int(total_epochs * self.config.stage1_ratio)
        stage2_end = int(total_epochs * (self.config.stage1_ratio + self.config.stage2_ratio))
        stage3_end = int(total_epochs * (self.config.stage1_ratio + self.config.stage2_ratio + self.config.stage3_ratio))

        if epoch < stage1_end:
            return 1
        elif epoch < stage2_end:
            return 2
        elif epoch < stage3_end:
            return 3
        else:
            return 4

    def _update_stage(self, epoch: int):
        """Update training stage if needed."""
        new_stage = self._get_stage_for_epoch(epoch)

        if new_stage != self.current_stage:
            print(f'\n{"="*50}')
            print(f'Transitioning from Stage {self.current_stage} to Stage {new_stage}')
            print(f'{"="*50}\n')

            self.current_stage = new_stage
            self.criterion.update_weights(**self.stage_configs[new_stage])

            # Log stage transition
            if self.logger is not None:
                self.logger.log({'stage': new_stage}, step=epoch)

    def train_epoch(
        self,
        train_loader: DataLoader,
        epoch: int,
    ) -> Dict[str, float]:
        """Train for one epoch with stage-aware loss weights."""
        # Update stage if needed
        self._update_stage(epoch)

        # Call parent train_epoch
        return super().train_epoch(train_loader, epoch)

    def train(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
        num_epochs: Optional[int] = None,
    ) -> Dict[str, List[float]]:
        """
        Full training loop with multi-stage strategy.

        Args:
            train_loader: Training data loader
            val_loader: Validation data loader
            num_epochs: Number of epochs (overrides config if provided)

        Returns:
            Training history
        """
        num_epochs = num_epochs or self.config.num_epochs
        history = {'train_loss': [], 'val_loss': [], 'val_accuracy': [], 'val_cam_iou': []}

        print(f'\nStarting Multi-Stage Training for {num_epochs} epochs')
        print(f'Stage 1 (Classification): epochs 0-{int(num_epochs * self.config.stage1_ratio) - 1}')
        print(f'Stage 2 (+ Attention): epochs {int(num_epochs * self.config.stage1_ratio)}-{int(num_epochs * (self.config.stage1_ratio + self.config.stage2_ratio)) - 1}')
        print(f'Stage 3 (+ Localization): epochs {int(num_epochs * (self.config.stage1_ratio + self.config.stage2_ratio))}-{int(num_epochs * (self.config.stage1_ratio + self.config.stage2_ratio + self.config.stage3_ratio)) - 1}')
        print(f'Stage 4 (Full): epochs {int(num_epochs * (self.config.stage1_ratio + self.config.stage2_ratio + self.config.stage3_ratio))}-{num_epochs - 1}')
        print()

        for epoch in range(num_epochs):
            # Train
            train_losses = self.train_epoch(train_loader, epoch)
            history['train_loss'].append(train_losses['total'])

            # Validate
            if epoch % self.config.eval_interval == 0:
                val_metrics = self.validate(val_loader, epoch)
                history['val_loss'].append(val_metrics['val_total'])
                history['val_accuracy'].append(val_metrics['accuracy'])
                history['val_cam_iou'].append(val_metrics['cam_iou'])

                print(f'\nEpoch {epoch} (Stage {self.current_stage}):')
                print(f'  Train Loss: {train_losses["total"]:.4f}')
                print(f'  Val Loss: {val_metrics["val_total"]:.4f}')
                print(f'  Val Accuracy: {val_metrics["accuracy"]:.4f}')
                print(f'  Val CAM-IoU: {val_metrics["cam_iou"]:.4f}')

                # Save best model
                if val_metrics['accuracy'] > self.best_metric:
                    self.best_metric = val_metrics['accuracy']
                    self.save_checkpoint(
                        os.path.join(self.config.checkpoint_dir, 'best_model.pth'),
                        epoch,
                        val_metrics
                    )

            # Periodic checkpoint
            if epoch % self.config.save_interval == 0:
                self.save_checkpoint(
                    os.path.join(self.config.checkpoint_dir, f'checkpoint_epoch_{epoch}.pth'),
                    epoch,
                    val_metrics if epoch % self.config.eval_interval == 0 else {}
                )

            # Update scheduler
            if self.scheduler is not None:
                self.scheduler.step()

        # Save final model
        self.save_checkpoint(
            os.path.join(self.config.checkpoint_dir, 'final_model.pth'),
            num_epochs - 1,
            val_metrics
        )

        return history
