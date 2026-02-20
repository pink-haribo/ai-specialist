#!/usr/bin/env python3
"""
Training Script for GAIN-MTL - Multi-Strategy Comparison

Trains 5 separate models, each with a different loss strategy:
    - Strategy 1: Classification only
    - Strategy 2: Classification + CAM Guidance (weight-based CAM supervised by GT mask)
    - Strategy 3: Classification + Attention Mining
    - Strategy 4: Classification + Attention Mining + Localization
    - Strategy 5: Full (All losses including Counterfactual)

This allows direct comparison of each strategy's performance.

Usage:
    python train.py --config configs/default.yaml
    python train.py --config configs/default.yaml --strategies 1 2 3 4 5
    python train.py --config configs/default.yaml --strategies 5  # Train only strategy 5

Backbone options (mmpretrain):
    b0  - EfficientNetV2-B0
    b1  - EfficientNetV2-B1
    s   - EfficientNetV2-S (Small)
    m   - EfficientNetV2-M (Medium)
    l   - EfficientNetV2-L (Large)
    xl  - EfficientNetV2-XL (Extra Large)
"""

import argparse
import json
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

import torch
import torch.nn as nn
from torch.optim import Adam, AdamW, SGD
from torch.optim.lr_scheduler import CosineAnnealingLR, StepLR, ReduceLROnPlateau

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

from src.models import GAINMTLModel
from src.losses import GAINMTLLoss
from src.training import GAINMTLTrainer, TrainingConfig
from src.data import create_dataloaders
from src.evaluation import evaluate_model
from src.utils import set_seed, get_device, load_config, save_config, print_model_summary


# Strategy configurations (fixed loss weights for each strategy)
#
# Strategy Guide:
# - Strategy 1: Classification only (baseline)
# - Strategy 2: Classification + CAM Guidance (weight-based CAM supervised by GT mask)
# - Strategy 3: Classification + Attention Mining
# - Strategy 4: Classification + Attention Mining + Localization
# - Strategy 5: Full (all losses including Counterfactual)
#
# Key difference between Strategy 2 and 3+:
# - Strategy 2: Uses weight-based CAM directly from classifier (no extra module)
# - Strategy 3+: Uses attention module with guided attention loss
STRATEGY_CONFIGS = {
    1: {
        'name': 'classification_only',
        'description': 'Classification loss only',
        'weights': {
            'lambda_cls': 1.0,
            'lambda_am': 0.0,
            'lambda_cam_guide': 0.0,
            'lambda_loc': 0.0,
            'lambda_guide': 0.0,
            'lambda_cf': 0.0,
            'lambda_consist': 0.0,
        }
    },
    2: {
        'name': 'cls_cam_guidance',
        'description': 'Classification + CAM Guidance (weight-based CAM supervised by GT mask)',
        'weights': {
            'lambda_cls': 1.0,
            'lambda_am': 0.0,
            'lambda_cam_guide': 0.5,
            'lambda_loc': 0.0,
            'lambda_guide': 0.0,
            'lambda_cf': 0.0,
            'lambda_consist': 0.0,
        }
    },
    3: {
        'name': 'cls_attention_mining',
        'description': 'Classification + Attention Mining + CAM Guidance',
        'weights': {
            'lambda_cls': 1.0,
            'lambda_am': 0.5,
            'lambda_cam_guide': 0.3,
            'lambda_loc': 0.0,
            'lambda_guide': 0.5,
            'lambda_cf': 0.0,
            'lambda_consist': 0.0,
        }
    },
    4: {
        'name': 'cls_attention_localization',
        'description': 'Classification + Attention Mining + Localization + CAM Guidance',
        'weights': {
            'lambda_cls': 1.0,
            'lambda_am': 0.5,
            'lambda_cam_guide': 0.3,
            'lambda_loc': 0.2,
            'lambda_guide': 0.5,
            'lambda_cf': 0.0,
            'lambda_consist': 0.2,
        },
        'loc_warmup_ratio': 0.5,
    },
    5: {
        'name': 'full',
        'description': 'Full training with all losses (including Counterfactual + CAM Guidance)',
        'weights': {
            'lambda_cls': 1.0,
            'lambda_am': 0.5,
            'lambda_cam_guide': 0.3,
            'lambda_loc': 0.2,
            'lambda_guide': 0.5,
            'lambda_cf': 0.3,
            'lambda_consist': 0.2,
        },
        'loc_warmup_ratio': 0.5,
    },
    6: {
        'name': 'full_with_gt_mask_attention',
        'description': 'Strategy 5 (full) + GT mask multiplicative fusion with curriculum',
        'weights': {
            'lambda_cls': 1.0,
            'lambda_am': 0.5,
            'lambda_cam_guide': 0.3,
            'lambda_loc': 0.2,
            'lambda_guide': 0.5,
            'lambda_cf': 0.3,
            'lambda_consist': 0.2,
        },
        'loc_warmup_ratio': 0.5,
        # Curriculum: blend alpha decays from start→end over first ratio of training
        # alpha=0.5: 50% GT replace + 50% fusion; alpha=0.0: pure fusion (matches inference)
        'ext_attn_alpha_start': 0.5,
        'ext_attn_alpha_end': 0.0,
        'ext_attn_alpha_ratio': 0.7,  # Decay over first 70% of epochs
    },
}


def parse_args():
    parser = argparse.ArgumentParser(description='Train GAIN-MTL models with different strategies')

    # Config
    parser.add_argument('--config', type=str, default='configs/default.yaml',
                        help='Path to config file')

    # Strategy selection
    parser.add_argument('--strategies', type=int, nargs='+', default=[1, 2, 3, 4, 5],
                        choices=[1, 2, 3, 4, 5, 6],
                        help='Which strategies to train (1-6). Default: 1-5')

    # Override config options
    parser.add_argument('--data_root', type=str, default=None,
                        help='Override data root directory')
    parser.add_argument('--backbone', type=str, default=None,
                        choices=['b0', 'b1', 's', 'm', 'l', 'xl'],
                        help='Override backbone arch (b0/b1/s/m/l/xl for EfficientNetV2)')
    parser.add_argument('--batch_size', type=int, default=None,
                        help='Override batch size')
    parser.add_argument('--epochs', type=int, default=None,
                        help='Override number of epochs')
    parser.add_argument('--lr', type=float, default=None,
                        help='Override learning rate')
    parser.add_argument('--device', type=str, default=None,
                        help='Override device')
    parser.add_argument('--val_ratio', type=float, default=None,
                        help='Ratio of training data for validation (0.0-1.0). If set, randomly splits train data.')

    # Experiment
    parser.add_argument('--name', type=str, default=None,
                        help='Experiment name prefix')
    parser.add_argument('--seed', type=int, default=None,
                        help='Random seed')

    # Logging
    parser.add_argument('--wandb', action='store_true',
                        help='Enable W&B logging')
    parser.add_argument('--no_tensorboard', action='store_true',
                        help='Disable TensorBoard logging')

    return parser.parse_args()


def create_model(config: dict, device: torch.device) -> GAINMTLModel:
    """Create a new GAIN-MTL model instance."""
    model = GAINMTLModel(
        backbone_arch=config['model']['backbone_arch'],
        num_classes=config['model']['num_classes'],
        pretrained=config['model']['pretrained'],
        fpn_channels=config['model']['fpn_channels'],
        attention_channels=config['model']['attention_channels'],
        use_counterfactual=config['model']['use_counterfactual'],
        freeze_backbone_stages=config['model']['freeze_backbone_stages'],
        out_indices=tuple(config['model'].get('out_indices', [3, 4, 5, 6])),
    )
    return model.to(device)


def create_criterion(config: dict, strategy_weights: dict) -> GAINMTLLoss:
    """Create loss function with strategy-specific weights."""
    return GAINMTLLoss(
        lambda_cls=strategy_weights['lambda_cls'],
        lambda_am=strategy_weights['lambda_am'],
        lambda_cam_guide=strategy_weights['lambda_cam_guide'],
        lambda_loc=strategy_weights['lambda_loc'],
        lambda_guide=strategy_weights['lambda_guide'],
        lambda_cf=strategy_weights['lambda_cf'],
        lambda_consist=strategy_weights['lambda_consist'],
        focal_gamma=config['loss']['focal_gamma'],
        focal_alpha=config['loss']['focal_alpha'],
    )


def create_optimizer(model: nn.Module, config: dict):
    """Create optimizer."""
    optimizer_name = config['training'].get('optimizer', 'adamw').lower()
    lr = config['training']['learning_rate']
    weight_decay = config['training']['weight_decay']

    if optimizer_name == 'adam':
        return Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    elif optimizer_name == 'adamw':
        return AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    elif optimizer_name == 'sgd':
        return SGD(model.parameters(), lr=lr, weight_decay=weight_decay, momentum=0.9)
    else:
        raise ValueError(f'Unknown optimizer: {optimizer_name}')


def create_scheduler(optimizer, config: dict):
    """Create learning rate scheduler."""
    scheduler_type = config['training'].get('scheduler_type', 'cosine')
    num_epochs = config['training']['num_epochs']

    if scheduler_type == 'cosine':
        return CosineAnnealingLR(
            optimizer,
            T_max=num_epochs,
            eta_min=config['training'].get('min_lr', 1e-6)
        )
    elif scheduler_type == 'step':
        return StepLR(optimizer, step_size=30, gamma=0.1)
    elif scheduler_type == 'plateau':
        return ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)
    else:
        return None


def train_single_strategy(
    strategy_id: int,
    config: dict,
    train_loader,
    val_loader,
    test_loader,
    device: torch.device,
    base_exp_dir: Path,
    logger=None,
) -> Dict:
    """
    Train a single model with a specific strategy.

    Args:
        strategy_id: Strategy number (1-4)
        config: Configuration dictionary
        train_loader: Training data loader
        val_loader: Validation data loader
        test_loader: Test data loader
        device: Device to train on
        base_exp_dir: Base experiment directory
        logger: Optional logger (wandb)

    Returns:
        Dictionary containing training history and test metrics
    """
    strategy = STRATEGY_CONFIGS[strategy_id]
    strategy_name = strategy['name']

    print('\n' + '=' * 70)
    print(f'STRATEGY {strategy_id}: {strategy_name.upper()}')
    print(f'Description: {strategy["description"]}')
    print('Loss weights:')
    for key, value in strategy['weights'].items():
        print(f'  {key}: {value}')
    print('=' * 70 + '\n')

    # Create experiment directory for this strategy
    exp_dir = base_exp_dir / f'strategy_{strategy_id}_{strategy_name}'
    exp_dir.mkdir(parents=True, exist_ok=True)

    # Reset seed for fair comparison
    set_seed(config['experiment']['seed'])

    # Create fresh model, criterion, optimizer, scheduler
    model = create_model(config, device)
    model.set_strategy(strategy_id)
    criterion = create_criterion(config, strategy['weights'])
    optimizer = create_optimizer(model, config)
    scheduler = create_scheduler(optimizer, config)

    print_model_summary(model)

    # Create training config with TensorBoard support
    training_config = TrainingConfig(
        num_epochs=config['training']['num_epochs'],
        batch_size=config['training']['batch_size'],
        learning_rate=config['training']['learning_rate'],
        weight_decay=config['training']['weight_decay'],
        # Set all stage ratios to 0 except stage4 (use fixed strategy throughout)
        stage1_ratio=0.0,
        stage2_ratio=0.0,
        stage3_ratio=0.0,
        stage4_ratio=1.0,
        gradient_accumulation_steps=config['training']['gradient_accumulation_steps'],
        max_grad_norm=config['training']['max_grad_norm'],
        use_amp=config['training']['use_amp'],
        log_interval=config['logging']['log_interval'],
        eval_interval=config['evaluation']['eval_interval'],
        save_interval=config['checkpoint']['save_interval'],
        checkpoint_dir=str(exp_dir),
        device=str(device),
        # TensorBoard settings
        use_tensorboard=config['logging'].get('use_tensorboard', True),
        tensorboard_dir=config['logging'].get('tensorboard_dir', './runs'),
    )

    # Create trainer (use base GAINMTLTrainer, not MultiStageTrainer)
    trainer = GAINMTLTrainer(
        model=model,
        criterion=criterion,
        optimizer=optimizer,
        scheduler=scheduler,
        config=training_config,
        logger=logger,
        experiment_name=f"{config['experiment']['name']}/strategy_{strategy_id}_{strategy_name}",
    )

    # Save strategy config
    strategy_config = {
        'strategy_id': strategy_id,
        'strategy_name': strategy_name,
        'description': strategy['description'],
        'loss_weights': strategy['weights'],
        'model_config': config['model'],
        'training_config': config['training'],
    }
    with open(exp_dir / 'strategy_config.json', 'w') as f:
        json.dump(strategy_config, f, indent=2)

    # Training loop
    num_epochs = config['training']['num_epochs']
    history = {'train_loss': [], 'val_loss': [], 'val_accuracy': [], 'val_cam_iou': []}

    # Localization warmup config
    loc_warmup_ratio = strategy.get('loc_warmup_ratio', 0.0)
    target_lambda_loc = strategy['weights']['lambda_loc']
    if loc_warmup_ratio > 0 and target_lambda_loc > 0:
        loc_warmup_epochs = int(num_epochs * loc_warmup_ratio)
        print(f'Localization warmup: lambda_loc 0.0 → {target_lambda_loc} over {loc_warmup_epochs} epochs')
    else:
        loc_warmup_epochs = 0

    # External attention alpha curriculum (Strategy 6)
    ext_attn_alpha_start = strategy.get('ext_attn_alpha_start', 0.0)
    ext_attn_alpha_end = strategy.get('ext_attn_alpha_end', 0.0)
    ext_attn_alpha_ratio = strategy.get('ext_attn_alpha_ratio', 0.7)
    use_alpha_schedule = ext_attn_alpha_start > 0 and hasattr(model, 'set_external_attention_alpha')
    if use_alpha_schedule:
        alpha_decay_epochs = int(num_epochs * ext_attn_alpha_ratio)
        print(f'External attention alpha: {ext_attn_alpha_start} → {ext_attn_alpha_end} over {alpha_decay_epochs} epochs')

    print(f'\nStarting training for {num_epochs} epochs...\n')

    for epoch in range(num_epochs):
        # Gradual localization warmup: linearly increase lambda_loc
        if loc_warmup_epochs > 0:
            if epoch < loc_warmup_epochs:
                warmup_loc = target_lambda_loc * (epoch / loc_warmup_epochs)
            else:
                warmup_loc = target_lambda_loc
            criterion.update_weights(lambda_loc=warmup_loc)

        # External attention alpha curriculum (Strategy 6):
        # Decays blend alpha from start→end so the model transitions
        # from GT-guided attention to pure multiplicative fusion
        if use_alpha_schedule:
            if epoch < alpha_decay_epochs:
                alpha = ext_attn_alpha_start + (ext_attn_alpha_end - ext_attn_alpha_start) * (epoch / alpha_decay_epochs)
            else:
                alpha = ext_attn_alpha_end
            model.set_external_attention_alpha(alpha)

        # Train
        train_losses = trainer.train_epoch(train_loader, epoch)
        history['train_loss'].append(train_losses['total'])

        # Validate
        if epoch % training_config.eval_interval == 0:
            val_metrics = trainer.validate(val_loader, epoch)
            history['val_loss'].append(val_metrics['val_total'])
            history['val_accuracy'].append(val_metrics['accuracy'])
            history['val_cam_iou'].append(val_metrics['cam_iou'])

            print(f'\nEpoch {epoch}:')
            print(f'  Train Loss: {train_losses["total"]:.4f}')
            print(f'  Val Loss: {val_metrics["val_total"]:.4f}')
            print(f'  Val Accuracy: {val_metrics["accuracy"]:.4f}')
            print(f'  Val CAM-IoU: {val_metrics["cam_iou"]:.4f}')

            # Save best model
            if val_metrics['accuracy'] > trainer.best_metric:
                trainer.best_metric = val_metrics['accuracy']
                trainer.save_checkpoint(
                    str(exp_dir / 'best_model.pth'),
                    epoch,
                    val_metrics
                )

        # Update scheduler
        if scheduler is not None:
            scheduler.step()

    # Save last model
    trainer.save_checkpoint(
        str(exp_dir / 'last_model.pth'),
        num_epochs - 1,
        val_metrics
    )

    # Final evaluation on test set
    print(f'\n{"="*50}')
    print(f'Final Evaluation - Strategy {strategy_id}: {strategy_name}')
    print(f'{"="*50}')

    # Load best model for evaluation
    best_checkpoint = exp_dir / 'best_model.pth'
    if best_checkpoint.exists():
        trainer.load_checkpoint(str(best_checkpoint))
        print('Loaded best model for evaluation')

    test_metrics = evaluate_model(model, test_loader, device, cam_source=model._cam_prob_key)

    print('\nTest Results:')
    print('-' * 40)
    for key, value in sorted(test_metrics.items()):
        print(f'{key}: {value:.4f}')

    # Save metrics
    results = {
        'strategy_id': strategy_id,
        'strategy_name': strategy_name,
        'history': history,
        'test_metrics': test_metrics,
        'best_val_accuracy': trainer.best_metric,
    }

    with open(exp_dir / 'results.json', 'w') as f:
        json.dump(results, f, indent=2)

    return results


def main():
    args = parse_args()

    # Load config
    config = load_config(args.config)

    # Override config with command line arguments
    if args.data_root:
        config['data']['data_root'] = args.data_root
    if args.backbone:
        config['model']['backbone_arch'] = args.backbone
    if args.batch_size:
        config['training']['batch_size'] = args.batch_size
    if args.epochs:
        config['training']['num_epochs'] = args.epochs
    if args.lr:
        config['training']['learning_rate'] = args.lr
    if args.device:
        config['experiment']['device'] = args.device
    if args.seed:
        config['experiment']['seed'] = args.seed
    if args.val_ratio is not None:
        config['data']['val_ratio'] = args.val_ratio
    if args.wandb:
        config['logging']['use_wandb'] = True
    if args.no_tensorboard:
        config['logging']['use_tensorboard'] = False

    # Experiment name
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    arch = config['model']['backbone_arch']
    if args.name:
        exp_name = f"{args.name}_{timestamp}"
    else:
        exp_name = f"strategy_comparison_{arch}_{timestamp}"

    config['experiment']['name'] = exp_name

    # Set seed
    set_seed(config['experiment']['seed'])

    # Device
    device = get_device(config['experiment'].get('device'))
    print(f'\nUsing device: {device}')

    # Create base output directory
    base_exp_dir = Path(config['checkpoint']['checkpoint_dir']) / exp_name
    base_exp_dir.mkdir(parents=True, exist_ok=True)

    # Save base config
    save_config(config, str(base_exp_dir / 'config.yaml'))
    print(f'Base experiment directory: {base_exp_dir}')

    # ============ Create Data Loaders ============
    print('\n' + '=' * 60)
    print('Loading Data')
    print('=' * 60)

    train_loader, val_loader, test_loader = create_dataloaders(
        data_root=config['data']['data_root'],
        batch_size=config['training']['batch_size'],
        image_size=tuple(config['data']['image_size']),
        num_workers=config['training']['num_workers'],
        pin_memory=config['training']['pin_memory'],
        dataset_type=config['data']['dataset_type'],
        category=config['data'].get('category', 'bottle'),
        val_ratio=config['data'].get('val_ratio', 0.0),
        seed=config['experiment']['seed'],
    )

    print(f'Train batches: {len(train_loader)}')
    print(f'Val batches: {len(val_loader)}')
    print(f'Test batches: {len(test_loader)}')

    # ============ Setup Logger ============
    logger = None
    if config['logging']['use_wandb']:
        try:
            import wandb
            wandb.init(
                project=config['logging']['wandb_project'],
                entity=config['logging'].get('wandb_entity'),
                name=exp_name,
                config=config,
            )
            logger = wandb
        except ImportError:
            print('W&B not installed, skipping...')

    # ============ Train Each Strategy ============
    strategies_to_train = sorted(args.strategies)
    all_results = {}

    print('\n' + '=' * 70)
    print(f'TRAINING {len(strategies_to_train)} STRATEGIES: {strategies_to_train}')
    print('=' * 70)

    for strategy_id in strategies_to_train:
        results = train_single_strategy(
            strategy_id=strategy_id,
            config=config,
            train_loader=train_loader,
            val_loader=val_loader,
            test_loader=test_loader,
            device=device,
            base_exp_dir=base_exp_dir,
            logger=logger,
        )
        all_results[strategy_id] = results

    # ============ Summary Comparison ============
    print('\n' + '=' * 70)
    print('STRATEGY COMPARISON SUMMARY')
    print('=' * 70)

    print(f'\n{"Strategy":<40} {"Val Acc":<12} {"Test Acc":<12} {"CAM-IoU":<12}')
    print('-' * 76)

    for strategy_id in strategies_to_train:
        results = all_results[strategy_id]
        strategy_name = STRATEGY_CONFIGS[strategy_id]['name']
        val_acc = results['best_val_accuracy']
        test_acc = results['test_metrics'].get('accuracy', 0)
        cam_iou = results['test_metrics'].get('cam_iou', 0)

        print(f'{strategy_id}. {strategy_name:<36} {val_acc:<12.4f} {test_acc:<12.4f} {cam_iou:<12.4f}')

    # Save comparison summary
    summary = {
        'experiment_name': exp_name,
        'strategies_trained': strategies_to_train,
        'results': {str(k): v for k, v in all_results.items()},
        'comparison': {
            str(sid): {
                'strategy_name': STRATEGY_CONFIGS[sid]['name'],
                'best_val_accuracy': all_results[sid]['best_val_accuracy'],
                'test_accuracy': all_results[sid]['test_metrics'].get('accuracy', 0),
                'test_cam_iou': all_results[sid]['test_metrics'].get('cam_iou', 0),
            }
            for sid in strategies_to_train
        }
    }

    with open(base_exp_dir / 'comparison_summary.json', 'w') as f:
        json.dump(summary, f, indent=2)

    print(f'\nAll training complete!')
    print(f'Results saved to: {base_exp_dir}')

    # Close logger
    if logger is not None and hasattr(logger, 'finish'):
        logger.finish()


if __name__ == '__main__':
    main()
