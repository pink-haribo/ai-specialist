#!/usr/bin/env python3
"""
Training Script for GAIN-MTL

Usage:
    python train.py --config configs/default.yaml
    python train.py --config configs/default.yaml --data_root /path/to/data
    python train.py --config configs/default.yaml --backbone m  # EfficientNetV2-M

Backbone options (mmpretrain):
    s   - EfficientNetV2-S (Small)
    m   - EfficientNetV2-M (Medium)
    l   - EfficientNetV2-L (Large)
    xl  - EfficientNetV2-XL (Extra Large)

For multi-GPU training:
    torchrun --nproc_per_node=4 train.py --config configs/default.yaml
"""

import argparse
import os
import sys
from datetime import datetime
from pathlib import Path

import torch
import torch.nn as nn
from torch.optim import Adam, AdamW, SGD
from torch.optim.lr_scheduler import CosineAnnealingLR, StepLR, ReduceLROnPlateau

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

from src.models import GAINMTLModel
from src.losses import GAINMTLLoss
from src.training import MultiStageTrainer, TrainingConfig
from src.data import create_dataloaders
from src.evaluation import evaluate_model
from src.utils import set_seed, get_device, load_config, save_config, print_model_summary


def parse_args():
    parser = argparse.ArgumentParser(description='Train GAIN-MTL model')

    # Config
    parser.add_argument('--config', type=str, default='configs/default.yaml',
                        help='Path to config file')

    # Override config options
    parser.add_argument('--data_root', type=str, default=None,
                        help='Override data root directory')
    parser.add_argument('--backbone', type=str, default=None,
                        choices=['s', 'm', 'l', 'xl'],
                        help='Override backbone arch (s/m/l/xl for EfficientNetV2)')
    parser.add_argument('--batch_size', type=int, default=None,
                        help='Override batch size')
    parser.add_argument('--epochs', type=int, default=None,
                        help='Override number of epochs')
    parser.add_argument('--lr', type=float, default=None,
                        help='Override learning rate')
    parser.add_argument('--device', type=str, default=None,
                        help='Override device')

    # Experiment
    parser.add_argument('--name', type=str, default=None,
                        help='Experiment name')
    parser.add_argument('--resume', type=str, default=None,
                        help='Resume from checkpoint')
    parser.add_argument('--seed', type=int, default=None,
                        help='Random seed')

    # Logging
    parser.add_argument('--wandb', action='store_true',
                        help='Enable W&B logging')
    parser.add_argument('--no_tensorboard', action='store_true',
                        help='Disable TensorBoard logging')

    return parser.parse_args()


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
    if args.wandb:
        config['logging']['use_wandb'] = True
    if args.no_tensorboard:
        config['logging']['use_tensorboard'] = False

    # Experiment name
    if args.name:
        config['experiment']['name'] = args.name
    else:
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        arch = config['model']['backbone_arch']
        config['experiment']['name'] = f"efficientnetv2_{arch}_{timestamp}"

    # Set seed
    set_seed(config['experiment']['seed'])

    # Device
    device = get_device(config['experiment'].get('device'))
    print(f'\nUsing device: {device}')

    # Create output directories
    exp_dir = Path(config['checkpoint']['checkpoint_dir']) / config['experiment']['name']
    exp_dir.mkdir(parents=True, exist_ok=True)

    # Save config
    save_config(config, str(exp_dir / 'config.yaml'))
    print(f'Experiment directory: {exp_dir}')

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
    )

    print(f'Train batches: {len(train_loader)}')
    print(f'Val batches: {len(val_loader)}')
    print(f'Test batches: {len(test_loader)}')

    # ============ Create Model ============
    print('\n' + '=' * 60)
    print('Creating Model')
    print('=' * 60)

    model = GAINMTLModel(
        backbone_arch=config['model']['backbone_arch'],
        num_classes=config['model']['num_classes'],
        pretrained=config['model']['pretrained'],
        fpn_channels=config['model']['fpn_channels'],
        attention_channels=config['model']['attention_channels'],
        use_counterfactual=config['model']['use_counterfactual'],
        freeze_backbone_stages=config['model']['freeze_backbone_stages'],
        out_indices=tuple(config['model'].get('out_indices', [1, 2, 3, 4])),
    )

    model = model.to(device)
    print_model_summary(model)

    # ============ Create Loss Function ============
    criterion = GAINMTLLoss(
        lambda_cls=config['loss']['lambda_cls'],
        lambda_am=config['loss']['lambda_am'],
        lambda_loc=config['loss']['lambda_loc'],
        lambda_guide=config['loss']['lambda_guide'],
        lambda_cf=config['loss']['lambda_cf'],
        lambda_consist=config['loss']['lambda_consist'],
        focal_gamma=config['loss']['focal_gamma'],
        focal_alpha=config['loss']['focal_alpha'],
    )

    # ============ Create Optimizer ============
    optimizer_name = config['training'].get('optimizer', 'adamw').lower()
    lr = config['training']['learning_rate']
    weight_decay = config['training']['weight_decay']

    if optimizer_name == 'adam':
        optimizer = Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    elif optimizer_name == 'adamw':
        optimizer = AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    elif optimizer_name == 'sgd':
        optimizer = SGD(model.parameters(), lr=lr, weight_decay=weight_decay, momentum=0.9)
    else:
        raise ValueError(f'Unknown optimizer: {optimizer_name}')

    # ============ Create Scheduler ============
    scheduler_type = config['training'].get('scheduler_type', 'cosine')
    num_epochs = config['training']['num_epochs']

    if scheduler_type == 'cosine':
        scheduler = CosineAnnealingLR(
            optimizer,
            T_max=num_epochs,
            eta_min=config['training'].get('min_lr', 1e-6)
        )
    elif scheduler_type == 'step':
        scheduler = StepLR(optimizer, step_size=30, gamma=0.1)
    elif scheduler_type == 'plateau':
        scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)
    else:
        scheduler = None

    # ============ Create Trainer ============
    training_config = TrainingConfig(
        num_epochs=num_epochs,
        batch_size=config['training']['batch_size'],
        learning_rate=lr,
        weight_decay=weight_decay,
        stage1_ratio=config['training']['stage1_ratio'],
        stage2_ratio=config['training']['stage2_ratio'],
        stage3_ratio=config['training']['stage3_ratio'],
        stage4_ratio=config['training']['stage4_ratio'],
        gradient_accumulation_steps=config['training']['gradient_accumulation_steps'],
        max_grad_norm=config['training']['max_grad_norm'],
        use_amp=config['training']['use_amp'],
        log_interval=config['logging']['log_interval'],
        eval_interval=config['evaluation']['eval_interval'],
        save_interval=config['checkpoint']['save_interval'],
        checkpoint_dir=str(exp_dir),
        device=str(device),
    )

    # Setup logger
    logger = None
    if config['logging']['use_wandb']:
        try:
            import wandb
            wandb.init(
                project=config['logging']['wandb_project'],
                entity=config['logging'].get('wandb_entity'),
                name=config['experiment']['name'],
                config=config,
            )
            logger = wandb
        except ImportError:
            print('W&B not installed, skipping...')

    trainer = MultiStageTrainer(
        model=model,
        criterion=criterion,
        optimizer=optimizer,
        scheduler=scheduler,
        config=training_config,
        logger=logger,
    )

    # ============ Resume from Checkpoint ============
    start_epoch = 0
    if args.resume:
        start_epoch = trainer.load_checkpoint(args.resume)
        print(f'Resumed from epoch {start_epoch}')

    # ============ Train ============
    print('\n' + '=' * 60)
    print('Starting Training')
    print('=' * 60)

    history = trainer.train(
        train_loader=train_loader,
        val_loader=val_loader,
        num_epochs=num_epochs,
    )

    # ============ Final Evaluation ============
    print('\n' + '=' * 60)
    print('Final Evaluation on Test Set')
    print('=' * 60)

    # Load best model
    best_checkpoint = exp_dir / 'best_model.pth'
    if best_checkpoint.exists():
        trainer.load_checkpoint(str(best_checkpoint))
        print('Loaded best model for evaluation')

    test_metrics = evaluate_model(model, test_loader, device)

    print('\nTest Results:')
    print('-' * 40)
    for key, value in sorted(test_metrics.items()):
        print(f'{key}: {value:.4f}')

    # Save final metrics
    import json
    with open(exp_dir / 'test_metrics.json', 'w') as f:
        json.dump(test_metrics, f, indent=2)

    print(f'\nTraining complete!')
    print(f'Results saved to: {exp_dir}')

    # Close logger
    if logger is not None and hasattr(logger, 'finish'):
        logger.finish()


if __name__ == '__main__':
    main()
