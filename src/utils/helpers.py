"""
Utility Functions for GAIN-MTL

General helper functions for:
- Configuration management
- Device setup
- Seeding
- Logging helpers
"""

import os
import random
from typing import Any, Dict, Optional, Union

import numpy as np
import torch
import yaml


def set_seed(seed: int = 42):
    """
    Set random seed for reproducibility.

    Args:
        seed: Random seed value
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    # For complete reproducibility (may impact performance)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_device(device: Optional[str] = None) -> torch.device:
    """
    Get appropriate device for computation.

    Args:
        device: Device string ('cuda', 'cpu', 'cuda:0', etc.)
                If None, auto-detect best available device

    Returns:
        torch.device object
    """
    if device is not None:
        return torch.device(device)

    if torch.cuda.is_available():
        return torch.device('cuda')
    elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        return torch.device('mps')
    else:
        return torch.device('cpu')


def count_parameters(model: torch.nn.Module, trainable_only: bool = True) -> int:
    """
    Count model parameters.

    Args:
        model: PyTorch model
        trainable_only: If True, count only trainable parameters

    Returns:
        Number of parameters
    """
    if trainable_only:
        return sum(p.numel() for p in model.parameters() if p.requires_grad)
    return sum(p.numel() for p in model.parameters())


def load_config(config_path: str) -> Dict[str, Any]:
    """
    Load configuration from YAML file.

    Args:
        config_path: Path to config file

    Returns:
        Configuration dictionary
    """
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def save_config(config: Dict[str, Any], save_path: str):
    """
    Save configuration to YAML file.

    Args:
        config: Configuration dictionary
        save_path: Path to save config
    """
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    with open(save_path, 'w') as f:
        yaml.dump(config, f, default_flow_style=False)


class AverageMeter:
    """
    Computes and stores the average and current value.

    Useful for tracking metrics during training.
    """

    def __init__(self, name: str = 'Metric'):
        self.name = name
        self.reset()

    def reset(self):
        """Reset all statistics."""
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0
        self.history = []

    def update(self, val: float, n: int = 1):
        """
        Update with new value.

        Args:
            val: New value
            n: Number of samples this value represents
        """
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count
        self.history.append(val)

    def __str__(self):
        return f'{self.name}: {self.avg:.4f}'


def format_time(seconds: float) -> str:
    """
    Format seconds into human readable string.

    Args:
        seconds: Time in seconds

    Returns:
        Formatted string (e.g., "1h 23m 45s")
    """
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)

    if hours > 0:
        return f'{hours}h {minutes}m {secs}s'
    elif minutes > 0:
        return f'{minutes}m {secs}s'
    else:
        return f'{secs}s'


def get_lr(optimizer: torch.optim.Optimizer) -> float:
    """Get current learning rate from optimizer."""
    for param_group in optimizer.param_groups:
        return param_group['lr']


def print_model_summary(model: torch.nn.Module, input_size: tuple = (1, 3, 512, 512)):
    """
    Print model summary with parameter counts.

    Args:
        model: PyTorch model
        input_size: Example input size
    """
    total_params = count_parameters(model, trainable_only=False)
    trainable_params = count_parameters(model, trainable_only=True)

    print('=' * 60)
    print(f'Model: {model.__class__.__name__}')
    print('=' * 60)
    print(f'Total Parameters: {total_params:,}')
    print(f'Trainable Parameters: {trainable_params:,}')
    print(f'Non-trainable Parameters: {total_params - trainable_params:,}')
    print(f'Model Size: {total_params * 4 / (1024 ** 2):.2f} MB (float32)')
    print('=' * 60)

    # Print layer-wise parameter counts
    print('\nLayer-wise Parameters:')
    print('-' * 60)
    for name, module in model.named_children():
        params = sum(p.numel() for p in module.parameters())
        print(f'{name}: {params:,}')
    print('-' * 60)
