#!/usr/bin/env python3
"""
Inference Speed Benchmark for GAIN-MTL

Compares full forward pass vs optimized inference (skip_unused=True) to
measure the speedup from skipping modules unused by the current strategy.

Usage:
    # Interactive: select checkpoint from available ones
    python benchmark_speed.py --data_root ./data

    # Direct checkpoint path
    python benchmark_speed.py --checkpoint checkpoints/strategy_3/best_model.pth --data_root ./data

    # Work directory mode
    python benchmark_speed.py --work_dir checkpoints/strategy_comparison_v1/strategy_3_cls_attention_mining

    # Options
    python benchmark_speed.py --checkpoint model.pth --data_root ./data --num_images 100 --device cuda --warmup 10
"""

import argparse
import glob
import os
import sys
import time
from pathlib import Path

import torch
import numpy as np

sys.path.insert(0, str(Path(__file__).parent))

from src.models import GAINMTLModel
from src.data import create_dataloaders, get_transforms, DefectDataset
from src.utils import get_device, load_config


def find_checkpoints(root_dir: str = '.') -> list:
    """Recursively find all .pth checkpoint files."""
    patterns = [
        os.path.join(root_dir, '**', '*.pth'),
    ]
    found = []
    for pattern in patterns:
        found.extend(glob.glob(pattern, recursive=True))
    # Sort by modification time (newest first)
    found.sort(key=os.path.getmtime, reverse=True)
    return found


def select_checkpoint_interactive() -> str:
    """Let user select a checkpoint interactively."""
    checkpoints = find_checkpoints('.')
    if not checkpoints:
        print("No .pth checkpoint files found in current directory tree.")
        sys.exit(1)

    print('\n' + '=' * 70)
    print('Available Checkpoints')
    print('=' * 70)
    for i, ckpt in enumerate(checkpoints):
        size_mb = os.path.getsize(ckpt) / (1024 * 1024)
        mtime = time.strftime('%Y-%m-%d %H:%M', time.localtime(os.path.getmtime(ckpt)))
        print(f'  [{i}] {ckpt}')
        print(f'       Size: {size_mb:.1f} MB | Modified: {mtime}')
    print('=' * 70)

    while True:
        try:
            idx = int(input(f'\nSelect checkpoint [0-{len(checkpoints)-1}]: '))
            if 0 <= idx < len(checkpoints):
                return checkpoints[idx]
            print(f'Invalid index. Please enter 0-{len(checkpoints)-1}.')
        except ValueError:
            print('Please enter a valid number.')
        except (KeyboardInterrupt, EOFError):
            print('\nCancelled.')
            sys.exit(0)


def parse_args():
    parser = argparse.ArgumentParser(description='Benchmark GAIN-MTL inference speed')

    parser.add_argument('--work_dir', type=str, default=None,
                        help='Work directory (derives checkpoint and config)')
    parser.add_argument('--checkpoint', type=str, default=None,
                        help='Path to model checkpoint')
    parser.add_argument('--config', type=str, default=None,
                        help='Path to config file')
    parser.add_argument('--data_root', type=str, default='./data',
                        help='Data root directory')
    parser.add_argument('--num_images', type=int, default=50,
                        help='Number of test images to benchmark (default: 50)')
    parser.add_argument('--warmup', type=int, default=5,
                        help='Number of warmup iterations before timing (default: 5)')
    parser.add_argument('--device', type=str, default=None,
                        help='Device to use (default: auto-detect)')
    parser.add_argument('--strategy', type=int, default=None, choices=[1, 2, 3, 4, 5, 6, 7, 8],
                        help='Training strategy (auto-detected from checkpoint if not specified)')
    parser.add_argument('--batch_size', type=int, default=1,
                        help='Batch size for inference (default: 1 for per-image timing)')

    args = parser.parse_args()

    # Derive paths from work_dir
    if args.work_dir:
        work_dir = Path(args.work_dir)
        if args.checkpoint is None:
            args.checkpoint = str(work_dir / 'best_model.pth')
        if args.config is None:
            args.config = str(work_dir.parent / 'config.yaml')

    return args


def run_benchmark(model, images_list, device, num_images, warmup, skip_unused=None):
    """Run inference benchmark with specified skip_unused mode.

    Args:
        model: Model in eval mode.
        images_list: Pre-loaded image tensors.
        device: torch device.
        num_images: Number of images to benchmark.
        warmup: Number of warmup iterations.
        skip_unused: Passed to model forward(). None=auto (optimized in eval).

    Returns:
        numpy array of latencies in milliseconds.
    """
    # Warmup
    warmup_img = images_list[0].to(device)
    with torch.no_grad():
        for _ in range(warmup):
            _ = model(warmup_img, skip_unused=skip_unused)
    if device.type == 'cuda':
        torch.cuda.synchronize()

    # Benchmark
    latencies = []
    with torch.no_grad():
        for i in range(num_images):
            img = images_list[i].to(device)

            if device.type == 'cuda':
                torch.cuda.synchronize()

            start = time.perf_counter()
            _ = model(img, skip_unused=skip_unused)

            if device.type == 'cuda':
                torch.cuda.synchronize()

            end = time.perf_counter()
            latencies.append((end - start) * 1000)  # ms

    return np.array(latencies)


def print_latency_stats(latencies, label):
    """Print latency statistics for a benchmark run."""
    print(f'  [{label}]')
    print(f'    Mean latency   : {latencies.mean():.2f} ms')
    print(f'    Std latency    : {latencies.std():.2f} ms')
    print(f'    Median latency : {np.median(latencies):.2f} ms')
    print(f'    Min latency    : {latencies.min():.2f} ms')
    print(f'    Max latency    : {latencies.max():.2f} ms')
    print(f'    P95 latency    : {np.percentile(latencies, 95):.2f} ms')
    print(f'    P99 latency    : {np.percentile(latencies, 99):.2f} ms')
    print(f'    Throughput     : {1000.0 / latencies.mean():.1f} FPS')


# Modules skipped per strategy group (for display)
SKIP_INFO = {
    'S1-S2': 'FPN, AttentionModule, MiningHead, FeatureAdapter, AttendedClsHead, LocalizationHead',
    'S3-S5,S7': 'FPN, ClassificationHead, CAM, LocalizationHead',
    'S6,S8': 'FPN, ClassificationHead, CAM, LocalizationHead',
}


def get_strategy_group(strategy):
    if strategy <= 2:
        return 'S1-S2'
    elif strategy in (6, 8):
        return 'S6,S8'
    else:
        return 'S3-S5,S7'


def main():
    args = parse_args()

    # ============ Checkpoint Selection ============
    if args.checkpoint is None:
        args.checkpoint = select_checkpoint_interactive()

    checkpoint_path = Path(args.checkpoint)
    if not checkpoint_path.exists():
        print(f'Checkpoint not found: {args.checkpoint}')
        sys.exit(1)

    print(f'\nCheckpoint: {args.checkpoint}')

    # ============ Device ============
    device = get_device(args.device)
    print(f'Device: {device}')

    # ============ Config ============
    checkpoint_dir = checkpoint_path.parent
    if args.config and Path(args.config).exists():
        config = load_config(args.config)
    elif (checkpoint_dir / 'config.yaml').exists():
        config = load_config(str(checkpoint_dir / 'config.yaml'))
    elif (checkpoint_dir.parent / 'config.yaml').exists():
        config = load_config(str(checkpoint_dir.parent / 'config.yaml'))
    else:
        config = load_config('configs/default.yaml')
        print('Using default config')

    if args.data_root:
        config['data']['data_root'] = args.data_root

    image_size = tuple(config['data']['image_size'])

    # ============ Load Model ============
    print('\nLoading model...')
    model = GAINMTLModel(
        backbone_arch=config['model']['backbone_arch'],
        num_classes=config['model']['num_classes'],
        pretrained=False,
        fpn_channels=config['model']['fpn_channels'],
        attention_channels=config['model']['attention_channels'],
        use_counterfactual=config['model']['use_counterfactual'],
        freeze_backbone_stages=config['model']['freeze_backbone_stages'],
        out_indices=tuple(config['model'].get('out_indices', [3, 4, 5, 6])),
    )

    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
        epoch = checkpoint.get('epoch', 'unknown')
        print(f'Loaded checkpoint from epoch {epoch}')

        # Auto-detect strategy from checkpoint
        if args.strategy is None:
            extra = checkpoint.get('extra_data', {})
            args.strategy = extra.get('strategy_id', 3)
    else:
        model.load_state_dict(checkpoint)

    if args.strategy is None:
        args.strategy = 3

    model = model.to(device)
    model.set_strategy(args.strategy)
    model.eval()

    param_count = sum(p.numel() for p in model.parameters())
    print(f'Model parameters: {param_count:,}')
    print(f'Strategy: {args.strategy}')

    # ============ Load Test Data ============
    print('\nLoading test data...')
    _, _, test_loader = create_dataloaders(
        data_root=config['data']['data_root'],
        batch_size=args.batch_size,
        image_size=image_size,
        num_workers=0,
        dataset_type=config['data']['dataset_type'],
        category=config['data'].get('category', 'bottle'),
    )

    # Collect first N test images
    num_images = args.num_images
    images_list = []
    for batch in test_loader:
        imgs = batch['image']
        for i in range(imgs.size(0)):
            images_list.append(imgs[i].unsqueeze(0))
            if len(images_list) >= num_images:
                break
        if len(images_list) >= num_images:
            break

    actual_count = len(images_list)
    if actual_count == 0:
        print('No test images found!')
        sys.exit(1)

    if actual_count < num_images:
        print(f'Note: Only {actual_count} test images available (requested {num_images})')
    num_images = actual_count

    print(f'Benchmarking on {num_images} images (batch_size={args.batch_size})')
    print(f'Image size: {image_size}')

    # ============ Benchmark: Full Forward ============
    print(f'\n--- Full forward (skip_unused=False) ---')
    print(f'Warmup ({args.warmup} iterations)...')
    latencies_full = run_benchmark(
        model, images_list, device, num_images, args.warmup, skip_unused=False,
    )

    # ============ Benchmark: Optimized Inference ============
    print(f'\n--- Optimized inference (skip_unused=True) ---')
    print(f'Warmup ({args.warmup} iterations)...')
    latencies_opt = run_benchmark(
        model, images_list, device, num_images, args.warmup, skip_unused=True,
    )

    # ============ Results ============
    strategy_group = get_strategy_group(args.strategy)
    skipped_modules = SKIP_INFO[strategy_group]

    print('\n' + '=' * 70)
    print(f'  Inference Speed Benchmark Results')
    print('=' * 70)
    print(f'  Checkpoint : {args.checkpoint}')
    print(f'  Device     : {device}')
    print(f'  Strategy   : S{args.strategy} ({strategy_group})')
    print(f'  Image size : {image_size[0]}x{image_size[1]}')
    print(f'  Num images : {num_images}')
    print(f'  Skipped    : {skipped_modules}')
    print('-' * 70)

    print_latency_stats(latencies_full, 'Full forward')
    print()
    print_latency_stats(latencies_opt, 'Optimized')

    # ============ Speedup Summary ============
    speedup = latencies_full.mean() / latencies_opt.mean()
    saved_ms = latencies_full.mean() - latencies_opt.mean()
    fps_full = 1000.0 / latencies_full.mean()
    fps_opt = 1000.0 / latencies_opt.mean()

    print()
    print('-' * 70)
    print(f'  Summary')
    print('-' * 70)
    print(f'  Full    : {latencies_full.mean():.2f} ms  ({fps_full:.1f} FPS)')
    print(f'  Optimized : {latencies_opt.mean():.2f} ms  ({fps_opt:.1f} FPS)')
    print(f'  Speedup   : {speedup:.2f}x  ({saved_ms:+.2f} ms per image)')
    print(f'  FPS gain  : +{fps_opt - fps_full:.1f} FPS')
    print('=' * 70)


if __name__ == '__main__':
    main()
