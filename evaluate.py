#!/usr/bin/env python3
"""
Evaluation Script for GAIN-MTL

Usage:
    python evaluate.py --checkpoint checkpoints/best_model.pth --data_root ./data
    python evaluate.py --checkpoint checkpoints/best_model.pth --visualize
    python evaluate.py --checkpoint checkpoints/best_model.pth --export_results results.json
"""

import argparse
import json
import os
import sys
from pathlib import Path

import torch
import numpy as np
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent))

from src.models import GAINMTLModel
from src.data import create_dataloaders, get_transforms, DefectDataset
from src.evaluation import (
    evaluate_model,
    compute_classification_metrics,
    compute_cam_metrics,
    compute_localization_metrics,
    DefectExplainer,
)
from src.utils import get_device, load_config


def parse_args():
    parser = argparse.ArgumentParser(description='Evaluate GAIN-MTL model')

    parser.add_argument('--checkpoint', type=str, required=True,
                        help='Path to model checkpoint')
    parser.add_argument('--config', type=str, default=None,
                        help='Path to config file (auto-detect from checkpoint dir)')
    parser.add_argument('--data_root', type=str, default='./data',
                        help='Data root directory')
    parser.add_argument('--split', type=str, default='test',
                        choices=['train', 'val', 'test'],
                        help='Data split to evaluate')
    parser.add_argument('--batch_size', type=int, default=32,
                        help='Batch size')
    parser.add_argument('--device', type=str, default='cuda',
                        help='Device to use')

    # Output options
    parser.add_argument('--export_results', type=str, default=None,
                        help='Export results to JSON file')
    parser.add_argument('--visualize', action='store_true',
                        help='Generate visualizations')
    parser.add_argument('--vis_dir', type=str, default='./visualizations',
                        help='Directory for visualizations')
    parser.add_argument('--num_vis', type=int, default=20,
                        help='Number of samples to visualize')

    # Analysis options
    parser.add_argument('--per_class', action='store_true',
                        help='Compute per-class metrics')
    parser.add_argument('--error_analysis', action='store_true',
                        help='Perform error analysis')
    parser.add_argument('--threshold', type=float, default=0.5,
                        help='Classification threshold')

    return parser.parse_args()


def main():
    args = parse_args()

    # Device
    device = get_device(args.device)
    print(f'Using device: {device}')

    # Load config
    checkpoint_dir = Path(args.checkpoint).parent
    if args.config:
        config = load_config(args.config)
    elif (checkpoint_dir / 'config.yaml').exists():
        config = load_config(str(checkpoint_dir / 'config.yaml'))
        print(f'Loaded config from: {checkpoint_dir / "config.yaml"}')
    else:
        # Use default config
        config = load_config('configs/default.yaml')
        print('Using default config')

    # Override data root
    if args.data_root:
        config['data']['data_root'] = args.data_root

    # ============ Load Model ============
    print('\n' + '=' * 60)
    print('Loading Model')
    print('=' * 60)

    model = GAINMTLModel(
        backbone_name=config['model']['backbone'],
        num_classes=config['model']['num_classes'],
        pretrained=False,  # Don't need pretrained when loading checkpoint
        fpn_channels=config['model']['fpn_channels'],
        attention_channels=config['model']['attention_channels'],
        use_counterfactual=config['model']['use_counterfactual'],
    )

    # Load checkpoint
    checkpoint = torch.load(args.checkpoint, map_location=device)
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
        print(f'Loaded checkpoint from epoch {checkpoint.get("epoch", "unknown")}')
        if 'metrics' in checkpoint:
            print(f'Checkpoint metrics: {checkpoint["metrics"]}')
    else:
        model.load_state_dict(checkpoint)

    model = model.to(device)
    model.eval()

    # ============ Load Data ============
    print('\n' + '=' * 60)
    print('Loading Data')
    print('=' * 60)

    _, val_loader, test_loader = create_dataloaders(
        data_root=config['data']['data_root'],
        batch_size=args.batch_size,
        image_size=tuple(config['data']['image_size']),
        num_workers=4,
        dataset_type=config['data']['dataset_type'],
        category=config['data'].get('category', 'bottle'),
    )

    if args.split == 'val':
        dataloader = val_loader
    else:
        dataloader = test_loader

    print(f'Evaluating on {args.split} set: {len(dataloader)} batches')

    # ============ Evaluate ============
    print('\n' + '=' * 60)
    print('Running Evaluation')
    print('=' * 60)

    metrics = evaluate_model(model, dataloader, device, args.threshold)

    print('\n' + '=' * 60)
    print('Results')
    print('=' * 60)

    # Classification metrics
    print('\nClassification Metrics:')
    print('-' * 40)
    cls_metrics = ['accuracy', 'precision', 'recall', 'f1', 'auc_roc']
    for m in cls_metrics:
        if m in metrics:
            print(f'  {m}: {metrics[m]:.4f}')

    # CAM metrics
    print('\nCAM Quality Metrics:')
    print('-' * 40)
    cam_metrics = ['cam_iou', 'point_game', 'energy_inside']
    for m in cam_metrics:
        if m in metrics:
            print(f'  {m}: {metrics[m]:.4f}')

    # Localization metrics
    print('\nLocalization Metrics:')
    print('-' * 40)
    loc_metrics = ['loc_iou', 'loc_dice', 'loc_pixel_accuracy']
    for m in loc_metrics:
        if m in metrics:
            print(f'  {m}: {metrics[m]:.4f}')

    # ============ Error Analysis ============
    if args.error_analysis:
        print('\n' + '=' * 60)
        print('Error Analysis')
        print('=' * 60)

        perform_error_analysis(model, dataloader, device, args.threshold)

    # ============ Visualizations ============
    if args.visualize:
        print('\n' + '=' * 60)
        print('Generating Visualizations')
        print('=' * 60)

        vis_dir = Path(args.vis_dir)
        vis_dir.mkdir(parents=True, exist_ok=True)

        generate_visualizations(
            model=model,
            dataloader=dataloader,
            device=device,
            output_dir=vis_dir,
            num_samples=args.num_vis,
        )

        print(f'Visualizations saved to: {vis_dir}')

    # ============ Export Results ============
    if args.export_results:
        with open(args.export_results, 'w') as f:
            json.dump(metrics, f, indent=2)
        print(f'\nResults exported to: {args.export_results}')

    print('\nEvaluation complete!')


def perform_error_analysis(model, dataloader, device, threshold=0.5):
    """Analyze model errors."""
    model.eval()

    false_positives = []
    false_negatives = []
    low_cam_iou = []

    with torch.no_grad():
        for batch in tqdm(dataloader, desc='Analyzing errors'):
            images = batch['image'].to(device)
            labels = batch['label'].to(device)
            defect_masks = batch['defect_mask'].to(device)
            has_defect = batch['has_defect'].to(device)
            image_paths = batch.get('image_path', [None] * len(images))

            outputs = model(images)
            preds = outputs['cls_logits'].argmax(dim=1)

            for i in range(len(images)):
                pred = preds[i].item()
                label = labels[i].item()
                path = image_paths[i] if image_paths else f'sample_{i}'

                if pred == 1 and label == 0:
                    false_positives.append({
                        'path': path,
                        'confidence': torch.softmax(outputs['cls_logits'][i], dim=0)[1].item()
                    })
                elif pred == 0 and label == 1:
                    false_negatives.append({
                        'path': path,
                        'confidence': torch.softmax(outputs['cls_logits'][i], dim=0)[0].item()
                    })

                # Check CAM quality for defective samples
                if has_defect[i]:
                    attn = outputs['attention_map'][i, 0]
                    mask = defect_masks[i, 0]

                    # Resize attention
                    attn_resized = torch.nn.functional.interpolate(
                        attn.unsqueeze(0).unsqueeze(0),
                        size=mask.shape,
                        mode='bilinear'
                    )[0, 0]

                    # Compute IoU
                    attn_bin = (attn_resized > threshold).float()
                    mask_bin = (mask > threshold).float()
                    intersection = (attn_bin * mask_bin).sum()
                    union = attn_bin.sum() + mask_bin.sum() - intersection
                    iou = (intersection / (union + 1e-6)).item()

                    if iou < 0.3:
                        low_cam_iou.append({
                            'path': path,
                            'cam_iou': iou,
                            'correct_pred': pred == label
                        })

    print(f'\nFalse Positives: {len(false_positives)}')
    if false_positives:
        print('  Top 5 by confidence:')
        for item in sorted(false_positives, key=lambda x: x['confidence'], reverse=True)[:5]:
            print(f"    {item['path']}: {item['confidence']:.4f}")

    print(f'\nFalse Negatives: {len(false_negatives)}')
    if false_negatives:
        print('  Top 5 by confidence:')
        for item in sorted(false_negatives, key=lambda x: x['confidence'], reverse=True)[:5]:
            print(f"    {item['path']}: {item['confidence']:.4f}")

    print(f'\nLow CAM-IoU samples (< 0.3): {len(low_cam_iou)}')
    correct_but_wrong_cam = sum(1 for x in low_cam_iou if x['correct_pred'])
    print(f'  Correct prediction but wrong attention: {correct_but_wrong_cam}')


def generate_visualizations(model, dataloader, device, output_dir, num_samples=20):
    """Generate visualization for random samples."""
    import matplotlib
    matplotlib.use('Agg')

    explainer = DefectExplainer(model, device)

    # Collect samples
    samples = []
    for batch in dataloader:
        for i in range(len(batch['image'])):
            samples.append({
                'image': batch['image'][i],
                'defect_mask': batch['defect_mask'][i],
                'label': batch['label'][i].item(),
                'path': batch.get('image_path', [None])[i] if 'image_path' in batch else None
            })
            if len(samples) >= num_samples:
                break
        if len(samples) >= num_samples:
            break

    # Generate visualizations
    for i, sample in enumerate(tqdm(samples, desc='Generating visualizations')):
        fig = explainer.visualize(
            image=sample['image'],
            defect_mask=sample['defect_mask'].numpy() if sample['defect_mask'] is not None else None,
            save_path=str(output_dir / f'sample_{i:03d}.png'),
        )
        import matplotlib.pyplot as plt
        plt.close(fig)


if __name__ == '__main__':
    main()
