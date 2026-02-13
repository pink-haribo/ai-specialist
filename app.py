#!/usr/bin/env python3
"""
Gradio Inference App for GAIN-MTL

Provides a web-based test/inference page with:
- Experiment and strategy selection
- Epoch dropdown that only shows epochs with checkpoints
- Single-image inference with visualization

Usage:
    python app.py
    python app.py --checkpoint_root ./checkpoints --port 7860
"""

import argparse
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import gradio as gr
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent))

from src.models import GAINMTLModel
from src.evaluation import DefectExplainer
from src.utils import get_device, load_config, get_available_checkpoints


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def list_experiments(checkpoint_root: str) -> List[str]:
    """Return experiment directory names under *checkpoint_root*."""
    root = Path(checkpoint_root)
    if not root.is_dir():
        return []
    return sorted(
        d.name for d in root.iterdir()
        if d.is_dir() and not d.name.startswith('.')
    )


def list_strategies(checkpoint_root: str, experiment: str) -> List[str]:
    """Return strategy directory names inside an experiment."""
    exp_dir = Path(checkpoint_root) / experiment
    if not exp_dir.is_dir():
        return []
    return sorted(
        d.name for d in exp_dir.iterdir()
        if d.is_dir() and d.name.startswith('strategy_')
    )


def _parse_strategy_id(strategy_name: str) -> int:
    """Extract strategy id from directory name like 'strategy_3_cls_attention_mining'."""
    try:
        return int(strategy_name.split('_')[1])
    except (IndexError, ValueError):
        return 3


# ---------------------------------------------------------------------------
# State
# ---------------------------------------------------------------------------

_current_model: Optional[GAINMTLModel] = None
_current_explainer: Optional[DefectExplainer] = None
_current_ckpt_path: Optional[str] = None
_device = get_device('cpu')


def _load_model(
    checkpoint_root: str,
    experiment: str,
    strategy: str,
    checkpoint_label: str,
) -> str:
    """Load model from the selected checkpoint. Returns a status message."""
    global _current_model, _current_explainer, _current_ckpt_path, _device

    _device = get_device(None)

    # Resolve checkpoint path
    work_dir = Path(checkpoint_root) / experiment / strategy
    checkpoints = get_available_checkpoints(str(work_dir))
    ckpt_path = None
    for c in checkpoints:
        if c['label'] == checkpoint_label:
            ckpt_path = c['path']
            break
    if ckpt_path is None:
        return f"Checkpoint '{checkpoint_label}' not found."

    if ckpt_path == _current_ckpt_path:
        return f"Already loaded: {checkpoint_label}"

    # Load config
    config_path = work_dir.parent / 'config.yaml'
    if not config_path.exists():
        return f"Config not found: {config_path}"
    config = load_config(str(config_path))

    # Build model
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

    checkpoint = torch.load(ckpt_path, map_location=_device, weights_only=False)
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint)

    strategy_id = _parse_strategy_id(strategy)
    model = model.to(_device)
    model.set_strategy(strategy_id)
    model.eval()

    _current_model = model
    _current_ckpt_path = ckpt_path
    _current_explainer = DefectExplainer(model, _device, strategy=strategy_id)

    epoch_info = checkpoint.get('epoch', '?')
    return f"Loaded: {strategy} / {checkpoint_label} (epoch {epoch_info})"


# ---------------------------------------------------------------------------
# Gradio callbacks
# ---------------------------------------------------------------------------

def on_experiment_change(checkpoint_root: str, experiment: str):
    strategies = list_strategies(checkpoint_root, experiment)
    first = strategies[0] if strategies else None
    return gr.update(choices=strategies, value=first)


def on_strategy_change(checkpoint_root: str, experiment: str, strategy: str):
    if not strategy:
        return gr.update(choices=[], value=None)
    work_dir = Path(checkpoint_root) / experiment / strategy
    checkpoints = get_available_checkpoints(str(work_dir))
    labels = [c['label'] for c in checkpoints]
    first = labels[0] if labels else None
    return gr.update(choices=labels, value=first)


def run_inference(
    checkpoint_root: str,
    experiment: str,
    strategy: str,
    checkpoint_label: str,
    image: Optional[np.ndarray],
):
    if image is None:
        return None, "Please upload an image."

    status = _load_model(checkpoint_root, experiment, strategy, checkpoint_label)
    if _current_explainer is None:
        return None, status

    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    fig = _current_explainer.visualize(image=image)
    fig.canvas.draw()
    w, h = fig.canvas.get_width_height()
    buf = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8).reshape(h, w, 3)
    plt.close(fig)

    return buf, status


# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------

def create_app(checkpoint_root: str = './checkpoints') -> gr.Blocks:
    experiments = list_experiments(checkpoint_root)
    initial_exp = experiments[0] if experiments else ''
    initial_strategies = list_strategies(checkpoint_root, initial_exp) if initial_exp else []
    initial_strategy = initial_strategies[0] if initial_strategies else ''

    if initial_exp and initial_strategy:
        work_dir = Path(checkpoint_root) / initial_exp / initial_strategy
        initial_ckpts = get_available_checkpoints(str(work_dir))
        initial_labels = [c['label'] for c in initial_ckpts]
    else:
        initial_labels = []

    with gr.Blocks(title='GAIN-MTL Inference') as app:
        gr.Markdown('# GAIN-MTL Test / Inference')

        ckpt_root_state = gr.State(checkpoint_root)

        with gr.Row():
            exp_dd = gr.Dropdown(
                choices=experiments,
                value=initial_exp,
                label='Experiment',
            )
            strategy_dd = gr.Dropdown(
                choices=initial_strategies,
                value=initial_strategy,
                label='Strategy',
            )
            epoch_dd = gr.Dropdown(
                choices=initial_labels,
                value=initial_labels[0] if initial_labels else None,
                label='Checkpoint (epoch)',
            )

        with gr.Row():
            input_image = gr.Image(label='Input Image', type='numpy')
            output_image = gr.Image(label='Result')

        status_box = gr.Textbox(label='Status', interactive=False)

        run_btn = gr.Button('Run Inference', variant='primary')

        # --- wiring ---
        exp_dd.change(
            fn=on_experiment_change,
            inputs=[ckpt_root_state, exp_dd],
            outputs=strategy_dd,
        )
        strategy_dd.change(
            fn=on_strategy_change,
            inputs=[ckpt_root_state, exp_dd, strategy_dd],
            outputs=epoch_dd,
        )
        run_btn.click(
            fn=run_inference,
            inputs=[ckpt_root_state, exp_dd, strategy_dd, epoch_dd, input_image],
            outputs=[output_image, status_box],
        )

    return app


def main():
    parser = argparse.ArgumentParser(description='GAIN-MTL Inference App')
    parser.add_argument('--checkpoint_root', type=str, default='./checkpoints',
                        help='Root directory containing experiment checkpoints')
    parser.add_argument('--port', type=int, default=7860, help='Server port')
    parser.add_argument('--share', action='store_true', help='Create public link')
    args = parser.parse_args()

    app = create_app(checkpoint_root=args.checkpoint_root)
    app.launch(server_port=args.port, share=args.share)


if __name__ == '__main__':
    main()
