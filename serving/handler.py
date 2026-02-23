"""
TorchServe Custom Handler for GAIN-MTL Defect Detection Model

Handles:
- Image preprocessing (resize, normalize with ImageNet stats)
- Inference with GAINMTLModel (classification + attention map)
- Postprocessing (class prediction, confidence, CAM heatmap)

Usage:
    # 1. Package the model archive (.mar)
    torch-model-archiver \
        --model-name gain_mtl \
        --version 1.0 \
        --handler serving/handler.py \
        --extra-files "src,configs/default.yaml" \
        --serialized-file checkpoints/best_model.pth \
        --export-path model_store

    # 2. Start TorchServe
    torchserve --start \
        --model-store model_store \
        --models gain_mtl=gain_mtl.mar \
        --ts-config serving/config.properties

    # 3. Inference request
    curl -X POST http://localhost:8080/predictions/gain_mtl \
        -T test_image.png

References:
    - TorchServe custom handler: https://pytorch.org/serve/custom_service.html
    - GAIN-MTL model: src/models/gain_mtl.py
"""

import base64
import io
import json
import logging
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from ts.torch_handler.base_handler import BaseHandler

logger = logging.getLogger(__name__)


class GAINMTLHandler(BaseHandler):
    """
    TorchServe handler for GAIN-MTL defect detection model.

    Supports:
    - Binary classification (normal vs defective)
    - Attention/CAM heatmap generation
    - Configurable strategy (1-6)
    - Configurable image size and model architecture
    """

    # ImageNet normalization constants
    IMAGENET_MEAN = [0.485, 0.456, 0.406]
    IMAGENET_STD = [0.229, 0.224, 0.225]

    CLASS_NAMES = ["normal", "defective"]

    def __init__(self):
        super().__init__()
        self.initialized = False
        self.image_size = (512, 512)
        self.strategy = 5
        self.threshold = 0.5
        self.return_cam = True

    def initialize(self, context):
        """
        Load the GAIN-MTL model and configuration.

        Called once when the model is first loaded by TorchServe.

        Args:
            context: TorchServe context object containing model metadata
        """
        properties = context.system_properties
        model_dir = properties.get("model_dir")

        # Add model directory to Python path so that `src` package is importable
        if model_dir not in sys.path:
            sys.path.insert(0, model_dir)

        from src.models import GAINMTLModel

        # ---- Load handler config (serving/config.json bundled as extra-file) ----
        handler_config = self._load_handler_config(model_dir)

        model_config = handler_config.get("model", {})
        self.image_size = tuple(handler_config.get("image_size", [512, 512]))
        self.strategy = handler_config.get("strategy", 5)
        if self.strategy not in (1, 2, 3, 4, 5, 6):
            raise ValueError(
                f"Invalid strategy: {self.strategy}. Must be 1-6."
            )
        self.threshold = handler_config.get("threshold", 0.5)
        self.return_cam = handler_config.get("return_cam", True)
        class_names = handler_config.get("class_names")
        if class_names:
            self.CLASS_NAMES = class_names

        # ---- Device ----
        self.device = torch.device(
            properties.get("gpu_id")
            if torch.cuda.is_available() and properties.get("gpu_id") is not None
            else "cpu"
        )
        if isinstance(self.device, int) or (
            isinstance(self.device, str) and self.device.isdigit()
        ):
            self.device = torch.device(f"cuda:{self.device}")

        logger.info("Device: %s", self.device)

        # ---- Build model ----
        self.model = GAINMTLModel(
            backbone_arch=model_config.get("backbone_arch", "s"),
            num_classes=model_config.get("num_classes", 2),
            pretrained=False,  # weights come from the checkpoint
            fpn_channels=model_config.get("fpn_channels", 256),
            attention_channels=model_config.get("attention_channels", 512),
            use_counterfactual=model_config.get("use_counterfactual", True),
            freeze_backbone_stages=model_config.get("freeze_backbone_stages", -1),
        )

        # ---- Load weights ----
        checkpoint_path = self._find_checkpoint(model_dir)
        if checkpoint_path is None:
            raise FileNotFoundError(
                f"No checkpoint (.pth/.pt) found in model_dir: {model_dir}"
            )

        logger.info("Loading checkpoint: %s", checkpoint_path)
        checkpoint = torch.load(checkpoint_path, map_location=self.device, weights_only=False)

        if "model_state_dict" in checkpoint:
            self.model.load_state_dict(checkpoint["model_state_dict"])
            epoch = checkpoint.get("epoch", "unknown")
            logger.info("Loaded model_state_dict from epoch %s", epoch)
        else:
            self.model.load_state_dict(checkpoint)

        self.model.set_strategy(self.strategy)
        self.model.to(self.device)
        self.model.eval()

        logger.info(
            "GAIN-MTL model initialized (backbone=%s, strategy=%d, image_size=%s)",
            model_config.get("backbone_arch", "s"),
            self.strategy,
            self.image_size,
        )
        self.initialized = True

    # ------------------------------------------------------------------
    # Preprocess
    # ------------------------------------------------------------------

    def preprocess(self, data: List[Dict[str, Any]]) -> torch.Tensor:
        """
        Convert raw input (image bytes / base64) into a normalized tensor.

        Accepts:
        - Binary image data (multipart/form-data or raw bytes)
        - Base64-encoded image string in JSON body

        Args:
            data: List of request payloads from TorchServe

        Returns:
            Batched tensor of shape (B, 3, H, W)
        """
        images = []
        for row in data:
            # Extract raw bytes from the request body
            image_data = row.get("data") or row.get("body")

            if isinstance(image_data, str):
                # Base64-encoded string
                image_data = base64.b64decode(image_data)
            if isinstance(image_data, (bytearray, memoryview)):
                image_data = bytes(image_data)

            image = Image.open(io.BytesIO(image_data)).convert("RGB")
            tensor = self._transform(image)
            images.append(tensor)

        return torch.stack(images).to(self.device)

    def _transform(self, image: Image.Image) -> torch.Tensor:
        """
        Apply inference transforms: resize -> to tensor -> normalize.

        Mirrors the validation/test transform from src/data/dataset.py
        without albumentations dependency (pure PIL + torch).

        Args:
            image: PIL Image in RGB mode

        Returns:
            Normalized tensor of shape (3, H, W)
        """
        image = image.resize((self.image_size[1], self.image_size[0]), Image.BILINEAR)

        # PIL -> numpy -> tensor  (H, W, C) -> (C, H, W)
        arr = np.array(image, dtype=np.float32) / 255.0
        tensor = torch.from_numpy(arr).permute(2, 0, 1)  # (3, H, W)

        # Normalize with ImageNet stats
        mean = torch.tensor(self.IMAGENET_MEAN).view(3, 1, 1)
        std = torch.tensor(self.IMAGENET_STD).view(3, 1, 1)
        tensor = (tensor - mean) / std

        return tensor

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------

    def inference(self, data: torch.Tensor, *args, **kwargs) -> Dict[str, torch.Tensor]:
        """
        Run model forward pass.

        Args:
            data: Preprocessed image tensor (B, 3, H, W)

        Returns:
            Raw model output dictionary
        """
        with torch.no_grad():
            outputs = self.model(data)
        return outputs

    # ------------------------------------------------------------------
    # Postprocess
    # ------------------------------------------------------------------

    def postprocess(self, inference_output: Dict[str, torch.Tensor]) -> List[Dict[str, Any]]:
        """
        Convert model outputs to JSON-serializable response.

        Each item in the returned list corresponds to one image in the batch.

        Response format per image:
        {
            "class_index": 1,
            "class_name": "defective",
            "confidence": 0.9876,
            "probabilities": {"normal": 0.0124, "defective": 0.9876},
            "defect_detected": true,
            "cam": [[0.0, 0.1, ...], ...]   // optional, H x W float array
        }
        """
        # Determine which logits to use based on strategy
        if self.strategy <= 2:
            cls_logits = inference_output["cls_logits"]
            cam_prob = inference_output["cam_prob"]
        else:
            cls_logits = inference_output["attended_cls_logits"]
            cam_prob = inference_output["attention_map_prob"]

        probs = F.softmax(cls_logits, dim=1)  # (B, num_classes)
        confidence, predictions = probs.max(dim=1)  # (B,)

        # Upsample CAM to input image size
        cam_upsampled = F.interpolate(
            cam_prob,
            size=self.image_size,
            mode="bilinear",
            align_corners=False,
        )  # (B, 1, H, W)

        batch_size = cls_logits.size(0)
        results = []
        for i in range(batch_size):
            pred_idx = predictions[i].item()
            result = {
                "class_index": pred_idx,
                "class_name": self.CLASS_NAMES[pred_idx] if pred_idx < len(self.CLASS_NAMES) else str(pred_idx),
                "confidence": round(confidence[i].item(), 4),
                "probabilities": {
                    name: round(probs[i, j].item(), 4)
                    for j, name in enumerate(self.CLASS_NAMES)
                },
                "defect_detected": bool((cam_upsampled[i] > self.threshold).any().item()),
            }

            if self.return_cam:
                # Squeeze (1, H, W) -> (H, W), convert to list of lists
                cam_np = cam_upsampled[i, 0].cpu().numpy()
                # Quantize to 3 decimal places to reduce payload size
                result["cam"] = np.round(cam_np, 3).tolist()

            results.append(result)

        return results

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _load_handler_config(self, model_dir: str) -> dict:
        """
        Load handler configuration from config.json in the model archive.

        Falls back to configs/default.yaml if config.json is not found.
        """
        # Try config.json first (recommended for TorchServe)
        config_path = os.path.join(model_dir, "config.json")
        if os.path.exists(config_path):
            logger.info("Loading handler config: %s", config_path)
            with open(config_path, "r") as f:
                return json.load(f)

        # Fallback: parse default.yaml
        yaml_path = os.path.join(model_dir, "configs", "default.yaml")
        if not os.path.exists(yaml_path):
            yaml_path = os.path.join(model_dir, "default.yaml")

        if os.path.exists(yaml_path):
            logger.info("Loading config from YAML: %s", yaml_path)
            import yaml
            with open(yaml_path, "r") as f:
                full_config = yaml.safe_load(f)
            return {
                "model": full_config.get("model", {}),
                "image_size": full_config.get("data", {}).get("image_size", [512, 512]),
                "threshold": full_config.get("inference", {}).get("threshold", 0.5),
                "strategy": 5,
                "return_cam": True,
            }

        logger.warning("No config found in %s, using defaults", model_dir)
        return {}

    def _find_checkpoint(self, model_dir: str) -> Optional[str]:
        """Find a serialized checkpoint file in model_dir."""
        # TorchServe places the --serialized-file at the model_dir root
        for name in os.listdir(model_dir):
            if name.endswith((".pth", ".pt", ".bin")):
                return os.path.join(model_dir, name)
        return None
