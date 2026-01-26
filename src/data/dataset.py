"""
Dataset Classes for Manufacturing Defect Detection

Supports:
- Generic defect dataset with classification + localization labels
- MVTec Anomaly Detection dataset
- Custom manufacturing datasets

Data format expected:
- Images: RGB images
- Labels: Binary (0=normal, 1=defective)
- Masks: Binary masks indicating defect regions (optional for normal samples)
"""

import os
import json
from typing import Any, Callable, Dict, List, Optional, Tuple, Union
from pathlib import Path

import numpy as np
import cv2
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader, Subset
from torch.utils.data import random_split
import albumentations as A
from albumentations.pytorch import ToTensorV2


class DefectDataset(Dataset):
    """
    Generic Dataset for Manufacturing Defect Detection.

    Expects data organized as:
    ```
    data_root/
    ├── images/
    │   ├── train/
    │   │   ├── normal/
    │   │   │   ├── img_001.png
    │   │   │   └── ...
    │   │   └── defective/
    │   │       ├── img_001.png
    │   │       └── ...
    │   └── test/
    │       └── ...
    ├── masks/
    │   └── defective/
    │       ├── img_001.png  (defect mask, same name as image)
    │       └── ...
    └── annotations.json (optional)
    ```

    Args:
        data_root: Root directory of the dataset
        split: Data split ('train', 'val', 'test')
        transform: Albumentations transform pipeline
        image_size: Target image size (H, W)
        mask_suffix: Suffix for mask files (e.g., '_mask')
    """

    def __init__(
        self,
        data_root: str,
        split: str = 'train',
        transform: Optional[A.Compose] = None,
        image_size: Tuple[int, int] = (512, 512),
        mask_suffix: str = '',
    ):
        self.data_root = Path(data_root)
        self.split = split
        self.image_size = image_size
        self.mask_suffix = mask_suffix

        # Default transform if none provided
        self.transform = transform or self._default_transform(split)

        # Load data
        self.samples = self._load_samples()

        print(f'Loaded {len(self.samples)} samples for {split} split')
        print(f'  Normal: {sum(1 for s in self.samples if s["label"] == 0)}')
        print(f'  Defective: {sum(1 for s in self.samples if s["label"] == 1)}')

    def _default_transform(self, split: str) -> A.Compose:
        """Create default augmentation pipeline."""
        if split == 'train':
            return A.Compose([
                A.Resize(self.image_size[0], self.image_size[1]),
                A.HorizontalFlip(p=0.5),
                A.VerticalFlip(p=0.5),
                A.RandomRotate90(p=0.5),
                A.OneOf([
                    A.GaussNoise(var_limit=(10, 50)),
                    A.GaussianBlur(blur_limit=(3, 7)),
                    A.MotionBlur(blur_limit=(3, 7)),
                ], p=0.3),
                A.OneOf([
                    A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2),
                    A.HueSaturationValue(hue_shift_limit=10, sat_shift_limit=20, val_shift_limit=10),
                ], p=0.3),
                A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                ToTensorV2(),
            ])
        else:
            return A.Compose([
                A.Resize(self.image_size[0], self.image_size[1]),
                A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                ToTensorV2(),
            ])

    def _load_samples(self) -> List[Dict[str, Any]]:
        """Load all samples from the dataset."""
        samples = []

        # Check for annotation file
        anno_path = self.data_root / 'annotations.json'
        if anno_path.exists():
            return self._load_from_annotations(anno_path)

        # Load from directory structure
        images_dir = self.data_root / 'images' / self.split
        masks_dir = self.data_root / 'masks'

        # Normal samples
        normal_dir = images_dir / 'normal'
        if normal_dir.exists():
            for img_path in normal_dir.glob('*'):
                if img_path.suffix.lower() in ['.png', '.jpg', '.jpeg', '.bmp']:
                    samples.append({
                        'image_path': str(img_path),
                        'mask_path': None,
                        'label': 0,  # Normal
                        'has_defect': False,
                    })

        # Defective samples
        defective_dir = images_dir / 'defective'
        if defective_dir.exists():
            for img_path in defective_dir.glob('*'):
                if img_path.suffix.lower() in ['.png', '.jpg', '.jpeg', '.bmp']:
                    # Find corresponding mask
                    mask_name = img_path.stem + self.mask_suffix + img_path.suffix
                    mask_path = masks_dir / 'defective' / mask_name

                    # Also check without suffix
                    if not mask_path.exists():
                        mask_path = masks_dir / 'defective' / img_path.name

                    samples.append({
                        'image_path': str(img_path),
                        'mask_path': str(mask_path) if mask_path.exists() else None,
                        'label': 1,  # Defective
                        'has_defect': True,
                    })

        return samples

    def _load_from_annotations(self, anno_path: Path) -> List[Dict[str, Any]]:
        """Load samples from annotations file."""
        with open(anno_path, 'r') as f:
            annotations = json.load(f)

        samples = []
        for anno in annotations.get(self.split, []):
            samples.append({
                'image_path': str(self.data_root / anno['image']),
                'mask_path': str(self.data_root / anno['mask']) if anno.get('mask') else None,
                'label': anno['label'],
                'has_defect': anno['label'] == 1,
            })

        return samples

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        sample = self.samples[idx]

        # Load image
        image = cv2.imread(sample['image_path'])
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Load mask (or create empty mask for normal samples)
        if sample['mask_path'] and os.path.exists(sample['mask_path']):
            mask = cv2.imread(sample['mask_path'], cv2.IMREAD_GRAYSCALE)
            mask = (mask > 127).astype(np.float32)  # Binarize
        else:
            mask = np.zeros((image.shape[0], image.shape[1]), dtype=np.float32)

        # Apply transforms
        transformed = self.transform(image=image, mask=mask)
        image = transformed['image']
        mask = transformed['mask']

        # Add channel dimension to mask
        if mask.ndim == 2:
            mask = mask.unsqueeze(0)

        return {
            'image': image,
            'defect_mask': mask,
            'label': torch.tensor(sample['label'], dtype=torch.long),
            'has_defect': torch.tensor(sample['has_defect'], dtype=torch.bool),
            'image_path': sample['image_path'],
        }


class MVTecDataset(Dataset):
    """
    MVTec Anomaly Detection Dataset.

    Standard benchmark for anomaly detection.
    Structure:
    ```
    mvtec_root/
    ├── bottle/
    │   ├── train/
    │   │   └── good/
    │   ├── test/
    │   │   ├── good/
    │   │   ├── broken_large/
    │   │   ├── broken_small/
    │   │   └── contamination/
    │   └── ground_truth/
    │       ├── broken_large/
    │       ├── broken_small/
    │       └── contamination/
    └── ...
    ```

    Args:
        data_root: Root directory of MVTec dataset
        category: Product category (e.g., 'bottle', 'cable', 'capsule')
        split: Data split ('train', 'test')
        transform: Albumentations transform pipeline
        image_size: Target image size
    """

    CATEGORIES = [
        'bottle', 'cable', 'capsule', 'carpet', 'grid',
        'hazelnut', 'leather', 'metal_nut', 'pill', 'screw',
        'tile', 'toothbrush', 'transistor', 'wood', 'zipper'
    ]

    def __init__(
        self,
        data_root: str,
        category: str,
        split: str = 'train',
        transform: Optional[A.Compose] = None,
        image_size: Tuple[int, int] = (256, 256),
    ):
        assert category in self.CATEGORIES, f'Unknown category: {category}'

        self.data_root = Path(data_root)
        self.category = category
        self.split = split
        self.image_size = image_size

        # Default transform
        if transform is None:
            transform = get_transforms(split, image_size)
        self.transform = transform

        # Load samples
        self.samples = self._load_samples()

        print(f'MVTec {category} - {split}: {len(self.samples)} samples')

    def _load_samples(self) -> List[Dict[str, Any]]:
        samples = []
        category_dir = self.data_root / self.category

        if self.split == 'train':
            # Training: only good samples
            good_dir = category_dir / 'train' / 'good'
            for img_path in good_dir.glob('*.png'):
                samples.append({
                    'image_path': str(img_path),
                    'mask_path': None,
                    'label': 0,
                    'has_defect': False,
                    'defect_type': 'good',
                })
        else:
            # Test: good + all defect types
            test_dir = category_dir / 'test'
            gt_dir = category_dir / 'ground_truth'

            for defect_type_dir in test_dir.iterdir():
                if not defect_type_dir.is_dir():
                    continue

                defect_type = defect_type_dir.name
                is_defective = defect_type != 'good'

                for img_path in defect_type_dir.glob('*.png'):
                    if is_defective:
                        # Find ground truth mask
                        mask_path = gt_dir / defect_type / f'{img_path.stem}_mask.png'
                    else:
                        mask_path = None

                    samples.append({
                        'image_path': str(img_path),
                        'mask_path': str(mask_path) if mask_path and mask_path.exists() else None,
                        'label': 1 if is_defective else 0,
                        'has_defect': is_defective,
                        'defect_type': defect_type,
                    })

        return samples

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        sample = self.samples[idx]

        # Load image
        image = cv2.imread(sample['image_path'])
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Load or create mask
        if sample['mask_path'] and os.path.exists(sample['mask_path']):
            mask = cv2.imread(sample['mask_path'], cv2.IMREAD_GRAYSCALE)
            mask = (mask > 0).astype(np.float32)
        else:
            mask = np.zeros((image.shape[0], image.shape[1]), dtype=np.float32)

        # Apply transforms
        transformed = self.transform(image=image, mask=mask)
        image = transformed['image']
        mask = transformed['mask']

        if mask.ndim == 2:
            mask = mask.unsqueeze(0)

        return {
            'image': image,
            'defect_mask': mask,
            'label': torch.tensor(sample['label'], dtype=torch.long),
            'has_defect': torch.tensor(sample['has_defect'], dtype=torch.bool),
            'defect_type': sample['defect_type'],
            'image_path': sample['image_path'],
        }


def get_transforms(
    split: str,
    image_size: Tuple[int, int] = (512, 512),
    augmentation_level: str = 'medium',
) -> A.Compose:
    """
    Get augmentation pipeline.

    Args:
        split: Data split ('train', 'val', 'test')
        image_size: Target image size
        augmentation_level: 'light', 'medium', 'heavy'

    Returns:
        Albumentations Compose transform
    """
    if split == 'train':
        if augmentation_level == 'light':
            return A.Compose([
                A.Resize(image_size[0], image_size[1]),
                A.HorizontalFlip(p=0.5),
                A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                ToTensorV2(),
            ])
        elif augmentation_level == 'medium':
            return A.Compose([
                A.Resize(image_size[0], image_size[1]),
                A.HorizontalFlip(p=0.5),
                A.VerticalFlip(p=0.5),
                A.RandomRotate90(p=0.5),
                A.ShiftScaleRotate(
                    shift_limit=0.1,
                    scale_limit=0.1,
                    rotate_limit=15,
                    p=0.5
                ),
                A.OneOf([
                    A.GaussNoise(var_limit=(10, 50)),
                    A.GaussianBlur(blur_limit=(3, 5)),
                ], p=0.2),
                A.RandomBrightnessContrast(
                    brightness_limit=0.2,
                    contrast_limit=0.2,
                    p=0.3
                ),
                A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                ToTensorV2(),
            ])
        else:  # heavy
            return A.Compose([
                A.Resize(image_size[0], image_size[1]),
                A.HorizontalFlip(p=0.5),
                A.VerticalFlip(p=0.5),
                A.RandomRotate90(p=0.5),
                A.ShiftScaleRotate(
                    shift_limit=0.15,
                    scale_limit=0.15,
                    rotate_limit=30,
                    p=0.7
                ),
                A.OneOf([
                    A.GaussNoise(var_limit=(10, 80)),
                    A.GaussianBlur(blur_limit=(3, 7)),
                    A.MotionBlur(blur_limit=(3, 7)),
                ], p=0.4),
                A.OneOf([
                    A.RandomBrightnessContrast(brightness_limit=0.3, contrast_limit=0.3),
                    A.HueSaturationValue(hue_shift_limit=20, sat_shift_limit=30, val_shift_limit=20),
                    A.CLAHE(clip_limit=4),
                ], p=0.5),
                A.CoarseDropout(
                    max_holes=8,
                    max_height=32,
                    max_width=32,
                    fill_value=0,
                    p=0.3
                ),
                A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                ToTensorV2(),
            ])
    else:
        return A.Compose([
            A.Resize(image_size[0], image_size[1]),
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ToTensorV2(),
        ])


class SubsetWithTransform(Subset):
    """
    Subset that allows overriding the transform of the underlying dataset.

    This is useful when splitting a dataset into train/val and wanting
    different augmentations for each split.
    """

    def __init__(self, dataset: Dataset, indices: List[int], transform: Optional[A.Compose] = None):
        super().__init__(dataset, indices)
        self.custom_transform = transform

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        # Get original sample info
        original_idx = self.indices[idx]
        sample = self.dataset.samples[original_idx]

        # Load image
        image = cv2.imread(sample['image_path'])
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Load mask
        if sample.get('mask_path') and os.path.exists(sample['mask_path']):
            mask = cv2.imread(sample['mask_path'], cv2.IMREAD_GRAYSCALE)
            mask = (mask > 127).astype(np.float32) if mask.max() > 1 else mask.astype(np.float32)
        else:
            mask = np.zeros((image.shape[0], image.shape[1]), dtype=np.float32)

        # Apply custom transform if provided, otherwise use dataset's transform
        transform = self.custom_transform if self.custom_transform else self.dataset.transform
        transformed = transform(image=image, mask=mask)
        image = transformed['image']
        mask = transformed['mask']

        if mask.ndim == 2:
            mask = mask.unsqueeze(0)

        result = {
            'image': image,
            'defect_mask': mask,
            'label': torch.tensor(sample['label'], dtype=torch.long),
            'has_defect': torch.tensor(sample.get('has_defect', sample['label'] == 1), dtype=torch.bool),
            'image_path': sample['image_path'],
        }

        # Include defect_type if available (for MVTec)
        if 'defect_type' in sample:
            result['defect_type'] = sample['defect_type']

        return result


def create_dataloaders(
    data_root: str,
    batch_size: int = 16,
    image_size: Tuple[int, int] = (512, 512),
    num_workers: int = 4,
    pin_memory: bool = True,
    dataset_type: str = 'generic',
    val_ratio: float = 0.0,
    seed: int = 42,
    **kwargs,
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    Create train, validation, and test data loaders.

    Args:
        data_root: Root directory of dataset
        batch_size: Batch size
        image_size: Target image size
        num_workers: Number of data loading workers
        pin_memory: Whether to pin memory
        dataset_type: 'generic' or 'mvtec'
        val_ratio: Ratio of training data to use for validation (0.0-1.0).
                   If > 0, randomly splits train data instead of using separate val directory.
        seed: Random seed for reproducible train/val split
        **kwargs: Additional arguments for dataset

    Returns:
        Tuple of (train_loader, val_loader, test_loader)
    """
    # Set seed for reproducibility
    generator = torch.Generator().manual_seed(seed)

    if dataset_type == 'mvtec':
        DatasetClass = MVTecDataset
        category = kwargs.get('category', 'bottle')

        train_dataset = DatasetClass(
            data_root, category, 'train',
            transform=get_transforms('train', image_size),
            image_size=image_size
        )

        if val_ratio > 0:
            # Split train into train/val
            total_size = len(train_dataset)
            val_size = int(total_size * val_ratio)
            train_size = total_size - val_size

            print(f'\nSplitting train data: {train_size} train, {val_size} val (ratio={val_ratio})')

            # Create indices for split
            indices = list(range(total_size))
            np.random.seed(seed)
            np.random.shuffle(indices)
            train_indices = indices[:train_size]
            val_indices = indices[train_size:]

            # Create subsets with appropriate transforms
            train_subset = SubsetWithTransform(
                train_dataset, train_indices,
                transform=get_transforms('train', image_size)
            )
            val_subset = SubsetWithTransform(
                train_dataset, val_indices,
                transform=get_transforms('val', image_size)
            )

            train_dataset = train_subset
            val_dataset = val_subset
        else:
            # MVTec doesn't have val split, use test for both
            val_dataset = DatasetClass(
                data_root, category, 'test',
                transform=get_transforms('val', image_size),
                image_size=image_size
            )

        test_dataset = DatasetClass(
            data_root, category, 'test',
            transform=get_transforms('test', image_size),
            image_size=image_size
        )
    else:
        DatasetClass = DefectDataset

        if val_ratio > 0:
            # Load all training data and split randomly
            full_train_dataset = DatasetClass(
                data_root, 'train',
                transform=get_transforms('train', image_size),
                image_size=image_size
            )

            total_size = len(full_train_dataset)
            val_size = int(total_size * val_ratio)
            train_size = total_size - val_size

            print(f'\nSplitting train data: {train_size} train, {val_size} val (ratio={val_ratio})')

            # Create indices for split
            indices = list(range(total_size))
            np.random.seed(seed)
            np.random.shuffle(indices)
            train_indices = indices[:train_size]
            val_indices = indices[train_size:]

            # Create subsets with appropriate transforms
            train_dataset = SubsetWithTransform(
                full_train_dataset, train_indices,
                transform=get_transforms('train', image_size)
            )
            val_dataset = SubsetWithTransform(
                full_train_dataset, val_indices,
                transform=get_transforms('val', image_size)
            )
        else:
            # Use separate directories
            train_dataset = DatasetClass(
                data_root, 'train',
                transform=get_transforms('train', image_size),
                image_size=image_size
            )
            val_dataset = DatasetClass(
                data_root, 'val',
                transform=get_transforms('val', image_size),
                image_size=image_size
            )

        test_dataset = DatasetClass(
            data_root, 'test',
            transform=get_transforms('test', image_size),
            image_size=image_size
        )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )

    return train_loader, val_loader, test_loader
