"""
PotteryDataset: Hierarchical fine-grained dataset with culture/type/era labels.
Supports stratified splits, long-tail handling, and multi-task training.
"""
import json
import os
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from PIL import Image
from torch.utils.data import Dataset
from sklearn.model_selection import StratifiedKFold, train_test_split
import torch

class PotteryDataset(Dataset):
    """Fine-grained painted pottery dataset with hierarchical labels."""

    def __init__(self, data_root, split='train', split_file=None,
                 transform=None, target_tasks=('culture', 'type', 'era'),
                 use_valid_images_only=True, min_samples_per_class=0):
        """
        Args:
            data_root: Path to project root (cc_pottery/)
            split: 'train', 'val', 'test', or 'all'
            split_file: Path to pre-computed split JSON (if None, auto-split)
            transform: Albumentations/torchvision transforms
            target_tasks: Which labels to return ('culture', 'type', 'era')
            use_valid_images_only: Filter to samples with existing images
            min_samples_per_class: Filter tail classes with fewer samples
        """
        self.data_root = Path(data_root)
        self.split = split
        self.transform = transform
        self.target_tasks = target_tasks
        self.use_valid_images_only = use_valid_images_only

        # Load annotations
        json_path = self.data_root / 'pottery_dataset_index.json'
        with open(json_path, 'r', encoding='utf-8') as f:
            self.annotations = json.load(f)

        # Filter to valid images
        if use_valid_images_only:
            self.annotations = [
                item for item in self.annotations
                if self._has_valid_images(item)
            ]

        # Build label mappings
        self._build_label_mappings()

        # Filter tail classes
        if min_samples_per_class > 0:
            self.annotations = self._filter_tail_classes(min_samples_per_class)
            # Rebuild mappings after filtering
            self._build_label_mappings()

        # Load or create splits
        if split_file and os.path.exists(split_file):
            self._load_splits(split_file)
        elif split != 'all':
            self._create_splits()

        print(f"[PotteryDataset] {split} split: {len(self.annotations)} samples")
        print(f"  Cultures: {self.num_cultures}, Types: {self.num_types}, Eras: {self.num_eras}")

    def _has_valid_images(self, item):
        """Check if at least one image file exists for this item."""
        for img_path in item['images']:
            if os.path.exists(img_path):
                return True
        return False

    def _build_label_mappings(self):
        """Build categorical label encodings for culture, type, era."""
        self.culture_to_idx = {}
        self.type_to_idx = {}
        self.era_to_idx = {}
        self.idx_to_culture = {}
        self.idx_to_type = {}
        self.idx_to_era = {}

        for item in self.annotations:
            c = item['culture']
            t = item['name']
            e = item['era']

            if c not in self.culture_to_idx:
                idx = len(self.culture_to_idx)
                self.culture_to_idx[c] = idx
                self.idx_to_culture[idx] = c

            if t not in self.type_to_idx:
                idx = len(self.type_to_idx)
                self.type_to_idx[t] = idx
                self.idx_to_type[idx] = t

            if e not in self.era_to_idx:
                idx = len(self.era_to_idx)
                self.era_to_idx[e] = idx
                self.idx_to_era[idx] = e

        self.num_cultures = len(self.culture_to_idx)
        self.num_types = len(self.type_to_idx)
        self.num_eras = len(self.era_to_idx)

        # Compute class weights for balanced loss
        culture_counts = Counter(item['culture'] for item in self.annotations)
        total = len(self.annotations)
        self.culture_weights = torch.tensor(
            [total / max(culture_counts[self.idx_to_culture[i]], 1)
             for i in range(self.num_cultures)],
            dtype=torch.float32
        )

        type_counts = Counter(item['name'] for item in self.annotations)
        self.type_weights = torch.tensor(
            [total / max(type_counts[self.idx_to_type[i]], 1)
             for i in range(self.num_types)],
            dtype=torch.float32
        )

    def _filter_tail_classes(self, min_samples):
        """Remove samples from classes with fewer than min_samples."""
        culture_counts = Counter(item['culture'] for item in self.annotations)
        keep = []
        for item in self.annotations:
            if culture_counts[item['culture']] >= min_samples:
                keep.append(item)
        return keep

    def _create_splits(self):
        """Create stratified train/val/test splits."""
        # Use culture+type as stratification key
        stratify_keys = [
            f"{item['culture']}|{item['name']}"
            for item in self.annotations
        ]

        # Group tail classes together for stratification
        key_counts = Counter(stratify_keys)
        valid_keys = [k if key_counts[k] >= 3 else 'TAIL' for k in stratify_keys]

        # First split: train+val vs test (80/20)
        train_val_idx, test_idx = train_test_split(
            np.arange(len(self.annotations)),
            test_size=0.15,
            stratify=valid_keys,
            random_state=42
        )

        # Second split: train vs val (85/15 of train_val)
        train_val_keys = [valid_keys[i] for i in train_val_idx]
        train_idx, val_idx = train_test_split(
            train_val_idx,
            test_size=0.176,  # 0.15 / 0.85 ≈ 0.176
            stratify=train_val_keys,
            random_state=42
        )

        if self.split == 'train':
            self.annotations = [self.annotations[i] for i in train_idx]
        elif self.split == 'val':
            self.annotations = [self.annotations[i] for i in val_idx]
        elif self.split == 'test':
            self.annotations = [self.annotations[i] for i in test_idx]

    def _load_splits(self, split_file):
        """Load pre-computed train/val/test split indices."""
        with open(split_file, 'r', encoding='utf-8') as f:
            split_data = json.load(f)
        uids = [item['uid'] for item in self.annotations]
        split_uids = split_data[self.split]
        self.annotations = [
            item for item in self.annotations
            if item['uid'] in split_uids
        ]

    def _get_valid_image(self, item):
        """Get the first valid image path for an item."""
        for img_path in item['images']:
            if os.path.exists(img_path):
                return img_path
        return item['images'][0]  # fallback

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, idx):
        item = self.annotations[idx]

        # Load image
        img_path = self._get_valid_image(item)
        try:
            image = Image.open(img_path).convert('RGB')
        except Exception:
            # Return a blank image on failure
            image = Image.new('RGB', (224, 224), (128, 128, 128))

        if self.transform:
            image = self.transform(image)
        else:
            from torchvision import transforms
            image = transforms.ToTensor()(image)

        # Build labels
        labels = {}
        if 'culture' in self.target_tasks:
            labels['culture'] = self.culture_to_idx.get(item['culture'], -1)
        if 'type' in self.target_tasks:
            labels['type'] = self.type_to_idx.get(item['name'], -1)
        if 'era' in self.target_tasks:
            labels['era'] = self.era_to_idx.get(item['era'], -1)

        # Return metadata for reasoning tasks
        meta = {
            'uid': item['uid'],
            'description': item['description'],
            'culture_name': item['culture'],
            'type_name': item['name'],
            'era_name': item['era'],
            'dimensions': item['dimensions'],
            'excavation': item['excavation'],
        }

        return image, labels, meta

    def get_label_info(self):
        """Return label mappings and statistics for model building."""
        return {
            'num_cultures': self.num_cultures,
            'num_types': self.num_types,
            'num_eras': self.num_eras,
            'culture_to_idx': self.culture_to_idx,
            'type_to_idx': self.type_to_idx,
            'era_to_idx': self.era_to_idx,
            'idx_to_culture': self.idx_to_culture,
            'idx_to_type': self.idx_to_type,
            'idx_to_era': self.idx_to_era,
            'culture_weights': self.culture_weights,
            'type_weights': self.type_weights,
        }

    def get_class_distribution(self):
        """Get per-class sample counts for analysis."""
        culture_counts = Counter(item['culture'] for item in self.annotations)
        type_counts = Counter(item['name'] for item in self.annotations)
        era_counts = Counter(item['era'] for item in self.annotations)
        return {
            'culture_distribution': dict(culture_counts.most_common()),
            'type_distribution': dict(type_counts.most_common()),
            'era_distribution': dict(era_counts.most_common()),
        }


def get_transforms(image_size=224, is_train=True):
    """Get standard image transforms for training/evaluation."""
    from torchvision import transforms

    if is_train:
        return transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(degrees=15),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.1, hue=0.05),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                               std=[0.229, 0.224, 0.225]),
        ])
    else:
        return transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                               std=[0.229, 0.224, 0.225]),
        ])


def collate_fn(batch):
    """Custom collate to handle hierarchical labels and metadata."""
    images = torch.stack([item[0] for item in batch])
    labels = {key: torch.tensor([item[1][key] for item in batch])
              for key in batch[0][1]}
    metas = [item[2] for item in batch]
    return images, labels, metas


if __name__ == '__main__':
    # Test dataset loading
    root = r'f:\考古\cc_pottery'
    ds = PotteryDataset(root, split='train', min_samples_per_class=5)
    info = ds.get_label_info()
    dist = ds.get_class_distribution()

    print(f"\nLoaded dataset with {len(ds)} training samples")
    print(f"Cultures: {info['num_cultures']}, Types: {info['num_types']}, Eras: {info['num_eras']}")

    # Test a sample
    img, labels, meta = ds[0]
    print(f"\nSample 0:")
    print(f"  Image shape: {img.shape}")
    print(f"  Labels: culture={labels['culture']}, type={labels['type']}, era={labels['era']}")
    print(f"  Culture: {meta['culture_name']}")
    print(f"  Type: {meta['type_name']}")
    print(f"  Description: {meta['description'][:100]}...")
