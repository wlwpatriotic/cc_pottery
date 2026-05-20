"""
Main experiment runner: trains and evaluates all baselines.
Usage: python run_experiments.py --model vit --task culture
       python run_experiments.py --model all --task all
"""
import sys
import os
import argparse
import json
import warnings
from pathlib import Path

import torch
from torch.utils.data import DataLoader

warnings.filterwarnings('ignore')

# Add project root
PROJECT_ROOT = Path(r'f:\考古\cc_pottery')
sys.path.insert(0, str(PROJECT_ROOT))

from utils.dataset import PotteryDataset, get_transforms, collate_fn
from utils.trainer import MultiTaskTrainer
from models.baselines import ViTFineTune, ResNetFineTune, HierarchicalViT


def build_loaders(root, batch_size=32, num_workers=0, image_size=224):
    """Build train/val/test dataloaders."""
    train_dataset = PotteryDataset(root, split='train', min_samples_per_class=5,
                                    transform=get_transforms(image_size, is_train=True))
    val_dataset = PotteryDataset(root, split='val', min_samples_per_class=5,
                                  transform=get_transforms(image_size, is_train=False))
    test_dataset = PotteryDataset(root, split='test', min_samples_per_class=5,
                                   transform=get_transforms(image_size, is_train=False))

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True,
                              num_workers=num_workers, collate_fn=collate_fn, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False,
                            num_workers=num_workers, collate_fn=collate_fn, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False,
                             num_workers=num_workers, collate_fn=collate_fn, pin_memory=True)

    return train_loader, val_loader, test_loader, train_dataset.get_label_info()


def run_baseline_experiment(model_name, device='cuda'):
    """Run a single baseline experiment."""
    print(f"\n{'='*70}")
    print(f"EXPERIMENT: {model_name}")
    print(f"{'='*70}")

    root = str(PROJECT_ROOT)
    train_loader, val_loader, test_loader, label_info = build_loaders(root)

    # Build model
    if model_name == 'vit':
        model = ViTFineTune(
            num_cultures=label_info['num_cultures'],
            num_types=label_info['num_types'],
            num_eras=label_info['num_eras'],
            pretrained=True,
            freeze_backbone=False,
        )
    elif model_name == 'resnet':
        model = ResNetFineTune(
            num_cultures=label_info['num_cultures'],
            num_types=label_info['num_types'],
            num_eras=label_info['num_eras'],
            pretrained=True,
        )
    elif model_name == 'hierarchical_vit':
        model = HierarchicalViT(
            num_cultures=label_info['num_cultures'],
            num_types=label_info['num_types'],
            num_eras=label_info['num_eras'],
            pretrained=True,
        )
    else:
        raise ValueError(f"Unknown model: {model_name}")

    # Output directory
    output_dir = PROJECT_ROOT / 'outputs' / f'{model_name}_baseline'
    output_dir.mkdir(parents=True, exist_ok=True)

    # Class weights for long-tail handling
    class_weights = {
        'culture': label_info['culture_weights'],
        'type': label_info['type_weights'],
    }

    # Train
    trainer = MultiTaskTrainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        test_loader=test_loader,
        device=device,
        output_dir=str(output_dir),
        class_weights=class_weights,
        task_weights={'culture': 1.0, 'type': 1.0, 'era': 0.3},
        lr=1e-4,
        weight_decay=1e-4,
    )

    history = trainer.train(num_epochs=30, patience=7, eval_every=1)

    # Save results
    results = {
        'model': model_name,
        'best_val_acc': trainer.best_val_acc,
        'final_test_metrics': history['test'][-1] if history['test'] else None,
        'label_info': {k: v for k, v in label_info.items()
                       if not k.startswith('_') and not isinstance(v, dict)},
    }
    with open(output_dir / 'results.json', 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False, default=str)

    print(f"\nResults saved to {output_dir}")
    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default='vit',
                       choices=['vit', 'resnet', 'hierarchical_vit', 'all'])
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--test-only', action='store_true')
    args = parser.parse_args()

    if not torch.cuda.is_available():
        print("WARNING: CUDA not available, using CPU")
        args.device = 'cpu'

    models_to_run = ['vit', 'resnet', 'hierarchical_vit'] if args.model == 'all' else [args.model]

    all_results = {}
    for model_name in models_to_run:
        try:
            results = run_baseline_experiment(model_name, args.device)
            all_results[model_name] = results
        except Exception as e:
            print(f"ERROR running {model_name}: {e}")
            import traceback
            traceback.print_exc()

    # Summary
    print(f"\n{'='*70}")
    print("SUMMARY")
    print(f"{'='*70}")
    for model_name, results in all_results.items():
        print(f"  {model_name}: best_val_acc={results.get('best_val_acc', 'N/A')}")


if __name__ == '__main__':
    main()
