"""
Multi-task trainer for hierarchical fine-grained pottery classification.
Supports weighted loss, mixed precision, and comprehensive logging.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.cuda.amp import GradScaler, autocast
import numpy as np
from collections import defaultdict
import time
import json
from pathlib import Path


class HierarchicalLoss(nn.Module):
    """Weighted sum of culture + type + era losses."""

    def __init__(self, culture_weight=1.0, type_weight=1.0, era_weight=0.5,
                 class_weights=None):
        super().__init__()
        self.culture_weight = culture_weight
        self.type_weight = type_weight
        self.era_weight = era_weight
        self.class_weights = class_weights or {}

    def forward(self, predictions, labels):
        losses = {}
        total = 0.0

        if 'culture' in predictions and 'culture' in labels:
            weights = self.class_weights.get('culture', None)
            loss = F.cross_entropy(predictions['culture'], labels['culture'],
                                   weight=weights.to(predictions['culture'].device) if weights is not None else None)
            losses['culture'] = loss.item()
            total += self.culture_weight * loss

        if 'type' in predictions and 'type' in labels:
            weights = self.class_weights.get('type', None)
            loss = F.cross_entropy(predictions['type'], labels['type'],
                                   weight=weights.to(predictions['type'].device) if weights is not None else None)
            losses['type'] = loss.item()
            total += self.type_weight * loss

        if 'era' in predictions and 'era' in labels:
            loss = F.cross_entropy(predictions['era'], labels['era'])
            losses['era'] = loss.item()
            total += self.era_weight * loss

        losses['total'] = total.item() if isinstance(total, torch.Tensor) else total
        return total, losses


class MetricsTracker:
    """Tracks per-class and overall metrics across epochs."""

    def __init__(self, task_names=('culture', 'type', 'era')):
        self.task_names = task_names
        self.reset()

    def reset(self):
        self.correct = defaultdict(int)
        self.total = 0
        self.per_class_correct = defaultdict(lambda: defaultdict(int))
        self.per_class_total = defaultdict(lambda: defaultdict(int))
        self.all_preds = defaultdict(list)
        self.all_labels = defaultdict(list)

    def update(self, predictions, labels):
        self.total += 1
        for task in self.task_names:
            if task not in predictions or task not in labels:
                continue
            pred = predictions[task].argmax(dim=-1)
            target = labels[task]

            # Per-sample accuracy
            self.correct[task] += (pred == target).sum().item()

            # Per-class tracking
            for p, t in zip(pred.cpu().numpy(), target.cpu().numpy()):
                self.per_class_correct[task][t] += (p == t)
                self.per_class_total[task][t] += 1

            self.all_preds[task].extend(pred.cpu().numpy().tolist())
            self.all_labels[task].extend(target.cpu().numpy().tolist())

    def compute(self):
        metrics = {'total_samples': max(1, self.total)}
        for task in self.task_names:
            metrics[f'{task}_acc'] = (
                self.correct[task] / max(1, sum(self.per_class_total[task].values()))
            )

            # Per-class F1
            per_class_f1 = []
            for cls in sorted(self.per_class_total[task].keys()):
                tp = self.per_class_correct[task][cls]
                total = self.per_class_total[task][cls]
                recall = tp / max(1, total)
                # Estimate precision from confusion
                pred_count = sum(1 for p, t in zip(self.all_preds[task], self.all_labels[task]) if p == cls)
                precision = tp / max(1, pred_count)
                f1 = 2 * precision * recall / max(1e-8, precision + recall)
                per_class_f1.append(f1)

            metrics[f'{task}_macro_f1'] = np.mean(per_class_f1) if per_class_f1 else 0.0

        return metrics


class MultiTaskTrainer:
    """Trainer with mixed precision, logging, and checkpointing."""

    def __init__(self, model, train_loader, val_loader, test_loader=None,
                 device='cuda', output_dir='./outputs', class_weights=None,
                 task_weights=None, lr=1e-4, weight_decay=1e-4):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader
        self.device = device
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.criterion = HierarchicalLoss(
            culture_weight=task_weights.get('culture', 1.0) if task_weights else 1.0,
            type_weight=task_weights.get('type', 1.0) if task_weights else 1.0,
            era_weight=task_weights.get('era', 0.5) if task_weights else 0.5,
            class_weights=class_weights,
        )

        self.optimizer = torch.optim.AdamW(
            self.model.parameters(), lr=lr, weight_decay=weight_decay
        )
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=50, eta_min=1e-6
        )
        self.scaler = GradScaler()

        self.best_val_acc = 0.0
        self.patience_counter = 0
        self.history = {'train': [], 'val': [], 'test': []}

    def train_epoch(self):
        self.model.train()
        tracker = MetricsTracker()

        for batch_idx, (images, labels, _) in enumerate(self.train_loader):
            images = images.to(self.device)
            labels = {k: v.to(self.device) for k, v in labels.items()}

            self.optimizer.zero_grad()

            with autocast():
                predictions = self.model(images)
                loss, loss_dict = self.criterion(predictions, labels)

            self.scaler.scale(loss).backward()
            self.scaler.step(self.optimizer)
            self.scaler.update()

            tracker.update(predictions, labels)

            if batch_idx % 50 == 0:
                print(f"  Batch {batch_idx}/{len(self.train_loader)}: "
                      f"loss={loss_dict.get('total', 0):.4f} "
                      f"culture_acc={tracker.correct.get('culture', 0)/max(1, tracker.total):.3f}")

        return tracker.compute()

    @torch.no_grad()
    def validate(self, loader, split_name='val'):
        self.model.eval()
        tracker = MetricsTracker()

        for images, labels, _ in loader:
            images = images.to(self.device)
            labels = {k: v.to(self.device) for k, v in labels.items()}

            predictions = self.model(images)  # No labels during validation
            tracker.update(predictions, labels)

        metrics = tracker.compute()
        metrics = {f'{split_name}_{k}': v for k, v in metrics.items()}
        return metrics

    def train(self, num_epochs=50, patience=10, eval_every=1):
        print(f"\n{'='*60}")
        print(f"Training on {self.device} for {num_epochs} epochs")
        print(f"Train samples: {len(self.train_loader.dataset)}")
        print(f"Val samples: {len(self.val_loader.dataset)}")
        print(f"{'='*60}\n")

        for epoch in range(num_epochs):
            t0 = time.time()

            # Train
            train_metrics = self.train_epoch()
            train_metrics['epoch'] = epoch
            train_metrics['lr'] = self.optimizer.param_groups[0]['lr']
            self.history['train'].append(train_metrics)

            # Validate
            if (epoch + 1) % eval_every == 0:
                val_metrics = self.validate(self.val_loader, 'val')
                val_metrics['epoch'] = epoch
                self.history['val'].append(val_metrics)

                # Print summary
                print(f"\nEpoch {epoch+1}/{num_epochs} ({time.time()-t0:.1f}s)")
                print(f"  Train: culture_acc={train_metrics.get('culture_acc', 0):.4f} "
                      f"type_acc={train_metrics.get('type_acc', 0):.4f}")
                print(f"  Val:   culture_acc={val_metrics.get('val_culture_acc', 0):.4f} "
                      f"type_acc={val_metrics.get('val_type_acc', 0):.4f}")

                # Checkpoint best
                current_val_acc = val_metrics.get('val_culture_acc', 0)
                if current_val_acc > self.best_val_acc:
                    self.best_val_acc = current_val_acc
                    self.patience_counter = 0
                    self.save_checkpoint('best_model.pt')
                    print(f"  -> New best! culture_acc={self.best_val_acc:.4f}")
                else:
                    self.patience_counter += 1

                if self.patience_counter >= patience:
                    print(f"\nEarly stopping at epoch {epoch+1}")
                    break

            self.scheduler.step()

        # Final test evaluation
        if self.test_loader:
            self.model = self.load_checkpoint(self.output_dir / 'best_model.pt')
            test_metrics = self.validate(self.test_loader, 'test')
            self.history['test'].append(test_metrics)
            print(f"\nTest Results:")
            for k, v in test_metrics.items():
                print(f"  {k}: {v:.4f}")

        # Save history
        with open(self.output_dir / 'training_history.json', 'w') as f:
            json.dump(self.history, f, indent=2)

        return self.history

    def save_checkpoint(self, filename):
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'best_val_acc': self.best_val_acc,
        }, self.output_dir / filename)

    def load_checkpoint(self, path):
        checkpoint = torch.load(path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        return self.model
