"""
Comprehensive evaluation pipeline with error analysis.
Tracks per-class metrics, confusion matrices, and hard case mining.
"""
import json
import numpy as np
import torch
import torch.nn.functional as F
from collections import defaultdict
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path


class EvaluationPipeline:
    """Comprehensive evaluation with hierarchical error analysis."""

    def __init__(self, model, dataloader, label_info, device='cuda', output_dir='./outputs'):
        self.model = model.to(device).eval()
        self.dataloader = dataloader
        self.label_info = label_info
        self.device = device
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.results = {}
        self.errors = []  # detailed error analysis per sample

    @torch.no_grad()
    def evaluate(self):
        """Run full evaluation."""
        all_preds = defaultdict(list)
        all_labels = defaultdict(list)
        all_confidences = defaultdict(list)
        all_features = []
        all_metas = []

        for images, labels, metas in self.dataloader:
            images = images.to(self.device)
            labels_gpu = {k: v.to(self.device) for k, v in labels.items()}

            outputs = self.model(images)  # NEVER pass labels during evaluation

            for task in ['culture', 'type', 'era']:
                if task not in outputs:
                    continue
                logits = outputs[task]
                probs = F.softmax(logits, dim=-1)
                preds = logits.argmax(dim=-1)

                all_preds[task].extend(preds.cpu().numpy())
                all_labels[task].extend(labels[task].cpu().numpy())
                all_confidences[task].extend(probs.max(dim=-1)[0].cpu().numpy())

            if 'features' in outputs:
                all_features.append(outputs['features'].cpu().numpy())

            all_metas.extend(metas)

        # Compute metrics
        for task in ['culture', 'type', 'era']:
            if task not in all_preds:
                continue
            self._compute_task_metrics(task, all_preds[task], all_labels[task],
                                       all_confidences[task], all_metas)

        # Cross-analysis: culture-type confusion
        self._cross_task_analysis(all_preds, all_labels, all_metas)

        return self.results

    def _compute_task_metrics(self, task, preds, labels, confidences, metas):
        """Compute per-class and overall metrics for a task."""
        class_names = {v: k for k, v in self.label_info[f'{task}_to_idx'].items()}
        preds_arr = np.array(preds)
        labels_arr = np.array(labels)
        unique_classes = sorted(set(labels))

        # Overall accuracy (manual computation, more reliable than sklearn)
        accuracy = float((preds_arr == labels_arr).mean())

        # Per-class metrics
        per_class_acc = {}
        per_class_f1 = {}
        for cls in unique_classes:
            mask = labels_arr == cls
            if mask.sum() > 0:
                tp = (preds_arr[mask] == cls).sum()
                recall = tp / mask.sum()
                pred_mask = preds_arr == cls
                precision = tp / max(1, pred_mask.sum())
                f1 = 2 * precision * recall / max(1e-8, precision + recall)
                per_class_acc[class_names.get(cls, cls)] = float(recall)
                per_class_f1[class_names.get(cls, cls)] = float(f1)

        macro_f1 = float(np.mean(list(per_class_f1.values()))) if per_class_f1 else 0.0

        # Weighted F1
        total = len(labels_arr)
        weighted_f1 = float(sum(
            per_class_f1.get(class_names.get(cls, cls), 0) * (labels_arr == cls).sum()
            for cls in unique_classes
        ) / max(1, total))

        # Confusion matrix
        cm = confusion_matrix(labels, preds, labels=unique_classes)

        # Top confusions
        top_confusions = self._get_top_confusions(cm, unique_classes, class_names)

        # Hard cases
        hard_cases = []
        for i, (p, l, conf, meta) in enumerate(zip(preds, labels, confidences, metas)):
            if p != l:
                hard_cases.append({
                    'uid': meta['uid'],
                    'true': class_names.get(l, f'class_{l}'),
                    'predicted': class_names.get(p, f'class_{p}'),
                    'confidence': float(conf),
                    'description': meta['description'][:200],
                })
        hard_cases.sort(key=lambda x: x['confidence'], reverse=True)

        self.results[task] = {
            'accuracy': accuracy,
            'macro_f1': macro_f1,
            'weighted_f1': weighted_f1,
            'per_class_accuracy': per_class_acc,
            'top_confusions': top_confusions[:15],
            'hard_cases': hard_cases[:20],
            'confusion_matrix': cm.tolist(),
            'num_classes': len(unique_classes),
        }

        print(f"\n{'='*50}")
        print(f"Task: {task}")
        print(f"{'='*50}")
        print(f"  Accuracy: {accuracy:.4f}")
        print(f"  Macro F1: {macro_f1:.4f}")
        print(f"  Weighted F1: {weighted_f1:.4f}")
        print(f"\n  Top Confusions:")
        for (true, pred, count) in self.results[task]['top_confusions'][:5]:
            print(f"    {true} -> {pred}: {count}x")
        print(f"\n  Hard Cases (top-3):")
        for case in self.results[task]['hard_cases'][:3]:
            print(f"    [{case['uid']}] {case['true']} -> {case['predicted']} "
                  f"(conf={case['confidence']:.3f})")

    def _get_top_confusions(self, cm, classes, class_names):
        """Extract top confusion pairs from confusion matrix."""
        confusions = []
        for i in range(len(classes)):
            for j in range(len(classes)):
                if i != j and cm[i][j] > 0:
                    confusions.append((
                        class_names.get(classes[i], f'c{classes[i]}'),
                        class_names.get(classes[j], f'c{classes[j]}'),
                        int(cm[i][j]),
                    ))
        confusions.sort(key=lambda x: x[2], reverse=True)
        return confusions

    def _cross_task_analysis(self, all_preds, all_labels, all_metas):
        """Analyze how culture misclassification affects type prediction."""
        if 'culture' not in all_preds or 'type' not in all_preds:
            return

        culture_preds = np.array(all_preds['culture'])
        culture_labels = np.array(all_labels['culture'])
        type_preds = np.array(all_preds['type'])
        type_labels = np.array(all_labels['type'])

        # When culture is correct, what's type accuracy?
        culture_correct = culture_preds == culture_labels
        culture_wrong = ~culture_correct

        type_acc_given_culture_correct = (
            (type_preds[culture_correct] == type_labels[culture_correct]).mean()
            if culture_correct.sum() > 0 else 0
        )
        type_acc_given_culture_wrong = (
            (type_preds[culture_wrong] == type_labels[culture_wrong]).mean()
            if culture_wrong.sum() > 0 else 0
        )

        self.results['cross_analysis'] = {
            'type_acc_given_culture_correct': float(type_acc_given_culture_correct),
            'type_acc_given_culture_wrong': float(type_acc_given_culture_wrong),
            'culture_correct_ratio': float(culture_correct.mean()),
        }

        print(f"\n{'='*50}")
        print(f"Cross-Task Analysis")
        print(f"{'='*50}")
        print(f"  Culture correct: {culture_correct.mean():.3f}")
        print(f"  Type acc (culture correct): {type_acc_given_culture_correct:.4f}")
        print(f"  Type acc (culture wrong):   {type_acc_given_culture_wrong:.4f}")

    def plot_confusion_matrix(self, task, top_k=20, save=True):
        """Plot confusion matrix for top-K classes."""
        if task not in self.results or 'confusion_matrix' not in self.results[task]:
            return

        cm = np.array(self.results[task]['confusion_matrix'])
        class_names = {v: k for k, v in self.label_info[f'{task}_to_idx'].items()}

        # Sort by total samples
        class_totals = cm.sum(axis=1)
        top_classes = np.argsort(class_totals)[-top_k:]

        cm_subset = cm[top_classes][:, top_classes]
        names = [class_names.get(i, f'c{i}') for i in top_classes]

        fig, ax = plt.subplots(figsize=(12, 10))
        im = ax.imshow(cm_subset, cmap='YlOrRd', aspect='auto')

        ax.set_xticks(range(len(names)))
        ax.set_yticks(range(len(names)))
        ax.set_xticklabels(names, rotation=90, fontsize=6)
        ax.set_yticklabels(names, fontsize=6)
        ax.set_xlabel('Predicted')
        ax.set_ylabel('True')
        ax.set_title(f'{task} Confusion Matrix (Top {top_k} Classes)')

        plt.colorbar(im)
        plt.tight_layout()

        if save:
            save_path = self.output_dir / f'confusion_{task}.png'
            plt.savefig(save_path, dpi=150)
            print(f"  Saved confusion matrix to {save_path}")

        plt.close()

    def save_results(self):
        """Save all evaluation results."""
        save_path = self.output_dir / 'evaluation_results.json'
        with open(save_path, 'w', encoding='utf-8') as f:
            json.dump(self.results, f, indent=2, ensure_ascii=False, default=str)
        print(f"\nResults saved to {save_path}")

    def generate_error_report(self):
        """Generate a human-readable error analysis report."""
        lines = []
        lines.append("=" * 70)
        lines.append("ERROR ANALYSIS REPORT")
        lines.append("=" * 70)

        for task in ['culture', 'type', 'era']:
            if task not in self.results:
                continue
            r = self.results[task]
            lines.append(f"\n## {task.upper()} Classification")
            lines.append(f"  Accuracy: {r['accuracy']:.4f}")
            lines.append(f"  Macro F1: {r['macro_f1']:.4f}")

            # Most confused pairs
            lines.append(f"\n  Top confusion pairs:")
            for true, pred, count in r['top_confusions'][:10]:
                lines.append(f"    {true} → {pred}: {count}x")

            # Worst-performing classes
            worst = sorted(r['per_class_accuracy'].items(), key=lambda x: x[1])[:5]
            lines.append(f"\n  Worst-performing classes:")
            for cls, acc in worst:
                lines.append(f"    {cls}: {acc:.3f}")

        # Cross-task analysis
        if 'cross_analysis' in self.results:
            ca = self.results['cross_analysis']
            lines.append(f"\n## Cross-Task Analysis")
            lines.append(f"  Culture correct rate: {ca['culture_correct_ratio']:.3f}")
            lines.append(f"  Type acc (culture correct): {ca['type_acc_given_culture_correct']:.4f}")
            lines.append(f"  Type acc (culture wrong): {ca['type_acc_given_culture_wrong']:.4f}")
            lines.append(f"  Drop: {(ca['type_acc_given_culture_correct'] - ca['type_acc_given_culture_wrong']):.4f}")

        report = '\n'.join(lines)
        report_path = self.output_dir / 'error_analysis.txt'
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write(report)

        print(f"\nError report saved to {report_path}")
        return report
