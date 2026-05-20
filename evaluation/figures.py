"""
Figure generation for pottery recognition paper.
Creates publication-quality visualizations.
"""
import json
import numpy as np
from pathlib import Path
from collections import Counter
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

# Set publication style
plt.rcParams.update({
    'font.family': 'serif',
    'font.size': 10,
    'axes.labelsize': 11,
    'axes.titlesize': 12,
    'legend.fontsize': 8,
    'figure.dpi': 150,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
})


def fig_dataset_distribution():
    """Figure 2: Dataset distribution - cultures, types, eras."""
    import sys
    sys.path.insert(0, str(Path(r'f:\考古\cc_pottery')))
    from utils.dataset import PotteryDataset

    ds = PotteryDataset(r'f:\考古\cc_pottery', 'all', min_samples_per_class=5)
    dist = ds.get_class_distribution()

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    # Culture distribution (top 15)
    cultures = list(dist['culture_distribution'].items())[:15]
    axes[0].barh([c[0] for c in cultures], [c[1] for c in cultures],
                 color='steelblue', edgecolor='white')
    axes[0].set_xlabel('Number of Samples')
    axes[0].set_title('Top 15 Archaeological Cultures')
    axes[0].invert_yaxis()

    # Type distribution
    types = list(dist['type_distribution'].items())[:10]
    axes[1].bar([t[0] for t in types], [t[1] for t in types],
                color='coral', edgecolor='white')
    axes[1].set_xlabel('Artifact Type')
    axes[1].set_ylabel('Number of Samples')
    axes[1].set_title('Top 10 Artifact Types')
    axes[1].tick_params(axis='x', rotation=45)

    # Era distribution
    eras = list(dist['era_distribution'].items())
    colors = plt.cm.Set3(np.linspace(0, 1, len(eras)))
    axes[2].pie([e[1] for e in eras], labels=[e[0] for e in eras],
                autopct='%1.1f%%', colors=colors)
    axes[2].set_title('Chronological Distribution')

    plt.suptitle('Painted Pottery Dataset Statistics', fontweight='bold', y=1.02)
    plt.tight_layout()

    save_path = Path(r'f:\考古\cc_pottery\outputs\fig_dataset.pdf')
    save_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path)
    plt.close()
    print(f"Saved {save_path}")


def fig_method_comparison(results_files):
    """Figure 3: Method comparison bar chart."""
    if not results_files:
        print("No results to compare yet")
        return

    methods = []
    culture_acc = []
    type_acc = []

    for method_name, filepath in results_files:
        if not Path(filepath).exists():
            continue
        with open(filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)

        methods.append(method_name)
        tm = data.get('final_test_metrics', {})
        culture_acc.append(tm.get('test_culture_acc', 0))
        type_acc.append(tm.get('test_type_acc', 0))

    if not methods:
        return

    fig, ax = plt.subplots(figsize=(10, 5))
    x = np.arange(len(methods))
    width = 0.35

    bars1 = ax.bar(x - width/2, culture_acc, width, label='Culture Accuracy',
                   color='steelblue', edgecolor='white')
    bars2 = ax.bar(x + width/2, type_acc, width, label='Type Accuracy',
                   color='coral', edgecolor='white')

    ax.set_ylabel('Accuracy')
    ax.set_title('Method Comparison on Pottery Recognition')
    ax.set_xticks(x)
    ax.set_xticklabels(methods, rotation=30)
    ax.legend()
    ax.set_ylim(0, 1.0)
    ax.grid(axis='y', alpha=0.3)

    # Add value labels
    for bar in bars1:
        height = bar.get_height()
        ax.annotate(f'{height:.3f}', xy=(bar.get_x() + bar.get_width()/2, height),
                   xytext=(0, 3), textcoords="offset points", ha='center', fontsize=8)
    for bar in bars2:
        height = bar.get_height()
        ax.annotate(f'{height:.3f}', xy=(bar.get_x() + bar.get_width()/2, height),
                   xytext=(0, 3), textcoords="offset points", ha='center', fontsize=8)

    plt.tight_layout()
    save_path = Path(r'f:\考古\cc_pottery\outputs\fig_comparison.pdf')
    plt.savefig(save_path)
    plt.close()
    print(f"Saved {save_path}")


def fig_ablation(results_files):
    """Figure 4: Ablation study results."""
    if not results_files:
        print("No ablation results yet")
        return

    fig, ax = plt.subplots(figsize=(8, 5))

    variants = []
    accs = []
    for name, filepath in results_files:
        if not Path(filepath).exists():
            continue
        with open(filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)
        variants.append(name)
        accs.append(data.get('best_val_acc', 0))

    colors = plt.cm.RdYlGn(np.linspace(0.2, 0.8, len(variants)))
    bars = ax.barh(variants, accs, color=colors, edgecolor='white')

    ax.set_xlabel('Best Validation Accuracy (Culture)')
    ax.set_title('Ablation Study')

    for bar, acc in zip(bars, accs):
        ax.text(bar.get_width() + 0.005, bar.get_y() + bar.get_height()/2,
                f'{acc:.4f}', va='center', fontsize=9)

    plt.tight_layout()
    save_path = Path(r'f:\考古\cc_pottery\outputs\fig_ablation.pdf')
    plt.savefig(save_path)
    plt.close()
    print(f"Saved {save_path}")


def fig_confusion_matrix(cm, class_names, title, save_path):
    """Plot confusion matrix."""
    fig, ax = plt.subplots(figsize=(12, 10))
    im = ax.imshow(cm, cmap='YlOrRd', aspect='auto')

    n = len(class_names)
    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    ax.set_xticklabels(class_names, rotation=90, fontsize=5)
    ax.set_yticklabels(class_names, fontsize=5)
    ax.set_xlabel('Predicted')
    ax.set_ylabel('True')
    ax.set_title(title)

    plt.colorbar(im)
    plt.tight_layout()
    plt.savefig(save_path, dpi=200)
    plt.close()


def fig_training_curves(history_file, save_path):
    """Plot training and validation curves."""
    if not Path(history_file).exists():
        print(f"History file not found: {history_file}")
        return

    with open(history_file, 'r', encoding='utf-8') as f:
        history = json.load(f)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    # Accuracy curves
    train_acc = [e.get('culture_acc', 0) for e in history.get('train', [])]
    val_epochs = [e['epoch'] for e in history.get('val', [])]
    val_acc = [e.get('val_culture_acc', 0) for e in history.get('val', [])]

    axes[0].plot(train_acc, label='Train', alpha=0.6)
    axes[0].plot(val_epochs, val_acc, label='Val', marker='o', markersize=3)
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Culture Accuracy')
    axes[0].set_title('Training Progress')
    axes[0].legend()
    axes[0].grid(alpha=0.3)

    # Loss curves
    train_loss = [e.get('total', 0) for e in history.get('train', [])]
    axes[1].plot(train_loss, alpha=0.6)
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Loss')
    axes[1].set_title('Training Loss')
    axes[1].grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    print(f"Saved {save_path}")


if __name__ == '__main__':
    fig_dataset_distribution()
    print("Dataset figures generated.")

    # Check for existing results
    outputs = Path(r'f:\考古\cc_pottery\outputs')
    results = list(outputs.glob('*/results.json'))
    if results:
        results_files = [(r.parent.name, r) for r in results]
        fig_method_comparison(results_files)
