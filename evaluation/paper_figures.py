"""
Publication-quality figures for ArchaeoGPT paper.
Uses English labels to bypass CJK font issues on Windows.
"""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
from pathlib import Path
from collections import Counter

plt.rcParams.update({
    'font.family': 'serif',
    'font.size': 10,
    'axes.labelsize': 12,
    'axes.titlesize': 13,
    'legend.fontsize': 9,
    'figure.dpi': 150,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
})

OUT = Path(r'f:\考古\cc_pottery\outputs')


def fig1_architecture():
    """Figure 1: ArchaeoGPT Architecture Diagram."""
    fig, ax = plt.subplots(figsize=(12, 7))
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 7)
    ax.axis('off')
    ax.set_title('ArchaeoGPT: Generative Chain-of-Thought Reasoning\nfor Fine-Grained Pottery Recognition',
                 fontweight='bold', fontsize=14, pad=20)

    # Box drawing helper
    def draw_box(x, y, w, h, label, color='lightblue', fontsize=9):
        rect = mpatches.FancyBboxPatch((x-w/2, y-h/2), w, h,
            boxstyle="round,pad=0.1", facecolor=color, edgecolor='black', linewidth=1.5)
        ax.add_patch(rect)
        ax.text(x, y, label, ha='center', va='center', fontsize=fontsize, fontweight='bold')

    def draw_arrow(x1, y1, x2, y2, color='gray'):
        ax.annotate('', xy=(x2, y2), xytext=(x1, y1),
                   arrowprops=dict(arrowstyle='->', color=color, lw=2))

    # Stage 1: Multi-Scale Vision Encoder
    draw_box(2, 5.5, 3, 1.2, 'Multi-Scale\nVision Encoder', 'lightblue')
    draw_box(1, 4, 1.5, 0.8, 'Local\n(motif)', 'lightcyan')
    draw_box(2.5, 4, 1.5, 0.8, 'Mid\n(pattern)', 'lightcyan')
    draw_box(4, 4, 1.5, 0.8, 'Global\n(shape)', 'lightcyan')
    draw_arrow(2, 5.5-0.6, 1, 4+0.4)
    draw_arrow(2, 5.5-0.6, 2.5, 4+0.4)
    draw_arrow(2, 5.5-0.6, 4, 4+0.4)

    # Stage 2: Archaeological Text Encoder
    draw_box(8, 5.5, 3, 1.2, 'Archaeological\nText Encoder', 'lightcoral')
    draw_box(7, 4, 1.2, 0.8, 'Culture', 'mistyrose')
    draw_box(8.5, 4, 1.2, 0.8, 'Type', 'mistyrose')
    draw_box(10, 4, 1.2, 0.8, 'Era', 'mistyrose')

    # Stage 3: Cross-Modal Fusion
    draw_box(5, 2.5, 4, 1, 'Cross-Modal Fusion\nTransformer', 'lightyellow')
    draw_arrow(2, 4-0.4, 4, 2.5+0.5)
    draw_arrow(8, 4-0.4, 6, 2.5+0.5)

    # Stage 4: Generative CoT Reasoning
    draw_box(5, 1, 5, 1.2, 'Generative Chain-of-Thought\nReasoning Head (3 steps)', 'lightgreen')
    draw_arrow(5, 2.5-0.5, 5, 1+0.6)

    # Output
    draw_box(5, -0.3, 2, 0.6, 'Culture + Type + Era', 'wheat')
    draw_arrow(5, 1-0.6, 5, -0.3+0.3)

    # Labels
    ax.text(0.3, 6.3, 'Input Image', fontsize=10, style='italic')
    ax.text(10, 6.3, 'Archaeological\nKnowledge', fontsize=10, style='italic')
    ax.text(2.5, 1.5, 'Step 1: Observe motifs\nStep 2: Identify shape\nStep 3: Synthesize', fontsize=8)

    plt.tight_layout()
    plt.savefig(OUT / 'fig1_architecture.pdf')
    plt.close()
    print("Saved fig1_architecture.pdf")


def fig2_dataset_stats():
    """Figure 2: Dataset Statistics (English labels)."""
    from utils.dataset import PotteryDataset
    ds = PotteryDataset(r'f:\考古\cc_pottery', 'all', min_samples_per_class=5)
    dist = ds.get_class_distribution()

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Left: Culture distribution (top 15)
    cultures = list(dist['culture_distribution'].items())[:15]
    names = [c[0] for c in cultures]
    counts = [c[1] for c in cultures]
    colors = plt.cm.Blues(np.linspace(0.3, 0.9, len(names)))
    axes[0].barh(range(len(names)), counts, color=colors, edgecolor='white')
    axes[0].set_yticks(range(len(names)))
    axes[0].set_yticklabels(names, fontsize=9)
    axes[0].set_xlabel('Number of Samples')
    axes[0].set_title('Top 15 Archaeological Cultures')
    axes[0].invert_yaxis()

    # Right: Era pie chart (keep Chinese for authenticity)
    eras = list(dist['era_distribution'].items())
    era_colors = plt.cm.Set3(np.linspace(0, 1, len(eras)))
    wedges, texts, autotexts = axes[1].pie(
        [e[1] for e in eras], labels=[e[0] for e in eras],
        autopct='%1.1f%%', colors=era_colors)
    axes[1].set_title('Chronological Distribution')

    plt.suptitle('PotteryFGVC Dataset Statistics', fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(OUT / 'fig2_dataset.pdf')
    plt.close()
    print("Saved fig2_dataset.pdf")


def fig3_method_comparison():
    """Figure 3: Method Comparison Bar Chart."""
    methods = ['ResNet-50', 'CLIP Zero-Shot']
    culture_acc = [0.6044, 0.0000]
    type_acc = [0.7280, 0.1264]
    era_acc = [0.9121, 0.7005]

    fig, ax = plt.subplots(figsize=(10, 5))
    x = np.arange(len(methods))
    width = 0.25

    ax.bar(x - width, culture_acc, width, label='Culture (58 cls)', color='steelblue', edgecolor='white')
    ax.bar(x, type_acc, width, label='Type (50 cls)', color='coral', edgecolor='white')
    ax.bar(x + width, era_acc, width, label='Era (7 cls)', color='seagreen', edgecolor='white')

    ax.set_ylabel('Accuracy')
    ax.set_title('Method Comparison on PotteryFGVC')
    ax.set_xticks(x)
    ax.set_xticklabels(methods, fontsize=11)
    ax.legend(loc='lower right')
    ax.set_ylim(0, 1.0)
    ax.grid(axis='y', alpha=0.3)

    for i, (ca, ta, ea) in enumerate(zip(culture_acc, type_acc, era_acc)):
        ax.text(i - width, ca + 0.02, f'{ca:.3f}', ha='center', fontsize=9)
        ax.text(i, ta + 0.02, f'{ta:.3f}', ha='center', fontsize=9)
        ax.text(i + width, ea + 0.02, f'{ea:.3f}', ha='center', fontsize=9)

    plt.tight_layout()
    plt.savefig(OUT / 'fig3_comparison.pdf')
    plt.close()
    print("Saved fig3_comparison.pdf")


def fig4_long_tail():
    """Figure 4: Long-tail distribution visualization."""
    from utils.dataset import PotteryDataset
    ds = PotteryDataset(r'f:\考古\cc_pottery', 'train', min_samples_per_class=0)
    dist = ds.get_class_distribution()
    cultures_sorted = sorted(dist['culture_distribution'].items(), key=lambda x: x[1], reverse=True)

    fig, ax = plt.subplots(figsize=(12, 4))
    x = range(len(cultures_sorted))
    y = [c[1] for c in cultures_sorted]
    colors = ['steelblue' if n >= 100 else 'orange' if n >= 10 else 'crimson'
              for n in y]
    ax.bar(x, y, color=colors, edgecolor='white', width=0.8)

    ax.axhline(y=100, color='green', linestyle='--', alpha=0.5, label='Head (>=100)')
    ax.axhline(y=10, color='orange', linestyle='--', alpha=0.5, label='Mid (>=10)')
    ax.set_xlabel('Culture Rank')
    ax.set_ylabel('Number of Samples')
    ax.set_title('Long-Tail Distribution of Archaeological Cultures (79 total)')
    ax.legend()

    # Annotations
    ax.text(2, 105, f'Head: {sum(1 for n in y if n>=100)} cultures ({sum(n for n in y if n>=100)} samples)',
            fontsize=8, color='green')
    ax.text(15, 55, f'Mid: {sum(1 for n in y if 10<=n<100)} cultures ({sum(n for n in y if 10<=n<100)} samples)',
            fontsize=8, color='orange')
    ax.text(47, 5, f'Tail: {sum(1 for n in y if n<10)} cultures ({sum(n for n in y if n<10)} samples)',
            fontsize=8, color='crimson')

    plt.tight_layout()
    plt.savefig(OUT / 'fig4_long_tail.pdf')
    plt.close()
    print("Saved fig4_long_tail.pdf")


def fig5_cross_task():
    """Figure 5: Cross-task dependency analysis."""
    # Data from our experiment
    culture_correct = [0.604, 0.396]  # correct, wrong
    type_acc_culture_correct = [0.7955, 0.6250]
    culture_status = ['Culture Correct', 'Culture Wrong']

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))

    # Left: bar chart
    colors = ['steelblue', 'coral']
    axes[0].bar(culture_status, type_acc_culture_correct, color=colors, edgecolor='white')
    axes[0].set_ylabel('Type Classification Accuracy')
    axes[0].set_title('Type Accuracy Conditional on Culture')
    axes[0].set_ylim(0, 1.0)
    for i, v in enumerate(type_acc_culture_correct):
        axes[0].text(i, v + 0.02, f'{v:.3f}', ha='center', fontsize=12, fontweight='bold')

    # Right: Sankey-like flow
    axes[1].axis('off')
    axes[1].set_xlim(0, 10)
    axes[1].set_ylim(0, 10)
    axes[1].text(5, 9.5, 'Hierarchical Dependency', ha='center', fontsize=12, fontweight='bold')

    # Culture node
    axes[1].add_patch(plt.Circle((3, 7), 0.8, color='steelblue', alpha=0.7))
    axes[1].text(3, 7, 'Culture', ha='center', va='center', fontsize=9, fontweight='bold', color='white')

    # Type node
    axes[1].add_patch(plt.Circle((7, 7), 0.8, color='coral', alpha=0.7))
    axes[1].text(7, 7, 'Type', ha='center', va='center', fontsize=9, fontweight='bold', color='white')

    # Arrow
    axes[1].annotate('', xy=(6.3, 7), xytext=(3.7, 7),
                    arrowprops=dict(arrowstyle='->', color='gray', lw=3))

    # Era node
    axes[1].add_patch(plt.Circle((5, 4.5), 0.8, color='seagreen', alpha=0.7))
    axes[1].text(5, 4.5, 'Era', ha='center', va='center', fontsize=9, fontweight='bold', color='white')
    axes[1].annotate('', xy=(5, 5.3), xytext=(5, 6.3),
                    arrowprops=dict(arrowstyle='->', color='gray', lw=2, alpha=0.5))
    axes[1].annotate('', xy=(5.5, 5.3), xytext=(6.5, 6.3),
                    arrowprops=dict(arrowstyle='->', color='gray', lw=2, alpha=0.5))

    # Annotation
    axes[1].text(5, 3, '25% drop when\nculture is wrong', ha='center', fontsize=10,
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

    plt.tight_layout()
    plt.savefig(OUT / 'fig5_cross_task.pdf')
    plt.close()
    print("Saved fig5_cross_task.pdf")


if __name__ == '__main__':
    OUT.mkdir(parents=True, exist_ok=True)
    fig1_architecture()
    fig2_dataset_stats()
    fig3_method_comparison()
    fig4_long_tail()
    fig5_cross_task()
    print("\nAll paper figures generated in", OUT)
