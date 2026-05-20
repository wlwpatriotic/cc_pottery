"""
Reasoning trace visualization for ArchaeoGPT.
Shows how multi-step reasoning refines predictions.
Even without text generation, we can visualize:
1. Feature trajectory across reasoning steps
2. Prediction confidence changes per step
3. Attention maps at each step
"""
import torch
import torch.nn.functional as F
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.decomposition import PCA


def visualize_reasoning_trajectory(model, images, labels, label_info, save_path):
    """
    Track how predictions evolve across reasoning steps.
    Uses PCA to visualize feature trajectory.
    """
    model.eval()
    device = next(model.parameters()).device
    images = images.to(device)

    with torch.no_grad():
        # Get visual features
        visual_features = model.vision_encoder(images)
        B = images.shape[0]

        # Aggregate
        if 'mid' in visual_features:
            visual_feat = torch.cat([visual_features['local'],
                                     visual_features['mid'],
                                     visual_features['global']], dim=-1)
        else:
            visual_feat = torch.cat([visual_features['local'],
                                     visual_features['global']], dim=-1)
        visual_feat = model.visual_aggregator(visual_feat)

        # Text features (zero in current setup)
        text_feat = model.text_encoder(None, None, None, batch_size=B)
        fused_feat = model.fusion(visual_feat, text_feat)

        # Run reasoning step by step
        state = fused_feat
        step_outputs = []
        step_preds = []
        step_confs = []

        for step in range(model.reasoning_head.num_reasoning_steps):
            query = model.reasoning_head.step_queries[step].unsqueeze(0).expand(B, -1)
            state_input = state + query
            state = model.reasoning_head.state_encoder(state_input, state)

            step_feat = model.reasoning_head.step_predictors[step](state)

            # Intermediate culture prediction
            culture_logits = model.reasoning_head.culture_classifier(step_feat)
            culture_probs = F.softmax(culture_logits, dim=-1)

            step_outputs.append(step_feat.cpu().numpy())
            step_preds.append(culture_logits.argmax(dim=-1).cpu().numpy())
            step_confs.append(culture_probs.max(dim=-1)[0].cpu().numpy())

        # Final prediction
        final_features = sum(step_outputs) / len(step_outputs)
        final_preds = model.reasoning_head.culture_classifier(
            torch.tensor(final_features, device=device)).argmax(dim=-1).cpu().numpy()

    # Generate visualization
    fig, axes = plt.subplots(2, 3, figsize=(14, 8))

    # Row 1: Feature trajectory (PCA)
    all_features = np.stack([fused_feat.cpu().numpy()] + step_outputs)  # (steps+1, B, D)
    n_steps = all_features.shape[0]
    all_flat = all_features.reshape(-1, all_features.shape[-1])
    pca = PCA(n_components=2)
    all_2d = pca.fit_transform(all_flat)
    all_2d = all_2d.reshape(n_steps, B, 2)

    idx_to_culture = label_info['idx_to_culture']
    culture_labels = labels['culture'].cpu().numpy()

    for b in range(min(B, 3)):  # Show 3 samples
        ax = axes[0, b]
        for s in range(n_steps):
            color = plt.cm.viridis(s / n_steps)
            ax.scatter(all_2d[s, b, 0], all_2d[s, b, 1],
                      color=[color], s=100, marker='o',
                      label=f'Step {s}' if b == 0 else '')
            if s > 0:
                ax.arrow(all_2d[s-1, b, 0], all_2d[s-1, b, 1],
                        all_2d[s, b, 0] - all_2d[s-1, b, 0],
                        all_2d[s, b, 1] - all_2d[s-1, b, 1],
                        head_width=0.05, color='gray', alpha=0.5)
        true_cls = idx_to_culture.get(culture_labels[b], f'cls_{culture_labels[b]}')
        pred_cls = idx_to_culture.get(final_preds[b], f'cls_{final_preds[b]}')
        ax.set_title(f'Sample {b+1}\nTrue: {true_cls[:8]}\nPred: {pred_cls[:8]}', fontsize=8)
        if b == 0:
            ax.legend(fontsize=6)

    # Row 2: Confidence evolution per step
    for b in range(min(B, 3)):
        ax = axes[1, b]
        confs = [step_confs[s][b] for s in range(len(step_confs))]
        ax.plot(range(1, len(confs)+1), confs, 'o-', linewidth=2, markersize=8)
        ax.set_xlabel('Reasoning Step')
        ax.set_ylabel('Confidence')
        ax.set_title(f'Sample {b+1} Confidence')
        ax.set_ylim(0, 1.05)
        ax.grid(alpha=0.3)

    plt.suptitle('ArchaeoGPT Multi-Step Reasoning Trajectory', fontweight='bold')
    plt.tight_layout()
    plt.savefig(save_path, dpi=200)
    plt.close()
    print(f"Saved reasoning visualization to {save_path}")


def visualize_feature_space(model, dataloader, label_info, save_path, n_samples=200):
    """PCA visualization of learned feature space, colored by culture."""
    model.eval()
    device = next(model.parameters()).device

    all_features = []
    all_cultures = []
    count = 0
    for images, labels, _ in dataloader:
        if count >= n_samples:
            break
        images = images.to(device)
        with torch.no_grad():
            outputs = model(images)
        features = outputs['features'].cpu().numpy()
        cultures = labels['culture'].cpu().numpy()
        all_features.append(features)
        all_cultures.append(cultures)
        count += len(images)

    all_features = np.concatenate(all_features)[:n_samples]
    all_cultures = np.concatenate(all_cultures)[:n_samples]

    pca = PCA(n_components=2)
    features_2d = pca.fit_transform(all_features)

    fig, ax = plt.subplots(figsize=(10, 8))
    idx_to_culture = label_info['idx_to_culture']

    # Top 10 cultures with distinct colors
    from collections import Counter
    culture_counts = Counter(all_cultures)
    top_cultures = [c for c, _ in culture_counts.most_common(10)]

    for c in top_cultures:
        mask = all_cultures == c
        name = idx_to_culture.get(c, f'cls_{c}')
        ax.scatter(features_2d[mask, 0], features_2d[mask, 1],
                  label=name[:12], alpha=0.6, s=20)

    ax.legend(fontsize=7, loc='upper right')
    ax.set_title('Feature Space PCA (Top 10 Cultures)')
    plt.tight_layout()
    plt.savefig(save_path, dpi=200)
    plt.close()
    print(f"Saved feature visualization to {save_path}")
