"""
Ablation model variants for ArchaeoGPT.
Each variant removes one component to measure its contribution.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from models.archaeogpt import (
    MultiScaleVisionEncoder, ArchaeologicalTextEncoder,
    CrossModalFusion, GenerativeReasoningHead, ArchaeoGPT
)


def make_ablation_variants(num_cultures=58, num_types=50, num_eras=7):
    """Create all ablation model variants."""
    variants = {}

    # Variant A: ArchaeoGPT 0-step reasoning (direct classification, no GRU)
    class ArchaeoGPT_0Step(ArchaeoGPT):
        def __init__(self, *args, **kwargs):
            kwargs['num_reasoning_steps'] = 0
            super().__init__(*args, **kwargs)
            # Replace reasoning head with direct classifier
            hidden_dim = kwargs.get('hidden_dim', 768)
            self.culture_head = nn.Linear(hidden_dim, num_cultures)
            self.type_head = nn.Linear(hidden_dim, num_types)
            self.era_head = nn.Linear(hidden_dim, num_eras)

        def forward(self, images, culture_ids=None, type_ids=None, era_ids=None):
            B = images.shape[0]
            visual_features = self.vision_encoder(images)
            parts = [visual_features['global']]
            if 'local' in visual_features: parts.append(visual_features['local'])
            if 'mid' in visual_features: parts.append(visual_features['mid'])
            visual_feat = self.visual_aggregator(torch.cat(parts, dim=-1)) if len(parts)>1 else self.visual_aggregator(visual_features['global'])
            text_feat = self.text_encoder(culture_ids, type_ids, era_ids, batch_size=B)
            fused_feat = self.fusion(visual_feat, text_feat)
            return {
                'culture': self.culture_head(fused_feat),
                'type': self.type_head(fused_feat),
                'era': self.era_head(fused_feat),
                'features': fused_feat,
            }
    variants['archaeogpt_0step'] = ArchaeoGPT_0Step

    # Variant B: ArchaeoGPT 1-step reasoning
    class ArchaeoGPT_1Step(ArchaeoGPT):
        def __init__(self, *args, **kwargs):
            kwargs['num_reasoning_steps'] = 1
            super().__init__(*args, **kwargs)
    variants['archaeogpt_1step'] = ArchaeoGPT_1Step

    # Variant C: No gating (use simple concatenation)
    class NoGateFusion(nn.Module):
        def __init__(self, hidden_dim=768):
            super().__init__()
            self.proj = nn.Sequential(
                nn.Linear(hidden_dim * 2, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
            )
        def forward(self, visual_feat, text_feat):
            return self.proj(torch.cat([visual_feat, text_feat], dim=-1))

    class ArchaeoGPT_NoGate(ArchaeoGPT):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.fusion = NoGateFusion(kwargs.get('hidden_dim', 768))
    variants['archaeogpt_nogate'] = ArchaeoGPT_NoGate

    # Variant D: Single-scale (global only, no multi-scale)
    class SingleScaleEncoder(nn.Module):
        def __init__(self, backbone='vit_b_16', hidden_dim=768):
            super().__init__()
            from torchvision import models
            self.backbone = models.vit_b_16(weights='DEFAULT')
            self.backbone.heads = nn.Identity()
            self.proj = nn.Linear(768, hidden_dim)
        def forward(self, x):
            f = self.backbone(x)
            return {'global': self.proj(f)}

    class ArchaeoGPT_SingleScale(ArchaeoGPT):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.vision_encoder = SingleScaleEncoder(kwargs.get('backbone', 'vit_b_16'),
                                                      kwargs.get('hidden_dim', 768))
            self.visual_aggregator = nn.Sequential(
                nn.Linear(kwargs.get('hidden_dim', 768), kwargs.get('hidden_dim', 768)),
                nn.LayerNorm(kwargs.get('hidden_dim', 768)),
                nn.ReLU(),
            )
        def forward(self, images, culture_ids=None, type_ids=None, era_ids=None):
            B = images.shape[0]
            visual_features = self.vision_encoder(images)
            visual_feat = self.visual_aggregator(visual_features['global'])
            text_feat = self.text_encoder(culture_ids, type_ids, era_ids, batch_size=B)
            fused_feat = self.fusion(visual_feat, text_feat)
            predictions = self.reasoning_head(fused_feat)
            return predictions
    variants['archaeogpt_singlescale'] = ArchaeoGPT_SingleScale

    # Variant E: ViT + deeper MLP (same depth as reasoning head, but no GRU)
    class ViTDeepMLP(nn.Module):
        def __init__(self, num_cultures, num_types, num_eras):
            super().__init__()
            from torchvision import models
            self.backbone = models.vit_b_16(weights='DEFAULT')
            hidden_dim = self.backbone.heads.head.in_features
            self.backbone.heads = nn.Identity()
            self.mlp = nn.Sequential(
                nn.Linear(hidden_dim, 1024), nn.ReLU(), nn.Dropout(0.3),
                nn.Linear(1024, 1024), nn.ReLU(), nn.Dropout(0.3),
                nn.Linear(1024, 512), nn.ReLU(), nn.Dropout(0.3),
            )
            self.culture_head = nn.Linear(512, num_cultures)
            self.type_head = nn.Linear(512, num_types)
            self.era_head = nn.Linear(512, num_eras)

        def forward(self, images, **kwargs):
            f = self.backbone(images)
            f = self.mlp(f)
            return {'culture': self.culture_head(f), 'type': self.type_head(f),
                    'era': self.era_head(f), 'features': f}
    variants['vit_deep_mlp'] = ViTDeepMLP

    return variants
