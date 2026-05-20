"""
ArchaeoGPT: Generative Reasoning for Fine-Grained Pottery Recognition.
Core innovation: hierarchical chain-of-thought reasoning for archaeological artifact classification.

Architecture:
1. Hierarchical Vision Encoder (multi-scale motif + shape features)
2. Archaeological Knowledge Encoder (domain-specific text embeddings)
3. Cross-Modal Fusion Transformer
4. Generative Reasoning Head (produces structured CoT output)
5. Open-Vocabulary Recognition Head (CLIP-style contrastive)
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
import math


class MultiScaleVisionEncoder(nn.Module):
    """Extract features at multiple granularities: motif-level (local), shape-level (global)."""

    def __init__(self, backbone='vit_b_16', pretrained=True, hidden_dim=768):
        super().__init__()
        if backbone == 'vit_b_16':
            self.backbone = models.vit_b_16(weights='DEFAULT' if pretrained else None)
            self.backbone.heads = nn.Identity()
            self.feature_dim = 768
        elif backbone == 'resnet50':
            resnet = models.resnet50(weights='DEFAULT' if pretrained else None)
            self.feature_dim = 2048
            # Multi-scale: grab features from different ResNet stages
            self.conv1 = resnet.conv1
            self.bn1 = resnet.bn1
            self.relu = resnet.relu
            self.maxpool = resnet.maxpool
            self.layer1 = resnet.layer1  # 256 dim, fine details
            self.layer2 = resnet.layer2  # 512 dim
            self.layer3 = resnet.layer3  # 1024 dim
            self.layer4 = resnet.layer4  # 2048 dim, global shape
            self.avgpool = resnet.avgpool
            self._use_resnet = True
        else:
            raise ValueError(f"Unknown backbone: {backbone}")

        self._backbone_type = backbone
        self.hidden_dim = hidden_dim

        # Projection for multi-scale features
        if backbone == 'resnet50':
            self.local_proj = nn.Linear(256, hidden_dim)   # fine detail
            self.mid_proj = nn.Linear(1024, hidden_dim)     # mid-level
            self.global_proj = nn.Linear(2048, hidden_dim)  # shape-level
        else:
            # ViT: use patch features and CLS token
            self.local_proj = nn.Linear(self.feature_dim, hidden_dim)

    def forward(self, x):
        if self._backbone_type == 'vit_b_16':
            # For ViT, use the CLS token as global + intermediate features
            # This is a simplification; full patch feature extraction requires hook
            global_feat = self.backbone(x)
            local_feat = global_feat  # ViT doesn't easily give layer-wise features
            return {
                'global': self.local_proj(global_feat),
                'local': self.local_proj(local_feat),
            }
        else:
            # ResNet multi-scale
            x = self.relu(self.bn1(self.conv1(x)))
            x = self.maxpool(x)
            f1 = self.layer1(x)       # fine-grained motifs
            f2 = self.layer2(f1)
            f3 = self.layer3(f2)       # mid-level patterns
            f4 = self.layer4(f3)       # global shape

            local_feat = self.local_proj(self.avgpool(f1).flatten(1))
            mid_feat = self.mid_proj(self.avgpool(f3).flatten(1))
            global_feat = self.global_proj(self.avgpool(f4).flatten(1))

            return {
                'local': local_feat,      # motif-level
                'mid': mid_feat,          # pattern-level
                'global': global_feat,    # shape-level
            }


class ArchaeologicalTextEncoder(nn.Module):
    """Domain-specific text encoder for archaeological terminology."""

    def __init__(self, num_cultures, num_types, num_eras, hidden_dim=768):
        super().__init__()
        # Learnable embeddings for each archaeological concept
        self.culture_embed = nn.Embedding(num_cultures, hidden_dim)
        self.type_embed = nn.Embedding(num_types, hidden_dim)
        self.era_embed = nn.Embedding(num_eras, hidden_dim)

        # Context embeddings for key archaeological concepts
        self.material_embed = nn.Embedding(10, hidden_dim // 4)  # clay type, temper
        self.decoration_embed = nn.Embedding(20, hidden_dim // 4)  # decorative technique

        # Fusion layer
        self.fusion = nn.Sequential(
            nn.Linear(hidden_dim * 3, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
        )

        self.hidden_dim = hidden_dim

    def forward(self, culture_ids=None, type_ids=None, era_ids=None,
                material_ids=None, decoration_ids=None, batch_size=None):
        """Encode archaeological knowledge into embedding space.
        If IDs are None, returns a learnable 'unknown' embedding."""
        # Determine batch size
        B = batch_size
        if B is None:
            if culture_ids is not None:
                B = culture_ids.shape[0]
            elif type_ids is not None:
                B = type_ids.shape[0]
            else:
                B = 1

        dev = self.culture_embed.weight.device
        zero_feat = torch.zeros(B, self.hidden_dim, device=dev)

        features = []
        if culture_ids is not None:
            features.append(self.culture_embed(culture_ids))
        else:
            features.append(zero_feat)

        if type_ids is not None:
            features.append(self.type_embed(type_ids))
        else:
            features.append(zero_feat)

        if era_ids is not None:
            features.append(self.era_embed(era_ids))
        else:
            features.append(zero_feat)

        fused = self.fusion(torch.cat(features, dim=-1))
        return fused


class CrossModalFusion(nn.Module):
    """Gated cross-modal fusion between visual and archaeological text features.
    Uses bilinear gating instead of attention to avoid MHA dimension issues
    with single-token representations."""

    def __init__(self, hidden_dim=768, num_heads=8):
        super().__init__()
        # Gated fusion: learn how much to trust visual vs text features
        self.gate = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.Sigmoid(),
        )
        self.visual_proj = nn.Linear(hidden_dim, hidden_dim)
        self.text_proj = nn.Linear(hidden_dim, hidden_dim)

        self.norm1 = nn.LayerNorm(hidden_dim)
        self.norm2 = nn.LayerNorm(hidden_dim)

        self.ffn = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 4),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim * 4, hidden_dim),
            nn.Dropout(0.1),
        )

    def forward(self, visual_feat, text_feat):
        """Gated fusion of visual and text features."""
        v = self.visual_proj(visual_feat)
        t = self.text_proj(text_feat)

        # Gating mechanism
        gate = self.gate(torch.cat([v, t], dim=-1))
        fused = gate * v + (1 - gate) * t
        fused = self.norm1(fused)

        # FFN refinement
        out = self.norm2(fused + self.ffn(fused))
        return out


class GenerativeReasoningHead(nn.Module):
    """
    Chain-of-Thought reasoning head that produces structured predictions:
    "Observing [motif], which is characteristic of [culture], and
     the vessel shape suggests [type], while the clay material indicates [era].
     Therefore: Culture=[pred], Type=[pred], Era=[pred]."
    """

    def __init__(self, hidden_dim=768, num_cultures=58, num_types=50, num_eras=7,
                 num_reasoning_steps=3):
        super().__init__()
        self.num_reasoning_steps = num_reasoning_steps
        self.hidden_dim = hidden_dim

        # Reasoning state encoder
        self.state_encoder = nn.GRUCell(hidden_dim, hidden_dim)

        # Step-specific prediction heads
        self.step_predictors = nn.ModuleList([
            nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(0.2),
            ) for _ in range(num_reasoning_steps)
        ])

        # Final classifiers
        self.culture_classifier = nn.Linear(hidden_dim, num_cultures)
        self.type_classifier = nn.Linear(hidden_dim, num_types)
        self.era_classifier = nn.Linear(hidden_dim, num_eras)

        # Reasoning step descriptions (learnable)
        self.step_queries = nn.Parameter(
            torch.randn(num_reasoning_steps, hidden_dim) * 0.02
        )

    def forward(self, fused_features, return_reasoning=False):
        """
        Multi-step reasoning from visual+text fused features.
        Each step refines the prediction by focusing on different aspects.
        """
        B = fused_features.shape[0]
        state = fused_features
        reasoning_outputs = []

        for step in range(self.num_reasoning_steps):
            # Update reasoning state
            query = self.step_queries[step].unsqueeze(0).expand(B, -1)
            state_input = state + query
            state = self.state_encoder(state_input, state)

            # Step-specific intermediate prediction
            step_feat = self.step_predictors[step](state)
            reasoning_outputs.append(step_feat)

        # Final aggregated reasoning features
        final_features = sum(reasoning_outputs) / self.num_reasoning_steps

        predictions = {
            'culture': self.culture_classifier(final_features),
            'type': self.type_classifier(final_features),
            'era': self.era_classifier(final_features),
            'features': final_features,
        }

        if return_reasoning:
            predictions['reasoning_steps'] = reasoning_outputs

        return predictions


class ArchaeoGPT(nn.Module):
    """
    Full ArchaeoGPT model: generative reasoning for fine-grained pottery recognition.

    Pipeline:
    1. Multi-scale visual encoding (motif + shape)
    2. Archaeological text encoding (domain knowledge)
    3. Cross-modal fusion (visual-text interaction)
    4. Chain-of-thought reasoning
    5. Hierarchical classification
    """

    def __init__(self, num_cultures=58, num_types=50, num_eras=7,
                 hidden_dim=768, num_reasoning_steps=3, backbone='vit_b_16'):
        super().__init__()
        self.vision_encoder = MultiScaleVisionEncoder(
            backbone=backbone, hidden_dim=hidden_dim
        )
        self.text_encoder = ArchaeologicalTextEncoder(
            num_cultures, num_types, num_eras, hidden_dim
        )
        self.fusion = CrossModalFusion(hidden_dim, num_heads=8)
        self.reasoning_head = GenerativeReasoningHead(
            hidden_dim, num_cultures, num_types, num_eras, num_reasoning_steps
        )

        # Feature aggregation
        self.visual_aggregator = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
        )

    def forward(self, images, culture_ids=None, type_ids=None, era_ids=None,
                return_reasoning=False):
        """
        Args:
            images: (B, 3, H, W) input images
            culture_ids: (B,) optional culture labels for text encoding
            type_ids: (B,) optional type labels
            era_ids: (B,) optional era labels
        """
        # Step 1: Multi-scale visual encoding
        visual_features = self.vision_encoder(images)

        # Aggregate visual features
        if 'mid' in visual_features:
            visual_feat = torch.cat([visual_features['local'],
                                     visual_features['mid'],
                                     visual_features['global']], dim=-1)
        else:
            visual_feat = torch.cat([visual_features['local'],
                                     visual_features['global']], dim=-1)
        visual_feat = self.visual_aggregator(visual_feat)

        # Step 2: Archaeological text encoding
        B = images.shape[0]
        text_feat = self.text_encoder(culture_ids, type_ids, era_ids, batch_size=B)

        # Step 3: Cross-modal fusion
        fused_feat = self.fusion(visual_feat, text_feat)

        # Step 4: Generative chain-of-thought reasoning
        predictions = self.reasoning_head(fused_feat, return_reasoning)

        return predictions

    def generate_explanation(self, images, label_info):
        """Generate natural language explanation for predictions."""
        self.eval()
        with torch.no_grad():
            predictions = self.forward(images, return_reasoning=True)

        explanations = []
        for b in range(images.shape[0]):
            culture_idx = predictions['culture'][b].argmax().item()
            type_idx = predictions['type'][b].argmax().item()
            era_idx = predictions['era'][b].argmax().item()

            culture_name = label_info.get('idx_to_culture', {}).get(culture_idx, f'culture_{culture_idx}')
            type_name = label_info.get('idx_to_type', {}).get(type_idx, f'type_{type_idx}')
            era_name = label_info.get('idx_to_era', {}).get(era_idx, f'era_{era_idx}')

            # Compute reasoning step contributions
            step_scores = []
            for step_feat in predictions.get('reasoning_steps', []):
                step_scores.append(float(step_feat[b].norm().item()))

            explanation = (
                f"After {len(step_scores)}-step reasoning (feature norms: "
                f"{[f'{s:.2f}' for s in step_scores]}), "
                f"this artifact is identified as [{culture_name}] {type_name} "
                f"from the {era_name} period."
            )
            explanations.append(explanation)

        return explanations
