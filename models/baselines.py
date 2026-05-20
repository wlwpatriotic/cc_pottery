"""
Baseline models for fine-grained pottery recognition.
Includes: CLIP zero-shot, ViT fine-tuning, DINOv2, and hierarchical classifiers.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
import numpy as np


class ViTFineTune(nn.Module):
    """Fine-tuned ViT-B/16 with hierarchical classification heads."""

    def __init__(self, num_cultures, num_types, num_eras,
                 pretrained=True, freeze_backbone=False):
        super().__init__()
        self.backbone = models.vit_b_16(weights='DEFAULT' if pretrained else None)
        hidden_dim = self.backbone.heads.head.in_features

        # Remove original classification head
        self.backbone.heads = nn.Identity()

        self.culture_head = nn.Sequential(
            nn.Linear(hidden_dim, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, num_cultures)
        )
        self.type_head = nn.Sequential(
            nn.Linear(hidden_dim, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, num_types)
        )
        self.era_head = nn.Sequential(
            nn.Linear(hidden_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, num_eras)
        )

        if freeze_backbone:
            for param in self.backbone.parameters():
                param.requires_grad = False

    def forward(self, x):
        features = self.backbone(x)  # (B, hidden_dim)
        return {
            'culture': self.culture_head(features),
            'type': self.type_head(features),
            'era': self.era_head(features),
            'features': features
        }


class ResNetFineTune(nn.Module):
    """Fine-tuned ResNet-50 baseline with hierarchical heads."""

    def __init__(self, num_cultures, num_types, num_eras,
                 pretrained=True, freeze_backbone=False):
        super().__init__()
        self.backbone = models.resnet50(weights='DEFAULT' if pretrained else None)
        hidden_dim = self.backbone.fc.in_features
        self.backbone.fc = nn.Identity()

        self.culture_head = nn.Linear(hidden_dim, num_cultures)
        self.type_head = nn.Linear(hidden_dim, num_types)
        self.era_head = nn.Linear(hidden_dim, num_eras)

        if freeze_backbone:
            for param in self.backbone.parameters():
                param.requires_grad = False

    def forward(self, x):
        features = self.backbone(x)
        return {
            'culture': self.culture_head(features),
            'type': self.type_head(features),
            'era': self.era_head(features),
            'features': features
        }


class CLIPZeroShot(nn.Module):
    """CLIP-based zero-shot classifier using archaeological text prompts."""

    def __init__(self, clip_model_name='ViT-B/32', label_info=None, device='cuda'):
        super().__init__()
        try:
            import open_clip
            self.model, _, self.preprocess = open_clip.create_model_and_transforms(
                clip_model_name, pretrained='laion2b_s34b_b79k'
            )
            self.tokenizer = open_clip.get_tokenizer(clip_model_name)
        except ImportError:
            import clip
            self.model, self.preprocess = clip.load(clip_model_name, device=device)
            self.tokenizer = clip.tokenize

        self.device = device
        self.label_info = label_info
        self.text_embeddings = None  # cached text embeddings

    def build_prompts(self, class_names, task='culture'):
        """Build archaeological text prompts for each class."""
        templates = [
            "a painted pottery {class_name} from ancient China",
            "{class_name} style ceramic vessel with painted decoration",
            "archaeological artifact: {class_name} painted pottery",
            "a {class_name} painted pottery vessel in museum collection",
        ]
        prompts = []
        for name in class_names:
            for tpl in templates:
                prompts.append(tpl.format(class_name=name))
        return prompts

    def encode_text(self, class_names, task='culture'):
        """Pre-compute text embeddings for zero-shot classification."""
        prompts = self.build_prompts(class_names, task)
        tokenized = self.tokenizer(prompts).to(self.device)
        with torch.no_grad():
            text_features = self.model.encode_text(tokenized)
            text_features = F.normalize(text_features, dim=-1)

        # Average across templates per class
        num_classes = len(class_names)
        num_templates = len(prompts) // num_classes
        text_features = text_features.reshape(num_classes, num_templates, -1)
        text_features = text_features.mean(dim=1)
        text_features = F.normalize(text_features, dim=-1)
        return text_features

    def forward(self, images):
        """Zero-shot classification."""
        image_features = self.model.encode_image(images)
        image_features = F.normalize(image_features, dim=-1)

        results = {}
        for task in ['culture', 'type', 'era']:
            if self.text_embeddings is not None and task in self.text_embeddings:
                text_features = self.text_embeddings[task]
                logits = 100.0 * image_features @ text_features.T
                results[task] = logits

        return results


class HierarchicalViT(nn.Module):
    """ViT with hierarchical label structure: culture -> type -> era."""

    def __init__(self, num_cultures, num_types, num_eras, pretrained=True):
        super().__init__()
        self.backbone = models.vit_b_16(weights='DEFAULT' if pretrained else None)
        hidden_dim = self.backbone.heads.head.in_features
        self.backbone.heads = nn.Identity()

        # Shared feature extractor
        self.shared_fc = nn.Sequential(
            nn.Linear(hidden_dim, 1024),
            nn.ReLU(),
            nn.Dropout(0.3),
        )

        # Culture classification (coarse)
        self.culture_head = nn.Linear(1024, num_cultures)

        # Type classification conditioned on culture
        self.type_head = nn.Sequential(
            nn.Linear(1024 + num_cultures, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, num_types),
        )

        # Era classification conditioned on culture + type
        self.era_head = nn.Sequential(
            nn.Linear(1024 + num_cultures + num_types, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, num_eras),
        )

    def forward(self, x):
        features = self.backbone(x)
        shared = self.shared_fc(features)

        # Culture prediction
        culture_logits = self.culture_head(shared)
        culture_probs = F.softmax(culture_logits, dim=-1)

        # Type prediction (conditioned on culture)
        type_logits = self.type_head(torch.cat([shared, culture_probs], dim=-1))

        # Era prediction (conditioned on culture + type)
        type_probs = F.softmax(type_logits, dim=-1)
        era_logits = self.era_head(torch.cat([shared, culture_probs, type_probs], dim=-1))

        return {
            'culture': culture_logits,
            'type': type_logits,
            'era': era_logits,
            'features': features
        }


class GeometricAugmentation:
    """Pottery-specific augmentations: warp, cylindrical projection simulation."""

    @staticmethod
    def apply_cylindrical_warp(image, strength=0.1):
        """Simulate cylindrical vessel surface for data augmentation."""
        _, h, w = image.shape
        grid_x = torch.linspace(-1, 1, w)
        grid_y = torch.linspace(-1, 1, h)
        grid_xx, grid_yy = torch.meshgrid(grid_x, grid_y, indexing='xy')

        # Cylindrical distortion
        theta = grid_xx * np.pi * strength
        grid_xx_warped = torch.sin(theta) / (np.pi * strength + 1e-8)
        grid_yy_warped = grid_yy / (torch.cos(theta) + 1e-8)

        grid = torch.stack([grid_xx_warped, grid_yy_warped], dim=-1).unsqueeze(0)
        return F.grid_sample(image.unsqueeze(0), grid, align_corners=True).squeeze(0)
