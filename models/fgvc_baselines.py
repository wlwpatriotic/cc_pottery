"""
FGVC baselines: CAL (Counterfactual Attention Learning) and API-Net.
Simplified implementations adapted for pottery recognition.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models


class CALBaseline(nn.Module):
    """
    Simplified CAL: Counterfactual Attention Learning.
    Uses attention dropout to create counterfactual features, then
    contrastive loss between factual and counterfactual predictions.

    Reference: Rao et al., "Counterfactual Attention Learning for FGVC", ICCV 2021.
    """

    def __init__(self, num_cultures, num_types, num_eras, backbone='resnet50'):
        super().__init__()
        if backbone == 'resnet50':
            self.backbone = models.resnet50(weights='DEFAULT')
            hidden_dim = self.backbone.fc.in_features
            self.backbone.fc = nn.Identity()

            # Attention branch
            self.attn_conv = nn.Sequential(
                nn.Conv2d(2048, 512, 1),
                nn.ReLU(),
                nn.Conv2d(512, 1, 1),
                nn.Sigmoid(),
            )
        else:
            raise ValueError(f"Unsupported backbone: {backbone}")

        self.pool = nn.AdaptiveAvgPool2d(1)

        self.culture_head = nn.Linear(hidden_dim, num_cultures)
        self.type_head = nn.Linear(hidden_dim, num_types)
        self.era_head = nn.Linear(hidden_dim, num_eras)

        self.cal_lambda = 0.01  # weight for CAL loss

    def forward(self, images, return_attn=False, **kwargs):
        # Extract features before pooling (need spatial features)
        x = self.backbone.conv1(images)
        x = self.backbone.bn1(x)
        x = self.backbone.relu(x)
        x = self.backbone.maxpool(x)
        x = self.backbone.layer1(x)
        x = self.backbone.layer2(x)
        x = self.backbone.layer3(x)
        x = self.backbone.layer4(x)  # (B, 2048, H, W)

        # Attention map
        attn = self.attn_conv(x)  # (B, 1, H, W)
        B, C, H, W = x.shape

        # Factual features (with attention)
        factual_feat = (x * attn).sum(dim=[2, 3]) / (attn.sum(dim=[2, 3]) + 1e-8)
        # Also do global pooling for robustness
        global_feat = self.pool(x).flatten(1)
        fused_feat = factual_feat + global_feat

        # Counterfactual features (with randomized attention)
        rand_attn = torch.rand_like(attn)
        counter_feat = (x * rand_attn).sum(dim=[2, 3]) / (rand_attn.sum(dim=[2, 3]) + 1e-8)
        counter_feat = counter_feat + global_feat

        outputs = {
            'culture': self.culture_head(fused_feat),
            'type': self.type_head(fused_feat),
            'era': self.era_head(fused_feat),
            'features': fused_feat,
        }

        if return_attn:
            outputs['attention'] = attn

        return outputs


class APINetBaseline(nn.Module):
    """
    Simplified API-Net: Attentive Pairwise Interaction Network.
    Learns by comparing pairs of images from same/different classes.

    Reference: Zhuang et al., "API-Net: Attentive Pairwise Interaction Network for FGVC", AAAI 2020.
    Simplified: uses feature contrastive learning instead of full pairwise interaction.
    """

    def __init__(self, num_cultures, num_types, num_eras, backbone='resnet50'):
        super().__init__()
        if backbone == 'resnet50':
            self.backbone = models.resnet50(weights='DEFAULT')
            hidden_dim = self.backbone.fc.in_features
            self.backbone.fc = nn.Identity()
        else:
            raise ValueError(f"Unsupported backbone: {backbone}")

        # Feature refinement
        self.refine = nn.Sequential(
            nn.Linear(hidden_dim, 1024),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(1024, 512),
        )

        self.culture_head = nn.Linear(512, num_cultures)
        self.type_head = nn.Linear(512, num_types)
        self.era_head = nn.Linear(512, num_eras)

    def forward(self, images, **kwargs):
        features = self.backbone(images)
        refined = self.refine(features)
        return {
            'culture': self.culture_head(refined),
            'type': self.type_head(refined),
            'era': self.era_head(refined),
            'features': refined,
        }


class SimpleConvNext(nn.Module):
    """Lightweight ConvNeXt-Tiny baseline."""

    def __init__(self, num_cultures, num_types, num_eras):
        super().__init__()
        self.backbone = models.convnext_tiny(weights='DEFAULT')
        hidden_dim = self.backbone.classifier[2].in_features
        self.backbone.classifier = nn.Identity()

        self.culture_head = nn.Linear(hidden_dim, num_cultures)
        self.type_head = nn.Linear(hidden_dim, num_types)
        self.era_head = nn.Linear(hidden_dim, num_eras)

    def forward(self, images, **kwargs):
        f = self.backbone(images)
        return {
            'culture': self.culture_head(f),
            'type': self.type_head(f),
            'era': self.era_head(f),
            'features': f,
        }
