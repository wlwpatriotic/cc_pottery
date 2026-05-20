"""
CLIP zero-shot baseline for pottery recognition.
Tests open-vocabulary capabilities of CLIP on archaeological artifacts.
"""
import sys; sys.path.insert(0, str(__import__('pathlib').Path(r'f:\考古\cc_pottery')))
import json
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from collections import defaultdict
from pathlib import Path
import numpy as np

from utils.dataset import PotteryDataset, get_transforms, collate_fn

try:
    import open_clip
    HAS_OPENCLIP = True
except ImportError:
    HAS_OPENCLIP = False


class CLIPZeroShotEvaluator:
    """Evaluate CLIP zero-shot on pottery recognition."""

    def __init__(self, model_name='ViT-B-32', device='cuda'):
        self.device = device

        if HAS_OPENCLIP:
            self.model, _, self.preprocess = open_clip.create_model_and_transforms(
                model_name, pretrained='laion2b_s34b_b79k'
            )
            self.tokenizer = open_clip.get_tokenizer(model_name)
        else:
            import clip
            self.model, self.preprocess = clip.load('ViT-B/32', device=device)
            self.tokenizer = clip.tokenize

        self.model = self.model.to(device).eval()

    def build_prompts(self, class_names, task='culture'):
        """Domain-specific prompts for archaeological pottery."""
        templates = {
            'culture': [
                "a painted pottery vessel from the {cls} archaeological culture",
                "{cls} style painted ceramic pottery from ancient China",
                "an excavated {cls} culture painted pottery artifact",
                "archaeological museum exhibit: {cls} painted pottery",
                "a {cls} painted pottery vessel with distinctive decoration",
            ],
            'type': [
                "a painted pottery {cls} vessel",
                "an ancient Chinese ceramic {cls} with painted designs",
                "archaeological artifact: {cls} shaped painted pottery",
                "a museum quality ancient painted pottery {cls}",
                "{cls} vessel from Neolithic China with painted patterns",
            ],
            'era': [
                "painted pottery from the {cls} period",
                "ancient Chinese ceramic from the {cls} era",
                "archaeological artifact dating to the {cls}",
                "{cls} period painted pottery vessel",
                "excavated pottery from {cls} China",
            ],
        }

        templates = templates.get(task, templates['culture'])
        prompts = []
        for name in class_names:
            for tpl in templates:
                prompts.append(tpl.format(cls=name))
        return prompts

    @torch.no_grad()
    def encode_text(self, class_names, task='culture'):
        """Pre-compute text embeddings."""
        prompts = self.build_prompts(class_names, task)

        if HAS_OPENCLIP:
            tokenized = self.tokenizer(prompts).to(self.device)
            text_features = self.model.encode_text(tokenized)
        else:
            tokenized = self.tokenizer(prompts).to(self.device)
            text_features = self.model.encode_text(tokenized)

        text_features = F.normalize(text_features, dim=-1)

        # Average across templates per class
        num_classes = len(class_names)
        num_templates = len(prompts) // num_classes
        text_features = text_features.reshape(num_classes, num_templates, -1)
        text_features = text_features.mean(dim=1)
        text_features = F.normalize(text_features, dim=-1)
        return text_features

    @torch.no_grad()
    def evaluate(self, dataloader, label_info, tasks=('culture', 'type', 'era')):
        """Run zero-shot evaluation."""
        # Pre-compute text embeddings for all classes
        text_features = {}
        for task in tasks:
            idx_to_name = label_info[f'idx_to_{task}']
            class_names = [idx_to_name[i] for i in range(len(idx_to_name))]
            text_features[task] = self.encode_text(class_names, task)

        all_preds = defaultdict(list)
        all_labels = defaultdict(list)
        all_metas = []

        for images, labels, metas in dataloader:
            images = images.to(self.device)

            if HAS_OPENCLIP:
                image_features = self.model.encode_image(images)
            else:
                image_features = self.model.encode_image(images)

            image_features = F.normalize(image_features, dim=-1)

            for task in tasks:
                if task not in labels:
                    continue
                logits = 100.0 * image_features @ text_features[task].T
                preds = logits.argmax(dim=-1)
                all_preds[task].extend(preds.cpu().numpy())
                all_labels[task].extend(labels[task].cpu().numpy())

            all_metas.extend(metas)

        # Compute metrics
        results = {}
        for task in tasks:
            if task not in all_preds:
                continue
            preds = np.array(all_preds[task])
            targets = np.array(all_labels[task])

            acc = (preds == targets).mean()
            results[f'{task}_acc'] = float(acc)

            # Per-class accuracy
            per_class = {}
            for cls in np.unique(targets):
                mask = targets == cls
                if mask.sum() > 0:
                    per_class[int(cls)] = float((preds[mask] == cls).mean())

            results[f'{task}_per_class'] = per_class

            print(f"  {task}: accuracy = {acc:.4f}")

        return results


def run_clip_benchmark():
    root = Path(r'f:\考古\cc_pottery')
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Running CLIP zero-shot benchmark on {device}...\n")

    # Test set
    test_ds = PotteryDataset(str(root), 'test', min_samples_per_class=5,
                             transform=get_transforms(224, False))
    test_loader = DataLoader(test_ds, 32, False, collate_fn=collate_fn)

    label_info = test_ds.get_label_info()

    # Evaluate
    evaluator = CLIPZeroShotEvaluator(device=device)
    results = evaluator.evaluate(test_loader, label_info)

    # Save
    output_dir = root / 'outputs' / 'clip_zero_shot'
    output_dir.mkdir(parents=True, exist_ok=True)
    with open(output_dir / 'results.json', 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False, default=str)

    print(f"\nResults saved to {output_dir / 'results.json'}")
    return results


if __name__ == '__main__':
    run_clip_benchmark()
