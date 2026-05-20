"""
Error case study analysis for paper.
Finds cases where ArchaeoGPT succeeds but ViT fails, and vice versa.
"""
import torch
import torch.nn.functional as F
import numpy as np
from collections import defaultdict
from pathlib import Path
import json


def analyze_prediction_differences(model_a, model_b, dataloader, label_info,
                                    name_a='ArchaeoGPT', name_b='ViT'):
    """
    Find cases where model_a and model_b disagree.
    Returns organized case studies for the paper.
    """
    device = next(model_a.parameters()).device

    both_right = []
    both_wrong = []
    a_right_b_wrong = []  # ArchaeoGPT wins
    a_wrong_b_right = []  # ViT wins

    idx_to_culture = label_info['idx_to_culture']

    all_samples = []
    for images, labels, metas in dataloader:
        images = images.to(device)

        with torch.no_grad():
            out_a = model_a(images)
            out_b = model_b(images)

        preds_a = out_a['culture'].argmax(dim=-1).cpu().numpy()
        preds_b = out_b['culture'].argmax(dim=-1).cpu().numpy()
        targets = labels['culture'].cpu().numpy()

        confs_a = F.softmax(out_a['culture'], dim=-1).max(dim=-1)[0].cpu().numpy()
        confs_b = F.softmax(out_b['culture'], dim=-1).max(dim=-1)[0].cpu().numpy()

        for i in range(len(images)):
            sample = {
                'uid': metas[i]['uid'],
                'true_culture': idx_to_culture.get(targets[i], f'cls_{targets[i]}'),
                'pred_a': idx_to_culture.get(preds_a[i], f'cls_{preds_a[i]}'),
                'pred_b': idx_to_culture.get(preds_b[i], f'cls_{preds_b[i]}'),
                'conf_a': float(confs_a[i]),
                'conf_b': float(confs_b[i]),
                'description': metas[i]['description'][:200],
                'type': metas[i]['type_name'],
            }

            correct_a = preds_a[i] == targets[i]
            correct_b = preds_b[i] == targets[i]

            if correct_a and correct_b:
                both_right.append(sample)
            elif not correct_a and not correct_b:
                both_wrong.append(sample)
            elif correct_a:
                a_right_b_wrong.append(sample)
            else:
                a_wrong_b_right.append(sample)

    # Sort by confidence difference
    a_right_b_wrong.sort(key=lambda x: x['conf_a'] - x['conf_b'], reverse=True)
    a_wrong_b_right.sort(key=lambda x: x['conf_b'] - x['conf_a'], reverse=True)

    report = {
        'summary': {
            f'{name_a} right, {name_b} right': len(both_right),
            f'{name_a} right, {name_b} wrong': len(a_right_b_wrong),
            f'{name_a} wrong, {name_b} right': len(a_wrong_b_right),
            'both wrong': len(both_wrong),
        },
        f'{name_a}_wins': a_right_b_wrong[:10],
        f'{name_b}_wins': a_wrong_b_right[:10],
        'both_wrong_examples': both_wrong[:5],
    }

    # Print summary
    print(f"\n{'='*60}")
    print(f"PREDICTION DIFFERENCE ANALYSIS: {name_a} vs {name_b}")
    print(f"{'='*60}")
    total = len(both_right) + len(a_right_b_wrong) + len(a_wrong_b_right) + len(both_wrong)
    print(f"  Both correct:        {len(both_right)} ({100*len(both_right)/total:.1f}%)")
    print(f"  {name_a} wins:        {len(a_right_b_wrong)} ({100*len(a_right_b_wrong)/total:.1f}%)")
    print(f"  {name_b} wins:         {len(a_wrong_b_right)} ({100*len(a_wrong_b_right)/total:.1f}%)")
    print(f"  Both wrong:          {len(both_wrong)} ({100*len(both_wrong)/total:.1f}%)")

    if a_right_b_wrong:
        print(f"\n  Top {name_a} wins (where {name_a} is right, {name_b} is wrong):")
        for case in a_right_b_wrong[:5]:
            print(f"    [{case['uid']}] True: {case['true_culture']}")
            print(f"      {name_a}: {case['pred_a']} (conf={case['conf_a']:.3f})")
            print(f"      {name_b}:  {case['pred_b']} (conf={case['conf_b']:.3f})")
            print(f"      Desc: {case['description'][:120]}...")

    if a_wrong_b_right:
        print(f"\n  Top {name_b} wins (where {name_b} is right, {name_a} is wrong):")
        for case in a_wrong_b_right[:5]:
            print(f"    [{case['uid']}] True: {case['true_culture']}")
            print(f"      {name_a}: {case['pred_a']} (conf={case['conf_a']:.3f})")
            print(f"      {name_b}:  {case['pred_b']} (conf={case['conf_b']:.3f})")
            print(f"      Desc: {case['description'][:120]}...")

    return report


def generate_hard_case_table(cases, output_path, max_cases=5):
    """Generate LaTeX table for hard case analysis."""
    lines = []
    lines.append("\\begin{table}[t]")
    lines.append("\\centering")
    lines.append("\\caption{Hard case analysis: Where ArchaeoGPT succeeds but ViT fails.}")
    lines.append("\\label{tab:hard_cases}")
    lines.append("\\small")
    lines.append("\\begin{tabular}{p{2cm} p{2.5cm} p{2.5cm} p{4.5cm}}")
    lines.append("\\toprule")
    lines.append("True Culture & ArchaeoGPT & ViT & Description \\\\")
    lines.append("\\midrule")

    for case in cases[:max_cases]:
        true = case['true_culture'][:10]
        pred_a = f"{case['pred_a'][:10]} ({case['conf_a']:.2f})"
        pred_b = f"{case['pred_b'][:10]} ({case['conf_b']:.2f})"
        desc = case['description'][:80] + "..."
        lines.append(f"{true} & {pred_a} & {pred_b} & {desc} \\\\")

    lines.append("\\bottomrule")
    lines.append("\\end{tabular}")
    lines.append("\\end{table}")

    table = '\n'.join(lines)
    if output_path:
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(table)
    return table
