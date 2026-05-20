"""
Cross-experiment comparison and LaTeX table generation.
"""
import json
from pathlib import Path
import numpy as np


def load_results(output_dir):
    """Load all result JSONs from experiment directories."""
    results = {}
    root = Path(output_dir)
    for exp_dir in root.iterdir():
        if not exp_dir.is_dir():
            continue
        results_file = exp_dir / 'results.json'
        eval_file = exp_dir / 'evaluation_results.json'
        if results_file.exists():
            with open(results_file, 'r', encoding='utf-8') as f:
                results[exp_dir.name] = json.load(f)
    return results


def generate_latex_table(all_results, save_path=None):
    """Generate LaTeX-formatted results table."""
    if not all_results:
        return "No results available yet."

    # Determine which metrics to show
    tasks = ['culture', 'type', 'era']
    metrics = ['acc', 'macro_f1']

    latex = []
    latex.append("\\begin{table}[t]")
    latex.append("\\centering")
    latex.append("\\caption{Comparison of methods on fine-grained pottery recognition.}")
    latex.append("\\label{tab:main_results}")
    latex.append("\\small")
    latex.append("\\begin{tabular}{l" + "c" * (len(tasks) * len(metrics)) + "}")
    latex.append("\\toprule")

    # Header
    header = " & "
    for task in tasks:
        for metric in metrics:
            header += f"\\multicolumn{{1}}{{c}}{{{task.capitalize()} {metric.upper()}}}"
            if not (task == tasks[-1] and metric == metrics[-1]):
                header += " & "
    header += " \\\\"
    latex.append(header)
    latex.append("\\midrule")

    # Body
    for method, results in all_results.items():
        method_display = method.replace('_', '\\_')
        row = f"  {method_display}"
        for task in tasks:
            for metric in metrics:
                key = f"{task}_{metric}"
                if key in results:
                    row += f" & {results[key]:.3f}"
                elif 'final_test_metrics' in results and results['final_test_metrics']:
                    val = results['final_test_metrics'].get(f'test_{key}', None)
                    row += f" & {val:.3f}" if val is not None else " & -"
                else:
                    row += " & -"
        row += " \\\\"
        latex.append(row)

    latex.append("\\bottomrule")
    latex.append("\\end{tabular}")
    latex.append("\\end{table}")

    table = '\n'.join(latex)
    print(table)

    if save_path:
        with open(save_path, 'w', encoding='utf-8') as f:
            f.write(table)

    return table


def generate_comparison_summary(all_results):
    """Generate a text summary comparing methods."""
    lines = []
    lines.append("=" * 70)
    lines.append("METHOD COMPARISON SUMMARY")
    lines.append("=" * 70)

    for method, results in sorted(all_results.items()):
        lines.append(f"\n## {method}")
        if 'best_val_acc' in results:
            lines.append(f"  Best val accuracy (culture): {results['best_val_acc']:.4f}")

        if 'final_test_metrics' in results and results['final_test_metrics']:
            tm = results['final_test_metrics']
            for key in ['test_culture_acc', 'test_type_acc', 'test_era_acc',
                        'test_culture_macro_f1', 'test_type_macro_f1']:
                if key in tm:
                    lines.append(f"  {key}: {tm[key]:.4f}")

    return '\n'.join(lines)


if __name__ == '__main__':
    results = load_results(r'f:\考古\cc_pottery\outputs')
    print(f"Found {len(results)} experiment results")
    generate_latex_table(results, r'f:\考古\cc_pottery\outputs\comparison_table.tex')
    summary = generate_comparison_summary(results)
    print(summary)
