#!/usr/bin/env python3
"""
整理三向消融结果
"""
import json
import numpy as np
from pathlib import Path

DATA_DIR = Path('data')

# 读取所有结果
results = {}
for mode in ['no_sharing', 'no_gating', 'full']:
    results[mode] = {'r1': [], 'r5': [], 'r10': [], 'loss': []}
    for seed in [42, 123, 456]:
        file = DATA_DIR / f'result_{mode}_r32_s{seed}.json'
        with open(file) as f:
            data = json.load(f)
            results[mode]['r1'].append(data['final_r1'])
            results[mode]['r5'].append(data['final_r5'])
            results[mode]['r10'].append(data['final_r10'])
            results[mode]['loss'].append(data['final_loss'])

# 计算 mean ± std
print('='*80)
print('  三向消融结果汇总（CIFAR-10 子集 500 样本，rank=32，50 epochs）')
print('='*80)
print()
print(f"{'Configuration':<20} {'R@1':<15} {'R@5':<15} {'R@10':<15} {'Loss':<10}")
print('-'*80)

for mode in ['no_sharing', 'no_gating', 'full']:
    r1_mean = np.mean(results[mode]['r1']) * 100
    r1_std = np.std(results[mode]['r1']) * 100
    r5_mean = np.mean(results[mode]['r5']) * 100
    r5_std = np.std(results[mode]['r5']) * 100
    r10_mean = np.mean(results[mode]['r10']) * 100
    r10_std = np.std(results[mode]['r10']) * 100
    loss_mean = np.mean(results[mode]['loss'])
    loss_std = np.std(results[mode]['loss'])
    
    mode_name = mode.replace('_', ' ').title()
    print(f'{mode_name:<20} {r1_mean:>5.1f}±{r1_std:<5.1f} {r5_mean:>5.1f}±{r5_std:<5.1f} {r10_mean:>5.1f}±{r10_std:<5.1f} {loss_mean:>5.2f}±{loss_std:<.2f}')

print()
print('='*80)
print()

# 保存 LaTeX 表格
latex = r'''\begin{table}[t]
\centering
\caption{Three-way ablation on CIFAR-10 subset (500 train, 100 test). Results averaged over 3 seeds.}
\label{tab:ablation}
\begin{tabular}{lccc}
\toprule
Configuration & R@1 (\%) & R@5 (\%) & R@10 (\%) \\
\midrule
'''

for mode in ['no_sharing', 'no_gating', 'full']:
    r1_mean = np.mean(results[mode]['r1']) * 100
    r1_std = np.std(results[mode]['r1']) * 100
    r5_mean = np.mean(results[mode]['r5']) * 100
    r5_std = np.std(results[mode]['r5']) * 100
    r10_mean = np.mean(results[mode]['r10']) * 100
    r10_std = np.std(results[mode]['r10']) * 100
    
    mode_name = mode.replace('_', ' ').title()
    if mode == 'full':
        latex += f'\\textbf{{{mode_name}}} & \\textbf{{{r1_mean:.1f}$\\pm${r1_std:.1f}}} & \\textbf{{{r5_mean:.1f}$\\pm${r5_std:.1f}}} & \\textbf{{{r10_mean:.1f}$\\pm${r10_std:.1f}}} \\\\\n'
    else:
        latex += f'{mode_name} & {r1_mean:.1f}$\\pm${r1_std:.1f} & {r5_mean:.1f}$\\pm${r5_std:.1f} & {r10_mean:.1f}$\\pm${r10_std:.1f} \\\\\n'

latex += r'''\bottomrule
\end{tabular}
\end{table}'''

print('LaTeX 表格代码：')
print(latex)
print()

# 保存到文件
with open(DATA_DIR / 'ablation_table.tex', 'w') as f:
    f.write(latex)

print(f'LaTeX 表格已保存到 {DATA_DIR / "ablation_table.tex"}')
