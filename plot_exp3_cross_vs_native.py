import json
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

# Load ES-aware cross-config results
with open('results/es_gnn_checkpoint_eval.json') as f:
    data = json.load(f)

# Get ep10000 cross-config values
cross = {
    'IPPO+GNN': {k: data['IPPO+GNN']['10000'][k] for k in ['2ES-3MD','2ES-5MD','3ES-7MD']},
    'ExplabOff+GNN': {k: data['ExplabOff+GNN']['10000'][k] for k in ['2ES-3MD','2ES-5MD','3ES-7MD']},
}

# Native-trained models (legacy/old models, as user permitted)
native = {
    'IPPO+GNN': {
        '2ES-3MD': {'cost': 0.4181, 'comp': 0.999},
        '2ES-5MD': {'cost': 0.4486, 'comp': 0.838},
    },
    'ExplabOff+GNN': {
        '2ES-3MD': {'cost': 0.4123, 'comp': 0.981},
        '2ES-5MD': {'cost': 0.380, 'comp': 0.97},
    },
}

configs = ['2ES-3MD', '2ES-5MD']
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5.5))

x = np.arange(len(configs))
width = 0.18

# Cost
for i, model in enumerate(['IPPO+GNN', 'ExplabOff+GNN']):
    cross_cost = [cross[model][c]['cost'] for c in configs]
    native_cost = [native[model][c]['cost'] for c in configs]
    offset = (i - 0.5) * 2 * width
    ax1.bar(x + offset, cross_cost, width, label=f'{model} (3ES-7MD trained)', color=['#FF6B6B','#4ECDC4'][i], alpha=0.85)
    ax1.bar(x + offset + width, native_cost, width, label=f'{model} (native trained)', color=['#C44569','#006266'][i], alpha=0.85, hatch='//')

ax1.set_ylabel('Average Cost', fontsize=12)
ax1.set_title('Cross-Config vs Native-Trained: Cost', fontsize=13, fontweight='bold')
ax1.set_xticks(x + width/2)
ax1.set_xticklabels(configs)
ax1.legend(fontsize=9, loc='upper left')
ax1.grid(axis='y', alpha=0.3)

# Completion
for i, model in enumerate(['IPPO+GNN', 'ExplabOff+GNN']):
    cross_comp = [cross[model][c]['comp']*100 for c in configs]
    native_comp = [native[model][c]['comp']*100 for c in configs]
    offset = (i - 0.5) * 2 * width
    ax2.bar(x + offset, cross_comp, width, label=f'{model} (3ES-7MD trained)', color=['#FF6B6B','#4ECDC4'][i], alpha=0.85)
    ax2.bar(x + offset + width, native_comp, width, label=f'{model} (native trained)', color=['#C44569','#006266'][i], alpha=0.85, hatch='//')

ax2.set_ylabel('Completion Rate (%)', fontsize=12)
ax2.set_title('Cross-Config vs Native-Trained: Completion', fontsize=13, fontweight='bold')
ax2.set_xticks(x + width/2)
ax2.set_xticklabels(configs)
ax2.legend(fontsize=9, loc='upper right')
ax2.grid(axis='y', alpha=0.3)
ax2.set_ylim([0, 105])

plt.tight_layout()
out_dir = Path('results/ppt_experiments')
out_dir.mkdir(exist_ok=True)
plt.savefig(out_dir / 'exp3_cross_vs_native.png', dpi=300, bbox_inches='tight')
print('Saved: results/ppt_experiments/exp3_cross_vs_native.png')

# Also print data table
print('\nExperiment 3 data:')
print('| Model | Train | Test | Cost | Comp |')
print('|-------|-------|------|------|------|')
for model in ['IPPO+GNN', 'ExplabOff+GNN']:
    for cfg in configs:
        c = cross[model][cfg]
        n = native[model][cfg]
        print(f'| {model} | 3ES-7MD | {cfg} | {c["cost"]:.4f} | {c["comp"]*100:.1f}% |')
        print(f'| {model} | {cfg} | {cfg} | {n["cost"]:.4f} | {n["comp"]*100:.1f}% |')
