import json
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
import torch

from envs.paper_accurate_env import PaperAccurateEnvV3
from greedy_baseline import greedy_baseline

# Load ES-aware cross-config results
with open('results/es_gnn_checkpoint_eval.json') as f:
    data = json.load(f)

# Get ep10000 cross-config values (trained on 3ES-7MD, tested on 2ES-3MD/2ES-5MD)
cross = {
    'IPPO+GNN': {k: data['IPPO+GNN']['10000'][k] for k in ['2ES-3MD','2ES-5MD']},
    'ExplabOff+GNN': {k: data['ExplabOff+GNN']['10000'][k] for k in ['2ES-3MD','2ES-5MD']},
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

# Greedy handwritten baseline - evaluate dynamically with fixed greedy_baseline.py
greedy = {}
for num_md, num_es, cfg_name in [(3, 2, '2ES-3MD'), (5, 2, '2ES-5MD')]:
    env = PaperAccurateEnvV3(num_devices=num_md, num_servers=num_es, seed=10000)
    cost_mean, cost_std, comp = greedy_baseline(env, seed=10000, episodes=100)
    greedy[cfg_name] = {'cost': cost_mean, 'cost_std': cost_std, 'comp': comp}

configs = ['2ES-3MD', '2ES-5MD']
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.5))

x = np.arange(len(configs))
width = 0.15

# Cost plot
ax1.bar(x - 2*width, [greedy[c]['cost'] for c in configs], width,
        label='Greedy (handwritten)', color='#95A5A6', alpha=0.9)
ax1.bar(x - width, [native['IPPO+GNN'][c]['cost'] for c in configs], width,
        label='IPPO+GNN (native)', color='#FF6B6B', alpha=0.85)
ax1.bar(x, [native['ExplabOff+GNN'][c]['cost'] for c in configs], width,
        label='ExplabOff+GNN (native)', color='#4ECDC4', alpha=0.85)
ax1.bar(x + width, [cross['IPPO+GNN'][c]['cost'] for c in configs], width,
        label='IPPO+GNN (3ES-7MD trained)', color='#C44569', alpha=0.85, hatch='//')
ax1.bar(x + 2*width, [cross['ExplabOff+GNN'][c]['cost'] for c in configs], width,
        label='ExplabOff+GNN (3ES-7MD trained)', color='#006266', alpha=0.85, hatch='//')

ax1.set_ylabel('Average Cost', fontsize=12)
ax1.set_title('Experiment 3: Cross-Config Generalization vs Handwritten Baseline', fontsize=13, fontweight='bold')
ax1.set_xticks(x)
ax1.set_xticklabels(configs)
ax1.legend(fontsize=8.5, loc='upper left')
ax1.grid(axis='y', alpha=0.3)

# Completion plot
ax2.bar(x - 2*width, [greedy[c]['comp']*100 for c in configs], width,
        label='Greedy (handwritten)', color='#95A5A6', alpha=0.9)
ax2.bar(x - width, [native['IPPO+GNN'][c]['comp']*100 for c in configs], width,
        label='IPPO+GNN (native)', color='#FF6B6B', alpha=0.85)
ax2.bar(x, [native['ExplabOff+GNN'][c]['comp']*100 for c in configs], width,
        label='ExplabOff+GNN (native)', color='#4ECDC4', alpha=0.85)
ax2.bar(x + width, [cross['IPPO+GNN'][c]['comp']*100 for c in configs], width,
        label='IPPO+GNN (3ES-7MD trained)', color='#C44569', alpha=0.85, hatch='//')
ax2.bar(x + 2*width, [cross['ExplabOff+GNN'][c]['comp']*100 for c in configs], width,
        label='ExplabOff+GNN (3ES-7MD trained)', color='#006266', alpha=0.85, hatch='//')

ax2.set_ylabel('Completion Rate (%)', fontsize=12)
ax2.set_title('Experiment 3: Completion Rate Comparison', fontsize=13, fontweight='bold')
ax2.set_xticks(x)
ax2.set_xticklabels(configs)
ax2.legend(fontsize=8.5, loc='upper right')
ax2.grid(axis='y', alpha=0.3)
ax2.set_ylim([0, 105])

plt.tight_layout()
out_dir = Path('results/ppt_experiments')
out_dir.mkdir(exist_ok=True)
plt.savefig(out_dir / 'exp3_cross_vs_native.png', dpi=300, bbox_inches='tight')
print('Saved: results/ppt_experiments/exp3_cross_vs_native.png')

print('\nExperiment 3 data (with Greedy baseline):')
print('| Method | Train | Test | Cost | Comp |')
print('|--------|-------|------|------|------|')
for cfg in configs:
    print(f'| Greedy | - | {cfg} | {greedy[cfg]["cost"]:.4f} | {greedy[cfg]["comp"]*100:.1f}% |')
for model in ['IPPO+GNN', 'ExplabOff+GNN']:
    for cfg in configs:
        n = native[model][cfg]
        print(f'| {model} (native) | {cfg} | {cfg} | {n["cost"]:.4f} | {n["comp"]*100:.1f}% |')
for model in ['IPPO+GNN', 'ExplabOff+GNN']:
    for cfg in configs:
        c = cross[model][cfg]
        print(f'| {model} (3ES-7MD) | 3ES-7MD | {cfg} | {c["cost"]:.4f} | {c["comp"]*100:.1f}% |')
