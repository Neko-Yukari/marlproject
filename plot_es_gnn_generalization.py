import json
import matplotlib.pyplot as plt
import numpy as np
import os

# Load data
with open('results/es_gnn_checkpoint_eval.json', 'r', encoding='utf-8') as f:
    data = json.load(f)

episodes = sorted([int(k) for k in data['IPPO+GNN'].keys()])
configs = ['2ES-3MD', '2ES-5MD', '3ES-7MD']
colors = {'2ES-3MD': '#e74c3c', '2ES-5MD': '#3498db', '3ES-7MD': '#2ecc71'}

fig, axes = plt.subplots(2, 2, figsize=(14, 10))

for idx, (model, title) in enumerate([('IPPO+GNN', 'IPPO + ES-aware GNN'), ('ExplabOff+GNN', 'ExplabOff + ES-aware GNN')]):
    ax_cost = axes[0, idx]
    ax_comp = axes[1, idx]
    
    for cfg in configs:
        costs = [data[model][str(ep)][cfg]['cost'] for ep in episodes]
        comps = [data[model][str(ep)][cfg]['comp'] * 100 for ep in episodes]
        ax_cost.plot(episodes, costs, marker='o', label=cfg, color=colors[cfg], linewidth=2)
        ax_comp.plot(episodes, comps, marker='s', label=cfg, color=colors[cfg], linewidth=2)
    
    ax_cost.set_title(f'{title} - Cost', fontsize=13, fontweight='bold')
    ax_cost.set_xlabel('Training Episode')
    ax_cost.set_ylabel('Avg Cost')
    ax_cost.legend()
    ax_cost.grid(True, alpha=0.3)
    ax_cost.set_ylim(0.3, 1.0)
    
    ax_comp.set_title(f'{title} - Completion Rate', fontsize=13, fontweight='bold')
    ax_comp.set_xlabel('Training Episode')
    ax_comp.set_ylabel('Completion (%)')
    ax_comp.legend()
    ax_comp.grid(True, alpha=0.3)
    ax_comp.set_ylim(50, 105)

# Side-by-side comparison at final checkpoint
ax_comp_final = axes[0, 0]  # placeholder not used
plt.tight_layout()
plt.savefig('results/es_gnn_generalization_curves.png', dpi=300, bbox_inches='tight')
print('Saved results/es_gnn_generalization_curves.png')

# Create comparison chart: IPPO vs ExplabOff on each config
fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
for idx, cfg in enumerate(configs):
    ax = axes[idx]
    for model, color, marker in [('IPPO+GNN', '#e67e22', 'o'), ('ExplabOff+GNN', '#9b59b6', 's')]:
        costs = [data[model][str(ep)][cfg]['cost'] for ep in episodes]
        ax.plot(episodes, costs, marker=marker, label=model, color=color, linewidth=2, markersize=5)
    ax.set_title(f'{cfg}', fontsize=13, fontweight='bold')
    ax.set_xlabel('Training Episode')
    ax.set_ylabel('Avg Cost')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0.35, 0.95)

plt.suptitle('IPPO vs ExplabOff (ES-aware GNN) Cross-Config Generalization', fontsize=14, fontweight='bold', y=1.02)
plt.tight_layout()
plt.savefig('results/es_gnn_ippo_vs_explaboff.png', dpi=300, bbox_inches='tight')
print('Saved results/es_gnn_ippo_vs_explaboff.png')
