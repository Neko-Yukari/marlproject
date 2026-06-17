import json
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

# Load ES-aware results
with open('results/es_gnn_checkpoint_eval.json') as f:
    es_data = json.load(f)

# Old index-based results (from generalization_report.json or hardcoded from b63/b79)
old_ippo_gnn = {
    '3ES-7MD': {'cost': 0.4614, 'comp': 0.734},
    '2ES-5MD': {'cost': 0.4148, 'comp': 0.942},
    '2ES-3MD': {'cost': 0.4568, 'comp': 0.761},
}
old_explaboff_gnn = {
    '3ES-7MD': {'cost': 0.4123, 'comp': 0.981},
    '2ES-5MD': {'cost': 0.4486, 'comp': 0.838},
    '2ES-3MD': {'cost': 0.4720, 'comp': 0.666},
}

# ES-aware final results
new_ippo_gnn = es_data['IPPO+GNN']['10000']
new_explaboff_gnn = es_data['ExplabOff+GNN']['10000']

configs = ['3ES-7MD\n(train)', '2ES-5MD', '2ES-3MD']
old_ippo_costs = [old_ippo_gnn[c]['cost'] for c in ['3ES-7MD', '2ES-5MD', '2ES-3MD']]
old_expl_costs = [old_explaboff_gnn[c]['cost'] for c in ['3ES-7MD', '2ES-5MD', '2ES-3MD']]
new_ippo_costs = [new_ippo_gnn[c]['cost'] for c in ['3ES-7MD', '2ES-5MD', '2ES-3MD']]
new_expl_costs = [new_explaboff_gnn[c]['cost'] for c in ['3ES-7MD', '2ES-5MD', '2ES-3MD']]

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

# Cost comparison
x = np.arange(len(configs))
width = 0.2
ax1.bar(x - 1.5*width, old_ippo_costs, width, label='Index IPPO+GNN', color='#FF6B6B', alpha=0.8)
ax1.bar(x - 0.5*width, old_expl_costs, width, label='Index ExplabOff+GNN', color='#4ECDC4', alpha=0.8)
ax1.bar(x + 0.5*width, new_ippo_costs, width, label='ES-aware IPPO+GNN', color='#C44569', alpha=0.9)
ax1.bar(x + 1.5*width, new_expl_costs, width, label='ES-aware ExplabOff+GNN', color='#006266', alpha=0.9)
ax1.set_ylabel('Average Cost', fontsize=12)
ax1.set_title('Cross-Config Generalization: Cost', fontsize=13, fontweight='bold')
ax1.set_xticks(x)
ax1.set_xticklabels(configs)
ax1.legend(fontsize=9)
ax1.grid(axis='y', alpha=0.3)

# Completion rate comparison
old_ippo_comp = [old_ippo_gnn[c]['comp']*100 for c in ['3ES-7MD', '2ES-5MD', '2ES-3MD']]
old_expl_comp = [old_explaboff_gnn[c]['comp']*100 for c in ['3ES-7MD', '2ES-5MD', '2ES-3MD']]
new_ippo_comp = [new_ippo_gnn[c]['comp']*100 for c in ['3ES-7MD', '2ES-5MD', '2ES-3MD']]
new_expl_comp = [new_explaboff_gnn[c]['comp']*100 for c in ['3ES-7MD', '2ES-5MD', '2ES-3MD']]

ax2.bar(x - 1.5*width, old_ippo_comp, width, label='Index IPPO+GNN', color='#FF6B6B', alpha=0.8)
ax2.bar(x - 0.5*width, old_expl_comp, width, label='Index ExplabOff+GNN', color='#4ECDC4', alpha=0.8)
ax2.bar(x + 0.5*width, new_ippo_comp, width, label='ES-aware IPPO+GNN', color='#C44569', alpha=0.9)
ax2.bar(x + 1.5*width, new_expl_comp, width, label='ES-aware ExplabOff+GNN', color='#006266', alpha=0.9)
ax2.set_ylabel('Completion Rate (%)', fontsize=12)
ax2.set_title('Cross-Config Generalization: Completion Rate', fontsize=13, fontweight='bold')
ax2.set_xticks(x)
ax2.set_xticklabels(configs)
ax2.legend(fontsize=9)
ax2.grid(axis='y', alpha=0.3)
ax2.set_ylim([0, 105])

plt.tight_layout()
output = Path('results/gnn_index_vs_esaware_comparison.png')
plt.savefig(output, dpi=300, bbox_inches='tight')
print(f'Saved: {output}')

# Also create a cleaner version with just improvement arrows
fig, ax = plt.subplots(figsize=(10, 6))
models = ['IPPO+GNN', 'ExplabOff+GNN']
cfg_list = ['3ES-7MD', '2ES-5MD', '2ES-3MD']
old_data = [old_ippo_gnn, old_explaboff_gnn]
new_data = [new_ippo_gnn, new_explaboff_gnn]

for m_idx, (model, old, new) in enumerate(zip(models, old_data, new_data)):
    costs_old = [old[c]['cost'] for c in cfg_list]
    costs_new = [new[c]['cost'] for c in cfg_list]
    offset = 0.15 if m_idx == 0 else -0.15
    x_pos = np.arange(len(cfg_list)) + offset
    ax.plot(x_pos, costs_old, 'o--', label=f'{model} (Index-based)', color='#FF6B6B' if m_idx==0 else '#4ECDC4', linewidth=2, markersize=8)
    ax.plot(x_pos, costs_new, 's-', label=f'{model} (ES-aware)', color='#C44569' if m_idx==0 else '#006266', linewidth=2, markersize=8)

ax.set_xticks(np.arange(len(cfg_list)))
ax.set_xticklabels(cfg_list)
ax.set_ylabel('Average Cost')
ax.set_title('GNN Generalization: Index-based vs ES-aware Policy Head', fontsize=13, fontweight='bold')
ax.legend(fontsize=10)
ax.grid(alpha=0.3)
plt.tight_layout()
output2 = Path('results/gnn_generalization_lines.png')
plt.savefig(output2, dpi=300, bbox_inches='tight')
print(f'Saved: {output2}')
