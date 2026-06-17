import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

RESULTS = Path('results')

# Parse comparison_log.txt for Hyper 2ES-5MD
def parse_hyper_2es5md():
    log_path = RESULTS / 'comparison_log.txt'
    episodes, costs, comps = [], [], []
    in_target = False
    with open(log_path) as f:
        for line in f:
            line = line.strip()
            if '] Hyper 2ES-5MD' in line:
                in_target = True
                continue
            if in_target:
                if 'Return code:' in line or '] ' in line and 'Hyper' not in line and '2ES-5MD' not in line:
                    # Stop at next experiment
                    if '] ' in line and ('Standard' in line or 'GNN' in line):
                        break
                if line.startswith('Ep') and 'Cost:' in line:
                    parts = line.split('|')
                    try:
                        ep = int(parts[0].replace('Ep', '').strip())
                        cost = float(parts[1].replace('Cost:', '').strip())
                        comp = float(parts[2].replace('Comp:', '').replace('%', '').strip())
                        episodes.append(ep)
                        costs.append(cost)
                        comps.append(comp)
                    except Exception:
                        pass
    return np.array(episodes), np.array(costs), np.array(comps)

eps, costs, comps = parse_hyper_2es5md()

fig, axes = plt.subplots(2, 1, figsize=(12, 8), sharex=True)
axes[0].plot(eps, costs, color='#78E08F', linewidth=2)
axes[0].axvline(x=9000, color='red', linestyle='--', alpha=0.7, label='divergence start (~ep9000)')
axes[0].set_ylabel('Cost')
axes[0].set_title('HyperNetwork 2ES-5MD Training: Cost Trajectory', fontweight='bold')
axes[0].grid(alpha=0.3)
axes[0].legend()

axes[1].plot(eps, comps, color='#78E08F', linewidth=2)
axes[1].axvline(x=9000, color='red', linestyle='--', alpha=0.7)
axes[1].set_xlabel('Episode')
axes[1].set_ylabel('Completion Rate (%)')
axes[1].set_title('HyperNetwork 2ES-5MD Training: Completion Rate', fontweight='bold')
axes[1].grid(alpha=0.3)

# Annotate min cost
min_idx = np.argmin(costs)
axes[0].scatter([eps[min_idx]], [costs[min_idx]], color='green', s=100, zorder=5)
axes[0].annotate(f'best={costs[min_idx]:.3f}\n@ep{eps[min_idx]}',
                 xy=(eps[min_idx], costs[min_idx]),
                 xytext=(eps[min_idx]-2000, costs[min_idx]+0.3),
                 arrowprops=dict(arrowstyle='->', color='green'),
                 fontsize=9)

# Annotate final cost
axes[0].scatter([eps[-1]], [costs[-1]], color='red', s=100, zorder=5)
axes[0].annotate(f'final={costs[-1]:.3f}\n@ep{eps[-1]}',
                 xy=(eps[-1], costs[-1]),
                 xytext=(eps[-1]-1500, costs[-1]-0.3),
                 arrowprops=dict(arrowstyle='->', color='red'),
                 fontsize=9)

fig.tight_layout()
out_path = RESULTS / 'ppt_experiments' / 'hyper_2es5md_divergence.png'
fig.savefig(out_path, dpi=300, bbox_inches='tight')
print(f'Saved: {out_path}')

# Print key stats
print(f'\nHyper 2ES-5MD divergence analysis:')
print(f'Best cost: {costs[min_idx]:.4f} at ep{eps[min_idx]}')
print(f'Final cost: {costs[-1]:.4f} at ep{eps[-1]}')
print(f'Degradation: +{costs[-1] - costs[min_idx]:.4f} ({(costs[-1]/costs[min_idx]-1)*100:.1f}%)')
print(f'Completion: best={comps[min_idx]:.1f}%, final={comps[-1]:.1f}%')

# Find divergence point: cost > 0.6 after ep6000
after_6k = eps > 6000
div_mask = after_6k & (costs > 0.6)
if np.any(div_mask):
    first_div = eps[np.argmax(div_mask)]
    print(f'First cost > 0.6 after ep6000: ep{first_div}, cost={costs[eps==first_div][0]:.4f}')
