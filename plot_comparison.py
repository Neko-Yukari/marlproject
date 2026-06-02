import matplotlib.pyplot as plt
import numpy as np
import json
import csv
from pathlib import Path

# Load baseline results
baseline_files = list(Path("results").glob("baselines_*/baselines.json"))
baselines = {}
if baseline_files:
    with open(baseline_files[-1]) as f:
        baselines = json.load(f)

# Load IPPO results
ippo_files = list(Path("results").glob("ippo_paper_v2_10k_*"))
ippo_data = []
if ippo_files:
    with open(sorted(ippo_files)[-1] / "ippo_paper_v2_10k.csv") as f:
        reader = csv.DictReader(f)
        for row in reader:
            ippo_data.append({
                'ep': int(row['ep']),
                'avg_cost': float(row['avg_cost']),
                'completion_rate': float(row['completion_rate'])
            })

# Load ExplabOff results
explaboff_files = list(Path("results").glob("explaboff_paper_v2_gpu_10k_*"))
explaboff_data = []
if explaboff_files:
    with open(sorted(explaboff_files)[-1] / "explaboff_paper_v2_gpu_10k.csv") as f:
        reader = csv.DictReader(f)
        for row in reader:
            explaboff_data.append({
                'ep': int(row['ep']),
                'avg_cost': float(row['avg_cost']),
                'completion_rate': float(row['completion_rate'])
            })

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

# Plot 1: AvgCost convergence
if ippo_data:
    eps = [d['ep'] for d in ippo_data]
    costs = [d['avg_cost'] for d in ippo_data]
    ax1.plot(eps, costs, 'b-o', label='IPPO', markersize=4)

if explaboff_data:
    eps = [d['ep'] for d in explaboff_data]
    costs = [d['avg_cost'] for d in explaboff_data]
    ax1.plot(eps, costs, 'r-s', label='ExplabOff', markersize=4)

# Add baselines as horizontal lines
colors = ['g', 'm', 'c', 'y', 'orange', 'purple', 'brown']
for i, (name, data) in enumerate(baselines.items()):
    ax1.axhline(y=data['avg_cost'], color=colors[i % len(colors)], 
                linestyle='--', alpha=0.7, label=f'{name} ({data["avg_cost"]:.3f})')

ax1.set_xlabel('Episode')
ax1.set_ylabel('Average Cost')
ax1.set_title('Convergence: AvgCost')
ax1.legend(fontsize=8, loc='upper right')
ax1.grid(True, alpha=0.3)

# Plot 2: Completion Rate
if ippo_data:
    eps = [d['ep'] for d in ippo_data]
    comps = [d['completion_rate'] * 100 for d in ippo_data]
    ax2.plot(eps, comps, 'b-o', label='IPPO', markersize=4)

if explaboff_data:
    eps = [d['ep'] for d in explaboff_data]
    comps = [d['completion_rate'] * 100 for d in explaboff_data]
    ax2.plot(eps, comps, 'r-s', label='ExplabOff', markersize=4)

# Add baselines
for i, (name, data) in enumerate(baselines.items()):
    ax2.axhline(y=data['completion_rate'] * 100, color=colors[i % len(colors)], 
                linestyle='--', alpha=0.7, label=f'{name} ({data["completion_rate"]*100:.1f}%)')

ax2.set_xlabel('Episode')
ax2.set_ylabel('Completion Rate (%)')
ax2.set_title('Convergence: SuccRate')
ax2.legend(fontsize=8, loc='lower right')
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('results/convergence_comparison.png', dpi=150, bbox_inches='tight')
print("Saved: results/convergence_comparison.png")

# Create summary table
print("\n" + "="*70)
print("SUMMARY: Algorithm Comparison")
print("="*70)
print(f"{'Algorithm':<20} {'AvgCost':>10} {'SuccRate':>10} {'Status':>15}")
print("-"*70)

# Add RL results
print(f"{'IPPO (10K)':<20} {0.427:10.3f} {93.3:9.1f}% {'RL':>15}")
print(f"{'ExplabOff (10K)':<20} {0.435:10.3f} {90.0:9.1f}% {'RL+MI':>15}")

# Add baselines
for name, data in baselines.items():
    print(f"{name:<20} {data['avg_cost']:10.3f} {data['completion_rate']*100:9.1f}% {'Baseline':>15}")

print("-"*70)
print(f"{'Optimal Heuristic':<20} {0.411:10.3f} {100.0:9.1f}% {'Size_Based':>15}")
print("="*70)
