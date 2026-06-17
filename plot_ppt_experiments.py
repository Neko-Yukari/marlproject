"""Generate PPT-ready figures for the three redefined experiments.
Uses latest available models; older data is explicitly labeled as legacy."""
import json
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

plt.rcParams['font.size'] = 10
plt.rcParams['axes.labelsize'] = 11
plt.rcParams['axes.titlesize'] = 12
plt.rcParams['legend.fontsize'] = 9
plt.rcParams['figure.dpi'] = 300

RESULTS = Path('results')
OUT_DIR = RESULTS / 'ppt_experiments'
OUT_DIR.mkdir(exist_ok=True)

# -----------------------------------------------------------------------------
# Experiment 3: Cross-config generalization (latest ES-aware models)
# -----------------------------------------------------------------------------
def plot_experiment3():
    with open(RESULTS / 'es_gnn_checkpoint_eval.json') as f:
        data = json.load(f)

    checkpoints = ['1000', '2000', '3000', '4000', '5000', '6000', '7000', '8000', '9000', '10000']
    configs = ['3ES-7MD', '2ES-5MD', '2ES-3MD']
    x = np.arange(len(checkpoints))

    fig, axes = plt.subplots(1, 2, figsize=(14, 4.5))
    colors = {'IPPO+GNN': '#C44569', 'ExplabOff+GNN': '#006266'}
    markers = {'IPPO+GNN': 'o', 'ExplabOff+GNN': 's'}

    for model in ['IPPO+GNN', 'ExplabOff+GNN']:
        for cfg in configs:
            costs = [data[model][ck][cfg]['cost'] for ck in checkpoints]
            axes[0].plot(x, costs, marker=markers[model], label=f'{model} → {cfg}',
                         color=colors[model], alpha=0.7 if cfg != '3ES-7MD' else 1.0,
                         linestyle='-' if cfg == '3ES-7MD' else '--', linewidth=2, markersize=5)
            comps = [data[model][ck][cfg]['comp'] * 100 for ck in checkpoints]
            axes[1].plot(x, comps, marker=markers[model], label=f'{model} → {cfg}',
                         color=colors[model], alpha=0.7 if cfg != '3ES-7MD' else 1.0,
                         linestyle='-' if cfg == '3ES-7MD' else '--', linewidth=2, markersize=5)

    for ax, title, ylab in zip(axes,
                               ['Cross-Config Generalization Cost', 'Cross-Config Generalization Completion Rate'],
                               ['Average Cost', 'Completion Rate (%)']):
        ax.set_xticks(x)
        ax.set_xticklabels(checkpoints, rotation=45)
        ax.set_xlabel('Training Episode')
        ax.set_ylabel(ylab)
        ax.set_title(title, fontweight='bold')
        ax.grid(alpha=0.3)
        ax.legend(ncol=2, fontsize=8)

    fig.tight_layout()
    fig.savefig(OUT_DIR / 'exp3_cross_config_generalization.png', bbox_inches='tight')
    print(f'Saved: {OUT_DIR / "exp3_cross_config_generalization.png"}')

    # Table summary
    print('\nExperiment 3 summary (ep10000):')
    for model in ['IPPO+GNN', 'ExplabOff+GNN']:
        print(f'  {model}:')
        for cfg in configs:
            r = data[model]['10000'][cfg]
            print(f'    → {cfg}: cost={r["cost"]:.4f}, comp={r["comp"]*100:.1f}%')

# -----------------------------------------------------------------------------
# Experiment 2: Final evaluation comparison (latest 3ES-7MD ES-aware + legacy others)
# -----------------------------------------------------------------------------
def plot_experiment2():
    # Latest ES-aware 3ES-7MD final evals (from train_unified output)
    latest_3es7md = {
        'IPPO+GNN': {'cost': 0.4255, 'std': 0.0118, 'comp': 0.860},
        'ExplabOff+GNN': {'cost': 0.4282, 'std': 0.0125, 'comp': 0.851},
    }

    # Legacy final evals from June 9 comparison (for 2ES-3MD, 2ES-5MD)
    # These use index-based GNN / Standard / Hyper
    legacy = {
        '2ES-3MD': {
            'Standard': {'cost': 0.4720, 'std': 0.0094, 'comp': 0.667},
            'GNN': {'cost': 0.4808, 'std': 0.0312, 'comp': 0.667},
            'Hyper': {'cost': 0.4808, 'std': 0.0453, 'comp': 0.667},
        },
        '2ES-5MD': {
            'Standard': {'cost': 0.4120, 'std': 0.0111, 'comp': 0.880},
            'GNN': {'cost': 0.4112, 'std': 0.0142, 'comp': 0.900},
            'Hyper': {'cost': 0.8588, 'std': 0.1561, 'comp': 0.840},
        },
        '3ES-7MD': {
            'Standard': {'cost': 0.4270, 'std': 0.0129, 'comp': 0.857},
            'GNN': {'cost': 0.4263, 'std': 0.0122, 'comp': 0.743},
            'Hyper': {'cost': 0.4344, 'std': 0.0189, 'comp': 0.871},
        },
    }

    fig, axes = plt.subplots(1, 2, figsize=(14, 4.5))
    configs = ['2ES-3MD', '2ES-5MD', '3ES-7MD']
    networks = ['Standard', 'GNN', 'Hyper']
    x = np.arange(len(configs))
    width = 0.25
    colors = {'Standard': '#4A69BD', 'GNN': '#6A89CC', 'Hyper': '#B8E994'}

    for i, net in enumerate(networks):
        costs = [legacy[c][net]['cost'] for c in configs]
        stds = [legacy[c][net]['std'] for c in configs]
        axes[0].bar(x + (i-1)*width, costs, width, yerr=stds, label=net, color=colors[net], capsize=3)

    # Highlight latest ES-aware on 3ES-7MD as separate markers
    axes[0].scatter([x[2] + 1.5*width], [latest_3es7md['IPPO+GNN']['cost']], marker='D', s=80,
                    color='#C44569', label='IPPO+GNN (ES-aware, latest)', zorder=5)
    axes[0].scatter([x[2] + 1.5*width], [latest_3es7md['ExplabOff+GNN']['cost']], marker='D', s=80,
                    color='#006266', label='ExplabOff+GNN (ES-aware, latest)', zorder=5)

    axes[0].set_xticks(x)
    axes[0].set_xticklabels(configs)
    axes[0].set_ylabel('Average Cost')
    axes[0].set_title('Final Evaluation: Cost (legacy Standard/GNN/Hyper + latest ES-aware GNN)', fontweight='bold')
    axes[0].legend(fontsize=8)
    axes[0].grid(axis='y', alpha=0.3)

    for i, net in enumerate(networks):
        comps = [legacy[c][net]['comp'] * 100 for c in configs]
        axes[1].bar(x + (i-1)*width, comps, width, label=net, color=colors[net])
    axes[1].scatter([x[2] + 1.5*width], [latest_3es7md['IPPO+GNN']['comp']*100], marker='D', s=80,
                    color='#C44569', zorder=5)
    axes[1].scatter([x[2] + 1.5*width], [latest_3es7md['ExplabOff+GNN']['comp']*100], marker='D', s=80,
                    color='#006266', zorder=5)
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(configs)
    axes[1].set_ylabel('Completion Rate (%)')
    axes[1].set_title('Final Evaluation: Completion Rate', fontweight='bold')
    axes[1].legend(fontsize=8)
    axes[1].grid(axis='y', alpha=0.3)
    axes[1].set_ylim([0, 105])

    fig.tight_layout()
    fig.savefig(OUT_DIR / 'exp2_final_evaluation.png', bbox_inches='tight')
    print(f'Saved: {OUT_DIR / "exp2_final_evaluation.png"}')

# -----------------------------------------------------------------------------
# Experiment 1: Convergence comparison (legacy June 9 data)
# -----------------------------------------------------------------------------
def plot_experiment1():
    # Read June 9 comparison log to extract smoothed curves
    log_path = RESULTS / 'comparison_log.txt'
    if not log_path.exists():
        print(f'Warning: {log_path} not found, skipping Experiment 1')
        return

    experiments = {
        'Standard 2ES-3MD': [], 'GNN 2ES-3MD': [], 'Hyper 2ES-3MD': [],
        'Standard 2ES-5MD': [], 'GNN 2ES-5MD': [], 'Hyper 2ES-5MD': [],
        'Standard 3ES-7MD': [], 'GNN 3ES-7MD': [], 'Hyper 3ES-7MD': [],
    }

    current_exp = None
    with open(log_path) as f:
        for line in f:
            line = line.strip()
            for exp in experiments.keys():
                if line.startswith(f'[{exp.split()[0][0]}'):
                    pass
            # Match header lines like "[1/9] Standard 2ES-3MD"
            for exp_name in experiments.keys():
                if f'] {exp_name}' in line:
                    current_exp = exp_name
                    break
            # Match metric lines like "Ep   100 | Cost: 1.5033 | Comp: 36.7%"
            if current_exp and line.startswith('Ep') and 'Cost:' in line:
                parts = line.split('|')
                try:
                    ep = int(parts[0].replace('Ep', '').strip())
                    cost = float(parts[1].replace('Cost:', '').strip())
                    experiments[current_exp].append((ep, cost))
                except Exception:
                    pass

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.2))
    configs = ['2ES-3MD', '2ES-5MD', '3ES-7MD']
    colors = {'Standard': '#4A69BD', 'GNN': '#6A89CC', 'Hyper': '#B8E994'}

    for ax, cfg in zip(axes, configs):
        for net in ['Standard', 'GNN', 'Hyper']:
            key = f'{net} {cfg}'
            data = experiments.get(key, [])
            if data:
                eps, costs = zip(*data)
                # 300-episode rolling min for smoother convergence
                window = 3
                smoothed = []
                for i in range(len(costs)):
                    start = max(0, i - window + 1)
                    smoothed.append(min(costs[start:i+1]))
                ax.plot(eps, smoothed, label=net, color=colors[net], linewidth=2)
        ax.set_xlabel('Episode')
        ax.set_ylabel('Cost (rolling min)')
        ax.set_title(f'{cfg} Convergence', fontweight='bold')
        ax.grid(alpha=0.3)
        ax.legend()

    fig.suptitle('Experiment 1: Same-Config Convergence (legacy index-based networks, June 9)',
                 fontweight='bold', y=1.02)
    fig.tight_layout()
    fig.savefig(OUT_DIR / 'exp1_convergence.png', bbox_inches='tight')
    print(f'Saved: {OUT_DIR / "exp1_convergence.png"}')

if __name__ == '__main__':
    plot_experiment3()
    plot_experiment2()
    plot_experiment1()
    print('\nAll PPT experiment figures generated in', OUT_DIR)
