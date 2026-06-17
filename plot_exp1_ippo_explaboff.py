import json
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

RESULTS_DIR = Path('results')
OUT_DIR = RESULTS_DIR / 'ppt_experiments'
OUT_DIR.mkdir(exist_ok=True)


def load_history(model_dir):
    """Load history.json, return (episodes, costs) or None."""
    path = Path(model_dir) / 'history.json'
    if not path.exists():
        return None
    with open(path) as f:
        data = json.load(f)
    if isinstance(data, list):
        eps = [h['episode'] for h in data]
        costs = [h.get('avg_cost', h.get('cost')) for h in data]
        return eps, costs
    if 'cost_history' in data and 'episode_history' in data:
        return data['episode_history'], data['cost_history']
    return None


def load_comparison_json(network, config):
    """Load IPPO comparison JSON, return (episodes, costs) or None."""
    path = RESULTS_DIR / f'comparison_{config}_{network}.json'
    if not path.exists():
        return None
    with open(path) as f:
        data = json.load(f)
    eps = [h['episode'] for h in data['history']]
    costs = [h['cost'] for h in data['history']]
    return eps, costs


def moving_min_safe(episodes, costs, window_eps=500):
    """Minimum over trailing window_eps episodes."""
    eps = np.asarray(episodes)
    vals = np.asarray(costs, dtype=float)
    result = np.full(len(eps), np.nan)
    for i in range(len(eps)):
        start = np.searchsorted(eps, eps[i] - window_eps)
        result[i] = vals[start:i+1].min()
    return result


def collect_exp1_data():
    data = {}
    configs = ['2ES-3MD', '2ES-5MD', '3ES-7MD']
    net_files = {'Standard-MLP': 'Standard', 'GNN': 'GNN', 'HyperNet': 'HyperNetwork'}
    net_prefix = {'Standard-MLP': 'standard', 'GNN': 'gnn', 'HyperNet': 'hyper'}
    algos = ['IPPO', 'ExplabOff']
    
    for algo in algos:
        data[algo] = {cfg: {net: None for net in net_files} for cfg in configs}
    
    # IPPO — from comparison JSONs
    for cfg in configs:
        for net, fname in net_files.items():
            curve = load_comparison_json(fname, cfg)
            if curve:
                data['IPPO'][cfg][net] = curve
    
    # IPPO 3ES-7MD GNN — override with latest ES-aware
    esp = sorted(RESULTS_DIR.glob('ippo_gnn_7md3es_*/'), key=lambda p: p.name)
    if esp:
        h = load_history(esp[-1])
        if h:
            data['IPPO']['3ES-7MD']['GNN'] = h
    
    # ExplabOff — from latest history.json
    for net in net_files:
        for cfg in configs:
            parts = cfg.split('-')  # "2ES-3MD"
            es_num = parts[0].replace('ES', '')
            md_num = parts[1].replace('MD', '')
            pat = f'explaboff_{net_prefix[net]}_{md_num}md{es_num}es_*'
            dirs = sorted([p for p in RESULTS_DIR.glob(pat) if p.is_dir()], key=lambda p: p.name)
            if dirs:
                h = load_history(dirs[-1])
                if h:
                    data['ExplabOff'][cfg][net] = h
    
    return data


def plot_experiment1():
    data = collect_exp1_data()
    configs = ['2ES-3MD', '2ES-5MD', '3ES-7MD']
    networks = ['Standard-MLP', 'GNN', 'HyperNet']
    
    colors = {
        ('IPPO', 'Standard-MLP'): '#E55039',
        ('IPPO', 'GNN'): '#4A69BD',
        ('IPPO', 'HyperNet'): '#78E08F',
        ('ExplabOff', 'Standard-MLP'): '#B33939',
        ('ExplabOff', 'GNN'): '#1E3799',
        ('ExplabOff', 'HyperNet'): '#218F66',
    }
    linestyles = {'IPPO': '-', 'ExplabOff': '--'}
    markers = {'Standard-MLP': 'o', 'GNN': 's', 'HyperNet': '^'}
    
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.5))
    
    all_handles, all_labels = [], []
    
    for idx, cfg in enumerate(configs):
        ax = axes[idx]
        for algo in ['IPPO', 'ExplabOff']:
            for net in networks:
                curve = data[algo][cfg][net]
                if curve is None or len(curve[0]) == 0:
                    continue
                eps, costs = curve
                smooth = moving_min_safe(eps, costs, window_eps=500)
                label = f'{algo}+{net}'
                line, = ax.plot(
                    eps, smooth,
                    color=colors[(algo, net)],
                    linestyle=linestyles[algo],
                    linewidth=1.6,
                    label=label
                )
                all_handles.append(line)
                all_labels.append(label)
        
        ax.set_title(cfg, fontsize=13, fontweight='bold')
        ax.set_xlabel('Episode')
        ax.set_xlim([0, 10000])
        ax.set_ylim([0.35, 1.6])
        ax.grid(alpha=0.3)
        if idx == 0:
            ax.set_ylabel('Cost (500-ep moving min)')
    
    # Single clean legend
    fig.legend(all_handles, all_labels, loc='lower center', ncol=6, fontsize=8)
    plt.tight_layout(rect=[0, 0.08, 1, 1])
    
    output = OUT_DIR / 'exp1_convergence.png'
    plt.savefig(output, dpi=300, bbox_inches='tight')
    print(f'Saved: {output}')


if __name__ == '__main__':
    plot_experiment1()
