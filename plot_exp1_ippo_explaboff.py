import json
import re
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

RESULTS_DIR = Path('results')
OUT_DIR = RESULTS_DIR / 'ppt_experiments'
OUT_DIR.mkdir(exist_ok=True)


def load_history(model_dir):
    path = Path(model_dir) / 'history.json'
    if not path.exists():
        return None
    with open(path) as f:
        data = json.load(f)
    # Support both list-of-dicts and dict-with-cost_history
    if isinstance(data, list):
        return {'cost_history': [h.get('avg_cost', h.get('cost', 0)) for h in data],
                'comp_history': [h.get('completion_rate', h.get('comp', 0)) for h in data]}
    return data


def moving_min(values, window=100):
    """Minimum over a sliding window."""
    values = np.asarray(values)
    n = len(values)
    out = np.full(n, np.nan)
    for i in range(n):
        start = max(0, i - window + 1)
        out[i] = values[start:i+1].min()
    return out


def parse_comparison_log():
    """Read legacy comparison_log.txt for IPPO Standard/GNN/Hyper across 3 configs."""
    log_path = RESULTS_DIR / 'comparison_log.txt'
    if not log_path.exists():
        return {}
    curves = {}
    # Format: "[1/9] Standard 2ES-3MD" or "[1/9] GNN 2ES-5MD"
    pattern = re.compile(r"\[\d+/\d+\]\s+(?P<net>\w+)\s+(?P<cfg>\dES-\dMD)")
    current = None
    with open(log_path, encoding='utf-8') as f:
        for line in f:
            m = pattern.search(line)
            if m:
                current = ('IPPO', m.group('net'), m.group('cfg'))
                curves.setdefault(current, {'cost': [], 'comp': []})
            if current and 'cost=' in line and 'comp=' in line:
                cm = re.search(r'cost=([0-9.]+).*?comp=([0-9.]+)', line)
                if cm:
                    curves[current]['cost'].append(float(cm.group(1)))
                    curves[current]['comp'].append(float(cm.group(2)))
    return curves


def load_comparison_json(network, config):
    """Load IPPO comparison JSON with history."""
    path = RESULTS_DIR / f'comparison_{config}_{network}.json'
    if not path.exists():
        return None
    with open(path) as f:
        data = json.load(f)
    return [h['cost'] for h in data.get('history', [])]


def get_latest_model_dirs():
    """Find latest ES-aware GNN model dirs for IPPO/ExplabOff on 3ES-7MD."""
    dirs = {}
    for algo, prefix in [('IPPO', 'ippo_gnn_7md3es'), ('ExplabOff', 'explaboff_gnn_7md3es')]:
        candidates = sorted(RESULTS_DIR.glob(f'{prefix}_*'), key=lambda p: p.name)
        if candidates:
            dirs[algo] = candidates[-1]
    return dirs


def collect_exp1_data():
    """Collect curves for Experiment 1."""
    data = {}
    configs = ['2ES-3MD', '2ES-5MD', '3ES-7MD']
    networks = ['Standard-MLP', 'GNN', 'HyperNet']
    net_files = {'Standard-MLP': 'Standard', 'GNN': 'GNN', 'HyperNet': 'HyperNetwork'}
    algos = ['IPPO', 'ExplabOff']
    
    for algo in algos:
        data[algo] = {cfg: {net: None for net in networks} for cfg in configs}
    
    # IPPO legacy comparison JSONs (and txt fallback)
    log_curves = parse_comparison_log()
    for cfg in configs:
        for net, fname in net_files.items():
            curve = load_comparison_json(fname, cfg)
            if curve is None:
                key = ('IPPO', net if net != 'Standard-MLP' else 'Standard', cfg)
                if key in log_curves:
                    curve = log_curves[key]['cost']
            data['IPPO'][cfg][net] = curve
    
    # ExplabOff 3ES-7MD from latest ES-aware models
    latest = get_latest_model_dirs()
    for algo in ['IPPO', 'ExplabOff']:
        if algo in latest:
            hist = load_history(latest[algo])
            if hist and 'cost_history' in hist:
                data[algo]['3ES-7MD']['GNN'] = hist['cost_history']
    
    # ExplabOff Standard/Hyper on 3ES-7MD from existing legacy results
    for net, pattern in [('Standard-MLP', 'explaboff_standard_7md3es_*'), 
                         ('HyperNet', 'explaboff_hyper_7md3es_*')]:
        candidates = sorted(RESULTS_DIR.glob(pattern), key=lambda p: p.name)
        if candidates:
            hist = load_history(candidates[-1])
            if hist and 'cost_history' in hist:
                data['ExplabOff']['3ES-7MD'][net] = hist['cost_history']
    
    return data


def plot_experiment1():
    data = collect_exp1_data()
    configs = ['2ES-3MD', '2ES-5MD', '3ES-7MD']
    networks = ['Standard-MLP', 'GNN', 'HyperNet']
    
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.5))
    
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
    
    for idx, cfg in enumerate(configs):
        ax = axes[idx]
        for algo in ['IPPO', 'ExplabOff']:
            for net in networks:
                curve = data[algo][cfg][net]
                if curve is None or len(curve) == 0:
                    continue
                curve_smooth = moving_min(curve, window=100)
                label = f'{algo}+{net}'
                ax.plot(
                    range(len(curve_smooth)), curve_smooth,
                    color=colors[(algo, net)],
                    linestyle=linestyles[algo],
                    linewidth=1.8,
                    label=label
                )
        ax.set_title(cfg, fontsize=13, fontweight='bold')
        ax.set_xlabel('Episode')
        if idx == 0:
            ax.set_ylabel('Cost (100-episode moving min)')
        ax.grid(alpha=0.3)
        ax.set_ylim([0.35, 1.6])
        if idx == 2:
            ax.legend(fontsize=8, loc='upper right')
    
    fig.suptitle('Experiment 1: IPPO vs ExplabOff Convergence Across Network Architectures', 
                 fontsize=14, fontweight='bold')
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    output = OUT_DIR / 'exp1_convergence.png'
    plt.savefig(output, dpi=300, bbox_inches='tight')
    print(f'Saved: {output}')


if __name__ == '__main__':
    plot_experiment1()
