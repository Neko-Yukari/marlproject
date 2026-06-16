"""
Analyze Generalization Evolution Across Training Checkpoints.

Loads checkpoints from IPPO+GNN and ExplabOff+GNN models trained on 3ES-7MD,
evaluates each checkpoint on 2ES-3MD, 2ES-5MD, 3ES-7MD, and tracks:
- Cost and completion evolution
- Action distribution per MD
- ES selection bias

Usage:
    python analyze_generalization_evolution.py \
        --ippo results/ippo_gnn_7md3es_20260617_032200 \
        --ippo_config results/ippo_gnn_7md3es_20260617_033932 \
        --explaboff results/explaboff_gnn_7md3es_20260617_042137 \
        --explaboff_config results/explaboff_gnn_7md3es_20260617_050536
"""
import sys; sys.path.insert(0, '.')

import argparse
import json
import numpy as np
import torch
import yaml
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from collections import defaultdict

from envs.paper_accurate_env import PaperAccurateEnvV3
from agents.ppo_agent import PPOAgent
from agents.standard_policy import StandardPolicy
from agents.hyper_policy import HyperPolicy
from agents.mi_plugin import MIPlugin
from agents.gnn_policy import GNNPolicy


TEST_CONFIGS = [
    {"name": "2ES-3MD", "num_md": 3, "num_es": 2},
    {"name": "2ES-5MD", "num_md": 5, "num_es": 2},
    {"name": "3ES-7MD", "num_md": 7, "num_es": 3},
]

EVAL_EPISODES = 100
EVAL_SEED_START = 10000


def load_config(model_dir: Path) -> dict:
    with open(model_dir / 'config.yaml', 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)


def create_policy(config: dict, device: torch.device):
    """Recreate policy network from config."""
    algo_cfg = config['algorithm']
    network_type = algo_cfg.get('network', 'StandardPolicy')
    
    M = config['environment']['num_md']
    E = config['environment']['num_es']
    obs_dim = 1 + E + E
    action_dim = E + 1
    
    if network_type == 'StandardPolicy':
        policy = StandardPolicy(
            state_dim=obs_dim,
            action_dim=action_dim,
            hidden_dim=algo_cfg.get('hidden_dim', 128),
            num_layers=algo_cfg.get('num_layers', 2)
        ).to(device)
    
    elif network_type == 'HyperPolicy':
        policy = HyperPolicy(
            max_obs_dim=algo_cfg.get('max_obs_dim', 7),
            max_action_dim=algo_cfg.get('max_action_dim', 4),
            hidden_dim=algo_cfg.get('hidden_dim', 256)
        ).to(device)
        policy.set_config(M, E)
    
    elif network_type == 'GNNPolicy':
        policy = GNNPolicy(
            max_action_dim=algo_cfg.get('max_action_dim', 4),
            hidden_dim=algo_cfg.get('hidden_dim', 128),
            gnn_layers=algo_cfg.get('gnn_layers', 1),
            node_dim=algo_cfg.get('node_dim', 4),
            max_md=algo_cfg.get('max_md', 10)
        ).to(device)
    
    else:
        raise ValueError(f"Unknown network type: {network_type}")
    
    return policy, network_type, obs_dim, action_dim


def load_checkpoint(model_dir: Path, checkpoint_dir: Path, device: torch.device):
    """Load policy (and mi_plugin if exists) from checkpoint."""
    config = load_config(model_dir)
    policy, network_type, obs_dim, action_dim = create_policy(config, device)
    
    policy.load_state_dict(torch.load(checkpoint_dir / 'policy.pt', map_location=device))
    policy.eval()
    
    mi_plugin = None
    mi_path = checkpoint_dir / 'mi_plugin.pt'
    if mi_path.exists():
        algo_cfg = config['algorithm']
        mi_plugin = MIPlugin(
            state_dim=obs_dim,
            action_dim=action_dim,
            hidden_dim=128,
            mu=algo_cfg.get('mi_mu', 3.5),
            nu=algo_cfg.get('mi_nu', 1.0),
            device=device
        )
        mi_plugin.load_state_dict(torch.load(mi_path, map_location=device))
    
    return policy, mi_plugin, network_type, config


def evaluate_checkpoint(policy, mi_plugin, network_type: str, test_cfg: dict,
                       device: torch.device, num_episodes: int = EVAL_EPISODES):
    """Evaluate a checkpoint on one test config.
    
    Returns:
        dict with cost_mean, cost_std, completion, action_counts, action_dist
    """
    env = PaperAccurateEnvV3(
        num_devices=test_cfg['num_md'],
        num_servers=test_cfg['num_es'],
        randomize_profile=True,
        profile_noise=0.05
    )
    
    if network_type == 'HyperPolicy':
        policy.set_config(test_cfg['num_md'], test_cfg['num_es'])
    
    costs, completions = [], []
    # action_counts[md_idx][action] = count
    action_counts = defaultdict(lambda: defaultdict(int))
    total_actions = 0
    
    for ep in range(num_episodes):
        obs, _ = env.reset(seed=EVAL_SEED_START + ep)
        
        for step in range(10):
            if network_type == 'GNNPolicy':
                policy.set_graph(env)
            
            actions = {}
            for agent_id in env.agents:
                md_idx = int(agent_id.split('_')[1])
                obs_tensor = torch.FloatTensor(obs[agent_id]).to(device)
                
                if network_type == 'GNNPolicy':
                    action_probs, _ = policy(obs_tensor.unsqueeze(0), agent_id=md_idx)
                else:
                    action_probs, _ = policy(obs_tensor.unsqueeze(0))
                
                action = int(torch.argmax(action_probs).item())
                actions[agent_id] = action
                action_counts[md_idx][action] += 1
                total_actions += 1
            
            obs, rewards, terms, truncs, infos = env.step(actions)
            if all(terms.values()) or all(truncs.values()):
                break
        
        metrics = env.get_episode_metrics()
        costs.append(metrics['avg_cost'])
        completions.append(metrics['completion_rate'])
    
    # Convert action_counts to normalized distribution
    action_dist = {}
    for md_idx in sorted(action_counts.keys()):
        total = sum(action_counts[md_idx].values())
        action_dist[md_idx] = {a: c / total for a, c in action_counts[md_idx].items()}
    
    # Overall action distribution (average across MDs)
    overall = defaultdict(float)
    for md_dist in action_dist.values():
        for a, p in md_dist.items():
            overall[a] += p / len(action_dist)
    
    return {
        'cost_mean': float(np.mean(costs)),
        'cost_std': float(np.std(costs)),
        'completion': float(np.mean(completions)),
        'action_counts': {str(k): dict(v) for k, v in action_counts.items()},
        'action_dist': {str(k): v for k, v in action_dist.items()},
        'overall_action_dist': dict(overall),
    }


def analyze_model(model_dir: Path, config_dir: Path, device: torch.device, label: str):
    """Analyze all checkpoints for one model."""
    print(f"\n{'='*60}")
    print(f"Analyzing: {label}")
    print(f"Model dir (checkpoints): {model_dir}")
    print(f"Config dir: {config_dir}")
    print(f"{'='*60}")
    
    checkpoint_dirs = sorted(model_dir.glob('checkpoint_ep*'))
    if not checkpoint_dirs:
        # Try final model dir as single checkpoint
        checkpoint_dirs = [model_dir]
    
    results = []
    for ckpt_dir in checkpoint_dirs:
        ep_str = ckpt_dir.name.replace('checkpoint_ep', '')
        episode = int(ep_str) if ep_str.isdigit() else 9999
        
        print(f"\n[Checkpoint ep{episode}]")
        policy, mi_plugin, network_type, config = load_checkpoint(config_dir, ckpt_dir, device)
        
        ckpt_result = {'episode': episode, 'test_configs': {}}
        for test_cfg in TEST_CONFIGS:
            print(f"  Evaluating on {test_cfg['name']}...", end='')
            res = evaluate_checkpoint(policy, mi_plugin, network_type, test_cfg, device)
            ckpt_result['test_configs'][test_cfg['name']] = res
            print(f" cost={res['cost_mean']:.4f}±{res['cost_std']:.4f}, "
                  f"comp={res['completion']*100:.1f}%")
        
        results.append(ckpt_result)
    
    return results


def plot_evolution(results_ippo: List[dict], results_explaboff: List[dict],
                   output_dir: Path):
    """Plot generalization evolution across checkpoints."""
    output_dir.mkdir(parents=True, exist_ok=True)
    
    episodes = [r['episode'] for r in results_ippo]
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    fig.suptitle('Generalization Evolution: IPPO+GNN vs ExplabOff+GNN (trained on 3ES-7MD)', fontsize=14)
    
    test_names = [c['name'] for c in TEST_CONFIGS]
    colors = {'IPPO+GNN': '#2E86AB', 'ExplabOff+GNN': '#A23B72'}
    
    for col, test_name in enumerate(test_names):
        # Top row: cost
        ax = axes[0, col]
        for label, results in [('IPPO+GNN', results_ippo), ('ExplabOff+GNN', results_explaboff)]:
            costs = [r['test_configs'][test_name]['cost_mean'] for r in results]
            ax.plot(episodes, costs, marker='o', label=label, color=colors[label])
        ax.set_xlabel('Training Episode')
        ax.set_ylabel('Avg Cost')
        ax.set_title(f'Test on {test_name}')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Bottom row: completion
        ax = axes[1, col]
        for label, results in [('IPPO+GNN', results_ippo), ('ExplabOff+GNN', results_explaboff)]:
            comps = [r['test_configs'][test_name]['completion'] * 100 for r in results]
            ax.plot(episodes, comps, marker='o', label=label, color=colors[label])
        ax.set_xlabel('Training Episode')
        ax.set_ylabel('Completion Rate (%)')
        ax.set_title(f'Completion on {test_name}')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'generalization_evolution.png', dpi=150)
    plt.close()
    print(f"\nSaved evolution plot: {output_dir / 'generalization_evolution.png'}")
    
    # Plot action distribution evolution for 2ES-3MD (where divergence is clearest)
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle('Action Distribution Evolution on 2ES-3MD (test set)', fontsize=14)
    
    actions = ['Local', 'ES1', 'ES2']
    for idx, (label, results) in enumerate([('IPPO+GNN', results_ippo), ('ExplabOff+GNN', results_explaboff)]):
        ax = axes[idx]
        
        # Stack data: rows=episodes, cols=actions
        data = np.zeros((len(results), 3))
        for i, r in enumerate(results):
            dist = r['test_configs']['2ES-3MD']['overall_action_dist']
            for a in range(3):
                data[i, a] = dist.get(a, 0.0)
        
        bottom = np.zeros(len(results))
        for a, action_name in enumerate(actions):
            ax.bar(episodes, data[:, a], bottom=bottom, label=action_name, width=600)
            bottom += data[:, a]
        
        ax.set_xlabel('Training Episode')
        ax.set_ylabel('Action Proportion')
        ax.set_title(label)
        ax.legend()
        ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig(output_dir / 'action_distribution_evolution_2es3md.png', dpi=150)
    plt.close()
    print(f"Saved action distribution plot: {output_dir / 'action_distribution_evolution_2es3md.png'}")


def print_report(results_ippo: List[dict], results_explaboff: List[dict]):
    """Print a concise analysis report."""
    print("\n" + "="*80)
    print("GENERALIZATION EVOLUTION REPORT")
    print("="*80)
    
    for test_name in [c['name'] for c in TEST_CONFIGS]:
        print(f"\n--- Test on {test_name} ---")
        print(f"{'Ep':>6} | {'IPPO Cost':>12} | {'IPPO Comp':>10} | "
              f"{'Expl Cost':>12} | {'Expl Comp':>10} | {'Gap':>8}")
        print("-" * 75)
        for r_ippo, r_expl in zip(results_ippo, results_explaboff):
            ep = r_ippo['episode']
            i_cost = r_ippo['test_configs'][test_name]['cost_mean']
            i_comp = r_ippo['test_configs'][test_name]['completion'] * 100
            e_cost = r_expl['test_configs'][test_name]['cost_mean']
            e_comp = r_expl['test_configs'][test_name]['completion'] * 100
            gap = e_cost - i_cost
            print(f"{ep:>6} | {i_cost:>12.4f} | {i_comp:>9.1f}% | "
                  f"{e_cost:>12.4f} | {e_comp:>9.1f}% | {gap:>+8.4f}")
    
    # Find when divergence starts
    print("\n--- Divergence Analysis (2ES-3MD) ---")
    diverged_ep = None
    for r_ippo, r_expl in zip(results_ippo, results_explaboff):
        i_cost = r_ippo['test_configs']['2ES-3MD']['cost_mean']
        e_cost = r_expl['test_configs']['2ES-3MD']['cost_mean']
        if e_cost - i_cost > 0.05 and diverged_ep is None:
            diverged_ep = r_ippo['episode']
            print(f"ExplabOff cost exceeds IPPO by >0.05 starting at episode {diverged_ep}")
            print(f"  IPPO: {i_cost:.4f}, ExplabOff: {e_cost:.4f}")
            break
    if diverged_ep is None:
        print("No significant divergence detected")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--ippo', type=str, required=True, help='IPPO+GNN checkpoint root directory')
    parser.add_argument('--ippo_config', type=str, default=None, help='IPPO+GNN config directory (default: same as --ippo)')
    parser.add_argument('--explaboff', type=str, required=True, help='ExplabOff+GNN checkpoint root directory')
    parser.add_argument('--explaboff_config', type=str, default=None, help='ExplabOff+GNN config directory (default: same as --explaboff)')
    parser.add_argument('--output_dir', type=str, default='results/generalization_evolution')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
    args = parser.parse_args()
    
    device = torch.device(args.device)
    ippo_dir = Path(args.ippo)
    ippo_config_dir = Path(args.ippo_config) if args.ippo_config else ippo_dir
    expl_dir = Path(args.explaboff)
    expl_config_dir = Path(args.explaboff_config) if args.explaboff_config else expl_dir
    output_dir = Path(args.output_dir)
    
    results_ippo = analyze_model(ippo_dir, ippo_config_dir, device, 'IPPO+GNN (3ES-7MD)')
    results_explaboff = analyze_model(expl_dir, expl_config_dir, device, 'ExplabOff+GNN (3ES-7MD)')
    
    # Save raw results
    all_results = {
        'ippo_gnn': results_ippo,
        'explaboff_gnn': results_explaboff,
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    with open(output_dir / 'evolution_results.json', 'w') as f:
        json.dump(all_results, f, indent=2)
    print(f"\nSaved results: {output_dir / 'evolution_results.json'}")
    
    # Plot
    plot_evolution(results_ippo, results_explaboff, output_dir)
    
    # Report
    print_report(results_ippo, results_explaboff)


if __name__ == '__main__':
    main()
