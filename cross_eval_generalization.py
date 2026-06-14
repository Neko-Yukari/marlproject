"""
Cross-Configuration Generalization Evaluation.

For each trained model (18 total: 3 networks × 2 algorithms × 3 train configs),
evaluates on all 3 test environments to measure generalization capability.

Usage:
    python cross_eval_generalization.py [--models DIR1 DIR2 ...] [--all] [--ippo_only] [--explaboff_only]
    
    # Evaluate specific models
    python cross_eval_generalization.py --models results/ippo_standard_3md2es_20260609_022445
    
    # Evaluate all IPPO models
    python cross_eval_generalization.py --ippo_only
    
    # Evaluate all trained models
    python cross_eval_generalization.py --all
"""
import sys; sys.path.insert(0, '.')

import argparse
import json
import numpy as np
import torch
import yaml
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any

from envs.paper_accurate_env import PaperAccurateEnvV3
from agents.ppo_agent import PPOAgent
from agents.standard_policy import StandardPolicy
from agents.hyper_policy import HyperPolicy
from agents.mi_plugin import MIPlugin


# ── Environment configs to test ──
TEST_CONFIGS = [
    {"name": "2ES-3MD", "num_md": 3, "num_es": 2, "obs_dim": 5, "action_dim": 3},
    {"name": "2ES-5MD", "num_md": 5, "num_es": 2, "obs_dim": 5, "action_dim": 3},
    {"name": "3ES-7MD", "num_md": 7, "num_es": 3, "obs_dim": 7, "action_dim": 4},
]

EVAL_EPISODES = 100
EVAL_SEED_START = 10000  # Different seeds from training


def can_evaluate(network_type: str, train_obs_dim: int, train_action_dim: int,
                 test_cfg: dict) -> bool:
    """Check if a model can be evaluated on a test config.
    
    Standard MLP: only compatible if obs_dim and action_dim match.
    GNN/Hyper: always compatible (architecture handles variable dimensions).
    """
    if network_type in ('GNNPolicy', 'HyperPolicy'):
        return True
    # Standard MLP: dimensions must match
    return (train_obs_dim == test_cfg['obs_dim'] and 
            train_action_dim == test_cfg['action_dim'])


def create_policy(config: dict, device: torch.device):
    """Recreate policy network from config (mirrors train_unified.py logic)."""
    algo_cfg = config['algorithm']
    network_type = algo_cfg.get('network', 'StandardPolicy')
    
    # Infer obs_dim/action_dim from config
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
        from agents.gnn_policy import GNNPolicy
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


def evaluate_model(policy, mi_plugin, model_dir: Path, device: torch.device):
    """Evaluate a loaded model on all compatible test configs.
    
    Returns:
        dict: {test_config_name: {"cost": float, "std": float, "completion": float}}
    """
    config = yaml.safe_load(open(model_dir / 'config.yaml'))
    _, network_type, train_obs_dim, train_action_dim = create_policy(config, device)
    
    results = {}
    
    for test_cfg in TEST_CONFIGS:
        if not can_evaluate(network_type, train_obs_dim, train_action_dim, test_cfg):
            results[test_cfg['name']] = {"compatible": False}
            continue
        
        # Create test environment
        env = PaperAccurateEnvV3(
            num_devices=test_cfg['num_md'],
            num_servers=test_cfg['num_es'],
            randomize_profile=True,
            profile_noise=0.05
        )
        
        # HyperNetwork: update config for test environment
        if network_type == 'HyperPolicy':
            policy.set_config(test_cfg['num_md'], test_cfg['num_es'])
        
        costs, completions = [], []
        
        for ep in range(EVAL_EPISODES):
            obs, _ = env.reset(seed=EVAL_SEED_START + ep)
            
            for step in range(10):
                # Handle GNN: build graph before action selection
                if network_type == 'GNNPolicy':
                    policy.set_graph(env)
                
                actions = {}
                for agent_id in env.agents:
                    md_idx = int(agent_id.split('_')[1])
                    obs_tensor = torch.FloatTensor(obs[agent_id]).to(device)
                    
                    if network_type == 'GNNPolicy':
                        # GNN: set_graph was called above, use obs to index cached embedding
                        action_probs, _ = policy(obs_tensor.unsqueeze(0), agent_id=md_idx)
                    else:
                        action_probs, _ = policy(obs_tensor.unsqueeze(0))
                    
                    # Deterministic: take argmax for evaluation
                    actions[agent_id] = int(torch.argmax(action_probs).item())
                
                obs, rewards, terms, truncs, infos = env.step(actions)
                if all(terms.values()) or all(truncs.values()):
                    break
            
            metrics = env.get_episode_metrics()
            costs.append(metrics['avg_cost'])
            completions.append(metrics['completion_rate'])
        
        results[test_cfg['name']] = {
            "compatible": True,
            "cost_mean": float(np.mean(costs)),
            "cost_std": float(np.std(costs)),
            "completion": float(np.mean(completions)),
            "best_cost": float(np.min(costs)),
            "worst_cost": float(np.max(costs)),
        }
        
        print(f"  {test_cfg['name']}: cost={np.mean(costs):.4f}±{np.std(costs):.4f}, "
              f"comp={np.mean(completions)*100:.1f}%")
    
    return results


def find_models(results_dir: Path, algo_filter: str = None) -> List[Path]:
    """Find all model directories (containing policy.pt and config.yaml)."""
    models = []
    for subdir in sorted(results_dir.iterdir()):
        if not subdir.is_dir():
            continue
        if (subdir / 'policy.pt').exists() and (subdir / 'config.yaml').exists():
            # Apply filter
            if algo_filter:
                dirname = subdir.name.lower()
                if algo_filter not in dirname:
                    continue
            models.append(subdir)
    return models


def parse_train_config(model_name: str) -> Optional[str]:
    """Parse training config from model directory name.
    
    Examples:
        'ippo_standard_3md2es_20260609_022445' → '2ES-3MD'
        'explaboff_gnn_7md3es_20260609_...' → '3ES-7MD'
    """
    for cfg_info in TEST_CONFIGS:
        es_md = f"{cfg_info['num_es']}es{cfg_info['num_md']}md"
        md_es = f"{cfg_info['num_md']}md{cfg_info['num_es']}es"
        if es_md in model_name.lower() or md_es in model_name.lower():
            return cfg_info['name']
    return None


def build_matrix(all_results: Dict[str, dict]):
    """Build generalization matrix from all results."""
    # Organize: network × algo × train_config → test_config results
    matrix = {}
    for model_name, results in all_results.items():
        parts = model_name.split('_')
        algo = 'explaboff' if 'explaboff' in model_name else 'ippo'
        
        if 'gnn' in parts:
            network = 'GNN'
        elif 'hyper' in parts:
            network = 'Hyper'
        else:
            network = 'Standard'
        
        train_cfg = parse_train_config(model_name)
        
        key = (network, algo, train_cfg)
        matrix[key] = results
    
    return matrix


def print_matrix(matrix: Dict, output_path: Path = None):
    """Print generalization matrix as a formatted table."""
    lines = []
    lines.append("=" * 80)
    lines.append("GENERALIZATION MATRIX: Train → Test (Cost ± std, Completion%)")
    lines.append("=" * 80)
    
    for (network, algo, train_cfg), results in sorted(matrix.items()):
        lines.append(f"\n--- {network} | {algo.upper()} | Trained on {train_cfg} ---")
        for test_cfg in TEST_CONFIGS:
            r = results.get(test_cfg['name'], {})
            if r.get('compatible') is False:
                lines.append(f"  {train_cfg:>10} -> {test_cfg['name']:<10}: INCOMPATIBLE")
            elif r:
                is_train = (test_cfg['name'] == train_cfg)
                marker = " [SAME]" if is_train else ""
                lines.append(
                    f"  {train_cfg:>10} -> {test_cfg['name']:<10}: "
                    f"cost={r['cost_mean']:.4f}+/-{r['cost_std']:.4f}, "
                    f"comp={r['completion']*100:.1f}%{marker}"
                )
    
    lines.append("\n" + "=" * 80)
    lines.append("GENERALIZATION GAP (cost on unseen config - cost on training config)")
    lines.append("=" * 80)
    
    for (network, algo, train_cfg), results in sorted(matrix.items()):
        train_cost = None
        gaps = []
        for test_cfg in TEST_CONFIGS:
            r = results.get(test_cfg['name'], {})
            if test_cfg['name'] == train_cfg and r.get('compatible'):
                train_cost = r['cost_mean']
                break
        
        if train_cost is None:
            continue
        
        lines.append(f"\n{network} | {algo.upper()} | Trained {train_cfg} (baseline={train_cost:.4f})")
        for test_cfg in TEST_CONFIGS:
            r = results.get(test_cfg['name'], {})
            if test_cfg['name'] == train_cfg:
                continue
            if r.get('compatible') is False:
                lines.append(f"  -> {test_cfg['name']:<10}: N/A (incompatible)")
            elif r:
                gap = r['cost_mean'] - train_cost
                status = "OK" if gap < 0.05 else "WARN" if gap < 0.15 else "BAD"
                lines.append(
                    f"  -> {test_cfg['name']:<10}: +{gap:+.4f} {status}"
                )
    
    output = "\n".join(lines)
    print(output)
    
    if output_path:
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(output)
        print(f"\nReport saved to: {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Cross-config generalization evaluation")
    parser.add_argument('--models', nargs='+', help='Specific model directories to evaluate')
    parser.add_argument('--all', action='store_true', help='Evaluate all trained models')
    parser.add_argument('--ippo_only', action='store_true', help='Only IPPO models')
    parser.add_argument('--explaboff_only', action='store_true', help='Only ExplabOff models')
    parser.add_argument('--output', type=str, default='results/generalization_report.txt',
                       help='Output report path')
    parser.add_argument('--device', type=str, default='cuda', help='Device (cuda/cpu)')
    args = parser.parse_args()
    
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    
    results_dir = Path('results')
    
    # Find models
    if args.models:
        model_dirs = [Path(m) for m in args.models]
    elif args.ippo_only:
        model_dirs = find_models(results_dir, algo_filter='ippo')
    elif args.explaboff_only:
        model_dirs = find_models(results_dir, algo_filter='explaboff')
    elif args.all:
        model_dirs = find_models(results_dir)
    else:
        # Default: latest IPPO models (June 9 comparison run only)
        model_dirs = find_models(results_dir, algo_filter='ippo')
        model_dirs = [m for m in model_dirs if '20260609' in str(m)]
        # Only keep unique configs (latest run for each)
        seen = set()
        filtered = []
        for m in sorted(model_dirs, reverse=True):
            # Key: network_algo_config (e.g., standard_ippo_3md2es)
            parts = m.name.split('_')
            if 'gnn' in parts: net='gnn'
            elif 'hyper' in parts: net='hyper'
            else: net='standard'
            cfg = parse_train_config(m.name.replace('-', ''))
            key = f"{net}_{cfg}"
            if key not in seen:
                seen.add(key)
                filtered.append(m)
        model_dirs = sorted(filtered)
        if not model_dirs:
            print("No June 9 IPPO models found. Use --models or --all.")
            return
    
    print(f"Found {len(model_dirs)} models to evaluate")
    
    all_results = {}
    
    for model_dir in model_dirs:
        name = model_dir.name
        print(f"\n{'='*60}")
        print(f"Evaluating: {name}")
        
        try:
            config = yaml.safe_load(open(model_dir / 'config.yaml'))
        except Exception as e:
            print(f"  SKIP: cannot read config.yaml ({e})")
            continue
        
        # Create policy
        try:
            policy, network_type, obs_dim, action_dim = create_policy(config, device)
        except Exception as e:
            print(f"  SKIP: cannot create policy ({e})")
            continue
        
        # Load weights
        try:
            policy.load_state_dict(torch.load(model_dir / 'policy.pt', map_location=device))
        except Exception as e:
            print(f"  SKIP: cannot load policy.pt ({e})")
            continue
        
        # Load MI plugin if exists
        mi_plugin = None
        mi_path = model_dir / 'mi_plugin.pt'
        if mi_path.exists():
            mi_plugin = MIPlugin(
                state_dim=obs_dim,
                action_dim=action_dim,
                hidden_dim=config['algorithm'].get('hidden_dim', 128),
                mu=config['algorithm'].get('mi_mu', 3.5),
                nu=config['algorithm'].get('mi_nu', 1.0),
                device=device
            )
            mi_plugin.load_state_dict(torch.load(mi_path, map_location=device))
        
        # Evaluate on all compatible test configs
        results = evaluate_model(policy, mi_plugin, model_dir, device)
        all_results[name] = results
    
    # Build and print matrix
    matrix = build_matrix(all_results)
    output_path = Path(args.output)
    print_matrix(matrix, output_path)
    
    # Also save JSON results
    json_path = output_path.with_suffix('.json')
    with open(json_path, 'w') as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f"\nJSON results saved to: {json_path}")


if __name__ == '__main__':
    main()
