"""
HyperNetwork Ablation Study - Automated Comparison.

Trains multiple HyperNetwork variants and compares performance.
"""
import os
os.environ['MKL_THREADING_LAYER'] = 'GNU'
os.environ['KMP_AFFINITY'] = 'disabled'

import torch
import numpy as np
import time
import json
from pathlib import Path

from envs.paper_accurate_env_v3 import PaperAccurateEnvV3
from agents.hypernetwork_variants import (
    HyperNetworkV1_LowLR,
    HyperNetworkV2_Large,
    HyperNetworkV3_LayerNorm,
    HyperNetworkV4_Curriculum,
    CrossScaleAgent
)
from utils.opencode_notifier import notify_training_complete


# Experiment configurations
EXPERIMENTS = [
    {
        "name": "V1_LowLR",
        "model_class": HyperNetworkV1_LowLR,
        "lr": 1e-5,
        "hidden_dim": 128,
        "description": "Lower learning rate (1e-5)"
    },
    {
        "name": "V2_Large",
        "model_class": HyperNetworkV2_Large,
        "lr": 5e-5,
        "hidden_dim": 256,
        "description": "Larger hidden dimension (256)"
    },
    {
        "name": "V3_LayerNorm",
        "model_class": HyperNetworkV3_LayerNorm,
        "lr": 5e-5,
        "hidden_dim": 128,
        "description": "LayerNorm for stability"
    },
    {
        "name": "V4_Curriculum",
        "model_class": HyperNetworkV4_Curriculum,
        "lr": 5e-5,
        "hidden_dim": 128,
        "description": "Curriculum learning strategy",
        "curriculum": True
    }
]


def make_serializable_config(exp_config):
    """Convert experiment config to JSON-serializable dict."""
    cfg = {}
    for k, v in exp_config.items():
        if k == 'model_class':
            cfg[k] = v.__name__ if hasattr(v, '__name__') else str(v)
        else:
            cfg[k] = v
    return cfg


def train_variant(exp_config, num_episodes=10000, eval_interval=1000, 
                  save_dir="results/hypernetwork_ablation"):
    """Train a single HyperNetwork variant."""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\n{'='*60}")
    print(f"Training: {exp_config['name']}")
    print(f"Description: {exp_config['description']}")
    print(f"{'='*60}")
    
    # Configurations
    configs = [
        (3, 2, "2ES-3MD"),
        (5, 2, "2ES-5MD"),
        (7, 3, "3ES-7MD"),
    ]
    
    # Create model
    model_class = exp_config['model_class']
    hyper_net = model_class(
        obs_dim=7,
        max_action_dim=4,
        hidden_dim=exp_config.get('hidden_dim', 128)
    ).to(device)
    
    print(f"Parameters: {sum(p.numel() for p in hyper_net.parameters())}")
    
    # Create environments and agents
    envs = {}
    agents = {}
    for M, E, name in configs:
        envs[name] = PaperAccurateEnvV3(num_devices=M, num_servers=E, randomize_profile=True)
        agents[name] = [CrossScaleAgent(i, hyper_net, device=device, lr=exp_config['lr']) 
                        for i in range(M)]
    
    # Training loop
    history = {name: [] for _, _, name in configs}
    best_costs = {name: float('inf') for _, _, name in configs}
    start_time = time.time()
    
    # Curriculum learning schedule
    curriculum_stage = 0
    curriculum_schedule = [
        (0, 0.5, ["2ES-3MD"]),  # First 50%: only 2ES-3MD
        (0.5, 0.8, ["2ES-3MD", "2ES-5MD"]),  # Next 30%: add 2ES-5MD
        (0.8, 1.0, ["2ES-3MD", "2ES-5MD", "3ES-7MD"])  # Last 20%: all configs
    ]
    
    for episode in range(num_episodes):
        # Determine active configs for curriculum
        if exp_config.get('curriculum', False):
            progress = episode / num_episodes
            for start, end, active in curriculum_schedule:
                if start <= progress < end:
                    active_configs = [(M, E, name) for M, E, name in configs if name in active]
                    break
            else:
                active_configs = configs
        else:
            active_configs = configs
        
        # Select random config from active ones
        M, E, config_name = active_configs[episode % len(active_configs)]
        env = envs[config_name]
        config_agents = agents[config_name]
        
        # Episode
        obs, _ = env.reset(seed=episode)
        for step in range(10):
            actions = {}
            for agent in config_agents:
                agent_id = f"device_{agent.agent_id}"
                mask = env.compute_action_mask(agent_id)
                action, log_prob, value = agent.select_action(obs[agent_id], M, E, mask)
                actions[agent_id] = action
                agent.store_transition(obs[agent_id], action, 0.0, value, log_prob, False)
            
            obs, rewards, _, _, _ = env.step(actions)
            for agent in config_agents:
                agent_id = f"device_{agent.agent_id}"
                agent.trajectory["rewards"][-1] = rewards[agent_id]
        
        # Update
        for agent in config_agents:
            agent.update(M, E)
            agent.clear_trajectory()
        
        # Evaluate
        if episode % eval_interval == 0:
            print(f"\n--- Episode {episode} ---")
            for _, _, name in configs:
                cost, comp = evaluate(envs[name], agents[name], envs[name].M, envs[name].E, device)
                history[name].append({"episode": episode, "cost": cost, "comp": comp})
                if cost < best_costs[name]:
                    best_costs[name] = cost
                print(f"  {name}: Cost={cost:.4f}, Comp={comp:.1%}, Best={best_costs[name]:.4f}")
    
    # Final evaluation
    final_results = {}
    print(f"\n{'='*60}")
    print(f"FINAL RESULTS: {exp_config['name']}")
    print(f"{'='*60}")
    
    for M, E, name in configs:
        cost, comp = evaluate(envs[name], agents[name], M, E, device, num_eps=50)
        final_results[name] = {"cost": cost, "comp": comp}
        print(f"{name}: Cost={cost:.4f}, Comp={comp:.1%}")
    
    # Cross-config test
    test_configs = [
        (4, 2, "2ES-4MD"),
        (6, 3, "3ES-6MD"),
    ]
    
    print(f"\nCross-Config Generalization:")
    for M, E, name in test_configs:
        env = PaperAccurateEnvV3(num_devices=M, num_servers=E, randomize_profile=True)
        test_agents = [CrossScaleAgent(i, hyper_net, device=device, lr=exp_config['lr']) 
                      for i in range(M)]
        cost, comp = evaluate(env, test_agents, M, E, device, num_eps=50)
        final_results[name] = {"cost": cost, "comp": comp}
        print(f"{name} (unseen): Cost={cost:.4f}, Comp={comp:.1%}")
    
    # Save results
    variant_dir = Path(save_dir) / exp_config['name']
    variant_dir.mkdir(parents=True, exist_ok=True)
    
    with open(variant_dir / "results.json", 'w') as f:
        json.dump({
            "config": make_serializable_config(exp_config),
            "history": history,
            "best_costs": best_costs,
            "final_results": final_results,
            "training_time": time.time() - start_time
        }, f, indent=2)
    
    torch.save(hyper_net.state_dict(), variant_dir / "final.pt")
    
    return final_results, best_costs


def evaluate(env, agents, M, E, device, num_eps=20):
    """Evaluate agents."""
    costs = []
    comps = []
    
    for _ in range(num_eps):
        obs, _ = env.reset()
        for step in range(10):
            actions = {}
            for agent in agents:
                agent_id = f"device_{agent.agent_id}"
                mask = env.compute_action_mask(agent_id)
                action, _, _ = agent.select_action(obs[agent_id], M, E, mask)
                actions[agent_id] = action
            obs, _, _, _, _ = env.step(actions)
        
        metrics = env.get_episode_metrics()
        costs.append(metrics["avg_cost"])
        comps.append(metrics["completion_rate"])
    
    return np.mean(costs), np.mean(comps)


def run_ablation_study():
    """Run all variants and generate comparison."""
    print("=" * 80)
    print("HYPERNETWORK ABLATION STUDY")
    print("=" * 80)
    
    all_results = {}
    
    for exp_config in EXPERIMENTS:
        variant_dir = Path("results/hypernetwork_ablation") / exp_config['name']
        results_path = variant_dir / "results.json"
        
        if results_path.exists():
            print(f"\n[{exp_config['name']}] Already completed. Loading existing results...")
            with open(results_path, 'r') as f:
                saved = json.load(f)
            all_results[exp_config['name']] = {
                'final': saved['final_results'],
                'best': saved['best_costs']
            }
            continue
        
        final_results, best_costs = train_variant(
            exp_config,
            num_episodes=10000,  # 10K per variant for quick comparison
            save_dir="results/hypernetwork_ablation"
        )
        all_results[exp_config['name']] = {
            'final': final_results,
            'best': best_costs
        }
    
    # Generate comparison report
    print(f"\n{'='*80}")
    print("COMPARISON SUMMARY")
    print(f"{'='*80}")
    
    print("\n| Variant | 2ES-3MD | 2ES-5MD | 3ES-7MD | Avg Best |")
    print("|---------|---------|---------|---------|----------|")
    
    for name, results in all_results.items():
        c1 = results['best'].get('2ES-3MD', 999)
        c2 = results['best'].get('2ES-5MD', 999)
        c3 = results['best'].get('3ES-7MD', 999)
        avg = np.mean([c1, c2, c3])
        print(f"| {name:8s} | {c1:7.4f} | {c2:7.4f} | {c3:7.4f} | {avg:8.4f} |")
    
    # Save comparison
    with open("results/hypernetwork_ablation/comparison.json", 'w') as f:
        json.dump(all_results, f, indent=2)
    
    # Notify
    notify_training_complete(
        job_name="HyperNetwork Ablation Study",
        best_metric=min([np.mean(list(r['best'].values())) for r in all_results.values()]),
        episodes=10000 * len(EXPERIMENTS),
        duration_seconds=0,  # Will be calculated
        results={k: v['final'] for k, v in all_results.items()}
    )
    
    print(f"\nResults saved to: results/hypernetwork_ablation/")


if __name__ == "__main__":
    run_ablation_study()
