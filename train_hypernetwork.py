"""
HyperNetwork Training Script - Cross-Scale MEC Offloading.

Trains a single hypernetwork on multiple (M, E) configurations.
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
from agents.hypernetwork import HyperNetwork, CrossScaleAgent
from utils.opencode_notifier import notify_training_complete


def train_mixed(num_episodes=50000, eval_interval=1000, save_dir="results/hypernetwork"):
    """Train hypernetwork on mixed configurations."""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    
    # Configurations to train on
    configs = [
        (3, 2, "2ES-3MD"),
        (5, 2, "2ES-5MD"),
        (7, 3, "3ES-7MD"),
    ]
    
    # Create shared hypernetwork
    hyper_net = HyperNetwork(
        obs_dim=7,  # Max observation dimension
        max_action_dim=4,
        hidden_dim=128
    ).to(device)
    
    print(f"HyperNetwork parameters: {sum(p.numel() for p in hyper_net.parameters())}")
    
    # Create environments
    envs = {}
    agents = {}
    for M, E, name in configs:
        envs[name] = PaperAccurateEnvV3(num_devices=M, num_servers=E, randomize_profile=True)
        agents[name] = [CrossScaleAgent(i, hyper_net, device=device) for i in range(M)]
    
    # Training metrics
    history = {name: [] for _, _, name in configs}
    best_costs = {name: float('inf') for _, _, name in configs}
    
    start_time = time.time()
    
    for episode in range(num_episodes):
        # Randomly select a configuration
        M, E, config_name = configs[episode % len(configs)]
        env = envs[config_name]
        config_agents = agents[config_name]
        
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
        
        # Update all agents with this configuration
        for agent in config_agents:
            agent.update(M, E)
            agent.clear_trajectory()
        
        # Evaluate
        if episode % eval_interval == 0:
            print(f"\n=== Episode {episode} ===")
            for M, E, name in configs:
                cost, comp = evaluate(envs[name], agents[name], M, E, device)
                history[name].append({"episode": episode, "cost": cost, "comp": comp})
                
                if cost < best_costs[name]:
                    best_costs[name] = cost
                
                print(f"  {name}: Cost={cost:.4f}, Comp={comp:.1%}, Best={best_costs[name]:.4f}")
            
            elapsed = time.time() - start_time
            print(f"  Time: {elapsed:.1f}s")
            
            # Save checkpoint
            Path(save_dir).mkdir(parents=True, exist_ok=True)
            torch.save(hyper_net.state_dict(), f"{save_dir}/hypernet_ep{episode}.pt")
    
    # Final evaluation
    print(f"\n{'='*60}")
    print("FINAL RESULTS")
    print(f"{'='*60}")
    
    final_results = {}
    for M, E, name in configs:
        cost, comp = evaluate(envs[name], agents[name], M, E, device, num_eps=50)
        final_results[name] = {"cost": cost, "comp": comp}
        print(f"{name}: Cost={cost:.4f}, Comp={comp:.1%}")
    
    # Cross-config generalization test
    print(f"\n{'='*60}")
    print("CROSS-CONFIG GENERALIZATION")
    print(f"{'='*60}")
    
    test_configs = [
        (4, 2, "2ES-4MD"),
        (6, 3, "3ES-6MD"),
    ]
    
    for M, E, name in test_configs:
        env = PaperAccurateEnvV3(num_devices=M, num_servers=E, randomize_profile=True)
        test_agents = [CrossScaleAgent(i, hyper_net, device=device) for i in range(M)]
        cost, comp = evaluate(env, test_agents, M, E, device, num_eps=50)
        final_results[name] = {"cost": cost, "comp": comp}
        print(f"{name} (unseen): Cost={cost:.4f}, Comp={comp:.1%}")
    
    # Save results
    with open(f"{save_dir}/results.json", 'w') as f:
        json.dump({
            "history": history,
            "best_costs": best_costs,
            "final_results": final_results,
            "training_time": time.time() - start_time
        }, f, indent=2)
    
    # Save final model
    torch.save(hyper_net.state_dict(), f"{save_dir}/final.pt")
    
    # Notify completion
    avg_best = np.mean(list(best_costs.values()))
    notify_training_complete(
        job_name="HyperNetwork Cross-Scale Training",
        best_metric=avg_best,
        episodes=num_episodes,
        duration_seconds=time.time() - start_time,
        results=final_results
    )
    
    print(f"\nTraining complete! Total time: {(time.time() - start_time)/3600:.1f} hours")
    return final_results


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


if __name__ == "__main__":
    print("=" * 60)
    print("HyperNetwork Cross-Scale Training")
    print("=" * 60)
    train_mixed(num_episodes=50000, save_dir="results/hypernetwork")
