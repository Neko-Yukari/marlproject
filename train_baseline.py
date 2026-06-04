"""
Baseline IPPO Training - Multi-Config for Comparison.

Trains separate IPPO models for each configuration.
Used as baseline to compare against HyperNetwork.
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
from agents.ippo_agent import IPPOAgent


def train_baseline(config_name, M, E, num_episodes=20000, save_dir="results/baseline"):
    """Train baseline IPPO for single config."""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    env = PaperAccurateEnvV3(num_devices=M, num_servers=E, randomize_profile=True)
    
    agents = []
    for i in range(M):
        agent = IPPOAgent(i, env.obs_dim, env.E + 1, hidden_dim=128, learning_rate=5e-5, device=device)
        agents.append(agent)
    
    best_cost = float('inf')
    history = []
    start_time = time.time()
    
    for episode in range(num_episodes):
        obs, _ = env.reset(seed=episode)
        
        for step in range(10):
            actions = {}
            for agent in agents:
                aid = f"device_{agent.agent_id}"
                mask = env.compute_action_mask(aid)
                a, lp, v = agent.select_action(obs[aid], mask)
                actions[aid] = a
                agent.store_transition(obs[aid], a, 0.0, v, lp, False)
            
            obs, rewards, _, _, _ = env.step(actions)
            for agent in agents:
                agent.trajectory["rewards"][-1] = rewards[f"device_{agent.agent_id}"]
        
        for agent in agents:
            agent.update()
            agent.clear_trajectory()
        
        if episode % 1000 == 0:
            cost, comp = evaluate(env, agents, device)
            history.append({"episode": episode, "cost": cost, "comp": comp})
            if cost < best_cost:
                best_cost = cost
            print(f"[{config_name}] Ep {episode:5d} | Cost: {cost:.4f} | Comp: {comp:.1%} | Best: {best_cost:.4f}")
    
    # Final eval
    final_cost, final_comp = evaluate(env, agents, device, num_eps=50)
    
    Path(save_dir).mkdir(parents=True, exist_ok=True)
    with open(f"{save_dir}/{config_name}_results.json", 'w') as f:
        json.dump({
            "config": config_name,
            "best_cost": best_cost,
            "final_cost": final_cost,
            "final_comp": final_comp,
            "history": history,
            "time": time.time() - start_time
        }, f)
    
    return final_cost, final_comp


def evaluate(env, agents, device, num_eps=20):
    costs = []
    comps = []
    for _ in range(num_eps):
        obs, _ = env.reset()
        for step in range(10):
            actions = {}
            for agent in agents:
                aid = f"device_{agent.agent_id}"
                mask = env.compute_action_mask(aid)
                a, _, _ = agent.select_action(obs[aid], mask)
                actions[aid] = a
            obs, _, _, _, _ = env.step(actions)
        metrics = env.get_episode_metrics()
        costs.append(metrics["avg_cost"])
        comps.append(metrics["completion_rate"])
    return np.mean(costs), np.mean(comps)


if __name__ == "__main__":
    configs = [
        ("2ES-3MD", 3, 2),
        ("2ES-5MD", 5, 2),
        ("3ES-7MD", 7, 3),
    ]
    
    results = {}
    for name, M, E in configs:
        print(f"\n{'='*60}")
        print(f"Training {name}")
        print(f"{'='*60}")
        cost, comp = train_baseline(name, M, E)
        results[name] = {"cost": cost, "comp": comp}
    
    print(f"\n{'='*60}")
    print("BASELINE RESULTS")
    print(f"{'='*60}")
    for name, res in results.items():
        print(f"{name}: Cost={res['cost']:.4f}, Comp={res['comp']:.1%}")
