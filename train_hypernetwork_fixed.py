"""
HyperNetwork Fixed Training - With Value Head, Buffer Accumulation, and Lower LR.

Fixes applied:
1. Value Head (non-zero) for proper GAE
2. Multi-episode buffer (update every N episodes instead of every episode)
3. Lower learning rate (1e-6)
4. Weight cache for stability
5. Higher entropy coefficient (0.05)
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
    HyperNetworkV2_Large,
    CrossScaleAgent
)
from utils.opencode_notifier import notify_training_complete


def train_fixed(num_episodes=10000, eval_interval=1000, 
                save_dir="results/hypernetwork_fixed"):
    """Train fixed HyperNetwork."""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\n{'='*60}")
    print(f"Training: HyperNetwork V2_Large FIXED")
    print(f"Device: {device}")
    print(f"{'='*60}")
    
    # Configurations
    configs = [
        (3, 2, "2ES-3MD"),
        (5, 2, "2ES-5MD"),
        (7, 3, "3ES-7MD"),
    ]
    
    # Create model
    hyper_net = HyperNetworkV2_Large(
        obs_dim=7,
        max_action_dim=4,
        hidden_dim=256
    ).to(device)
    
    print(f"Parameters: {sum(p.numel() for p in hyper_net.parameters())}")
    
    # FIXED: Lower learning rate and buffer accumulation
    lr = 1e-6
    update_interval = 10
    print(f"Learning Rate: {lr}")
    print(f"Update Interval: {update_interval} episodes")
    
    # Create environments and agents
    envs = {}
    agents = {}
    for M, E, name in configs:
        envs[name] = PaperAccurateEnvV3(num_devices=M, num_servers=E, randomize_profile=True)
        agents[name] = [CrossScaleAgent(i, hyper_net, device=device, lr=lr, 
                                        update_interval=update_interval) 
                        for i in range(M)]
    
    # Training loop
    history = {name: [] for _, _, name in configs}
    best_costs = {name: float('inf') for _, _, name in configs}
    start_time = time.time()
    
    for episode in range(num_episodes):
        # FIXED: Select random config each episode
        M, E, config_name = configs[episode % len(configs)]
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
        
        # FIXED: End episode (move to buffer)
        for agent in config_agents:
            agent.end_episode()
        
        # FIXED: Update only when buffer is full
        if config_agents[0].should_update():
            for agent in config_agents:
                info = agent.update(M, E)
                if info:
                    pass  # Could log loss here
        
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
    print(f"FINAL RESULTS: HyperNetwork FIXED")
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
        test_agents = [CrossScaleAgent(i, hyper_net, device=device, lr=lr) 
                      for i in range(M)]
        cost, comp = evaluate(env, test_agents, M, E, device, num_eps=50)
        final_results[name] = {"cost": cost, "comp": comp}
        print(f"{name} (unseen): Cost={cost:.4f}, Comp={comp:.1%}")
    
    # Save results
    variant_dir = Path(save_dir)
    variant_dir.mkdir(parents=True, exist_ok=True)
    
    with open(variant_dir / "results_fixed.json", 'w') as f:
        json.dump({
            "config": {
                "name": "V2_Large_FIXED",
                "lr": lr,
                "hidden_dim": 256,
                "update_interval": update_interval,
                "description": "Fixed: Value Head + Buffer + Low LR"
            },
            "history": history,
            "best_costs": best_costs,
            "final_results": final_results,
            "training_time": time.time() - start_time
        }, f, indent=2)
    
    torch.save(hyper_net.state_dict(), variant_dir / "final_fixed.pt")
    
    # Print comparison with old results
    print(f"\n{'='*60}")
    print("COMPARISON: FIXED vs ORIGINAL")
    print(f"{'='*60}")
    old_results = {
        "2ES-3MD": 0.4681,
        "2ES-5MD": 0.4349,
        "3ES-7MD": 0.4492
    }
    for name in ["2ES-3MD", "2ES-5MD", "3ES-7MD"]:
        old = old_results[name]
        new = final_results[name]["cost"]
        delta = new - old
        print(f"{name}: {old:.4f} → {new:.4f} ({delta:+.4f})")
    
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


if __name__ == "__main__":
    final_results, best_costs = train_fixed(num_episodes=10000)
    
    print(f"\n{'='*60}")
    print("TRAINING COMPLETE")
    print(f"{'='*60}")
    print(f"Best costs: {best_costs}")
    print(f"Results saved to: results/hypernetwork_fixed/")
