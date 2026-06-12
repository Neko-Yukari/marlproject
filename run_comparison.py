"""
Complete comparison: GNN vs Standard vs HyperNetwork
3 configs × 3 networks = 9 experiments
Saves convergence curves for visualization.
"""
import sys; sys.path.insert(0, '.')
import time
import json
import torch
import numpy as np
from envs.paper_accurate_env import PaperAccurateEnvV3
from agents import StandardPolicy, GNNPolicy, HyperPolicy, PPOAgent

# Configs
CONFIGS = [
    {'M': 3, 'E': 2, 'name': '2ES-3MD'},
    {'M': 5, 'E': 2, 'name': '2ES-5MD'},
    {'M': 7, 'E': 3, 'name': '3ES-7MD'},
]

NETWORKS = {
    'Standard': lambda M, E: StandardPolicy(
        state_dim=1+E+E, action_dim=E+1, hidden_dim=128, num_layers=2
    ),
    'GNN': lambda M, E: GNNPolicy(
        max_action_dim=4, hidden_dim=128, gnn_layers=2
    ),
    'HyperNetwork': lambda M, E: HyperPolicy(
        max_obs_dim=1+3+3, max_action_dim=4, hidden_dim=256
    ),
}

NUM_EPISODES = 10000
LOG_INTERVAL = 500
UPDATE_EVERY = 500
BATCH_SIZE = 256

def train_network(network_name, network_fn, M, E, config_name):
    """Train one network on one config."""
    print(f"\n{'='*70}")
    print(f"Training {network_name} on {config_name}")
    print(f"{'='*70}")
    
    env = PaperAccurateEnvV3(num_devices=M, num_servers=E)
    device = torch.device('cuda')
    
    policy = network_fn(M, E).to(device)
    if hasattr(policy, 'set_config'):
        policy.set_config(M, E)
    
    agents = [PPOAgent(agent_id=i, policy_network=policy, device=device,
                       learning_rate=5e-5, entropy_coeff=0.01)
              for i in range(M)]
    
    history = []
    start = time.time()
    best_cost = float('inf')
    
    for ep in range(NUM_EPISODES):
        obs, _ = env.reset(seed=(42 + ep))
        
        for step in range(10):
            # GNN: build graph
            if hasattr(policy, 'set_graph'):
                policy.set_graph(env)
            
            actions = {}
            for i, name in enumerate(env.agents):
                a, lp, v = agents[i].select_action(obs[name], agent_id=i)
                actions[name] = a
                agents[i]._last_value = v
                agents[i]._last_log_prob = lp
                
                # Store embedding for GNN
                if hasattr(policy, 'get_embedding'):
                    emb = policy.get_embedding(i)
                    if emb is not None:
                        agents[i]._last_embedding = emb.detach().cpu()
            
            next_obs, rewards, terms, truncs, infos = env.step(actions)
            
            for i, name in enumerate(env.agents):
                emb = getattr(agents[i], '_last_embedding', None)
                agents[i].store_transition(
                    obs[name], actions[name], rewards[name],
                    agents[i]._last_value, agents[i]._last_log_prob,
                    terms[name] or truncs[name],
                    embedding=emb
                )
            
            if hasattr(policy, 'clear_cache'):
                policy.clear_cache()
            obs = next_obs
        
        if (ep + 1) % UPDATE_EVERY == 0:
            for agent in agents:
                if len(agent.trajectory['states']) > 0:
                    agent.update(batch_size=BATCH_SIZE, num_epochs=4)
        
        if ep % LOG_INTERVAL == 0 or ep == NUM_EPISODES - 1:
            m = env.get_episode_metrics()
            elapsed = time.time() - start
            eps_per_sec = (ep + 1) / elapsed
            
            is_best = m['avg_cost'] < best_cost
            if is_best:
                best_cost = m['avg_cost']
            
            print(f"Ep {ep:5d} | Cost: {m['avg_cost']:.4f} | Comp: {m['completion_rate']:.1%} | "
                  f"Speed: {eps_per_sec:.1f} eps/s | {'BEST' if is_best else ''}")
            
            history.append({
                'episode': ep,
                'cost': float(m['avg_cost']),
                'completion': float(m['completion_rate']),
                'time': elapsed
            })
    
    return {
        'network': network_name,
        'config': config_name,
        'history': history,
        'best_cost': float(best_cost),
        'total_time': time.time() - start
    }

# Run all experiments
results = []
for cfg in CONFIGS:
    for net_name, net_fn in NETWORKS.items():
        result = train_network(net_name, net_fn, cfg['M'], cfg['E'], cfg['name'])
        results.append(result)
        
        # Save intermediate results
        with open(f'results/comparison_{cfg["name"]}_{net_name}.json', 'w') as f:
            json.dump(result, f, indent=2)

# Save combined results
with open('results/comparison_all.json', 'w') as f:
    json.dump(results, f, indent=2)

print(f"\n{'='*70}")
print("ALL EXPERIMENTS COMPLETE")
print(f"{'='*70}")
for r in results:
    print(f"{r['network']:15s} | {r['config']:10s} | Best Cost: {r['best_cost']:.4f} | Time: {r['total_time']:.0f}s")
