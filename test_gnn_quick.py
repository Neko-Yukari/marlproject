"""
Simple GNN training test - 5K episodes, CPU only.
Quick check if GNN actually learns.
"""
import sys; sys.path.insert(0, '.')
import time
import torch
import numpy as np
from envs.paper_accurate_env import PaperAccurateEnvV3
from agents import GNNPolicy, PPOAgent

print("GNN Learning Test - 5000 episodes")
print("="*60)

env = PaperAccurateEnvV3(num_devices=3, num_servers=2)
policy = GNNPolicy(max_action_dim=4, hidden_dim=64, gnn_layers=2)
agents = [PPOAgent(agent_id=i, policy_network=policy, device=torch.device('cpu'))
          for i in range(3)]

history = []
start = time.time()

for ep in range(5000):
    obs, _ = env.reset(seed=(42 + ep))
    
    for step in range(10):
        policy.set_graph(env)
        
        actions = {}
        for i, name in enumerate(env.agents):
            a, lp, v = agents[i].select_action(obs[name], agent_id=i)
            actions[name] = a
            agents[i]._last_value = v
            agents[i]._last_log_prob = lp
        
        next_obs, rewards, terms, truncs, infos = env.step(actions)
        
        for i, name in enumerate(env.agents):
            agents[i].store_transition(
                obs[name], actions[name], rewards[name],
                agents[i]._last_value, agents[i]._last_log_prob,
                terms[name] or truncs[name]
            )
        
        policy.clear_cache()
        obs = next_obs
    
    if (ep + 1) % 10 == 0:
        for agent in agents:
            if len(agent.trajectory['states']) > 0:
                agent.update(batch_size=64, num_epochs=2)
    
    if ep % 500 == 0:
        m = env.get_episode_metrics()
        history.append((ep, m['avg_cost'], m['completion_rate']))
        print(f"Ep {ep:4d} | Cost: {m['avg_cost']:.4f} | Comp: {m['completion_rate']:.1%} | Time: {time.time()-start:.1f}s")

# Analyze learning
costs = [c for _, c, _ in history]
if len(costs) >= 2:
    first = np.mean(costs[:len(costs)//2])
    second = np.mean(costs[len(costs)//2:])
    print(f"\nLearning Analysis:")
    print(f"  First half avg:  {first:.4f}")
    print(f"  Second half avg: {second:.4f}")
    print(f"  Trend:           {'LEARNING' if second < first else 'NOT LEARNING'}")

print(f"\nBest cost: {min(costs):.4f}")
print(f"Time: {time.time()-start:.1f}s")
