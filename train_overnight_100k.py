"""Overnight training: 100K episodes with full algorithms."""
import sys; sys.path.insert(0, '.')
import numpy as np, json, time, torch
from pathlib import Path
from datetime import datetime
from envs.paper_accurate_env_v2 import PaperAccurateEnv
from agents.ippo_agent import IPPOAgent
from agents.explaboff_agent import ExplabOffAgent

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Device: {device}")
if device.type == 'cuda':
    print(f"GPU: {torch.cuda.get_device_name(0)}")

def train_algorithm(name, agent_class, episodes, M=3, E=2, 
                    use_mi=False, update_every=100, log_interval=1000):
    """Train one algorithm."""
    print(f"\n{'='*60}")
    print(f"Training: {name} | {episodes} episodes")
    print(f"{'='*60}")
    
    env = PaperAccurateEnv(M, E, seed=42)
    dim, adim = env.obs_dim, E + 1
    
    if use_mi:
        agents = [agent_class(i, dim, adim, hidden_dim=128, lr=5e-5,
                              mi_mu=3.5, mi_nu=1.0, device=device)
                  for i in range(M)]
    else:
        agents = [agent_class(i, dim, adim, hidden_dim=128, lr=5e-5, device=device)
                  for i in range(M)]
    
    accum_traj = {i: {"states": [], "actions": [], "rewards": [], 
                       "values": [], "log_probs": [], "dones": []}
                  for i in range(M)}
    
    hist = []
    t0 = time.time()
    update_count = 0
    
    for ep in range(episodes):
        obs, _ = env.reset(seed=(42 + ep))
        ep_reward_sum = 0.0
        
        for step in range(10):
            actions = {}
            for i, agent in enumerate(agents):
                a_name = f"device_{i}"
                a, lp, v = agent.select_action(obs[a_name])
                actions[a_name] = a
                
                accum_traj[i]["states"].append(obs[a_name])
                accum_traj[i]["actions"].append(a)
                accum_traj[i]["values"].append(v)
                accum_traj[i]["log_probs"].append(lp)
                accum_traj[i]["dones"].append(False)
                accum_traj[i]["rewards"].append(0.0)
            
            next_obs, rewards, terms, _, _ = env.step(actions)
            
            for i, agent in enumerate(agents):
                a_name = f"device_{i}"
                if use_mi:
                    mi_r = agent.compute_mi_reward(obs[a_name], actions[a_name])
                    combined = rewards[a_name] + mi_r
                else:
                    combined = rewards[a_name]
                accum_traj[i]["rewards"][-1] = combined
                ep_reward_sum += combined
            
            obs = next_obs
            if any(terms.values()):
                break
        
        # Update every N episodes
        if (ep + 1) % update_every == 0:
            for i, agent in enumerate(agents):
                agent.trajectory = {
                    "states": accum_traj[i]["states"][:],
                    "actions": accum_traj[i]["actions"][:],
                    "rewards": accum_traj[i]["rewards"][:],
                    "values": accum_traj[i]["values"][:],
                    "log_probs": accum_traj[i]["log_probs"][:],
                    "dones": accum_traj[i]["dones"][:]
                }
                if use_mi:
                    agent.classify_episode(ep_reward_sum)
                agent.update(batch_size=256, num_epochs=4)
                
                for k in accum_traj[i]:
                    accum_traj[i][k].clear()
            
            update_count += 1
            if device.type == 'cuda':
                torch.cuda.empty_cache()
        
        if ep % log_interval == 0 or ep == episodes - 1:
            m = env.get_episode_metrics()
            el = time.time() - t0
            hist.append({
                'ep': ep, 'avg_cost': float(m['avg_cost']),
                'completion_rate': float(m['completion_rate']),
                'avg_latency': float(m['avg_latency']),
                'avg_energy': float(m['avg_energy']),
                'ep_reward': float(ep_reward_sum), 'time': el
            })
            print(f"[{name}] ep{ep:6d} cost={m['avg_cost']:6.3f} comp={m['completion_rate']:.3f} "
                  f"lat={m['avg_latency']:6.2f} t={el:7.0f}s")
    
    return hist

# Train both algorithms
results = {}
results['IPPO_100K'] = train_algorithm('IPPO', IPPOAgent, 100000, use_mi=False)
results['ExplabOff_100K'] = train_algorithm('ExplabOff', ExplabOffAgent, 100000, use_mi=True)

# Save results
out = Path("results") / f"overnight_100k_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
out.mkdir(parents=True, exist_ok=True)

with open(out / "results.json", "w") as f:
    json.dump(results, f, indent=2)

for name, data in results.items():
    with open(out / f"{name.lower()}.csv", "w") as f:
        f.write("ep,avg_cost,completion_rate,avg_latency,avg_energy,ep_reward\n")
        for row in data:
            f.write(f"{row['ep']},{row['avg_cost']:.4f},{row['completion_rate']:.4f},"
                    f"{row['avg_latency']:.4f},{row['avg_energy']:.4f},{row['ep_reward']:.4f}\n")

print(f"\n{'='*60}")
print("Overnight training complete!")
print(f"{'='*60}")
for name, data in results.items():
    final = data[-1]
    print(f"{name:20s}: cost={final['avg_cost']:.3f} comp={final['completion_rate']:.3f} "
          f"lat={final['avg_latency']:.2f} time={final['time']:.0f}s")
print(f"\nSaved to {out}")
