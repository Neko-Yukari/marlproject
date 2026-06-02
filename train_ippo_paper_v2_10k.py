"""Train IPPO on paper-accurate environment v2 — 10K episodes."""
import sys; sys.path.insert(0, '.')
import numpy as np, json, time, torch
from pathlib import Path
from datetime import datetime
from envs.paper_accurate_env_v2 import PaperAccurateEnv
from agents.ippo_agent import IPPOAgent

def train(name, episodes, log_interval=500):
    device = torch.device('cpu')
    env = PaperAccurateEnv(3, 2, seed=42)
    dim, adim = env.obs_dim, 3  # 3 actions: local, ES1, ES2
    agents = [IPPOAgent(i, dim, adim, hidden_dim=128, learning_rate=5e-5, device=device) 
              for i in range(3)]
    hist = []
    t0 = time.time()
    
    for ep in range(episodes):
        obs, _ = env.reset(seed=(42 + ep))
        for agent in agents:
            agent.clear_trajectory()
        ep_reward_sum = 0.0
        
        for step in range(10):  # 10 slots per episode
            actions = {}
            for i, agent in enumerate(agents):
                a_name = f"device_{i}"
                a, lp, v = agent.select_action(obs[a_name])
                actions[a_name] = a
                agent.store_transition(obs[a_name], a, 0.0, v, lp, False)
            
            next_obs, rewards, terms, truncs, _ = env.step(actions)
            
            for i, agent in enumerate(agents):
                a_name = f"device_{i}"
                agent.trajectory["rewards"][-1] = rewards[a_name]
                ep_reward_sum += rewards[a_name]
            
            obs = next_obs
            if any(terms.values()):
                break
        
        for agent in agents:
            agent.update(batch_size=64, num_epochs=4)
        
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
            print(f"  [{name}] ep{ep:6d} | cost={m['avg_cost']:6.3f} | comp={m['completion_rate']:.3f} | "
                  f"lat={m['avg_latency']:6.2f} | rew={ep_reward_sum:8.1f} | t={el:6.0f}s")
    
    return hist

print(f"\n{'='*70}")
print(f"IPPO 10K | Paper-Accurate Env v2 (Mb tasks) | 2ES-3MD")
print(f"{'='*70}")
results = {'IPPO_paper_10K': train('IPPO', 10000, 500)}

out = Path("results") / f"ippo_paper_v2_10k_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
out.mkdir(parents=True, exist_ok=True)
with open(out / "results.json", "w") as f:
    json.dump(results, f, indent=2)

with open(out / "ippo_paper_v2_10k.csv", "w") as f:
    f.write("ep,avg_cost,completion_rate,avg_latency,avg_energy,ep_reward\n")
    for row in results['IPPO_paper_10K']:
        f.write(f"{row['ep']},{row['avg_cost']:.4f},{row['completion_rate']:.4f},"
                f"{row['avg_latency']:.4f},{row['avg_energy']:.4f},{row['ep_reward']:.4f}\n")

final = results['IPPO_paper_10K'][-1]
print(f"\nFinal: cost={final['avg_cost']:.3f} comp={final['completion_rate']:.3f} "
      f"lat={final['avg_latency']:.2f} time={final['time']:.0f}s")
print(f"Saved to {out}")