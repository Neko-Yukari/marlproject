"""Train ExplabOff on paper-accurate environment v2 — 10K episodes."""
import sys; sys.path.insert(0, '.')
import numpy as np, json, time, torch
from pathlib import Path
from datetime import datetime
from envs.paper_accurate_env_v2 import PaperAccurateEnv
from agents.explaboff_agent import ExplabOffAgent

def train(M, E, episodes, log_interval=500):
    device = torch.device('cpu')
    env = PaperAccurateEnv(M, E, seed=42)
    dim, adim = env.obs_dim, E + 1
    agents = [ExplabOffAgent(i, dim, adim, hidden_dim=128, lr=5e-5,
                             mi_mu=0.01, mi_nu=0.01, device=device)
              for i in range(M)]
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
                actions[a_name] = a  # Direct integer action
                agent.store_transition(obs[a_name], a, 0.0, v, lp, False)
            
            next_obs, rewards, terms, _, _ = env.step(actions)
            
            # Apply MI-enhanced rewards
            for i, agent in enumerate(agents):
                a_name = f"device_{i}"
                mi_r = agent.compute_mi_reward(obs[a_name], agent.trajectory["actions"][-1])
                combined = rewards[a_name] + mi_r
                agent.trajectory["rewards"][-1] = combined
                ep_reward_sum += combined
            
            obs = next_obs
            if any(terms.values()):
                break
        
        # Update each agent
        for agent in agents:
            agent.classify_episode(ep_reward_sum)
            agent.update(batch_size=64, num_epochs=4)
        
        if ep % log_interval == 0 or ep == episodes - 1:
            m = env.get_episode_metrics()
            el = time.time() - t0
            hist.append({'ep': ep, 'avg_cost': float(m['avg_cost']),
                        'completion_rate': float(m['completion_rate']),
                        'avg_latency': float(m['avg_latency']),
                        'avg_energy': float(m['avg_energy']),
                        'ep_reward': float(ep_reward_sum), 'time': el})
            print(f"ep{ep:6d} cost={m['avg_cost']:6.3f} comp={m['completion_rate']:.3f} "
                  f"lat={m['avg_latency']:6.2f} t={el:6.0f}s")
    
    return hist

print("="*60)
print("ExplabOff 10K | Paper-Accurate Env v2 | 2ES-3MD")
print("="*60)
results = {'ExplabOff_paper_10K': train(3, 2, 10000, 500)}

out = Path("results") / f"explaboff_paper_v2_10k_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
out.mkdir(parents=True, exist_ok=True)
with open(out / "results.json", "w") as f:
    json.dump(results, f, indent=2)

with open(out / "explaboff_paper_v2_10k.csv", "w") as f:
    f.write("ep,avg_cost,completion_rate,avg_latency,avg_energy,ep_reward\n")
    for row in results['ExplabOff_paper_10K']:
        f.write(f"{row['ep']},{row['avg_cost']:.4f},{row['completion_rate']:.4f},"
                f"{row['avg_latency']:.4f},{row['avg_energy']:.4f},{row['ep_reward']:.4f}\n")

final = results['ExplabOff_paper_10K'][-1]
print(f"\nFinal: cost={final['avg_cost']:.3f} comp={final['completion_rate']:.3f} "
      f"lat={final['avg_latency']:.2f} time={final['time']:.0f}s")
print(f"Saved to {out}")
