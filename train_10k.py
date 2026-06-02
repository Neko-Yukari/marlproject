"""10K episodes training script for IPPO and ExplabOff.

Matches paper: 2ES-3MD, tight deadlines, multi-slot execution.
Logs every 500 episodes to avoid console spam.
"""
import sys; sys.path.insert(0, '.')
import numpy as np, json, time, torch
from pathlib import Path
from datetime import datetime
from envs.edge_offload_env import EdgeOffloadEnv
from agents.ippo_agent import IPPOAgent
from agents.explaboff_agent import ExplabOffAgent

def discrete_to_dict(a, E):
    """Map discrete action to env action dict."""
    if a == 0:
        return {"offload_ratio": np.array([0.0], np.float32), "target_es": 0}
    return {"offload_ratio": np.array([1.0], np.float32), "target_es": min(a, E)}


def train_ippo(name, M, E, episodes, ep_len, log_interval=500):
    """Train IPPO baseline."""
    device = torch.device('cpu')
    # Paper params: MD=1e9, ES=6-12e9. Use 3e9/10e9 for tighter learning signal.
    env = EdgeOffloadEnv(M, E, ep_len, device_cpu=3e9, server_cpu=10e9, energy_budget=500.0)
    dim, adim = env.obs_dim, E + 1
    
    # Create agents with shared network (parameter sharing)
    agents = [IPPOAgent(i, dim, adim, hidden_dim=128, learning_rate=5e-5, device=device) 
              for i in range(M)]
    
    hist = []
    t0 = time.time()
    
    for ep in range(episodes):
        obs, _ = env.reset()
        for agent in agents:
            agent.clear_trajectory()
        
        ep_reward_sum = 0.0
        for step in range(ep_len):
            actions = {}
            for i, agent in enumerate(agents):
                a_name = f"device_{i}"
                a, lp, v = agent.select_action(obs[a_name])
                actions[a_name] = discrete_to_dict(a, E)
                agent.store_transition(obs[a_name], a, 0.0, v, lp, False)
            
            next_obs, rewards, terms, _, _ = env.step(actions)
            
            # Store rewards in trajectories
            for i, agent in enumerate(agents):
                a_name = f"device_{i}"
                agent.trajectory["rewards"][-1] = rewards[a_name]
                ep_reward_sum += rewards[a_name]
            
            obs = next_obs
            if any(terms.values()):
                break
        
        # PPO update for each agent
        for agent in agents:
            agent.update(batch_size=64, num_epochs=4)
        
        # Logging
        if ep % log_interval == 0 or ep == episodes - 1:
            m = env.get_episode_metrics()
            el = time.time() - t0
            hist.append({
                'ep': ep,
                'avg_cost': float(m['avg_cost']),
                'completion_rate': float(m['completion_rate']),
                'avg_latency': float(m['avg_latency']),
                'avg_energy': float(m['avg_energy']),
                'ep_reward': float(ep_reward_sum),
                'time': el
            })
            print(f"  [{name}] ep{ep:6d} | cost={m['avg_cost']:6.3f} | comp={m['completion_rate']:.3f} | "
                  f"lat={m['avg_latency']:6.2f} | rew={ep_reward_sum:8.1f} | t={el:6.0f}s")
    
    return hist


def train_explaboff(name, M, E, episodes, ep_len, log_interval=500):
    """Train ExplabOff with MI enhancement."""
    device = torch.device('cpu')
    env = EdgeOffloadEnv(M, E, ep_len, device_cpu=3e9, server_cpu=10e9, energy_budget=500.0)
    dim, adim = env.obs_dim, E + 1
    
    agents = [ExplabOffAgent(i, dim, adim, hidden_dim=128, lr=5e-5,
                             mi_mu=0.01, mi_nu=0.01, device=device)
              for i in range(M)]
    
    hist = []
    t0 = time.time()
    
    for ep in range(episodes):
        obs, _ = env.reset()
        for agent in agents:
            agent.clear_trajectory()
        
        ep_reward_sum = 0.0
        for step in range(ep_len):
            actions = {}
            for i, agent in enumerate(agents):
                a_name = f"device_{i}"
                a, lp, v = agent.select_action(obs[a_name])
                actions[a_name] = discrete_to_dict(a, E)
                
                # Compute MI-enhanced reward
                mi_reward = agent.compute_mi_reward(obs[a_name], a)
                combined_reward = 0.0  # Will be replaced after step
                agent.store_transition(obs[a_name], a, combined_reward, v, lp, False)
            
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
            # Classify episode for dual buffers
            agent.classify_episode(ep_reward_sum)
            agent.update(batch_size=64, num_epochs=4)
        
        # Logging
        if ep % log_interval == 0 or ep == episodes - 1:
            m = env.get_episode_metrics()
            el = time.time() - t0
            hist.append({
                'ep': ep,
                'avg_cost': float(m['avg_cost']),
                'completion_rate': float(m['completion_rate']),
                'avg_latency': float(m['avg_latency']),
                'avg_energy': float(m['avg_energy']),
                'ep_reward': float(ep_reward_sum),
                'time': el
            })
            print(f"  [{name}] ep{ep:6d} | cost={m['avg_cost']:6.3f} | comp={m['completion_rate']:.3f} | "
                  f"lat={m['avg_latency']:6.2f} | rew={ep_reward_sum:8.1f} | t={el:6.0f}s")
    
    return hist


def main():
    EPISODES = 10000
    EP_LEN = 100
    LOG_INT = 500
    
    results = {}
    configs = [
        ('IPPO_2ES3MD_10K', 3, 2, 'ippo'),
        ('ExplabOff_2ES3MD_10K', 3, 2, 'explaboff'),
    ]
    
    for name, M, E, algo in configs:
        print(f"\n{'='*70}")
        print(f"Training: {name} | M={M} E={E} | Episodes={EPISODES}")
        print(f"{'='*70}")
        
        if algo == 'ippo':
            results[name] = train_ippo(name, M, E, EPISODES, EP_LEN, LOG_INT)
        else:
            results[name] = train_explaboff(name, M, E, EPISODES, EP_LEN, LOG_INT)
    
    # Save results
    out = Path("results") / f"10k_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    out.mkdir(parents=True, exist_ok=True)
    
    # Save JSON (strip time for compactness)
    with open(out / "results.json", "w") as f:
        json.dump({
            k: [{kk: vv for kk, vv in v.items() if kk != 'time'} for v in vs]
            for k, vs in results.items()
        }, f, indent=2)
    
    # Save CSV for easy plotting
    for name, hist in results.items():
        with open(out / f"{name}.csv", "w") as f:
            f.write("ep,avg_cost,completion_rate,avg_latency,avg_energy,ep_reward\n")
            for row in hist:
                f.write(f"{row['ep']},{row['avg_cost']:.4f},{row['completion_rate']:.4f},"
                        f"{row['avg_latency']:.4f},{row['avg_energy']:.4f},{row['ep_reward']:.4f}\n")
    
    print(f"\n{'='*70}")
    print(f"Results saved to: {out}")
    print(f"{'='*70}")
    for name, hist in results.items():
        if hist:
            final = hist[-1]
            print(f"  {name}:")
            print(f"    Final Cost:      {final['avg_cost']:.3f}")
            print(f"    Completion:      {final['completion_rate']:.3f}")
            print(f"    Avg Latency:     {final['avg_latency']:.2f}")
            print(f"    Total Time:      {final['time']:.0f}s")


if __name__ == "__main__":
    main()
