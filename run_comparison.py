"""Stage 1 vs Stage 2 comparison training (quick)."""
import sys; sys.path.insert(0, '.')
import numpy as np
import json
from pathlib import Path
from envs.edge_offload_env import EdgeOffloadEnv
from agents import IPPOAgent
from agents.explaboff_agent import ExplabOffAgent

EPISODES = 50
EP_LENGTH = 20
M, E = 3, 2  # small scale for quick test

def discrete_to_dict(action, E):
    if action == 0:
        return {"offload_ratio": np.array([0.0], dtype=np.float32), "target_es": 0}
    return {"offload_ratio": np.array([1.0], dtype=np.float32), "target_es": min(action, E)}

def run_algorithm(name, agent_cls, episodes, **kwargs):
    env = EdgeOffloadEnv(num_devices=M, num_servers=E, max_slots=EP_LENGTH)
    obs_dim = env.observation_spaces['device_0'].shape[0]
    act_dim = E + 1
    agents = [agent_cls(agent_id=i, state_dim=obs_dim, action_dim=act_dim, **kwargs) for i in range(M)]
    if name == 'IPPO':
        shared = agents[0].network
        for a in agents[1:]:
            a.network = shared
            a.optimizer = __import__('torch').optim.Adam(shared.parameters(), lr=5e-5)

    history = []
    for ep in range(episodes):
        obs, _ = env.reset()
        ep_metrics = {'episode': ep, 'total_reward': 0.0, 'steps': 0}
        for step in range(EP_LENGTH):
            agent_data = {}
            actions = {}
            for i, agent in enumerate(agents):
                a_id = f"device_{i}"
                act, logp, val = agent.select_action(obs[a_id])
                agent_data[a_id] = (act, logp, val)
                actions[a_id] = discrete_to_dict(act, E)

            next_obs, rewards, terms, truncs, _ = env.step(actions)
            for i, agent in enumerate(agents):
                a_id = f"device_{i}"
                act, logp, val = agent_data[a_id]
                r = rewards[a_id]
                if name == 'ExplabOff':
                    r += agent.compute_mi_reward(obs[a_id], act)
                agent.store_transition(obs[a_id], act, r, val, logp, terms[a_id])
            obs = next_obs
            ep_metrics['total_reward'] += sum(rewards.values())
            ep_metrics['steps'] += 1
            if any(terms.values()): break

        for agent in agents:
            agent.update(batch_size=16, num_epochs=2)
            agent.clear_trajectory()

        if name == 'ExplabOff':
            for agent in agents:
                agent.classify_episode(ep_metrics['total_reward'])

        m = env.get_episode_metrics()
        ep_metrics.update(m)
        history.append(ep_metrics)
        if ep % 10 == 0:
            print(f"  {name} ep {ep}: reward={ep_metrics['total_reward']:.1f}, completion={m['completion_rate']:.2%}")

    # Final summary
    last10 = history[-10:]
    summary = {
        'algorithm': name,
        'episodes': episodes,
        'final_completion_rate': float(np.mean([h['completion_rate'] for h in last10])),
        'final_avg_latency': float(np.mean([h['avg_latency'] for h in last10])),
        'final_avg_energy': float(np.mean([h['avg_energy'] for h in last10])),
        'final_avg_reward': float(np.mean([h['total_reward'] for h in last10])),
    }
    return summary, history

print("=" * 60)
print("IPPO vs ExplabOff Comparison")
print(f"Episodes: {EPISODES}, Devices: {M}, Servers: {E}")
print("=" * 60)

print("\n--- Stage 1: IPPO ---")
ippo_summary, ippo_hist = run_algorithm('IPPO', IPPOAgent, EPISODES)

print("\n--- Stage 2: ExplabOff ---")
expl_summary, expl_hist = run_algorithm('ExplabOff', ExplabOffAgent, EPISODES)

# Print comparison
print("\n" + "=" * 60)
print("COMPARISON RESULTS (last 10 episodes avg)")
print("=" * 60)
for metric in ['final_completion_rate', 'final_avg_latency', 'final_avg_energy', 'final_avg_reward']:
    i_val = ippo_summary[metric]
    e_val = expl_summary[metric]
    delta = e_val - i_val
    symbol = "↑" if delta > 0 else "↓"
    print(f"  {metric}: IPPO={i_val:.4f} → ExplabOff={e_val:.4f} ({symbol}{abs(delta):.4f})")

# Save results
out = Path("results/comparison.json")
out.parent.mkdir(parents=True, exist_ok=True)
with open(out, "w") as f:
    json.dump({"ippo": ippo_summary, "explaboff": expl_summary, "ippo_history": ippo_hist, "explaboff_history": expl_hist}, f, indent=2)
print(f"\nResults saved to {out}")
