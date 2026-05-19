"""Comprehensive MARL comparison experiments."""
import sys; sys.path.insert(0, '.')
import numpy as np
import json
import time
from pathlib import Path
from datetime import datetime
from envs.edge_offload_env import EdgeOffloadEnv
from agents import IPPOAgent
from agents.explaboff_agent import ExplabOffAgent

def discrete_to_dict(action, E):
    if action == 0:
        return {"offload_ratio": np.array([0.0], dtype=np.float32), "target_es": 0}
    return {"offload_ratio": np.array([1.0], dtype=np.float32), "target_es": min(action, E)}

def run_experiment(name, agent_cls, M, E, episodes, ep_length, **kwargs):
    """Single experiment run."""
    t0 = time.time()
    device = torch.device('cuda')
    env = EdgeOffloadEnv(num_devices=M, num_servers=E, max_slots=ep_length,
                         device_cpu=3e9, server_cpu=10e9, energy_budget=500.0)
    obs_dim = env.observation_spaces['device_0'].shape[0]
    act_dim = E + 1
    agents = [agent_cls(agent_id=i, state_dim=obs_dim, action_dim=act_dim, device=device, **kwargs) for i in range(M)]
    if agent_cls == IPPOAgent:
        shared = agents[0].network
        for a in agents[1:]: a.network = shared

    history = []
    for ep in range(episodes):
        obs, _ = env.reset()
        ep_r_total = 0.0
        for step in range(ep_length):
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
                r = float(np.clip(rewards[a_id], -100, 0))
                if name.startswith('ExplabOff') and 'no_mi' not in name and ep > 10:
                    r += agent.compute_mi_reward(obs[a_id], act)
                agent.store_transition(obs[a_id], act, r, val, logp, terms[a_id])
            obs = next_obs
            ep_r_total += sum(rewards.values())
            if any(terms.values()): break

        for agent in agents:
            agent.update(batch_size=16, num_epochs=2)
            agent.clear_trajectory()

        if name.startswith('ExplabOff') and 'no_mi' not in name:
            for agent in agents:
                agent.classify_episode(float(ep_r_total))
            if ep % 5 == 0:
                for agent in agents:
                    agent.update_mi_estimators()

        m = env.get_episode_metrics()
        history.append({'ep': ep, 'reward': ep_r_total, **m})

    elapsed = time.time() - t0
    last20 = history[-20:]
    return {
        'name': name, 'M': M, 'E': E, 'episodes': episodes,
        'time_sec': elapsed,
        'completion_rate': float(np.mean([h['completion_rate'] for h in last20])),
        'failure_rate': float(np.mean([h['failure_rate'] for h in last20])),
        'avg_latency': float(np.mean([h['avg_latency'] for h in last20])),
        'avg_energy': float(np.mean([h['avg_energy'] for h in last20])),
        'avg_reward': float(np.mean([h['reward'] for h in last20])),
        'history': history,
    }

import torch
results = []
ts = datetime.now().strftime("%Y%m%d_%H%M%S")

# ═══════════════════════════════════════════════════════
# Experiment 1: IPPO vs ExplabOff, 3 devices, 500 eps
# ═══════════════════════════════════════════════════════
print("=" * 70)
print("EXP 1: IPPO vs ExplabOff, 3MD 2ES, 500 episodes")
print("=" * 70)
r1a = run_experiment('IPPO', IPPOAgent, M=3, E=2, episodes=500, ep_length=50)
print(f"  IPPO: completion={r1a['completion_rate']:.3f}, time={r1a['time_sec']:.0f}s")
r1b = run_experiment('ExplabOff', ExplabOffAgent, M=3, E=2, episodes=500, ep_length=50)
print(f"  ExplabOff: completion={r1b['completion_rate']:.3f}, time={r1b['time_sec']:.0f}s")
results.extend([r1a, r1b])

# ═══════════════════════════════════════════════════════
# Experiment 2: Scale comparison — 5 devices, 500 eps
# ═══════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("EXP 2: IPPO vs ExplabOff, 5MD 3ES, 500 episodes")
print("=" * 70)
r2a = run_experiment('IPPO', IPPOAgent, M=5, E=3, episodes=500, ep_length=50)
print(f"  IPPO: completion={r2a['completion_rate']:.3f}, time={r2a['time_sec']:.0f}s")
r2b = run_experiment('ExplabOff', ExplabOffAgent, M=5, E=3, episodes=500, ep_length=50)
print(f"  ExplabOff: completion={r2b['completion_rate']:.3f}, time={r2b['time_sec']:.0f}s")
results.extend([r2a, r2b])

# ═══════════════════════════════════════════════════════
# Experiment 3: Ablation — ExplabOff without MI reward
# ═══════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("EXP 3: Ablation — ExplabOff without MI, 3MD 2ES, 500 eps")
print("=" * 70)
r3a = run_experiment('ExplabOff_noMI', ExplabOffAgent, M=3, E=2, episodes=500, ep_length=50)
print(f"  ExplabOff_noMI: completion={r3a['completion_rate']:.3f}, time={r3a['time_sec']:.0f}s")
results.append(r3a)

# ═══════════════════════════════════════════════════════
# Experiment 4: Convergence speed — check every 50 eps
# ═══════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("EXP 4: Convergence analysis — tracking per-50-eps")
print("=" * 70)
convergence = {'IPPO': [], 'ExplabOff': []}
for alg_key, agent_cls in [('IPPO', IPPOAgent), ('ExplabOff', ExplabOffAgent)]:
    env = EdgeOffloadEnv(num_devices=3, num_servers=2, max_slots=50,
                         device_cpu=3e9, server_cpu=10e9, energy_budget=500.0)
    obs_dim = env.observation_spaces['device_0'].shape[0]
    agents = [agent_cls(agent_id=i, state_dim=obs_dim, action_dim=env.E+1) for i in range(3)]
    if agent_cls == IPPOAgent:
        shared = agents[0].network
        for a in agents[1:]: a.network = shared

    snapshots = []
    for ep in range(500):
        obs, _ = env.reset()
        ep_r = 0.0
        for step in range(50):
            agent_data = {}
            actions = {}
            for i, agent in enumerate(agents):
                a_id = f"device_{i}"
                act, logp, val = agent.select_action(obs[a_id])
                agent_data[a_id] = (act, logp, val)
                actions[a_id] = discrete_to_dict(act, env.E)
            next_obs, rewards, terms, truncs, _ = env.step(actions)
            for i, agent in enumerate(agents):
                a_id = f"device_{i}"
                act, logp, val = agent_data[a_id]
                r = float(np.clip(rewards[a_id], -100, 0))
                if alg_key == 'ExplabOff' and ep > 10:
                    r += agent.compute_mi_reward(obs[a_id], act)
                agent.store_transition(obs[a_id], act, r, val, logp, terms[a_id])
            obs = next_obs
            ep_r += sum(rewards.values())
            if any(terms.values()): break
        for agent in agents:
            agent.update(batch_size=16, num_epochs=2)
            agent.clear_trajectory()
        if alg_key == 'ExplabOff':
            for agent in agents: agent.classify_episode(float(ep_r))
            if ep % 5 == 0:
                for agent in agents: agent.update_mi_estimators()
        if ep % 50 == 0:
            m = env.get_episode_metrics()
            snapshots.append({'ep': ep, 'completion': m['completion_rate'], 'reward': ep_r})
    convergence[alg_key] = snapshots

# ═══════════════════════════════════════════════════════
# Summary
# ═══════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("FINAL RESULTS")
print("=" * 70)
print(f"{'Experiment':<30} {'Completion':>10} {'Latency':>10} {'Energy':>10} {'Time(s)':>8}")
print("-" * 70)
for r in results:
    print(f"{r['name']:<30} {r['completion_rate']:>10.4f} {r['avg_latency']:>10.2f} {r['avg_energy']:>10.2f} {r['time_sec']:>8.0f}")

# Save
out_dir = Path("results") / f"comprehensive_{ts}"
out_dir.mkdir(parents=True, exist_ok=True)
summary_data = [{'name':r['name'],'completion':r['completion_rate'],'latency':r['avg_latency'],
                  'energy':r['avg_energy'],'reward':r['avg_reward'],'time':r['time_sec'],
                  'M':r['M'],'E':r['E']} for r in results]
with open(out_dir / "summary.json", "w") as f: json.dump(summary_data, f, indent=2)
with open(out_dir / "convergence.json", "w") as f: json.dump(convergence, f, indent=2)
print(f"\nSaved to {out_dir}")
