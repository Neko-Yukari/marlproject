"""
Multi-Environment Benchmark — Overnight Training
=================================================
Runs IPPO vs ExplabOff vs Baselines across all 3 MEC configs:
  2ES-3MD, 2ES-5MD, 3ES-7MD

Training: 20K episodes per algorithm per config (with LR decay)
Random task profiles: new profile selected each episode
"""
import sys; sys.path.insert(0, '.')
import numpy as np, json, time, torch
from pathlib import Path
from datetime import datetime
from envs.paper_accurate_env_v3 import PaperAccurateEnvV3, ALL_PROFILES, ES_CPU_DB

RUN_ID = datetime.now().strftime('%Y%m%d_%H%M%S')
OUT_BASE = Path("results") / f"multi_env_benchmark_{RUN_ID}"
OUT_BASE.mkdir(parents=True, exist_ok=True)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Device: {device} | GPU: {torch.cuda.get_device_name(0) if device.type == 'cuda' else 'N/A'}")
print(f"Output: {OUT_BASE}\n")

# ══════════════════════════════════════════════════════
# Standard GPU-Optimized Configuration (RTX 4080 SUPER)
# Memory: ~290-675MB (2-4% of 16GB VRAM at batch=4096,hdim=1024)
# Bottleneck: CPU env simulation, not GPU compute
# ══════════════════════════════════════════════════════
STD = {
    "hidden_dim": 1024,
    "update_every": 500,
    "batch_size": 4096,
    "num_epochs": 10,
    "lr": 5e-5,
    "lr_step": 5000,
    "lr_gamma": 0.5,
}


# ═══════════════════════════════════════════════════
# Baselines
# ═══════════════════════════════════════════════════
def run_baselines(M, E, n_episodes=500):
    """Run all baselines on given MEC config."""
    env = PaperAccurateEnvV3(M, E, randomize_profile=True)
    es_cpus = env.es_cpu_list
    
    strategies = {
        "All_Local": lambda task_sizes, cpus: [0] * M,
        "All_BestES": lambda task_sizes, cpus: [E] * M,
        "Random": lambda task_sizes, cpus: [np.random.randint(0, E+1) for _ in range(M)],
        "Round_Robin": lambda task_sizes, cpus: [1 + (i % E) for i in range(M)],
        "Greedy": lambda task_sizes, cpus: [
            np.argmin([(s*1e6/10e6 + s*1e6*900/(c if c else 1e9)) for c in [1e9] + es_cpus])
            for s in task_sizes
        ],
        "Size_Based": lambda task_sizes, cpus: size_based_alloc(task_sizes, cpus),
    }
    
    results = {}
    for name, strategy in strategies.items():
        costs, comps = [], []
        for ep in range(n_episodes):
            obs, _ = env.reset(seed=42+ep)
            # Extract task sizes directly
            task_sizes = env._current_means
            actions_list = strategy(task_sizes, es_cpus)
            actions = {f"device_{i}": a for i, a in enumerate(actions_list)}
            _, rewards, _, _, _ = env.step(actions)
            m = env.get_episode_metrics()
            costs.append(m['avg_cost'])
            comps.append(m['completion_rate'])
        
        avg_cost = np.mean(costs)
        avg_comp = np.mean(comps)
        print(f"  {name:16s} | cost={avg_cost:.4f} | comp={avg_comp:.3f}")
        results[name] = {"avg_cost": avg_cost, "avg_comp": avg_comp}
    
    return results


def size_based_alloc(task_sizes, es_cpus):
    """Allocate largest tasks to fastest ES, smaller to slower ES.
    Greedy: sort tasks descending, assign each to ES that gives earliest completion."""
    sorted_idx = np.argsort(task_sizes)[::-1]
    sorted_cpus = sorted([(c, i) for i, c in enumerate(es_cpus)], key=lambda x: x[0], reverse=True)
    es_load = [0.0] * len(es_cpus)
    tx_rate = 10e6
    cpu_cycles = 900
    
    assignments = [0] * len(task_sizes)
    for idx in sorted_idx:
        s = task_sizes[idx]
        best_time = float('inf')
        best_es = 0
        for es_i, (cpu, _) in enumerate(sorted_cpus):
            t_edge = s * 1e6 / tx_rate + es_load[es_i] + s * 1e6 * cpu_cycles / cpu
            if t_edge < best_time:
                best_time = t_edge
                best_es = es_i
        es_load[best_es] += task_sizes[idx] * 1e6 * cpu_cycles / sorted_cpus[best_es][0]
        assignments[idx] = best_es + 1  # ES1-indexed
    
    return assignments


# ═══════════════════════════════════════════════════
# IPPO Training (with LR Decay)
# ═══════════════════════════════════════════════════
def train_ippo(M, E, episodes=20000, log_interval=1000):
    """Train IPPO on GPU with batched updates + LR decay."""
    from agents.ippo_agent import IPPOAgent
    
    env = PaperAccurateEnvV3(M, E, randomize_profile=True)
    adim = E + 1
    agents = [IPPOAgent(i, env.obs_dim, adim, hidden_dim=STD["hidden_dim"],
                        learning_rate=STD["lr"], device=device)
              for i in range(M)]
    
    # LR decay
    for agent in agents:
        agent.scheduler = torch.optim.lr_scheduler.StepLR(
            agent.optimizer, step_size=STD["lr_step"], gamma=STD["lr_gamma"])
    
    # Accumulate trajectories for batched GPU updates
    accum_traj = {i: {"states": [], "actions": [], "rewards": [], 
                       "values": [], "log_probs": [], "dones": []}
                  for i in range(M)}
    
    hist, t0 = [], time.time()
    
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
                accum_traj[i]["rewards"][-1] = rewards[a_name]
                ep_reward_sum += rewards[a_name]
            
            obs = next_obs
            if any(terms.values()):
                break
        
        # Batch update every N episodes
        if (ep + 1) % STD["update_every"] == 0:
            for i, agent in enumerate(agents):
                agent.trajectory = {k: list(v) for k, v in accum_traj[i].items()}
                agent.update(batch_size=STD["batch_size"], num_epochs=STD["num_epochs"])
                agent.scheduler.step()
                for k in accum_traj[i]:
                    accum_traj[i][k].clear()
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
                'time': el
            })
            eta = (episodes - ep) * (el / max(ep, 1)) if ep > 0 else 0
            print(f"  [IPPO] ep{ep:6d} cost={m['avg_cost']:.4f} comp={m['completion_rate']:.3f} "
                  f"lat={m['avg_latency']:.2f} t={el:.0f}s eta={eta:.0f}s")

    return hist


# ═══════════════════════════════════════════════════
# ExplabOff Training (GPU + Large Batch)
# ═══════════════════════════════════════════════════
def train_explaboff(M, E, episodes=20000, log_interval=1000):
    """Train ExplabOff with full MI on given MEC config."""
    from agents.explaboff_agent import ExplabOffAgent

    env = PaperAccurateEnvV3(M, E, randomize_profile=True)
    adim = E + 1
    agents = [ExplabOffAgent(i, env.obs_dim, adim, hidden_dim=STD["hidden_dim"],
                             lr=STD["lr"], mi_mu=3.5, mi_nu=1.0, device=device)
              for i in range(M)]
    
    accum_traj = {i: {"states": [], "actions": [], "rewards": [], 
                       "values": [], "log_probs": [], "dones": []}
                  for i in range(M)}
    episode_rewards = []
    hist, t0 = [], time.time()
    
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
                mi_r = agent.compute_mi_reward(obs[a_name], actions[a_name])
                combined = rewards[a_name] + mi_r
                accum_traj[i]["rewards"][-1] = combined
                ep_reward_sum += combined
            
            obs = next_obs
            if any(terms.values()):
                break
        
        episode_rewards.append(ep_reward_sum)
        
        # Batch update
        if (ep + 1) % STD["update_every"] == 0:
            for i, agent in enumerate(agents):
                agent.trajectory = {k: list(v) for k, v in accum_traj[i].items()}
                avg_reward = np.mean(episode_rewards[-STD["update_every"]:])
                agent.classify_episode(avg_reward)
                agent.update(batch_size=STD["batch_size"], num_epochs=STD["num_epochs"])
                for k in accum_traj[i]:
                    accum_traj[i][k].clear()
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
                'time': el
            })
            eta = (episodes - ep) * (el / max(ep, 1)) if ep > 0 else 0
            print(f"  [ExplabOff] ep{ep:6d} cost={m['avg_cost']:.4f} comp={m['completion_rate']:.3f} "
                  f"lat={m['avg_latency']:.2f} t={el:.0f}s eta={eta:.0f}s")
    
    return hist


# ═══════════════════════════════════════════════════
# Main: Run all benchmarks
# ═══════════════════════════════════════════════════
CONFIGS = [
    (2, 3, "2ES-3MD"),
    (2, 5, "2ES-5MD"),
    (3, 7, "3ES-7MD"),
]

all_results = {}

for E, M, name in CONFIGS:
    print(f"\n{'='*60}")
    print(f"Config: {name} ({E} Edge Servers, {M} Mobile Devices)")
    print(f"ES CPUs: {[f'{c/1e9:.1f}GHz' for c in ES_CPU_DB[(E, M)]]}")
    print(f"{'='*60}")
    
    # 1. Baselines
    print(f"\n--- Baselines ---")
    baselines = run_baselines(M, E, n_episodes=500)
    
    # 2. IPPO
    print(f"\n--- IPPO Training (20K episodes) ---")
    ippo_hist = train_ippo(M, E, episodes=20000, log_interval=1000)
    
    # 3. ExplabOff
    print(f"\n--- ExplabOff Training (20K episodes, Full MI) ---")
    explaboff_hist = train_explaboff(M, E, episodes=20000, log_interval=1000)
    
    all_results[name] = {
        "config": {"E": E, "M": M, "es_cpus": ES_CPU_DB[(E, M)]},
        "baselines": baselines,
        "ippo": {
            "best_cost": min(h['avg_cost'] for h in ippo_hist),
            "final_cost": ippo_hist[-1]['avg_cost'],
            "final_comp": ippo_hist[-1]['completion_rate'],
            "history": ippo_hist,
        },
        "explaboff": {
            "best_cost": min(h['avg_cost'] for h in explaboff_hist),
            "final_cost": explaboff_hist[-1]['avg_cost'],
            "final_comp": explaboff_hist[-1]['completion_rate'],
            "history": explaboff_hist,
        }
    }
    
    # Save intermediate
    with open(OUT_BASE / "all_results.json", "w") as f:
        # Convert to serializable format
        json.dump(all_results, f, indent=2, default=str)

# ═══════════════════════════════════════════════════
# Summary
# ═══════════════════════════════════════════════════
print(f"\n{'='*60}")
print(f"BENCHMARK COMPLETE")
print(f"{'='*60}")

for name, data in all_results.items():
    print(f"\n{name}:")
    print(f"  Baselines:")
    for bname, bres in data["baselines"].items():
        print(f"    {bname:16s}: cost={bres['avg_cost']:.4f} comp={bres['avg_comp']:.3f}")
    print(f"  IPPO:       cost={data['ippo']['final_cost']:.4f} comp={data['ippo']['final_comp']:.3f}")
    print(f"  ExplabOff:  cost={data['explaboff']['final_cost']:.4f} comp={data['explaboff']['final_comp']:.3f}")

print(f"\nResults saved to: {OUT_BASE}")
