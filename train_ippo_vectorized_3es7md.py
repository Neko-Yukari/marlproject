"""IPPO with Vectorized Environments — 3ES-7MD."""
import sys; sys.path.insert(0, '.')
import numpy as np, json, time, torch
from datetime import datetime
from pathlib import Path
from envs.vectorized_env import VectorizedEnv
from agents.ippo_agent import IPPOAgent
from utils.reporter import TrainingReporter

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Device: {device}")

M, E = 7, 3
N_ENVS = 8  # Parallel environments
EPISODES = 20000 // N_ENVS  # Each env runs this many
LOG_INTERVAL = 100
CKPT_INTERVAL = 500

STD = {
    "hidden_dim": 256,
    "update_every": 100,
    "batch_size": 2048,
    "num_epochs": 10,
    "lr": 5e-5,
}

run_dir = f"results/bench_3es7md_ippo_vec_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
reporter = TrainingReporter(run_dir, config={"M": M, "E": E, "algorithm": "IPPO+Vec", "n_envs": N_ENVS, **STD})

# Create vectorized environment
vec_env = VectorizedEnv(M, E, n_envs=N_ENVS, randomize_profile=True)

# Create shared agents (same networks for all envs)
adim = E + 1
agents = [IPPOAgent(i, vec_env.envs[0].obs_dim, adim, hidden_dim=STD["hidden_dim"],
                    learning_rate=STD["lr"], device=device)
          for i in range(M)]

accum_traj = {i: {"states": [], "actions": [], "rewards": [], 
                   "values": [], "log_probs": [], "dones": []}
              for i in range(M)}

t0 = time.time()

for ep in range(EPISODES):
    # Reset all environments with different seeds
    seeds = [ep * N_ENVS + i + 1000 for i in range(N_ENVS)]
    obs_list = vec_env.reset(seeds)
    
    # Collect trajectories from all environments
    for env_idx in range(N_ENVS):
        obs = obs_list[env_idx]
        
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
            
            next_obs, rewards, terms, _, _ = vec_env.envs[env_idx].step(actions)
            
            for i in range(M):
                a_name = f"device_{i}"
                accum_traj[i]["rewards"][-1] = rewards[a_name]
            
            obs = next_obs
            if any(terms.values()):
                break
    
    # Update every N episodes
    if (ep + 1) % STD["update_every"] == 0:
        for i, agent in enumerate(agents):
            agent.trajectory = {k: list(v) for k, v in accum_traj[i].items()}
            agent.update(batch_size=STD["batch_size"], num_epochs=STD["num_epochs"])
            for k in accum_traj[i]:
                accum_traj[i][k].clear()
        if device.type == 'cuda':
            torch.cuda.empty_cache()
    
    # Log (use first env's metrics)
    if ep % LOG_INTERVAL == 0 or ep == EPISODES - 1:
        metrics = vec_env.get_metrics()
        m = metrics[0]  # First env
        reporter.log_episode(ep * N_ENVS, m)  # Scale episode count
        el = time.time() - t0
        total_eps = (ep + 1) * N_ENVS
        print(f"\n{'='*60}")
        print(f"[IPPO+Vec] ep{total_eps:6d} cost={m['avg_cost']:.4f} comp={m['completion_rate']:.3f} "
              f"lat={m['avg_latency']:.2f} t={el:.0f}s")
        
        report_path = reporter.generate_report(ep * N_ENVS)
        with open(report_path) as f:
            r = json.load(f)
        print(f"  REPORT: {report_path}")
        print(f"     Best: cost={r['metrics']['best']['avg_cost']:.4f} @ ep{r['metrics']['best']['episode']}")
        print(f"     Speed: {r['training_stats']['episodes_per_second']:.2f} eps/s")
        print(f"{'='*60}")
    
    if (ep + 1) % CKPT_INTERVAL == 0:
        path = reporter.save_checkpoint(ep * N_ENVS, agents)
        print(f"  Checkpoint: {path}")

results = reporter.finalize(agents)
print(f"\nDone! Best: {results['best']['avg_cost']:.4f} @ ep{results['best']['episode']}")
print(f"Total time: {results['total_time_seconds']:.0f}s")
