"""Evaluate saved checkpoints and generate final comparison."""
import sys; sys.path.insert(0, '.')
import numpy as np
import json
import torch
from pathlib import Path
from envs.edge_offload_env import EdgeOffloadEnv
from agents.networks.actor_critic import ActorCriticNetwork

def discrete_to_dict(action, E):
    if action == 0:
        return {"offload_ratio": np.array([0.0], dtype=np.float32), "target_es": 0}
    return {"offload_ratio": np.array([1.0], dtype=np.float32), "target_es": min(action, E)}

def evaluate_checkpoint(ckpt_path, M, E, num_eps=50):
    """Evaluate a checkpoint over multiple episodes."""
    env = EdgeOffloadEnv(num_devices=M, num_servers=E, max_slots=100,
                         device_cpu=3e9, server_cpu=10e9, energy_budget=500.0)
    obs_dim = env.observation_spaces['device_0'].shape[0]
    network = ActorCriticNetwork(obs_dim, E+1, 128)
    network.load_state_dict(torch.load(ckpt_path, map_location='cpu', weights_only=True))
    network.eval()
    
    metrics = {'completion': [], 'latency': [], 'energy': []}
    for _ in range(num_eps):
        obs_dict, _ = env.reset()
        for step in range(100):
            obs_batch = np.array([obs_dict[f"device_{i}"] for i in range(M)])
            with torch.no_grad():
                probs, _ = network(torch.from_numpy(obs_batch).float())
                actions = probs.argmax(-1).numpy()
            actions_dict = {f"device_{i}": discrete_to_dict(int(actions[i]), E) for i in range(M)}
            next_obs, rewards, terms, _, _ = env.step(actions_dict)
            obs_dict = next_obs
            if any(terms.values()): break
        m = env.get_episode_metrics()
        metrics['completion'].append(m['completion_rate'])
        metrics['latency'].append(m['avg_latency'])
        metrics['energy'].append(m['avg_energy'])
    
    return {k: (float(np.mean(v)), float(np.std(v))) for k, v in metrics.items()}

results = {}
for name, M, E in [('IPPO_3MD', 3, 2), ('ExplabOff_3MD', 3, 2)]:
    ckpt_dir = Path(f"results/long_run/{name}")
    for ep_tag in [5000, 10000, 15000]:
        ckpt = ckpt_dir / f"ep{ep_tag}.pt"
        if not ckpt.exists(): continue
        r = evaluate_checkpoint(str(ckpt), M, E)
        results[f"{name}_ep{ep_tag}"] = r
        print(f"{name} @ ep{ep_tag}: complete={r['completion'][0]:.3f}±{r['completion'][1]:.3f} "
              f"lat={r['latency'][0]:.1f} en={r['energy'][0]:.1f}")

print("\n=== CONVERGENCE SUMMARY ===")
for alg in ['IPPO_3MD', 'ExplabOff_3MD']:
    print(f"\n{alg}:")
    for ep in [5000, 10000, 15000]:
        k = f"{alg}_ep{ep}"
        if k in results:
            r = results[k]
            print(f"  ep{ep:6d}: {r['completion'][0]:.3f} ±{r['completion'][1]:.3f}")

with open("results/long_run/eval_results.json", "w") as f:
    json.dump(results, f, indent=2)
print("\nSaved to results/long_run/eval_results.json")
