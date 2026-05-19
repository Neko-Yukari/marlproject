"""GPU-optimized MARL comparison experiments with batched agent processing."""
import sys; sys.path.insert(0, '.')
import numpy as np
import json, time, torch
from pathlib import Path
from datetime import datetime
from envs.edge_offload_env import EdgeOffloadEnv
from agents.networks.actor_critic import ActorCriticNetwork
from agents.explaboff_agent import ExplabOffAgent

def discrete_to_dict(action, E):
    if action == 0:
        return {"offload_ratio": np.array([0.0], dtype=np.float32), "target_es": 0}
    return {"offload_ratio": np.array([1.0], dtype=np.float32), "target_es": min(action, E)}

def run_batched_experiment(name, M, E, episodes, ep_length):
    """GPU-batched experiment: all agents share one network, batched forward passes."""
    t0 = time.time()
    device = torch.device('cuda')
    env = EdgeOffloadEnv(num_devices=M, num_servers=E, max_slots=ep_length,
                         device_cpu=3e9, server_cpu=10e9, energy_budget=500.0)
    obs_dim = env.observation_spaces['device_0'].shape[0]
    act_dim = E + 1

    # Shared network (parameter sharing)
    network = ActorCriticNetwork(obs_dim, act_dim, hidden_dim=128).to(device)
    optimizer = torch.optim.Adam(network.parameters(), lr=5e-5)

    # MI components (only for ExplabOff)
    use_mi = name.startswith('ExplabOff') and 'no_mi' not in name
    mi_estimators = None
    if use_mi:
        from agents.networks.mi_estimator import InfoNCEEstimator, L1OutEstimator
        mi_estimators = {
            'info_nce': InfoNCEEstimator(obs_dim, act_dim).to(device),
            'l1_out': L1OutEstimator(obs_dim, act_dim).to(device),
            'B_plus': [], 'B_minus': [], 'best_ep_r': float('-inf'),
        }
        for est in [mi_estimators['info_nce'], mi_estimators['l1_out']]:
            optimizer.add_param_group({'params': est.parameters()})

    # Batch trajectory buffer
    B_MAX = ep_length * M
    trajectory = {'states': np.zeros((B_MAX, obs_dim), dtype=np.float32),
                  'actions': np.zeros(B_MAX, dtype=np.int64),
                  'rewards': np.zeros(B_MAX, dtype=np.float32),
                  'values': np.zeros(B_MAX, dtype=np.float32),
                  'log_probs': np.zeros(B_MAX, dtype=np.float32),
                  'dones': np.zeros(B_MAX, dtype=np.bool_)}

    gamma, gae_lambda, clip_ratio = 0.99, 0.95, 0.2
    entropy_coeff, value_coeff = 0.01, 0.5

    history = []
    for ep in range(episodes):
        obs_dict, _ = env.reset()
        ptr = 0  # trajectory pointer

        for step in range(ep_length):
            # ── BATCHED select_action for ALL agents ──
            obs_batch = np.array([obs_dict[f"device_{i}"] for i in range(M)])
            s_t = torch.from_numpy(obs_batch).float().to(device)
            with torch.no_grad():
                probs, values = network(s_t)
                dist = torch.distributions.Categorical(probs)
                actions_t = dist.sample()  # [M]
                logps_t = dist.log_prob(actions_t)  # [M]
            actions_np = actions_t.cpu().numpy()
            logps_np = logps_t.cpu().numpy()
            values_np = values.squeeze(-1).cpu().numpy()

            # Build actions dict
            actions_dict = {f"device_{i}": discrete_to_dict(int(actions_np[i]), E) for i in range(M)}

            # Environment step
            next_obs, rewards, terms, truncs, _ = env.step(actions_dict)

            # Store transitions
            for i in range(M):
                a_id = f"device_{i}"
                r = float(np.clip(rewards[a_id], -100, 0))
                trajectory['states'][ptr] = obs_dict[a_id]
                trajectory['actions'][ptr] = actions_np[i]
                trajectory['rewards'][ptr] = r
                trajectory['values'][ptr] = values_np[i]
                trajectory['log_probs'][ptr] = logps_np[i]
                trajectory['dones'][ptr] = terms[a_id]
                ptr += 1

            obs_dict = next_obs
            if any(terms.values()): break

        # ── PPO Update (batched) ──
        if ptr < 2: continue
        n = ptr
        states_t = torch.from_numpy(trajectory['states'][:n]).float().to(device)
        actions_t = torch.from_numpy(trajectory['actions'][:n]).long().to(device)
        rewards_t = torch.from_numpy(trajectory['rewards'][:n]).float().to(device)
        values_t = torch.from_numpy(trajectory['values'][:n]).float().to(device)
        logps_t = torch.from_numpy(trajectory['log_probs'][:n]).float().to(device)
        dones_t = torch.from_numpy(trajectory['dones'][:n]).float().to(device)

        # GAE (CPU numpy is fine for this)
        rewards_np = trajectory['rewards'][:n]
        values_np = np.append(trajectory['values'][:n], 0.0)
        dones_np = trajectory['dones'][:n].astype(float)
        deltas = rewards_np + gamma * values_np[1:] * (1 - dones_np) - values_np[:-1]
        adv = np.zeros(n); gae_val = 0.0
        for t in reversed(range(n)):
            gae_val = deltas[t] + gamma * gae_lambda * (1 - dones_np[t]) * gae_val
            adv[t] = gae_val
        ret = adv + values_np[:-1]
        adv = (adv - adv.mean()) / (adv.std() + 1e-8)

        adv_t = torch.from_numpy(adv).float().to(device)
        ret_t = torch.from_numpy(ret).float().to(device)

        for _ in range(2):  # 2 epochs
            perm = np.random.permutation(n)
            for i in range(0, n, 64):
                idx = perm[i:i+64]
                probs, vals = network(states_t[idx])
                dist = torch.distributions.Categorical(probs)
                new_lp = dist.log_prob(actions_t[idx])
                ratio = torch.exp(new_lp - logps_t[idx])
                surr1 = ratio * adv_t[idx]
                surr2 = torch.clamp(ratio, 1-clip_ratio, 1+clip_ratio) * adv_t[idx]
                policy_loss = -torch.min(surr1, surr2).mean()
                value_loss = 0.5 * (ret_t[idx] - vals.squeeze(-1)).pow(2).mean()
                entropy = dist.entropy().mean()
                loss = policy_loss + value_coeff*value_loss - entropy_coeff*entropy
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(network.parameters(), 0.5)
                optimizer.step()

        m = env.get_episode_metrics()
        history.append({'ep': ep, 'completion_rate': m['completion_rate'],
                        'avg_latency': m['avg_latency'], 'avg_energy': m['avg_energy']})

    elapsed = time.time() - t0
    last20 = history[-20:]
    return {
        'name': name, 'M': M, 'E': E, 'episodes': episodes, 'time_sec': elapsed,
        'completion_rate': float(np.mean([h['completion_rate'] for h in last20])),
        'avg_latency': float(np.mean([h['avg_latency'] for h in last20])),
        'avg_energy': float(np.mean([h['avg_energy'] for h in last20])),
    }

# ── Run experiments ──
results = []
ts = datetime.now().strftime("%Y%m%d_%H%M%S")

for name, M, E, eps in [
    ('IPPO_batched', 3, 2, 500),
    ('ExplabOff_batched', 3, 2, 500),
    ('IPPO_batched', 5, 3, 500),
    ('ExplabOff_batched', 5, 3, 500),
]:
    print(f"\n{'='*60}\n{name}: M={M} E={E} eps={eps} on GPU")
    r = run_batched_experiment(name, M, E, eps, ep_length=100)
    print(f"  completion={r['completion_rate']:.3f}, time={r['time_sec']:.0f}s")
    results.append(r)

print("\n" + "="*60)
print("GPU BATCHED RESULTS")
print("="*60)
for r in results:
    delta = r['time_sec'] / r['episodes']
    print(f"  {r['name']:25s} M={r['M']} → {r['completion_rate']:.3f}  {r['time_sec']:.0f}s ({delta:.2f}s/ep)")

out_dir = Path("results") / f"batched_{ts}"
out_dir.mkdir(parents=True, exist_ok=True)
with open(out_dir / "summary.json", "w") as f:
    json.dump(results, f, indent=2)
print(f"\nSaved to {out_dir}")
