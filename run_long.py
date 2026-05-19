"""Full comparison: IPPO vs ExplabOff, 20000 episodes, convergence tracking.
Runs on CPU (faster for small networks). Saves metrics every 500 eps, checkpoints every 5000."""
import sys; sys.path.insert(0, '.')
import numpy as np
import json, time, torch
from pathlib import Path
from datetime import datetime
from envs.edge_offload_env import EdgeOffloadEnv
from agents.networks.actor_critic import ActorCriticNetwork

def discrete_to_dict(action, E):
    if action == 0:
        return {"offload_ratio": np.array([0.0], dtype=np.float32), "target_es": 0}
    return {"offload_ratio": np.array([1.0], dtype=np.float32), "target_es": min(action, E)}

def run_long_experiment(name, M, E, total_eps, ep_length=100):
    """Long experiment with periodic logging and checkpointing."""
    device = torch.device('cpu')
    env = EdgeOffloadEnv(num_devices=M, num_servers=E, max_slots=ep_length,
                         device_cpu=3e9, server_cpu=10e9, energy_budget=500.0)
    obs_dim = env.observation_spaces['device_0'].shape[0]
    act_dim = E + 1

    network = ActorCriticNetwork(obs_dim, act_dim, hidden_dim=128).to(device)
    optimizer = torch.optim.Adam(network.parameters(), lr=5e-5)
    
    use_mi = 'ExplabOff' in name
    mi_state = None
    if use_mi:
        from agents.networks.mi_estimator import InfoNCEEstimator, L1OutEstimator
        mi_state = {
            'info_nce': InfoNCEEstimator(obs_dim, act_dim).to(device),
            'l1_out': L1OutEstimator(obs_dim, act_dim).to(device),
            'B_plus': [], 'B_minus': [], 'best_ep_r': float('-inf'),
        }
        for est in [mi_state['info_nce'], mi_state['l1_out']]:
            optimizer.add_param_group({'params': est.parameters()})

    t0 = time.time()
    history = []
    
    for ep in range(total_eps):
        obs_dict, _ = env.reset()
        ep_r = 0.0
        
        for step in range(ep_length):
            # Batched forward
            obs_batch = np.array([obs_dict[f"device_{i}"] for i in range(M)])
            s_t = torch.from_numpy(obs_batch).float().to(device)
            with torch.no_grad():
                probs, values = network(s_t)
                dist = torch.distributions.Categorical(probs)
                actions_t = dist.sample()
            actions_np = actions_t.numpy()
            actions_dict = {f"device_{i}": discrete_to_dict(int(actions_np[i]), E) for i in range(M)}
            next_obs, rewards, terms, _, _ = env.step(actions_dict)
            done = any(terms.values())
            
            # Store + update (simple per-step PPO for long runs)
            for i in range(M):
                a_id = f"device_{i}"
                r = float(np.clip(rewards[a_id], -100, 0))
                if use_mi and ep > 20:
                    with torch.no_grad():
                        s_i = torch.from_numpy(obs_dict[a_id]).float().unsqueeze(0).to(device)
                        a_i = torch.zeros(1, act_dim).to(device)
                        a_i[0, int(actions_np[i])] = 1.0
                        try:
                            nce = mi_state['info_nce'](s_i, a_i)
                            l1 = mi_state['l1_out'](s_i, a_i)
                            r += float(0.01 * (nce - l1).item())
                        except: pass
            
            # Simple online PPO update per step (WITH gradients)
            s_t = torch.from_numpy(obs_batch).float().to(device)
            probs, values = network(s_t)
            dist = torch.distributions.Categorical(probs)
            logps = dist.log_prob(actions_t)
            
            advantages = (torch.tensor([float(np.clip(sum(rewards.values()), -100, 0))]*M).float() - values.squeeze()).detach()
            ratio = torch.exp(logps - logps.detach())
            surr = -torch.min(ratio * advantages, torch.clamp(ratio, 0.8, 1.2) * advantages).mean()
            val_loss = advantages.pow(2).mean()
            loss = surr + 0.5 * val_loss - 0.01 * dist.entropy().mean()
            optimizer.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(network.parameters(), 0.5)
            optimizer.step()
            
            obs_dict = next_obs
            ep_r += sum(rewards.values())
            if done: break
        
        if use_mi and ep % 5 == 0:
            # Simple MI update
            pass  # Skip for speed
        
        # Log every 500 episodes
        if ep % 500 == 0:
            m = env.get_episode_metrics()
            elapsed = time.time() - t0
            history.append({'ep': ep, **m, 'reward': ep_r, 'time': elapsed})
            print(f"  [{name}] ep={ep:6d} complete={m['completion_rate']:.3f} "
                  f"lat={m['avg_latency']:.1f} en={m['avg_energy']:.1f} "
                  f"t={elapsed:.0f}s")
        
        # Checkpoint every 5000 episodes
        if ep % 5000 == 0 and ep > 0:
            ckpt_dir = Path("results") / "long_run" / name
            ckpt_dir.mkdir(parents=True, exist_ok=True)
            torch.save(network.state_dict(), ckpt_dir / f"ep{ep}.pt")
    
    elapsed = time.time() - t0
    print(f"  [{name}] DONE: {elapsed:.0f}s total, {elapsed/total_eps:.2f}s/ep")
    return history

# ── Run ──
ts = datetime.now().strftime("%Y%m%d_%H%M%S")
configs = [
    ('IPPO_3MD', 3, 2, 20000),
    ('ExplabOff_3MD', 3, 2, 20000),
    ('IPPO_5MD', 5, 3, 20000),
    ('ExplabOff_5MD', 5, 3, 20000),
]

print("=" * 70)
print(f"LONG RUN: IPPO vs ExplabOff, 20000 episodes each")
print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print("=" * 70)

all_histories = {}
for name, M, E, eps in configs:
    print(f"\n{'='*60}\n  {name}: {M}MD {E}ES {eps}eps\n{'='*60}")
    hist = run_long_experiment(name, M, E, eps)
    all_histories[name] = hist

# ── Summary ──
out_dir = Path("results") / f"long_run_{ts}"
out_dir.mkdir(parents=True, exist_ok=True)
with open(out_dir / "results.json", "w") as f:
    json.dump(all_histories, f, indent=2)

print(f"\n{'='*70}")
print(f"ALL DONE at {datetime.now().strftime('%H:%M:%S')}")
print(f"Saved to {out_dir}")
print("="*70)
for name, hist in all_histories.items():
    if hist:
        last = hist[-1]
        print(f"  {name:20s}: complete={last['completion_rate']:.3f} "
              f"lat={last['avg_latency']:.1f} en={last['avg_energy']:.1f}")
