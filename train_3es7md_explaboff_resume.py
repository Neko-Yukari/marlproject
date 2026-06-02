"""ExplabOff 3ES-7MD - Auto-resume training."""
import sys; sys.path.insert(0, '.')
import numpy as np, json, time, torch
from datetime import datetime
from pathlib import Path
from envs.paper_accurate_env_v3 import PaperAccurateEnvV3
from agents.explaboff_agent import ExplabOffAgent

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Device: {device}")

M, E = 7, 3
SEGMENT = 5000
TOTAL = 20000

STD = {
    "hidden_dim": 1024,
    "update_every": 500,
    "batch_size": 2048,
    "num_epochs": 10,
    "lr": 5e-5,
    "mi_mu": 3.5,
    "mi_nu": 1.0,
}

env = PaperAccurateEnvV3(M, E, randomize_profile=True)
adim = E + 1
agents = [ExplabOffAgent(i, env.obs_dim, adim, hidden_dim=STD["hidden_dim"], lr=STD["lr"],
                         mi_mu=STD["mi_mu"], mi_nu=STD["mi_nu"], device=device)
          for i in range(M)]

# Try to load checkpoint
ckpt_dir = Path("results/explaboff_3es7md_checkpoints")
ckpt_dir.mkdir(parents=True, exist_ok=True)
latest = sorted(ckpt_dir.glob("ep_*.pt"), key=lambda x: int(x.stem.split('_')[1]), reverse=True)
start_ep = 0
if latest:
    print(f"Loading checkpoint: {latest[0]}")
    ckpt = torch.load(latest[0], weights_only=False)
    for i, agent in enumerate(agents):
        agent.network.load_state_dict(ckpt[f'agent_{i}'])
    start_ep = int(latest[0].stem.split('_')[1])
    print(f"Resuming from ep{start_ep}")

accum_traj = {i: {"states": [], "actions": [], "rewards": [], 
                   "values": [], "log_probs": [], "dones": []}
              for i in range(M)}

t0 = time.time()
best_cost = float('inf')
best_ep = 0

for ep in range(start_ep, TOTAL):
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
            accum_traj[i]["rewards"][-1] = rewards[a_name] + mi_r
            ep_reward_sum += rewards[a_name] + mi_r
        
        obs = next_obs
        if any(terms.values()):
            break
    
    if (ep + 1) % STD["update_every"] == 0:
        for i, agent in enumerate(agents):
            agent.trajectory = {k: list(v) for k, v in accum_traj[i].items()}
            agent.update(batch_size=STD["batch_size"], num_epochs=STD["num_epochs"])
            for k in accum_traj[i]:
                accum_traj[i][k].clear()
        if device.type == 'cuda':
            torch.cuda.empty_cache()
    
    if ep % 500 == 0:
        m = env.get_episode_metrics()
        el = time.time() - t0
        if m['avg_cost'] < best_cost:
            best_cost = m['avg_cost']
            best_ep = ep
        print(f"ep{ep:5d} cost={m['avg_cost']:.4f} comp={m['completion_rate']:.3f} "
              f"lat={m['avg_latency']:.2f} t={el:.0f}s best={best_cost:.4f}@{best_ep}")
    
    # Save checkpoint every segment
    if (ep + 1) % SEGMENT == 0:
        ckpt_path = ckpt_dir / f"ep_{ep+1}.pt"
        torch.save({f'agent_{i}': agent.network.state_dict() for i, agent in enumerate(agents)}, ckpt_path)
        print(f"  Checkpoint saved: {ckpt_path}")

print(f"\nDone! Best: {best_cost:.4f} @ ep{best_ep}")
