"""IPPO with Large Network (512h, 4L) + Action Masking — 3ES-7MD."""
import sys; sys.path.insert(0, '.')
import numpy as np, time, torch
from envs.paper_accurate_env_v3 import PaperAccurateEnvV3
from agents.ippo_agent import IPPOAgent

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Device: {device}")

M, E = 7, 3
EPISODES = 20000

CONFIG = {
    "hidden_dim": 512,
    "num_layers": 4,
    "update_every": 500,
    "batch_size": 2048,
    "num_epochs": 10,
    "lr": 5e-5,
}

env = PaperAccurateEnvV3(M, E, randomize_profile=True)
agents = [IPPOAgent(i, env.obs_dim, E+1, hidden_dim=CONFIG["hidden_dim"],
                    learning_rate=CONFIG["lr"], device=device)
          for i in range(M)]

for agent in agents:
    agent.network = agent.network.__class__(env.obs_dim, E+1, 
                                            hidden_dim=CONFIG["hidden_dim"],
                                            num_layers=CONFIG["num_layers"]).to(device)
    agent.optimizer = torch.optim.Adam(agent.network.parameters(), lr=CONFIG["lr"])

accum_traj = {i: {"states": [], "actions": [], "rewards": [], 
                   "values": [], "log_probs": [], "dones": []}
              for i in range(M)}

t0 = time.time()
best_cost = float('inf')

for ep in range(EPISODES):
    obs, _ = env.reset(seed=(42 + ep))
    ep_reward_sum = 0.0
    
    for step in range(10):
        actions = {}
        for i, agent in enumerate(agents):
            a_name = f"device_{i}"
            mask = env.compute_action_mask(a_name)
            a, lp, v = agent.select_action(obs[a_name], action_mask=mask)
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
    
    if (ep + 1) % CONFIG["update_every"] == 0:
        for i, agent in enumerate(agents):
            agent.trajectory = {k: list(v) for k, v in accum_traj[i].items()}
            agent.update(batch_size=CONFIG["batch_size"], num_epochs=CONFIG["num_epochs"])
            for k in accum_traj[i]:
                accum_traj[i][k].clear()
        if device.type == 'cuda':
            torch.cuda.empty_cache()
    
    if ep % 2000 == 0 or ep == EPISODES - 1:
        m = env.get_episode_metrics()
        el = time.time() - t0
        if m['avg_cost'] < best_cost:
            best_cost = m['avg_cost']
        print(f"ep{ep:5d} cost={m['avg_cost']:.4f} comp={m['completion_rate']:.3f} "
              f"lat={m['avg_latency']:.2f} best={best_cost:.4f} t={el:.0f}s")

print(f"\nDone! Best cost: {best_cost:.4f}")
