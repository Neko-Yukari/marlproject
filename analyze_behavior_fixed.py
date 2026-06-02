"""Re-analyze behavior with fixed env."""
import sys; sys.path.insert(0, '.')
import torch, numpy as np
from envs.paper_accurate_env_v3 import PaperAccurateEnvV3
from agents.ippo_agent import IPPOAgent
from agents.explaboff_agent import ExplabOffAgent

device = torch.device('cpu')
env = PaperAccurateEnvV3(7, 3, randomize_profile=True)

# Load IPPO checkpoint
ippo_ckpt = torch.load('results/bench_3es7md_ippo_20260530_000029/checkpoints/ep_14999.pt', map_location=device, weights_only=False)

# Create and load agents
ippo_agents = [IPPOAgent(i, env.obs_dim, 4, hidden_dim=1024, learning_rate=5e-5, device=device) for i in range(7)]
for i, agent in enumerate(ippo_agents):
    agent.network.load_state_dict(ippo_ckpt['agents'][i]['network'])

# Test 100 episodes
N = 100
actions_all = {i: [] for i in range(7)}

for ep in range(N):
    obs, _ = env.reset(seed=(1000 + ep))
    actions = {}
    for i, agent in enumerate(ippo_agents):
        a, _, _ = agent.select_action(obs[f'device_{i}'])
        actions_all[i].append(a)
        actions[f'device_{i}'] = a
    env.step(actions)

print("="*60)
print("IPPO Behavior (Fixed Environment)")
print("="*60)

for i in range(7):
    counts = np.bincount(actions_all[i], minlength=4)
    dist = counts / counts.sum() * 100
    print(f"\nDevice {i}:")
    for a in range(4):
        names = ['Local', 'ES1', 'ES2', 'ES3']
        print(f"  {names[a]}: {dist[a]:5.1f}%")

# ES load distribution
es_load = {0: 0, 1: 0, 2: 0, 3: 0}
for ep in range(N):
    for i in range(7):
        a = actions_all[i][ep]
        es_load[a] += 1

print(f"\nES Load Distribution:")
for es in range(4):
    names = ['Local', 'ES1', 'ES2', 'ES3']
    pct = es_load[es] / (N * 7) * 100
    print(f"  {names[es]}: {pct:5.1f}%")
