"""Analyze ExplabOff behavior."""
import sys; sys.path.insert(0, '.')
import torch, numpy as np
from envs.paper_accurate_env_v3 import PaperAccurateEnvV3
from agents.explaboff_agent import ExplabOffAgent

device = torch.device('cpu')
env = PaperAccurateEnvV3(7, 3, randomize_profile=True)

# Load checkpoint
ckpt = torch.load('results/bench_3es7md_explaboff_20260530_005324/checkpoints/ep_4999.pt', map_location=device, weights_only=False)

agents = [ExplabOffAgent(i, env.obs_dim, 4, hidden_dim=1024, lr=5e-5, mi_mu=3.5, mi_nu=1.0, device=device) for i in range(7)]
for i, agent in enumerate(agents):
    agent.network.load_state_dict(ckpt['agents'][i]['network'])

N = 100
actions_all = {i: [] for i in range(7)}

for ep in range(N):
    obs, _ = env.reset(seed=(2000 + ep))
    for i, agent in enumerate(agents):
        a, _, _ = agent.select_action(obs[f'device_{i}'])
        actions_all[i].append(a)
    env.step({f'device_{i}': actions_all[i][-1] for i in range(7)})

print("="*60)
print("ExplabOff Behavior (Fixed Environment)")
print("="*60)

for i in range(7):
    counts = np.bincount(actions_all[i], minlength=4)
    dist = counts / counts.sum() * 100
    print(f"\nDevice {i}:")
    for a in range(4):
        names = ['Local', 'ES1', 'ES2', 'ES3']
        print(f"  {names[a]}: {dist[a]:5.1f}%")

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
