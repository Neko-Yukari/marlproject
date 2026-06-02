"""Behavior Analysis: IPPO vs ExplabOff task allocation patterns."""
import sys; sys.path.insert(0, '.')
import torch, numpy as np, json
from pathlib import Path
from envs.paper_accurate_env_v3 import PaperAccurateEnvV3
from agents.ippo_agent import IPPOAgent
from agents.explaboff_agent import ExplabOffAgent

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
M, E = 7, 3
NUM_TEST_EPISODES = 100

# Find checkpoints
ippo_ckpts = sorted(Path("results").glob("bench_3es7md_ippo_*/checkpoints/*.pt"), 
                    key=lambda x: x.stat().st_mtime, reverse=True)
expl_ckpts = sorted(Path("results").glob("bench_3es7md_explaboff_*/checkpoints/*.pt"), 
                    key=lambda x: x.stat().st_mtime, reverse=True)

print(f"Found IPPO checkpoints: {len(ippo_ckpts)}")
print(f"Found ExplabOff checkpoints: {len(expl_ckpts)}")

if not ippo_ckpts or not expl_ckpts:
    print("Missing checkpoints! Run training first.")
    sys.exit(1)

# Load IPPO
env = PaperAccurateEnvV3(M, E, randomize_profile=True)
ippo_agents = [IPPOAgent(i, env.obs_dim, E+1, hidden_dim=1024, device=device) for i in range(M)]
ckpt = torch.load(ippo_ckpts[0], map_location=device)
for i, agent in enumerate(ippo_agents):
    agent.network.load_state_dict(ckpt['agents'][i]['network'])
    agent.network.eval()

# Load ExplabOff  
expl_agents = [ExplabOffAgent(i, env.obs_dim, E+1, hidden_dim=1024, mi_mu=3.5, mi_nu=1.0, device=device) 
               for i in range(M)]
ckpt = torch.load(expl_ckpts[0], map_location=device)
for i, agent in enumerate(expl_agents):
    agent.network.load_state_dict(ckpt['agents'][i]['network'])
    agent.network.eval()

# Test function
def evaluate_agents(agents, name):
    action_counts = {f"device_{i}": {a: 0 for a in range(E+1)} for i in range(M)}
    total_costs = []
    completions = []
    latency_dist = []
    energy_dist = []
    action_matrix = []  # per-episode action distribution
    
    for ep in range(NUM_TEST_EPISODES):
        obs, _ = env.reset(seed=(10000 + ep))
        ep_actions = []
        
        for step in range(10):
            actions = {}
            for i, agent in enumerate(agents):
                a_name = f"device_{i}"
                a, _, _ = agent.select_action(obs[a_name])
                actions[a_name] = a
                action_counts[a_name][a] += 1
            
            ep_actions.append(actions)
            next_obs, rewards, terms, _, _ = env.step(actions)
            obs = next_obs
            if any(terms.values()):
                break
        
        m = env.get_episode_metrics()
        total_costs.append(m['avg_cost'])
        completions.append(m['completion_rate'])
        latency_dist.append(m['avg_latency'])
        energy_dist.append(m['avg_energy'])
        action_matrix.append(ep_actions)
    
    # Analysis
    print(f"\n{'='*70}")
    print(f"{name} Behavior Analysis ({NUM_TEST_EPISODES} episodes)")
    print(f"{'='*70}")
    print(f"\nOverall Performance:")
    print(f"  Mean Cost: {np.mean(total_costs):.4f} (±{np.std(total_costs):.4f})")
    print(f"  Mean Completion: {np.mean(completions):.1%} (±{np.std(completions):.1%})")
    print(f"  Mean Latency: {np.mean(latency_dist):.3f}s")
    print(f"  Mean Energy: {np.mean(energy_dist):.4f}")
    
    print(f"\nAction Distribution (per device, %):")
    action_labels = ['Local', 'ES1', 'ES2', 'ES3']
    print(f"  {'Device':<10} {'Local':>8} {'ES1':>8} {'ES2':>8} {'ES3':>8}")
    print(f"  {'-'*46}")
    for i in range(M):
        a_name = f"device_{i}"
        total = sum(action_counts[a_name].values())
        if total > 0:
            dist = [action_counts[a_name][a]/total*100 for a in range(E+1)]
            print(f"  {a_name:<10} {dist[0]:>7.1f}% {dist[1]:>7.1f}% {dist[2]:>7.1f}% {dist[3]:>7.1f}%")
    
    # ES utilization
    es_load = {f"ES{i}": 0 for i in range(1, E+1)}
    for i in range(M):
        a_name = f"device_{i}"
        total = sum(action_counts[a_name].values())
        if total > 0:
            for es in range(1, E+1):
                es_load[f"ES{es}"] += action_counts[a_name][es]
    
    print(f"\nES Total Load (tasks assigned):")
    total_all = sum(es_load.values())
    for es, count in es_load.items():
        if total_all > 0:
            print(f"  {es}: {count} ({count/total_all*100:.1f}%)")
    
    # Identify patterns
    print(f"\nKey Patterns:")
    local_heavy = sum(action_counts[f"device_{i}"][0] for i in range(M))
    offload_heavy = sum(sum(action_counts[f"device_{i}"][a] for a in range(1, E+1)) for i in range(M))
    total_all = local_heavy + offload_heavy
    if total_all > 0:
        print(f"  Local vs Offload ratio: {local_heavy/total_all*100:.1f}% : {offload_heavy/total_all*100:.1f}%")
    
    return {
        'name': name,
        'cost_mean': float(np.mean(total_costs)),
        'cost_std': float(np.std(total_costs)),
        'completion_mean': float(np.mean(completions)),
        'action_counts': action_counts,
        'es_load': es_load,
        'latency_mean': float(np.mean(latency_dist)),
        'energy_mean': float(np.mean(energy_dist))
    }

# Evaluate both
ippo_results = evaluate_agents(ippo_agents, "IPPO")
expl_results = evaluate_agents(expl_agents, "ExplabOff")

# Comparison
print(f"\n{'='*70}")
print(f"HEAD-TO-HEAD COMPARISON")
print(f"{'='*70}")
print(f"\n{'Metric':<25} {'IPPO':>12} {'ExplabOff':>12} {'Diff':>12}")
print(f"{'-'*65}")
print(f"{'Cost':<25} {ippo_results['cost_mean']:>12.4f} {expl_results['cost_mean']:>12.4f} {ippo_results['cost_mean']-expl_results['cost_mean']:>+12.4f}")
print(f"{'Completion':<25} {ippo_results['completion_mean']:>11.1%} {expl_results['completion_mean']:>11.1%} {ippo_results['completion_mean']-expl_results['completion_mean']:>+11.1%}")
print(f"{'Latency':<25} {ippo_results['latency_mean']:>12.3f} {expl_results['latency_mean']:>12.3f} {ippo_results['latency_mean']-expl_results['latency_mean']:>+12.3f}")
print(f"{'Energy':<25} {ippo_results['energy_mean']:>12.4f} {expl_results['energy_mean']:>12.4f} {ippo_results['energy_mean']-expl_results['energy_mean']:>+12.4f}")

# Save report
report = {
    'ippo': ippo_results,
    'explaboff': expl_results,
    'analysis': {
        'ippobetter': ippo_results['cost_mean'] < expl_results['cost_mean'],
        'costdiff': float(ippo_results['cost_mean'] - expl_results['cost_mean'])
    }
}

with open('behavior_analysis.json', 'w') as f:
    json.dump(report, f, indent=2)

print(f"\nSaved to behavior_analysis.json")
print(f"\n{'='*70}")
