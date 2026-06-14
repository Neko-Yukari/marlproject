"""
GPU-optimized GNN training with dynamic batch size calculation.
Automatically computes max batch size based on available VRAM.
"""
import sys; sys.path.insert(0, '.')
import time
import torch
import numpy as np
from envs.paper_accurate_env import PaperAccurateEnvV3
from agents import GNNPolicy, PPOAgent

# ── Dynamic GPU Configuration ─────────────────────────────────────
device = torch.device('cuda')

def compute_optimal_batch():
    """Compute batch size and update frequency based on free VRAM."""
    torch.cuda.empty_cache()
    free_vram = torch.cuda.memory_reserved(device) - torch.cuda.memory_allocated(device)
    free_vram = max(free_vram, torch.cuda.get_device_properties(device).total_memory * 0.7)
    
    # Estimate memory per trajectory sample (obs + action + reward + value + log_prob + advantage)
    # Rough estimate: ~1KB per sample per agent for our small network
    bytes_per_sample = 2048  # Conservative estimate
    
    # Target using 80% of free VRAM for trajectory buffer
    target_buffer_bytes = free_vram * 0.8
    max_samples = int(target_buffer_bytes / bytes_per_sample)
    
    # For PPO, we want batch_size to be a power of 2, at least 64
    batch_size = 64
    while batch_size * 2 <= max_samples and batch_size < 2048:
        batch_size *= 2
    
    # update_every should accumulate at least 4x batch_size
    update_every = max(batch_size * 4, 500)
    
    return batch_size, update_every, free_vram / 1e9

BATCH_SIZE, UPDATE_EVERY, FREE_VRAM_GB = compute_optimal_batch()
NUM_EPISODES = 10000
LOG_INTERVAL = 1000
HIDDEN_DIM = 256
GNN_LAYERS = 3
LR = 1e-4

print(f"Device: {device} | {torch.cuda.get_device_name(0)}")
print(f"Free VRAM: {FREE_VRAM_GB:.1f} GB")
print(f"Dynamic Config: BATCH_SIZE={BATCH_SIZE}, UPDATE_EVERY={UPDATE_EVERY}")
print(f"Network: hidden={HIDDEN_DIM}, layers={GNN_LAYERS}, lr={LR}")
print("="*70)

# ── Training Setup ────────────────────────────────────────────────
env = PaperAccurateEnvV3(num_devices=3, num_servers=2)
policy = GNNPolicy(max_action_dim=4, hidden_dim=HIDDEN_DIM, gnn_layers=GNN_LAYERS).to(device)

agents = [PPOAgent(agent_id=i, policy_network=policy, device=device,
                   learning_rate=LR, entropy_coeff=0.01)
          for i in range(3)]

history = []
start = time.time()
best_cost = float('inf')

for ep in range(NUM_EPISODES):
    obs, _ = env.reset(seed=(42 + ep))
    
    for step in range(10):
        policy.set_graph(env)
        
        actions = {}
        for i, name in enumerate(env.agents):
            a, lp, v = agents[i].select_action(obs[name], agent_id=i)
            actions[name] = a
            agents[i]._last_value = v
            agents[i]._last_log_prob = lp
        
        next_obs, rewards, terms, truncs, infos = env.step(actions)
        
        for i, name in enumerate(env.agents):
            agents[i].store_transition(
                obs[name], actions[name], rewards[name],
                agents[i]._last_value, agents[i]._last_log_prob,
                terms[name] or truncs[name]
            )
        
        policy.clear_cache()
        obs = next_obs
    
    # Update with dynamically computed frequency
    if (ep + 1) % UPDATE_EVERY == 0:
        for agent in agents:
            if len(agent.trajectory['states']) > 0:
                agent.update(batch_size=BATCH_SIZE, num_epochs=4)
    
    if ep % LOG_INTERVAL == 0 or ep == NUM_EPISODES - 1:
        m = env.get_episode_metrics()
        elapsed = time.time() - start
        eps_per_sec = (ep + 1) / elapsed if elapsed > 0 else 0
        
        is_best = m['avg_cost'] < best_cost
        if is_best:
            best_cost = m['avg_cost']
        
        # Memory usage
        mem_alloc = torch.cuda.memory_allocated(device) / 1e9
        mem_res = torch.cuda.memory_reserved(device) / 1e9
        
        print(f"Ep {ep:5d} | Cost: {m['avg_cost']:.4f} | Comp: {m['completion_rate']:.1%} | "
              f"Speed: {eps_per_sec:.1f} eps/s | GPU: {mem_alloc:.1f}/{mem_res:.1f}GB | "
              f"{'BEST' if is_best else ''}")
        
        history.append({
            'episode': ep,
            'cost': float(m['avg_cost']),
            'completion': float(m['completion_rate']),
            'time': elapsed,
            'speed': eps_per_sec,
            'gpu_alloc': mem_alloc,
            'gpu_res': mem_res
        })

# ── Final Analysis ────────────────────────────────────────────────
costs = [h['cost'] for h in history]
if len(costs) >= 2:
    first = np.mean(costs[:len(costs)//2])
    second = np.mean(costs[len(costs)//2:])
    improvement = first - second
    
    print(f"\n{'='*70}")
    print("LEARNING ANALYSIS")
    print(f"{'='*70}")
    print(f"Batch size used:      {BATCH_SIZE}")
    print(f"Update every:         {UPDATE_EVERY}")
    print(f"First half avg cost:  {first:.4f}")
    print(f"Second half avg cost: {second:.4f}")
    print(f"Improvement:          {improvement:.4f} ({improvement/first*100:.1f}%)")
    print(f"Best cost achieved:   {best_cost:.4f}")
    print(f"Final cost:           {costs[-1]:.4f}")
    print(f"Total time:           {time.time()-start:.0f}s")
    
    if second < first:
        print(f"Status: LEARNING CONFIRMED")
    else:
        print(f"Status: WARNING - Cost did not decrease")

# Save
import json
with open('results/gnn_gpu_10k.json', 'w') as f:
    json.dump({
        'config': {
            'batch_size': BATCH_SIZE,
            'update_every': UPDATE_EVERY,
            'hidden_dim': HIDDEN_DIM,
            'gnn_layers': GNN_LAYERS,
            'lr': LR,
            'free_vram_gb': FREE_VRAM_GB
        },
        'history': history,
        'best_cost': float(best_cost),
        'learning': bool(second < first) if len(costs) >= 2 else False
    }, f, indent=2)

print(f"\nResults saved to results/gnn_gpu_10k.json")
