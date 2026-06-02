"""Train MB-MERL on 3ES-7MD with meta-learning."""
import sys; sys.path.insert(0, '.')
import numpy as np, json, time, torch
from datetime import datetime
from pathlib import Path
from envs.paper_accurate_env_v3 import PaperAccurateEnvV3
from agents.mbmerl_agent import MBMERLAgent

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Device: {device}")

M, E = 7, 3
META_EPISODES = 5000  # Meta-training episodes
ADAPT_EPISODES = 100  # Adaptation episodes per test profile
TEST_PROFILES = 10    # Number of test profiles

# Create agents
agents = [MBMERLAgent(i, hidden_dim=64, meta_lr=1e-3, inner_lr=1e-2, device=device) 
          for i in range(M)]

env = PaperAccurateEnvV3(M, E, randomize_profile=True)

print("="*60)
print("MB-MERL Training")
print("="*60)

# Phase 1: Meta-training
print("\nPhase 1: Meta-training...")
t0 = time.time()

for ep in range(META_EPISODES):
    obs, _ = env.reset(seed=ep)
    
    # Collect experiences
    for step in range(10):
        actions = {}
        for i, agent in enumerate(agents):
            a_name = f"device_{i}"
            # Get ES loads and CPUs from obs
            es_loads = obs[a_name][1:1+E].tolist()
            es_cpus = obs[a_name][1+E:1+2*E].tolist()
            a = agent.select_action(obs[a_name], es_loads, es_cpus)
            actions[a_name] = a
        
        next_obs, rewards, terms, _, info = env.step(actions)
        
        # Store experiences
        for i, agent in enumerate(agents):
            a_name = f"device_{i}"
            task_size = obs[a_name][0]
            action = actions[a_name]
            cost = -rewards[a_name]  # Convert reward to cost
            
            if action == 0:
                # Local execution
                agent.store_experience(task_size, 0.0, 0.0, cost)
            else:
                es_idx = action - 1
                es_load = obs[a_name][1 + es_idx]
                es_cpu = obs[a_name][1 + E + es_idx]
                agent.store_experience(task_size, es_load, es_cpu, cost)
        
        obs = next_obs
        if any(terms.values()):
            break
    
    # Adapt every 50 episodes
    if (ep + 1) % 50 == 0:
        for agent in agents:
            agent.adapt(num_steps=5, batch_size=32)
    
    if ep % 500 == 0:
        m = env.get_episode_metrics()
        el = time.time() - t0
        print(f"ep{ep:5d} cost={m['avg_cost']:.4f} comp={m['completion_rate']:.3f} t={el:.0f}s")

# Phase 2: Few-shot adaptation on test profiles
print("\nPhase 2: Testing with few-shot adaptation...")
test_results = []

for test_idx in range(TEST_PROFILES):
    # Reset to new profile
    test_seed = 10000 + test_idx
    obs, _ = env.reset(seed=test_seed)
    
    # Collect adaptation data
    for ep in range(ADAPT_EPISODES):
        for step in range(10):
            actions = {}
            for i, agent in enumerate(agents):
                a_name = f"device_{i}"
                es_loads = obs[a_name][1:1+E].tolist()
                es_cpus = obs[a_name][1+E:1+2*E].tolist()
                a = agent.select_action(obs[a_name], es_loads, es_cpus)
                actions[a_name] = a
            
            next_obs, rewards, terms, _, _ = env.step(actions)
            
            for i, agent in enumerate(agents):
                a_name = f"device_{i}"
                task_size = obs[a_name][0]
                action = actions[a_name]
                cost = -rewards[a_name]
                
                if action == 0:
                    agent.store_experience(task_size, 0.0, 0.0, cost)
                else:
                    es_idx = action - 1
                    es_load = obs[a_name][1 + es_idx]
                    es_cpu = obs[a_name][1 + E + es_idx]
                    agent.store_experience(task_size, es_load, es_cpu, cost)
            
            obs = next_obs
            if any(terms.values()):
                break
        
        obs, _ = env.reset(seed=(test_seed + ep + 1))
    
    # Adapt
    for agent in agents:
        agent.adapt(num_steps=10, batch_size=64)
    
    # Evaluate
    eval_costs = []
    eval_comps = []
    for eval_ep in range(20):
        obs, _ = env.reset(seed=(test_seed + 1000 + eval_ep))
        for step in range(10):
            actions = {}
            for i, agent in enumerate(agents):
                a_name = f"device_{i}"
                es_loads = obs[a_name][1:1+E].tolist()
                es_cpus = obs[a_name][1+E:1+2*E].tolist()
                a = agent.select_action(obs[a_name], es_loads, es_cpus)
                actions[a_name] = a
            
            next_obs, rewards, terms, _, _ = env.step(actions)
            obs = next_obs
            if any(terms.values()):
                break
        
        m = env.get_episode_metrics()
        eval_costs.append(m['avg_cost'])
        eval_comps.append(m['completion_rate'])
    
    avg_cost = np.mean(eval_costs)
    avg_comp = np.mean(eval_comps)
    test_results.append({'profile': test_idx, 'cost': float(avg_cost), 'completion': float(avg_comp)})
    print(f"Profile {test_idx}: cost={avg_cost:.4f} comp={avg_comp:.3f}")

# Summary
print("\n" + "="*60)
print("MB-MERL Results")
print("="*60)
avg_cost = np.mean([r['cost'] for r in test_results])
avg_comp = np.mean([r['completion'] for r in test_results])
print(f"Average Cost: {avg_cost:.4f}")
print(f"Average Completion: {avg_comp:.3f}")

# Save results
out = Path("results") / f"mbmerl_3es7md_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
out.mkdir(parents=True, exist_ok=True)
with open(out / "results.json", "w") as f:
    json.dump({'meta_episodes': META_EPISODES, 'adapt_episodes': ADAPT_EPISODES,
               'test_profiles': TEST_PROFILES, 'results': test_results,
               'avg_cost': float(avg_cost), 'avg_completion': float(avg_comp)}, f, indent=2)

print(f"Saved to {out}")
