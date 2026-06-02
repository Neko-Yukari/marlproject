"""Training Pipeline Validation — End-to-end verification."""
import sys; sys.path.insert(0, '.')
import torch, numpy as np
from pathlib import Path
from envs.paper_accurate_env_v3 import PaperAccurateEnvV3
from agents.ippo_agent import IPPOAgent
from agents.explaboff_agent import ExplabOffAgent

print("="*60)
print("Training Pipeline Validation")
print("="*60)

# ── 1. Environment Check ──
print("\n[1/6] Checking environment...")
try:
    env = PaperAccurateEnvV3(3, 2, randomize_profile=True)
    print("  [OK] Environment created")
    
    # PettingZoo API check
    assert hasattr(env, 'agents'), "Missing agents attribute"
    assert hasattr(env, 'possible_agents'), "Missing possible_agents"
    print(f"  [OK] Agents: {env.possible_agents}")
    
    # Space check
    obs_space = env.observation_space('device_0')
    act_space = env.action_space('device_0')
    print(f"  [OK] Observation space: {obs_space.shape}")
    print(f"  [OK] Action space: {act_space.n}")
    
except Exception as e:
    print(f"  [FAIL] Environment failed: {e}")
    sys.exit(1)

# ── 2. Agent Creation ──
print("\n[2/6] Creating agents...")
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"  Device: {device}")

try:
    # IPPO
    ippo_agents = [IPPOAgent(i, env.obs_dim, 3, hidden_dim=64, device=device) 
                   for i in range(3)]
    print(f"  [OK] IPPO agents: {len(ippo_agents)}")
    
    # ExplabOff
    explab_agents = [ExplabOffAgent(i, env.obs_dim, 3, hidden_dim=64, 
                                    mi_mu=0.01, mi_nu=0.01, device=device)
                     for i in range(3)]
    print(f"  [OK] ExplabOff agents: {len(explab_agents)}")
    
except Exception as e:
    print(f"  [FAIL] Agent creation failed: {e}")
    sys.exit(1)

# ── 3. Training Loop (Mini) ──
print("\n[3/6] Running mini training (50 episodes)...")
try:
    agents = ippo_agents  # Test with IPPO
    for ep in range(50):
        obs, _ = env.reset(seed=ep)
        for agent in agents:
            agent.clear_trajectory()
        
        for step in range(10):
            actions = {}
            for i, agent in enumerate(agents):
                a, lp, v = agent.select_action(obs[f'device_{i}'])
                actions[f'device_{i}'] = a
                agent.store_transition(obs[f'device_{i}'], a, 0.0, v, lp, False)
            
            next_obs, rewards, terms, _, _ = env.step(actions)
            
            for i, agent in enumerate(agents):
                agent.trajectory['rewards'][-1] = rewards[f'device_{i}']
            
            obs = next_obs
            if any(terms.values()):
                break
        
        # Update every 10 episodes
        if (ep + 1) % 10 == 0:
            for agent in agents:
                agent.update(batch_size=32, num_epochs=2)
    
    print(f"  [OK] Training completed")
    
except Exception as e:
    print(f"  [FAIL] Training failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# ── 4. Evaluation ──
print("\n[4/6] Evaluating trained model...")
try:
    eval_costs = []
    for ep in range(10):
        obs, _ = env.reset(seed=1000 + ep)
        for step in range(10):
            actions = {}
            for i, agent in enumerate(agents):
                a, _, _ = agent.select_action(obs[f'device_{i}'])
                actions[f'device_{i}'] = a
            
            next_obs, rewards, terms, _, _ = env.step(actions)
            obs = next_obs
            if any(terms.values()):
                break
        
        m = env.get_episode_metrics()
        eval_costs.append(m['avg_cost'])
    
    avg_cost = np.mean(eval_costs)
    print(f"  [OK] Evaluation cost: {avg_cost:.4f}")
    
except Exception as e:
    print(f"  [FAIL] Evaluation failed: {e}")
    sys.exit(1)

# ── 5. Save/Load Check ──
print("\n[5/6] Testing save/load...")
try:
    save_dir = Path("results") / "pipeline_test"
    save_dir.mkdir(parents=True, exist_ok=True)
    
    # Save
    for i, agent in enumerate(agents):
        agent.save(str(save_dir / f"agent_{i}.pt"))
    print(f"  [OK] Models saved to {save_dir}")
    
    # Load
    new_agents = [IPPOAgent(i, env.obs_dim, 3, hidden_dim=64, device=device) 
                  for i in range(3)]
    for i, agent in enumerate(new_agents):
        agent.load(str(save_dir / f"agent_{i}.pt"))
    print(f"  [OK] Models loaded")
    
except Exception as e:
    print(f"  [FAIL] Save/load failed: {e}")
    sys.exit(1)

# ── 6. Report Generation ──
print("\n[6/6] Generating report...")
try:
    report = {
        'status': 'SUCCESS',
        'device': str(device),
        'env_config': {'M': 3, 'E': 2},
        'eval_cost': float(avg_cost),
        'eval_std': float(np.std(eval_costs)),
        'agents_trained': len(agents),
        'episodes_trained': 50,
        'save_load': 'OK'
    }
    
    import json
    with open(save_dir / "report.json", "w") as f:
        json.dump(report, f, indent=2)
    
    print(f"  [OK] Report saved to {save_dir / 'report.json'}")
    
except Exception as e:
    print(f"  [FAIL] Report generation failed: {e}")
    sys.exit(1)

print("\n" + "="*60)
print("[OK] Pipeline validation PASSED")
print("="*60)
print(f"\nSummary:")
print(f"  - Environment: OK (PettingZoo API)")
print(f"  - Agents: OK (IPPO + ExplabOff)")
print(f"  - Training: OK (50 episodes)")
print(f"  - Evaluation: OK (cost={avg_cost:.4f})")
print(f"  - Save/Load: OK")
print(f"  - Report: OK")
print(f"\nReady for full-scale training!")
