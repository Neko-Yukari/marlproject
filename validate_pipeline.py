"""Pipeline Validation — End-to-end verification of unified architecture."""
import sys; sys.path.insert(0, '.')
import torch, numpy as np
from pathlib import Path

print("="*60)
print("MARL Edge Computing - Pipeline Validation")
print("="*60)

# ── 1. Environment Check ──
print("\n[1/6] Checking environment...")
try:
    from envs.paper_accurate_env import PaperAccurateEnvV3
    env = PaperAccurateEnvV3(3, 2, randomize_profile=True)
    print("  [OK] Environment created: 3MD-2ES")
    
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
    print(f"  [FAIL] {e}")
    sys.exit(1)

# ── 2. Policy Check ──
print("\n[2/6] Checking policy networks...")
try:
    from agents.policy_interface import PolicyNetwork
    from agents.standard_policy import StandardPolicy
    from agents.hyper_policy import HyperPolicy
    
    # Standard policy
    std_policy = StandardPolicy(state_dim=5, action_dim=3, hidden_dim=128, num_layers=2)
    print("  [OK] StandardPolicy created")
    
    # Hyper policy
    hyper_policy = HyperPolicy(max_obs_dim=7, max_action_dim=4, hidden_dim=256)
    hyper_policy.set_config(3, 2)
    print("  [OK] HyperPolicy created")
    
except Exception as e:
    print(f"  [FAIL] {e}")
    sys.exit(1)

# ── 3. Agent Check ──
print("\n[3/6] Checking PPO agent...")
try:
    from agents.ppo_agent import PPOAgent
    from agents.mi_plugin import MIPlugin
    
    # IPPO agent
    ippo_agent = PPOAgent(
        agent_id=0,
        policy_network=std_policy,
        mi_plugin=None,
        learning_rate=5e-5
    )
    print("  [OK] PPOAgent (IPPO mode) created")
    
    # ExplabOff agent with MI
    mi_plugin = MIPlugin(state_dim=5, action_dim=3, hidden_dim=128)
    explaboff_agent = PPOAgent(
        agent_id=0,
        policy_network=std_policy,
        mi_plugin=mi_plugin,
        learning_rate=5e-5
    )
    print("  [OK] PPOAgent (ExplabOff mode) created")
    
except Exception as e:
    print(f"  [FAIL] {e}")
    sys.exit(1)

# ── 4. Unified Trainer Check ──
print("\n[4/6] Checking unified trainer...")
try:
    from train_unified import UnifiedTrainer, build_config
    import argparse
    
    args = argparse.Namespace(
        config=None, network='standard', algorithm='ippo',
        md=3, es=2, episodes=10, seed=42, device='cpu',
        mode='train', save=None, load=None
    )
    config = build_config(args)
    trainer = UnifiedTrainer(config)
    print(f"  [OK] UnifiedTrainer created: {trainer.env.M}MD-{trainer.env.E}ES")
    print(f"  [OK] Policy: {type(trainer.agents['agents'][0].policy).__name__}")
    print(f"  [OK] MI Plugin: {trainer.agents['mi_plugin'] is None}")
    
except Exception as e:
    print(f"  [FAIL] {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# ── 5. Training Loop Check ──
print("\n[5/6] Checking training loop...")
try:
    obs = env.reset(seed=42)
    episode_reward = 0
    
    for step in range(env.slots):
        actions = {}
        for i, agent in enumerate(trainer.agents['agents']):
            agent_name = f'device_{i}'
            action = agent.select_action(obs[agent_name], deterministic=False)
            actions[agent_name] = action
        
        obs, rewards, dones, infos = env.step(actions)
        episode_reward += sum(rewards.values())
    
    print(f"  [OK] 1 episode completed, reward={episode_reward:.4f}")
    
except Exception as e:
    print(f"  [FAIL] {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# ── 6. Save/Load Check ──
print("\n[6/6] Checking save/load...")
try:
    import tempfile
    with tempfile.TemporaryDirectory() as tmpdir:
        trainer.save(tmpdir)
        print(f"  [OK] Model saved to {tmpdir}")
        
        trainer.load(tmpdir)
        print("  [OK] Model loaded successfully")
        
except Exception as e:
    print(f"  [FAIL] {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# ── Summary ──
print("\n" + "="*60)
print("ALL CHECKS PASSED!")
print("="*60)
print("\nProject is ready for training.")
print("\nQuick start:")
print("  python train_unified.py --network standard --algorithm ippo --md 3 --es 2")
print("  python train_unified.py --network hyper --algorithm explaboff --md 5 --es 2")
