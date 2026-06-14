"""
Quick test script for GNN Policy integration.
Verifies GNNPolicy works with existing PPOAgent and training loop.
"""
import sys; sys.path.insert(0, '.')
import torch
import numpy as np
from envs.paper_accurate_env import PaperAccurateEnvV3
from agents.gnn_policy import GNNPolicy
from agents.ppo_agent import PPOAgent


def test_gnn_policy_basic():
    """Test basic GNNPolicy functionality."""
    print("=" * 60)
    print("Test 1: GNNPolicy Basic Functionality")
    print("=" * 60)
    
    # Create environment
    env = PaperAccurateEnvV3(num_devices=3, num_servers=2)
    obs, _ = env.reset(seed=42)
    
    # Create GNN policy
    policy = GNNPolicy(max_action_dim=4, hidden_dim=64, gnn_layers=2)
    
    # Build graph
    policy.set_graph(env)
    
    # Test forward for each agent
    for i, agent_name in enumerate(env.agents):
        agent_obs = obs[agent_name]
        obs_tensor = torch.from_numpy(agent_obs).float().unsqueeze(0)
        
        probs, value = policy(obs_tensor, agent_id=i)
        
        print(f"  Agent {i}: probs={probs.detach().numpy()}, value={value.item():.4f}")
        assert probs.shape == torch.Size([3]), f"Expected [3], got {probs.shape}"  # E+1=3
        assert torch.isclose(probs.sum(), torch.tensor(1.0), atol=1e-5), "Probs should sum to 1"
    
    policy.clear_cache()
    print("  [PASS] Basic functionality passed")
    return True


def test_gnn_with_ppo_agent():
    """Test GNNPolicy with PPOAgent."""
    print("\n" + "=" * 60)
    print("Test 2: GNNPolicy + PPOAgent Integration")
    print("=" * 60)
    
    env = PaperAccurateEnvV3(num_devices=3, num_servers=2)
    obs, _ = env.reset(seed=42)
    
    policy = GNNPolicy(max_action_dim=4, hidden_dim=64, gnn_layers=2)
    
    # Create agent
    agent = PPOAgent(
        agent_id=0,
        policy_network=policy,
        learning_rate=5e-5,
        device=torch.device('cpu')
    )
    
    # Set graph (normally done in training loop)
    policy.set_graph(env)
    
    # Select action
    action, log_prob, value = agent.select_action(obs['device_0'], agent_id=0)
    
    print(f"  Action: {action}, LogProb: {log_prob:.4f}, Value: {value:.4f}")
    assert 0 <= action < 3, f"Action {action} out of range"
    
    policy.clear_cache()
    print("  [PASS] PPOAgent integration passed")
    return True


def test_gnn_multi_step():
    """Test GNNPolicy across multiple steps."""
    print("\n" + "=" * 60)
    print("Test 3: Multi-Step GNN Execution")
    print("=" * 60)
    
    env = PaperAccurateEnvV3(num_devices=3, num_servers=2)
    obs, _ = env.reset(seed=42)
    
    policy = GNNPolicy(max_action_dim=4, hidden_dim=64, gnn_layers=2)
    agents = [
        PPOAgent(agent_id=i, policy_network=policy, device=torch.device('cpu'))
        for i in range(3)
    ]
    
    for step in range(5):
        # Set graph for this step
        policy.set_graph(env)
        
        # All agents select actions
        actions = {}
        for i, agent_name in enumerate(env.agents):
            action, _, _ = agents[i].select_action(obs[agent_name], agent_id=i)
            actions[agent_name] = action
        
        # Step environment
        obs, rewards, terms, truncs, infos = env.step(actions)
        
        # Clear cache
        policy.clear_cache()
        
        print(f"  Step {step}: actions={actions}, rewards={[f'{v:.3f}' for v in rewards.values()]}")
    
    print("  [PASS] Multi-step execution passed")
    return True


def test_gnn_training_loop():
    """Test mini training loop with GNN."""
    print("\n" + "=" * 60)
    print("Test 4: Mini Training Loop (100 episodes)")
    print("=" * 60)
    
    env = PaperAccurateEnvV3(num_devices=3, num_servers=2)
    policy = GNNPolicy(max_action_dim=4, hidden_dim=64, gnn_layers=2)
    
    agents = [
        PPOAgent(agent_id=i, policy_network=policy, device=torch.device('cpu'))
        for i in range(3)
    ]
    
    costs = []
    for ep in range(100):
        obs, _ = env.reset(seed=(42 + ep))
        ep_reward = 0.0
        
        for step in range(10):
            policy.set_graph(env)
            
            actions = {}
            for i, agent_name in enumerate(env.agents):
                action, log_prob, value = agents[i].select_action(obs[agent_name], agent_id=i)
                actions[agent_name] = action
                agents[i]._last_value = value
                agents[i]._last_log_prob = log_prob
            
            next_obs, rewards, terms, truncs, infos = env.step(actions)
            
            # Store transitions
            for i, agent_name in enumerate(env.agents):
                agents[i].store_transition(
                    obs[agent_name], actions[agent_name], rewards[agent_name],
                    agents[i]._last_value, agents[i]._last_log_prob,
                    terms[agent_name] or truncs[agent_name]
                )
            
            policy.clear_cache()
            obs = next_obs
        
        # Update every 10 episodes
        if (ep + 1) % 10 == 0:
            for agent in agents:
                if len(agent.trajectory['states']) > 0:
                    agent.update(batch_size=64, num_epochs=2)
        
        metrics = env.get_episode_metrics()
        costs.append(metrics['avg_cost'])
        
        if ep % 20 == 0:
            print(f"  Ep {ep}: cost={metrics['avg_cost']:.4f}, comp={metrics['completion_rate']:.1%}")
    
    avg_cost = np.mean(costs[-10:])
    print(f"  Final 10 episodes avg cost: {avg_cost:.4f}")
    print("  [PASS] Training loop passed")
    return True


def test_cross_config():
    """Test GNN on different (M, E) configurations."""
    print("\n" + "=" * 60)
    print("Test 5: Cross-Config Generalization")
    print("=" * 60)
    
    policy = GNNPolicy(max_action_dim=4, hidden_dim=64, gnn_layers=2)
    
    configs = [(3, 2), (5, 2), (7, 3)]
    
    for M, E in configs:
        env = PaperAccurateEnvV3(num_devices=M, num_servers=E)
        obs, _ = env.reset(seed=42)
        
        policy.set_graph(env)
        
        # Test first agent
        action, _, _ = PPOAgent(
            agent_id=0, policy_network=policy, device=torch.device('cpu')
        ).select_action(obs['device_0'], agent_id=0)
        
        print(f"  Config {M}MD-{E}ES: action={action}, valid={0 <= action < E+1}")
        
        policy.clear_cache()
    
    print("  [PASS] Cross-config test passed")
    return True


if __name__ == '__main__':
    print("\n" + "=" * 70)
    print(" GNN Policy Integration Test Suite")
    print("=" * 70 + "\n")
    
    try:
        test_gnn_policy_basic()
        test_gnn_with_ppo_agent()
        test_gnn_multi_step()
        test_gnn_training_loop()
        test_cross_config()
        
        print("\n" + "=" * 70)
        print(" ALL TESTS PASSED [PASS]")
        print("=" * 70)
    except Exception as e:
        print(f"\n  [FAIL] TEST FAILED: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
