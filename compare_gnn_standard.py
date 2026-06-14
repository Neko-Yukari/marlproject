"""
Compare GNN vs Standard Policy training (10K episodes each).
Monitors actual learning by tracking cost trends.
"""
import sys; sys.path.insert(0, '.')
import time
import torch
import numpy as np
from envs.paper_accurate_env import PaperAccurateEnvV3
from agents import StandardPolicy, GNNPolicy, PPOAgent


def run_training(policy_type, num_episodes=10000, log_interval=500):
    """Run training and return history."""
    env = PaperAccurateEnvV3(num_devices=3, num_servers=2)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Create policy
    if policy_type == 'standard':
        policy = StandardPolicy(state_dim=5, action_dim=3, hidden_dim=128, num_layers=2)
    else:  # gnn
        policy = GNNPolicy(max_action_dim=4, hidden_dim=128, gnn_layers=2)
    
    policy = policy.to(device)
    
    # Create agents
    agents = [
        PPOAgent(agent_id=i, policy_network=policy, device=device,
                learning_rate=5e-5, entropy_coeff=0.01)
        for i in range(3)
    ]
    
    history = []
    start_time = time.time()
    best_cost = float('inf')
    
    print(f"\n{'='*60}")
    print(f"Training {policy_type.upper()} on 2ES-3MD | Device: {device}")
    print(f"{'='*60}\n")
    
    for ep in range(num_episodes):
        obs, _ = env.reset(seed=(42 + ep))
        
        # GNN: build graph once per step
        for step in range(10):
            if hasattr(policy, 'set_graph'):
                policy.set_graph(env)
            
            actions = {}
            for i, agent_name in enumerate(env.agents):
                action, log_prob, value = agents[i].select_action(
                    obs[agent_name], agent_id=i
                )
                actions[agent_name] = action
                agents[i]._last_value = value
                agents[i]._last_log_prob = log_prob
            
            next_obs, rewards, terms, truncs, infos = env.step(actions)
            
            for i, agent_name in enumerate(env.agents):
                agents[i].store_transition(
                    obs[agent_name], actions[agent_name], rewards[agent_name],
                    agents[i]._last_value, agents[i]._last_log_prob,
                    terms[agent_name] or truncs[agent_name]
                )
            
            if hasattr(policy, 'clear_cache'):
                policy.clear_cache()
            obs = next_obs
        
        # Update every 10 episodes
        if (ep + 1) % 10 == 0:
            for agent in agents:
                if len(agent.trajectory['states']) > 0:
                    agent.update(batch_size=64, num_epochs=4)
        
        # Log
        if ep % log_interval == 0 or ep == num_episodes - 1:
            metrics = env.get_episode_metrics()
            elapsed = time.time() - start_time
            
            record = {
                'episode': ep,
                'cost': float(metrics['avg_cost']),
                'completion': float(metrics['completion_rate']),
                'time': elapsed
            }
            history.append(record)
            
            status = "BEST" if metrics['avg_cost'] < best_cost else "    "
            if metrics['avg_cost'] < best_cost:
                best_cost = metrics['avg_cost']
            
            print(f"Ep {ep:5d} | Cost: {metrics['avg_cost']:.4f} | "
                  f"Comp: {metrics['completion_rate']:.1%} | "
                  f"Time: {elapsed:.1f}s | {status}")
    
    return history, best_cost


def analyze_learning(history, label):
    """Analyze if actual learning occurred."""
    costs = [h['cost'] for h in history]
    
    # Check trend
    first_half = np.mean(costs[:len(costs)//2])
    second_half = np.mean(costs[len(costs)//2:])
    improvement = first_half - second_half
    
    # Check if cost is trending down
    from scipy import stats
    x = np.arange(len(costs))
    slope, intercept, r_value, p_value, std_err = stats.linregress(x, costs)
    
    print(f"\n  {label} Learning Analysis:")
    print(f"    First half avg:  {first_half:.4f}")
    print(f"    Second half avg: {second_half:.4f}")
    print(f"    Improvement:     {improvement:.4f} ({improvement/first_half*100:.1f}%)")
    print(f"    Trend slope:     {slope:.6f} (negative = learning)")
    print(f"    R-squared:       {r_value**2:.4f}")
    print(f"    P-value:         {p_value:.4f} (<0.05 = significant)")
    
    if slope < -0.001 and p_value < 0.05:
        print(f"    [LEARNING CONFIRMED] Significant downward trend")
    elif slope < 0:
        print(f"    [WEAK LEARNING] Slight downward trend")
    else:
        print(f"    [NO LEARNING] Cost is not decreasing")
    
    return slope < 0 and p_value < 0.05


if __name__ == '__main__':
    print("=" * 70)
    print(" GNN vs Standard Policy - 10K Episode Comparison")
    print(" Monitoring actual learning...")
    print("=" * 70)
    
    # Run both
    std_history, std_best = run_training('standard', num_episodes=10000)
    gnn_history, gnn_best = run_training('gnn', num_episodes=10000)
    
    # Analyze
    print("\n" + "=" * 70)
    print(" RESULTS")
    print("=" * 70)
    
    std_learning = analyze_learning(std_history, "Standard")
    gnn_learning = analyze_learning(gnn_history, "GNN")
    
    print(f"\n  Standard Best Cost: {std_best:.4f}")
    print(f"  GNN Best Cost:      {gnn_best:.4f}")
    print(f"  Winner:             {'GNN' if gnn_best < std_best else 'Standard'}")
    
    # Save results
    import json
    with open('results/gnn_vs_standard_10k.json', 'w') as f:
        json.dump({
            'standard': {'history': std_history, 'best_cost': std_best, 'learning': std_learning},
            'gnn': {'history': gnn_history, 'best_cost': gnn_best, 'learning': gnn_learning}
        }, f, indent=2)
    print("\n  Results saved to results/gnn_vs_standard_10k.json")
