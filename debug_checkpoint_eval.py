"""
Debug checkpoint loading and cross-config evaluation for GNNPolicy.
Compare multiple checkpoints and final model.
"""
import sys; sys.path.insert(0, '.')
import torch
import numpy as np
from pathlib import Path

from envs.paper_accurate_env import PaperAccurateEnvV3
from agents.gnn_policy import GNNPolicy

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Paths
config_dir = Path('results/ippo_gnn_7md3es_20260617_033932')
ckpt_root = Path('results/ippo_gnn_7md3es_20260617_032200')

def load_policy(ckpt_path):
    import yaml
    with open(config_dir / 'config.yaml', 'r') as f:
        config = yaml.safe_load(f)
    
    algo_cfg = config['algorithm']
    policy = GNNPolicy(
        max_action_dim=algo_cfg.get('max_action_dim', 4),
        hidden_dim=algo_cfg.get('hidden_dim', 128),
        gnn_layers=algo_cfg.get('gnn_layers', 1),
        node_dim=algo_cfg.get('node_dim', 4),
        max_md=algo_cfg.get('max_md', 10)
    ).to(DEVICE)
    
    policy.load_state_dict(torch.load(ckpt_path, map_location=DEVICE))
    policy.eval()
    return policy

def eval_on_config(policy, num_md, num_es, num_eps=100):
    env = PaperAccurateEnvV3(num_devices=num_md, num_servers=num_es, randomize_profile=True, profile_noise=0.05)
    costs = []
    completions = []
    action_counts = {}
    total_actions = 0
    
    for ep in range(num_eps):
        obs, _ = env.reset(seed=10000 + ep)
        for step in range(10):
            policy.set_graph(env)
            actions = {}
            for agent_id in env.agents:
                md_idx = int(agent_id.split('_')[1])
                obs_t = torch.FloatTensor(obs[agent_id]).to(DEVICE)
                with torch.no_grad():
                    probs, _ = policy(obs_t.unsqueeze(0), agent_id=md_idx)
                action = int(torch.argmax(probs).item())
                actions[agent_id] = action
                
                if md_idx not in action_counts:
                    action_counts[md_idx] = {}
                action_counts[md_idx][action] = action_counts[md_idx].get(action, 0) + 1
                total_actions += 1
            
            obs, rewards, terms, truncs, infos = env.step(actions)
            if all(terms.values()) or all(truncs.values()):
                break
        
        metrics = env.get_episode_metrics()
        costs.append(metrics['avg_cost'])
        completions.append(metrics['completion_rate'])
    
    # Normalize action counts
    action_dist = {}
    for md_idx, counts in action_counts.items():
        action_dist[md_idx] = {a: c / total_actions * len(action_counts) for a, c in counts.items()}
    
    return np.mean(costs), np.std(costs), np.mean(completions), action_dist

# Checkpoints to test
checkpoints = [1000, 5000, 10000]

for ckpt_ep in checkpoints:
    ckpt_path = ckpt_root / f'checkpoint_ep{ckpt_ep}' / 'policy.pt'
    print(f"\n{'='*60}")
    print(f"Checkpoint ep{ckpt_ep}: {ckpt_path}")
    policy = load_policy(ckpt_path)
    
    for cfg_name, num_md, num_es in [('2ES-3MD', 3, 2), ('2ES-5MD', 5, 2), ('3ES-7MD', 7, 3)]:
        cost, std, comp, dist = eval_on_config(policy, num_md, num_es)
        print(f"  {cfg_name}: cost={cost:.4f}±{std:.4f}, comp={comp*100:.1f}%, actions={dist}")

# Final model
print(f"\n{'='*60}")
print(f"FINAL MODEL: {config_dir / 'policy.pt'}")
policy = load_policy(config_dir / 'policy.pt')
for cfg_name, num_md, num_es in [('2ES-3MD', 3, 2), ('2ES-5MD', 5, 2), ('3ES-7MD', 7, 3)]:
    cost, std, comp, dist = eval_on_config(policy, num_md, num_es)
    print(f"  {cfg_name}: cost={cost:.4f}±{std:.4f}, comp={comp*100:.1f}%, actions={dist}")
