"""Debug ES-aware GNN training path."""
import sys
sys.path.insert(0, 'E:\\MARL-IPPOAndMore')

import torch
import yaml
from envs.paper_accurate_env import PaperAccurateEnvV3
from agents.ppo_agent import PPOAgent
from agents.gnn_policy import GNNPolicy
from agents.policy_interface import PolicyNetwork

# Load config
with open('configs/ippo_gnn_7md3es.yaml', 'r') as f:
    config = yaml.safe_load(f)

device = torch.device('cuda')

# Create env and policy
env = PaperAccurateEnvV3(num_devices=7, num_servers=3, seed=42)
obs, _ = env.reset(seed=42)

policy = GNNPolicy(
    max_action_dim=4,
    hidden_dim=128,
    gnn_layers=1,
    node_dim=4,
    max_md=10,
)

agent = PPOAgent(
    agent_id=0,
    policy_network=policy,
    learning_rate=config['algorithm']['lr'],
    gamma=config['algorithm']['gamma'],
    gae_lambda=config['algorithm']['gae_lambda'],
    clip_ratio=config['algorithm']['clip_ratio'],
    entropy_coeff=config['algorithm']['entropy_coeff'],
    value_coeff=config['algorithm']['value_coeff'],
    max_grad_norm=config['algorithm']['max_grad_norm'],
    device=device,
)

print("=== Step 1: check forward path ===")
policy.set_graph(env)
obs_t = torch.tensor(obs['device_0'], dtype=torch.float32, device=device).unsqueeze(0)
probs, log_prob = policy.forward(obs_t, agent_id=0)
print(f"probs shape: {probs.shape}, sum: {probs.sum():.4f}")
print(f"probs: {probs.detach().cpu().numpy().round(4)}")

print("\n=== Step 2: check get_embedding returns tuple ===")
emb = policy.get_embedding(0)
print(f"embedding type: {type(emb)}")
if isinstance(emb, tuple):
    md_emb, es_emb = emb
    print(f"md_emb shape: {md_emb.shape}, es_emb shape: {es_emb.shape}")
else:
    print(f"embedding shape: {emb.shape}")

print("\n=== Step 3: simulate one episode and store transitions ===")
for step in range(10):
    policy.set_graph(env)
    actions = {}
    for i in range(env.M):
        obs_i = obs[f'device_{i}']
        action, log_prob, value = agent.select_action(obs_i, agent_id=i)
        actions[f'device_{i}'] = action
        emb = policy.get_embedding(i)
        if isinstance(emb, tuple):
            emb = tuple(e.detach().cpu() for e in emb)
        else:
            emb = emb.detach().cpu()
        agent._last_embedding = emb
    
    next_obs, rewards, dones, truncations, infos = env.step(actions)
    
    for i in range(env.M):
            agent.store_transition(
                obs[f'device_{i}'],
                actions[f'device_{i}'],
                rewards[f'device_{i}'],
                agent._last_value,
                agent._last_log_prob,
                dones[f'device_{i}'],
                embedding=getattr(agent, '_last_embedding', None)
            )
    obs = next_obs
    if any(dones.values()):
        break

print(f"Stored {len(agent.trajectory['states'])} transitions")

print("\n=== Step 4: check update path ===")
# Check if embeddings are tuples in buffer
first_emb = agent.trajectory["embeddings"][0]
print(f"First embedding type: {type(first_emb)}")
if isinstance(first_emb, tuple):
    print("Buffer stores tuple embeddings - correct")
else:
    print("Buffer stores non-tuple embeddings - BUG")

# Check if es_score_head gets gradients
for name, param in policy.named_parameters():
    if 'es_score_head' in name or 'local_score_head' in name:
        print(f"Parameter {name}: requires_grad={param.requires_grad}, shape={param.shape}")

# Run update
losses = agent.update()
print(f"\nUpdate losses: {losses}")

print("\n=== Step 5: check gradients ===")
for name, param in policy.named_parameters():
    if param.grad is not None and ('es_score_head' in name or 'local_score_head' in name):
        print(f"Gradient {name}: norm={param.grad.norm().item():.6f}")
    elif param.grad is None and ('es_score_head' in name or 'local_score_head' in name):
        print(f"NO GRADIENT for {name}")

print("\n=== Step 6: verify forward_from_embedding ===")
md_embs = torch.stack([agent.trajectory["embeddings"][i][0] for i in range(len(agent.trajectory["embeddings"]))[:5]]).to(device)
es_embs = torch.stack([agent.trajectory["embeddings"][i][1] for i in range(len(agent.trajectory["embeddings"]))[:5]]).to(device)
# es_embs shape is [E, hidden], need to expand for each md
es_embs = es_embs.unsqueeze(0).expand(5, -1, -1)  # [5, E, hidden]
print(f"md_embs shape: {md_embs.shape}, es_embs shape: {es_embs.shape}")
probs2, _ = policy.forward_from_embedding((md_embs, es_embs))
print(f"forward_from_embedding output shape: {probs2.shape}")
print(f"probs2[0]: {probs2[0].detach().cpu().numpy().round(4)}")

print("\nDebug complete.")
