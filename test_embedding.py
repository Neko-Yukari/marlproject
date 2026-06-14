import sys; sys.path.insert(0, '.')
from agents.gnn_policy import GNNPolicy
from agents.ppo_agent import PPOAgent
from envs.paper_accurate_env import PaperAccurateEnvV3
import torch

print('Testing embedding storage...')
env = PaperAccurateEnvV3(num_devices=3, num_servers=2)
obs, _ = env.reset(seed=42)

policy = GNNPolicy(max_action_dim=4, hidden_dim=64, gnn_layers=2)
agent = PPOAgent(agent_id=0, policy_network=policy, device=torch.device('cpu'))

policy.set_graph(env)
action, log_prob, value = agent.select_action(obs['device_0'], agent_id=0)

# Get embedding
emb = policy.get_embedding(0)
print(f'Embedding shape: {emb.shape}')

# Store with embedding
agent.store_transition(obs['device_0'], action, -1.0, value, log_prob, False, embedding=emb)
key = 'embeddings'
print(f'Trajectory embeddings: {len(agent.trajectory[key])}')

# Test forward_from_embedding
probs, val = policy.forward_from_embedding(emb.unsqueeze(0))
print(f'Forward from embedding: probs={probs.detach().numpy()}, value={val.item():.4f}')

print('Embedding test passed!')
