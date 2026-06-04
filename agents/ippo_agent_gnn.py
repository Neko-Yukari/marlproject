"""
IPPO Agent with GNN Backbone.

Uses shared GNN network for cross-config generalization.
Integrates with PettingZoo's ParallelEnv features:
- Uses env.agents for dynamic agent enumeration
- Supports variable number of agents across configs
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from typing import Dict, Tuple, List, Optional, Any

from .networks.gnn_backbone import GNNActorCritic


class IPPOAgentGNN:
    """IPPO agent with GNN backbone for cross-config support."""
    
    def __init__(self, agent_id: int, shared_network: GNNActorCritic,
                 learning_rate: float = 5e-5, gamma: float = 0.99,
                 gae_lambda: float = 0.95, clip_ratio: float = 0.2,
                 entropy_coeff: float = 0.01, value_coeff: float = 0.5,
                 max_grad_norm: float = 0.5,
                 device: torch.device = torch.device("cpu")):
        """
        Args:
            agent_id: Index of this agent in the MD list (matches env.agents ordering)
            shared_network: Shared GNNActorCritic instance (FIX C4: parameter sharing)
        """
        self.agent_id = agent_id
        self.network = shared_network  # FIX C4: shared reference
        self.device = device
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.clip_ratio = clip_ratio
        self.entropy_coeff = entropy_coeff
        self.value_coeff = value_coeff
        self.max_grad_norm = max_grad_norm
        
        # Optimizer operates on shared network parameters
        self.optimizer = optim.Adam(self.network.parameters(), lr=learning_rate)
        
        # Trajectory storage
        self.trajectory: Dict[str, List] = {
            "states": [], "actions": [], "rewards": [],
            "values": [], "log_probs": [], "dones": [],
            "graph_data": []  # Store graph data for each step
        }
    
    def select_action(self, obs: np.ndarray, graph_data: Tuple, 
                     action_mask: Optional[np.ndarray] = None) -> Tuple[int, float, float]:
        """
        Select action using GNN-processed graph data.
        
        Args:
            obs: Original observation (kept for compatibility, not used by GNN)
            graph_data: (node_features, node_types, edge_index)
            action_mask: [action_dim] bool array
        Returns:
            action, log_prob, value
        """
        node_features, node_types, edge_index = graph_data
        
        # Infer num_md and num_es from node_types
        num_md = int((node_types == 0).sum())
        num_es = int((node_types == 1).sum())
        
        # Run GNN once (FIX M7: shared computation across agents)
        with torch.no_grad():
            # Convert to tensors
            nf = node_features.to(self.device) if isinstance(node_features, torch.Tensor) else \
                 torch.from_numpy(node_features).float().to(self.device)
            nt = node_types.to(self.device) if isinstance(node_types, torch.Tensor) else \
                 torch.from_numpy(node_types).long().to(self.device)
            ei = edge_index.to(self.device) if isinstance(edge_index, torch.Tensor) else \
                 torch.from_numpy(edge_index).long().to(self.device)
            
            # Handle mask
            mask = None
            if action_mask is not None:
                mask = torch.from_numpy(action_mask).bool().unsqueeze(0).unsqueeze(0).to(self.device)
                # [1, 1, action_dim] -> will be broadcast to [1, num_md, max_action_dim]
                # Need to expand to [1, num_md, max_action_dim]
                max_dim = self.network.max_action_dim
                full_mask = torch.zeros(1, num_md, max_dim, dtype=torch.bool, device=self.device)
                actual_dim = num_es + 1
                full_mask[:, :, :actual_dim] = mask.expand(1, num_md, actual_dim)
                mask = full_mask
            
            policies, values = self.network(nf, nt, ei, num_md, num_es, action_mask=mask)
            
            # Extract this agent's policy (FIX C4: all agents share same forward pass)
            policy = policies[self.agent_id]  # [action_dim]
            value = values[self.agent_id]     # [1]
            
            # Handle numerical issues
            if torch.isnan(policy).any() or torch.isinf(policy).any():
                # Fallback to uniform
                policy = torch.zeros_like(policy)
                policy[:num_es+1] = 1.0 / (num_es + 1)
            
            dist = torch.distributions.Categorical(logits=policy)
            action = dist.sample()
            
            return action.item(), dist.log_prob(action).item(), value.item()
    
    def store_transition(self, state, action, reward, value, log_prob, done, graph_data=None):
        """Store a transition in the trajectory."""
        self.trajectory["states"].append(state)
        self.trajectory["actions"].append(action)
        self.trajectory["rewards"].append(reward)
        self.trajectory["values"].append(value)
        self.trajectory["log_probs"].append(log_prob)
        self.trajectory["dones"].append(done)
        if graph_data is not None:
            self.trajectory["graph_data"].append(graph_data)
    
    def clear_trajectory(self):
        """Clear stored trajectory."""
        for k in self.trajectory:
            self.trajectory[k].clear()
    
    def compute_gae(self, next_value: float = 0.0) -> Tuple[np.ndarray, np.ndarray]:
        """Compute Generalized Advantage Estimation."""
        rewards = np.array(self.trajectory["rewards"])
        values = np.array(self.trajectory["values"] + [next_value])
        dones = np.array(self.trajectory["dones"])
        
        deltas = rewards + self.gamma * values[1:] * (1 - dones) - values[:-1]
        adv = np.zeros_like(rewards)
        gae = 0.0
        
        for t in reversed(range(len(rewards))):
            gae = deltas[t] + self.gamma * self.gae_lambda * (1 - dones[t]) * gae
            adv[t] = gae
        
        ret = adv + values[:-1]
        adv = (adv - adv.mean()) / (adv.std() + 1e-8)
        
        return adv, ret
    
    def update(self, batch_size: int = 64, num_epochs: int = 4,
               graph_data_list: Optional[List] = None) -> Dict[str, float]:
        """Update policy using PPO clipped objective.
        
        Args:
            batch_size: Mini-batch size
            num_epochs: Number of optimization epochs
            graph_data_list: List of graph_data tuples for each timestep.
                           If None, uses stored trajectory graph_data.
        """
        if not self.trajectory["states"]:
            return {}
        
        # Use stored graph data if available
        if graph_data_list is None and len(self.trajectory.get("graph_data", [])) > 0:
            graph_data_list = self.trajectory["graph_data"]
        
        actions = np.array(self.trajectory["actions"])
        old_lp = np.array(self.trajectory["log_probs"])
        old_vals = np.array(self.trajectory["values"])
        adv, ret = self.compute_gae()
        
        a_t = torch.from_numpy(actions).long().to(self.device)
        o_t = torch.from_numpy(old_lp).float().to(self.device)
        v_t = torch.from_numpy(old_vals).float().to(self.device)
        d_t = torch.from_numpy(adv).float().to(self.device)
        r_t = torch.from_numpy(ret).float().to(self.device)
        
        losses = []
        n = len(actions)
        
        for _ in range(num_epochs):
            perm = np.random.permutation(n)
            for i in range(0, n, batch_size):
                idx = perm[i:i+batch_size]
                
                # If graph data provided, recompute log_probs and values with GNN
                if graph_data_list is not None and len(graph_data_list) > 0:
                    # Recompute log_probs and values for each step in batch
                    batch_graphs = [graph_data_list[i] for i in idx]
                    new_lps = []
                    new_vals = []
                    
                    for j, gd in enumerate(batch_graphs):
                        nf, nt, ei = gd
                        num_md = int((nt == 0).sum())
                        num_es = int((nt == 1).sum())
                        nf_t = nf.to(self.device) if isinstance(nf, torch.Tensor) else \
                               torch.from_numpy(nf).float().to(self.device)
                        nt_t = nt.to(self.device) if isinstance(nt, torch.Tensor) else \
                               torch.from_numpy(nt).long().to(self.device)
                        ei_t = ei.to(self.device) if isinstance(ei, torch.Tensor) else \
                               torch.from_numpy(ei).long().to(self.device)
                        
                        # Add batch dimension if needed
                        if nf_t.dim() == 2:
                            nf_t = nf_t.unsqueeze(0)
                            nt_t = nt_t.unsqueeze(0)
                        
                        policies, values = self.network(nf_t, nt_t, ei_t, num_md, num_es)
                        
                        # Extract this agent's policy and value
                        if policies.dim() == 3:  # [batch=1, num_md, action_dim]
                            policy = policies[0, self.agent_id]
                            value = values[0, self.agent_id]
                        else:  # [num_md, action_dim]
                            policy = policies[self.agent_id]
                            value = values[self.agent_id]
                        
                        dist = torch.distributions.Categorical(logits=policy)
                        action_idx = a_t[idx][j]  # Get the j-th action in batch
                        new_lps.append(dist.log_prob(action_idx))
                        new_vals.append(value.squeeze())
                    
                    new_lp = torch.stack(new_lps)
                    new_val = torch.stack(new_vals)
                else:
                    # Fallback: use stored values (approximate)
                    new_lp = o_t[idx]
                    new_val = v_t[idx]
                
                ratio = torch.exp(new_lp - o_t[idx])
                surr1 = ratio * d_t[idx]
                surr2 = torch.clamp(ratio, 1 - self.clip_ratio, 1 + self.clip_ratio) * d_t[idx]
                policy_loss = -torch.min(surr1, surr2).mean()
                
                value_loss = 0.5 * (r_t[idx] - new_val).pow(2).mean()
                
                # Compute entropy from the last dist (approximation)
                entropy = dist.entropy().mean() if 'dist' in dir() else torch.tensor(0.1)
                
                loss = policy_loss + self.value_coeff * value_loss - self.entropy_coeff * entropy
                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.network.parameters(), self.max_grad_norm)
                self.optimizer.step()
                losses.append(loss.item())
        
        return {"total_loss": float(np.mean(losses)) if losses else 0.0}
    
    def state_dict(self) -> dict:
        return {
            "network": self.network.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "agent_id": self.agent_id
        }
    
    def load_state_dict(self, d: dict):
        self.network.load_state_dict(d["network"])
        self.optimizer.load_state_dict(d["optimizer"])
    
    def save(self, path: str):
        torch.save(self.state_dict(), path)
    
    def load(self, path: str):
        self.load_state_dict(torch.load(path, map_location=self.device, weights_only=False))
