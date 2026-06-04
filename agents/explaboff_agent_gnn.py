"""
ExplabOff Agent with GNN Backbone.

Combines MI-enhanced rewards with GNN-based cross-config generalization.
Uses GNN embeddings (md_context) as input to MI estimators (FIX C3).
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from typing import Dict, Tuple, List, Optional

from .networks.gnn_backbone import GNNActorCritic
from .networks.mi_estimator import InfoNCEEstimator, L1OutEstimator


class ExplabOffAgentGNN:
    """ExplabOff agent with GNN backbone for cross-config support."""
    
    def __init__(self, agent_id: int, shared_network: GNNActorCritic,
                 learning_rate: float = 5e-5, gamma: float = 0.99,
                 gae_lambda: float = 0.95, clip_ratio: float = 0.2,
                 entropy_coeff: float = 0.01, value_coeff: float = 0.5,
                 max_grad_norm: float = 0.5,
                 mi_mu: float = 3.5, mi_nu: float = 1.0,
                 device: torch.device = torch.device("cpu")):
        """
        Args:
            agent_id: Index of this agent in the MD list
            shared_network: Shared GNNActorCritic instance (FIX C4)
            mi_mu, mi_nu: MI reward weights (paper: 3.5 and 1.0)
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
        self.mi_mu = mi_mu
        self.mi_nu = mi_nu
        
        # FIX C3: MI estimators with correct input dims
        # Input: md_context [hidden_dim*2], action [max_action_dim=4]
        hidden_dim = shared_network.hidden_dim
        self.info_nce = InfoNCEEstimator(hidden_dim * 2, 4, hidden_dim).to(device)
        self.l1_out = L1OutEstimator(hidden_dim * 2, 4, hidden_dim).to(device)
        
        # Optimizer: shared network + MI estimators
        params = (list(self.network.parameters()) + 
                  list(self.info_nce.parameters()) + 
                  list(self.l1_out.parameters()))
        self.optimizer = optim.Adam(params, lr=learning_rate)
        
        # Trajectory storage
        self.trajectory: Dict[str, List] = {
            "states": [], "actions": [], "rewards": [],
            "values": [], "log_probs": [], "dones": [],
            "graph_data": []  # Store graph data for each step
        }
        
        # Dual buffers for MI (FIX C3)
        self.B_plus: List[Tuple] = []
        self.B_minus: List[Tuple] = []
        self._best_ep_reward = float("-inf")
        
        # FIX M7: Cache GNN embedding for MI computation
        self._cached_gnn_embedding = None
    
    def select_action(self, obs: np.ndarray, graph_data: Tuple,
                      action_mask: Optional[np.ndarray] = None) -> Tuple[int, float, float]:
        """Select action using GNN-processed graph data."""
        node_features, node_types, edge_index = graph_data
        
        num_md = int((node_types == 0).sum())
        num_es = int((node_types == 1).sum())
        
        with torch.no_grad():
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
                max_dim = self.network.max_action_dim
                full_mask = torch.zeros(1, num_md, max_dim, dtype=torch.bool, device=self.device)
                actual_dim = num_es + 1
                full_mask[:, :, :actual_dim] = mask.expand(1, num_md, actual_dim)
                mask = full_mask
            
            # FIX C3: Request embeddings alongside policies
            policies, values, embeddings = self.network(
                nf, nt, ei, num_md, num_es, 
                action_mask=mask, return_embeddings=True
            )
            
            # FIX M7: Cache GNN embedding (md_context) for MI computation
            self._cached_gnn_embedding = embeddings.detach()
            
            policy = policies[self.agent_id]
            value = values[self.agent_id]
            
            if torch.isnan(policy).any() or torch.isinf(policy).any():
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
        self._cached_gnn_embedding = None
    
    # ── MI-enhanced reward ──
    def compute_mi_reward(self, state: np.ndarray, action: int) -> float:
        """
        FIX M7: Use cached GNN embedding instead of recomputing.
        FIX C3: embedding dim = hidden_dim*2, matching InfoNCEEstimator input.
        """
        if self._cached_gnn_embedding is None:
            return 0.0
        
        try:
            with torch.no_grad():
                # Extract this agent's embedding from cached GNN output
                embedding = self._cached_gnn_embedding[self.agent_id]  # [hidden_dim*2]
                
                # Action one-hot (max_action_dim=4)
                action_onehot = torch.zeros(4, device=self.device)
                action_onehot[action] = 1.0
                
                i_nce = self.info_nce(embedding.unsqueeze(0), action_onehot.unsqueeze(0))
                i_l1 = self.l1_out(embedding.unsqueeze(0), action_onehot.unsqueeze(0))
                mi = self.mi_mu * i_nce - self.mi_nu * i_l1
                
                if torch.isnan(mi) or torch.isinf(mi):
                    return 0.0
                return float(mi.item()) * 0.01
        except Exception:
            return 0.0
    
    # ── Dual buffers ──
    def classify_episode(self, ep_reward: float, max_buf: int = 1000):
        """Classify episode into B+ or B- based on reward.
        
        Stores (state, action) pairs. If GNN embedding is available, use it; otherwise use raw state.
        """
        # Use GNN embedding if available, otherwise raw state
        if self._cached_gnn_embedding is not None:
            embedding = self._cached_gnn_embedding[self.agent_id].cpu().numpy()
            samples = [(embedding, a) for a in self.trajectory["actions"]]
        else:
            samples = list(zip(self.trajectory["states"], self.trajectory["actions"]))
        if ep_reward > self._best_ep_reward:
            self._best_ep_reward = ep_reward
            self.B_plus.extend(samples)
            if len(self.B_plus) > max_buf:
                self.B_plus = self.B_plus[-max_buf:]
        else:
            self.B_minus.extend(samples)
            if len(self.B_minus) > max_buf:
                self.B_minus = self.B_minus[-max_buf:]
    
    def update_mi_estimators(self, batch_size: int = 32):
        """Update MI estimators using B+ and B- buffers."""
        if len(self.B_plus) >= batch_size:
            idx = np.random.choice(len(self.B_plus), batch_size, replace=False)
            s_b = torch.from_numpy(np.array([self.B_plus[i][0] for i in idx])).float().to(self.device)
            a_b = torch.zeros(batch_size, 4).to(self.device)  # max_action_dim=4
            for j, i in enumerate(idx):
                a_b[j, self.B_plus[i][1]] = 1.0
            loss = self.info_nce.compute_loss(s_b, a_b)
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
        
        if len(self.B_minus) >= batch_size:
            idx = np.random.choice(len(self.B_minus), batch_size, replace=False)
            s_b = torch.from_numpy(np.array([self.B_minus[i][0] for i in idx])).float().to(self.device)
            a_b = torch.zeros(batch_size, 4).to(self.device)
            for j, i in enumerate(idx):
                a_b[j, self.B_minus[i][1]] = 1.0
            loss = self.l1_out.compute_loss(s_b, a_b)
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
    
    # ── GAE ──
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
    
    # ── PPO + MI update ──
    def update(self, batch_size: int = 64, num_epochs: int = 4,
               graph_data_list: Optional[List] = None) -> Dict[str, float]:
        """Update policy using PPO clipped objective with GNN."""
        if not self.trajectory["states"]:
            return {}
        
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
                
                # Recompute log_probs and values with GNN
                dist = None  # Initialize to avoid unbound variable warning
                if graph_data_list is not None and len(graph_data_list) > 0:
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
                        
                        if nf_t.dim() == 2:
                            nf_t = nf_t.unsqueeze(0)
                            nt_t = nt_t.unsqueeze(0)
                        
                        policies, values = self.network(nf_t, nt_t, ei_t, num_md, num_es)
                        
                        if policies.dim() == 3:
                            policy = policies[0, self.agent_id]
                            value = values[0, self.agent_id]
                        else:
                            policy = policies[self.agent_id]
                            value = values[self.agent_id]
                        
                        dist = torch.distributions.Categorical(logits=policy)
                        action_idx = a_t[idx][j]
                        new_lps.append(dist.log_prob(action_idx))
                        new_vals.append(value.squeeze())
                    
                    new_lp = torch.stack(new_lps)
                    new_val = torch.stack(new_vals)
                else:
                    new_lp = o_t[idx]
                    new_val = v_t[idx]
                
                ratio = torch.exp(new_lp - o_t[idx])
                surr1 = ratio * d_t[idx]
                surr2 = torch.clamp(ratio, 1 - self.clip_ratio, 1 + self.clip_ratio) * d_t[idx]
                policy_loss = -torch.min(surr1, surr2).mean()
                
                value_loss = 0.5 * (r_t[idx] - new_val).pow(2).mean()
                
                # Compute entropy from the last dist if available, else use default
                entropy = torch.tensor(0.1, device=self.device)
                if dist is not None:
                    entropy = dist.entropy().mean()
                
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
            "agent_id": self.agent_id,
            "mi_mu": self.mi_mu,
            "mi_nu": self.mi_nu
        }
    
    def load_state_dict(self, d: dict):
        self.network.load_state_dict(d["network"])
        self.optimizer.load_state_dict(d["optimizer"])
    
    def save(self, path: str):
        torch.save(self.state_dict(), path)
    
    def load(self, path: str):
        self.load_state_dict(torch.load(path, map_location=self.device, weights_only=False))
