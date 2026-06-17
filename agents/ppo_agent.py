"""
Unified PPO Agent - Orthogonal Architecture.

Works with any PolicyNetwork (StandardPolicy, HyperPolicy, etc.)
and optionally MIPlugin for ExplabOff behavior.

Replaces: IPPOAgent, ExplabOffAgent, CrossScaleAgent
"""
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from typing import Dict, Tuple, List, Optional

from .policy_interface import PolicyNetwork


class PPOAgent:
    """
    Unified PPO agent supporting any policy network and optional MI reward.
    
    Args:
        agent_id: unique agent identifier
        policy_network: PolicyNetwork instance (injected)
        mi_plugin: optional MIPlugin for ExplabOff behavior
        learning_rate: learning rate
        gamma: discount factor
        gae_lambda: GAE lambda
        clip_ratio: PPO clip ratio
        entropy_coeff: entropy bonus coefficient
        value_coeff: value loss coefficient
        max_grad_norm: gradient clipping norm
        device: torch device
    """
    
    def __init__(self, agent_id: int, policy_network: PolicyNetwork,
                 mi_plugin=None,
                 learning_rate: float = 5e-5,
                 gamma: float = 0.99, gae_lambda: float = 0.95,
                 clip_ratio: float = 0.2, entropy_coeff: float = 0.01,
                 value_coeff: float = 0.5, max_grad_norm: float = 0.5,
                 device: torch.device = torch.device("cpu")):
        self.agent_id = agent_id
        self.policy = policy_network.to(device)
        self.mi_plugin = mi_plugin
        self.device = device
        
        # PPO hyperparameters
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.clip_ratio = clip_ratio
        self.entropy_coeff = entropy_coeff
        self.value_coeff = value_coeff
        self.max_grad_norm = max_grad_norm
        
        # Build optimizer parameters
        opt_params = list(self.policy.parameters())
        if mi_plugin is not None:
            opt_params.extend(mi_plugin.get_parameters())
        self.optimizer = optim.Adam(opt_params, lr=learning_rate)
        
        # Trajectory buffer
        self.trajectory: Dict[str, List] = {
            "states": [], "actions": [], "rewards": [],
            "values": [], "log_probs": [], "dones": [],
            "embeddings": []  # For GNN: cached node embeddings
        }
        
        # Episode reward tracking (for MI buffer classification)
        self._ep_reward_sum = 0.0
        
        # Temporary storage for value and log_prob (used by training loop)
        self._last_value = 0.0
        self._last_log_prob = 0.0
    
    def select_action(self, state: np.ndarray, action_mask: Optional[np.ndarray] = None, agent_id: int = 0) -> Tuple[int, float, float]:
        """
        Select action using policy network.
        
        Args:
            state: observation array
            action_mask: optional action mask
            agent_id: agent index for GNN cached embeddings
        
        Returns:
            action: int
            log_prob: float
            value: float
        """
        with torch.no_grad():
            s = torch.from_numpy(state).float().unsqueeze(0).to(self.device)
            
            mask = None
            if action_mask is not None:
                mask = torch.from_numpy(action_mask).float().unsqueeze(0).to(self.device)
            
            probs, value = self.policy(s, mask, agent_id=agent_id)
            
            # Handle case where all valid actions have prob 0
            if mask is not None and probs.sum() < 1e-6:
                probs = mask / mask.sum()
            
            dist = torch.distributions.Categorical(probs)
            action = dist.sample()
            
            return action.item(), dist.log_prob(action).item(), value.item()
    
    def store_transition(self, state, action, reward, value, log_prob, done, embedding=None):
        """Store a transition in the trajectory buffer."""
        self.trajectory["states"].append(state)
        self.trajectory["actions"].append(action)
        self.trajectory["rewards"].append(reward)
        self.trajectory["values"].append(value)
        self.trajectory["log_probs"].append(log_prob)
        self.trajectory["dones"].append(done)
        if embedding is not None:
            self.trajectory["embeddings"].append(embedding)
        self._ep_reward_sum += reward
    
    def clear_trajectory(self):
        """Clear trajectory buffer."""
        for k in self.trajectory:
            self.trajectory[k].clear()
        self._ep_reward_sum = 0.0
    
    def compute_mi_reward(self, state: np.ndarray, action: int) -> float:
        """
        Compute MI reward if plugin is attached.
        
        Returns:
            mi_reward: float (0.0 if no plugin)
        """
        if self.mi_plugin is None:
            return 0.0
        return self.mi_plugin.compute_reward(state, action)
    
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
    
    def update(self, batch_size: int = 64, num_epochs: int = 4) -> Dict[str, float]:
        """
        PPO policy update.
        
        Returns:
            dict with training metrics
        """
        if not self.trajectory["states"]:
            return {}
        
        states = np.array(self.trajectory["states"])
        actions = np.array(self.trajectory["actions"])
        old_lp = np.array(self.trajectory["log_probs"])
        adv, ret = self.compute_gae()
        
        s_t = torch.from_numpy(states).float().to(self.device)
        a_t = torch.from_numpy(actions).long().to(self.device)
        o_t = torch.from_numpy(old_lp).float().to(self.device)
        d_t = torch.from_numpy(adv).float().to(self.device)
        r_t = torch.from_numpy(ret).float().to(self.device)
        
        # Check if we have cached embeddings (for GNN)
        has_embeddings = len(self.trajectory["embeddings"]) > 0
        emb_t = None
        if has_embeddings:
            embeddings = self.trajectory["embeddings"]
            # ES-aware GNN stores tuple (md_emb, es_embs)
            if isinstance(embeddings[0], tuple):
                md_embs = torch.stack([e[0] for e in embeddings]).to(self.device)
                es_embs = torch.stack([e[1] for e in embeddings]).to(self.device)
                emb_t = (md_embs, es_embs)
            else:
                emb_t = torch.stack(embeddings).to(self.device)
        
        def _index_embedding(embedding, idx):
            """Index into either tensor embedding or tuple embedding."""
            if isinstance(embedding, tuple):
                return tuple(e[idx] for e in embedding)
            return embedding[idx]
        
        losses = []
        n = len(states)
        
        for _ in range(num_epochs):
            perm = np.random.permutation(n)
            for i in range(0, n, batch_size):
                idx = perm[i:i+batch_size]
                
                # Forward pass
                if has_embeddings:
                    # Use cached embeddings directly
                    sub_emb = _index_embedding(emb_t, idx)
                    probs, values = self.policy.forward_from_embedding(sub_emb)
                else:
                    probs, values = self.policy(s_t[idx])
                
                # Handle NaN
                if torch.isnan(probs).any():
                    continue
                
                probs = torch.clamp(probs, 1e-8, 1.0)
                probs = probs / probs.sum(-1, keepdim=True)
                
                dist = torch.distributions.Categorical(probs)
                new_lp = dist.log_prob(a_t[idx])
                entropy = dist.entropy().mean()
                
                # PPO loss
                ratio = torch.exp(new_lp - o_t[idx])
                surr1 = ratio * d_t[idx]
                surr2 = torch.clamp(ratio, 1 - self.clip_ratio, 1 + self.clip_ratio) * d_t[idx]
                policy_loss = -torch.min(surr1, surr2).mean()
                
                value_loss = 0.5 * (r_t[idx] - values.squeeze(-1)).pow(2).mean()
                
                loss = policy_loss + self.value_coeff * value_loss - self.entropy_coeff * entropy
                
                # Backward
                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
                if self.mi_plugin is not None:
                    nn.utils.clip_grad_norm_(
                        self.mi_plugin.get_parameters(), self.max_grad_norm
                    )
                self.optimizer.step()
                
                losses.append(loss.item())
        
        # Update MI estimators if plugin exists
        if self.mi_plugin is not None:
            # Classify episode
            self.mi_plugin.classify_episode(
                self.trajectory["states"],
                self.trajectory["actions"],
                self._ep_reward_sum
            )
            # Update estimators
            self.mi_plugin.update_estimators(optimizer=self.optimizer)
        
        return {"total_loss": float(np.mean(losses)) if losses else 0.0}
    
    def state_dict(self) -> dict:
        """Get agent state dict."""
        state = {
            "policy": self.policy.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "agent_id": self.agent_id
        }
        if self.mi_plugin is not None:
            state["mi_plugin"] = self.mi_plugin.state_dict()
        return state
    
    def load_state_dict(self, d: dict):
        """Load agent state dict."""
        self.policy.load_state_dict(d["policy"])
        self.optimizer.load_state_dict(d["optimizer"])
        if self.mi_plugin is not None and "mi_plugin" in d:
            self.mi_plugin.load_state_dict(d["mi_plugin"])
    
    def save(self, path: str):
        """Save agent to file."""
        torch.save(self.state_dict(), path)
    
    def load(self, path: str):
        """Load agent from file."""
        ckpt = torch.load(path, map_location=self.device, weights_only=False)
        self.load_state_dict(ckpt)
