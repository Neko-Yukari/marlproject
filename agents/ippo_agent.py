"""
IPPO Agent — Independent PPO for MARL edge offloading (Stage 1).

Each MD learns independently with shared parameters, no inter-agent communication.
Reference: de Witt et al., ICLR 2020.
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from typing import Dict, Tuple, List, Optional

from .networks.actor_critic import ActorCriticNetwork


class IPPOAgent:
    """Independent PPO agent for a single mobile device."""

    def __init__(self, agent_id: int, state_dim: int, action_dim: int,
                 hidden_dim: int = 128, learning_rate: float = 5e-5,
                 gamma: float = 0.99, gae_lambda: float = 0.95,
                 clip_ratio: float = 0.2, entropy_coeff: float = 0.01,
                 value_coeff: float = 0.5, max_grad_norm: float = 0.5,
                 device: torch.device = torch.device("cpu")):
        self.agent_id = agent_id
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.device = device
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.clip_ratio = clip_ratio
        self.entropy_coeff = entropy_coeff
        self.value_coeff = value_coeff
        self.max_grad_norm = max_grad_norm

        self.network = ActorCriticNetwork(state_dim, action_dim, hidden_dim).to(device)
        self.optimizer = optim.Adam(self.network.parameters(), lr=learning_rate)
        self.trajectory: Dict[str, List] = {"states":[],"actions":[],"rewards":[],
                                             "values":[],"log_probs":[],"dones":[]}

    def select_action(self, state: np.ndarray, action_mask: Optional[np.ndarray] = None) -> Tuple[int, float, float]:
        with torch.no_grad():
            s = torch.from_numpy(state).float().unsqueeze(0).to(self.device)
            mask = torch.from_numpy(action_mask).float().unsqueeze(0).to(self.device) if action_mask is not None else None
            probs, value = self.network(s, mask)
            # Handle case where all valid actions have prob 0 (shouldn't happen with proper mask)
            if mask is not None and probs.sum() < 1e-6:
                # Fallback: uniform over valid actions
                probs = mask / mask.sum()
            dist = torch.distributions.Categorical(probs)
            a = dist.sample()
            return a.item(), dist.log_prob(a).item(), value.item()

    def get_action_probs(self, state: np.ndarray) -> np.ndarray:
        with torch.no_grad():
            s = torch.from_numpy(state).float().unsqueeze(0).to(self.device)
            probs, _ = self.network(s)
            return probs.cpu().numpy().squeeze(0)

    def store_transition(self, state, action, reward, value, log_prob, done):
        self.trajectory["states"].append(state)
        self.trajectory["actions"].append(action)
        self.trajectory["rewards"].append(reward)
        self.trajectory["values"].append(value)
        self.trajectory["log_probs"].append(log_prob)
        self.trajectory["dones"].append(done)

    def clear_trajectory(self):
        for k in self.trajectory: self.trajectory[k].clear()

    def compute_gae(self, next_value: float = 0.0) -> Tuple[np.ndarray, np.ndarray]:
        rewards = np.array(self.trajectory["rewards"])
        values  = np.array(self.trajectory["values"] + [next_value])
        dones   = np.array(self.trajectory["dones"])
        deltas  = rewards + self.gamma * values[1:] * (1 - dones) - values[:-1]
        adv = np.zeros_like(rewards)
        gae = 0.0
        for t in reversed(range(len(rewards))):
            gae = deltas[t] + self.gamma * self.gae_lambda * (1 - dones[t]) * gae
            adv[t] = gae
        ret = adv + values[:-1]
        adv = (adv - adv.mean()) / (adv.std() + 1e-8)
        return adv, ret

    def update(self, batch_size: int = 64, num_epochs: int = 4) -> Dict[str, float]:
        if not self.trajectory["states"]: return {}
        states  = np.array(self.trajectory["states"])
        actions = np.array(self.trajectory["actions"])
        old_lp  = np.array(self.trajectory["log_probs"])
        adv, ret = self.compute_gae()

        s_t = torch.from_numpy(states).float().to(self.device)
        a_t = torch.from_numpy(actions).long().to(self.device)
        o_t = torch.from_numpy(old_lp).float().to(self.device)
        d_t = torch.from_numpy(adv).float().to(self.device)
        r_t = torch.from_numpy(ret).float().to(self.device)

        losses = []
        n = len(states)
        for _ in range(num_epochs):
            perm = np.random.permutation(n)
            for i in range(0, n, batch_size):
                idx = perm[i:i+batch_size]
                probs, values = self.network(s_t[idx])
                dist = torch.distributions.Categorical(probs)
                new_lp  = dist.log_prob(a_t[idx])
                entropy = dist.entropy().mean()
                ratio   = torch.exp(new_lp - o_t[idx])
                surr1   = ratio * d_t[idx]
                surr2   = torch.clamp(ratio, 1 - self.clip_ratio, 1 + self.clip_ratio) * d_t[idx]
                policy_loss = -torch.min(surr1, surr2).mean()
                value_loss  = 0.5 * (r_t[idx] - values.squeeze(-1)).pow(2).mean()
                loss = policy_loss + self.value_coeff * value_loss - self.entropy_coeff * entropy
                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.network.parameters(), self.max_grad_norm)
                self.optimizer.step()
                losses.append(loss.item())
        return {"total_loss": float(np.mean(losses))}

    def state_dict(self) -> dict:
        return {"network": self.network.state_dict(),
                "optimizer": self.optimizer.state_dict(), "agent_id": self.agent_id}

    def load_state_dict(self, d: dict):
        self.network.load_state_dict(d["network"])
        self.optimizer.load_state_dict(d["optimizer"])

    def save(self, path: str): torch.save(self.state_dict(), path)
    def load(self, path: str): self.load_state_dict(torch.load(path, map_location=self.device, weights_only=False))
