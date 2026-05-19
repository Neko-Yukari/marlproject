"""
Actor-Critic network architecture for PPO-based MARL edge offloading.
"""
import torch
import torch.nn as nn
from typing import Tuple


class ActorCriticNetwork(nn.Module):
    """
    Shared-backbone Actor-Critic with separate policy and value heads.

    Outputs:
        - action_probs [batch, action_dim] via softmax
        - value [batch, 1]
    """

    def __init__(self, state_dim: int, action_dim: int, hidden_dim: int = 128, num_layers: int = 2):
        super().__init__()
        layers = []
        layers.append(nn.Linear(state_dim, hidden_dim))
        layers.append(nn.ReLU())
        for _ in range(num_layers - 1):
            layers.append(nn.Linear(hidden_dim, hidden_dim))
            layers.append(nn.ReLU())
        self.backbone = nn.Sequential(*layers)
        self.actor_head = nn.Linear(hidden_dim, action_dim)
        self.critic_head = nn.Linear(hidden_dim, 1)

    def forward(self, state: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        features = self.backbone(state)
        logits = self.actor_head(features)
        action_probs = torch.softmax(logits, dim=-1)
        value = self.critic_head(features)
        return action_probs, value
