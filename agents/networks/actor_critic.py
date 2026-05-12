"""
Actor-Critic network architecture for PPO-based MARL.

Standard MLP backbone shared between actor (policy head)
and critic (value head). Used by IPPO, MAPPO, and ExplabOff agents.
"""

import torch
import torch.nn as nn
from typing import Tuple, Dict


class ActorCriticNetwork(nn.Module):
    """
    Shared-backbone Actor-Critic network.

    Architecture:
        Input (state_dim) → Hidden layers → Actor head (action_dim) + Critic head (1)

    Outputs:
        - Action probabilities (softmax over discrete action space)
        - State value estimate (scalar)
    """

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        hidden_dim: int = 128,
        num_layers: int = 2,
    ):
        super().__init__()
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.hidden_dim = hidden_dim

        # --- TO IMPLEMENT: Shared backbone ---
        layers = []
        layers.append(nn.Linear(state_dim, hidden_dim))
        layers.append(nn.ReLU())
        for _ in range(num_layers - 1):
            layers.append(nn.Linear(hidden_dim, hidden_dim))
            layers.append(nn.ReLU())
        self.backbone = nn.Sequential(*layers)

        # --- TO IMPLEMENT: Actor head (policy) ---
        self.actor_head: nn.Module = None  # Linear(hidden_dim, action_dim) + Softmax

        # --- TO IMPLEMENT: Critic head (value) ---
        self.critic_head: nn.Module = None  # Linear(hidden_dim, 1)

    def forward(
        self, state: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass.

        Args:
            state: [batch_size, state_dim]

        Returns:
            (action_probs [batch_size, action_dim], value [batch_size, 1])

        --- TO IMPLEMENT ---
        """
        raise NotImplementedError("ActorCriticNetwork.forward not implemented")


class ContinuousActorCriticNetwork(nn.Module):
    """
    Actor-Critic for continuous/partially-continuous action spaces.

    For Stage 3 mixed action: discrete (target ES) + continuous (offload ratio).
    """

    def __init__(
        self,
        state_dim: int,
        es_count: int,
        hidden_dim: int = 128,
    ):
        super().__init__()
        # --- TO IMPLEMENT: Separate heads for discrete and continuous actions ---
        pass

    def forward(self, state: torch.Tensor):
        raise NotImplementedError
