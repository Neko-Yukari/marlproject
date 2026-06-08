"""
Standard Policy - wraps ActorCriticNetwork to implement PolicyNetwork interface.
"""
import torch
import torch.nn as nn
from typing import Tuple, Optional

from .networks.actor_critic import ActorCriticNetwork
from .policy_interface import PolicyNetwork


class StandardPolicy(PolicyNetwork):
    """
    Standard policy using ActorCriticNetwork.
    
    Each (M, E) configuration has its own StandardPolicy instance
    with fixed architecture.
    """
    
    def __init__(self, state_dim: int, action_dim: int, hidden_dim: int = 128, num_layers: int = 2):
        super().__init__()
        self._state_dim = state_dim
        self._action_dim = action_dim
        self.net = ActorCriticNetwork(state_dim, action_dim, hidden_dim, num_layers)
    
    def forward(self, obs: torch.Tensor, action_mask: Optional[torch.Tensor] = None, **kwargs) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass through ActorCriticNetwork.
        
        Args:
            obs: [batch, state_dim] or [state_dim]
            action_mask: optional mask
        
        Returns:
            action_probs: [batch, action_dim] or [action_dim]
            value: [batch, 1] or [1]
        """
        return self.net(obs, action_mask)
    
    def get_action_dim(self) -> int:
        return self._action_dim
    
    def get_obs_dim(self) -> int:
        return self._state_dim
