"""
Policy Network Interface for Orthogonal Architecture.

All policy networks (Standard, HyperNetwork, etc.) must implement this interface
to be used interchangeably with PPOAgent.
"""
import torch
import torch.nn as nn
from typing import Tuple, Optional


class PolicyNetwork(nn.Module):
    """
    Abstract base class for all policy networks.
    
    Guarantees:
    - forward() returns (action_probs, value)
    - action_probs: [batch, action_dim] or [action_dim] (sum to 1)
    - value: [batch, 1] or [1]
    """
    
    def __init__(self):
        super().__init__()
    
    def forward(self, obs: torch.Tensor, action_mask: Optional[torch.Tensor] = None, **kwargs) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass.
        
        Args:
            obs: [batch, obs_dim] or [obs_dim]
            action_mask: optional [batch, action_dim] or [action_dim], 0=masked, 1=valid
            **kwargs: additional args (e.g., M, E for HyperNetwork)
        
        Returns:
            action_probs: [batch, action_dim] or [action_dim]
            value: [batch, 1] or [1]
        """
        raise NotImplementedError
    
    def get_action_dim(self) -> int:
        """Return the actual action dimension for current configuration."""
        raise NotImplementedError
    
    def get_obs_dim(self) -> int:
        """Return the actual observation dimension for current configuration."""
        raise NotImplementedError
