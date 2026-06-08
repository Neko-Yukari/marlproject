"""
Hyper Policy - wraps HyperNetwork to implement PolicyNetwork interface.

Dynamically generates weights based on (M, E) configuration.
Single model serves all configurations.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional

from .hypernetwork import HyperNetwork
from .policy_interface import PolicyNetwork


class HyperPolicy(PolicyNetwork):
    """
    HyperNetwork-based policy.
    
    Uses HyperNetwork to dynamically generate weights for different (M, E) configs.
    Must call set_config(M, E) before forward pass.
    """
    
    def __init__(self, max_obs_dim: int = 7, max_action_dim: int = 4, hidden_dim: int = 256):
        super().__init__()
        self.hyper = HyperNetwork(
            obs_dim=max_obs_dim,
            max_action_dim=max_action_dim,
            hidden_dim=hidden_dim
        )
        self._M = None
        self._E = None
        self._max_obs_dim = max_obs_dim
        self._max_action_dim = max_action_dim
    
    def set_config(self, M: int, E: int):
        """Set current configuration. Must be called before forward."""
        self._M = M
        self._E = E
    
    def forward(self, obs: torch.Tensor, action_mask: Optional[torch.Tensor] = None, **kwargs) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass with dynamically generated weights.
        
        Args:
            obs: [batch, obs_dim] or [obs_dim]
            action_mask: optional mask
            **kwargs: can pass M, E here if not set via set_config
        
        Returns:
            action_probs: [batch, action_dim] or [action_dim]
            value: [batch, 1] or [1]
        """
        # Get M, E from kwargs or stored values
        M = kwargs.get('M', self._M)
        E = kwargs.get('E', self._E)
        
        if M is None or E is None:
            raise ValueError("Must call set_config(M, E) or pass M, E in kwargs")
        
        # HyperNetwork returns logits, not probs
        logits, value = self.hyper(obs, M, E, action_mask)
        
        # Handle single vs batch
        if logits.dim() == 1:
            logits = logits.unsqueeze(0)
            single = True
        else:
            single = False
        
        # Softmax to get probs
        probs = F.softmax(logits, dim=-1)
        
        if single:
            probs = probs.squeeze(0)
            value = value.squeeze(0)
        
        return probs, value
    
    def get_action_dim(self) -> int:
        if self._E is None:
            raise ValueError("Must call set_config first")
        return self._E + 1
    
    def get_obs_dim(self) -> int:
        # HyperNetwork can handle variable obs_dim via padding
        return self._max_obs_dim
