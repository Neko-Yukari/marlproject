"""
HyperNetwork-based Policy for Cross-Scale MEC Offloading.

A single model that generates policy weights for different (M, E) configurations.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class ConfigEncoder(nn.Module):
    """Encode (M, E) configuration into a latent vector."""
    
    def __init__(self, max_M=10, max_E=5, embed_dim=32):
        super().__init__()
        self.M_embed = nn.Embedding(max_M, embed_dim)
        self.E_embed = nn.Embedding(max_E, embed_dim)
        self.fc = nn.Sequential(
            nn.Linear(embed_dim * 2, 64),
            nn.ReLU(),
            nn.Linear(64, 64)
        )
    
    def forward(self, M, E, device=None):
        """
        Args:
            M: int or tensor - number of mobile devices
            E: int or tensor - number of edge servers
            device: target device
        Returns:
            config_vec: [batch, 64] or [64]
        """
        if device is None:
            device = next(self.parameters()).device
        
        if isinstance(M, int):
            M = torch.tensor([M], device=device)
        elif isinstance(M, torch.Tensor):
            M = M.to(device)
        
        if isinstance(E, int):
            E = torch.tensor([E], device=device)
        elif isinstance(E, torch.Tensor):
            E = E.to(device)
        
        m_vec = self.M_embed(M)
        e_vec = self.E_embed(E)
        combined = torch.cat([m_vec, e_vec], dim=-1)
        return self.fc(combined)


class HyperNetwork(nn.Module):
    """Generate policy network weights based on configuration."""
    
    def __init__(self, 
                 obs_dim=7,  # Max observation dimension
                 max_action_dim=4,  # Max action dimension (3ES + local)
                 hidden_dim=128,
                 config_embed_dim=64):
        super().__init__()
        self.obs_dim = obs_dim
        self.max_action_dim = max_action_dim
        self.hidden_dim = hidden_dim
        
        # Configuration encoder
        self.config_encoder = ConfigEncoder(max_M=10, max_E=5, embed_dim=32)
        
        # Hypernetwork: generates weights for the policy network
        # We generate: W1 (hidden_dim x obs_dim), b1 (hidden_dim)
        #              W2 (max_action_dim x hidden_dim), b2 (max_action_dim)
        self.weight_gen = nn.Sequential(
            nn.Linear(config_embed_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 512),
            nn.ReLU()
        )
        
        # Output heads for different weight components
        self.W1_head = nn.Linear(512, hidden_dim * obs_dim)
        self.b1_head = nn.Linear(512, hidden_dim)
        self.W2_head = nn.Linear(512, max_action_dim * hidden_dim)
        self.b2_head = nn.Linear(512, max_action_dim)
        
        # Learnable scaling factors for each configuration
        self.scale_M = nn.Parameter(torch.ones(1))
        self.scale_E = nn.Parameter(torch.ones(1))
    
    def generate_weights(self, M, E):
        """Generate policy weights for given configuration."""
        device = next(self.parameters()).device
        
        if isinstance(M, int):
            M = torch.tensor([M], device=device)
        elif isinstance(M, torch.Tensor):
            M = M.to(device)
            
        if isinstance(E, int):
            E = torch.tensor([E], device=device)
        elif isinstance(E, torch.Tensor):
            E = E.to(device)
        
        config_vec = self.config_encoder(M, E)  # [batch, 64] or [64]
        
        if config_vec.dim() == 1:
            config_vec = config_vec.unsqueeze(0)
        
        features = self.weight_gen(config_vec)  # [batch, 512]
        
        # Generate weights
        W1 = self.W1_head(features).view(-1, self.hidden_dim, self.obs_dim)
        b1 = self.b1_head(features).view(-1, self.hidden_dim)
        W2 = self.W2_head(features).view(-1, self.max_action_dim, self.hidden_dim)
        b2 = self.b2_head(features).view(-1, self.max_action_dim)
        
        return W1, b1, W2, b2
    
    def forward(self, obs, M, E, action_mask=None):
        """
        Forward pass with dynamically generated weights.
        
        Args:
            obs: [batch, obs_dim] or [obs_dim]
            M: int - number of devices
            E: int - number of servers
            action_mask: optional mask [batch, action_dim]
        Returns:
            logits: [batch, action_dim] policy logits
            value: [batch, 1] state value
        """
        # Generate weights
        W1, b1, W2, b2 = self.generate_weights(M, E)
        
        # Handle single vs batch
        if obs.dim() == 1:
            obs = obs.unsqueeze(0)
            single = True
        else:
            single = False
        
        batch_size = obs.size(0)
        actual_obs_dim = obs.size(1)
        
        # Pad obs to max_obs_dim if needed
        if actual_obs_dim < self.obs_dim:
            padding = torch.zeros(batch_size, self.obs_dim - actual_obs_dim, 
                                device=obs.device, dtype=obs.dtype)
            obs = torch.cat([obs, padding], dim=1)
        
        # Apply generated weights
        # Expand weights to batch size if needed
        if W1.size(0) == 1 and batch_size > 1:
            W1 = W1.expand(batch_size, -1, -1)
            b1 = b1.expand(batch_size, -1)
            W2 = W2.expand(batch_size, -1, -1)
            b2 = b2.expand(batch_size, -1)
        
        # Layer 1: obs -> hidden
        h = torch.bmm(W1, obs.unsqueeze(-1)).squeeze(-1) + b1
        h = F.relu(h)
        
        # Layer 2: hidden -> action logits
        logits = torch.bmm(W2, h.unsqueeze(-1)).squeeze(-1) + b2
        
        # Value head (shared features)
        value = torch.zeros(batch_size, 1, device=obs.device)
        
        # Apply action mask
        if action_mask is not None:
            logits = logits.masked_fill(~action_mask, float('-inf'))
        
        # Truncate to actual action dimension
        actual_action_dim = E + 1
        logits = logits[:, :actual_action_dim]
        
        if single:
            logits = logits.squeeze(0)
            value = value.squeeze(0)
        
        return logits, value


# NOTE: CrossScaleAgent removed.
# Use agents.ppo_agent.PPOAgent with agents.hyper_policy.HyperPolicy instead.
# Example:
#   policy = HyperPolicy(max_obs_dim=7, max_action_dim=4)
#   policy.set_config(M=3, E=2)
#   agent = PPOAgent(agent_id=0, policy_network=policy)
