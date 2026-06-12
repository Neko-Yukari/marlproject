"""
GNN Policy - Graph Neural Network policy network for variable (M, E) configurations.

Compatible with existing PPOAgent via PolicyNetwork interface.
Uses cached node embeddings from batch graph computation.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional, Dict
import numpy as np

from .policy_interface import PolicyNetwork


class GNNLayer(nn.Module):
    """Simple Graph Convolution Layer with node type embeddings."""
    
    def __init__(self, in_dim: int, out_dim: int):
        super().__init__()
        self.linear = nn.Linear(in_dim, out_dim)
        self.norm = nn.LayerNorm(out_dim)
    
    def forward(self, x, edge_index):
        """
        Args:
            x: [N, in_dim]
            edge_index: [2, E]
        Returns:
            x: [N, out_dim]
        """
        # Message passing: aggregate neighbor features
        src, dst = edge_index
        
        # Gather source features
        src_features = x[src]  # [E, in_dim]
        
        # Aggregate to destination (scatter_add)
        out = torch.zeros(x.size(0), x.size(1), device=x.device, dtype=x.dtype)
        out.index_add_(0, dst, src_features)
        
        # Add self-connection
        out = out + x
        
        # Linear + Norm + ReLU
        out = self.linear(out)
        out = self.norm(out)
        out = F.relu(out)
        
        return out


class GNNBackbone(nn.Module):
    """GNN backbone for processing MEC graph.
    
    V4: 1-layer + per-agent learned embedding to prevent over-smoothing.
    MD nodes get unique identity vectors; ES nodes get zeros.
    """
    
    def __init__(self, node_dim: int, hidden_dim: int, num_layers: int = 1,
                 num_node_types: int = 2, max_md: int = 10):
        super().__init__()
        self.node_type_embed = nn.Embedding(num_node_types, 4)
        self.agent_embed = nn.Embedding(max_md, hidden_dim)  # V4: per-MD identity
        
        # Input projection (node_features + type_emb + agent_emb)
        self.input_proj = nn.Linear(node_dim + 4 + hidden_dim, hidden_dim)
        
        # GNN layers (default 1 to prevent over-smoothing)
        self.gnn_layers = nn.ModuleList([
            GNNLayer(hidden_dim, hidden_dim)
            for _ in range(num_layers)
        ])
        
        self.norm = nn.LayerNorm(hidden_dim)
    
    def forward(self, x, node_types, edge_index):
        """
        Args:
            x: [N, node_dim] node features
            node_types: [N] - 0=MD, 1=ES
            edge_index: [2, E]
        Returns:
            embeddings: [N, hidden_dim]
        """
        # Node type embedding (MD vs ES)
        type_emb = self.node_type_embed(node_types)  # [N, 4]
        
        # Per-agent identity embedding (MDs only, ES get zeros)
        M = (node_types == 0).sum().item()
        agent_idx = torch.arange(M, device=x.device)
        agent_emb_md = self.agent_embed(agent_idx)  # [M, hidden_dim]
        pad_es = torch.zeros(x.size(0) - M, agent_emb_md.size(1), device=x.device)
        agent_emb = torch.cat([agent_emb_md, pad_es], dim=0)  # [N, hidden_dim]
        
        # Concatenate all features
        x = torch.cat([x, type_emb, agent_emb], dim=-1)  # [N, node_dim+4+hidden]
        
        # Input projection
        x = self.input_proj(x)
        x = self.norm(x)
        x = F.relu(x)
        
        # GNN layers (1 layer: MD↔ES, no MD↔MD mixing)
        for layer in self.gnn_layers:
            x_new = layer(x, edge_index)
            x = x + x_new  # Residual
        
        return x


class GNNPolicy(PolicyNetwork):
    """
    GNN Policy using cached node embeddings.
    
    Compatible with existing PPOAgent - forward() takes single agent's obs
    but uses cached graph embeddings from set_graph().
    
    Usage in training loop:
        policy.set_graph(env)  # Call once per step
        action = agent.select_action(obs, agent_id=i)  # Each agent uses cached embedding
    """
    
    def __init__(self, max_action_dim: int = 4, hidden_dim: int = 128,
                 gnn_layers: int = 1, node_dim: int = 4, max_md: int = 10):
        super().__init__()
        self.max_action_dim = max_action_dim
        self.hidden_dim = hidden_dim
        
        # GNN backbone
        self.gnn = GNNBackbone(node_dim, hidden_dim, gnn_layers, max_md=max_md)
        
        # Output heads (shared across all nodes)
        self.actor = nn.Linear(hidden_dim, max_action_dim)
        self.critic = nn.Linear(hidden_dim, 1)
        
        # Cached embeddings
        self._node_embeddings = None
        self._current_e = None
        self._agent_name_to_idx = {}
        
        # Fallback MLP for update phase (when no graph cache)
        self._fallback_net = None
    
    def set_graph(self, env):
        """
        Build graph and compute all node embeddings.
        Call this once per step before agent actions.
        
        Args:
            env: PaperAccurateEnvV3 with get_graph_data() method
        """
        import torch
        
        # Get graph data from environment
        node_features, node_types, edge_index = env.get_graph_data()
        
        # Move to same device as model
        device = next(self.parameters()).device
        node_features = node_features.to(device)
        node_types = node_types.to(device)
        edge_index = edge_index.to(device)
        
        # Compute embeddings
        self._node_embeddings = self.gnn(node_features, node_types, edge_index)
        
        # Store mapping
        self._current_e = env.E
        self._agent_name_to_idx = {
            agent_name: i for i, agent_name in enumerate(env.agents)
        }
    
    def get_embedding(self, agent_id: int = 0) -> torch.Tensor:
        """Get cached embedding for agent_id. Returns None if not cached."""
        if self._node_embeddings is None:
            return None
        if agent_id >= self._node_embeddings.size(0):
            agent_id = 0
        return self._node_embeddings[agent_id]
    
    def forward(self, obs: torch.Tensor, action_mask: Optional[torch.Tensor] = None,
                **kwargs) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass for single agent.
        
        Uses cached embeddings from set_graph() if available.
        Otherwise uses obs directly (for PPO update phase).
        agent_id passed via kwargs['agent_id'].
        """
        agent_id = kwargs.get('agent_id', 0)
        
        if self._node_embeddings is not None:
            # Use cached GNN embedding
            if agent_id >= self._node_embeddings.size(0):
                agent_id = 0
            embedding = self._node_embeddings[agent_id]  # [hidden_dim]
        else:
            # Fallback: use obs directly (for PPO update phase)
            if obs.dim() == 1:
                embedding = obs
            else:
                # For batch, project obs to hidden_dim
                if not hasattr(self, '_obs_proj'):
                    self._obs_proj = nn.Linear(obs.shape[-1], self.hidden_dim).to(obs.device)
                embedding = self._obs_proj(obs)
        
        # Actor: action logits
        logits = self.actor(embedding)
        
        # Apply action mask
        if action_mask is not None:
            if action_mask.dim() == 2 and logits.dim() == 1:
                action_mask = action_mask.squeeze(0)
            logits = logits.masked_fill(action_mask == 0, float('-inf'))
        
        probs = F.softmax(logits, dim=-1)
        
        # Truncate to actual action dimension
        actual_action_dim = self._current_e + 1 if self._current_e is not None else self.max_action_dim
        if actual_action_dim < self.max_action_dim:
            if probs.dim() == 1:
                probs = probs[:actual_action_dim]
                probs = probs / (probs.sum() + 1e-10)
            else:
                probs = probs[:, :actual_action_dim]
                probs = probs / (probs.sum(dim=-1, keepdim=True) + 1e-10)
        
        value = self.critic(embedding)
        return probs, value
    
    def forward_from_embedding(self, embedding: torch.Tensor, 
                                action_mask: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass using pre-computed embedding (for PPO update with cached embeddings).
        
        Args:
            embedding: [batch, hidden_dim] or [hidden_dim]
            action_mask: optional action mask
        
        Returns:
            probs, value
        """
        # Actor: action logits
        logits = self.actor(embedding)
        
        # Apply action mask
        if action_mask is not None:
            if action_mask.dim() == 2 and logits.dim() == 1:
                action_mask = action_mask.squeeze(0)
            logits = logits.masked_fill(action_mask == 0, float('-inf'))
        
        probs = F.softmax(logits, dim=-1)
        
        # Truncate to actual action dimension
        actual_action_dim = self._current_e + 1 if self._current_e is not None else self.max_action_dim
        if actual_action_dim < self.max_action_dim:
            if probs.dim() == 1:
                probs = probs[:actual_action_dim]
                probs = probs / (probs.sum() + 1e-10)
            else:
                probs = probs[:, :actual_action_dim]
                probs = probs / (probs.sum(dim=-1, keepdim=True) + 1e-10)
        
        value = self.critic(embedding)
        return probs, value
    
    def get_action_dim(self) -> int:
        return self.max_action_dim
    
    def get_obs_dim(self) -> int:
        return 4  # Graph node feature dim (not used directly)
    
    def clear_cache(self):
        """Clear cached embeddings (call at end of episode)."""
        self._node_embeddings = None
        self._current_e = None
        self._agent_name_to_idx = {}
