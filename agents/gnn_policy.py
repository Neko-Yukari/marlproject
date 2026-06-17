"""
GNN Policy - Graph Neural Network policy network for variable (M, E) configurations.

Compatible with existing PPOAgent via PolicyNetwork interface.
Uses cached node embeddings from batch graph computation.

ES-aware design: instead of outputting discrete action indices, the policy head
scores each candidate (local execution or offloading to a specific ES) using
pairwise MD-ES features. This forces the network to learn ES-semantic reasoning
and generalize across different numbers of edge servers.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional, Dict, Union
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
    GNN Policy using cached node embeddings with ES-aware action scoring.
    
    Compatible with existing PPOAgent - forward() takes single agent's obs
    but uses cached graph embeddings from set_graph().
    
    ES-aware scoring:
        score(i, local) = local_score_head(md_embedding[i])
        score(i, j)     = es_score_head(concat(md_embedding[i], es_embedding[j]))
        policy(i)       = softmax([score(i, local), score(i, 1), ..., score(i, E)])
    
    This replaces the old index-based actor that learned "pick action index k"
    and overfit to the training configuration's action-space size.
    
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
        
        # ES-aware scoring heads
        # Local execution score (no ES involved)
        self.local_score_head = nn.Linear(hidden_dim, 1)
        # Per-ES score from pairwise MD-ES features
        self.es_score_head = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )
        
        # Value function
        self.critic = nn.Linear(hidden_dim, 1)
        
        # Fallback actor for update/eval when graph cache is unavailable.
        # This is index-based and NOT ES-aware; real training always uses cache.
        self._fallback_actor = nn.Linear(hidden_dim, max_action_dim)
        self._obs_proj = None  # Created lazily based on actual obs dim
        
        # Cached embeddings
        self._md_embeddings = None
        self._es_embeddings = None
        self._current_e = None
        self._agent_name_to_idx = {}
    
    def set_graph(self, env):
        """
        Build graph and compute all node embeddings.
        Call this once per step before agent actions.
        """
        # Get graph data from environment
        node_features, node_types, edge_index = env.get_graph_data()
        
        # Move to same device as model
        device = next(self.parameters()).device
        node_features = node_features.to(device)
        node_types = node_types.to(device)
        edge_index = edge_index.to(device)
        
        # Compute embeddings
        node_embeddings = self.gnn(node_features, node_types, edge_index)
        
        # Split MD and ES embeddings
        num_md = (node_types == 0).sum().item()
        self._md_embeddings = node_embeddings[:num_md]
        self._es_embeddings = node_embeddings[num_md:]
        
        # Store mapping
        self._current_e = env.E
        self._agent_name_to_idx = {
            agent_name: i for i, agent_name in enumerate(env.agents)
        }
    
    def get_embedding(self, agent_id: int = 0) -> Optional[Tuple[torch.Tensor, torch.Tensor]]:
        """Get cached embedding for agent_id as (md_emb, es_embs)."""
        if self._md_embeddings is None:
            return None
        if agent_id >= self._md_embeddings.size(0):
            agent_id = 0
        return (self._md_embeddings[agent_id], self._es_embeddings)
    
    def _compute_logits(self, md_emb: torch.Tensor,
                        es_embs: torch.Tensor) -> torch.Tensor:
        """
        Compute ES-aware action logits.
        
        Args:
            md_emb: [hidden_dim] or [B, hidden_dim]
            es_embs: [E, hidden_dim] or [B, E, hidden_dim]
        Returns:
            logits: [E+1] or [B, E+1]
        """
        single = md_emb.dim() == 1
        if single:
            md_emb = md_emb.unsqueeze(0)      # [1, H]
            es_embs = es_embs.unsqueeze(0)    # [1, E, H]
        
        B, E, H = es_embs.shape
        
        # Local execution score
        local_score = self.local_score_head(md_emb)  # [B, 1]
        
        # Per-ES score: pairwise MD-ES features
        md_expanded = md_emb.unsqueeze(1).expand(-1, E, -1)           # [B, E, H]
        pair_features = torch.cat([md_expanded, es_embs], dim=-1)     # [B, E, 2H]
        es_scores = self.es_score_head(pair_features)                 # [B, E, 1]
        
        # Concatenate local + ES scores
        logits = torch.cat([local_score, es_scores.squeeze(-1)], dim=-1)  # [B, E+1]
        
        if single:
            logits = logits.squeeze(0)  # [E+1]
        
        return logits
    
    def _apply_mask_and_softmax(self, logits: torch.Tensor,
                                action_mask: Optional[torch.Tensor],
                                actual_action_dim: int) -> torch.Tensor:
        """Apply action mask, softmax, and truncate to actual action dimension."""
        if action_mask is not None:
            if action_mask.dim() == 2 and logits.dim() == 1:
                action_mask = action_mask.squeeze(0)
            logits = logits.masked_fill(action_mask == 0, float('-inf'))
        
        probs = F.softmax(logits, dim=-1)
        
        # Truncate to actual action dimension
        if actual_action_dim < probs.size(-1):
            if probs.dim() == 1:
                probs = probs[:actual_action_dim]
                probs = probs / (probs.sum() + 1e-10)
            else:
                probs = probs[:, :actual_action_dim]
                probs = probs / (probs.sum(dim=-1, keepdim=True) + 1e-10)
        
        return probs
    
    def forward(self, obs: torch.Tensor, action_mask: Optional[torch.Tensor] = None,
                **kwargs) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass for single agent.
        
        Uses cached embeddings from set_graph() if available.
        Otherwise falls back to an index-based actor (not ES-aware).
        """
        agent_id = kwargs.get('agent_id', 0)
        
        if self._md_embeddings is not None:
            # Use cached ES-aware embeddings
            if agent_id >= self._md_embeddings.size(0):
                agent_id = 0
            md_emb = self._md_embeddings[agent_id]
            es_embs = self._es_embeddings
            logits = self._compute_logits(md_emb, es_embs)
            actual_action_dim = self._current_e + 1 if self._current_e is not None else self.max_action_dim
            embedding_for_value = md_emb
        else:
            # Fallback: project obs to hidden_dim then use index-based actor
            if obs.dim() == 1:
                obs_batch = obs.unsqueeze(0)
            else:
                obs_batch = obs
            if self._obs_proj is None:
                self._obs_proj = nn.Linear(obs_batch.shape[-1], self.hidden_dim).to(obs.device)
            projected = self._obs_proj(obs_batch)
            logits = self._fallback_actor(projected)
            if obs.dim() == 1 and logits.dim() == 2:
                logits = logits.squeeze(0)
            actual_action_dim = self.max_action_dim
            embedding_for_value = projected.squeeze(0) if projected.dim() == 2 else projected
        
        probs = self._apply_mask_and_softmax(logits, action_mask, actual_action_dim)
        value = self.critic(embedding_for_value)
        
        # Ensure value shape is consistent
        if value.dim() == 0:
            value = value.unsqueeze(0)
        
        return probs, value
    
    def forward_from_embedding(self, embedding: Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]],
                               action_mask: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass using pre-computed ES-aware embedding tuple (md_emb, es_embs).
        
        Args:
            embedding: tuple (md_emb, es_embs) where
                md_emb: [batch, hidden_dim] or [hidden_dim]
                es_embs: [batch, E, hidden_dim] or [E, hidden_dim]
            action_mask: optional action mask
        
        Returns:
            probs, value
        """
        if isinstance(embedding, tuple):
            md_emb, es_embs = embedding
        else:
            # Legacy: single tensor embedding - use old path with fallback actor
            logits = self._fallback_actor(embedding)
            probs = self._apply_mask_and_softmax(logits, action_mask, self.max_action_dim)
            value = self.critic(embedding)
            return probs, value
        
        logits = self._compute_logits(md_emb, es_embs)
        actual_action_dim = self._current_e + 1 if self._current_e is not None else self.max_action_dim
        
        probs = self._apply_mask_and_softmax(logits, action_mask, actual_action_dim)
        value = self.critic(md_emb)
        
        if value.dim() == 0:
            value = value.unsqueeze(0)
        
        return probs, value
    
    def get_action_dim(self) -> int:
        return self.max_action_dim
    
    def get_obs_dim(self) -> int:
        return 4  # Graph node feature dim (not used directly)
    
    def clear_cache(self):
        """Clear cached embeddings (call at end of episode)."""
        self._md_embeddings = None
        self._es_embeddings = None
        self._current_e = None
        self._agent_name_to_idx = {}
