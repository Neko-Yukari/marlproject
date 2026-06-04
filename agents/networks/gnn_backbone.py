import torch
import torch.nn as nn
import torch.nn.functional as F


class GATLayer(nn.Module):
    """
    Graph Attention Layer with per-destination softmax.
    
    Fixes:
    - C2: Per-destination softmax instead of global
    - M3: Vectorized edge aggregation (no Python loop)
    - M4: LayerNorm + residual connections
    """
    
    def __init__(self, in_dim, out_dim, num_heads=4, dropout=0.1):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = out_dim // num_heads
        self.out_dim = out_dim
        
        self.W = nn.Linear(in_dim, out_dim)
        self.att_src = nn.Parameter(torch.Tensor(1, num_heads, self.head_dim))
        self.att_dst = nn.Parameter(torch.Tensor(1, num_heads, self.head_dim))
        
        # M4: LayerNorm
        self.norm = nn.LayerNorm(out_dim)
        # M4: Residual projection if dimensions differ
        self.residual = nn.Linear(in_dim, out_dim) if in_dim != out_dim else nn.Identity()
        
        self.dropout = nn.Dropout(dropout)
        
        nn.init.xavier_uniform_(self.att_src)
        nn.init.xavier_uniform_(self.att_dst)
    
    def forward(self, x, edge_index):
        """
        Args:
            x: [batch_size, num_nodes, in_dim] or [num_nodes, in_dim]
            edge_index: [2, num_edges] (same for all batches)
        Returns:
            out: same shape as x but with out_dim
        """
        # Handle both batched and unbatched
        if x.dim() == 2:
            x = x.unsqueeze(0)  # [1, num_nodes, in_dim]
            squeeze = True
        else:
            squeeze = False
        
        batch_size, num_nodes, in_dim = x.shape
        
        # Linear projection
        h = self.W(x)  # [batch, num_nodes, out_dim]
        h = h.view(batch_size, num_nodes, self.num_heads, self.head_dim)
        # [batch, num_nodes, num_heads, head_dim]
        
        # Compute attention scores
        src_nodes = edge_index[0]  # [num_edges]
        dst_nodes = edge_index[1]  # [num_edges]
        num_edges = edge_index.size(1)
        
        # Gather source and destination features
        h_src = h[:, src_nodes]  # [batch, num_edges, num_heads, head_dim]
        h_dst = h[:, dst_nodes]  # [batch, num_edges, num_heads, head_dim]
        
        # Attention scores
        src_att = (h_src * self.att_src).sum(dim=-1)  # [batch, num_edges, num_heads]
        dst_att = (h_dst * self.att_dst).sum(dim=-1)  # [batch, num_edges, num_heads]
        att_score = F.leaky_relu(src_att + dst_att, negative_slope=0.2)
        
        # FIX C2: Per-destination softmax
        # Create sparse matrix [batch, num_nodes, num_heads, max_neighbors]
        # Use scatter_softmax
        att_weight = self._scatter_softmax(att_score, dst_nodes, num_nodes)
        # [batch, num_edges, num_heads]
        
        # FIX M3: Vectorized message passing
        # weighted messages: [batch, num_edges, num_heads, head_dim]
        messages = att_weight.unsqueeze(-1) * h_src
        
        # Aggregate by destination
        out = torch.zeros(batch_size, num_nodes, self.num_heads, self.head_dim, 
                         device=x.device, dtype=x.dtype)
        dst_expanded = dst_nodes.view(1, -1, 1, 1).expand(batch_size, -1, self.num_heads, self.head_dim)
        out.scatter_add_(1, dst_expanded, messages)
        
        # Reshape
        out = out.view(batch_size, num_nodes, self.out_dim)
        
        # M4: Residual + LayerNorm
        out = self.norm(out + self.residual(x))
        out = self.dropout(out)
        
        if squeeze:
            out = out.squeeze(0)
        
        return out
    
    def _scatter_softmax(self, att_score, dst_nodes, num_nodes):
        """
        FIX C2: Per-destination softmax.
        
        Args:
            att_score: [batch, num_edges, num_heads]
            dst_nodes: [num_edges] - destination node indices
            num_nodes: int
        Returns:
            att_weight: [batch, num_edges, num_heads]
        """
        batch_size = att_score.size(0)
        num_edges = att_score.size(1)
        num_heads = att_score.size(2)
        
        # For numerical stability
        att_max = torch.full((batch_size, num_nodes, num_heads), float('-inf'), 
                            device=att_score.device, dtype=att_score.dtype)
        
        # Scatter max per destination
        dst_idx = dst_nodes.view(1, -1, 1).expand(batch_size, -1, num_heads)
        att_max.scatter_reduce_(1, dst_idx, att_score, reduce='amax', include_self=False)
        
        # Gather max for each edge
        max_per_edge = att_max.gather(1, dst_idx)  # [batch, num_edges, num_heads]
        
        # Compute exp(att - max)
        att_exp = torch.exp(att_score - max_per_edge)
        
        # Sum exp per destination
        att_sum = torch.zeros(batch_size, num_nodes, num_heads, 
                             device=att_score.device, dtype=att_score.dtype)
        att_sum.scatter_add_(1, dst_idx, att_exp)
        
        # Normalize
        sum_per_edge = att_sum.gather(1, dst_idx)  # [batch, num_edges, num_heads]
        att_weight = att_exp / (sum_per_edge + 1e-8)
        
        return att_weight


class GNNActorCritic(nn.Module):
    """
    GNN-based Actor-Critic with cross-config support.
    
    Fixes:
    - C1: _max_action_dim=4 for all configs
    - C4: Single shared network, agents use shared parameters
    - M1: Full batch support [batch, num_nodes, feat_dim]
    - M5: Node type embeddings
    - M6: Consistent action mask handling
    """
    
    def __init__(self, hidden_dim=128, num_gnn_layers=2, max_action_dim=4, 
                 node_types=2, dropout=0.1):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.max_action_dim = max_action_dim  # FIX C1: explicit max
        self.num_gnn_layers = num_gnn_layers
        
        # M5: Node type embedding (MD=0, ES=1)
        self.node_type_embed = nn.Embedding(node_types, 4)
        
        # Node encoder: 4-dim features -> hidden_dim
        self.node_encoder = nn.Linear(4, hidden_dim)
        
        # GNN layers
        self.gnn_layers = nn.ModuleList([
            GATLayer(hidden_dim, hidden_dim, num_heads=4, dropout=dropout)
            for _ in range(num_gnn_layers)
        ])
        
        # FIX C1: Fixed output dim for actor
        self.actor_head = nn.Linear(hidden_dim * 2, max_action_dim)
        self.critic_head = nn.Linear(hidden_dim * 2, 1)
        
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, node_features, node_types, edge_index, num_md, num_es, 
                action_mask=None, return_embeddings=False):
        """
        Args:
            node_features: [batch, num_md+num_es, 4] or [num_nodes, 4]
            node_types: [batch, num_nodes] or [num_nodes] - 0=MD, 1=ES
            edge_index: [2, num_edges]
            num_md: int
            num_es: int
            action_mask: Optional[Tensor] [batch, num_md, max_action_dim] bool
            return_embeddings: bool - if True, also return md_context for MI computation
        Returns:
            policies: [batch, num_md, max_action_dim] logits
            values: [batch, num_md, 1] values
            embeddings: [batch, num_md, hidden_dim*2] (only if return_embeddings=True)
        """
        # Handle unbatched input
        if node_features.dim() == 2:
            node_features = node_features.unsqueeze(0)
            if node_types.dim() == 1:
                node_types = node_types.unsqueeze(0)
            squeeze = True
        else:
            squeeze = False
        
        batch_size = node_features.size(0)
        num_nodes = node_features.size(1)
        
        # M5: Add node type embedding
        type_emb = self.node_type_embed(node_types)  # [batch, num_nodes, 4]
        x = node_features + type_emb  # [batch, num_nodes, 4]
        
        # Encode nodes
        x = self.node_encoder(x)  # [batch, num_nodes, hidden_dim]
        x = self.dropout(x)
        
        # Message passing
        for gnn in self.gnn_layers:
            x = F.relu(gnn(x, edge_index))
        
        # Split MD and ES
        md_repr = x[:, :num_md]  # [batch, num_md, hidden_dim]
        es_repr = x[:, num_md:]  # [batch, num_es, hidden_dim]
        
        # Pool ES info
        es_pooled = es_repr.mean(dim=1, keepdim=True)  # [batch, 1, hidden_dim]
        es_pooled = es_pooled.expand(-1, num_md, -1)  # [batch, num_md, hidden_dim]
        md_context = torch.cat([md_repr, es_pooled], dim=-1)  # [batch, num_md, hidden_dim*2]
        
        # Per-MD outputs (FIX C4: shared head for all MDs)
        policies = self.actor_head(md_context)  # [batch, num_md, max_action_dim]
        values = self.critic_head(md_context)   # [batch, num_md, 1]
        
        # M6: Apply action mask consistently
        if action_mask is not None:
            # action_mask: [batch, num_md, max_action_dim] bool
            # True = valid, False = invalid
            policies = policies.masked_fill(~action_mask, float('-inf'))
        
        # FIX M2: Truncate to actual action_dim (num_es + 1)
        actual_action_dim = num_es + 1
        assert actual_action_dim <= self.max_action_dim, \
            f"Action dim {actual_action_dim} exceeds max {self.max_action_dim}"
        if actual_action_dim < self.max_action_dim:
            policies = policies[:, :, :actual_action_dim]
        
        if squeeze:
            policies = policies.squeeze(0)
            values = values.squeeze(0)
            md_context = md_context.squeeze(0)
        
        if return_embeddings:
            return policies, values, md_context
        return policies, values


class GNNExplabOffNetwork(nn.Module):
    """
    GNN backbone for ExplabOff with MI computation support.
    Extends GNNActorCritic with embedding caching for MI rewards.
    """
    
    def __init__(self, hidden_dim=128, num_gnn_layers=2, max_action_dim=4,
                 node_types=2, dropout=0.1):
        super().__init__()
        self.gnn = GNNActorCritic(hidden_dim, num_gnn_layers, max_action_dim,
                                   node_types, dropout)
        self.hidden_dim = hidden_dim
        
    def forward(self, node_features, node_types, edge_index, num_md, num_es,
                action_mask=None, return_embeddings=False):
        """Delegate to base GNN."""
        return self.gnn(node_features, node_types, edge_index, num_md, num_es,
                        action_mask, return_embeddings)
    
    def get_embeddings(self, node_features, node_types, edge_index, num_md, num_es):
        """Get MD embeddings for MI computation (FIX M7: caching)."""
        _, _, embeddings = self.gnn(node_features, node_types, edge_index,
                                     num_md, num_es, return_embeddings=True)
        return embeddings  # [batch, num_md, hidden_dim*2]
