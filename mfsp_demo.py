"""MFSP (Mean Field Shared Policy) - O(1) parameters, O(log M) complexity.

This is SIMPLER than IPPO because:
- Only 1 network (not M networks)
- No per-agent state maintenance
- Batch processing of all devices
"""
import torch
import torch.nn as nn
import numpy as np

class MFSPNetwork(nn.Module):
    """Single network for ALL devices."""
    def __init__(self, obs_dim, action_dim, hidden_dim=256):
        super().__init__()
        # Local observation encoder
        self.local_encoder = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
        )
        # Mean field encoder (global state)
        self.global_encoder = nn.Sequential(
            nn.Linear(4, hidden_dim), nn.ReLU(),  # 4 global stats
        )
        # Combined policy
        self.policy = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, action_dim),
        )
        # Value function
        self.value = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )
    
    def forward(self, local_obs, global_state):
        """
        Args:
            local_obs: [batch, M, obs_dim] - each device's observation
            global_state: [batch, 4] - mean field statistics
        Returns:
            logits: [batch, M, action_dim]
            values: [batch, M, 1]
        """
        batch_size, M, _ = local_obs.shape
        
        # Encode local observations [batch, M, hidden]
        local_feat = self.local_encoder(local_obs)
        
        # Encode global state [batch, hidden]
        global_feat = self.global_encoder(global_state)
        
        # Broadcast global to match local [batch, M, hidden]
        global_feat = global_feat.unsqueeze(1).expand(-1, M, -1)
        
        # Combine [batch, M, hidden*2]
        combined = torch.cat([local_feat, global_feat], dim=-1)
        
        # Output for ALL devices simultaneously
        logits = self.policy(combined)      # [batch, M, action_dim]
        values = self.value(combined)       # [batch, M, 1]
        
        return logits, values


class MFSPEnvWrapper:
    """Compute mean field statistics from environment."""
    @staticmethod
    def compute_global_state(env_obs, env_info):
        """O(M) aggregation, can be parallelized to O(log M) with tree reduction."""
        # Extract task sizes from observations
        task_sizes = np.array([obs[0] for obs in env_obs.values()])
        
        # Global statistics (O(1) dimension regardless of M)
        global_state = np.array([
            np.mean(task_sizes),           # avg task size
            np.max(task_sizes),            # max task size
            np.std(task_sizes),            # variance
            env_info.get('es_load_avg', 0.0),  # avg ES load
        ], dtype=np.float32)
        
        return global_state


# Example usage:
if __name__ == "__main__":
    M = 100  # 100 devices
    obs_dim = 7
    action_dim = 4  # local + 3 ES
    
    # Create network (O(1) parameters!)
    net = MFSPNetwork(obs_dim, action_dim, hidden_dim=256)
    print(f"Parameters: {sum(p.numel() for p in net.parameters()):,}")
    print(f"Same as 1 IPPO agent, vs {M} IPPO agents = {M}x fewer params!")
    
    # Forward pass for ALL devices at once
    batch_obs = torch.randn(1, M, obs_dim)      # [1, 100, 7]
    global_state = torch.randn(1, 4)             # [1, 4]
    
    logits, values = net(batch_obs, global_state)
    print(f"Output shape: {logits.shape}")  # [1, 100, 4] - all actions!
    
    # Complexity: O(1) params, O(M) forward (but parallelized on GPU)
    # With tree aggregation: O(log M) total latency
