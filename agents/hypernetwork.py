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


class CrossScaleAgent:
    """Agent using hypernetwork for cross-scale policy."""
    
    def __init__(self, agent_id, hyper_net, device='cpu', lr=5e-5):
        self.agent_id = agent_id
        self.hyper_net = hyper_net
        self.device = device
        self.optimizer = torch.optim.Adam(hyper_net.parameters(), lr=lr)
        self.trajectory = {
            "states": [], "actions": [], "rewards": [],
            "values": [], "log_probs": [], "dones": []
        }
        self.gamma = 0.99
        self.gae_lambda = 0.95
        self.clip_ratio = 0.2
        self.entropy_coeff = 0.01
        self.value_coeff = 0.5
    
    def select_action(self, obs, M, E, action_mask=None):
        with torch.no_grad():
            obs_t = torch.from_numpy(obs).float().to(self.device)
            if action_mask is not None:
                mask_t = torch.from_numpy(action_mask).bool().to(self.device)
                # Pad to max_action_dim
                max_dim = self.hyper_net.max_action_dim
                if mask_t.size(0) < max_dim:
                    padding = torch.ones(max_dim - mask_t.size(0), dtype=torch.bool, device=self.device)
                    mask_t = torch.cat([mask_t, padding])
                mask_t = mask_t.unsqueeze(0)  # [1, max_action_dim]
            else:
                mask_t = None
            
            logits, value = self.hyper_net(obs_t, M, E, mask_t)
            
            if torch.isnan(logits).any():
                logits = torch.zeros_like(logits)
                logits[:E+1] = 1.0 / (E + 1)
            
            dist = torch.distributions.Categorical(logits=logits)
            action = dist.sample()
            
            return action.item(), dist.log_prob(action).item(), value.item()
    
    def store_transition(self, state, action, reward, value, log_prob, done):
        self.trajectory["states"].append(state)
        self.trajectory["actions"].append(action)
        self.trajectory["rewards"].append(reward)
        self.trajectory["values"].append(value)
        self.trajectory["log_probs"].append(log_prob)
        self.trajectory["dones"].append(done)
    
    def clear_trajectory(self):
        for k in self.trajectory:
            self.trajectory[k].clear()
    
    def compute_gae(self, next_value=0.0):
        rewards = np.array(self.trajectory["rewards"])
        values = np.array(self.trajectory["values"] + [next_value])
        dones = np.array(self.trajectory["dones"])
        
        deltas = rewards + self.gamma * values[1:] * (1 - dones) - values[:-1]
        adv = np.zeros_like(rewards)
        gae = 0.0
        
        for t in reversed(range(len(rewards))):
            gae = deltas[t] + self.gamma * self.gae_lambda * (1 - dones[t]) * gae
            adv[t] = gae
        
        ret = adv + values[:-1]
        adv = (adv - adv.mean()) / (adv.std() + 1e-8)
        
        return adv, ret
    
    def update(self, M, E, batch_size=64, num_epochs=4):
        if not self.trajectory["states"]:
            return {}
        
        states = np.array(self.trajectory["states"])
        actions = np.array(self.trajectory["actions"])
        old_lp = np.array(self.trajectory["log_probs"])
        old_vals = np.array(self.trajectory["values"])
        adv, ret = self.compute_gae()
        
        s_t = torch.from_numpy(states).float().to(self.device)
        a_t = torch.from_numpy(actions).long().to(self.device)
        o_t = torch.from_numpy(old_lp).float().to(self.device)
        d_t = torch.from_numpy(adv).float().to(self.device)
        r_t = torch.from_numpy(ret).float().to(self.device)
        
        losses = []
        n = len(actions)
        
        for _ in range(num_epochs):
            perm = np.random.permutation(n)
            for i in range(0, n, batch_size):
                idx = perm[i:i+batch_size]
                
                logits, values = self.hyper_net(s_t[idx], M, E)
                
                dist = torch.distributions.Categorical(logits=logits)
                new_lp = dist.log_prob(a_t[idx])
                
                ratio = torch.exp(new_lp - o_t[idx])
                surr1 = ratio * d_t[idx]
                surr2 = torch.clamp(ratio, 1 - self.clip_ratio, 1 + self.clip_ratio) * d_t[idx]
                policy_loss = -torch.min(surr1, surr2).mean()
                
                value_loss = 0.5 * (r_t[idx] - values.squeeze()).pow(2).mean()
                entropy = dist.entropy().mean()
                
                loss = policy_loss + self.value_coeff * value_loss - self.entropy_coeff * entropy
                self.optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.hyper_net.parameters(), 0.5)
                self.optimizer.step()
                losses.append(loss.item())
        
        return {"loss": np.mean(losses)}
