"""
HyperNetwork Variants for Ablation Study.

Different architectures and hyperparameters to find optimal cross-scale policy.
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


class HyperNetworkV1_LowLR(nn.Module):
    """Variant 1: Lower learning rate baseline (lr=1e-5)."""
    
    def __init__(self, obs_dim=7, max_action_dim=4, hidden_dim=128):
        super().__init__()
        self.obs_dim = obs_dim
        self.max_action_dim = max_action_dim
        self.hidden_dim = hidden_dim
        
        self.config_encoder = ConfigEncoder(max_M=10, max_E=5, embed_dim=32)
        
        self.weight_gen = nn.Sequential(
            nn.Linear(64, 256),
            nn.ReLU(),
            nn.Linear(256, 512),
            nn.ReLU()
        )
        
        self.W1_head = nn.Linear(512, hidden_dim * obs_dim)
        self.b1_head = nn.Linear(512, hidden_dim)
        self.W2_head = nn.Linear(512, max_action_dim * hidden_dim)
        self.b2_head = nn.Linear(512, max_action_dim)
    
    def generate_weights(self, M, E):
        device = next(self.parameters()).device
        
        if isinstance(M, int):
            M = torch.tensor([M], device=device)
        elif isinstance(M, torch.Tensor):
            M = M.to(device)
        
        if isinstance(E, int):
            E = torch.tensor([E], device=device)
        elif isinstance(E, torch.Tensor):
            E = E.to(device)
        
        config_vec = self.config_encoder(M, E)
        
        if config_vec.dim() == 1:
            config_vec = config_vec.unsqueeze(0)
        
        features = self.weight_gen(config_vec)
        
        W1 = self.W1_head(features).view(-1, self.hidden_dim, self.obs_dim)
        b1 = self.b1_head(features).view(-1, self.hidden_dim)
        W2 = self.W2_head(features).view(-1, self.max_action_dim, self.hidden_dim)
        b2 = self.b2_head(features).view(-1, self.max_action_dim)
        
        return W1, b1, W2, b2
    
    def forward(self, obs, M, E, action_mask=None):
        W1, b1, W2, b2 = self.generate_weights(M, E)
        
        if obs.dim() == 1:
            obs = obs.unsqueeze(0)
            single = True
        else:
            single = False
        
        batch_size = obs.size(0)
        actual_obs_dim = obs.size(1)
        
        if actual_obs_dim < self.obs_dim:
            padding = torch.zeros(batch_size, self.obs_dim - actual_obs_dim, 
                                device=obs.device, dtype=obs.dtype)
            obs = torch.cat([obs, padding], dim=1)
        
        if W1.size(0) == 1 and batch_size > 1:
            W1 = W1.expand(batch_size, -1, -1)
            b1 = b1.expand(batch_size, -1)
            W2 = W2.expand(batch_size, -1, -1)
            b2 = b2.expand(batch_size, -1)
        
        h = torch.bmm(W1, obs.unsqueeze(-1)).squeeze(-1) + b1
        h = F.relu(h)
        
        logits = torch.bmm(W2, h.unsqueeze(-1)).squeeze(-1) + b2
        value = torch.zeros(batch_size, 1, device=obs.device)
        
        if action_mask is not None:
            logits = logits.masked_fill(~action_mask, float('-inf'))
        
        actual_action_dim = E + 1
        logits = logits[:, :actual_action_dim]
        
        if single:
            logits = logits.squeeze(0)
            value = value.squeeze(0)
        
        return logits, value


class HyperNetworkV2_Large(nn.Module):
    """Variant 2: Larger hidden dimension (256) - FIXED with Value Head and Weight Cache."""
    
    def __init__(self, obs_dim=7, max_action_dim=4, hidden_dim=256):
        super().__init__()
        self.obs_dim = obs_dim
        self.max_action_dim = max_action_dim
        self.hidden_dim = hidden_dim
        
        self.config_encoder = ConfigEncoder(max_M=10, max_E=5, embed_dim=32)
        
        self.weight_gen = nn.Sequential(
            nn.Linear(64, 512),
            nn.ReLU(),
            nn.Linear(512, 1024),
            nn.ReLU()
        )
        
        self.W1_head = nn.Linear(1024, hidden_dim * obs_dim)
        self.b1_head = nn.Linear(1024, hidden_dim)
        self.W2_head = nn.Linear(1024, max_action_dim * hidden_dim)
        self.b2_head = nn.Linear(1024, max_action_dim)
        
        # FIXED: Add Value Head (from config embedding, not obs)
        self.value_head = nn.Sequential(
            nn.Linear(64, 128),
            nn.ReLU(),
            nn.Linear(128, 1)
        )
        
        # FIXED: Weight cache for stability
        self.weight_cache = {}
        self.cache_update_counter = 0
    
    def generate_weights(self, M, E):
        device = next(self.parameters()).device
        
        if isinstance(M, int):
            M = torch.tensor([M], device=device)
        elif isinstance(M, torch.Tensor):
            M = M.to(device)
        
        if isinstance(E, int):
            E = torch.tensor([E], device=device)
        elif isinstance(E, torch.Tensor):
            E = E.to(device)
        
        config_vec = self.config_encoder(M, E)
        
        if config_vec.dim() == 1:
            config_vec = config_vec.unsqueeze(0)
        
        features = self.weight_gen(config_vec)
        
        W1 = self.W1_head(features).view(-1, self.hidden_dim, self.obs_dim)
        b1 = self.b1_head(features).view(-1, self.hidden_dim)
        W2 = self.W2_head(features).view(-1, self.max_action_dim, self.hidden_dim)
        b2 = self.b2_head(features).view(-1, self.max_action_dim)
        
        return W1, b1, W2, b2
    
    def forward(self, obs, M, E, action_mask=None):
        W1, b1, W2, b2 = self.generate_weights(M, E)
        
        if obs.dim() == 1:
            obs = obs.unsqueeze(0)
            single = True
        else:
            single = False
        
        batch_size = obs.size(0)
        actual_obs_dim = obs.size(1)
        
        if actual_obs_dim < self.obs_dim:
            padding = torch.zeros(batch_size, self.obs_dim - actual_obs_dim, 
                                device=obs.device, dtype=obs.dtype)
            obs = torch.cat([obs, padding], dim=1)
        
        if W1.size(0) == 1 and batch_size > 1:
            W1 = W1.expand(batch_size, -1, -1)
            b1 = b1.expand(batch_size, -1)
            W2 = W2.expand(batch_size, -1, -1)
            b2 = b2.expand(batch_size, -1)
        
        h = torch.bmm(W1, obs.unsqueeze(-1)).squeeze(-1) + b1
        h = F.relu(h)
        
        logits = torch.bmm(W2, h.unsqueeze(-1)).squeeze(-1) + b2
        
        # FIXED: Use value head instead of zeros
        config_vec = self.config_encoder(
            torch.tensor([M], device=obs.device) if isinstance(M, int) else M.to(obs.device),
            torch.tensor([E], device=obs.device) if isinstance(E, int) else E.to(obs.device)
        )
        if config_vec.dim() == 1:
            config_vec = config_vec.unsqueeze(0)
        value = self.value_head(config_vec)
        
        if action_mask is not None:
            logits = logits.masked_fill(~action_mask, float('-inf'))
        
        actual_action_dim = E + 1
        logits = logits[:, :actual_action_dim]
        
        if single:
            logits = logits.squeeze(0)
            value = value.squeeze(0)
        
        return logits, value


class HyperNetworkV3_LayerNorm(nn.Module):
    """Variant 3: Add LayerNorm for stability."""
    
    def __init__(self, obs_dim=7, max_action_dim=4, hidden_dim=128):
        super().__init__()
        self.obs_dim = obs_dim
        self.max_action_dim = max_action_dim
        self.hidden_dim = hidden_dim
        
        self.config_encoder = ConfigEncoder(max_M=10, max_E=5, embed_dim=32)
        
        self.weight_gen = nn.Sequential(
            nn.Linear(64, 256),
            nn.LayerNorm(256),
            nn.ReLU(),
            nn.Linear(256, 512),
            nn.LayerNorm(512),
            nn.ReLU()
        )
        
        self.W1_head = nn.Linear(512, hidden_dim * obs_dim)
        self.b1_head = nn.Linear(512, hidden_dim)
        self.W2_head = nn.Linear(512, max_action_dim * hidden_dim)
        self.b2_head = nn.Linear(512, max_action_dim)
    
    def generate_weights(self, M, E):
        device = next(self.parameters()).device
        
        if isinstance(M, int):
            M = torch.tensor([M], device=device)
        elif isinstance(M, torch.Tensor):
            M = M.to(device)
        
        if isinstance(E, int):
            E = torch.tensor([E], device=device)
        elif isinstance(E, torch.Tensor):
            E = E.to(device)
        
        config_vec = self.config_encoder(M, E)
        
        if config_vec.dim() == 1:
            config_vec = config_vec.unsqueeze(0)
        
        features = self.weight_gen(config_vec)
        
        W1 = self.W1_head(features).view(-1, self.hidden_dim, self.obs_dim)
        b1 = self.b1_head(features).view(-1, self.hidden_dim)
        W2 = self.W2_head(features).view(-1, self.max_action_dim, self.hidden_dim)
        b2 = self.b2_head(features).view(-1, self.max_action_dim)
        
        return W1, b1, W2, b2
    
    def forward(self, obs, M, E, action_mask=None):
        W1, b1, W2, b2 = self.generate_weights(M, E)
        
        if obs.dim() == 1:
            obs = obs.unsqueeze(0)
            single = True
        else:
            single = False
        
        batch_size = obs.size(0)
        actual_obs_dim = obs.size(1)
        
        if actual_obs_dim < self.obs_dim:
            padding = torch.zeros(batch_size, self.obs_dim - actual_obs_dim, 
                                device=obs.device, dtype=obs.dtype)
            obs = torch.cat([obs, padding], dim=1)
        
        if W1.size(0) == 1 and batch_size > 1:
            W1 = W1.expand(batch_size, -1, -1)
            b1 = b1.expand(batch_size, -1)
            W2 = W2.expand(batch_size, -1, -1)
            b2 = b2.expand(batch_size, -1)
        
        h = torch.bmm(W1, obs.unsqueeze(-1)).squeeze(-1) + b1
        h = F.layer_norm(h, h.shape[1:])
        h = F.relu(h)
        
        logits = torch.bmm(W2, h.unsqueeze(-1)).squeeze(-1) + b2
        value = torch.zeros(batch_size, 1, device=obs.device)
        
        if action_mask is not None:
            logits = logits.masked_fill(~action_mask, float('-inf'))
        
        actual_action_dim = E + 1
        logits = logits[:, :actual_action_dim]
        
        if single:
            logits = logits.squeeze(0)
            value = value.squeeze(0)
        
        return logits, value


class HyperNetworkV4_Curriculum(nn.Module):
    """Variant 4: Curriculum learning - same as V1 but training strategy differs."""
    
    def __init__(self, obs_dim=7, max_action_dim=4, hidden_dim=128):
        super().__init__()
        self.obs_dim = obs_dim
        self.max_action_dim = max_action_dim
        self.hidden_dim = hidden_dim
        
        self.config_encoder = ConfigEncoder(max_M=10, max_E=5, embed_dim=32)
        
        self.weight_gen = nn.Sequential(
            nn.Linear(64, 256),
            nn.ReLU(),
            nn.Linear(256, 512),
            nn.ReLU()
        )
        
        self.W1_head = nn.Linear(512, hidden_dim * obs_dim)
        self.b1_head = nn.Linear(512, hidden_dim)
        self.W2_head = nn.Linear(512, max_action_dim * hidden_dim)
        self.b2_head = nn.Linear(512, max_action_dim)
    
    def generate_weights(self, M, E):
        device = next(self.parameters()).device
        
        if isinstance(M, int):
            M = torch.tensor([M], device=device)
        elif isinstance(M, torch.Tensor):
            M = M.to(device)
        
        if isinstance(E, int):
            E = torch.tensor([E], device=device)
        elif isinstance(E, torch.Tensor):
            E = E.to(device)
        
        config_vec = self.config_encoder(M, E)
        
        if config_vec.dim() == 1:
            config_vec = config_vec.unsqueeze(0)
        
        features = self.weight_gen(config_vec)
        
        W1 = self.W1_head(features).view(-1, self.hidden_dim, self.obs_dim)
        b1 = self.b1_head(features).view(-1, self.hidden_dim)
        W2 = self.W2_head(features).view(-1, self.max_action_dim, self.hidden_dim)
        b2 = self.b2_head(features).view(-1, self.max_action_dim)
        
        return W1, b1, W2, b2
    
    def forward(self, obs, M, E, action_mask=None):
        W1, b1, W2, b2 = self.generate_weights(M, E)
        
        if obs.dim() == 1:
            obs = obs.unsqueeze(0)
            single = True
        else:
            single = False
        
        batch_size = obs.size(0)
        actual_obs_dim = obs.size(1)
        
        if actual_obs_dim < self.obs_dim:
            padding = torch.zeros(batch_size, self.obs_dim - actual_obs_dim, 
                                device=obs.device, dtype=obs.dtype)
            obs = torch.cat([obs, padding], dim=1)
        
        if W1.size(0) == 1 and batch_size > 1:
            W1 = W1.expand(batch_size, -1, -1)
            b1 = b1.expand(batch_size, -1)
            W2 = W2.expand(batch_size, -1, -1)
            b2 = b2.expand(batch_size, -1)
        
        h = torch.bmm(W1, obs.unsqueeze(-1)).squeeze(-1) + b1
        h = F.relu(h)
        
        logits = torch.bmm(W2, h.unsqueeze(-1)).squeeze(-1) + b2
        value = torch.zeros(batch_size, 1, device=obs.device)
        
        if action_mask is not None:
            logits = logits.masked_fill(~action_mask, float('-inf'))
        
        actual_action_dim = E + 1
        logits = logits[:, :actual_action_dim]
        
        if single:
            logits = logits.squeeze(0)
            value = value.squeeze(0)
        
        return logits, value


# FIXED: Agent wrapper with buffer accumulation and delayed updates
class CrossScaleAgent:
    """Agent using hypernetwork for cross-scale policy - FIXED version."""
    
    def __init__(self, agent_id, hyper_net, device='cpu', lr=1e-6,  # FIXED: lr from 5e-5 to 1e-6
                 update_interval=10):  # FIXED: Update every N episodes
        self.agent_id = agent_id
        self.hyper_net = hyper_net
        self.device = device
        self.optimizer = torch.optim.Adam(hyper_net.parameters(), lr=lr)
        
        # Per-episode trajectory (cleared each episode)
        self.trajectory = {
            "states": [], "actions": [], "rewards": [],
            "values": [], "log_probs": [], "dones": []
        }
        
        # FIXED: Multi-episode buffer (accumulates before update)
        self.update_buffer = {
            "states": [], "actions": [], "rewards": [],
            "values": [], "log_probs": [], "dones": []
        }
        self.update_interval = update_interval
        self.episode_count = 0
        
        self.gamma = 0.99
        self.gae_lambda = 0.95
        self.clip_ratio = 0.2
        self.entropy_coeff = 0.05  # FIXED: Increased from 0.01 for more exploration
        self.value_coeff = 0.5
    
    def select_action(self, obs, M, E, action_mask=None):
        with torch.no_grad():
            obs_t = torch.from_numpy(obs).float().to(self.device)
            if action_mask is not None:
                mask_t = torch.from_numpy(action_mask).bool().to(self.device)
                max_dim = self.hyper_net.max_action_dim
                if mask_t.size(0) < max_dim:
                    padding = torch.ones(max_dim - mask_t.size(0), dtype=torch.bool, device=self.device)
                    mask_t = torch.cat([mask_t, padding])
                mask_t = mask_t.unsqueeze(0)
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
    
    def end_episode(self):
        """FIXED: Move trajectory to buffer and clear."""
        for key in self.trajectory:
            self.update_buffer[key].extend(self.trajectory[key])
        self.clear_trajectory()
        self.episode_count += 1
    
    def clear_trajectory(self):
        for k in self.trajectory:
            self.trajectory[k].clear()
    
    def should_update(self):
        """FIXED: Check if enough episodes accumulated."""
        return self.episode_count >= self.update_interval
    
    def compute_gae(self, rewards, values, dones, next_value=0.0):
        rewards = np.array(rewards)
        values = np.array(values + [next_value])
        dones = np.array(dones)
        
        deltas = rewards + self.gamma * values[1:] * (1 - dones) - values[:-1]
        adv = np.zeros_like(rewards)
        gae = 0.0
        
        for t in reversed(range(len(rewards))):
            gae = deltas[t] + self.gamma * self.gae_lambda * (1 - dones[t]) * gae
            adv[t] = gae
        
        ret = adv + values[:-1]
        adv = (adv - adv.mean()) / (adv.std() + 1e-8)
        
        return adv, ret
    
    def update(self, M, E, batch_size=256, num_epochs=4):  # FIXED: Larger batch_size
        """FIXED: Update from accumulated buffer, not single episode."""
        if not self.update_buffer["states"]:
            return {}
        
        states = np.array(self.update_buffer["states"])
        actions = np.array(self.update_buffer["actions"])
        old_lp = np.array(self.update_buffer["log_probs"])
        old_vals = np.array(self.update_buffer["values"])
        rewards = np.array(self.update_buffer["rewards"])
        dones = np.array(self.update_buffer["dones"])
        
        # FIXED: Compute GAE over entire buffer
        adv, ret = self.compute_gae(rewards.tolist(), old_vals.tolist(), dones.tolist())
        
        s_t = torch.from_numpy(states).float().to(self.device)
        a_t = torch.from_numpy(actions).long().to(self.device)
        o_t = torch.from_numpy(old_lp).float().to(self.device)
        d_t = torch.from_numpy(adv).float().to(self.device)
        r_t = torch.from_numpy(ret).float().to(self.device)
        
        losses = []
        n = len(actions)
        
        # FIXED: Better batch handling
        actual_batch = min(batch_size, n)
        
        for _ in range(num_epochs):
            perm = np.random.permutation(n)
            for i in range(0, n, actual_batch):
                idx = perm[i:i+actual_batch]
                
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
                torch.nn.utils.clip_grad_norm_(self.hyper_net.parameters(), 1.0)  # FIXED: Less aggressive clipping
                self.optimizer.step()
                losses.append(loss.item())
        
        # FIXED: Clear buffer after update
        for k in self.update_buffer:
            self.update_buffer[k].clear()
        self.episode_count = 0
        
        return {"loss": np.mean(losses), "buffer_size": n}
