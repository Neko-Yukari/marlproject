"""
Mutual Information Plugin for ExplabOff.

Optional plugin that computes MI-enhanced rewards.
Can be attached to any PPOAgent to turn it into ExplabOff.
"""
import torch
import numpy as np
from typing import List, Tuple

from .networks.mi_estimator import InfoNCEEstimator, L1OutEstimator


class MIPlugin:
    """
    MI reward computation plugin.
    
    Computes InfoNCE and L1Out rewards to encourage exploration and collaboration.
    """
    
    def __init__(self, state_dim: int, action_dim: int, hidden_dim: int = 128,
                 mu: float = 0.01, nu: float = 0.01,
                 device: torch.device = torch.device("cpu")):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.mu = mu
        self.nu = nu
        self.device = device
        
        self.info_nce = InfoNCEEstimator(state_dim, action_dim, hidden_dim).to(device)
        self.l1_out = L1OutEstimator(state_dim, action_dim, hidden_dim).to(device)
        
        # Dual buffers for discriminative MI learning
        self.B_plus: List[Tuple] = []   # Superior episodes
        self.B_minus: List[Tuple] = []  # Inferior episodes
        self._best_ep_reward = float("-inf")
    
    def compute_reward(self, state: np.ndarray, action: int) -> float:
        """
        Compute MI-enhanced reward for a single (state, action) pair.
        
        Returns:
            mi_reward: scalar MI reward (small value, typically scaled by 0.01)
        """
        try:
            with torch.no_grad():
                s = torch.from_numpy(state).float().unsqueeze(0).to(self.device)
                a_onehot = torch.zeros(1, self.action_dim).to(self.device)
                a_onehot[0, action] = 1.0
                
                i_nce = self.info_nce(s, a_onehot)
                i_l1 = self.l1_out(s, a_onehot)
                
                mi = self.mu * i_nce - self.nu * i_l1
                
                if torch.isnan(mi) or torch.isinf(mi):
                    return 0.0
                
                return float(mi.item()) * 0.01
        except Exception:
            return 0.0
    
    def classify_episode(self, states: List, actions: List, ep_reward: float, max_buf: int = 1000):
        """
        Classify episode as superior or inferior based on reward.
        
        Args:
            states: list of states from the episode
            actions: list of actions from the episode
            ep_reward: total episode reward
            max_buf: maximum buffer size
        """
        samples = list(zip(states, actions))
        
        if ep_reward > self._best_ep_reward:
            self._best_ep_reward = ep_reward
            self.B_plus.extend(samples)
            if len(self.B_plus) > max_buf:
                self.B_plus = self.B_plus[-max_buf:]
        else:
            self.B_minus.extend(samples)
            if len(self.B_minus) > max_buf:
                self.B_minus = self.B_minus[-max_buf:]
    
    def update_estimators(self, batch_size: int = 32, optimizer=None):
        """
        Update MI estimators using stored buffers.
        
        Args:
            batch_size: batch size for training
            optimizer: shared optimizer (optional, if None uses internal)
        """
        # Update InfoNCE on B_plus
        if len(self.B_plus) >= batch_size:
            idx = np.random.choice(len(self.B_plus), batch_size, replace=False)
            s_b = torch.from_numpy(np.array([self.B_plus[i][0] for i in idx])).float().to(self.device)
            a_b = torch.zeros(batch_size, self.action_dim).to(self.device)
            for j, i in enumerate(idx):
                a_b[j, self.B_plus[i][1]] = 1.0
            
            loss = self.info_nce.compute_loss(s_b, a_b)
            if optimizer is not None:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
        
        # Update L1Out on B_minus
        if len(self.B_minus) >= batch_size:
            idx = np.random.choice(len(self.B_minus), batch_size, replace=False)
            s_b = torch.from_numpy(np.array([self.B_minus[i][0] for i in idx])).float().to(self.device)
            a_b = torch.zeros(batch_size, self.action_dim).to(self.device)
            for j, i in enumerate(idx):
                a_b[j, self.B_minus[i][1]] = 1.0
            
            loss = self.l1_out.compute_loss(s_b, a_b)
            if optimizer is not None:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
    
    def get_parameters(self):
        """Return all MI estimator parameters for optimizer."""
        return list(self.info_nce.parameters()) + list(self.l1_out.parameters())
    
    def state_dict(self):
        return {
            "info_nce": self.info_nce.state_dict(),
            "l1_out": self.l1_out.state_dict(),
            "best_reward": self._best_ep_reward
        }
    
    def load_state_dict(self, d: dict):
        self.info_nce.load_state_dict(d["info_nce"])
        self.l1_out.load_state_dict(d["l1_out"])
        self._best_ep_reward = d.get("best_reward", float("-inf"))
