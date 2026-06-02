"""
Mutual Information Estimators for ExplabOff (INFOCOM 2025).
1. InfoNCE — Lower bound I(s;a) for superior episodes (B+)
2. L1Out  — Upper bound I(s;a) for inferior episodes (B-)
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class InfoNCEEstimator(nn.Module):
    """I_NCE(s;a) = log(K) - L_NCE. Maximize for B+."""
    def __init__(self, state_dim: int, action_dim: int, hidden_dim: int = 128, K: int = 32):
        super().__init__()
        self.K = K
        self.score_fn = nn.Sequential(
            nn.Linear(state_dim + action_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, 1))

    def forward(self, states: torch.Tensor, actions: torch.Tensor) -> torch.Tensor:
        """Compute I_NCE lower bound (scalar)."""
        B = states.size(0)
        sa = torch.cat([states, actions], dim=-1)       # [B, S+A]
        scores = self.score_fn(sa).squeeze(-1)           # [B]
        # Contrastive: positive pairs vs negatives (excluding self)
        L_NCE = 0.0
        for i in range(B):
            pos = scores[i]
            # Exclude self from negatives
            all_idx = torch.arange(B, device=states.device)
            valid_neg = all_idx[all_idx != i]
            if len(valid_neg) == 0:
                continue  # Skip if B=1 (no negatives)
            neg_idx = valid_neg[torch.randperm(len(valid_neg))[:self.K]]
            neg = scores[neg_idx]
            L_NCE = L_NCE - torch.log(torch.exp(pos) / (torch.exp(pos) + torch.exp(neg).sum()))
        L_NCE = L_NCE / B
        return torch.log(torch.tensor(self.K, dtype=torch.float32, device=states.device)) - L_NCE

    def compute_loss(self, states, actions) -> torch.Tensor:
        B = states.size(0)
        sa = torch.cat([states, actions], dim=-1)
        scores = self.score_fn(sa).squeeze(-1)
        loss = 0.0
        for i in range(B):
            pos = scores[i]
            # Exclude self from negatives
            all_idx = torch.arange(B, device=states.device)
            valid_neg = all_idx[all_idx != i]
            if len(valid_neg) == 0:
                continue  # Skip if B=1 (no negatives)
            neg_idx = valid_neg[torch.randperm(len(valid_neg))[:self.K]]
            neg = scores[neg_idx]
            log_softmax = pos - torch.log(torch.exp(pos) + torch.exp(neg).sum())
            loss = loss - log_softmax
        return loss / B


class L1OutEstimator(nn.Module):
    """I_L1Out(s;a) upper bound. Minimize for B-."""
    def __init__(self, state_dim: int, action_dim: int, hidden_dim: int = 128):
        super().__init__()
        self.var_dist = nn.Sequential(
            nn.Linear(state_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, action_dim))

    def forward(self, states: torch.Tensor, actions: torch.Tensor) -> torch.Tensor:
        B = states.size(0)
        pred_a = self.var_dist(states)
        log_q = -F.mse_loss(pred_a, actions, reduction='none').mean(-1)  # [B]
        # log E_{s_j} q(a_i|s_j) = log(1/B * sum_j q(a_i|s_j)) = logsumexp(cross) - log(B)
        cross_log_q = []
        for i in range(B):
            pred_j = self.var_dist(states)
            cross = -F.mse_loss(pred_j, actions[i:i+1].expand_as(pred_j), reduction='none').mean(-1)
            # log(1/B * sum_j exp(cross_j)) = logsumexp(cross) - log(B)
            cross_log_q_i = torch.logsumexp(cross, dim=0) - torch.log(torch.tensor(B, dtype=cross.dtype))
            cross_log_q.append(cross_log_q_i)
        return log_q.mean() - torch.stack(cross_log_q).mean()

    def compute_loss(self, states, actions) -> torch.Tensor:
        # For B- (inferior episodes), minimize I_L1Out upper bound
        return self.forward(states, actions)
