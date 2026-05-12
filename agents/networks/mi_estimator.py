"""
Mutual Information Estimators for ExplabOff.

Implements two neural MI estimators as described in:
Ren et al., "ExplabOff: Towards Explorative and Collaborative Task
Offloading via Mutual Information-Enhanced MARL", INFOCOM 2025.

1. InfoNCE — Lower bound of I(s; a) for superior episodes (B+)
2. L1Out  — Upper bound of I(s; a) for inferior episodes (B-)
"""

import torch
import torch.nn as nn


class InfoNCEEstimator(nn.Module):
    """
    InfoNCE lower bound estimator for MI.

    I_{NCE}(s; a) = log(K) - L_{NCE}
    where L_{NCE} = -E[ log( exp(q(s, a)) / E_{a_j} exp(q(s, a_j)) ) ]

    Used for B+ buffer: maximize MI to strengthen optimal collaborations.

    Args:
        state_dim: Global state dimension
        action_dim: Joint action dimension (or encoded action dim)
        hidden_dim: Hidden layer size for score function
        K: Number of negative samples
    """

    def __init__(self, state_dim: int, action_dim: int, hidden_dim: int = 128, K: int = 32):
        super().__init__()
        self.K = K

        # Score function q_ψ(s, a): joint state-action → scalar
        self.score_fn = nn.Sequential(
            nn.Linear(state_dim + action_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, states: torch.Tensor, actions: torch.Tensor) -> torch.Tensor:
        """
        Compute I_NCE lower bound.

        Args:
            states: [batch_size, state_dim]
            actions: [batch_size, action_dim]

        Returns:
            Estimated MI lower bound (scalar)

        --- TO IMPLEMENT: InfoNCE contrastive loss ---
        """
        raise NotImplementedError("InfoNCE forward not implemented")

    def compute_loss(self, states: torch.Tensor, actions: torch.Tensor) -> torch.Tensor:
        """
        Compute InfoNCE loss for training the scorer.

        --- TO IMPLEMENT ---
        """
        raise NotImplementedError


class L1OutEstimator(nn.Module):
    """
    L1Out upper bound estimator for MI.

    I_{L1Out}(s; a) = E_{(s,a)~B-}[ log q(a|s) - log E_{s_j≠s} q(a|s_j) ]

    Used for B- buffer: minimize MI to break sub-optimal collaborations.

    Args:
        state_dim: Global state dimension
        action_dim: Joint action dimension
        hidden_dim: Hidden layer size for variational distribution
    """

    def __init__(self, state_dim: int, action_dim: int, hidden_dim: int = 128):
        super().__init__()
        # Variational approximation q_ψ(a|s) to p(a|s)
        self.var_dist = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim),
        )

    def forward(self, states: torch.Tensor, actions: torch.Tensor) -> torch.Tensor:
        """
        Compute I_L1Out upper bound.

        Args:
            states: [batch_size, state_dim]
            actions: [batch_size, action_dim]

        Returns:
            Estimated MI upper bound (scalar)

        --- TO IMPLEMENT: L1Out bound computation ---
        """
        raise NotImplementedError("L1Out forward not implemented")

    def compute_loss(self, states: torch.Tensor, actions: torch.Tensor) -> torch.Tensor:
        """
        Compute L1Out loss for training the variational distribution.

        --- TO IMPLEMENT ---
        """
        raise NotImplementedError
