"""
IPPO Agent — Independent PPO for MARL edge offloading.

Stage 1 baseline (no inter-agent communication).
Each MD learns independently with shared network parameters.

Reference: de Witt et al., "Is Independent Learning All You Need?
           in the StarCraft Multi-Agent Challenge?", ICLR 2020.
"""

import torch
import numpy as np
from typing import Dict, Tuple, List, Optional

from .networks.actor_critic import ActorCriticNetwork


class IPPOAgent:
    """
    Independent PPO agent for a single mobile device.

    Each agent maintains its own actor-critic network and trajectory buffer.
    Parameters are optionally shared across agents (controlled by trainer).

    Hyperparameters:
        learning_rate: 5e-5 (from ExplabOff paper)
        gamma: 0.99
        gae_lambda: 0.95
        clip_ratio: 0.2
        entropy_coeff: 0.01
        value_coeff: 0.5
        max_grad_norm: 0.5
    """

    def __init__(
        self,
        agent_id: int,
        state_dim: int,
        action_dim: int,
        hidden_dim: int = 128,
        learning_rate: float = 5e-5,
        gamma: float = 0.99,
        gae_lambda: float = 0.95,
        clip_ratio: float = 0.2,
        entropy_coeff: float = 0.01,
        value_coeff: float = 0.5,
        max_grad_norm: float = 0.5,
        device: torch.device = torch.device("cpu"),
    ):
        self.agent_id = agent_id
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.device = device
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.clip_ratio = clip_ratio
        self.entropy_coeff = entropy_coeff
        self.value_coeff = value_coeff
        self.max_grad_norm = max_grad_norm

        # --- TO IMPLEMENT: Network ---
        self.network: Optional[ActorCriticNetwork] = None

        # --- TO IMPLEMENT: Optimizer ---
        self.optimizer: Optional[torch.optim.Optimizer] = None

        # Trajectory buffer
        self.trajectory: Dict[str, List] = {
            "states": [],
            "actions": [],
            "rewards": [],
            "values": [],
            "log_probs": [],
            "dones": [],
        }

    def select_action(self, state: np.ndarray) -> Tuple[int, float, float]:
        """
        Select action using current policy.

        Args:
            state: Local observation vector [state_dim]

        Returns:
            (action, log_prob, value_estimate)

        --- TO IMPLEMENT ---
        """
        raise NotImplementedError("IPPOAgent.select_action not implemented")

    def store_transition(
        self,
        state: np.ndarray,
        action: int,
        reward: float,
        value: float,
        log_prob: float,
        done: bool,
    ):
        """Store transition in trajectory buffer."""

    def compute_gae_advantages(self, next_value: float = 0.0) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute Generalized Advantage Estimation.

        Returns:
            (advantages, returns)

        --- TO IMPLEMENT ---
        """
        raise NotImplementedError("IPPOAgent.compute_gae_advantages not implemented")

    def update(self, batch_size: int = 64, num_epochs: int = 4) -> Dict[str, float]:
        """
        Update policy using PPO loss.

        Returns:
            Dict with 'total_loss', 'policy_loss', 'value_loss', 'entropy_loss'

        --- TO IMPLEMENT ---
        """
        raise NotImplementedError("IPPOAgent.update not implemented")

    def clear_trajectory(self):
        """Reset trajectory buffer for next episode."""
        for k in self.trajectory:
            self.trajectory[k].clear()

    def save(self, path: str):
        """Save agent state to checkpoint."""

    def load(self, path: str):
        """Load agent state from checkpoint."""
