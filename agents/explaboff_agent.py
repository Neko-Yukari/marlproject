"""
ExplabOff Agent — Explorative and Collaborative Task Offloading
via Mutual Information-Enhanced MARL.

Stage 2 core implementation. Strictly follows the INFOCOM 2025 paper:
Ren et al., "ExplabOff: Towards Explorative and Collaborative Task
Offloading via Mutual Information-Enhanced MARL", IEEE INFOCOM 2025.

Key innovations:
1. MI-augmented critic objective: J(Q) = E[γ^{n-1}·(r + I(s;a))]
2. Superior/Inferior episode buffer split (B+ / B-)
3. InfoNCE (lower bound) for B+, L1Out (upper bound) for B-
4. Synthetic MI: Î(s;a) = μ·I_NCE − ν·I_L1Out
"""

import torch
import numpy as np
from typing import Dict, Tuple, List, Optional

from .networks.actor_critic import ActorCriticNetwork
from .networks.mi_estimator import InfoNCEEstimator, L1OutEstimator


class ExplabOffAgent:
    """
    ExplabOff agent: PPO extended with MI-enhanced exploration and collaboration.

    Architecture (per agent):
        - Actor: π_φ(s_m) → action  (local obs, decentralized execution)
        - Critic: Q_θ(s, a) → value  (global state, centralized training)
        - MI Estimators: InfoNCE + L1Out
        - Dual experience buffers: B+ (superior episodes), B- (inferior episodes)

    Hyperparameters (from ExplabOff Table):
        learning_rate: 5e-5
        μ (MI weight): 0.01
        ν (MI weight): 0.01
        buffer sizes: configurable
    """

    def __init__(
        self,
        agent_id: int,
        state_dim: int,
        global_state_dim: int,
        action_dim: int,
        hidden_dim: int = 128,
        learning_rate: float = 5e-5,
        gamma: float = 0.99,
        gae_lambda: float = 0.95,
        clip_ratio: float = 0.2,
        entropy_coeff: float = 0.01,
        value_coeff: float = 0.5,
        max_grad_norm: float = 0.5,
        mi_weight_mu: float = 0.01,
        mi_weight_nu: float = 0.01,
        superior_buffer_size: int = 1000,
        inferior_buffer_size: int = 1000,
        device: torch.device = torch.device("cpu"),
    ):
        self.agent_id = agent_id
        self.state_dim = state_dim
        self.global_state_dim = global_state_dim
        self.action_dim = action_dim
        self.device = device
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.clip_ratio = clip_ratio
        self.entropy_coeff = entropy_coeff
        self.value_coeff = value_coeff
        self.mi_weight_mu = mi_weight_mu
        self.mi_weight_nu = mi_weight_nu

        # --- TO IMPLEMENT: Actor-Critic network ---
        self.network: Optional[ActorCriticNetwork] = None

        # --- TO IMPLEMENT: MI estimators ---
        self.info_nce: Optional[InfoNCEEstimator] = None
        self.l1_out: Optional[L1OutEstimator] = None

        # --- TO IMPLEMENT: Optimizer ---
        self.optimizer: Optional[torch.optim.Optimizer] = None

        # Dual experience buffers
        self.superior_buffer: List[Dict] = []   # B+
        self.inferior_buffer: List[Dict] = []    # B-
        self.max_superior = superior_buffer_size
        self.max_inferior = inferior_buffer_size
        self._best_episode_reward = float("-inf")

        # Trajectory buffer (current episode)
        self.trajectory: Dict[str, List] = {
            "states": [],
            "global_states": [],
            "actions": [],
            "rewards": [],
            "values": [],
            "log_probs": [],
            "dones": [],
        }

    def select_action(self, state: np.ndarray) -> Tuple[int, float, float]:
        """Select action using decentralized actor."""
        raise NotImplementedError("ExplabOffAgent.select_action not implemented")

    def compute_mi_enhanced_reward(
        self, state: torch.Tensor, action: torch.Tensor
    ) -> float:
        """
        Compute MI-enhanced reward: r̂ = r + Î(s; a).

        Î(s; a) = μ·I_NCE(s; a) − ν·I_L1Out(s; a)

        --- TO IMPLEMENT ---
        """
        raise NotImplementedError("ExplabOffAgent.compute_mi_enhanced_reward not implemented")

    def classify_episode(self, episode_reward: float):
        """
        Classify current episode as superior or inferior.
        Update B+ and B- buffers accordingly.
        """
        if episode_reward > self._best_episode_reward:
            self._best_episode_reward = episode_reward
            self._push_to_buffer(self.superior_buffer, self.max_superior)
        else:
            self._push_to_buffer(self.inferior_buffer, self.max_inferior)

    def _push_to_buffer(self, buffer: List, max_size: int):
        """Push current trajectory to buffer (FIFO if full)."""

    def update(self, batch_size: int = 64, num_epochs: int = 4) -> Dict[str, float]:
        """Update policy, critic, and MI estimators."""
        raise NotImplementedError("ExplabOffAgent.update not implemented")

    def update_mi_estimators(self):
        """
        Periodically update InfoNCE (on B+) and L1Out (on B-).

        --- TO IMPLEMENT ---
        """
        raise NotImplementedError("ExplabOffAgent.update_mi_estimators not implemented")

    def clear_trajectory(self):
        for k in self.trajectory:
            self.trajectory[k].clear()
