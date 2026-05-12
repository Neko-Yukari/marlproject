"""
MAPPO Agent — Multi-Agent PPO with centralized critic.

Stage 1 comparison baseline (CTDE: centralized critic, decentralized actor).
The critic has access to global state during training.

Reference: Yu et al., "The Surprising Effectiveness of PPO in
           Cooperative Multi-Agent Games", NeurIPS 2022.
"""

import torch
import numpy as np
from typing import Dict, Tuple, List, Optional

from .networks.actor_critic import ActorCriticNetwork


class MAPPOAgent:
    """
    Multi-Agent PPO agent with centralized critic.

    During centralized training, the critic conditions on global state
    (concatenation of all agents' observations). During decentralized
    execution, only the actor is used with local observation.

    Hyperparameters: Same as IPPO baseline.
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
        self.max_grad_norm = max_grad_norm

        # --- TO IMPLEMENT: Actor network (local obs) ---
        self.actor: Optional[ActorCriticNetwork] = None

        # --- TO IMPLEMENT: Critic network (global obs) ---
        self.critic: Optional[torch.nn.Module] = None

        # --- TO IMPLEMENT: Optimizers ---
        self.actor_optimizer: Optional[torch.optim.Optimizer] = None
        self.critic_optimizer: Optional[torch.optim.Optimizer] = None

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
        """Select action with decentralized actor (local obs only)."""
        raise NotImplementedError("MAPPOAgent.select_action not implemented")

    def compute_value(self, global_state: np.ndarray) -> float:
        """Compute centralized value estimate."""
        raise NotImplementedError("MAPPOAgent.compute_value not implemented")

    def update(self, batch_size: int = 64, num_epochs: int = 4) -> Dict[str, float]:
        """Update actor and critic."""
        raise NotImplementedError("MAPPOAgent.update not implemented")

    def clear_trajectory(self):
        for k in self.trajectory:
            self.trajectory[k].clear()
