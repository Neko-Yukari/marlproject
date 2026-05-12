"""
Stage 1 Trainer: IPPO Baseline Training Loop.

Trains M independent PPO agents on the PettingZoo EdgeOffloadEnv.
Each agent uses local observation — no inter-agent communication.

Reference: de Witt et al., ICLR 2020.
"""

import torch
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional
from torch.utils.tensorboard import SummaryWriter

from envs import EdgeOffloadEnv
from agents import IPPOAgent
from utils import MetricsCollector


class Stage1Trainer:
    """
    IPPO training loop.

    Key design decisions:
    - Parameter sharing: All agents share network weights
    - Agent ID embedding: Distinguish agents via learnable embedding
    - Global reward: All agents receive the same system-level scalar reward
    - GAE + PPO clip for stable policy updates

    TO IMPLEMENT:
    - Training loop (collect rollouts, compute GAE, PPO update)
    - TensorBoard logging
    - Checkpoint save/load
    - Periodic evaluation
    """

    def __init__(
        self,
        env: EdgeOffloadEnv,
        config: Dict,
        experiment_name: str = "stage1_ippo",
    ):
        self.env = env
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.total_episodes = config["marl"]["total_episodes"]
        self.episode_length = config["marl"]["episode_length"]
        self.batch_size = config["algorithm"]["batch_size"]
        self.num_epochs = config["algorithm"]["num_epochs"]
        self.save_interval = config["evaluation"]["save_interval"]

        # --- TO IMPLEMENT: Create agents ---
        self.agents: List[IPPOAgent] = []

        # --- TO IMPLEMENT: Experiment dirs, TensorBoard, MetricsCollector ---
        self.experiment_dir: Optional[Path] = None

    def train_episode(self) -> float:
        """
        Train one episode: collect rollout → update policies.

        Returns:
            Average reward across all agents.

        --- TO IMPLEMENT ---
        """
        raise NotImplementedError("Stage1Trainer.train_episode not implemented")

    def evaluate(self, num_episodes: int = 10) -> Dict[str, float]:
        """
        Evaluate current policies (deterministic, no exploration).

        --- TO IMPLEMENT ---
        """
        raise NotImplementedError("Stage1Trainer.evaluate not implemented")

    def train(self):
        """
        Full training loop over total_episodes.

        --- TO IMPLEMENT ---
        """
        raise NotImplementedError("Stage1Trainer.train not implemented")

    def save_checkpoint(self, episode: int):
        """Save all agent networks and optimizer states."""

    def load_checkpoint(self, path: str):
        """Restore training from checkpoint."""
