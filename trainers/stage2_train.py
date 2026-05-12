"""
Stage 2 Trainer: ExplabOff Training Loop.

Implements the MI-enhanced MARL training from ExplabOff (INFOCOM 2025).
Key differences from IPPO:
- MI-augmented reward: r̂ = r + Î(s; a)
- Dual experience buffers: B+ (superior episodes), B- (inferior episodes)
- Periodic MI estimator updates (InfoNCE on B+, L1Out on B-)
- Centralized critic with global state during training

Reference: Ren et al., INFOCOM 2025.
"""

import torch
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional

from envs import EdgeOffloadEnv
from agents import ExplabOffAgent
from utils import MetricsCollector


class Stage2Trainer:
    """
    ExplabOff training loop.

    TO IMPLEMENT:
    - CENTRALIZED TRAINING: Critic uses global state (all MD states + all ES queues)
    - DECENTRALIZED EXECUTION: Actor uses only local observation
    - MI computation: I_NCE(s; a) lower bound, I_L1Out(s; a) upper bound
    - Episode classification: superior (r_episode > best) vs inferior
    - Dual buffer management: B+ (max MI), B- (min MI)
    - MI-enhanced reward injection into critic loss
    """

    def __init__(
        self,
        env: EdgeOffloadEnv,
        config: Dict,
        experiment_name: str = "stage2_explaboff",
    ):
        self.env = env
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # --- TO IMPLEMENT ---
        self.agents: List[ExplabOffAgent] = []

    def collect_episode(self) -> Dict:
        """
        Collect one episode of joint offloading experiences.

        --- TO IMPLEMENT ---
        """
        raise NotImplementedError

    def classify_and_buffer(self, episode_reward: float):
        """Classify episode into B+ or B- buffer."""

    def update_mi_estimators(self):
        """Update InfoNCE (B+) and L1Out (B-)."""

    def train_episode(self) -> float:
        """Train one episode with MI-enhanced reward."""
        raise NotImplementedError

    def train(self):
        """Full training loop."""
        raise NotImplementedError
