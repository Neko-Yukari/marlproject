"""
Stage 1 Trainer: IPPO Baseline Training Loop.
Trains M independent PPO agents on PettingZoo EdgeOffloadEnv.
Reference: de Witt et al., ICLR 2020.
"""

import torch
import numpy as np
import json
from pathlib import Path
from datetime import datetime
from typing import Dict, List
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from envs import EdgeOffloadEnv
from agents import IPPOAgent
from utils import MetricsCollector


def discrete_to_dict_action(action: int, num_servers: int) -> Dict:
    """Map discrete action to environment Dict action.
    action=0 → local only; action=e+1 → full offload to server e."""
    if action == 0:
        return {"offload_ratio": np.array([0.0], dtype=np.float32), "target_es": 0}
    es = min(action - 1, num_servers - 1)
    return {"offload_ratio": np.array([1.0], dtype=np.float32), "target_es": es + 1}


class Stage1Trainer:
    def __init__(self, env: EdgeOffloadEnv, config: Dict, experiment_name: str = None):
        self.env = env
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.M = config["environment"]["num_devices"]
        self.E = config["environment"]["num_servers"]
        obs_dim = self.env.observation_spaces[env.agents[0]].shape[0]
        action_dim = self.E + 1  # local + E servers

        self.total_episodes = config["marl"]["total_episodes"]
        self.episode_length = config["marl"]["episode_length"]
        self.batch_size = config["algorithm"]["batch_size"]
        self.num_epochs = config["algorithm"]["num_epochs"]
        self.save_interval = config["evaluation"]["save_interval"]

        # Parameter-shared agents (all share one network)
        self._shared_network = None  # will be set after first agent creation
        self.agents = []
        for i in range(self.M):
            agent = IPPOAgent(
                agent_id=i, state_dim=obs_dim, action_dim=action_dim,
                hidden_dim=config["algorithm"]["hidden_dim"],
                learning_rate=config["algorithm"]["learning_rate"],
                gamma=config["algorithm"]["gamma"],
                gae_lambda=config["algorithm"]["gae_lambda"],
                clip_ratio=config["algorithm"]["clip_ratio"],
                entropy_coeff=config["algorithm"]["entropy_coeff"],
                value_coeff=config["algorithm"]["value_coeff"],
                max_grad_norm=config["algorithm"]["max_grad_norm"],
                device=self.device,
            )
            if i == 0:
                self._shared_network = agent.network
            else:
                agent.network = self._shared_network
                agent.optimizer = torch.optim.Adam(agent.network.parameters(),
                                                   lr=config["algorithm"]["learning_rate"])
            self.agents.append(agent)

        # Experiment tracking
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.exp_name = experiment_name or f"IPPO_{ts}"
        self.exp_dir = Path("results") / self.exp_name
        self.ckpt_dir = self.exp_dir / "checkpoints"
        self.ckpt_dir.mkdir(parents=True, exist_ok=True)
        self.writer = SummaryWriter(log_dir=str(self.exp_dir / "logs"))
        self.metrics = MetricsCollector()
        with open(self.exp_dir / "config.yaml", "w") as f:
            json.dump(config, f, indent=2)

    def train_episode(self, episode: int) -> float:
        obs, _ = self.env.reset()
        total_r = 0.0
        for step in range(self.episode_length):
            # Collect actions from all agents
            agent_data = {}  # a_id -> (action_idx, log_prob, value)
            actions = {}
            for i, agent in enumerate(self.agents):
                a_id = f"device_{i}"
                act_idx, logp, val = agent.select_action(obs[a_id])
                agent_data[a_id] = (act_idx, logp, val)
                actions[a_id] = discrete_to_dict_action(act_idx, self.E)

            # Environment step
            next_obs, rewards, terms, truncs, _ = self.env.step(actions)

            # Store transitions per agent
            for i, agent in enumerate(self.agents):
                a_id = f"device_{i}"
                act_idx, logp, val = agent_data[a_id]
                agent.store_transition(
                    obs[a_id], act_idx, rewards[a_id], val, logp, terms[a_id])
            obs = next_obs
            total_r += sum(rewards.values())
            if any(terms.values()):
                break
        avg_r = total_r / max(len(rewards) if 'rewards' in dir() else 1, 1)
        self.writer.add_scalar("episode/reward", avg_r, episode)
        return avg_r

    def evaluate(self, num_eps: int = 10) -> Dict[str, float]:
        results = {"completion_rate": [], "avg_latency": [], "avg_energy": []}
        for _ in range(num_eps):
            obs, _ = self.env.reset()
            for _ in range(self.episode_length):
                actions = {}
                for i, agent in enumerate(self.agents):
                    a_id = f"device_{i}"
                    probs = agent.get_action_probs(obs[a_id])
                    actions[a_id] = discrete_to_dict_action(int(np.argmax(probs)), self.E)
                obs, _, terms, _, _ = self.env.step(actions)
                if any(terms.values()): break
            m = self.env.get_episode_metrics()
            results["completion_rate"].append(m["completion_rate"])
            results["avg_latency"].append(m["avg_latency"])
            results["avg_energy"].append(m["avg_energy"])
        return {k: float(np.mean(v)) for k, v in results.items()}

    def train(self):
        print(f"\n{'='*60}\nIPPO Training — {self.exp_name}\nEpisodes: {self.total_episodes}\nAgents: {self.M}\n{'='*60}\n")
        best_reward = float("-inf")
        for ep in tqdm(range(self.total_episodes), desc="IPPO"):
            avg_r = self.train_episode(ep)
            for agent in self.agents:
                loss = agent.update(self.batch_size, self.num_epochs)
                if loss:
                    self.writer.add_scalar("loss/total", loss["total_loss"], ep)
            for agent in self.agents:
                agent.clear_trajectory()
            if (ep + 1) % self.save_interval == 0:
                eval_m = self.evaluate(3)
                for k, v in eval_m.items():
                    self.writer.add_scalar(f"eval/{k}", v, ep)
                if avg_r > best_reward:
                    best_reward = avg_r
                    self.save_checkpoint(ep, "best")
        self.save_checkpoint(self.total_episodes - 1, "final")
        self.writer.close()
        print(f"\nTraining complete. Best reward: {best_reward:.4f}")

    def save_checkpoint(self, episode: int, tag: str = ""):
        name = f"ep{episode}" + (f"_{tag}" if tag else "")
        torch.save({
            "episode": episode,
            "network": self._shared_network.state_dict(),
            "config": self.config,
        }, self.ckpt_dir / f"{name}.pt")

    def load_checkpoint(self, path: str):
        ckpt = torch.load(path, map_location=self.device, weights_only=False)
        self._shared_network.load_state_dict(ckpt["network"])
        return ckpt.get("episode", 0)
