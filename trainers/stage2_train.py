"""
Stage 2 Trainer: ExplabOff Training Loop (INFOCOM 2025).
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
from agents import ExplabOffAgent
from utils import MetricsCollector


def discrete_to_dict_action(action: int, num_servers: int) -> Dict:
    if action == 0:
        return {"offload_ratio": np.array([0.0], dtype=np.float32), "target_es": 0}
    es = min(action - 1, num_servers - 1)
    return {"offload_ratio": np.array([1.0], dtype=np.float32), "target_es": es + 1}


class Stage2Trainer:
    def __init__(self, env: EdgeOffloadEnv, config: Dict, experiment_name: str = None):
        self.env = env; self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.M = config["environment"]["num_devices"]
        self.E = config["environment"]["num_servers"]
        obs_dim = env.observation_spaces[env.agents[0]].shape[0]
        action_dim = self.E + 1

        self.total_episodes = config["marl"]["total_episodes"]
        self.episode_length = config["marl"]["episode_length"]
        self.batch_size = config["algorithm"]["batch_size"]
        self.num_epochs = config["algorithm"]["num_epochs"]
        self.save_interval = config["evaluation"]["save_interval"]

        self._shared_network = None
        self.agents = []
        for i in range(self.M):
            agent = ExplabOffAgent(
                agent_id=i, state_dim=obs_dim, action_dim=action_dim,
                hidden_dim=config["algorithm"]["hidden_dim"],
                lr=config["algorithm"]["learning_rate"],
                gamma=config["algorithm"]["gamma"],
                gae_lambda=config["algorithm"]["gae_lambda"],
                clip_ratio=config["algorithm"]["clip_ratio"],
                entropy_coeff=config["algorithm"]["entropy_coeff"],
                value_coeff=config["algorithm"]["value_coeff"],
                max_grad_norm=config["algorithm"]["max_grad_norm"],
                mi_mu=config["algorithm"].get("mi_weight_mu", 0.01),
                mi_nu=config["algorithm"].get("mi_weight_nu", 0.01),
                device=self.device)
            if i == 0: self._shared_network = agent.network
            else: agent.network = self._shared_network
            self.agents.append(agent)

        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.exp_name = experiment_name or f"ExplabOff_{ts}"
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
        ep_rewards = []  # track for B+/B- classification

        for step in range(self.episode_length):
            agent_data = {}
            actions = {}
            for i, agent in enumerate(self.agents):
                a_id = f"device_{i}"
                act_idx, logp, val = agent.select_action(obs[a_id])
                agent_data[a_id] = (act_idx, logp, val)
                actions[a_id] = discrete_to_dict_action(act_idx, self.E)

            next_obs, rewards, terms, truncs, _ = self.env.step(actions)

            for i, agent in enumerate(self.agents):
                a_id = f"device_{i}"
                act_idx, logp, val = agent_data[a_id]
                # MI-enhanced reward
                mi_bonus = agent.compute_mi_reward(obs[a_id], act_idx)
                enhanced_r = rewards[a_id] + mi_bonus
                agent.store_transition(obs[a_id], act_idx, enhanced_r, val, logp, terms[a_id])
                ep_rewards.append(rewards[a_id])

            obs = next_obs
            total_r += sum(rewards.values())
            if any(terms.values()): break

        # Episode classification (B+/B-)
        avg_ep_r = np.mean(ep_rewards) if ep_rewards else 0.0
        for agent in self.agents:
            agent.classify_episode(avg_ep_r)

        avg_r = total_r / max(step + 1, 1)
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
        print(f"\n{'='*60}\nExplabOff Training — {self.exp_name}\nEpisodes: {self.total_episodes}\n{'='*60}\n")
        best_reward = float("-inf")
        for ep in tqdm(range(self.total_episodes), desc="ExplabOff"):
            avg_r = self.train_episode(ep)
            for agent in self.agents:
                loss = agent.update(self.batch_size, self.num_epochs)
                if loss: self.writer.add_scalar("loss/total", loss["total_loss"], ep)
            for agent in self.agents: agent.clear_trajectory()
            if (ep + 1) % self.save_interval == 0:
                eval_m = self.evaluate(3)
                for k, v in eval_m.items(): self.writer.add_scalar(f"eval/{k}", v, ep)
                if avg_r > best_reward:
                    best_reward = avg_r
                    self.save_checkpoint(ep, "best")
            # Log MI buffer sizes
            self.writer.add_scalar("mi/b_plus_size", len(self.agents[0].B_plus), ep)
            self.writer.add_scalar("mi/b_minus_size", len(self.agents[0].B_minus), ep)
        self.save_checkpoint(self.total_episodes - 1, "final")
        self.writer.close()
        print(f"\nTraining complete. Best reward: {best_reward:.4f}")

    def save_checkpoint(self, episode: int, tag: str = ""):
        name = f"ep{episode}" + (f"_{tag}" if tag else "")
        torch.save({"episode": episode, "network": self._shared_network.state_dict(),
                    "config": self.config}, self.ckpt_dir / f"{name}.pt")

    def load_checkpoint(self, path: str):
        ckpt = torch.load(path, map_location=self.device, weights_only=False)
        self._shared_network.load_state_dict(ckpt["network"])
        return ckpt.get("episode", 0)
