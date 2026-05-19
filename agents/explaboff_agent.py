"""
ExplabOff Agent — MI-enhanced MARL for edge offloading (Stage 2).
Strictly follows: Ren et al., INFOCOM 2025.
"""
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from typing import Dict, Tuple, List

from .networks.actor_critic import ActorCriticNetwork
from .networks.mi_estimator import InfoNCEEstimator, L1OutEstimator


class ExplabOffAgent:
    def __init__(self, agent_id: int, state_dim: int, action_dim: int,
                 hidden_dim: int = 128, lr: float = 5e-5,
                 gamma: float = 0.99, gae_lambda: float = 0.95,
                 clip_ratio: float = 0.2, entropy_coeff: float = 0.01,
                 value_coeff: float = 0.5, max_grad_norm: float = 0.5,
                 mi_mu: float = 0.01, mi_nu: float = 0.01,
                 device: torch.device = torch.device("cpu")):
        self.agent_id = agent_id
        self.state_dim = state_dim; self.action_dim = action_dim
        self.device = device
        self.gamma = gamma; self.gae_lambda = gae_lambda
        self.clip_ratio = clip_ratio; self.entropy_coeff = entropy_coeff
        self.value_coeff = value_coeff; self.max_grad_norm = max_grad_norm
        self.mi_mu = mi_mu; self.mi_nu = mi_nu

        self.network = ActorCriticNetwork(state_dim, action_dim, hidden_dim).to(device)
        self.info_nce = InfoNCEEstimator(state_dim, action_dim, hidden_dim).to(device)
        self.l1_out   = L1OutEstimator(state_dim, action_dim, hidden_dim).to(device)

        params = (list(self.network.parameters()) + list(self.info_nce.parameters())
                  + list(self.l1_out.parameters()))
        self.optimizer = optim.Adam(params, lr=lr)

        self.trajectory: Dict[str, List] = {"states":[],"actions":[],"rewards":[],
                                             "values":[],"log_probs":[],"dones":[]}
        self.B_plus: List[Tuple] = []
        self.B_minus: List[Tuple] = []
        self._best_ep_reward = float("-inf")

    # ── Action ──
    def select_action(self, state: np.ndarray) -> Tuple[int, float, float]:
        with torch.no_grad():
            s = torch.from_numpy(state).float().unsqueeze(0).to(self.device)
            probs, value = self.network(s)
            dist = torch.distributions.Categorical(probs)
            a = dist.sample()
            return a.item(), dist.log_prob(a).item(), value.item()

    def get_action_probs(self, state: np.ndarray) -> np.ndarray:
        with torch.no_grad():
            s = torch.from_numpy(state).float().unsqueeze(0).to(self.device)
            probs, _ = self.network(s)
            return probs.cpu().numpy().squeeze(0)

    # ── Trajectory ──
    def store_transition(self, state, action, reward, value, log_prob, done):
        for k, v in zip(["states","actions","rewards","values","log_probs","dones"],
                        [state, action, reward, value, log_prob, done]):
            self.trajectory[k].append(v)

    def clear_trajectory(self):
        for k in self.trajectory: self.trajectory[k].clear()

    # ── MI-enhanced reward ──
    def compute_mi_reward(self, state: np.ndarray, action: int) -> float:
        with torch.no_grad():
            s = torch.from_numpy(state).float().unsqueeze(0).to(self.device)
            a_onehot = torch.zeros(1, self.action_dim).to(self.device)
            a_onehot[0, action] = 1.0
            i_nce = self.info_nce(s, a_onehot).item()
            i_l1  = self.l1_out(s, a_onehot).item()
            return self.mi_mu * i_nce - self.mi_nu * i_l1

    # ── Dual buffers ──
    def classify_episode(self, ep_reward: float, max_buf: int = 1000):
        samples = list(zip(self.trajectory["states"], self.trajectory["actions"]))
        if ep_reward > self._best_ep_reward:
            self._best_ep_reward = ep_reward
            self.B_plus.extend(samples)
            if len(self.B_plus) > max_buf: self.B_plus = self.B_plus[-max_buf:]
        else:
            self.B_minus.extend(samples)
            if len(self.B_minus) > max_buf: self.B_minus = self.B_minus[-max_buf:]

    def update_mi_estimators(self, batch_size: int = 32):
        if len(self.B_plus) >= batch_size:
            idx = np.random.choice(len(self.B_plus), batch_size, replace=False)
            s_b = torch.from_numpy(np.array([self.B_plus[i][0] for i in idx])).float().to(self.device)
            a_b = torch.zeros(batch_size, self.action_dim).to(self.device)
            for j, i in enumerate(idx):
                a_b[j, self.B_plus[i][1]] = 1.0
            loss = self.info_nce.compute_loss(s_b, a_b)
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
        if len(self.B_minus) >= batch_size:
            idx = np.random.choice(len(self.B_minus), batch_size, replace=False)
            s_b = torch.from_numpy(np.array([self.B_minus[i][0] for i in idx])).float().to(self.device)
            a_b = torch.zeros(batch_size, self.action_dim).to(self.device)
            for j, i in enumerate(idx):
                a_b[j, self.B_minus[i][1]] = 1.0
            loss = self.l1_out.compute_loss(s_b, a_b)
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

    # ── GAE ──
    def compute_gae(self, next_value: float = 0.0) -> Tuple[np.ndarray, np.ndarray]:
        rewards = np.array(self.trajectory["rewards"])
        values  = np.array(self.trajectory["values"] + [next_value])
        dones   = np.array(self.trajectory["dones"])
        deltas  = rewards + self.gamma * values[1:] * (1 - dones) - values[:-1]
        adv = np.zeros_like(rewards); gae = 0.0
        for t in reversed(range(len(rewards))):
            gae = deltas[t] + self.gamma * self.gae_lambda * (1 - dones[t]) * gae
            adv[t] = gae
        ret = adv + values[:-1]
        adv = (adv - adv.mean()) / (adv.std() + 1e-8)
        return adv, ret

    # ── PPO + MI update ──
    def update(self, batch_size: int = 64, num_epochs: int = 4) -> Dict[str, float]:
        if not self.trajectory["states"]: return {}
        states  = np.array(self.trajectory["states"])
        actions = np.array(self.trajectory["actions"])
        old_lp  = np.array(self.trajectory["log_probs"])
        adv, ret = self.compute_gae()

        s_t = torch.from_numpy(states).float().to(self.device)
        a_t = torch.from_numpy(actions).long().to(self.device)
        o_t = torch.from_numpy(old_lp).float().to(self.device)
        d_t = torch.from_numpy(adv).float().to(self.device)
        r_t = torch.from_numpy(ret).float().to(self.device)

        losses = []
        n = len(states)
        for _ in range(num_epochs):
            perm = np.random.permutation(n)
            for i in range(0, n, batch_size):
                idx = perm[i:i+batch_size]
                probs, values = self.network(s_t[idx])
                dist = torch.distributions.Categorical(probs)
                new_lp  = dist.log_prob(a_t[idx])
                entropy = dist.entropy().mean()
                ratio   = torch.exp(new_lp - o_t[idx])
                surr1   = ratio * d_t[idx]
                surr2   = torch.clamp(ratio, 1-self.clip_ratio, 1+self.clip_ratio) * d_t[idx]
                policy_loss = -torch.min(surr1, surr2).mean()
                value_loss  = 0.5 * (r_t[idx] - values.squeeze(-1)).pow(2).mean()
                loss = policy_loss + self.value_coeff*value_loss - self.entropy_coeff*entropy
                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.network.parameters(), self.max_grad_norm)
                self.optimizer.step()
                losses.append(loss.item())

        # Update MI estimators periodically
        if len(self.B_plus) >= 32 or len(self.B_minus) >= 32:
            self.update_mi_estimators()

        return {"total_loss": float(np.mean(losses)) if losses else 0.0}

    def save(self, path: str):
        torch.save({"network": self.network.state_dict(),
                    "info_nce": self.info_nce.state_dict(),
                    "l1_out": self.l1_out.state_dict(),
                    "agent_id": self.agent_id}, path)

    def load(self, path: str):
        ckpt = torch.load(path, map_location=self.device, weights_only=False)
        self.network.load_state_dict(ckpt["network"])
        self.info_nce.load_state_dict(ckpt["info_nce"])
        self.l1_out.load_state_dict(ckpt["l1_out"])
