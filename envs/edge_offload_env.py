"""
PettingZoo ParallelEnv for MARL edge computing task offloading.

M devices act as independent agents in parallel each time slot,
deciding how much of their task to offload to which edge server.

Based on: ExplabOff (Ren et al., INFOCOM 2025)
"""

import numpy as np
from typing import Dict, List, Tuple, Optional
from functools import lru_cache

from pettingzoo import ParallelEnv
from gymnasium import spaces

from utils.task_device import Task, Device
from utils.helpers import (
    create_task,
    calculate_transmission_rate,
    calculate_local_latency,
    calculate_edge_latency,
    calculate_local_energy,
    calculate_transmission_energy,
    compute_fairness_index,
)


class EdgeOffloadEnv(ParallelEnv):
    """
    Multi-agent edge computing task offloading environment.

    Agents: 'device_0', 'device_1', ..., 'device_{M-1}'
    Each agent independently decides offload_ratio and target_es per slot.

    State (per agent, local observation):
        - Current task info (data_size, max_latency)
        - Device status (remaining energy)
        - ES queue estimates (recently known)

    Action (per agent):
        - offload_ratio: float in [0, 1] (0=all local, 1=all offload)
        - target_es: int in {0, 1, ..., E} (0=local only, 1..E=edge server)

    References:
        Ren et al. "ExplabOff: Towards Explorative and Collaborative
        Task Offloading via Mutual Information-Enhanced MARL", INFOCOM 2025.
    """

    metadata = {"render_modes": ["human"], "name": "edge_offload_v1"}

    def __init__(self,
                 num_devices: int = 5,
                 num_servers: int = 3,
                 max_slots: int = 100,
                 device_cpu: float = 1e9,
                 server_cpu: float = 7e9,
                 energy_budget: float = 100.0,
                 tx_power: float = 0.1,
                 bandwidth: float = 10e6,
                 noise_power: float = 1e-11,
                 energy_coeff: float = 1e-28,
                 cost_weight: float = 0.5,
                 seed: Optional[int] = None):
        """
        Initialize the environment.

        Args:
            num_devices: Number of mobile devices (M)
            num_servers: Number of edge servers (E)
            max_slots: Maximum time slots per episode
            device_cpu: MD CPU capacity (cycles/s)
            server_cpu: ES CPU capacity (cycles/s)
            energy_budget: Initial energy per MD (Joules)
            tx_power: Transmission power (Watts)
            bandwidth: System bandwidth (Hz)
            noise_power: Noise power (Watts)
            energy_coeff: CMOS energy coefficient ξ
            cost_weight: η for latency vs energy tradeoff
            seed: Random seed
        """
        super().__init__()
        self.M = num_devices
        self.E = num_servers
        self.max_slots = max_slots
        self.device_cpu = device_cpu
        self.server_cpu = server_cpu
        self.energy_budget = energy_budget
        self.tx_power = tx_power
        self.bandwidth = bandwidth
        self.noise_power = noise_power
        self.energy_coeff = energy_coeff
        self.cost_weight = cost_weight

        # Agent IDs
        self.possible_agents = [f"device_{i}" for i in range(self.M)]
        self.agents = self.possible_agents[:]

        # Observation: [data_size, max_latency, energy_ratio, es_queue_0, ..., es_queue_{E-1}]
        self.obs_dim = 3 + self.E

        # Per-agent spaces
        self._observation_spaces = {
            agent: spaces.Box(low=0.0, high=1.0, shape=(self.obs_dim,), dtype=np.float32)
            for agent in self.possible_agents
        }

        # Action: {offload_ratio: [0,1], target_es: {0..E}}
        self._action_spaces = {
            agent: spaces.Dict({
                "offload_ratio": spaces.Box(low=0.0, high=1.0, shape=(1,), dtype=np.float32),
                "target_es": spaces.Discrete(self.E + 1),  # 0 = local only
            })
            for agent in self.possible_agents
        }

        # Internal state
        self.md_devices: Dict[str, Device] = {}
        self.es_devices: List[Device] = []
        self.current_slot: int = 0
        self.task_counter: int = 0
        self.episode_metrics: dict = {}

        # Channel model state (Jakes + shadowing)
        self._channel_state: Dict[Tuple[int, int], float] = {}
        self._shadowing_map: Dict[Tuple[int, int], float] = {}

        if seed is not None:
            np.random.seed(seed)

    # ── PettingZoo API ──────────────────────────────────────────────

    @property
    def observation_spaces(self) -> Dict[str, spaces.Space]:
        return {a: self._observation_spaces[a] for a in self.agents}

    @property
    def action_spaces(self) -> Dict[str, spaces.Space]:
        return {a: self._action_spaces[a] for a in self.agents}

    def reset(self, seed=None, options=None) -> Tuple[Dict[str, np.ndarray], Dict]:
        """Reset the environment for a new episode."""
        if seed is not None:
            np.random.seed(seed)

        self.agents = self.possible_agents[:]
        self.current_slot = 0
        self.task_counter = 0

        # Initialize mobile devices
        self.md_devices = {}
        for i in range(self.M):
            agent_id = f"device_{i}"
            self.md_devices[agent_id] = Device(
                device_id=i,
                device_type="mobile",
                cpu_capacity=self.device_cpu,
                energy_budget=self.energy_budget,
                location=(np.random.uniform(0, 500), np.random.uniform(0, 500)),
            )

        # Initialize edge servers
        self.es_devices = [
            Device(
                device_id=self.M + e,
                device_type="edge",
                cpu_capacity=self.server_cpu,
                energy_budget=float("inf"),
                location=(np.random.uniform(0, 500), np.random.uniform(0, 500)),
            )
            for e in range(self.E)
        ]

        # Initialize channel state
        self._init_channel_state()

        # Generate initial tasks for each MD
        for agent_id in self.agents:
            task = self._generate_task()
            self.md_devices[agent_id].tasks_queue.append(task)

        # Episode metrics
        self.episode_metrics = {
            "total_tasks": 0,
            "completed_tasks": 0,
            "failed_tasks": 0,
            "total_energy": 0.0,
            "total_latency": 0.0,
        }

        return {a: self._get_obs(a) for a in self.agents}, {}

    def step(self, actions: Dict[str, Dict]) -> Tuple:
        """
        Execute one time slot with all agents' actions.

        Lifecycle per slot:
          1. Collect each MD's decision on its current task
          2. Compute edge transmission (with interference)
          3. Compute local computation
          4. Mark task completion → compute reward BEFORE popping
          5. Pop completed tasks and generate new ones
          6. Build observations for next slot

        Args:
            actions: {agent_id: {"offload_ratio": float, "target_es": int}}

        Returns:
            observations, rewards, terminations, truncations, infos
        """
        rewards = {}
        terminations = {a: False for a in self.agents}
        truncations = {a: False for a in self.agents}

        # ── Phase 1: Apply actions to current tasks ──
        offload_requests = []
        for agent_id in self.agents:
            md = self.md_devices[agent_id]
            if not md.tasks_queue:
                continue
            task = md.tasks_queue[0]
            action = actions.get(agent_id, {"offload_ratio": 0.0, "target_es": 0})
            offload_ratio = float(np.clip(np.asarray(action.get("offload_ratio", 0.0)).item(), 0.0, 1.0))
            target_es = int(action.get("target_es", 0))
            task.offload_ratio = offload_ratio
            task.target_es = None
            if offload_ratio > 0 and target_es > 0:
                es_idx = min(target_es - 1, self.E - 1)
                task.target_es = es_idx
                offload_data = task.data_size * offload_ratio
                offload_requests.append((agent_id, es_idx, offload_data, task))

        # ── Phase 2: Compute transmission (with interference) ──
        tx_rates = self._compute_transmission_rates(offload_requests)

        # ── Phase 3: Edge computation ──
        for agent_id, es_idx, offload_data, task in offload_requests:
            rate = tx_rates.get((agent_id, es_idx), 1e8)
            t_tx = calculate_transmission_energy(offload_data, 1.0, rate)  # dummy, just rate check
            # Edge latency: transmission + queuing + compute
            es = self.es_devices[es_idx]
            t_edge = calculate_edge_latency(
                data_bits=offload_data,
                cpu_per_bit=task.cpu_cycles_per_bit,
                server_cpu=es.cpu_capacity,
                tx_rate=rate,
                queue_load=es.queue_load,
                startup_time=0.0,
            )
            e_edge = calculate_transmission_energy(offload_data, self.tx_power, rate)
            task.latency = max(task.latency, t_edge)
            task.energy_consumed += e_edge
            self.md_devices[agent_id].current_energy -= e_edge
            # Add to ES queue (keeps reference for later cleanup)
            es.tasks_queue.append(task)

        # ── Phase 4: Local computation ──
        for agent_id in self.agents:
            md = self.md_devices[agent_id]
            if not md.tasks_queue:
                continue
            task = md.tasks_queue[0]
            local_ratio = 1.0 - task.offload_ratio
            if local_ratio > 0:
                local_data = task.data_size * local_ratio
                t_local = calculate_local_latency(local_data, task.cpu_cycles_per_bit, md.cpu_capacity)
                e_local = calculate_local_energy(local_data, task.cpu_cycles_per_bit, md.cpu_capacity, self.energy_coeff)
                task.latency = max(task.latency, t_local)
                task.energy_consumed += e_local
                md.current_energy -= e_local

        # ── Phase 5: Complete tasks & compute rewards (BEFORE popping) ──
        for agent_id in self.agents:
            md = self.md_devices[agent_id]
            if not md.tasks_queue:
                continue
            task = md.tasks_queue[0]
            task.completion_time = self.current_slot
            task.completed = True
            task.deadline_met = task.latency <= task.max_latency

            # Update episode-level metrics
            self.episode_metrics["total_tasks"] += 1
            if task.deadline_met:
                self.episode_metrics["completed_tasks"] += 1
            else:
                self.episode_metrics["failed_tasks"] += 1
            self.episode_metrics["total_energy"] += task.energy_consumed
            self.episode_metrics["total_latency"] += task.latency

            # ── Compute per-agent reward ──
            cost = self.cost_weight * task.latency + (1 - self.cost_weight) * task.energy_consumed
            energy_penalty = 10.0 if md.current_energy < 0 else 0.0
            deadline_penalty = 10.0 if not task.deadline_met else 0.0
            rewards[agent_id] = float(-cost - energy_penalty - deadline_penalty)

        # ── Phase 6: Cleanup (pop completed, remove from ES queues) ──
        for agent_id in self.agents:
            md = self.md_devices[agent_id]
            if not md.tasks_queue:
                continue
            task = md.tasks_queue[0]
            if task.target_es is not None:
                es = self.es_devices[task.target_es]
                if task in es.tasks_queue:
                    es.tasks_queue.remove(task)
            md.tasks_queue.pop(0)

        # ── Phase 7: Generate new tasks ──
        for agent_id in self.agents:
            if not self.md_devices[agent_id].tasks_queue:
                new_task = self._generate_task()
                new_task.arrival_time = self.current_slot
                self.md_devices[agent_id].tasks_queue.append(new_task)

        # ── Phase 8: Check termination ──
        all_energy_depleted = all(d.current_energy <= 0 for d in self.md_devices.values())
        self.current_slot += 1
        if self.current_slot >= self.max_slots or all_energy_depleted:
            terminations = {a: True for a in self.agents}
            truncations = {a: (self.current_slot >= self.max_slots) for a in self.agents}

        # ── Phase 9: Build observations for next step ──
        observations = {a: self._get_obs(a) for a in self.agents}
        infos = {a: {"energy_remaining": self.md_devices[a].current_energy,
                      "slot": self.current_slot} for a in self.agents}

        return observations, rewards, terminations, truncations, infos

    # ── Internal Methods ───────────────────────────────────────────

    def _generate_task(self) -> Task:
        """Generate a random task for an MD."""
        task = create_task(
            task_id=self.task_counter,
            data_range=(1e5, 1e6),       # 0.1-1 Mbits
            cpu_per_bit=1000.0,           # typical video task
            latency_range=(0.1, 0.5),     # 100-500 ms
        )
        self.task_counter += 1
        return task

    def _get_obs(self, agent_id: str) -> np.ndarray:
        """
        Build local observation for a specific MD.

        Observation vector:
            [0]: data_size (normalized)
            [1]: max_latency (normalized)
            [2]: energy_ratio (remaining/total)
            [3:3+E]: ES queue load estimates (normalized)
        """
        md = self.md_devices[agent_id]
        obs = np.zeros(self.obs_dim, dtype=np.float32)

        if md.tasks_queue:
            task = md.tasks_queue[0]
            obs[0] = min(task.data_size / 1e6, 1.0)       # normalize to ~1
            obs[1] = min(task.max_latency / 1.0, 1.0)
        obs[2] = max(md.current_energy / max(md.energy_budget, 1.0), 0.0)

        for e in range(self.E):
            es = self.es_devices[e]
            queue_ratio = min(es.queue_load / (es.cpu_capacity * 10), 1.0)
            obs[3 + e] = queue_ratio

        return obs

    def _init_channel_state(self):
        """Initialize Jakes fading and shadowing for all MD-ES pairs."""
        self._channel_state = {}
        self._shadowing_map = {}
        for agent_id in self.agents:
            i = int(agent_id.split("_")[1])
            for e in range(self.E):
                # Small-scale fading: initial value ~ Rayleigh
                self._channel_state[(i, e)] = np.random.rayleigh(scale=0.7)
                # Large-scale shadowing: log-normal (dB)
                self._shadowing_map[(i, e)] = np.random.lognormal(mean=0.0, sigma=8.0)

    def _update_channel(self, md_idx: int, es_idx: int):
        """
        Update channel gain using Jakes first-order Gauss-Markov model.
        g_{n+1} = α·g_n + √(1-α²)·w, where w ~ CN(0,1), α = J_0(2π·f_d·Δt)
        """
        alpha = 0.95  # correlation coefficient (simplified)
        prev_gain = self._channel_state[(md_idx, es_idx)]
        noise = np.random.rayleigh(scale=0.3)
        new_gain = alpha * prev_gain + np.sqrt(1 - alpha ** 2) * noise
        self._channel_state[(md_idx, es_idx)] = max(new_gain, 0.01)

    def _compute_transmission_rates(
        self, offload_requests: list
    ) -> Dict[Tuple[str, int], float]:
        """
        Compute Shannon rates for all offloading links with interference.

        For each MD-ES link, interference comes from all other MDs offloading
        to ANY server simultaneously—just as in the ExplabOff paper Eq.(3)-(4).

        Returns:
            Dictionary mapping (agent_id, es_idx) → rate in bps.
        """
        rates = {}

        for agent_id, es_idx, _, _ in offload_requests:
            md_idx = int(agent_id.split("_")[1])
            self._update_channel(md_idx, es_idx)

            # Small-scale fading gain
            fading_gain = self._channel_state[(md_idx, es_idx)]
            # Large-scale: path loss based on distance (simplified urban micro)
            md_pos = self.md_devices[agent_id].location
            es_pos = self.es_devices[es_idx].location
            distance = np.linalg.norm(np.array(md_pos) - np.array(es_pos))
            path_loss = 128.1 + 37.6 * np.log10(max(distance / 1000, 0.001))
            path_loss_linear = 10 ** (-path_loss / 10)
            # Shadowing
            shadow = self._shadowing_map[(md_idx, es_idx)]
            channel_gain = fading_gain * path_loss_linear * shadow

            # Interference from other offloading MDs
            interference = 0.0
            for other_agent, other_es, _, _ in offload_requests:
                if other_agent == agent_id:
                    continue
                other_idx = int(other_agent.split("_")[1])
                other_pos = self.md_devices[other_agent].location
                other_dist = np.linalg.norm(np.array(other_pos) - np.array(es_pos))
                other_pl = 128.1 + 37.6 * np.log10(max(other_dist / 1000, 0.001))
                other_pl_linear = 10 ** (-other_pl / 10)
                # Approximate: interference power from other MD to this ES
                interference += self.tx_power * other_pl_linear * 0.1

            rate = calculate_transmission_rate(
                self.tx_power, channel_gain, interference,
                self.bandwidth, self.noise_power
            )
            rates[(agent_id, es_idx)] = rate

        return rates

    # ── Rendering ───────────────────────────────────────────────────

    def render(self) -> None:
        """Simple text-based rendering of the environment state."""
        print(f"\n=== Slot {self.current_slot} ===")
        for agent_id in self.agents:
            md = self.md_devices[agent_id]
            task_info = ""
            if md.tasks_queue:
                t = md.tasks_queue[0]
                task_info = f"task=[{t.data_size/1e3:.0f}kb, ρ={t.offload_ratio:.2f}]"
            print(f"  {agent_id}: energy={md.current_energy:.1f}J {task_info}")
        for e in range(self.E):
            es = self.es_devices[e]
            print(f"  server_{e}: queue={len(es.tasks_queue)} load={es.queue_load/1e9:.2f}Gcyc")
        metrics = self.get_episode_metrics()
        if metrics:
            print(f"  completion_rate={metrics.get('completion_rate', 0):.2%}")
        print()

    def get_episode_metrics(self) -> Dict[str, float]:
        """Get current episode-level metrics."""
        total = max(self.episode_metrics.get("total_tasks", 1), 1)
        return {
            "completion_rate": self.episode_metrics["completed_tasks"] / total,
            "failure_rate": self.episode_metrics["failed_tasks"] / total,
            "avg_energy": self.episode_metrics["total_energy"] / total,
            "avg_latency": self.episode_metrics["total_latency"] / total,
        }
