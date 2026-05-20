"""
PettingZoo ParallelEnv for MARL edge offloading — paper-accurate multi-slot execution.
Each MD processes f_m cycles/slot locally; each ES processes f_e cycles/slot for queued tasks.
Tasks span multiple slots until all cycles are processed.
"""
import numpy as np
from typing import Dict, List, Tuple, Optional
from pettingzoo import ParallelEnv
from gymnasium import spaces
from utils.task_device import Task, Device
from utils.helpers import create_task, calculate_transmission_rate, calculate_local_energy, calculate_transmission_energy, compute_fairness_index

class EdgeOffloadEnv(ParallelEnv):
    metadata = {"render_modes": ["human"], "name": "edge_offload_v2"}

    def __init__(self, num_devices=5, num_servers=3, max_slots=100,
                 device_cpu=1e9, server_cpu=7e9, energy_budget=500.0,
                 tx_power=0.1, bandwidth=10e6, noise_power=1e-11, energy_coeff=1e-28,
                 cost_weight=0.5, seed=None):
        super().__init__()
        self.M, self.E = num_devices, num_servers
        self.max_slots = max_slots
        self.device_cpu, self.server_cpu = device_cpu, server_cpu
        self.energy_budget = energy_budget
        self.tx_power, self.bandwidth, self.noise_power = tx_power, bandwidth, noise_power
        self.energy_coeff, self.cost_weight = energy_coeff, cost_weight

        self.possible_agents = [f"device_{i}" for i in range(self.M)]
        self.agents = self.possible_agents[:]
        self.obs_dim = 4 + self.E

        self._observation_spaces = {a: spaces.Box(0.0, 1.0, (self.obs_dim,), np.float32) for a in self.possible_agents}
        self._action_spaces = {a: spaces.Dict({
            "offload_ratio": spaces.Box(0.0, 1.0, (1,), np.float32),
            "target_es": spaces.Discrete(self.E + 1)
        }) for a in self.possible_agents}

        self._channel_state: Dict[Tuple[int,int], float] = {}
        self._shadowing_map: Dict[Tuple[int,int], float] = {}
        if seed is not None: np.random.seed(seed)

    @property
    def observation_spaces(self): return {a: self._observation_spaces[a] for a in self.agents}
    @property
    def action_spaces(self): return {a: self._action_spaces[a] for a in self.agents}

    # ═══════════════════════════════════════════════════════════════
    def reset(self, seed=None, options=None):
        if seed is not None: np.random.seed(seed)
        self.agents = self.possible_agents[:]
        self.current_slot = 0
        self.task_counter = 0

        # Clear all queues, reset devices
        self.md_devices = {f"device_{i}": Device(i, "mobile", self.device_cpu, self.energy_budget,
                           (np.random.uniform(0,500), np.random.uniform(0,500))) for i in range(self.M)}
        self.es_devices = [Device(self.M+e, "edge", self.server_cpu, float("inf"),
                           (np.random.uniform(0,500), np.random.uniform(0,500))) for e in range(self.E)]
        self._init_channel_state()

        self.episode_metrics = {"total_tasks":0, "completed_tasks":0, "failed_tasks":0,
                                "total_energy":0.0, "total_latency":0.0, "total_cost":0.0}
        return {a: self._get_obs(a) for a in self.agents}, {}

    # ═══════════════════════════════════════════════════════════════
    def step(self, actions: Dict[str, Dict]):
        rewards = {a: 0.0 for a in self.agents}
        terminations = {a: False for a in self.agents}
        truncations = {a: False for a in self.agents}

        # ── Phase 1: NEW task arrives for EVERY MD every slot (paper: j_m^n each slot) ──
        for agent_id in self.agents:
            task = self._generate_task()
            task.arrival_time = self.current_slot
            self.md_devices[agent_id].tasks_queue.append({
                'task': task, 'rem_local': task.total_cpu_cycles,
                'rem_edge': 0.0, 'target_es': None, 'submitted': False,
                'arrival': self.current_slot, 'edge_energy': 0.0
            })

        # ── Phase 2: Each MD decides offloading for its FRONT task (if not yet submitted) ──
        offload_requests = []
        for agent_id in self.agents:
            q = self.md_devices[agent_id].tasks_queue
            if not q: continue
            tdata = q[0]
            if tdata['submitted']: continue

            action = actions.get(agent_id, {"offload_ratio": [0.0], "target_es": 0})
            rho = float(np.clip(np.asarray(action.get("offload_ratio", [0.0])).item(), 0.0, 1.0))
            es_choice = int(action.get("target_es", 0))

            tdata['submitted'] = True
            if rho > 0 and es_choice > 0:
                es_idx = min(es_choice - 1, self.E - 1)
                tdata['target_es'] = es_idx
                total = tdata['task'].total_cpu_cycles
                tdata['rem_edge'] = total * rho
                tdata['rem_local'] = total * (1 - rho)
                offload_data = tdata['task'].data_size * rho
                offload_requests.append((agent_id, es_idx, offload_data, tdata))
            else:
                tdata['target_es'] = None
                tdata['rem_local'] = tdata['task'].total_cpu_cycles
                tdata['rem_edge'] = 0.0

        # ── Phase 3: Compute transmission energy ──
        tx_rates = self._compute_transmission_rates(offload_requests)
        for agent_id, es_idx, offload_data, tdata in offload_requests:
            rate = max(tx_rates.get((agent_id, es_idx), 1e8), 1e4)
            e_edge = calculate_transmission_energy(offload_data, self.tx_power, rate)
            tdata['edge_energy'] = e_edge
            self.md_devices[agent_id].current_energy -= e_edge
            self.es_devices[es_idx].tasks_queue.append(tdata)

        # ── Phase 4: Process local computation (f_m cycles across front tasks, round-robin) ──
        for agent_id in self.agents:
            md = self.md_devices[agent_id]
            remaining_cpu = md.cpu_capacity
            for tdata in md.tasks_queue:
                if remaining_cpu <= 0: break
                if tdata['rem_local'] <= 0: continue
                processed = min(tdata['rem_local'], remaining_cpu)
                tdata['rem_local'] -= processed
                remaining_cpu -= processed

        # ── Phase 5: Process edge computation (f_e cycles across ES queue, FIFO) ──
        for es in self.es_devices:
            remaining_cpu = es.cpu_capacity
            for tdata in list(es.tasks_queue):
                if remaining_cpu <= 0: break
                if tdata['rem_edge'] <= 0:
                    es.tasks_queue.remove(tdata)
                    continue
                processed = min(tdata['rem_edge'], remaining_cpu)
                tdata['rem_edge'] -= processed
                remaining_cpu -= processed

        # ── Phase 6: Complete tasks from MD queues ──
        for agent_id in self.agents:
            q = self.md_devices[agent_id].tasks_queue
            completed = []
            for tdata in q:
                if tdata['rem_local'] <= 0 and tdata['rem_edge'] <= 0:
                    task = tdata['task']
                    task.completion_time = self.current_slot
                    task.completed = True
                    elapsed = task.completion_time - task.arrival_time
                    task.deadline_met = elapsed <= task.max_latency
                    task.latency = elapsed

                    self.episode_metrics["total_tasks"] += 1
                    if task.deadline_met:
                        self.episode_metrics["completed_tasks"] += 1
                    else:
                        self.episode_metrics["failed_tasks"] += 1
                    self.episode_metrics["total_latency"] += elapsed

                    cost = self.cost_weight * elapsed + (1 - self.cost_weight) * tdata.get('edge_energy', 0)
                    self.episode_metrics["total_cost"] += cost
                    rewards[agent_id] = float(-cost - (10.0 if not task.deadline_met else 0.0))
                    completed.append(tdata)

                    # Remove from ES queue if present
                    if tdata['target_es'] is not None:
                        es = self.es_devices[tdata['target_es']]
                        if tdata in es.tasks_queue:
                            es.tasks_queue.remove(tdata)

            for tdata in completed:
                q.remove(tdata)

        # ── Phase 7: Termination ──
        self.current_slot += 1
        if self.current_slot >= self.max_slots:
            terminations = {a: True for a in self.agents}
            truncations = {a: True for a in self.agents}

        observations = {a: self._get_obs(a) for a in self.agents}
        infos = {a: {"energy": self.md_devices[a].current_energy} for a in self.agents}
        return observations, rewards, terminations, truncations, infos

    # ═══════════════════════════════════════════════════════════════
    def _generate_task(self):
        task = create_task(task_id=self.task_counter, data_range=(2e6, 7e6),
                          cpu_per_bit=1000.0, latency_range=(3.0, 8.0))  # tight: forces offloading of heavy tasks
        self.task_counter += 1
        return task

    def _get_obs(self, agent_id):
        md = self.md_devices[agent_id]
        obs = np.zeros(self.obs_dim, np.float32)
        q = md.tasks_queue
        if q:
            front = q[0]
            task = front['task']
            obs[0] = min(task.data_size / 7e6, 1.0)
            obs[1] = min(task.max_latency / 8.0, 1.0)
            total = sum(td['rem_local'] + td['rem_edge'] for td in q)
            obs[2] = min(total / (md.cpu_capacity * 20), 1.0)  # queue backlog
        obs[3] = max(md.current_energy / max(md.energy_budget, 1.0), 0.0)
        for e in range(self.E):
            es = self.es_devices[e]
            load = sum(td['rem_edge'] for td in es.tasks_queue)
            obs[4 + e] = min(load / (es.cpu_capacity * 10), 1.0)
        return obs

    def _init_channel_state(self):
        self._channel_state = {}
        self._shadowing_map = {}
        for i in range(self.M):
            for e in range(self.E):
                self._channel_state[(i, e)] = np.random.rayleigh(scale=0.7)
                self._shadowing_map[(i, e)] = np.random.lognormal(mean=0.0, sigma=8.0)

    def _update_channel(self, md_idx, es_idx):
        alpha = 0.95
        prev = self._channel_state[(md_idx, es_idx)]
        noise = np.random.rayleigh(scale=0.3)
        self._channel_state[(md_idx, es_idx)] = max(alpha * prev + np.sqrt(1 - alpha**2) * noise, 0.05)

    def _compute_transmission_rates(self, offload_requests):
        rates = {}
        for agent_id, es_idx, _, _ in offload_requests:
            md_idx = int(agent_id.split("_")[1])
            self._update_channel(md_idx, es_idx)
            fading = self._channel_state[(md_idx, es_idx)]
            md_pos = self.md_devices[agent_id].location
            es_pos = self.es_devices[es_idx].location
            dist = np.linalg.norm(np.array(md_pos) - np.array(es_pos))
            path_loss = 128.1 + 37.6 * np.log10(max(dist / 1000, 0.01))
            pl_linear = 10 ** (-path_loss / 10)
            channel_gain = fading * pl_linear * self._shadowing_map[(md_idx, es_idx)]
            interference = 0.0
            for other_a, other_es, _, _ in offload_requests:
                if other_a == agent_id: continue
                o_idx = int(other_a.split("_")[1])
                o_pos = self.md_devices[other_a].location
                o_dist = np.linalg.norm(np.array(o_pos) - np.array(es_pos))
                o_pl = 128.1 + 37.6 * np.log10(max(o_dist / 1000, 0.01))
                interference += self.tx_power * (10 ** (-o_pl / 10)) * 0.1
            rate = calculate_transmission_rate(self.tx_power, channel_gain, interference, self.bandwidth, self.noise_power)
            rates[(agent_id, es_idx)] = max(rate, 1e4)
        return rates

    def get_episode_metrics(self):
        total = max(self.episode_metrics["total_tasks"], 1)
        return {"completion_rate": self.episode_metrics["completed_tasks"] / total,
                "failure_rate": self.episode_metrics["failed_tasks"] / total,
                "avg_latency": self.episode_metrics["total_latency"] / total,
                "avg_energy": self.episode_metrics["total_energy"] / total,
                "avg_cost": self.episode_metrics["total_cost"] / total}

    def render(self):
        print(f"\n=== Slot {self.current_slot} ===")
        for i, a in enumerate(self.agents):
            tdata = self.md_tasks.get(a)
            status = f"rem={tdata['rem_local']/1e9:.1f}/{tdata['rem_edge']/1e9:.1f}G" if tdata else "idle"
            print(f"  {a}: {status} energy={self.md_devices[a].current_energy:.1f}J")
        for e in range(self.E):
            q = self.es_devices[e].tasks_queue
            print(f"  server_{e}: queue={len(q)} load={sum(td['rem_edge'] for td in q)/1e9:.1f}G")
        print()
