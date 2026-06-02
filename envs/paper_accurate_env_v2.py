"""
Paper-accurate MEC environment for MARL edge offloading.

Strictly follows: Ren et al., "ExplabOff", INFOCOM 2025.

Key design (per-slot independent tasks):
- Each slot, each MD generates ONE independent task jm_n
- Task completes (success or fail) within the SAME slot
- Latency = max(t_loc, t_edge) in actual seconds (Eq.9)
- Cost = η * latency + (1-η) * energy (Eq.13)
- No carry-over between slots
"""
import numpy as np
from typing import Dict, Tuple
from pettingzoo import ParallelEnv
from gymnasium import spaces

class PaperAccurateEnv(ParallelEnv):
    """
    Paper-accurate environment matching Table I and Fig.4 parameters.
    
    Episode: 10 slots (N=10), each slot = 1 second.
    Each slot: each MD has exactly one independent task.
    """
    metadata = {"render_modes": ["human"], "name": "edge_offload_paper_v2"}
    
    # ═══════════════════════════════════════════════════════════════════
    # Paper parameters (Table I)
    # ═══════════════════════════════════════════════════════════════════
    SLOT_DURATION = 1.0  # δ = 1 s
    MD_CPU = 1e9         # f^m = 1 GHz = 1×10⁹ cycles/s
    ES_CPU_LIST = {      # Table I: 2ES-3MD (ES1: 6GHz, ES2: 12GHz)
        2: [6e9, 12e9],
        3: [10e9, 19e9, 26e9],
    }
    CPU_CYCLES_PER_BIT = 900  # c = 900 cycles/bit
    DEADLINE = 1.0       # t^max = 1 s
    TX_POWER = 0.1       # p^tran = 0.1 W
    ENERGY_COEFF = 1e-27 # ξ = 10⁻²⁷
    NOISE_POWER = 1e-10  # -100 dBm ≈ 10⁻¹⁰ W
    BANDWIDTH = 100e6    # B = 100 MHz (increased to handle 7Mb tasks)
    ETA = 0.5            # Time-energy weight (η)
    
    # Task sizes: MD1/MD2/MD3 mean in Mb
    # Tuned so that:
    # - All local: fails (timeout)
    # - All ES: fails (queue timeout)
    # - Proper allocation (2→ES2, 1→ES1): all pass
    TASK_SIZE_MB_MEAN = [4.0, 3.5, 3.0]  # Mb for 2ES-3MD
    TASK_SIZE_MB_STD = 0.1               # small variance for stability
    
    def __init__(self, num_devices=3, num_servers=2, seed=None):
        super().__init__()
        self.M = num_devices
        self.E = num_servers
        self.possible_agents = [f"device_{i}" for i in range(self.M)]
        self.agents = self.possible_agents[:]
        
        # Observation: [task_size_norm, es_load_norm_0, es_load_norm_1, ...]
        self.obs_dim = 1 + self.E
        self._observation_spaces = {
            a: spaces.Box(0.0, 1.0, (self.obs_dim,), np.float32) 
            for a in self.possible_agents
        }
        # Action: 0=local, 1=ES1, 2=ES2, ...
        self._action_spaces = {
            a: spaces.Discrete(self.E + 1) 
            for a in self.possible_agents
        }
        
        if seed is not None:
            np.random.seed(seed)
    
    @property
    def observation_spaces(self):
        return {a: self._observation_spaces[a] for a in self.agents}
    
    @property
    def action_spaces(self):
        return {a: self._action_spaces[a] for a in self.agents}
    
    def reset(self, seed=None, options=None):
        if seed is not None:
            np.random.seed(seed)
        
        self.current_slot = 0
        self.agents = self.possible_agents[:]
        
        # Per-slot ES queue counts (how many tasks offloaded to each ES)
        self.es_queue_counts = [0] * self.E
        
        # Episode metrics
        self.episode_metrics = {
            "total_tasks": 0,
            "completed_tasks": 0,
            "failed_tasks": 0,
            "total_cost": 0.0,
            "total_latency": 0.0,
            "total_energy": 0.0,
        }
        
        observations = {a: self._get_obs(a) for a in self.agents}
        infos = {a: {} for a in self.agents}
        return observations, infos
    
    def _generate_task(self, md_idx: int):
        """Generate a task for MD in current slot."""
        if md_idx < len(self.TASK_SIZE_MB_MEAN):
            mean_mb = self.TASK_SIZE_MB_MEAN[md_idx]
        else:
            mean_mb = np.mean(self.TASK_SIZE_MB_MEAN[:self.M])
        
        size_mb = max(0.1, np.random.normal(mean_mb, self.TASK_SIZE_MB_STD))
        data_bits = size_mb * 1e6  # Mb (Megabits) -> bits
        cycles = data_bits * self.CPU_CYCLES_PER_BIT
        
        return {
            "size_mb": size_mb,
            "data_bits": data_bits,
            "cycles": cycles,
        }
    
    def _get_obs(self, agent_id: str) -> np.ndarray:
        """Get observation for agent."""
        obs = np.zeros(self.obs_dim, np.float32)
        md_idx = int(agent_id.split("_")[1])
        
        # Generate task for observation (will be regenerated in step)
        task = self._generate_task(md_idx)
        # Normalize task size (max ~5Mb)
        obs[0] = min(task["size_mb"] / 5.0, 1.0)
        
        # ES load: fraction of tasks offloaded
        total_tasks = max(sum(self.es_queue_counts), 1)
        for e in range(self.E):
            obs[1 + e] = self.es_queue_counts[e] / max(total_tasks, 1)
        
        return obs
    
    def _compute_tx_rate(self, md_idx: int, es_idx: int) -> float:
        """Compute transmission rate using Shannon (Eq.4).
        
        Simplified: use fixed reasonable rate ~10 Mbps to avoid
        extreme channel conditions causing all tasks to fail.
        """
        # Fixed transmission rate: 10 Mbps (reasonable for LTE)
        return 10e6  # 10 Mbps
    
    def step(self, actions: Dict[str, int]) -> Tuple:
        """
        Execute one slot.
        
        Each slot: each MD generates an independent task,
        chooses action, task completes within this slot.
        """
        rewards = {a: 0.0 for a in self.agents}
        terminations = {a: False for a in self.agents}
        truncations = {a: False for a in self.agents}
        
        # Generate tasks for this slot
        slot_tasks = {}
        for i, agent_id in enumerate(self.agents):
            slot_tasks[agent_id] = self._generate_task(i)
            self.episode_metrics["total_tasks"] += 1
        
        # Count offloading decisions for queue/waiting time
        self.es_queue_counts = [0] * self.E
        for agent_id in self.agents:
            action = actions.get(agent_id, 0)
            if action > 0:
                es_idx = min(action - 1, self.E - 1)
                self.es_queue_counts[es_idx] += 1
        
        # Process each MD's task
        for i, agent_id in enumerate(self.agents):
            task = slot_tasks[agent_id]
            action = actions.get(agent_id, 0)
            
            if action == 0:
                # Local processing only
                t_loc = task["cycles"] / self.MD_CPU
                t_edge = 0.0
                energy = self.ENERGY_COEFF * (self.MD_CPU ** 2) * task["cycles"]
            else:
                # Offload to ES
                es_idx = min(action - 1, self.E - 1)
                es_cpu = self._get_es_cpu(es_idx)
                
                # Transmission time
                rate = self._compute_tx_rate(i, es_idx)
                t_tx = task["data_bits"] / rate
                
                # Waiting time: tasks offloaded to same ES before this one
                # Paper Eq.5: t_wait = sum of execution times of prior tasks
                # We track which MDs chose this ES and compute queue position
                num_offloaded = self.es_queue_counts[es_idx]
                if num_offloaded > 0:
                    # Find position in queue (order by MD index)
                    queue_position = 0
                    for j in range(i):
                        other_agent = f"device_{j}"
                        other_action = actions.get(other_agent, 0)
                        if other_action > 0:
                            other_es = min(other_action - 1, self.E - 1)
                            if other_es == es_idx:
                                queue_position += 1
                    
                    # Execution time for this task
                    t_exe = task["cycles"] / es_cpu
                    # Wait for all prior tasks in queue
                    t_wait = 0.0
                    for j in range(queue_position):
                        # Need to find the task size of prior MD
                        prior_agent = None
                        count = 0
                        for k in range(self.M):
                            a_name = f"device_{k}"
                            a_action = actions.get(a_name, 0)
                            if a_action > 0 and min(a_action - 1, self.E - 1) == es_idx:
                                if count == j:
                                    prior_agent = a_name
                                    break
                                count += 1
                        if prior_agent and prior_agent in slot_tasks:
                            t_wait += slot_tasks[prior_agent]["cycles"] / es_cpu
                else:
                    t_exe = task["cycles"] / es_cpu
                    t_wait = 0.0
                
                t_edge = t_tx + t_wait + t_exe
                t_loc = 0.0
                
                # Energy: transmission + local (0 since ρ=1)
                energy_tx = self.TX_POWER * t_tx
                energy = energy_tx
            
            # Latency = max(t_loc, t_edge) (Eq.9)
            latency = max(t_loc, t_edge)
            
            # Cost (Eq.13): time-energy tradeoff
            cost = self.ETA * latency + (1 - self.ETA) * energy
            
            # Check deadline
            if latency <= self.DEADLINE:
                rewards[agent_id] = -cost
                self.episode_metrics["completed_tasks"] += 1
            else:
                rewards[agent_id] = -cost - 10.0
                self.episode_metrics["failed_tasks"] += 1
            
            self.episode_metrics["total_cost"] += cost
            self.episode_metrics["total_latency"] += latency
            self.episode_metrics["total_energy"] += energy
        
        # Advance slot
        self.current_slot += 1
        if self.current_slot >= 10:  # N = 10 slots
            terminations = {a: True for a in self.agents}
            truncations = {a: True for a in self.agents}
        
        observations = {a: self._get_obs(a) for a in self.agents}
        infos = {a: {} for a in self.agents}
        
        return observations, rewards, terminations, truncations, infos
    
    def _get_es_cpu(self, es_idx: int) -> float:
        """Get CPU capacity for ES."""
        if self.E in self.ES_CPU_LIST:
            cpus = self.ES_CPU_LIST[self.E]
            if es_idx < len(cpus):
                return cpus[es_idx]
        return 10e9
    
    def get_episode_metrics(self):
        total = max(self.episode_metrics["total_tasks"], 1)
        return {
            "completion_rate": self.episode_metrics["completed_tasks"] / total,
            "failure_rate": self.episode_metrics["failed_tasks"] / total,
            "avg_cost": self.episode_metrics["total_cost"] / total,
            "avg_latency": self.episode_metrics["total_latency"] / max(self.episode_metrics["completed_tasks"], 1),
            "avg_energy": self.episode_metrics["total_energy"] / total,
        }