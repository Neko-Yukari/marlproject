"""
Paper-accurate MEC environment for MARL edge offloading.

Strictly follows: Ren et al., "ExplabOff", INFOCOM 2025.

Key design:
- Each slot, each MD has at most ONE active task
- Task completed within the slot OR carried over to next slot
- Local and edge processing are PARALLEL (Eq.9)
- Cost = eta * latency + (1-eta) * energy (Eq.13)
"""
import numpy as np
from typing import Dict, Tuple
from pettingzoo import ParallelEnv
from gymnasium import spaces

class PaperAccurateEnv(ParallelEnv):
    """
    Paper-accurate environment matching Table I and Fig.4 parameters.
    
    Episode: 10 slots (N=10), each slot = 1 second.
    Each MD processes at most 1 task at a time.
    """
    metadata = {"render_modes": ["human"], "name": "edge_offload_paper_v1"}
    
    # ═══════════════════════════════════════════════════════════════════
    # Paper parameters (Table I + Fig.4)
    # ═══════════════════════════════════════════════════════════════════
    SLOT_DURATION = 1.0  # δ = 1 s
    MD_CPU = 1e9         # f^m = 1 GHz = 1×10⁹ cycles/s
    ES_CPU_LIST = {      # Fig.4: ES1=6GHz, ES2=12GHz for 2ES-3MD
        2: [6e9, 12e9],  # 2 ESs
        3: [10e9, 19e9, 26e9],  # 3 ESs (from text)
    }
    CPU_CYCLES_PER_BIT = 900  # c = 900 cycles/bit
    DEADLINE = 1.0       # t^max = 1 s = 1 slot
    TX_POWER = 0.1       # p^tran = 0.1 W
    ENERGY_COEFF = 1e-27 # ξ = 10⁻²⁷
    NOISE_POWER = 1e-10  # -100 dBm ≈ 10⁻¹⁰ W (for 1Hz, scaled)
    BANDWIDTH = 10e6     # B = 10 MHz (typical LTE)
    ETA = 0.5            # Time-energy weight (η, not specified, default 0.5)
    
    # Task sizes: MD0/MD1/MD2 mean task sizes in kb (from paper text)
    # Paper: "The task sizes of MDs 1-7 follow Gaussian distributions, 
    # with mean value being 7/6/3/4/5/4.5/5.5 kb and variance 1 kb"
    # For 2ES-3MD: MD1=7kb, MD2=6kb, MD3=3kb
    TASK_SIZE_KB_MEAN = [7.0, 6.0, 3.0]  # kb
    TASK_SIZE_KB_STD = 1.0                # variance 1 kb
    
    def __init__(self, num_devices=3, num_servers=2, seed=None):
        super().__init__()
        self.M = num_devices
        self.E = num_servers
        self.possible_agents = [f"device_{i}" for i in range(self.M)]
        self.agents = self.possible_agents[:]
        
        # Observation: [task_cycles_norm, deadline_norm, es_load_0, es_load_1, ...]
        self.obs_dim = 2 + self.E
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
        
        # Per-MD state: active task or None
        # Each MD has at most one task at a time
        self.md_tasks: Dict[str, dict] = {}  # agent_id -> task_info
        self.es_queues: Dict[int, list] = {e: [] for e in range(self.E)}  # ES queues
        
        # Episode metrics
        self.episode_metrics = {
            "total_tasks": 0,
            "completed_tasks": 0,
            "failed_tasks": 0,
            "total_cost": 0.0,
            "total_latency": 0.0,
            "total_energy": 0.0,
        }
        
        # Generate initial tasks for each MD
        for i, agent in enumerate(self.agents):
            self._generate_task_for_md(agent, i)
        
        observations = {a: self._get_obs(a) for a in self.agents}
        infos = {a: {} for a in self.agents}
        return observations, infos
    
    def _generate_task_for_md(self, agent_id: str, md_idx: int):
        """Generate a new task for MD.
        
        Paper: task sizes in kb, e.g., MD1=7kb, MD2=6kb, MD3=3kb
        Convert to cycles: cycles = data_bits * CPU_CYCLES_PER_BIT
        """
        if md_idx < len(self.TASK_SIZE_KB_MEAN):
            mean_kb = self.TASK_SIZE_KB_MEAN[md_idx]
        else:
            mean_kb = np.mean(self.TASK_SIZE_KB_MEAN[:self.M])
        
        # Generate task size in kb (Gaussian, min 1 kb)
        size_kb = max(1.0, np.random.normal(mean_kb, self.TASK_SIZE_KB_STD))
        
        # Convert to bits and then to CPU cycles
        data_bits = size_kb * 1000  # kb -> bits
        cycles = data_bits * self.CPU_CYCLES_PER_BIT
        
        self.md_tasks[agent_id] = {
            "cycles_total": cycles,
            "cycles_remaining_local": 0.0,
            "cycles_remaining_edge": 0.0,
            "data_bits": data_bits,
            "arrival_slot": self.current_slot + 1,  # Task arrives at NEXT slot
            "target_es": None,
            "completed": False,
            "failed": False,
            "energy_consumed": 0.0,
            "action_assigned": False,
        }
        self.episode_metrics["total_tasks"] += 1
    
    def _get_obs(self, agent_id: str) -> np.ndarray:
        """Get observation for agent."""
        obs = np.zeros(self.obs_dim, np.float32)
        
        task = self.md_tasks.get(agent_id)
        if task and not task["completed"] and not task["failed"]:
            # Normalize task cycles (0-1, max ~10×10⁶ for 7kb tasks)
            obs[0] = min(task["cycles_total"] / 10e6, 1.0)
            # Deadline (always 1 slot, normalized)
            obs[1] = 1.0
        
        # ES load: fraction of capacity used by queued tasks
        for e in range(self.E):
            queue_cycles = sum(t["cycles_remaining_edge"] for t in self.es_queues[e])
            es_cpu = self._get_es_cpu(e)
            obs[2 + e] = min(queue_cycles / es_cpu, 1.0)
        
        return obs
    
    def _get_es_cpu(self, es_idx: int) -> float:
        """Get CPU capacity for ES."""
        if self.E in self.ES_CPU_LIST:
            cpus = self.ES_CPU_LIST[self.E]
            if es_idx < len(cpus):
                return cpus[es_idx]
        # Default fallback
        return 10e9
    
    def step(self, actions: Dict[str, int]) -> Tuple:
        """
        Execute one slot.
        
        Phase 1: MDs make offloading decisions (actions)
        Phase 2: Process computation (local + edge in parallel)
        Phase 3: Check completions and compute rewards
        Phase 4: Generate new tasks for idle MDs
        """
        rewards = {a: 0.0 for a in self.agents}
        terminations = {a: False for a in self.agents}
        truncations = {a: False for a in self.agents}
        
        # ── Phase 1: Apply offloading decisions ──
        for agent_id in self.agents:
            task = self.md_tasks.get(agent_id)
            if not task or task["completed"] or task["failed"]:
                continue
            
            # Only apply decision if task is new (just generated this slot)
            # or if it's the first decision for this task
            # Check if task has been assigned an action before
            if task.get("action_assigned", False):
                # Task already has an ongoing decision, skip re-assignment
                continue
            
            action = actions.get(agent_id, 0)
            task["action_assigned"] = True
            
            if action == 0:
                # Local processing
                task["target_es"] = None
                task["cycles_remaining_edge"] = 0.0
                task["cycles_remaining_local"] = task["cycles_total"]
            else:
                # Offload to ES
                es_idx = min(action - 1, self.E - 1)
                task["target_es"] = es_idx
                task["cycles_remaining_edge"] = task["cycles_total"]
                task["cycles_remaining_local"] = 0.0
                self.es_queues[es_idx].append(task)
        
        # ── Phase 2: Process computation (parallel) ──
        # Local processing
        for agent_id in self.agents:
            task = self.md_tasks.get(agent_id)
            if not task or task["completed"] or task["failed"]:
                continue
            if task["target_es"] is None:
                # Process locally
                processed = min(task["cycles_remaining_local"], self.MD_CPU * self.SLOT_DURATION)
                task["cycles_remaining_local"] -= processed
                # Local energy: e_loc = ξ * (f^m)² * cycles
                energy_local = self.ENERGY_COEFF * (self.MD_CPU ** 2) * processed
                task["energy_consumed"] += energy_local
        
        # Edge processing (share ES CPU among queued tasks)
        for es_idx in range(self.E):
            queue = self.es_queues[es_idx]
            if not queue:
                continue
            
            es_cpu = self._get_es_cpu(es_idx)
            # Fair sharing: each task gets CPU / num_tasks
            cpu_per_task = es_cpu * self.SLOT_DURATION / len(queue)
            
            for task in list(queue):
                processed = min(task["cycles_remaining_edge"], cpu_per_task)
                task["cycles_remaining_edge"] -= processed
                
                # Edge energy: transmission + computation at ES
                # Simplified: we count transmission energy once
                if task["cycles_remaining_edge"] <= 0:
                    # Task completed at edge
                    task_data = task["data_bits"]
                    # Transmission rate (simplified Shannon)
                    rate = self.BANDWIDTH * np.log2(1 + self.TX_POWER / self.NOISE_POWER)
                    tx_time = task_data / max(rate, 1e6)
                    tx_energy = self.TX_POWER * tx_time
                    task["energy_consumed"] += tx_energy
        
        # ── Phase 3: Check completions and compute costs ──
        slot_cost_sum = 0.0
        active_count = 0
        
        for agent_id in self.agents:
            task = self.md_tasks.get(agent_id)
            if not task or task["completed"] or task["failed"]:
                continue
            
            active_count += 1
            
            # Check if task completed
            if task["cycles_remaining_local"] <= 0 and task["cycles_remaining_edge"] <= 0:
                task["completed"] = True
                latency = self.current_slot + 1 - task["arrival_slot"]  # Slots elapsed
                
                # Cost = η * latency + (1-η) * energy
                cost = self.ETA * latency + (1 - self.ETA) * task["energy_consumed"]
                slot_cost_sum += cost
                
                self.episode_metrics["total_cost"] += cost
                self.episode_metrics["total_latency"] += latency
                self.episode_metrics["total_energy"] += task["energy_consumed"]
                
                if latency <= self.DEADLINE:
                    # Success: reward inversely correlated with cost
                    rewards[agent_id] = -cost
                    self.episode_metrics["completed_tasks"] += 1
                else:
                    # Failure: penalty
                    rewards[agent_id] = -cost - 10.0
                    self.episode_metrics["failed_tasks"] += 1
                
                # Remove from ES queue if present
                if task["target_es"] is not None:
                    es_idx = task["target_es"]
                    if task in self.es_queues[es_idx]:
                        self.es_queues[es_idx].remove(task)
            else:
                # Task not completed, continue to next slot
                # Small negative reward for delay
                rewards[agent_id] = -0.1
        
        # ── Phase 4: Generate new tasks for idle MDs ──
        # Paper: each slot each MD has a task jm_n
        # Generate new task for any MD that doesn't have an active task
        for i, agent_id in enumerate(self.agents):
            task = self.md_tasks.get(agent_id)
            if not task or task["completed"] or task["failed"]:
                self._generate_task_for_md(agent_id, i)
        
        # ── Phase 5: Check episode end ──
        self.current_slot += 1
        if self.current_slot >= 10:  # N = 10 slots per episode
            terminations = {a: True for a in self.agents}
            truncations = {a: True for a in self.agents}
        
        observations = {a: self._get_obs(a) for a in self.agents}
        infos = {a: {} for a in self.agents}
        
        return observations, rewards, terminations, truncations, infos
    
    def get_episode_metrics(self):
        total = max(self.episode_metrics["total_tasks"], 1)
        return {
            "completion_rate": self.episode_metrics["completed_tasks"] / total,
            "failure_rate": self.episode_metrics["failed_tasks"] / total,
            "avg_cost": self.episode_metrics["total_cost"] / max(total, 1),
            "avg_latency": self.episode_metrics["total_latency"] / max(self.episode_metrics["completed_tasks"], 1),
            "avg_energy": self.episode_metrics["total_energy"] / max(total, 1),
        }
