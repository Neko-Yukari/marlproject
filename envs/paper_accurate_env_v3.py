"""
Paper-accurate MEC environment v3 — Multi-config + Random task sizes.
Strictly follows: Ren et al., "ExplabOff", INFOCOM 2025.

Key improvements over v2:
1. Multiple MEC configurations (2ES-3MD, 2ES-5MD, 3ES-7MD)
2. Random task size profiles per episode (paper mentions random task sizes)
3. Automatic constraint verification for generated profiles
4. Normalized observation to handle variable M/E
"""
import numpy as np
from typing import Dict, Tuple, List, Optional
from pettingzoo import ParallelEnv
from gymnasium import spaces
import copy


# ═══════════════════════════════════════════════════════════════════════
# Pre-computed valid task size profiles for each MEC configuration
# Each profile must satisfy: all-local fails, all-fastest-ES fails,
# optimal distribution passes.
# ═══════════════════════════════════════════════════════════════════════

# 2ES-3MD: ES [6, 12] GHz
# Condition for profile [s0, s1, s2]:
#   All local fail: each s_i > 1.11
#   All ES2 fail: 0.175*s2 + 0.075*(s0+s1) > 1.0
#   Optimal (s0,s1→ES2, s2→ES1): 0.175*s1+0.075*s0<=1 and 0.25*s2<=1
TASK_PROFILES_2ES_3MD = [
    [4.0, 3.5, 3.0],  # Base tuned
    [4.2, 3.3, 3.2],  # Variation 1
    [3.8, 3.7, 3.0],  # Variation 2
    [4.5, 3.0, 3.5],  # Variation 3
    [4.0, 3.0, 3.5],  # Variation 4
    [4.2, 3.5, 2.8],  # Variation 5
    [3.5, 4.0, 3.0],  # Variation 6
    [4.3, 3.2, 3.3],  # Variation 7
]

# 2ES-5MD: ES [15, 26] GHz
# Local: 1GHz, t_loc = 0.9*d, fail if d > 1.11
# ES2(26GHz): t_exe = d*9e8/26e9 = 0.0346*d
# ES1(15GHz): t_exe = d*9e8/15e9 = 0.06*d
# t_tx = 0.1*d (at 10Mbps)
# For all-ES2 fail with 5 MDs, last in queue:
#   0.1*s4 + 0.0346*(s0+s1+s2+s3+s4) > 1
# Optimal: 3→ES2, 2→ES1 usually
TASK_PROFILES_2ES_5MD = [
    [5.0, 4.5, 4.0, 3.5, 3.0],
    [5.5, 4.0, 4.5, 3.0, 3.5],
    [4.8, 5.0, 3.8, 4.0, 2.8],
    [5.2, 4.2, 4.2, 3.2, 3.2],
    [4.5, 5.0, 3.5, 4.5, 2.5],
]

# 3ES-7MD: ES [10, 19, 26] GHz
# t_exe(10GHz)=0.09d, t_exe(19GHz)=0.0474d, t_exe(26GHz)=0.0346d
# Max ~2 tasks per ES to stay under 1s deadline
TASK_PROFILES_3ES_7MD = [
    [5.5, 5.0, 4.5, 4.0, 3.5, 3.0, 2.5],
    [5.0, 5.5, 4.0, 4.5, 3.0, 3.5, 2.3],
    [5.3, 4.8, 4.7, 3.8, 3.8, 3.2, 2.4],
    [5.2, 5.3, 4.2, 4.3, 3.2, 3.3, 2.6],
    [4.8, 5.2, 4.3, 4.2, 3.3, 2.8, 3.0],
]

# Map MEC config to profiles
ALL_PROFILES = {
    (2, 3): TASK_PROFILES_2ES_3MD,
    (2, 5): TASK_PROFILES_2ES_5MD,
    (3, 7): TASK_PROFILES_3ES_7MD,
}

# ES CPU speeds from Table I (Hz)
ES_CPU_DB = {
    (2, 3): [6e9, 12e9],
    (2, 5): [15e9, 26e9],
    (3, 7): [10e9, 19e9, 26e9],
}


def verify_profile(es_cpus: List[float], sizes_mb: List[float],
                   tx_rate: float = 10e6, deadline: float = 1.0,
                   cpu_cycles_per_bit: int = 900, 
                   md_cpu: float = 1e9) -> bool:
    """Verify a task size profile satisfies constraints.
    
    Returns True if: all-local fails, all-best-ES fails, optimal passes.
    """
    M = len(sizes_mb)
    best_es = max(es_cpus)
    worst_es = min(es_cpus)
    
    # 1. All local must fail: each task alone > deadline
    for s in sizes_mb:
        t_loc = s * 1e6 * cpu_cycles_per_bit / md_cpu
        if t_loc <= deadline:
            return False
    
    # 2. All to best ES must fail
    total_texe = sum(s * 1e6 * cpu_cycles_per_bit / best_es for s in sizes_mb)
    t_last = sizes_mb[-1] * 1e6 / tx_rate + total_texe
    if t_last <= deadline:
        return False
    
    # 3. Check if optimal distribution exists (greedy: largest to fastest ES)
    # Sort sizes descending, assign to ES from fastest to slowest
    sorted_sizes = sorted(sizes_mb, reverse=True)
    # Simple greedy allocation: fill fastest ES first
    es_completion = [0.0] * len(es_cpus)  # time each ES finishes
    sorted_cpus = sorted(es_cpus, reverse=True)
    
    for s in sorted_sizes:
        t_tx = s * 1e6 / tx_rate
        t_exe = s * 1e6 * cpu_cycles_per_bit / sorted_cpus[0]  # fastest ES
        es_completion[0] += t_exe
        t_total = t_tx + es_completion[0]
    
    # If fastest ES can't handle first task alone, check if spreading helps
    # Simpler check: can all tasks complete if assigned to fastest ES?
    # This is a conservative check
    # Actually, let me just check if the greedy allocation works
    
    # Reset for proper greedy allocation
    es_time = [0.0] * len(es_cpus)
    
    for s in sorted_sizes:
        best_t = float('inf')
        best_e = 0
        for e_idx, cpu in enumerate(sorted_cpus):
            t_edge = s * 1e6 / tx_rate + es_time[e_idx] + s * 1e6 * cpu_cycles_per_bit / cpu
            if t_edge < best_t:
                best_t = t_edge
                best_e = e_idx
        if best_t > deadline:
            return False
        es_time[best_e] += s * 1e6 * cpu_cycles_per_bit / sorted_cpus[best_e]
    
    return True


class PaperAccurateEnvV3(ParallelEnv):
    """Multi-config MEC environment with random task size profiles.
    
    Supports: 2ES-3MD, 2ES-5MD, 3ES-7MD.
    Each episode randomly selects a task size profile from pre-verified set.
    """
    metadata = {"render_modes": ["human"], "name": "edge_offload_paper_v3"}
    
    # Paper parameters (Table I)
    SLOT_DURATION = 1.0
    MD_CPU = 1e9
    CPU_CYCLES_PER_BIT = 900
    DEADLINE = 1.0
    TX_POWER = 0.1
    ENERGY_COEFF = 1e-27
    BANDWIDTH = 10e6
    ETA = 0.5
    TASK_SIZE_STD = 0.1  # Small per-task variance
    
    def __init__(self, num_devices: int = 3, num_servers: int = 2,
                 randomize_profile: bool = True, seed: Optional[int] = None,
                 profile_noise: float = 0.05):  # ±5% noise per task
        super().__init__()
        self.M = num_devices
        self.E = num_servers
        self.randomize_profile = randomize_profile
        self.profile_noise = profile_noise
        
        # Get ES CPUs from database, fallback to linear scaling
        self.es_cpu_list = ES_CPU_DB.get((self.E, self.M), 
                                          [12e9 + i * 5e9 for i in range(self.E)])
        
        # Select task profiles for this config
        self._base_profiles = ALL_PROFILES.get((self.E, self.M), 
                                                self._generate_default_profiles())
        self._current_profile_idx = 0
        
        self.possible_agents = [f"device_{i}" for i in range(self.M)]
        self.agents = self.possible_agents[:]
        
        # Observation: [task_size_norm] + [es_load_norm_0, ..., es_load_norm_{E-1}]
        # + [es_cpu_norm_0, ..., es_cpu_norm_{E-1}]  (context)
        self.obs_dim = 1 + self.E + self.E
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
    
    # ── PettingZoo API: Per-agent spaces ──
    def observation_space(self, agent):
        """Return observation space for a single agent."""
        return self._observation_spaces[agent]
    
    def action_space(self, agent):
        """Return action space for a single agent."""
        return self._action_spaces[agent]
    
    # ── PettingZoo API: Global state ──
    def state(self):
        """Return global environment state (optional for decentralized training)."""
        # Return summary of current episode state
        if not hasattr(self, '_slot_tasks') or not self._slot_tasks:
            return None
        return {
            'task_sizes': {agent_id: task['size_mb'] 
                          for agent_id, task in self._slot_tasks.items()},
            'es_cpu': self.es_cpu_list,
            'slot': self.current_slot,
        }
    
    # ── PettingZoo API: Rendering ──
    def render(self):
        """Render environment state (text-based)."""
        if not hasattr(self, '_slot_tasks'):
            print("Environment not initialized. Call reset() first.")
            return
        print(f"[Render] Slot {self.current_slot}/{self.NUM_SLOTS}")
        for agent_id, task in sorted(self._slot_tasks.items()):
            md_idx = int(agent_id.split('_')[1])
            status = "DONE" if task['completed'] else "PENDING"
            print(f"  MD{md_idx}: {task['size_mb']:.1f}Mb | {status}")
    
    # ── PettingZoo API: Cleanup ──
    def close(self):
        """Clean up environment resources."""
        # Nothing to clean up for this environment
        pass
    
    def _generate_default_profiles(self) -> List[List[float]]:
        """Generate reasonable default profiles for unsupported configs."""
        # Scale: larger M needs smaller individual tasks
        base = max(2.0, 6.0 - self.M * 0.5)
        return [[base + (self.M - i) * 0.5 for i in range(self.M)]]
    
    @property
    def observation_spaces(self):
        return {a: self._observation_spaces[a] for a in self.agents}
    
    @property
    def action_spaces(self):
        return {a: self._action_spaces[a] for a in self.agents}
    
    def _pick_profile(self) -> List[float]:
        """Pick a task size profile for this episode."""
        idx = np.random.randint(0, len(self._base_profiles))
        self._current_profile_idx = idx
        base = self._base_profiles[idx]
        
        if self.profile_noise > 0:
            # Add small perturbation, clamp to reasonable range
            noisy = []
            for s in base:
                n = s + np.random.uniform(-self.profile_noise, self.profile_noise)
                n = max(0.5, min(10.0, n))
                noisy.append(n)
            return noisy
        return list(base)
    
    def reset(self, seed=None, options=None):
        if seed is not None:
            np.random.seed(seed)
        
        self.current_slot = 0
        self.agents = self.possible_agents[:]
        
        # Pick task size profile for this episode
        if self.randomize_profile:
            self._current_means = self._pick_profile()
        else:
            self._current_means = list(self._base_profiles[0])
        
        self.es_queue_counts = [0] * self.E
        self._slot_tasks = {}  # Cache tasks for current slot
        
        self.episode_metrics = {
            "total_tasks": 0, "completed_tasks": 0, "failed_tasks": 0,
            "total_cost": 0.0, "total_latency": 0.0, "total_energy": 0.0,
        }
        
        observations = {a: self._get_obs(a) for a in self.agents}
        infos = {a: {} for a in self.agents}
        return observations, infos
    
    def _generate_task(self, md_idx: int):
        """Generate a task with per-episode randomized mean size."""
        mean_mb = self._current_means[min(md_idx, len(self._current_means) - 1)]
        size_mb = max(0.1, np.random.normal(mean_mb, self.TASK_SIZE_STD))
        data_bits = size_mb * 1e6
        cycles = data_bits * self.CPU_CYCLES_PER_BIT
        return {"size_mb": size_mb, "data_bits": data_bits, "cycles": cycles}
    
    def _get_obs(self, agent_id: str) -> np.ndarray:
        obs = np.zeros(self.obs_dim, np.float32)
        md_idx = int(agent_id.split("_")[1])
        
        # Task size (normalized to max ~10Mb)
        # Use cached task if available, otherwise generate
        if agent_id not in self._slot_tasks:
            self._slot_tasks[agent_id] = self._generate_task(md_idx)
        task = self._slot_tasks[agent_id]
        obs[0] = min(task["size_mb"] / 10.0, 1.0)
        
        # ES load: fraction of tasks offloaded per ES
        total_tasks = max(sum(self.es_queue_counts), 1)
        base_es = 1
        for e in range(self.E):
            obs[base_es + e] = self.es_queue_counts[e] / max(total_tasks, 1)
        
        # ES CPU context (normalized to max ~30GHz)
        base_cpu = 1 + self.E
        max_cpu = 30e9
        for e in range(self.E):
            cpu = self._get_es_cpu(e)
            obs[base_cpu + e] = min(cpu / max_cpu, 1.0)
        
        return obs
    
    def compute_action_mask(self, agent_id: str) -> np.ndarray:
        """Compute action mask based on prior knowledge.
        
        Mask action if it's guaranteed to cause timeout.
        Returns: array of shape (action_dim,) with 1=valid, 0=invalid
        """
        md_idx = int(agent_id.split("_")[1])
        action_dim = self.E + 1  # local + ESs
        mask = np.ones(action_dim, dtype=np.float32)
        
        # Get cached task
        if agent_id not in self._slot_tasks:
            self._slot_tasks[agent_id] = self._generate_task(md_idx)
        task = self._slot_tasks[agent_id]
        
        # Check local execution
        t_loc = task["cycles"] / self.MD_CPU
        if t_loc > self.DEADLINE:
            mask[0] = 0  # Mask local
        
        # Check each ES
        for es_idx in range(self.E):
            es_cpu = self._get_es_cpu(es_idx)
            rate = self._compute_tx_rate(md_idx, es_idx)
            t_tx = task["data_bits"] / rate
            t_exe = task["cycles"] / es_cpu
            # Conservative: assume no wait (best case)
            t_edge_min = t_tx + t_exe
            if t_edge_min > self.DEADLINE:
                mask[es_idx + 1] = 0  # Mask this ES
        
        # Ensure at least one action is valid
        if mask.sum() == 0:
            # All masked: allow the fastest option
            mask[0] = 1  # Fallback to local
        
        return mask
    
    def _compute_tx_rate(self, md_idx: int, es_idx: int) -> float:
        return 10e6  # Fixed 10 Mbps
    
    def step(self, actions: Dict[str, int]) -> Tuple:
        rewards = {a: 0.0 for a in self.agents}
        terminations = {a: False for a in self.agents}
        truncations = {a: False for a in self.agents}
        
        # Generate tasks for this slot (cache for consistency)
        self._slot_tasks = {}
        for i, agent_id in enumerate(self.agents):
            self._slot_tasks[agent_id] = self._generate_task(i)
            self.episode_metrics["total_tasks"] += 1
        
        # Count offloading decisions
        self.es_queue_counts = [0] * self.E
        for agent_id in self.agents:
            action = actions.get(agent_id, 0)
            if action > 0:
                es_idx = min(action - 1, self.E - 1)
                self.es_queue_counts[es_idx] += 1
        
        # Process each MD's task (same logic as v2)
        for i, agent_id in enumerate(self.agents):
            task = self._slot_tasks[agent_id]
            action = actions.get(agent_id, 0)
            
            if action == 0:
                t_loc = task["cycles"] / self.MD_CPU
                t_edge = 0.0
                energy = self.ENERGY_COEFF * (self.MD_CPU ** 2) * task["cycles"]
            else:
                es_idx = min(action - 1, self.E - 1)
                es_cpu = self._get_es_cpu(es_idx)
                rate = self._compute_tx_rate(i, es_idx)
                t_tx = task["data_bits"] / rate
                t_exe = task["cycles"] / es_cpu
                
                # Compute waiting time from all tasks assigned to same ES
                t_wait = 0.0
                for j in range(self.M):
                    if j == i:
                        continue
                    other_agent = f"device_{j}"
                    other_action = actions.get(other_agent, 0)
                    if other_action > 0:
                        other_es = min(other_action - 1, self.E - 1)
                        if other_es == es_idx:
                            # Device with lower index executes first
                            if j < i:
                                t_wait += self._slot_tasks[other_agent]["cycles"] / es_cpu
                
                t_edge = t_tx + t_wait + t_exe
                t_loc = 0.0
                energy = self.TX_POWER * t_tx
            
            latency = max(t_loc, t_edge)
            cost = self.ETA * latency + (1 - self.ETA) * energy
            
            if latency <= self.DEADLINE:
                rewards[agent_id] = -cost
                self.episode_metrics["completed_tasks"] += 1
            else:
                rewards[agent_id] = -cost - 10.0
                self.episode_metrics["failed_tasks"] += 1
            
            self.episode_metrics["total_cost"] += cost
            self.episode_metrics["total_latency"] += latency
            self.episode_metrics["total_energy"] += energy
        
        self.current_slot += 1
        if self.current_slot >= 10:
            terminations = {a: True for a in self.agents}
            truncations = {a: True for a in self.agents}
        
        observations = {a: self._get_obs(a) for a in self.agents}
        infos = {a: {} for a in self.agents}
        
        return observations, rewards, terminations, truncations, infos
    
    def _get_es_cpu(self, es_idx: int) -> float:
        if es_idx < len(self.es_cpu_list):
            return self.es_cpu_list[es_idx]
        return 10e9
    
    def get_graph_data(self):
        """
        Return graph structure data for GNN processing.
        Uses PettingZoo's self.agents for dynamic agent enumeration.
        
        Enhanced features for better GNN learning:
        - MD: [task_size_norm, local_deadline_margin, relative_power, slot_progress]
        - ES: [cpu_norm, queue_fill_ratio, service_capacity, avg_wait_proxy]
        
        Returns:
            node_features: [num_md + num_es, 4]
            node_types: [num_md + num_es] - 0=MD, 1=ES
            edge_index: [2, num_md * num_es * 2] - fully connected bipartite
        """
        import torch
        
        # Use PettingZoo's agents list (dynamic)
        active_agents = self.agents
        num_md = len(active_agents)
        num_es = self.E
        num_nodes = num_md + num_es
        max_cpu = max(max(self.es_cpu_list), self.MD_CPU) if self.es_cpu_list else 30e9
        
        node_features = torch.zeros(num_nodes, 4)
        node_types = torch.zeros(num_nodes, dtype=torch.long)
        
        # MD nodes: [task_size_norm, local_deadline_margin, relative_power, slot_progress]
        for i, agent_id in enumerate(active_agents):
            md_idx = int(agent_id.split("_")[1])
            # Use cached task if available
            if agent_id in self._slot_tasks:
                task = self._slot_tasks[agent_id]
            else:
                task = self._generate_task(md_idx)
                self._slot_tasks[agent_id] = task
            
            # Compute local execution time and deadline margin
            t_loc = task["cycles"] / self.MD_CPU
            local_margin = (self.DEADLINE - t_loc) / self.DEADLINE
            
            node_features[i] = torch.tensor([
                task["size_mb"] / 10.0,              # Normalized task size
                local_margin,                         # Negative if local fails
                self.MD_CPU / max_cpu,                # Relative compute power
                float(self.current_slot) / 10.0      # Episode progress
            ])
            node_types[i] = 0  # MD
        
        # ES nodes: [cpu_norm, queue_fill_ratio, service_capacity, wait_proxy]
        max_es_cpu = max(self.es_cpu_list) if self.es_cpu_list else 30e9
        for j in range(num_es):
            es_cpu = self._get_es_cpu(j)
            # Queue fill ratio: how full is this ES relative to fair share
            fair_share = num_md / num_es if num_es > 0 else num_md
            queue_fill = self.es_queue_counts[j] / max(fair_share, 1.0)
            # Service capacity: relative to fastest ES
            service_cap = es_cpu / max_es_cpu if max_es_cpu > 0 else 1.0
            # Wait proxy: estimated wait based on queue length
            wait_proxy = min(self.es_queue_counts[j] * 0.2, 1.0)
            
            node_features[num_md + j] = torch.tensor([
                es_cpu / max_cpu,                    # Normalized CPU
                queue_fill,                          # Queue saturation
                service_cap,                         # Relative speed
                wait_proxy                           # Estimated congestion
            ])
            node_types[num_md + j] = 1  # ES
        
        # Edge index: fully connected bipartite (both directions)
        edge_list = []
        for i in range(num_md):
            for j in range(num_es):
                edge_list.append([i, num_md + j])  # MD -> ES
                edge_list.append([num_md + j, i])  # ES -> MD
        edge_index = torch.tensor(edge_list, dtype=torch.long).t()  # [2, num_edges]
        
        return node_features, node_types, edge_index
    
    def get_episode_metrics(self):
        total = max(self.episode_metrics["total_tasks"], 1)
        return {
            "completion_rate": self.episode_metrics["completed_tasks"] / total,
            "failure_rate": self.episode_metrics["failed_tasks"] / total,
            "avg_cost": self.episode_metrics["total_cost"] / total,
            "avg_latency": self.episode_metrics["total_latency"] / max(self.episode_metrics["completed_tasks"], 1),
            "avg_energy": self.episode_metrics["total_energy"] / total,
        }
    
    def get_config_info(self):
        return {
            "M": self.M, "E": self.E,
            "es_cpus": self.es_cpu_list,
            "profile": self._current_means,
            "profile_idx": self._current_profile_idx,
        }
