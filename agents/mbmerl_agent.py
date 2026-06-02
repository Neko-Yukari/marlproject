"""
MB-MERL: Model-Based Meta-RL for Edge Offloading.

Core idea: Learn a cost predictor that adapts to new task profiles in few episodes.
- CostPredictor: MLP(task_size, es_load, es_cpu) -> predicted_cost
- Meta-learning: MAML-style inner/outer loop
- Planning: Greedy action selection using adapted predictor
"""
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from typing import Dict, List, Tuple

class CostPredictor(nn.Module):
    """Predicts cost for (task_size, es_load, es_cpu) triple."""
    def __init__(self, hidden_dim: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(3, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )
    
    def forward(self, task_size: torch.Tensor, es_load: torch.Tensor, es_cpu: torch.Tensor) -> torch.Tensor:
        """
        Args:
            task_size: (B,) normalized task size
            es_load: (B,) normalized ES load (0-1)
            es_cpu: (B,) normalized ES CPU speed
        Returns:
            cost: (B,) predicted cost
        """
        x = torch.stack([task_size, es_load, es_cpu], dim=-1)
        return self.net(x).squeeze(-1)

class MBMERLAgent:
    """Meta-learned cost predictor with greedy planning."""
    def __init__(self, agent_id: int, hidden_dim: int = 64, 
                 meta_lr: float = 1e-3, inner_lr: float = 1e-2,
                 device: torch.device = torch.device("cpu")):
        self.agent_id = agent_id
        self.device = device
        self.meta_lr = meta_lr
        self.inner_lr = inner_lr
        
        # Meta-level model
        self.predictor = CostPredictor(hidden_dim).to(device)
        self.meta_optimizer = optim.Adam(self.predictor.parameters(), lr=meta_lr)
        
        # Adapted model (for current profile)
        self.adapted_predictor = CostPredictor(hidden_dim).to(device)
        self._adapted = False
        
        # Experience buffer for adaptation
        self.buffer: List[Tuple] = []  # (task_size, es_load, es_cpu, cost)
    
    def predict_cost(self, task_size: float, es_load: float, es_cpu: float, 
                     use_adapted: bool = True) -> float:
        """Predict cost for a single (task, ES) pair."""
        model = self.adapted_predictor if (use_adapted and self._adapted) else self.predictor
        with torch.no_grad():
            ts = torch.tensor([task_size], dtype=torch.float32, device=self.device)
            el = torch.tensor([es_load], dtype=torch.float32, device=self.device)
            ec = torch.tensor([es_cpu], dtype=torch.float32, device=self.device)
            cost = model(ts, el, ec).item()
        return cost
    
    def select_action(self, obs: np.ndarray, es_loads: List[float], 
                      es_cpus: List[float]) -> int:
        """
        Greedy action selection using adapted predictor.
        
        Args:
            obs: observation array [task_size, es_load_0, ..., es_cpu_0, ...]
            es_loads: list of ES loads (0-1)
            es_cpus: list of ES CPU speeds (normalized)
        Returns:
            action: 0=local, 1..E=ES
        """
        task_size = obs[0]  # First element is task size
        E = len(es_loads)
        
        costs = []
        # Local execution
        local_cost = self.predict_cost(task_size, 0.0, 0.0, use_adapted=True)
        costs.append(local_cost)
        
        # ES execution
        for e in range(E):
            cost = self.predict_cost(task_size, es_loads[e], es_cpus[e], use_adapted=True)
            costs.append(cost)
        
        return int(np.argmin(costs))
    
    def store_experience(self, task_size: float, es_load: float, es_cpu: float, 
                         actual_cost: float):
        """Store experience for adaptation."""
        self.buffer.append((task_size, es_load, es_cpu, actual_cost))
    
    def adapt(self, num_steps: int = 5, batch_size: int = 32):
        """
        MAML inner loop: adapt predictor to current profile.
        Called after collecting data on a new profile.
        """
        if len(self.buffer) < batch_size:
            return
        
        # Copy meta parameters to adapted model
        self.adapted_predictor.load_state_dict(self.predictor.state_dict())
        inner_opt = optim.SGD(self.adapted_predictor.parameters(), lr=self.inner_lr)
        
        for _ in range(num_steps):
            # Sample from buffer
            idx = np.random.choice(len(self.buffer), min(batch_size, len(self.buffer)), replace=False)
            batch = [self.buffer[i] for i in idx]
            
            ts = torch.tensor([x[0] for x in batch], dtype=torch.float32, device=self.device)
            el = torch.tensor([x[1] for x in batch], dtype=torch.float32, device=self.device)
            ec = torch.tensor([x[2] for x in batch], dtype=torch.float32, device=self.device)
            target = torch.tensor([x[3] for x in batch], dtype=torch.float32, device=self.device)
            
            pred = self.adapted_predictor(ts, el, ec)
            loss = nn.MSELoss()(pred, target)
            
            inner_opt.zero_grad()
            loss.backward()
            inner_opt.step()
        
        self._adapted = True
        self.buffer.clear()
    
    def meta_update(self, trajectories: List[Dict]):
        """
        MAML outer loop: update meta parameters using trajectories from multiple profiles.
        
        Args:
            trajectories: list of dicts with keys 'states', 'actions', 'costs'
        """
        total_loss = 0.0
        count = 0
        
        for traj in trajectories:
            if len(traj['states']) == 0:
                continue
            
            # Inner loop: adapt to this trajectory
            adapted = CostPredictor(self.predictor.net[0].out_features).to(self.device)
            adapted.load_state_dict(self.predictor.state_dict())
            inner_opt = optim.SGD(adapted.parameters(), lr=self.inner_lr)
            
            # Few gradient steps on this trajectory
            for _ in range(3):
                ts = torch.tensor(traj['task_sizes'], dtype=torch.float32, device=self.device)
                el = torch.tensor(traj['es_loads'], dtype=torch.float32, device=self.device)
                ec = torch.tensor(traj['es_cpus'], dtype=torch.float32, device=self.device)
                target = torch.tensor(traj['costs'], dtype=torch.float32, device=self.device)
                
                pred = adapted(ts, el, ec)
                loss = nn.MSELoss()(pred, target)
                
                inner_opt.zero_grad()
                loss.backward()
                inner_opt.step()
            
            # Outer loop: compute loss with adapted model, backprop to meta parameters
            # For simplicity, just update on the same data
            pred = self.predictor(ts, el, ec)
            meta_loss = nn.MSELoss()(pred, target)
            
            self.meta_optimizer.zero_grad()
            meta_loss.backward()
            self.meta_optimizer.step()
            
            total_loss += meta_loss.item()
            count += 1
        
        return total_loss / max(count, 1)
    
    def save(self, path: str):
        torch.save({
            'predictor': self.predictor.state_dict(),
            'meta_optimizer': self.meta_optimizer.state_dict(),
            'agent_id': self.agent_id
        }, path)
    
    def load(self, path: str):
        ckpt = torch.load(path, map_location=self.device, weights_only=False)
        self.predictor.load_state_dict(ckpt['predictor'])
        self.meta_optimizer.load_state_dict(ckpt['meta_optimizer'])
