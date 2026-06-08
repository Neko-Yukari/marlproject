"""
Unified Training Interface for MARL Edge Offloading.

Usage:
    python train_unified.py --config configs/ippo_2es3md.yaml
    python train_unified.py --config configs/explaboff_3es7md.yaml
    python train_unified.py --config configs/hypernetwork.yaml
"""
import sys; sys.path.insert(0, '.')
import argparse
import yaml
import json
import time
import torch
import numpy as np
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional, Tuple, Any, Union

from envs.paper_accurate_env_v3 import PaperAccurateEnvV3


class UnifiedTrainer:
    """Unified trainer supporting multiple algorithms and configurations."""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Extract config sections
        self.env_cfg = config['environment']
        self.algo_cfg = config['algorithm']
        self.train_cfg = config['training']
        
        # Create environment
        self.env = self._create_env()
        
        # Create agents
        self.agents = self._create_agents()
        
        # Training state
        self.episode = 0
        self.history: List[Dict[str, Any]] = []
        self.best_cost = float('inf')
        
    def _create_env(self) -> PaperAccurateEnvV3:
        """Create environment from config."""
        return PaperAccurateEnvV3(
            num_devices=self.env_cfg['num_md'],
            num_servers=self.env_cfg['num_es'],
            randomize_profile=self.env_cfg.get('randomize_profile', True),
            profile_noise=self.env_cfg.get('profile_noise', 0.05)
        )
    
    def _create_agents(self) -> Dict[str, Any]:
        """Create agents based on algorithm type."""
        algo_type = self.algo_cfg['type']
        M = self.env_cfg['num_md']
        E = self.env_cfg['num_es']
        obs_dim = self.env.obs_dim
        action_dim = E + 1
        
        if algo_type == 'IPPO':
            from agents.ippo_agent import IPPOAgent
            return {
                'type': 'IPPO',
                'agents': [
                    IPPOAgent(
                        agent_id=i,
                        state_dim=obs_dim,
                        action_dim=action_dim,
                        hidden_dim=self.algo_cfg.get('hidden_dim', 128),
                        learning_rate=self.algo_cfg.get('lr', 5e-5),
                        gamma=self.algo_cfg.get('gamma', 0.99),
                        gae_lambda=self.algo_cfg.get('gae_lambda', 0.95),
                        clip_ratio=self.algo_cfg.get('clip_ratio', 0.2),
                        entropy_coeff=self.algo_cfg.get('entropy_coeff', 0.01),
                        device=self.device
                    )
                    for i in range(M)
                ]
            }
        
        elif algo_type == 'ExplabOff':
            from agents.explaboff_agent import ExplabOffAgent
            return {
                'type': 'ExplabOff',
                'agents': [
                    ExplabOffAgent(
                        agent_id=i,
                        state_dim=obs_dim,
                        action_dim=action_dim,
                        hidden_dim=self.algo_cfg.get('hidden_dim', 128),
                        lr=self.algo_cfg.get('lr', 5e-5),
                        mi_mu=self.algo_cfg.get('mi_mu', 0.01),
                        mi_nu=self.algo_cfg.get('mi_nu', 0.01),
                        device=self.device
                    )
                    for i in range(M)
                ]
            }
        
        elif algo_type == 'HyperNetwork':
            from agents.hypernetwork import HyperNetwork
            from agents.hypernetwork_variants import CrossScaleAgent
            
            hyper_net = HyperNetwork(
                obs_dim=self.algo_cfg.get('obs_dim', 7),
                max_action_dim=self.algo_cfg.get('max_action_dim', 4),
                hidden_dim=self.algo_cfg.get('hidden_dim', 256)
            ).to(self.device)
            
            return {
                'type': 'HyperNetwork',
                'network': hyper_net,
                'agents': [
                    CrossScaleAgent(
                        agent_id=i,
                        hyper_net=hyper_net,
                        device=self.device,
                        lr=self.algo_cfg.get('lr', 1e-6),
                        update_interval=self.algo_cfg.get('update_interval', 10)
                    )
                    for i in range(M)
                ]
            }
        
        elif algo_type == 'Baseline':
            return {
                'type': 'Baseline',
                'strategy': self.algo_cfg['strategy']
            }
        
        else:
            raise ValueError(f"Unknown algorithm type: {algo_type}")
    
    def _select_actions(self, obs: Dict[str, np.ndarray]) -> Dict[str, int]:
        """Select actions for all agents."""
        algo_type = self.agents['type']
        actions: Dict[str, int] = {}
        
        if algo_type in ['IPPO', 'ExplabOff']:
            agent_list = self.agents['agents']
            for i, agent_id in enumerate(self.env.agents):
                a, _, _ = agent_list[i].select_action(obs[agent_id])
                actions[agent_id] = a
        
        elif algo_type == 'HyperNetwork':
            agent_list = self.agents['agents']
            M = self.env_cfg['num_md']
            E = self.env_cfg['num_es']
            for i, agent_id in enumerate(self.env.agents):
                a = agent_list[i].select_action(obs[agent_id], M, E)
                actions[agent_id] = a
        
        elif algo_type == 'Baseline':
            strategy = self.agents['strategy']
            task_sizes = self.env._current_means
            es_cpus = self.env.es_cpu_list
            
            if strategy == 'Random':
                actions_list = [np.random.randint(0, len(es_cpus) + 1) for _ in range(len(task_sizes))]
            elif strategy == 'All_Local':
                actions_list = [0] * len(task_sizes)
            elif strategy == 'All_BestES':
                actions_list = [len(es_cpus)] * len(task_sizes)
            elif strategy == 'Greedy':
                actions_list = self._greedy_actions(task_sizes, es_cpus)
            elif strategy == 'Size_Based':
                actions_list = self._size_based_actions(task_sizes, es_cpus)
            else:
                raise ValueError(f"Unknown baseline strategy: {strategy}")
            
            actions = {f"device_{i}": a for i, a in enumerate(actions_list)}
        
        return actions
    
    def _greedy_actions(self, task_sizes, es_cpus):
        """Greedy allocation."""
        tx_rate = self.env.BANDWIDTH
        cpu_cycles = self.env.CPU_CYCLES_PER_BIT
        deadline = self.env.DEADLINE
        
        actions = []
        es_time = [0.0] * len(es_cpus)
        
        for s in task_sizes:
            best_time = float('inf')
            best_action = 0
            
            # Check local
            t_loc = s * 1e6 * cpu_cycles / self.env.MD_CPU
            if t_loc <= deadline:
                best_time = t_loc
                best_action = 0
            
            # Check each ES
            for es_idx, cpu in enumerate(es_cpus):
                t_tx = s * 1e6 / tx_rate
                t_exe = s * 1e6 * cpu_cycles / cpu
                t_edge = t_tx + es_time[es_idx] + t_exe
                if t_edge < best_time and t_edge <= deadline:
                    best_time = t_edge
                    best_action = es_idx + 1
            
            if best_action > 0:
                es_time[best_action - 1] += s * 1e6 * cpu_cycles / es_cpus[best_action - 1]
            
            actions.append(best_action)
        
        return actions
    
    def _size_based_actions(self, task_sizes, es_cpus):
        """Size-based allocation."""
        sorted_idx = np.argsort(task_sizes)[::-1]
        sorted_cpus = sorted([(c, i) for i, c in enumerate(es_cpus)], key=lambda x: x[0], reverse=True)
        es_time = [0.0] * len(es_cpus)
        tx_rate = self.env.BANDWIDTH
        cpu_cycles = self.env.CPU_CYCLES_PER_BIT
        
        assignments = [0] * len(task_sizes)
        for idx in sorted_idx:
            s = task_sizes[idx]
            best_time = float('inf')
            best_es = 0
            
            for es_i, (cpu, _) in enumerate(sorted_cpus):
                t_edge = s * 1e6 / tx_rate + es_time[es_i] + s * 1e6 * cpu_cycles / cpu
                if t_edge < best_time:
                    best_time = t_edge
                    best_es = es_i
            
            es_time[best_es] += task_sizes[idx] * 1e6 * cpu_cycles / sorted_cpus[best_es][0]
            assignments[idx] = best_es + 1
        
        return assignments
    
    def _store_transitions(self, obs, actions, rewards, dones):
        """Store transitions for training."""
        algo_type = self.agents['type']
        
        if algo_type in ['IPPO', 'ExplabOff']:
            agent_list = self.agents['agents']
            for i, agent_id in enumerate(self.env.agents):
                agent = agent_list[i]
                with torch.no_grad():
                    state = torch.FloatTensor(obs[agent_id]).unsqueeze(0).to(self.device)
                    _, value = agent.network(state)
                    v = value.item()
                agent.store_transition(obs[agent_id], actions[agent_id], rewards[agent_id], dones[agent_id], v, 0.0)
    
    def _update_agents(self):
        """Update all agents."""
        algo_type = self.agents['type']
        
        if algo_type in ['IPPO', 'ExplabOff']:
            for agent in self.agents['agents']:
                if len(agent.trajectory['states']) > 0:
                    agent.update()
        
        elif algo_type == 'HyperNetwork':
            pass
    
    def train(self, num_episodes: int = 0):
        """Run training loop."""
        if num_episodes <= 0:
            num_episodes = self.train_cfg.get('episodes', 10000)
        log_interval = self.train_cfg.get('log_interval', 1000)
        update_every = self.train_cfg.get('update_every', 500)
        
        print(f"\n{'='*60}")
        print(f"Training: {self.agents['type']}")
        print(f"Config: {self.env_cfg['num_md']}MD-{self.env_cfg['num_es']}ES")
        print(f"Episodes: {num_episodes}")
        print(f"Device: {self.device}")
        print(f"{'='*60}\n")
        
        start_time = time.time()
        
        for ep in range(num_episodes):
            obs, _ = self.env.reset(seed=(42 + ep))
            ep_reward = 0.0
            
            for step in range(10):
                actions = self._select_actions(obs)
                next_obs, rewards, terms, truncs, infos = self.env.step(actions)
                ep_reward += sum(rewards.values())
                
                dones = {a: terms[a] or truncs[a] for a in self.env.agents}
                self._store_transitions(obs, actions, rewards, dones)
                
                obs = next_obs
                if all(dones.values()):
                    break
            
            if (ep + 1) % update_every == 0:
                self._update_agents()
            
            if ep % log_interval == 0 or ep == num_episodes - 1:
                metrics = self.env.get_episode_metrics()
                elapsed = time.time() - start_time
                
                record = {
                    'episode': ep,
                    'avg_cost': float(metrics['avg_cost']),
                    'completion_rate': float(metrics['completion_rate']),
                    'avg_latency': float(metrics['avg_latency']),
                    'avg_energy': float(metrics['avg_energy']),
                    'time': elapsed
                }
                self.history.append(record)
                
                print(f"Ep {ep:5d} | Cost: {record['avg_cost']:.4f} | "
                      f"Comp: {record['completion_rate']:.1%} | "
                      f"Time: {elapsed:.1f}s")
                
                if record['avg_cost'] < self.best_cost:
                    self.best_cost = record['avg_cost']
        
        print(f"\nTraining complete. Best cost: {self.best_cost:.4f}")
        return self.history
    
    def evaluate(self, num_episodes: int = 100) -> Dict[str, float]:
        """Evaluate trained model."""
        costs = []
        completions = []
        
        for ep in range(num_episodes):
            obs, _ = self.env.reset(seed=(10000 + ep))
            
            for step in range(10):
                actions = self._select_actions(obs)
                obs, rewards, terms, truncs, infos = self.env.step(actions)
                if all(terms.values()) or all(truncs.values()):
                    break
            
            metrics = self.env.get_episode_metrics()
            costs.append(metrics['avg_cost'])
            completions.append(metrics['completion_rate'])
        
        return {
            'avg_cost': float(np.mean(costs)),
            'std_cost': float(np.std(costs)),
            'avg_completion': float(np.mean(completions)),
            'best_cost': float(np.min(costs)),
            'worst_cost': float(np.max(costs))
        }
    
    def save(self, path: Union[str, Path]):
        """Save model and config."""
        save_path = Path(path)
        save_path.mkdir(parents=True, exist_ok=True)
        
        with open(save_path / 'config.yaml', 'w') as f:
            yaml.dump(self.config, f)
        
        with open(save_path / 'history.json', 'w') as f:
            json.dump(self.history, f, indent=2)
        
        algo_type = self.agents['type']
        if algo_type in ['IPPO', 'ExplabOff']:
            for i, agent in enumerate(self.agents['agents']):
                torch.save(agent.network.state_dict(), save_path / f'agent_{i}.pt')
        elif algo_type == 'HyperNetwork':
            torch.save(self.agents['network'].state_dict(), save_path / 'hypernetwork.pt')
        
        print(f"Saved to {save_path}")
    
    def load(self, path: Union[str, Path]):
        """Load model."""
        load_path = Path(path)
        
        algo_type = self.agents['type']
        if algo_type in ['IPPO', 'ExplabOff']:
            for i, agent in enumerate(self.agents['agents']):
                agent.network.load_state_dict(torch.load(load_path / f'agent_{i}.pt'))
        elif algo_type == 'HyperNetwork':
            self.agents['network'].load_state_dict(torch.load(load_path / 'hypernetwork.pt'))
        
        print(f"Loaded from {load_path}")


def main():
    parser = argparse.ArgumentParser(description='Unified MARL Training')
    parser.add_argument('--config', type=str, required=True, help='Path to config YAML')
    parser.add_argument('--mode', type=str, default='train', choices=['train', 'eval'], 
                       help='Train or evaluate')
    parser.add_argument('--save', type=str, default=None, help='Save directory')
    parser.add_argument('--load', type=str, default=None, help='Load directory')
    args = parser.parse_args()
    
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    
    trainer = UnifiedTrainer(config)
    
    if args.load:
        trainer.load(args.load)
    
    if args.mode == 'train':
        history = trainer.train()
        
        results = trainer.evaluate()
        print(f"\nEvaluation:")
        print(f"  Avg Cost: {results['avg_cost']:.4f} ± {results['std_cost']:.4f}")
        print(f"  Completion: {results['avg_completion']:.1%}")
        
        if args.save:
            trainer.save(args.save)
        else:
            save_dir = Path('results') / f"{config['name']}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            trainer.save(save_dir)
    
    elif args.mode == 'eval':
        if not args.load:
            print("Error: --load required for evaluation")
            return
        results = trainer.evaluate()
        print(f"Evaluation Results:")
        print(f"  Avg Cost: {results['avg_cost']:.4f} ± {results['std_cost']:.4f}")
        print(f"  Completion: {results['avg_completion']:.1%}")


if __name__ == '__main__':
    main()
