"""
Unified Training Interface for MARL Edge Offloading - Orthogonal Architecture.

Usage:
    # Method 1: Full config file
    python train_unified.py --config configs/ippo_2es3md.yaml
    
    # Method 2: Separated config (network + algorithm + env)
    python train_unified.py --network standard --algorithm ippo --md 3 --es 2
    python train_unified.py --network hyper --algorithm explaboff --md 5 --es 2
    
    # Method 3: Mixed (config file + overrides)
    python train_unified.py --config configs/base.yaml --md 7 --es 3
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

from envs.paper_accurate_env import PaperAccurateEnvV3


# Default configurations for separation
DEFAULT_NETWORK_CONFIGS = {
    'standard': {
        'type': 'StandardPolicy',
        'hidden_dim': 128,
        'num_layers': 2,
    },
    'hyper': {
        'type': 'HyperPolicy',
        'hidden_dim': 256,
        'num_layers': 3,
        'max_obs_dim': 7,
        'max_action_dim': 4,
    },
    'gnn': {
        'type': 'GNNPolicy',
        'hidden_dim': 128,
        'gnn_layers': 1,
        'node_dim': 4,
        'max_action_dim': 4,
        'max_md': 10,
    },
}

DEFAULT_ALGORITHM_CONFIGS = {
    'ippo': {
        'name': 'ippo',
        'use_mi': False,
        'lr': 5e-5,
        'gamma': 0.99,
        'gae_lambda': 0.95,
        'clip_ratio': 0.2,
        'entropy_coeff': 0.01,
        'value_coeff': 0.5,
        'max_grad_norm': 0.5,
        'update_every': 100,
        'num_epochs': 4,
        'batch_size': 128,
    },
    'explaboff': {
        'name': 'explaboff',
        'use_mi': True,
        'mi_mu': 3.5,
        'mi_nu': 1.0,
        'mi_buffer_size': 1000,
        'lr': 5e-5,
        'gamma': 0.99,
        'gae_lambda': 0.95,
        'clip_ratio': 0.2,
        'entropy_coeff': 0.01,
        'value_coeff': 0.5,
        'max_grad_norm': 0.5,
        'update_every': 100,
        'num_epochs': 4,
        'batch_size': 128,
    },
}

DEFAULT_ENV_CONFIG = {
        'name': 'paper_accurate_env',
    'slots': 10,
    'randomize_profile': True,
    'profile_noise': 0.05,
}

DEFAULT_TRAINING_CONFIG = {
    'num_episodes': 1000,
    'log_interval': 100,
    'seed': 42,
}

DEFAULT_EVAL_CONFIG = {
    'num_episodes': 100,
    'log_interval': 100,
}

DEFAULT_CHECKPOINT_CONFIG = {
    'save_interval': 1000,
}

# Map user-friendly network names to PolicyNetwork class names
NETWORK_TYPE_MAP = {
    'standard': 'StandardPolicy',
    'hyper': 'HyperPolicy',
    'gnn': 'GNNPolicy',
}


class UnifiedTrainer:
    """
    Unified trainer supporting orthogonal combinations:
    - Policy: StandardPolicy, HyperPolicy
    - Algorithm: IPPO (no MI), ExplabOff (with MI)
    """
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        device_str = config.get('device', 'cuda' if torch.cuda.is_available() else 'cpu')
        self.device = torch.device(device_str)
        
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
        """
        Create agents using orthogonal architecture.
        
        Config structure:
            algorithm:
                type: PPO                          # Always PPO
                network: StandardPolicy | HyperPolicy  # Which network
                use_mi: true | false               # Whether to use MI reward
                hidden_dim: 128
                lr: 5e-5
                # ... other hyperparameters
        """
        from agents.ppo_agent import PPOAgent
        from agents.standard_policy import StandardPolicy
        from agents.hyper_policy import HyperPolicy
        from agents.mi_plugin import MIPlugin
        
        M = self.env_cfg['num_md']
        E = self.env_cfg['num_es']
        obs_dim = self.env.obs_dim
        action_dim = E + 1
        
        # Get network type
        network_type = self.algo_cfg.get('network', 'StandardPolicy')
        use_mi = self.algo_cfg.get('use_mi', False)
        
        # Create shared policy network
        if network_type == 'StandardPolicy':
            policy = StandardPolicy(
                state_dim=obs_dim,
                action_dim=action_dim,
                hidden_dim=self.algo_cfg.get('hidden_dim', 128),
                num_layers=self.algo_cfg.get('num_layers', 2)
            ).to(self.device)
        
        elif network_type == 'HyperPolicy':
            policy = HyperPolicy(
                max_obs_dim=self.algo_cfg.get('max_obs_dim', 7),
                max_action_dim=self.algo_cfg.get('max_action_dim', 4),
                hidden_dim=self.algo_cfg.get('hidden_dim', 256)
            ).to(self.device)
            policy.set_config(M, E)
        
        elif network_type == 'GNNPolicy':
            from agents.gnn_policy import GNNPolicy
            policy = GNNPolicy(
                max_action_dim=self.algo_cfg.get('max_action_dim', 4),
                hidden_dim=self.algo_cfg.get('hidden_dim', 128),
                gnn_layers=self.algo_cfg.get('gnn_layers', 1),
                node_dim=self.algo_cfg.get('node_dim', 4),
                max_md=self.algo_cfg.get('max_md', 10)
            ).to(self.device)
        
        else:
            raise ValueError(f"Unknown network type: {network_type}")
        
        # Create optional MI plugin
        mi_plugin = None
        if use_mi:
            mi_plugin = MIPlugin(
                state_dim=obs_dim,
                action_dim=action_dim,
                hidden_dim=self.algo_cfg.get('hidden_dim', 128),
                mu=self.algo_cfg.get('mi_mu', 0.01),
                nu=self.algo_cfg.get('mi_nu', 0.01),
                device=self.device
            )
        
        # Create agents (all share the same policy network if parameter sharing)
        agent_list = []
        for i in range(M):
            agent = PPOAgent(
                agent_id=i,
                policy_network=policy,  # Inject policy!
                mi_plugin=mi_plugin,     # Optional MI
                learning_rate=float(self.algo_cfg.get('lr', 5e-5)),
                gamma=float(self.algo_cfg.get('gamma', 0.99)),
                gae_lambda=float(self.algo_cfg.get('gae_lambda', 0.95)),
                clip_ratio=float(self.algo_cfg.get('clip_ratio', 0.2)),
                entropy_coeff=float(self.algo_cfg.get('entropy_coeff', 0.01)),
                value_coeff=float(self.algo_cfg.get('value_coeff', 0.5)),
                max_grad_norm=float(self.algo_cfg.get('max_grad_norm', 0.5)),
                device=self.device
            )
            agent_list.append(agent)
        
        return {
            'type': 'PPO',
            'network_type': network_type,
            'use_mi': use_mi,
            'agents': agent_list,
            'policy': policy,
            'mi_plugin': mi_plugin
        }
    
    def _select_actions(self, obs: Dict[str, np.ndarray]) -> Dict[str, int]:
        """Select actions for all agents - unified!"""
        actions: Dict[str, int] = {}
        
        # If using GNN policy, build graph once per step
        policy = self.agents['policy']
        if hasattr(policy, 'set_graph'):
            policy.set_graph(self.env)
        
        for i, agent_id in enumerate(self.env.agents):
            agent = self.agents['agents'][i]
            action, log_prob, value = agent.select_action(obs[agent_id], agent_id=i)
            actions[agent_id] = action
            # Store value and log_prob for transition
            agent._last_value = value
            agent._last_log_prob = log_prob
            # Store embedding for GNN policy
            if hasattr(policy, 'get_embedding'):
                embedding = policy.get_embedding(i)
                if embedding is not None:
                    agent._last_embedding = embedding.detach().cpu()
        
        return actions
    
    def _store_transitions(self, obs, actions, rewards, dones):
        """Store transitions for all agents."""
        for i, agent_id in enumerate(self.env.agents):
            agent = self.agents['agents'][i]
            
            # Compute MI reward if plugin exists
            mi_reward = agent.compute_mi_reward(obs[agent_id], actions[agent_id])
            total_reward = rewards[agent_id] + mi_reward
            
            # Get embedding if available (for GNN policy)
            embedding = getattr(agent, '_last_embedding', None)
            
            # Store transition
            agent.store_transition(
                obs[agent_id],
                actions[agent_id],
                total_reward,
                agent._last_value,
                agent._last_log_prob,
                dones[agent_id],
                embedding=embedding
            )
    
    def _update_agents(self):
        """Update all agents."""
        for agent in self.agents['agents']:
            if len(agent.trajectory['states']) > 0:
                agent.update(
                    batch_size=self.train_cfg.get('batch_size', 64),
                    num_epochs=self.train_cfg.get('num_epochs', 4)
                )
                agent.clear_trajectory()
    
    def train(self, num_episodes: int = 0):
        """Run training loop."""
        if num_episodes <= 0:
            num_episodes = self.train_cfg.get('num_episodes', 10000)
        log_interval = self.train_cfg.get('log_interval', 1000)
        update_every = self.train_cfg.get('update_every', 500)
        
        network_type = self.agents['network_type']
        use_mi = self.agents['use_mi']
        
        print(f"\n{'='*60}")
        print(f"Training: PPO + {network_type} + {'MI' if use_mi else 'No MI'}")
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
                
                # Clear GNN cache after storing transitions
                policy = self.agents['policy']
                if hasattr(policy, 'clear_cache'):
                    policy.clear_cache()
                
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
        
        # Save shared policy
        policy = self.agents['policy']
        torch.save(policy.state_dict(), save_path / 'policy.pt')
        
        # Save MI plugin if exists
        if self.agents['mi_plugin'] is not None:
            torch.save(self.agents['mi_plugin'].state_dict(), save_path / 'mi_plugin.pt')
        
        print(f"Saved to {save_path}")
    
    def load(self, path: Union[str, Path]):
        """Load model."""
        load_path = Path(path)
        
        policy = self.agents['policy']
        policy.load_state_dict(torch.load(load_path / 'policy.pt', map_location=self.device))
        
        if self.agents['mi_plugin'] is not None:
            self.agents['mi_plugin'].load_state_dict(
                torch.load(load_path / 'mi_plugin.pt', map_location=self.device)
            )
        
        print(f"Loaded from {load_path}")


def build_config(args) -> Dict[str, Any]:
    """Build config: YAML-first with CLI override capability.

    Priority: YAML file > CLI args > Python defaults
    """
    config = {}

    # Step 1: Load YAML if provided (primary configuration source)
    if args.config:
        with open(args.config, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
    else:
        # Step 2: No YAML → build from CLI + Python defaults
        # Network
        if args.network:
            network_type = args.network.lower()
            net_defaults = DEFAULT_NETWORK_CONFIGS[network_type]
            config.setdefault('algorithm', {})['network'] = NETWORK_TYPE_MAP[network_type]
            for key, value in net_defaults.items():
                if key != 'type':
                    config['algorithm'].setdefault(key, value)

        # Algorithm
        if args.algorithm:
            algo_defaults = DEFAULT_ALGORITHM_CONFIGS[args.algorithm.lower()]
            config.setdefault('algorithm', {})
            for key, value in algo_defaults.items():
                config['algorithm'].setdefault(key, value)

        # Environment
        config.setdefault('environment', DEFAULT_ENV_CONFIG.copy())
        if args.md is not None:
            config['environment']['num_md'] = args.md
        if args.es is not None:
            config['environment']['num_es'] = args.es

        # Training
        config.setdefault('training', DEFAULT_TRAINING_CONFIG.copy())

    # Step 3: CLI overrides always win (regardless of YAML or CLI mode)
    if args.network and not args.config:
        pass  # already handled above
    if args.algorithm and not args.config:
        pass  # already handled above
    if args.md is not None:
        config.setdefault('environment', {})['num_md'] = args.md
    if args.es is not None:
        config.setdefault('environment', {})['num_es'] = args.es
    if args.episodes:
        config.setdefault('training', {})['num_episodes'] = args.episodes

    # Ensure required fields
    config.setdefault('environment', {}).setdefault('num_md', 3)
    config['environment'].setdefault('num_es', 2)
    config.setdefault('training', {}).setdefault('num_episodes', 10000)
    config.setdefault('training', {}).setdefault('log_interval', 100)
    config.setdefault('evaluation', DEFAULT_EVAL_CONFIG.copy())
    config.setdefault('checkpoint', DEFAULT_CHECKPOINT_CONFIG.copy())

    # Device and seed
    config['device'] = args.device if args.device else config.get('device', 'cuda' if torch.cuda.is_available() else 'cpu')
    config['seed'] = args.seed if args.seed is not None else config.get('seed', 42)

    # Merge top-level network section into algorithm (YAML compatibility)
    # MUST happen before name generation
    if 'network' in config and isinstance(config['network'], dict):
        net_cfg = config.pop('network')
        config.setdefault('algorithm', {})
        if 'type' in net_cfg:
            net_type = net_cfg.pop('type')
            net_cfg['network'] = NETWORK_TYPE_MAP.get(net_type, net_type)
        config['algorithm'].update(net_cfg)

    # Map friendly network names to class names (safety net)
    if 'algorithm' in config and 'network' in config['algorithm']:
        raw = config['algorithm']['network']
        config['algorithm']['network'] = NETWORK_TYPE_MAP.get(raw, raw)

    # Auto-generate name if not in YAML
    if 'name' not in config:
        net_key = args.network or config.get('algorithm', {}).get('network', 'standard')
        net_name = net_key.lower().replace('policy', '')
        algo_name = args.algorithm or config.get('algorithm', {}).get('name', 'ippo')
        md = config['environment']['num_md']
        es = config['environment']['num_es']
        config['name'] = f'{algo_name}_{net_name}_{md}md{es}es'

    return config


def main():
    parser = argparse.ArgumentParser(description='Unified MARL Training - Orthogonal Architecture')
    
    # Config file (optional, for full config or base config)
    parser.add_argument('--config', type=str, default=None, help='Path to base config YAML (optional)')
    
    # Separated configuration
    parser.add_argument('--network', type=str, choices=['standard', 'hyper', 'gnn'], 
                       help='Network architecture: standard, hyper, or gnn')
    parser.add_argument('--algorithm', type=str, choices=['ippo', 'explaboff'],
                       help='Training algorithm: ippo or explaboff')
    parser.add_argument('--md', type=int, help='Number of mobile devices')
    parser.add_argument('--es', type=int, help='Number of edge servers')
    
    # Training parameters
    parser.add_argument('--episodes', type=int, help='Number of training episodes')
    parser.add_argument('--seed', type=int, help='Random seed')
    parser.add_argument('--device', type=str, choices=['cpu', 'cuda'], help='Device to use')
    
    # Mode
    parser.add_argument('--mode', type=str, default='train', choices=['train', 'eval'], 
                       help='Train or evaluate')
    parser.add_argument('--save', type=str, default=None, help='Save directory')
    parser.add_argument('--load', type=str, default=None, help='Load directory')
    
    args = parser.parse_args()
    
    # Validate: either config file or separated params must be provided
    if not args.config and not (args.network and args.algorithm):
        parser.error("Either --config or both --network and --algorithm must be provided")
    
    # Build config
    config = build_config(args)
    
    print("="*60)
    print("Unified MARL Training - Orthogonal Architecture")
    print("="*60)
    print(f"Network: {config['algorithm'].get('network', 'standard')}")
    print(f"Algorithm: {config['algorithm']['name']}")
    print(f"Environment: {config['environment']['num_md']}MD-{config['environment']['num_es']}ES")
    print(f"Episodes: {config['training']['num_episodes']}")
    print(f"Device: {config['device']}")
    print(f"Seed: {config['seed']}")
    print("="*60)
    
    trainer = UnifiedTrainer(config)
    
    if args.load:
        trainer.load(args.load)
    
    if args.mode == 'train':
        history = trainer.train(num_episodes=config['training'].get('num_episodes', 10000))
        
        results = trainer.evaluate()
        print(f"\n{'='*60}")
        print("Training Complete!")
        print(f"{'='*60}")
        print(f"Evaluation:")
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
