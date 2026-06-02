"""Vectorized environment for parallel training."""
import numpy as np
from envs.paper_accurate_env_v3 import PaperAccurateEnvV3

class VectorizedEnv:
    """Run N environments in parallel with different seeds."""
    
    def __init__(self, M, E, n_envs=8, randomize_profile=True):
        self.n_envs = n_envs
        self.envs = [PaperAccurateEnvV3(M, E, randomize_profile=randomize_profile) for _ in range(n_envs)]
        self.M = M
        self.E = E
        self.agents = [f"device_{i}" for i in range(M)]
        
    def reset(self, seeds):
        """Reset all environments."""
        obs_list = []
        for env, seed in zip(self.envs, seeds):
            obs, _ = env.reset(seed=seed)
            obs_list.append(obs)
        return obs_list
    
    def step(self, actions_list):
        """Step all environments. Returns list of results."""
        results = []
        for env, actions in zip(self.envs, actions_list):
            result = env.step(actions)
            results.append(result)
        return results
    
    def get_metrics(self):
        """Get metrics from all environments."""
        return [env.get_episode_metrics() for env in self.envs]

# Test
if __name__ == "__main__":
    vec_env = VectorizedEnv(7, 3, n_envs=4)
    obs = vec_env.reset(seeds=[42, 43, 44, 45])
    print(f"Environments: {len(obs)}")
    print(f"Agents per env: {len(obs[0])}")
    print(f"Obs shape: {obs[0]['device_0'].shape}")
    print("[OK] Vectorized env ready")
