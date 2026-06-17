import torch
import numpy as np
from envs.paper_accurate_env import PaperAccurateEnvV3
from collections import defaultdict

def greedy_baseline(env, seed=10000, episodes=100):
    """
    Greedy baseline: for each MD, independently choose action (local or ES)
    that minimizes expected cost, considering current ES queue lengths.
    """
    costs, completions = [], []
    for ep in range(episodes):
        obs, _ = env.reset(seed=seed + ep)
        es_queue = [0] * env.E
        env_done = False
        while not env_done:
            actions = {}
            # Compute current tasks
            tasks = {}
            for i, agent_id in enumerate(env.agents):
                tasks[agent_id] = env._generate_task(i)
            # Greedy action selection per MD
            for i, agent_id in enumerate(env.agents):
                task = tasks[agent_id]
                best_action = 0
                best_cost = float('inf')
                # Local
                t_loc = task['cycles'] / env.MD_CPU
                energy_loc = env.ENERGY_COEFF * (env.MD_CPU ** 2) * task['cycles']
                cost_loc = env.ETA * t_loc + (1 - env.ETA) * energy_loc
                best_cost = cost_loc
                best_action = 0
                # Each ES
                for es_idx in range(env.E):
                    es_cpu = env._get_es_cpu(es_idx)
                    rate = env._compute_tx_rate(i, es_idx)
                    t_tx = task['data_bits'] / rate
                    t_exe = task['cycles'] / es_cpu
                    # Greedy waiting estimate: tasks currently in queue * t_exe
                    t_wait = es_queue[es_idx] * t_exe
                    t_edge = t_tx + t_wait + t_exe
                    energy_tx = env.TX_POWER * t_tx
                    cost_edge = env.ETA * t_edge + (1 - env.ETA) * energy_tx
                    if cost_edge < best_cost:
                        best_cost = cost_edge
                        best_action = es_idx + 1
                actions[agent_id] = best_action
                if best_action > 0:
                    es_queue[best_action - 1] += 1
            obs, rewards, terminations, truncations, _ = env.step(actions)
            env_done = all(terminations.values()) or all(truncations.values())
        metrics = env.get_episode_metrics()
        costs.append(metrics['avg_cost'])
        completions.append(metrics['completion_rate'])
    return np.mean(costs), np.std(costs), np.mean(completions)

if __name__ == '__main__':
    for num_md, num_es in [(3, 2), (5, 2)]:
        env = PaperAccurateEnvV3(num_devices=num_md, num_servers=num_es, seed=10000)
        cost_mean, cost_std, comp = greedy_baseline(env, episodes=100)
        print(f'Greedy {num_md}MD-{num_es}ES: cost={cost_mean:.4f}+/-{cost_std:.4f}, comp={comp*100:.1f}%')
