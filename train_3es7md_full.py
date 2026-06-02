"""3ES-7MD Training — with checkpointing + reporting."""
import sys; sys.path.insert(0, '.')
import numpy as np, json, time, torch
from datetime import datetime
from pathlib import Path
from envs.paper_accurate_env_v3 import PaperAccurateEnvV3
from agents.ippo_agent import IPPOAgent
from agents.explaboff_agent import ExplabOffAgent
from utils.reporter import TrainingReporter

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Device: {device}")
if device.type == 'cuda':
    print(f"GPU: {torch.cuda.get_device_name(0)}")

# ═══════════════════════════════════════════════
# Config
# ═══════════════════════════════════════════════
M, E = 7, 3
EPISODES = 20000
LOG_INTERVAL = 1000
CKPT_INTERVAL = 5000

STD = {
    "hidden_dim": 1024,
    "update_every": 500,
    "batch_size": 4096,
    "num_epochs": 10,
    "lr": 5e-5,
    "lr_step": 5000,
    "lr_gamma": 0.5,
}

# ═══════════════════════════════════════════════
# IPPO Training
# ═══════════════════════════════════════════════
def train_ippo():
    run_dir = f"results/bench_3es7md_ippo_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    reporter = TrainingReporter(run_dir, config={
        "M": M, "E": E, "algorithm": "IPPO", **STD
    })
    
    env = PaperAccurateEnvV3(M, E, randomize_profile=True)
    adim = E + 1
    agents = [IPPOAgent(i, env.obs_dim, adim, hidden_dim=STD["hidden_dim"],
                        learning_rate=STD["lr"], device=device)
              for i in range(M)]
    
    # LR decay
    for agent in agents:
        agent.scheduler = torch.optim.lr_scheduler.StepLR(
            agent.optimizer, step_size=STD["lr_step"], gamma=STD["lr_gamma"])
    
    accum_traj = {i: {"states": [], "actions": [], "rewards": [], 
                       "values": [], "log_probs": [], "dones": []}
                  for i in range(M)}
    
    t0 = time.time()
    
    for ep in range(EPISODES):
        obs, _ = env.reset(seed=(42 + ep))
        ep_reward_sum = 0.0
        
        for step in range(10):
            actions = {}
            for i, agent in enumerate(agents):
                a_name = f"device_{i}"
                a, lp, v = agent.select_action(obs[a_name])
                actions[a_name] = a
                accum_traj[i]["states"].append(obs[a_name])
                accum_traj[i]["actions"].append(a)
                accum_traj[i]["values"].append(v)
                accum_traj[i]["log_probs"].append(lp)
                accum_traj[i]["dones"].append(False)
                accum_traj[i]["rewards"].append(0.0)
            
            next_obs, rewards, terms, _, _ = env.step(actions)
            
            for i, agent in enumerate(agents):
                a_name = f"device_{i}"
                accum_traj[i]["rewards"][-1] = rewards[a_name]
                ep_reward_sum += rewards[a_name]
            
            obs = next_obs
            if any(terms.values()):
                break
        
        # Batch update
        if (ep + 1) % STD["update_every"] == 0:
            for i, agent in enumerate(agents):
                agent.trajectory = {k: list(v) for k, v in accum_traj[i].items()}
                agent.update(batch_size=STD["batch_size"], num_epochs=STD["num_epochs"])
                agent.scheduler.step()
                for k in accum_traj[i]:
                    accum_traj[i][k].clear()
            if device.type == 'cuda':
                torch.cuda.empty_cache()
        
        # Logging
        if ep % LOG_INTERVAL == 0 or ep == EPISODES - 1:
            m = env.get_episode_metrics()
            reporter.log_episode(ep, m)
            el = time.time() - t0
            print(f"[IPPO] ep{ep:6d} cost={m['avg_cost']:.4f} comp={m['completion_rate']:.3f} "
                  f"lat={m['avg_latency']:.2f} t={el:.0f}s")
            
            # Generate report
            reporter.generate_report(ep)
        
        # Checkpoint
        if (ep + 1) % CKPT_INTERVAL == 0:
            path = reporter.save_checkpoint(ep, agents)
            print(f"  Checkpoint saved: {path}")
    
    # Finalize
    results = reporter.finalize(agents)
    print(f"\nIPPO Final: best_cost={results['best']['avg_cost']:.4f} @ ep{results['best']['episode']}")
    return results

# ═══════════════════════════════════════════════
# ExplabOff Training
# ═══════════════════════════════════════════════
def train_explaboff():
    run_dir = f"results/bench_3es7md_explaboff_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    reporter = TrainingReporter(run_dir, config={
        "M": M, "E": E, "algorithm": "ExplabOff", **STD
    })
    
    env = PaperAccurateEnvV3(M, E, randomize_profile=True)
    adim = E + 1
    agents = [ExplabOffAgent(i, env.obs_dim, adim, hidden_dim=STD["hidden_dim"],
                             lr=STD["lr"], mi_mu=3.5, mi_nu=1.0, device=device)
              for i in range(M)]
    
    accum_traj = {i: {"states": [], "actions": [], "rewards": [], 
                       "values": [], "log_probs": [], "dones": []}
                  for i in range(M)}
    episode_rewards = []
    
    t0 = time.time()
    
    for ep in range(EPISODES):
        obs, _ = env.reset(seed=(42 + ep))
        ep_reward_sum = 0.0
        
        for step in range(10):
            actions = {}
            for i, agent in enumerate(agents):
                a_name = f"device_{i}"
                a, lp, v = agent.select_action(obs[a_name])
                actions[a_name] = a
                accum_traj[i]["states"].append(obs[a_name])
                accum_traj[i]["actions"].append(a)
                accum_traj[i]["values"].append(v)
                accum_traj[i]["log_probs"].append(lp)
                accum_traj[i]["dones"].append(False)
                accum_traj[i]["rewards"].append(0.0)
            
            next_obs, rewards, terms, _, _ = env.step(actions)
            
            for i, agent in enumerate(agents):
                a_name = f"device_{i}"
                mi_r = agent.compute_mi_reward(obs[a_name], actions[a_name])
                combined = rewards[a_name] + mi_r
                accum_traj[i]["rewards"][-1] = combined
                ep_reward_sum += combined
            
            obs = next_obs
            if any(terms.values()):
                break
        
        episode_rewards.append(ep_reward_sum)
        
        # Batch update
        if (ep + 1) % STD["update_every"] == 0:
            for i, agent in enumerate(agents):
                agent.trajectory = {k: list(v) for k, v in accum_traj[i].items()}
                avg_reward = np.mean(episode_rewards[-STD["update_every"]:])
                agent.classify_episode(avg_reward)
                agent.update(batch_size=STD["batch_size"], num_epochs=STD["num_epochs"])
                for k in accum_traj[i]:
                    accum_traj[i][k].clear()
            if device.type == 'cuda':
                torch.cuda.empty_cache()
        
        # Logging
        if ep % LOG_INTERVAL == 0 or ep == EPISODES - 1:
            m = env.get_episode_metrics()
            reporter.log_episode(ep, m)
            el = time.time() - t0
            print(f"[ExplabOff] ep{ep:6d} cost={m['avg_cost']:.4f} comp={m['completion_rate']:.3f} "
                  f"lat={m['avg_latency']:.2f} t={el:.0f}s")
            reporter.generate_report(ep)
        
        # Checkpoint
        if (ep + 1) % CKPT_INTERVAL == 0:
            path = reporter.save_checkpoint(ep, agents)
            print(f"  Checkpoint saved: {path}")
    
    # Finalize
    results = reporter.finalize(agents)
    print(f"\nExplabOff Final: best_cost={results['best']['avg_cost']:.4f} @ ep{results['best']['episode']}")
    return results

# ═══════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════
if __name__ == "__main__":
    print(f"\n{'='*60}")
    print(f"3ES-7MD Training | 20K episodes | GPU + Reporter")
    print(f"{'='*60}\n")
    
    ippo_results = train_ippo()
    explaboff_results = train_explaboff()
    
    print(f"\n{'='*60}")
    print("Summary")
    print(f"{'='*60}")
    print(f"IPPO:      best={ippo_results['best']['avg_cost']:.4f} @ ep{ippo_results['best']['episode']}")
    print(f"ExplabOff: best={explaboff_results['best']['avg_cost']:.4f} @ ep{explaboff_results['best']['episode']}")
