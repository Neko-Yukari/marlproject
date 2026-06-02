"""Training reporter — generates checkpoints + reports + curves."""
import json, csv, time
from pathlib import Path
from datetime import datetime
import torch

class TrainingReporter:
    def __init__(self, run_dir: str, config: dict):
        self.run_dir = Path(run_dir)
        self.run_dir.mkdir(parents=True, exist_ok=True)
        for sub in ["checkpoints", "reports", "curves"]:
            (self.run_dir / sub).mkdir(exist_ok=True)
        
        self.config = config
        self.metadata = {
            "start_time": datetime.now().isoformat(),
            "config": config,
        }
        with open(self.run_dir / "metadata.json", "w") as f:
            json.dump(self.metadata, f, indent=2)
        
        # Curve CSVs
        self.curve_fields = ["episode", "avg_cost", "completion_rate", "avg_latency", "avg_energy", "time_seconds"]
        for name in ["cost", "completion", "latency", "energy"]:
            with open(self.run_dir / "curves" / f"{name}.csv", "w", newline="") as f:
                pass  # headers written on first save
        self.curve_file = self.run_dir / "curves" / "all_metrics.csv"
        with open(self.curve_file, "w", newline="") as f:
            csv.writer(f).writerow(self.curve_fields)
        
        self.history = []
        self.best = {"avg_cost": float("inf"), "episode": 0}
        self.start_time = time.time()
    
    def log_episode(self, ep: int, metrics: dict):
        """Log single episode metrics."""
        row = {
            "episode": ep,
            "avg_cost": metrics["avg_cost"],
            "completion_rate": metrics["completion_rate"],
            "avg_latency": metrics.get("avg_latency", 0),
            "avg_energy": metrics.get("avg_energy", 0),
            "time_seconds": time.time() - self.start_time,
        }
        self.history.append(row)
        
        # Update best
        if metrics["avg_cost"] < self.best["avg_cost"]:
            self.best = {"avg_cost": metrics["avg_cost"], "episode": ep}
        
        # Append to CSV
        with open(self.curve_file, "a", newline="") as f:
            csv.writer(f).writerow([row[k] for k in self.curve_fields])
    
    def save_checkpoint(self, episode: int, agents: list, label: str = ""):
        """Save model weights + optimizer states."""
        ckpt = {
            "episode": episode,
            "config": self.config,
            "history": self.history[-1000:],  # last 1000 episodes
            "agents": [],
        }
        for agent in agents:
            agent_ckpt = {}
            if hasattr(agent, 'network'):
                agent_ckpt["network"] = agent.network.state_dict()
            if hasattr(agent, 'optimizer'):
                agent_ckpt["optimizer"] = agent.optimizer.state_dict()
            if hasattr(agent, '_best_ep_reward'):
                agent_ckpt["best_ep_reward"] = agent._best_ep_reward
            ckpt["agents"].append(agent_ckpt)
        
        tag = f"_{label}" if label else ""
        path = self.run_dir / "checkpoints" / f"ep_{episode}{tag}.pt"
        torch.save(ckpt, path)
        return str(path)
    
    def generate_report(self, episode: int, comparison: dict = None):
        """Generate阶段性JSON report."""
        # Compute convergence stats
        recent = [r["avg_cost"] for r in self.history[-1000:]] if len(self.history) >= 1000 else [r["avg_cost"] for r in self.history]
        variance = sum((x - sum(recent)/len(recent))**2 for x in recent) / len(recent) if recent else 0
        
        report = {
            "timestamp": datetime.now().isoformat(),
            "episode": episode,
            "config": self.config,
            "metrics": {
                "current": self.history[-1] if self.history else None,
                "best": self.best,
                "convergence": {
                    "stable_since": self._find_stable_point(),
                    "variance_last_1000": round(variance, 6),
                }
            },
            "training_stats": {
                "episodes_per_second": round(episode / max(self.history[-1]["time_seconds"], 0.1), 2) if self.history else 0,
                "total_time_seconds": round(time.time() - self.start_time, 1),
            },
            "comparison": comparison or {},
        }
        
        path = self.run_dir / "reports" / f"report_ep{episode}.json"
        with open(path, "w") as f:
            json.dump(report, f, indent=2)
        return str(path)
    
    def _find_stable_point(self):
        """Find episode where cost stabilized (variance < 0.01 for 500 eps)."""
        if len(self.history) < 500:
            return None
        for i in range(500, len(self.history)):
            window = [r["avg_cost"] for r in self.history[i-500:i]]
            var = sum((x - sum(window)/len(window))**2 for x in window) / len(window)
            if var < 0.01:
                return self.history[i]["episode"]
        return None
    
    def finalize(self, agents: list, comparison: dict = None):
        """Final save after training completes."""
        final_ep = self.history[-1]["episode"] if self.history else 0
        ckpt_path = self.save_checkpoint(final_ep, agents, "final")
        report_path = self.generate_report(final_ep, comparison)
        
        final_results = {
            "config": self.config,
            "best": self.best,
            "final": self.history[-1] if self.history else None,
            "total_episodes": len(self.history),
            "total_time_seconds": round(time.time() - self.start_time, 1),
            "checkpoint": ckpt_path,
            "final_report": report_path,
        }
        with open(self.run_dir / "final_results.json", "w") as f:
            json.dump(final_results, f, indent=2)
        
        return final_results
