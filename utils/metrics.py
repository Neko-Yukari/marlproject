"""
Metrics collection for MARL edge offloading training.
"""
import numpy as np
from typing import Dict, List


class MetricsCollector:
    """Collects and aggregates evaluation metrics across episodes."""

    def __init__(self):
        self.history: List[Dict[str, float]] = []

    def record_episode(self, metrics: Dict[str, float]):
        """Record metrics for one completed episode."""
        self.history.append(metrics)

    def get_summary(self, last_n: int = 0) -> Dict[str, Dict[str, float]]:
        """Get aggregate statistics over last N episodes (0 = all)."""
        data = self.history[-last_n:] if last_n > 0 else self.history
        if not data:
            return {}

        summary = {}
        for key in data[0].keys():
            values = [d[key] for d in data]
            summary[key] = {
                'mean': float(np.mean(values)),
                'std': float(np.std(values)),
                'min': float(np.min(values)),
                'max': float(np.max(values)),
            }
        return summary

    def latest(self) -> Dict[str, float]:
        """Return the most recent episode's metrics."""
        return self.history[-1] if self.history else {}
