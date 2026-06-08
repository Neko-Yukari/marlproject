"""
Neural network architectures for MARL edge offloading.
"""
from .actor_critic import ActorCriticNetwork
from .mi_estimator import InfoNCEEstimator, L1OutEstimator

__all__ = [
    "ActorCriticNetwork",
    "InfoNCEEstimator",
    "L1OutEstimator",
]
