"""Public API of the refactored ``ppo`` package.

See ``specs/003-fix-mountaincar-train/contracts/agent-api.md`` for the
authoritative surface. Only symbols re-exported here are stable; everything
else under ``ppo.*`` is implementation detail.
"""
from .networks import ActorNetwork, CriticNetwork
from .ppo import DEFAULT_SEED, PPOAgent, _AGENT_REGISTRY, register_agent
from .rollout_buffer import RolloutBuffer

__all__ = [
    "ActorNetwork",
    "CriticNetwork",
    "DEFAULT_SEED",
    "PPOAgent",
    "RolloutBuffer",
    "_AGENT_REGISTRY",
    "register_agent",
]
