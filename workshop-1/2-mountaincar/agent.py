"""MountainCarContinuous-v0 PPOAgent subclass stub.

This file demonstrates the override-contract pattern from
``specs/001-ppo-skeleton/contracts/agent-interface.md`` and serves as
the extension point for stage 2 (``workshop-1/2-mountaincar/``).

The stage-1 base ``PPOAgent`` already trains MountainCarContinuous-v0
in script mode (``workshop-1/1-ppo/ppo_skeleton.py``), so this stub
is intentionally minimal — ``preprocess`` is identity. Stage 2 may
extend this subclass with reward shaping, an observation normalizer,
or hyperparameter tuning to actually solve the environment.
"""
from __future__ import annotations

import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
_PPO_DIR = os.path.normpath(os.path.join(_HERE, "..", "1-ppo"))
if _PPO_DIR not in sys.path:
    sys.path.insert(0, _PPO_DIR)

from ppo_skeleton import PPOAgent, register_agent  # noqa: E402


@register_agent
class MountainCarPPOAgent(PPOAgent):
    """PPOAgent for MountainCarContinuous-v0.

    Vector observation; preprocess is identity in this minimal stub.
    Stage 2 may override with a normalization wrapper or with
    domain-specific feature engineering.
    """

    def preprocess(self, obs):
        return obs
