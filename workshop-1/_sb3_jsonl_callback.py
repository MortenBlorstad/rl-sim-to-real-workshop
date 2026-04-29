"""SB3 BaseCallback that emits canonical-schema records to a RunLogger.

Maps SB3's ``logger.name_to_value`` keys into the schema documented in
``specs/002-training-and-visualization/contracts/run-format.md`` so SB3-produced
runs are interchangeable with custom-PPO runs in ``analyze.ipynb``.
"""
from __future__ import annotations

import math
import time
from typing import Any

try:
    from stable_baselines3.common.callbacks import BaseCallback
except ImportError as exc:
    raise ImportError(
        "stable-baselines3 is not installed. "
        "Install with `uv sync --group workshop1`."
    ) from exc


def _safe_float(x: Any) -> float:
    try:
        v = float(x)
    except (TypeError, ValueError):
        return float("nan")
    if not math.isfinite(v):
        return float("nan")
    return v


class Sb3JsonlCallback(BaseCallback):
    """Emit one record per rollout end using the canonical metrics schema.

    SB3 publishes its scalars to ``self.logger.name_to_value`` after every
    rollout. We read it once per ``_on_rollout_end`` and translate into the
    canonical schema (notably flipping the sign on ``train/entropy_loss``,
    since SB3 logs the negation of entropy).

    Args:
        run_logger: a ``RunLogger`` instance whose ``__call__`` accepts the
            canonical metrics dict.
    """

    def __init__(self, run_logger):
        super().__init__()
        self.run_logger = run_logger
        self._start_time: float | None = None

    def _on_training_start(self) -> None:
        self._start_time = time.monotonic()

    def _on_rollout_end(self) -> None:
        nv = dict(self.logger.name_to_value)

        # log_std_mean: read from the SB3 policy's log_std parameter when present
        # (continuous action spaces only).
        log_std_mean: float | None = None
        policy = getattr(self.model, "policy", None)
        if policy is not None and hasattr(policy, "log_std"):
            try:
                log_std_mean = float(policy.log_std.exp().mean().item())
            except Exception:
                log_std_mean = None

        record = {
            "update": int(nv.get("time/iterations", 0)),
            "timesteps": int(
                nv.get("time/total_timesteps", self.num_timesteps)
            ),
            "policy_loss": _safe_float(nv.get("train/policy_gradient_loss")),
            "value_loss": _safe_float(nv.get("train/value_loss")),
            # SB3 logs `entropy_loss = -mean(entropy)`; flip sign for the canonical
            # positive-entropy convention.
            "entropy": -_safe_float(nv.get("train/entropy_loss")),
            "mean_return": _safe_float(nv.get("rollout/ep_rew_mean")),
            "log_std_mean": log_std_mean,
            "grad_norm": None,  # SB3 doesn't expose post-clip grad norm
            "wall_time_seconds": time.monotonic()
            - (self._start_time or time.monotonic()),
        }
        self.run_logger(record)

    def _on_step(self) -> bool:  # required by BaseCallback
        return True
