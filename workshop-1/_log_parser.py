"""Parse the formatted update lines emitted by ``ppo.train()``'s log_fn callback
and turn them into structured metric records suitable for ``RunLogger``.

This indirection exists because we deliberately do NOT modify ``ppo.py`` to add
a ``metrics_fn`` kwarg (per project preference: keep ``ppo.py`` teaching-clean).
Instead, we hook the existing ``log_fn`` parameter and parse the formatted line
emitted by ``format_update_line``.
"""
from __future__ import annotations

import re
import time
from typing import Callable

import numpy as np


# Matches the prefix produced by ``ppo.format_update_line``:
#     "[update  3/50] timesteps=  1024  policy_loss=+0.123  ..."
_UPDATE_PREFIX = re.compile(r"\[update\s+(\d+)/\d+\]")


def parse_update_line(line: str) -> dict | None:
    """Return a partial metrics dict from one formatted log line, or ``None``.

    Extracts: ``update``, ``timesteps``, ``policy_loss``, ``value_loss``,
    ``entropy``, ``mean_return``. Caller is responsible for adding
    ``log_std_mean``, ``grad_norm``, and ``wall_time_seconds``.
    """
    m = _UPDATE_PREFIX.match(line)
    if not m:
        return None
    metrics: dict = {"update": int(m.group(1))}
    for tok in line.split():
        if "=" not in tok:
            continue
        key, _, raw = tok.partition("=")
        try:
            metrics[key] = float(raw)
        except ValueError:
            continue
    if "timesteps" in metrics:
        metrics["timesteps"] = int(metrics["timesteps"])
    return metrics


def make_log_fn(
    run_logger,
    agent,
    *,
    also_print: bool = True,
) -> Callable[[str], None]:
    """Return a ``log_fn(line)`` suitable for passing to ``ppo.train()``.

    The returned function:
    1. Optionally echoes the line to stdout (default), so workshop participants
       still see fixed-width progress output.
    2. Parses the line, augments with ``log_std_mean``, ``grad_norm``, and
       ``wall_time_seconds``, and emits a JSONL record via ``run_logger(...)``.
    """
    start = time.monotonic()

    def log_fn(line: str) -> None:
        if also_print:
            print(line)
        parsed = parse_update_line(line)
        if parsed is None:
            return
        try:
            log_std_arr = agent.log_std.detach().cpu().numpy()
            parsed["log_std_mean"] = float(np.exp(log_std_arr).mean())
        except Exception:
            parsed["log_std_mean"] = float("nan")
        parsed["grad_norm"] = None
        parsed["wall_time_seconds"] = time.monotonic() - start
        run_logger(parsed)

    return log_fn
