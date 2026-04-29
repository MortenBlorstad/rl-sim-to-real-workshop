"""RunLogger — append-only JSONL writer for PPO training metrics.

See ``specs/002-training-and-visualization/contracts/run-format.md`` for the
on-disk schema. Writes are best-effort: an OSError during ``__call__`` is
logged once and swallowed so that training is never blocked by disk issues.
"""
from __future__ import annotations

import json
import math
import platform
import shutil
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


class RunDirectoryExistsError(FileExistsError):
    """Raised when the resolved run directory already exists and force=False."""


def _utc_iso8601() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _git_sha() -> str:
    try:
        out = subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            stderr=subprocess.DEVNULL,
            timeout=5,
        )
        return out.decode().strip()
    except Exception:
        return "unknown"


def _library_version(name: str) -> str:
    try:
        mod = __import__(name)
        return getattr(mod, "__version__", "unknown")
    except ImportError:
        return "unknown"


_UNSET = object()


_DEFAULT_METRIC_DEFINITIONS: dict[str, str] = {
    "update": "1-indexed PPO update counter",
    "timesteps": "cumulative environment steps after this update",
    "policy_loss": "PPO clipped surrogate loss (last minibatch of last epoch)",
    "value_loss": "MSE value loss (last minibatch of last epoch)",
    "entropy": "mean policy entropy this update (positive number; higher = more exploration)",
    "mean_return": "rolling mean of last 10 episode returns; current partial return if no episode finished",
    "log_std_mean": "exp(log_std).mean() — current exploration scale",
    "grad_norm": "post-clip gradient L2 norm; null for SB3 runs",
    "wall_time_seconds": "seconds since RunLogger initialization",
}


def _sanitize(value: Any) -> Any:
    """Replace NaN / Inf floats with None so JSON serializes cleanly."""
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


class RunLogger:
    """Best-effort JSONL writer + meta.json owner for one training run.

    Use as a context manager:

        with RunLogger(stage="mountaincar", ...) as runlog:
            runlog({"update": 1, "timesteps": 1024, ...})
    """

    def __init__(
        self,
        stage: str,
        hyperparameters: dict,
        env_id: str,
        agent_class: str,
        seed: int,
        total_timesteps: int,
        run_name: str | None = None,
        force: bool = False,
        runs_root: Path = Path("runs"),
        metric_definitions: dict[str, str] | None = None,
        network_arch: str | None = None,
        hf_repo_id=_UNSET,
        hf_filename=_UNSET,
    ) -> None:
        self.stage = stage
        if run_name is None:
            run_name = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
        self._run_dir = (Path(runs_root) / stage / run_name).resolve()

        if self._run_dir.exists():
            if not force:
                raise RunDirectoryExistsError(
                    f"run directory '{self._run_dir}' already exists. "
                    f"Pick a different --run-name or pass --force to overwrite."
                )
            shutil.rmtree(self._run_dir)
        self._run_dir.mkdir(parents=True, exist_ok=False)

        self._start_monotonic = time.monotonic()
        self._meta = {
            "stage": stage,
            "env_id": env_id,
            "agent_class": agent_class,
            "seed": int(seed),
            "total_timesteps": int(total_timesteps),
            "hyperparameters": dict(hyperparameters),
            "git_sha": _git_sha(),
            "started_at": _utc_iso8601(),
            "finished_at": None,
            "status": "running",
            "python_version": platform.python_version(),
            "torch_version": _library_version("torch"),
            "gymnasium_version": _library_version("gymnasium"),
            "metric_definitions": dict(
                metric_definitions or _DEFAULT_METRIC_DEFINITIONS
            ),
        }
        if network_arch is not None:
            self._meta["network_arch"] = network_arch
        # Always-present-with-null contract for SB3 runs: if either kwarg is
        # passed (even as None), record both. The SB3 driver always passes
        # them; custom-PPO drivers omit them entirely (the keys won't appear).
        # See specs/005-carracing-drivers/contracts/meta-fields.md.
        if hf_repo_id is not _UNSET or hf_filename is not _UNSET:
            self._meta["hf_repo_id"] = hf_repo_id if hf_repo_id is not _UNSET else None
            self._meta["hf_filename"] = hf_filename if hf_filename is not _UNSET else None
        self._write_meta()

        self._jsonl_path = self._run_dir / "metrics.jsonl"
        self._jsonl_fh = self._jsonl_path.open("w", encoding="utf-8")
        self._oserror_warned = False
        self._closed = False

    @property
    def run_dir(self) -> Path:
        return self._run_dir

    def _write_meta(self) -> None:
        meta_path = self._run_dir / "meta.json"
        meta_path.write_text(
            json.dumps(self._meta, indent=2) + "\n", encoding="utf-8"
        )

    def __call__(self, metrics: dict) -> None:
        if self._closed:
            return
        record = {k: _sanitize(v) for k, v in metrics.items()}
        try:
            self._jsonl_fh.write(json.dumps(record) + "\n")
            self._jsonl_fh.flush()
        except OSError as exc:
            if not self._oserror_warned:
                print(
                    f"[runlog] Warning: failed to write metrics.jsonl ({exc!r}); "
                    f"continuing best-effort.",
                    file=sys.stderr,
                )
                self._oserror_warned = True

    def close(self, status: str = "ok") -> None:
        if self._closed:
            return
        try:
            self._jsonl_fh.close()
        except Exception:
            pass
        self._meta["finished_at"] = _utc_iso8601()
        self._meta["status"] = status
        try:
            self._write_meta()
        except OSError as exc:
            print(
                f"[runlog] Warning: failed to update meta.json ({exc!r}).",
                file=sys.stderr,
            )
        self._closed = True

    def __enter__(self) -> "RunLogger":
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> bool:
        if exc_type is None:
            self.close("ok")
        elif issubclass(exc_type, KeyboardInterrupt):
            self.close("interrupted")
        else:
            self.close("error")
        return False  # never swallow exceptions
