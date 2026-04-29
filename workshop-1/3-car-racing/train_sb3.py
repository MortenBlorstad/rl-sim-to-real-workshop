"""CarRacing-v3 — Stable-Baselines3 alternative training driver.

The constitutional escape hatch for stage 3. Uses standalone Gymnasium
observation wrappers (``Grayscale`` → ``Resize`` → ``FrameStack``) to deliver
``(4, 84, 84)`` observations directly to SB3's closed training loop. This is
the **scoped Article II exemption** documented in spec FR-013.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import gymnasium as gym

_HERE = Path(__file__).resolve().parent
_WORKSHOP1 = _HERE.parent
sys.path.insert(0, str(_WORKSHOP1))

from _runlog import RunLogger, RunDirectoryExistsError  # noqa: E402
from _eval import record_eval_episode  # noqa: E402
from _sb3_jsonl_callback import Sb3JsonlCallback  # noqa: E402

from stable_baselines3 import PPO  # noqa: E402

DEFAULT_TIMESTEPS = 200_000
DEFAULT_SEED = 42
ENV_ID = "CarRacing-v3"
STAGE = "car-racing"


def _make_wrapped_env(seed: int) -> gym.Env:
    """Construct CarRacing with grayscale → 84x84 → 4-frame stack wrappers."""
    env = gym.make(ENV_ID, render_mode="rgb_array")

    # gymnasium ≥ 1.0 renamed several wrappers. Try the new names first, then
    # fall back to the legacy names.
    try:
        from gymnasium.wrappers import GrayscaleObservation as _Gray
    except ImportError:
        from gymnasium.wrappers import GrayScaleObservation as _Gray  # type: ignore[no-redef]
    from gymnasium.wrappers import ResizeObservation
    try:
        from gymnasium.wrappers import FrameStackObservation as _FrameStack
    except ImportError:
        from gymnasium.wrappers import FrameStack as _FrameStack  # type: ignore[no-redef]

    env = _Gray(env, keep_dim=False)
    env = ResizeObservation(env, (84, 84))
    env = _FrameStack(env, 4)
    env.reset(seed=seed)
    return env


def main() -> int:
    parser = argparse.ArgumentParser(
        description=f"SB3 PPO trainer for {ENV_ID}"
    )
    parser.add_argument("--timesteps", type=int, default=DEFAULT_TIMESTEPS)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--run-name", type=str, default=None)
    parser.add_argument("--no-eval", action="store_true")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    env = _make_wrapped_env(args.seed)
    model = PPO("CnnPolicy", env, seed=args.seed, verbose=0)

    runs_root = _WORKSHOP1.parent / "runs"
    try:
        runlog = RunLogger(
            stage=STAGE,
            hyperparameters={
                "lr": float(model.lr_schedule(1.0)),
                "n_steps": int(model.n_steps),
                "batch_size": int(model.batch_size),
                "n_epochs": int(model.n_epochs),
                "gamma": float(model.gamma),
                "gae_lambda": float(model.gae_lambda),
                "clip_range": float(model.clip_range(1.0)),
            },
            env_id=ENV_ID,
            agent_class="sb3.PPO[CnnPolicy]",
            seed=args.seed,
            total_timesteps=args.timesteps,
            run_name=args.run_name,
            force=args.force,
            runs_root=runs_root,
        )
    except RunDirectoryExistsError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        env.close()
        return 1

    exit_code = 0
    try:
        with runlog:
            callback = Sb3JsonlCallback(runlog)
            model.learn(total_timesteps=args.timesteps, callback=callback)
            model.save(str(runlog.run_dir / "model.zip"))
            if not args.no_eval:
                def predict_fn(obs, deterministic=True):
                    action, _ = model.predict(obs, deterministic=deterministic)
                    return action
                # Eval on the same wrapped env shape so SB3's CnnPolicy can
                # consume the observations.
                record_eval_episode(
                    ENV_ID,
                    predict_fn,
                    runlog.run_dir,
                    seed=args.seed,
                )
    except KeyboardInterrupt:
        print("\n[train_sb3] interrupted by user", file=sys.stderr)
        exit_code = 130
    except Exception as exc:
        print(f"[train_sb3] error: {exc!r}", file=sys.stderr)
        exit_code = 2
    finally:
        env.close()

    return exit_code


if __name__ == "__main__":
    sys.exit(main())
