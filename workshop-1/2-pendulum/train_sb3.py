"""Pendulum-v1 — Stable-Baselines3 alternative training driver.

The constitutional escape hatch: if you couldn't finish the stage-1 PPO TODOs,
this driver still produces a working trained agent + the same canonical
``runs/<stage>/<run-name>/`` layout that ``analyze.ipynb`` expects.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import gymnasium as gym

_HERE = Path(__file__).resolve().parent
_WORKSHOP1 = _HERE.parent
sys.path.insert(0, str(_WORKSHOP1 / "1-ppo"))

from ppo.utils import (  # noqa: E402
    RunDirectoryExistsError,
    RunLogger,
    Sb3JsonlCallback,
    record_eval_episode,
)

from stable_baselines3 import PPO  # noqa: E402

DEFAULT_TIMESTEPS = 200_000
DEFAULT_SEED = 42
ENV_ID = "Pendulum-v1"
STAGE = "pendulum"


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

    env = gym.make(ENV_ID)
    model = PPO(
        "MlpPolicy",
        env,
        gamma=0.98,
        # Using https://proceedings.mlr.press/v164/raffin22a.html
        use_sde=False,
        #sde_sample_freq=4,
        learning_rate=1e-3,
        n_epochs=10,
        ent_coef=0.0,
        clip_range=0.2,
        n_steps=1024,
        verbose=1,
    )

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
            agent_class="sb3.PPO[MlpPolicy]",
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
