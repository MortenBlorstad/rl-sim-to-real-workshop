"""MountainCarContinuous-v0 — custom-PPO training driver.

Writes ``runs/mountaincar/<run-name>/{meta.json, metrics.jsonl, model.pt,
eval.mp4}``.

Thin wrapper around the stage-1 ``PPOAgent.train()`` and ``PPOAgent.evaluate()``:

  1. Build env (with the ``NormalizeObs`` driver-level wrapper) and agent.
  2. Open a ``RunLogger`` (writes meta.json, opens metrics.jsonl).
  3. Call ``agent.train(...)`` with a ``log_fn`` that parses each formatted
     log line into a JSONL record via the RunLogger.
  4. ``agent.save(<run-dir>/model.pt)``.
  5. Unless ``--no-eval``: ``agent.evaluate(env, n_episodes=1, record_video=True,
     video_dir=<run-dir>)`` → ``eval.mp4``.

Usage::

    uv run python workshop-1/2-mountaincar/train.py
    uv run python workshop-1/2-mountaincar/train.py --timesteps 4096 --run-name smoke --force

The seed is configured in source via ``hyperparameters["random_state"]``;
there is intentionally no ``--seed`` CLI flag.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import gymnasium as gym
from gymnasium.vector import AutoresetMode

_HERE = Path(__file__).resolve().parent
_WORKSHOP1 = _HERE.parent
sys.path.insert(0, str(_WORKSHOP1 / "1-ppo"))

from ppo import PPOAgent
from ppo.utils import (
    RunDirectoryExistsError,
    RunLogger,
    make_log_fn,
    seed_everything,
)


DEFAULT_TIMESTEPS = 200_000
ENV_ID = "Pendulum-v1"
STAGE = "pendulum"
NUM_ENVS = 4

hyperparameters: dict = {
    "rollout_size": 2048,
    "n_epochs": 10,
    "batch_size": 64,
    "lr": 1e-3,
    "gamma": 0.98,
    "gae_lambda": 0.95,
    "clip_eps": 0.2,
    "value_coef": 0.5,
    "entropy_coef": 0.0,
    "max_grad_norm": 0.5,
    "log_std_init": 0.0,
    "random_state": 42,
}


class ObsToState(gym.ObservationWrapper):
    

    def __init__(self, env: gym.Env):
        super().__init__(env)
        self.obs_low = self.observation_space.low
        self.obs_high = self.observation_space.high

    def observation(self, obs):
        return obs


def main() -> int:
    parser = argparse.ArgumentParser(
        description=f"Custom PPO trainer for {ENV_ID}"
    )
    parser.add_argument("--timesteps", type=int, default=DEFAULT_TIMESTEPS)
    parser.add_argument("--run-name", type=str, default=None)
    parser.add_argument("--no-eval", action="store_true")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    seed = int(hyperparameters["random_state"])
    seed_everything(seed)

    env = gym.make_vec(
        ENV_ID,
        num_envs=NUM_ENVS,
        vectorization_mode="sync",
        wrappers=[ObsToState],
        vector_kwargs={"autoreset_mode": AutoresetMode.SAME_STEP},
    )
    agent = PPOAgent(env, hyperparameters=hyperparameters)

    runs_root = _WORKSHOP1.parent / "runs"
    try:
        runlog = RunLogger(
            stage=STAGE,
            hyperparameters=hyperparameters,
            env_id=ENV_ID,
            agent_class=type(agent).__name__,
            seed=seed,
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
            log_fn = make_log_fn(runlog, agent)
            agent.train(
                env,
                total_timesteps=args.timesteps,
                random_state=seed,
                log_fn=log_fn,
            )
            agent.save(str(runlog.run_dir / "model.pt"))
            if not args.no_eval:
                agent.evaluate(
                    env,
                    n_episodes=1,
                    record_video=True,
                    video_dir=runlog.run_dir,
                )
    except KeyboardInterrupt:
        print("\n[train] interrupted by user", file=sys.stderr)
        exit_code = 130
    except NotImplementedError as exc:
        print(
            "[train] Looks like a TODO is still raising NotImplementedError. "
            "Fill it in and re-run.",
            file=sys.stderr,
        )
        print(f"[train] underlying error: {exc}", file=sys.stderr)
        exit_code = 3
    except Exception as exc:
        print(f"[train] error: {exc!r}", file=sys.stderr)
        exit_code = 2
    finally:
        env.close()

    return exit_code


if __name__ == "__main__":
    sys.exit(main())
