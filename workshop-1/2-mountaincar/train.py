"""MountainCarContinuous-v0 — custom-PPO training driver.

Writes ``runs/mountaincar/<run-name>/{meta.json, metrics.jsonl, model.pt,
eval.mp4}`` per
``specs/002-training-and-visualization/contracts/run-format.md``.

The driver is a thin wrapper around the stage-1 ``ppo.train()`` function:

  1. Build env + agent.
  2. Open a ``RunLogger`` (writes meta.json, opens metrics.jsonl).
  3. Call ``ppo.train(...)`` with a custom ``log_fn`` that parses the
     formatted log line and emits one JSONL record per PPO update via the
     RunLogger.
  4. Run one greedy evaluation episode with video recording.
  5. Save model.

Usage::

    uv run python workshop-1/2-mountaincar/train.py
    uv run python workshop-1/2-mountaincar/train.py --timesteps 100000 --seed 7
    uv run python workshop-1/2-mountaincar/train.py --no-eval --run-name foo
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import gymnasium as gym
import numpy as np

_HERE = Path(__file__).resolve().parent
_WORKSHOP1 = _HERE.parent
sys.path.insert(0, str(_HERE))           # for agent.py
sys.path.insert(0, str(_WORKSHOP1))      # for _runlog, _eval, _log_parser
sys.path.insert(0, str(_WORKSHOP1 / "1-ppo"))  # for ppo

from agent import MountainCarPPOAgent  
from _runlog import RunLogger, RunDirectoryExistsError  
from _eval import record_eval_episode  
from _log_parser import make_log_fn  
from ppo import _seed_everything  

DEFAULT_TIMESTEPS = 200_000
DEFAULT_SEED = 42
ENV_ID = "MountainCarContinuous-v0"
STAGE = "mountaincar"


def main() -> int:
    parser = argparse.ArgumentParser(
        description=f"Custom PPO trainer for {ENV_ID}"
    )
    parser.add_argument("--timesteps", type=int, default=DEFAULT_TIMESTEPS)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--run-name", type=str, default=None)
    parser.add_argument("--no-eval", action="store_true")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    _seed_everything(args.seed)
    env = gym.make(ENV_ID)
    obs_dim = int(np.prod(env.observation_space.shape))
    action_dim = int(np.prod(env.action_space.shape))
    agent = MountainCarPPOAgent(env)

    runs_root = _WORKSHOP1.parent / "runs"
    
    runlog = RunLogger(
            stage=STAGE,
            hyperparameters=agent.hyperparameters,
            env_id=ENV_ID,
            agent_class=type(agent).__name__,
            seed=args.seed,
            total_timesteps=args.timesteps,
            run_name=args.run_name,
            force=args.force,
            runs_root=runs_root,
        )
    

    exit_code = 0
    print(agent.hyperparameters)
    print(f"action bounds: {agent.action_min} to {agent.action_max}")
    print(f"log_std init: {agent.log_std.data}")
    with runlog:
        log_fn = make_log_fn(runlog, agent)
        agent.train(
            env,
            total_timesteps=args.timesteps,
            log_fn=log_fn,
        )
        agent.save(str(runlog.run_dir / "model.pt"))
        if not args.no_eval:
            record_eval_episode(
                ENV_ID,
                agent.predict,
                runlog.run_dir,
                seed=args.seed,
                )
    
    
    env.close()

    return exit_code


if __name__ == "__main__":
    sys.exit(main())
