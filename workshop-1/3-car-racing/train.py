"""CarRacing-v3 — custom-PPO training driver.

Writes ``runs/car-racing/<run-name>/{meta.json, metrics.jsonl, model.pt,
eval.mp4}`` per
``specs/002-training-and-visualization/contracts/run-format.md``.

NOTE — current state: the stage-1 ``ppo.train()`` function does NOT call
``agent.preprocess()`` during its rollout loop. Until that integration lands,
the inner ``ppo.train(...)`` call here will raise on shape mismatch (the CNN
expects ``(4, 84, 84)`` preprocessed observations; the env emits raw
``(96, 96, 3)`` frames). The driver wraps the call in ``try/except`` so that
the run directory still completes — meta.json + a (possibly empty)
metrics.jsonl are written, ``model.pt`` records the random-init weights, and
``eval.mp4`` is recorded via ``agent.predict`` (which DOES call
``preprocess`` internally), so the eval pipeline is verifiable today.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import gymnasium as gym
import numpy as np

_HERE = Path(__file__).resolve().parent
_WORKSHOP1 = _HERE.parent
sys.path.insert(0, str(_HERE))
sys.path.insert(0, str(_WORKSHOP1))
sys.path.insert(0, str(_WORKSHOP1 / "1-ppo"))

from agent import CarRacingPPOAgent  # noqa: E402
from _runlog import RunLogger, RunDirectoryExistsError  # noqa: E402
from _eval import record_eval_episode  # noqa: E402
from _log_parser import make_log_fn  # noqa: E402
from ppo import train as ppo_train, _seed_everything  # noqa: E402

DEFAULT_TIMESTEPS = 200_000
DEFAULT_SEED = 42
ENV_ID = "CarRacing-v3"
STAGE = "car-racing"


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
    env = gym.make(ENV_ID, render_mode="rgb_array")
    obs_dim = int(np.prod(env.observation_space.shape))
    action_dim = int(np.prod(env.action_space.shape))
    agent = CarRacingPPOAgent(obs_dim, action_dim)

    runs_root = _WORKSHOP1.parent / "runs"
    try:
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
    except RunDirectoryExistsError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        env.close()
        return 1

    exit_code = 0
    try:
        with runlog:
            log_fn = make_log_fn(runlog, agent)
            try:
                ppo_train(
                    env,
                    agent.actor,
                    agent.value,
                    agent.log_std,
                    total_timesteps=args.timesteps,
                    seed=args.seed,
                    log_fn=log_fn,
                    **agent.hyperparameters,
                )
            except Exception as exc:
                # Expected today: stage-1 train() doesn't call agent.preprocess(),
                # so passing raw frames into the CNN raises a shape error. Surface
                # clearly and continue to eval — run dir still completes.
                print(
                    f"[train] WARNING: training failed "
                    f"({exc.__class__.__name__}: {exc}); continuing to eval with "
                    f"current (random) weights. This is expected until ppo.train() "
                    f"learns to call agent.preprocess() — see spec FR-001b.",
                    file=sys.stderr,
                )
            agent.save(str(runlog.run_dir / "model.pt"))
            if not args.no_eval:
                # CarRacing's preprocess holds a stateful frame buffer; clear it
                # before the eval episode so warmup frames aren't leftover.
                def reset_state():
                    if hasattr(agent, "_frame_buffer"):
                        agent._frame_buffer.clear()
                record_eval_episode(
                    ENV_ID,
                    agent.predict,
                    runlog.run_dir,
                    seed=args.seed,
                    reset_preprocess_state_fn=reset_state,
                )
    except KeyboardInterrupt:
        print("\n[train] interrupted by user", file=sys.stderr)
        exit_code = 130
    except Exception as exc:
        print(f"[train] error: {exc!r}", file=sys.stderr)
        exit_code = 2
    finally:
        env.close()

    return exit_code


if __name__ == "__main__":
    sys.exit(main())
