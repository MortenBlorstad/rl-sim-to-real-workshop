"""Pendulum-v1 — Stable-Baselines3 alternative training driver.

The constitutional escape hatch: if you couldn't finish the stage-1 PPO TODOs,
this driver still produces a working trained agent + the same canonical
``runs/<stage>/<run-name>/``.
"""

#### imports ###############
from __future__ import annotations
from dataclasses import dataclass
import argparse
import sys
from pathlib import Path
import warnings
warnings.filterwarnings("ignore", message="pkg_resources is deprecated")

_HERE = Path(__file__).resolve().parent
_WORKSHOP1 = _HERE.parent
sys.path.insert(0, str(_WORKSHOP1 / "1-ppo"))
from ppo.utils import silence_objc_dup_class_warnings
silence_objc_dup_class_warnings()

import gymnasium as gym




from ppo.utils import (  # noqa: E402
    RunDirectoryExistsError,
    RunLogger,
    Sb3JsonlCallback,
    record_eval_episode,
    get_device,
)

from stable_baselines3 import PPO  # noqa: E402

###############


DEFAULT_TIMESTEPS = 200_000
DEFAULT_SEED = 42
ENV_ID = "Pendulum-v1"
STAGE = "pendulum"

@dataclass
class EnvConfig:
    env_id: str = "Pendulum-v1"

    def make_train(self, seed: int = 0) -> gym.Env:
        return gym.make(self.env_id)

    def make_eval(self, seed: int = 0) -> gym.Env:
        return gym.make(self.env_id, render_mode="rgb_array")


def main() -> int:
    parser = argparse.ArgumentParser(
        description=f"SB3 PPO trainer for {ENV_ID}"
    )
    parser.add_argument("--timesteps", type=int, default=DEFAULT_TIMESTEPS)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--run-name", type=str, default=None)
    parser.add_argument("--no-eval", action="store_true")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--from-hub", action="store_true",
                    help="Load pretrained weights from HuggingFace sb3/ppo-Pendulum-v1")
    parser.add_argument("--eval-only", action="store_true",
                    help="Skip training, only run evaluation (useful with --from-hub)")
    args = parser.parse_args()

    cfg = EnvConfig()
    env = cfg.make_train(seed=args.seed)

    device = "cpu" #str(get_device()) # cpu is faster than gpu for this tiny model/env. 
    if args.from_hub:
        from huggingface_sb3 import load_from_hub
        sys.modules["gym"] = gym 

        checkpoint = load_from_hub(
            repo_id="sb3/ppo-Pendulum-v1",
            filename="ppo-Pendulum-v1.zip",
        )
        model = PPO.load(
            checkpoint,
            env=env,
            custom_objects={
                "observation_space": env.observation_space,
                "action_space": env.action_space,
                "clip_range": 0.2, 
            },
            device=device,
            tensorboard_log=None,
        )
        print("Loaded pretrained weights from sb3/ppo-Pendulum-v1")
    else:
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
            device=device,
            seed=args.seed,
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
            if not args.eval_only:
                callback = Sb3JsonlCallback(runlog)
                model.learn(total_timesteps=args.timesteps, callback=callback)
            model.save(str(runlog.run_dir / "model.zip"))
            if not args.no_eval:
                def predict_fn(obs, deterministic=True):
                    action, _ = model.predict(obs, deterministic=deterministic)
                    return action
                record_eval_episode(
                cfg.make_eval(args.seed),
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
