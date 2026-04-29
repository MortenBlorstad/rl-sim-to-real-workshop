"""CarRacing-v3 — Stable-Baselines3 alternative training driver.

The constitutional escape hatch for stage 3 (Article VI). Builds a 4-env
SyncVectorEnv with the canonical ``Grayscale → Resize(84, 84) → FrameStack(4)``
wrapper chain, trains SB3's ``PPO("CnnPolicy", ...)``, writes the canonical
run directory under ``runs/car-racing/<run-name>/``.

Optional ``--hf-repo``: download a pretrained checkpoint from HuggingFace Hub,
initialise SB3's PPO from those weights, and continue training (fine-tune)
for ``--timesteps`` steps before evaluating. See
``specs/005-carracing-drivers/contracts/cli.md`` for full flag documentation.

Usage::

    # From-scratch
    uv run python workshop-1/3-car-racing/train_sb3.py --timesteps 10000 --run-name smoke --force

    # Fine-tune from HuggingFace
    uv run python workshop-1/3-car-racing/train_sb3.py \
        --hf-repo sb3/ppo-CarRacing-v0 --timesteps 10000 --run-name finetune --force
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import gymnasium as gym
from gymnasium.wrappers import (
    FrameStackObservation,
    GrayscaleObservation,
    ResizeObservation,
)

_HERE = Path(__file__).resolve().parent
_WORKSHOP1 = _HERE.parent
sys.path.insert(0, str(_HERE))
sys.path.insert(0, str(_WORKSHOP1 / "1-ppo"))

from ppo.utils import (  # noqa: E402
    RunDirectoryExistsError,
    RunLogger,
    Sb3JsonlCallback,
    get_device,
    record_eval_episode,
)

from stable_baselines3 import PPO  # noqa: E402
from stable_baselines3.common.env_util import make_vec_env  # noqa: E402
from stable_baselines3.common.vec_env import VecFrameStack  # noqa: E402

DEFAULT_TIMESTEPS = 200_000
DEFAULT_SEED = 42
ENV_ID = "CarRacing-v3"
STAGE = "car-racing"
NUM_ENVS = 4


def _make_carracing_base(**kwargs) -> gym.Env:
    """Per-env factory used by SB3's make_vec_env. Applies the per-env half
    of the wrapper chain (grayscale + resize, channel-last HWC). The frame
    stack is added on the vec side via VecFrameStack."""
    env = gym.make(ENV_ID, **kwargs)
    # Resize first (on RGB), then grayscale. The reverse order trips
    # ResizeObservation on Gymnasium 1.x because cv2 collapses the channel
    # dim when resizing a (H, W, 1) array.
    env = ResizeObservation(env, (84, 84))
    env = GrayscaleObservation(env, keep_dim=True)
    return env


def main() -> int:
    parser = argparse.ArgumentParser(description=f"SB3 PPO trainer for {ENV_ID}")
    parser.add_argument("--timesteps", type=int, default=DEFAULT_TIMESTEPS)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--run-name", type=str, default=None)
    parser.add_argument("--no-eval", action="store_true")
    parser.add_argument("--force", action="store_true")
    parser.add_argument(
        "--hf-repo",
        type=str,
        default=None,
        help="HuggingFace repo id (e.g. 'sb3/ppo-CarRacing-v0'). When set, "
        "fine-tune from this checkpoint instead of training from scratch.",
    )
    parser.add_argument(
        "--hf-filename",
        type=str,
        default=None,
        help="Filename within the HF repo. Defaults to '<basename(repo_id)>.zip'. "
        "Requires --hf-repo.",
    )
    args = parser.parse_args()

    if args.hf_filename and not args.hf_repo:
        parser.error("--hf-filename requires --hf-repo")

    env = make_vec_env(
        _make_carracing_base,
        n_envs=NUM_ENVS,
        seed=args.seed,
    )
    env = VecFrameStack(env, n_stack=4, channels_order="last")

    device = str(get_device())
    resolved_filename = args.hf_filename
    if args.hf_repo is not None:
        from _huggingface import (  # noqa: E402  (sibling-of-driver import)
            HuggingFaceLoadError,
            _default_filename,
            download_pretrained,
        )

        resolved_filename = args.hf_filename or _default_filename(args.hf_repo)
        try:
            local_path = download_pretrained(args.hf_repo, resolved_filename)
        except HuggingFaceLoadError as exc:
            print(f"Error: {exc}", file=sys.stderr)
            env.close()
            return 1
        try:
            model = PPO.load(local_path, env=env, device=device)
        except (RuntimeError, ValueError, ModuleNotFoundError, ImportError) as exc:
            print(
                f"Error: HuggingFace checkpoint {args.hf_repo!r}/{resolved_filename!r} "
                f"could not be loaded into the local SB3 PPO config.\n"
                f"Underlying error: {type(exc).__name__}: {exc}\n"
                f"\n"
                f"Common causes:\n"
                f"  - Architecture/hyperparameter mismatch (different network width or n_steps).\n"
                f"  - Old SB3 v1.x checkpoint that references the legacy 'gym' module\n"
                f"    instead of 'gymnasium'. Pick a checkpoint trained with SB3 >= 2.0.\n"
                f"\n"
                f"Try a different repo, or fall back to from-scratch training "
                f"(omit --hf-repo).",
                file=sys.stderr,
            )
            env.close()
            return 1
        agent_class = "sb3.PPO[CnnPolicy]+hf-finetune"
    else:
        model = PPO(
            "CnnPolicy",
            env,
            seed=args.seed,
            device=device,
            verbose=0,
        )
        agent_class = "sb3.PPO[CnnPolicy]"

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
            agent_class=agent_class,
            seed=args.seed,
            total_timesteps=args.timesteps,
            run_name=args.run_name,
            force=args.force,
            runs_root=runs_root,
            network_arch="cnn",
            hf_repo_id=args.hf_repo,        # None for from-scratch
            hf_filename=resolved_filename,   # auto-derived when only --hf-repo is set
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

                # Eval env: keep_dim=False so grayscale gives (H, W) 2D,
                # then FrameStackObservation gives (4, H, W) CHW which is
                # the shape SB3's policy expects (channels-first after the
                # internal VecTransposeImage during training).
                eval_wrappers = [
                    lambda e: ResizeObservation(e, (84, 84)),
                    lambda e: GrayscaleObservation(e, keep_dim=False),
                    lambda e: FrameStackObservation(e, 4),
                ]
                record_eval_episode(
                    ENV_ID,
                    predict_fn,
                    runlog.run_dir,
                    seed=args.seed,
                    wrappers=eval_wrappers,
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
