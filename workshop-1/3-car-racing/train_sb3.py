"""CarRacing-v3 — Stable-Baselines3 alternative training driver.

The constitutional escape hatch for stage 3 (Article VI). Builds a 4-env
SyncVectorEnv with the canonical ``Grayscale → Resize(84, 84) → FrameStack(4)``
wrapper chain, trains SB3's ``PPO("CnnPolicy", ...)``, writes the canonical
run directory under ``runs/car-racing/<run-name>/``.
"""

#### imports ###############
from __future__ import annotations
from dataclasses import dataclass
import argparse
import sys
import warnings
from pathlib import Path
warnings.filterwarnings("ignore", message="pkg_resources is deprecated")

_HERE = Path(__file__).resolve().parent
_WORKSHOP1 = _HERE.parent
sys.path.insert(0, str(_HERE))
sys.path.insert(0, str(_WORKSHOP1 / "1-ppo"))

from ppo.utils import silence_objc_dup_class_warnings
silence_objc_dup_class_warnings()

import gymnasium as gym
from gymnasium.wrappers import (
    FrameStackObservation,
    GrayscaleObservation,
    ResizeObservation,
)


from ppo.utils import (  
    RunDirectoryExistsError,
    RunLogger,
    Sb3JsonlCallback,
    get_device,
    record_eval_episode,
)

from stable_baselines3 import PPO  
from stable_baselines3.common.env_util import make_vec_env 
from stable_baselines3.common.vec_env import VecFrameStack  
import numpy as np
###############


DEFAULT_TIMESTEPS = 100
DEFAULT_SEED = 42
ENV_ID = "CarRacing-v3"
STAGE = "car-racing"
NUM_ENVS = 4


class FrameSkip(gym.Wrapper):
    def __init__(self, env, skip=2):
        super().__init__(env)
        self._skip = skip

    def step(self, action):
        total_reward = 0.0
        for _ in range(self._skip):
            obs, reward, terminated, truncated, info = self.env.step(action)
            total_reward += reward
            if terminated or truncated:
                break
        return obs, total_reward, terminated, truncated, info

def make_env(resize=64, frame_skip=None, **kwargs) -> gym.Env:
    """Per-env factory. Wrapper chain is configurable to match HuggingFace checkpoint."""
    env = gym.make(ENV_ID, **kwargs)
    if frame_skip is not None:
        env = FrameSkip(env, skip=frame_skip)
    env = ResizeObservation(env, (resize, resize))
    env = GrayscaleObservation(env, keep_dim=True)
    return env





@dataclass
class EnvConfig:
    env_id: str = "CarRacing-v3"
    resize: int = 64
    frame_skip: int | None = None
    n_stack: int = 2
    n_envs: int = 4

    # Presets
    @classmethod
    def fresh(cls) -> "EnvConfig":
        return cls()

    @classmethod
    def hub(cls) -> "EnvConfig":
        """Match sb3/ppo-CarRacing-v0 checkpoint."""
        return cls(resize=64, frame_skip=2, n_stack=2)

    def make_single(self, **kwargs) -> gym.Env:
        env = gym.make(self.env_id, **kwargs)
        if self.frame_skip is not None:
            env = FrameSkip(env, skip=self.frame_skip)
        env = ResizeObservation(env, (self.resize, self.resize))
        env = GrayscaleObservation(env, keep_dim=True)
        return env

    def make_train(self, seed: int = 0):
        env = make_vec_env(self.make_single, n_envs=self.n_envs, seed=seed)
        return VecFrameStack(env, n_stack=self.n_stack, channels_order="last")

    def make_eval(self, seed: int = 0) -> gym.Env:
        env = gym.make(self.env_id, render_mode="rgb_array")
        if self.frame_skip is not None:
            env = FrameSkip(env, skip=self.frame_skip)
        env = ResizeObservation(env, (self.resize, self.resize))
        env = GrayscaleObservation(env, keep_dim=False)
        env = FrameStackObservation(env, self.n_stack)
        return env



def main() -> int:
    parser = argparse.ArgumentParser(description=f"SB3 PPO trainer for {ENV_ID}")
    parser.add_argument("--timesteps", type=int, default=DEFAULT_TIMESTEPS)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--run-name", type=str, default=None)
    parser.add_argument("--no-eval", action="store_true")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--from-hub", action="store_true",
                    help="Load pretrained weights from HuggingFace sb3/ppo-CarRacing-v0")
    parser.add_argument("--eval-only", action="store_true",
                    help="Skip training, only run evaluation (useful with --from-hub)")
    args = parser.parse_args()

    

    
    cfg = EnvConfig.hub() if args.from_hub else EnvConfig.fresh()
    env = cfg.make_train(seed=args.seed)
    device = str(get_device())

    if args.from_hub:
        from huggingface_sb3 import load_from_hub
        sys.modules["gym"] = gym 

        checkpoint = load_from_hub(
            repo_id="sb3/ppo-CarRacing-v0",
            filename="ppo-CarRacing-v0.zip",
        )
        obs_space_chw = gym.spaces.Box(
            low=0, high=255,
            shape=(cfg.n_stack, cfg.resize, cfg.resize),
            dtype=np.uint8,
        )
        model = PPO.load(
            checkpoint,
            env=env,
            custom_objects={
                "observation_space": obs_space_chw,
                "action_space": env.action_space,
                "learning_rate": 1e-4,
                "lr_schedule": lambda _: 1e-4,
                "clip_range": 0.2,
                "use_sde": True,
                "sde_sample_freq": 4,
            },
            device=device,
            tensorboard_log=None,
        )
        
        
        print("Loaded pretrained weights from sb3/ppo-CarRacing-v0")
    else:
        model = PPO(
            "CnnPolicy",
            env,
            seed=args.seed,
            device=device,
            verbose=1,
        )
   

    runs_root = _WORKSHOP1.parent / "runs"
    
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
        network_arch="cnn",
    )
    
    
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
   
    env.close()

    return 0


if __name__ == "__main__":
    sys.exit(main())
