from __future__ import annotations

import argparse
import sys
import warnings
from pathlib import Path

warnings.filterwarnings("ignore", message="pkg_resources is deprecated")


_HERE = Path(__file__).resolve().parent
_WORKSHOP1 = _HERE.parent
sys.path.insert(0, str(_WORKSHOP1 / "1-ppo"))
from ppo.utils import silence_objc_dup_class_warnings 

silence_objc_dup_class_warnings()

import gymnasium as gym
from gymnasium.vector import AutoresetMode
from gymnasium.wrappers import (
    GrayscaleObservation,
    ResizeObservation,
    FrameStackObservation
)



from ppo import PPOAgent  
from ppo.utils import (  
    
    RunLogger,
    make_log_fn,
    seed_everything,
)


DEFAULT_TIMESTEPS = 200_000
ENV_ID = "CarRacing-v3"
STAGE = "car-racing"
NUM_ENVS = 4

hyperparameters: dict = {
    "rollout_size": 2048,
    "n_epochs": 4,
    "batch_size": 64,
    "lr": 3e-4,
    "gamma": 0.99,
    "gae_lambda": 0.95,
    "clip_eps": 0.2,
    "value_coef": 0.5,
    "entropy_coef": 0.01,
    "max_grad_norm": 0.5,
    "log_std_init": -0.5,
    "random_state": 42,
}


def obs_to_state(env: gym.Env) -> gym.Env:
    """Per-env wrapper chain: Resize → Grayscale (2D) → (FrameStack added by
    gym.make_vec via the ``wrappers=`` arg, see main()).

    keep_dim=False means grayscale output is (H, W) 2D so a downstream
    FrameStackObservation(4) gives (4, 84, 84) CHW — the shape PPOAgent's
    CnnActorNetwork expects.
    """
    env = ResizeObservation(env, (84, 84))
    env = GrayscaleObservation(env, keep_dim=False)
    env = FrameStackObservation(env, 4)
    return env





def main() -> int:
    parser = argparse.ArgumentParser(description=f"Custom PPO trainer for {ENV_ID}")
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
        wrappers=[obs_to_state],
        vector_kwargs={"autoreset_mode": AutoresetMode.SAME_STEP},
    )
    agent = PPOAgent(env, hyperparameters=hyperparameters, device="auto")

    runs_root = _WORKSHOP1.parent / "runs"
    
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
        network_arch=agent.network_arch,
    )
    
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
    
    env.close()

    return 0


if __name__ == "__main__":
    sys.exit(main())
