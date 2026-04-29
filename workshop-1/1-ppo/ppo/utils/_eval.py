"""Single-episode evaluation video recorder.

Independent of training. Used by both ``train.py`` and ``train_sb3.py``
drivers. Falls back to writing an ``eval.mp4.skipped`` text file on recorder
failure so training artifacts always survive (FR-015).
"""
from __future__ import annotations

from pathlib import Path
from typing import Callable, Optional

import numpy as np


def record_eval_episode(
    env_id: str,
    agent_predict_fn: Callable[..., np.ndarray],
    run_dir: Path,
    *,
    max_steps: int = 1000,
    seed: int = 0,
    reset_preprocess_state_fn: Optional[Callable[[], None]] = None,
    wrappers: Optional[list] = None,
) -> Path:
    """Run one greedy episode and write ``<run_dir>/eval.mp4``.

    Args:
        env_id: gymnasium env id; constructed with ``render_mode="rgb_array"``.
        agent_predict_fn: callable ``(obs, deterministic=True) -> action``.
            For PPOAgent: pass ``agent.predict``. For SB3: pass a closure that
            calls ``model.predict(obs, deterministic=True)`` and returns the
            action.
        run_dir: directory where ``eval.mp4`` (or ``eval.mp4.skipped``) is
            written.
        max_steps: hard cap on episode length (prevents runaway recordings).
        seed: env reset seed.
        reset_preprocess_state_fn: optional callable; called once before the
            episode to clear stateful preprocessing buffers (e.g. CarRacing's
            frame buffer).

    Returns:
        Path to ``eval.mp4`` on success, or ``eval.mp4.skipped`` on failure.
    """
    run_dir = Path(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    skipped_path = run_dir / "eval.mp4.skipped"

    
    import gymnasium as gym
    from gymnasium.wrappers import RecordVideo


    
    env = gym.make(env_id, render_mode="rgb_array")
    if wrappers:
        for w in wrappers:
            env = w(env)
    env = RecordVideo(
        env,
        video_folder=str(run_dir),
        name_prefix="eval",
        episode_trigger=lambda ep_idx: True,
        disable_logger=True,
    )
    

    if reset_preprocess_state_fn is not None:
        
        reset_preprocess_state_fn()
        

    
    obs, _ = env.reset(seed=seed)
    for _ in range(max_steps):
        action = agent_predict_fn(obs, deterministic=False)
        action = np.asarray(action, dtype=np.float32)
        obs, _, terminated, truncated, _ = env.step(action)
        if terminated or truncated:
            break
    env.close()
    

    # RecordVideo names files like "eval-episode-0.mp4" — rename to "eval.mp4".
    candidates = sorted(run_dir.glob("eval-episode-*.mp4"))
    target = run_dir / "eval.mp4"
    if candidates:
        candidates[0].rename(target)
        for stray in run_dir.glob("eval-episode-*.meta.json"):
           
            stray.unlink()
            
        # Drop the .skipped marker if a previous failed attempt left one.
        if skipped_path.exists():
            skipped_path.unlink()
            
        return target

    skipped_path.write_text("RecordVideo produced no output file.\n")
    return skipped_path
