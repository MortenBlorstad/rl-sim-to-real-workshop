"""Agent contract test for the refactored ``PPOAgent``.

Run::

    uv run python workshop-1/1-ppo/ppo/tests/test_agent_interface.py --agent ppo

(``--agent sb3`` is reserved for a follow-up spec and exits 2.)

Verifies the agent contract on the refactored package:
    C1.  registry membership
    C3.  predict accepts raw observations and produces a valid action
         (and is deterministic when ``deterministic=True``)
    C4.  train method exists and returns the expected stats dict
    C5.  save/load round-trip including subclass class restoration via
         ``_AGENT_REGISTRY``
    C7.  evaluate returns a list of episode returns; with
         ``record_video=True`` writes ``eval.mp4`` (or ``eval.mp4.skipped``)

C2 (preprocess identity/determinism) and C6 (_get/_set_preprocess_state)
from the pre-refactor convention are intentionally omitted — those
methods no longer exist after the refactor.

Spec:    specs/003-fix-mountaincar-train/spec.md
Contract: specs/003-fix-mountaincar-train/contracts/tests.md
"""
from __future__ import annotations

import argparse
import os
import sys
from typing import Callable

# Make ``ppo`` importable regardless of current working directory.
_HERE = os.path.dirname(os.path.abspath(__file__))
_PACKAGE_ROOT = os.path.abspath(os.path.join(_HERE, "..", ".."))
if _PACKAGE_ROOT not in sys.path:
    sys.path.insert(0, _PACKAGE_ROOT)


STEPS: dict[int, tuple[str, Callable[[], None]]] = {}


def step(n: int, name: str):
    def decorator(fn: Callable[[], None]) -> Callable[[], None]:
        if n in STEPS:
            raise RuntimeError(f"step {n} already registered as {STEPS[n][0]!r}")
        STEPS[n] = (name, fn)
        return fn

    return decorator


def _run_step(n: int) -> str:
    name, fn = STEPS[n]
    try:
        fn()
    except NotImplementedError as e:
        print(f"TEST {n} not yet implemented ({name}): {e}")
        return "NOT_IMPLEMENTED"
    except AssertionError as e:
        print(f"FAIL: TEST {n} ({name}): {e}")
        return "FAIL"
    print(f"TEST {n} OK! ({name})")
    return "PASS"


# ===========================================================================
# C1 — registry membership
# ===========================================================================


@step(1, "C1 registry contains PPOAgent")
def test_registry_contains_ppoagent() -> None:
    from ppo import _AGENT_REGISTRY

    assert "PPOAgent" in _AGENT_REGISTRY, (
        f"PPOAgent must be registered via @register_agent. "
        f"Currently registered: {sorted(_AGENT_REGISTRY)}."
    )


# ===========================================================================
# C3 — predict
# ===========================================================================


@step(2, "C3 predict raw obs shape/dtype/range")
def test_predict_raw_obs_shape_dtype_range() -> None:
    import gymnasium as gym
    import numpy as np

    from ppo import PPOAgent

    env = gym.make("MountainCarContinuous-v0")
    try:
        agent = PPOAgent(env, hyperparameters={"random_state": 0})
        raw = env.observation_space.sample()
        action = agent.predict(raw)

        assert isinstance(action, np.ndarray), (
            f"predict must return a numpy array, got {type(action).__name__}"
        )
        assert action.shape == (agent.action_dim,), (
            f"action shape: expected ({agent.action_dim},), got {action.shape}"
        )
        assert action.dtype == np.float32, (
            f"action dtype: expected float32, got {action.dtype}"
        )
        amin = float(agent.action_min.min().item())
        amax = float(agent.action_max.max().item())
        val = float(action[0])
        assert amin <= val <= amax, (
            f"action {val} not in [{amin}, {amax}]. "
            f"Did you forget to clamp in sample_action?"
        )
    finally:
        env.close()


@step(3, "C3 predict deterministic flag")
def test_predict_deterministic_flag() -> None:
    import gymnasium as gym
    import numpy as np

    from ppo import PPOAgent

    env = gym.make("MountainCarContinuous-v0")
    try:
        agent = PPOAgent(env, hyperparameters={"random_state": 0})
        raw = np.array([0.1, -0.2], dtype=np.float32)
        a1 = agent.predict(raw, deterministic=True)
        a2 = agent.predict(raw, deterministic=True)
        assert np.array_equal(a1, a2), (
            f"deterministic=True must produce the same action across calls; "
            f"got {a1!r} then {a2!r}"
        )
    finally:
        env.close()


# ===========================================================================
# C4 — train smoke
# ===========================================================================


@step(4, "C4 train method smoke")
def test_train_method_smoke() -> None:
    import gymnasium as gym
    from gymnasium.vector import AutoresetMode

    from ppo import PPOAgent

    env = gym.make_vec(
        "MountainCarContinuous-v0",
        num_envs=2,
        vectorization_mode="sync",
        vector_kwargs={"autoreset_mode": AutoresetMode.SAME_STEP},
    )
    try:
        agent = PPOAgent(
            env,
            hyperparameters={
                "rollout_size": 256,
                "n_epochs": 2,
                "batch_size": 64,
                "random_state": 0,
            },
        )

        stats = agent.train(env, total_timesteps=512, random_state=0)

        assert isinstance(stats, dict), (
            f"PPOAgent.train must return a dict, got {type(stats).__name__}"
        )
        required = {"mean_reward", "policy_loss", "value_loss", "entropy", "n_updates"}
        missing = required - set(stats.keys())
        assert not missing, (
            f"stats dict missing required keys: {sorted(missing)}. "
            f"Got keys: {sorted(stats.keys())}."
        )
    finally:
        env.close()


# ===========================================================================
# C5 — save / load round-trip
# ===========================================================================


@step(5, "C5 save/load round-trip (base class)")
def test_save_load_roundtrip_base() -> None:
    import os
    import tempfile

    import gymnasium as gym
    import numpy as np

    from ppo import PPOAgent

    env = gym.make("MountainCarContinuous-v0")
    try:
        agent = PPOAgent(env, hyperparameters={"random_state": 0})
        raw = np.array([0.3, -0.4], dtype=np.float32)
        expected = agent.predict(raw, deterministic=True)

        fd, path = tempfile.mkstemp(suffix=".pt")
        os.close(fd)
        try:
            agent.save(path)
            loaded = PPOAgent.load(path, env)
            assert type(loaded).__name__ == "PPOAgent", (
                f"loaded agent class: expected PPOAgent, got {type(loaded).__name__}"
            )
            actual = loaded.predict(raw, deterministic=True)
            assert np.array_equal(actual, expected), (
                f"loaded agent produces a different action than the original. "
                f"original={expected!r}, loaded={actual!r}. "
                f"Save/load is dropping state somewhere — check that actor + critic "
                f"+ log_std are all serialized."
            )
        finally:
            if os.path.exists(path):
                os.remove(path)
    finally:
        env.close()


@step(6, "C5 subclass class restored on load")
def test_save_load_roundtrip_subclass_class_restored() -> None:
    import os
    import tempfile

    import gymnasium as gym

    from ppo import PPOAgent, register_agent

    @register_agent
    class _RoundTripAgent(PPOAgent):
        pass

    env = gym.make("MountainCarContinuous-v0")
    try:
        agent = _RoundTripAgent(env, hyperparameters={"random_state": 0})
        fd, path = tempfile.mkstemp(suffix=".pt")
        os.close(fd)
        try:
            agent.save(path)
            loaded = PPOAgent.load(path, env)
            assert type(loaded).__name__ == "_RoundTripAgent", (
                f"loaded class: expected _RoundTripAgent, got {type(loaded).__name__}. "
                f"PPOAgent.load must look up the class via _AGENT_REGISTRY by name."
            )
        finally:
            if os.path.exists(path):
                os.remove(path)
    finally:
        env.close()


# ===========================================================================
# C7 — evaluate (new in 003-fix-mountaincar-train)
# ===========================================================================


@step(7, "C7 evaluate returns finite list and writes video artifact")
def test_evaluate() -> None:
    import math
    import tempfile
    from pathlib import Path

    import gymnasium as gym

    from ppo import PPOAgent

    env = gym.make("MountainCarContinuous-v0")
    try:
        agent = PPOAgent(env, hyperparameters={"random_state": 0})

        # No-video path.
        returns = agent.evaluate(env, n_episodes=2, record_video=False)
        assert isinstance(returns, list), (
            f"evaluate must return a list, got {type(returns).__name__}"
        )
        assert len(returns) == 2, (
            f"evaluate(n_episodes=2) must return 2 returns, got {len(returns)}"
        )
        for i, r in enumerate(returns):
            assert isinstance(r, float), (
                f"returns[{i}] must be float, got {type(r).__name__}"
            )
            assert math.isfinite(r), f"returns[{i}] not finite: {r}"

        # Video path — must produce eval.mp4 OR eval.mp4.skipped.
        with tempfile.TemporaryDirectory() as td:
            agent.evaluate(env, n_episodes=1, record_video=True, video_dir=td)
            artifacts = sorted(p.name for p in Path(td).iterdir())
            has_video = "eval.mp4" in artifacts
            has_skipped = "eval.mp4.skipped" in artifacts
            assert has_video or has_skipped, (
                f"evaluate(record_video=True) must produce eval.mp4 or "
                f"eval.mp4.skipped under video_dir; got {artifacts}"
            )
            if has_video:
                size = (Path(td) / "eval.mp4").stat().st_size
                assert size > 0, "eval.mp4 was created but is empty"
    finally:
        env.close()


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Agent contract test for the refactored PPOAgent."
    )
    parser.add_argument(
        "--agent",
        choices=["ppo", "sb3"],
        required=True,
        help="Which agent to test. Only 'ppo' is available in this spec.",
    )
    return parser.parse_args()


def _main() -> int:
    args = _parse_args()

    if args.agent == "sb3":
        print(
            "FAIL: --agent sb3 is reserved for a follow-up spec and is not "
            "implemented yet. See specs/003-fix-mountaincar-train/research.md "
            "→ R6.",
            file=sys.stderr,
        )
        return 2

    results: dict[int, str] = {}
    for n in sorted(STEPS):
        results[n] = _run_step(n)

    total = len(results)
    passed = sum(1 for r in results.values() if r == "PASS")
    print(f"\n=== Summary: {passed} / {total} passed ===")
    for n in sorted(results):
        print(f"  TEST {n}: {results[n]}")

    return 0 if passed == total else 1


if __name__ == "__main__":
    sys.exit(_main())
