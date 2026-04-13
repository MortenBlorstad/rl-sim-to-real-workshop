"""Article II Agent contract test for PPOAgent.

Run:

    uv run python workshop-1/1-ppo/test_agent_interface.py --agent ppo

(``--agent sb3`` is reserved for the Path B follow-up spec and exits 2.)

Verifies that ``PPOAgent`` honors every clause of the Constitution
Article II ``Agent`` contract as pinned in
``specs/001-ppo-skeleton/contracts/agent-interface.md``:

  C1. registry membership
  C2. preprocess identity, determinism, and subclass override path
  C3. predict accepts raw observations and produces a valid action
  C4. train method exists and returns the expected stats dict
  C5. save/load round-trip including subclass class restoration
  C6. _get_preprocess_state / _set_preprocess_state base no-op

Tests that need ``sample_action`` (TODO 2) or ``train`` (TODO 5) will
report NOT_IMPLEMENTED on an unmodified skeleton; the structural
checks (C1, C2 identity, C6) MUST pass even on a fresh skeleton.

Each test does a LOCAL import of only the symbols it needs so a
single failing TODO does not cascade through unrelated structural
checks.
"""
from __future__ import annotations

import argparse
import os
import sys
from typing import Callable

# Make ``ppo_skeleton`` importable regardless of current working directory.
_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)


# ---------------------------------------------------------------------------
# Step registry — kept LOCAL to this file so we don't share mutable state
# with test_ppo.STEPS.
# ---------------------------------------------------------------------------

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
    from ppo_skeleton import _AGENT_REGISTRY

    assert "PPOAgent" in _AGENT_REGISTRY, (
        f"PPOAgent must be registered via @register_agent. "
        f"Currently registered: {sorted(_AGENT_REGISTRY)}."
    )


# ===========================================================================
# C2 — preprocess identity + determinism + override path
# ===========================================================================


@step(2, "C2 preprocess identity (base class)")
def test_preprocess_identity() -> None:
    import numpy as np

    from ppo_skeleton import PPOAgent

    agent = PPOAgent(obs_dim=2, action_dim=1)
    x = np.array([0.1, -0.2], dtype=np.float32)
    out = agent.preprocess(x)
    assert isinstance(out, np.ndarray), (
        f"preprocess() must return a numpy array, got {type(out).__name__}"
    )
    assert np.array_equal(out, x), (
        f"base PPOAgent.preprocess must be identity; got {out!r} for input {x!r}"
    )


@step(3, "C2 preprocess deterministic")
def test_preprocess_deterministic() -> None:
    import numpy as np

    from ppo_skeleton import PPOAgent

    agent = PPOAgent(obs_dim=2, action_dim=1)
    x = np.random.RandomState(0).randn(2).astype(np.float32)
    a = agent.preprocess(x)
    b = agent.preprocess(x)
    assert np.array_equal(a, b), (
        f"preprocess must be a pure function — same input must give same output. "
        f"First call: {a!r}, second call: {b!r}"
    )


@step(4, "C2 subclass override is used by predict")
def test_subclass_override_used_by_predict() -> None:
    import numpy as np

    from ppo_skeleton import PPOAgent, register_agent

    @register_agent
    class _ScalingAgent(PPOAgent):
        def preprocess(self, obs):
            return (obs * 0.5).astype(np.float32)

    agent = _ScalingAgent(obs_dim=2, action_dim=1)
    raw = np.array([0.4, -0.6], dtype=np.float32)

    # The override itself must be reachable.
    transformed = agent.preprocess(raw)
    assert np.array_equal(transformed, np.array([0.2, -0.3], dtype=np.float32)), (
        f"_ScalingAgent.preprocess override is not being called; got {transformed!r}"
    )

    # And predict must call self.preprocess (not the bare base method).
    # We sanity-check this by asserting that predict accepts the raw obs and
    # returns a valid action — sample_action is invoked downstream of
    # self.preprocess, so a working predict here proves the override path.
    action = agent.predict(raw)
    assert action.shape == (1,), (
        f"predict via override path: expected action shape (1,), got {action.shape}"
    )


# ===========================================================================
# C3 — predict accepts raw obs
# ===========================================================================


@step(5, "C3 predict raw obs shape/dtype/range")
def test_predict_raw_obs_shape_dtype_range() -> None:
    import numpy as np

    from ppo_skeleton import PPOAgent

    agent = PPOAgent(obs_dim=2, action_dim=1)
    raw = np.array([0.1, -0.2], dtype=np.float32)
    action = agent.predict(raw)

    assert isinstance(action, np.ndarray), (
        f"predict must return a numpy array, got {type(action).__name__}"
    )
    assert action.shape == (1,), (
        f"action shape: expected (1,), got {action.shape}"
    )
    assert action.dtype == np.float32, (
        f"action dtype: expected float32, got {action.dtype}"
    )
    val = float(action[0])
    assert -1.0 <= val <= 1.0, (
        f"action {val} not in [-1, 1]. Did you forget to clamp in sample_action?"
    )


@step(6, "C3 predict deterministic flag")
def test_predict_deterministic_flag() -> None:
    import numpy as np

    from ppo_skeleton import PPOAgent

    agent = PPOAgent(obs_dim=2, action_dim=1)
    raw = np.array([0.1, -0.2], dtype=np.float32)
    a1 = agent.predict(raw, deterministic=True)
    a2 = agent.predict(raw, deterministic=True)
    assert np.array_equal(a1, a2), (
        f"deterministic=True must produce the same action across calls; "
        f"got {a1!r} then {a2!r}"
    )


# ===========================================================================
# C4 — train smoke
# ===========================================================================


@step(7, "C4 train method smoke")
def test_train_method_smoke() -> None:
    import gymnasium as gym

    from ppo_skeleton import PPOAgent

    env = gym.make("MountainCarContinuous-v0")
    env.reset(seed=0)
    agent = PPOAgent(obs_dim=2, action_dim=1)

    # Override the rollout/epoch params so the test runs fast.
    agent.hyperparameters["rollout_size"] = 256
    agent.hyperparameters["n_epochs"] = 2
    agent.hyperparameters["batch_size"] = 64

    stats = agent.train(env, total_timesteps=512)
    env.close()

    assert isinstance(stats, dict), (
        f"PPOAgent.train must return a dict, got {type(stats).__name__}"
    )
    required = {"mean_reward", "policy_loss", "value_loss", "entropy", "n_updates"}
    missing = required - set(stats.keys())
    assert not missing, (
        f"stats dict missing required keys: {sorted(missing)}. "
        f"Got keys: {sorted(stats.keys())}."
    )


# ===========================================================================
# C5 — save / load round-trip
# ===========================================================================


@step(8, "C5 save/load round-trip (base class)")
def test_save_load_roundtrip_base() -> None:
    import os
    import tempfile

    import numpy as np

    from ppo_skeleton import PPOAgent

    agent = PPOAgent(obs_dim=2, action_dim=1)
    raw = np.array([0.3, -0.4], dtype=np.float32)
    expected = agent.predict(raw, deterministic=True)

    fd, path = tempfile.mkstemp(suffix=".pt")
    os.close(fd)
    try:
        agent.save(path)
        loaded = PPOAgent.load(path)
        assert type(loaded).__name__ == "PPOAgent", (
            f"loaded agent class: expected PPOAgent, got {type(loaded).__name__}"
        )
        actual = loaded.predict(raw, deterministic=True)
        assert np.array_equal(actual, expected), (
            f"loaded agent produces a different action than the original. "
            f"original={expected!r}, loaded={actual!r}. "
            f"Save/load is dropping state somewhere — check that actor + value "
            f"+ log_std are all serialized."
        )
    finally:
        if os.path.exists(path):
            os.remove(path)


@step(9, "C5 subclass class restored on load")
def test_save_load_roundtrip_subclass_class_restored() -> None:
    import os
    import tempfile

    import numpy as np

    from ppo_skeleton import PPOAgent, register_agent

    @register_agent
    class _RoundTripAgent(PPOAgent):
        def preprocess(self, obs):
            return obs.clip(-0.5, 0.5).astype(np.float32)

    agent = _RoundTripAgent(obs_dim=2, action_dim=1)

    fd, path = tempfile.mkstemp(suffix=".pt")
    os.close(fd)
    try:
        agent.save(path)
        loaded = PPOAgent.load(path)
        assert type(loaded).__name__ == "_RoundTripAgent", (
            f"loaded class: expected _RoundTripAgent, got {type(loaded).__name__}. "
            f"PPOAgent.load must look up the class via _AGENT_REGISTRY by name."
        )

        # The override must still be reachable on the loaded instance.
        test_input = np.array([2.0, -2.0], dtype=np.float32)
        out = loaded.preprocess(test_input)
        expected = np.array([0.5, -0.5], dtype=np.float32)
        assert np.array_equal(out, expected), (
            f"_RoundTripAgent.preprocess override is not active on the loaded "
            f"instance; expected {expected!r}, got {out!r}"
        )
    finally:
        if os.path.exists(path):
            os.remove(path)


# ===========================================================================
# C6 — _get/_set_preprocess_state base no-op
# ===========================================================================


@step(10, "C6 base preprocess_state is no-op")
def test_get_set_preprocess_state_base_no_op() -> None:
    from ppo_skeleton import PPOAgent

    agent = PPOAgent(obs_dim=2, action_dim=1)
    assert agent._get_preprocess_state() == {}, (
        f"base _get_preprocess_state() must return {{}}, "
        f"got {agent._get_preprocess_state()!r}"
    )
    agent._set_preprocess_state({})  # must not raise
    agent._set_preprocess_state({"unused": 42})  # base must ignore extra keys


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Article II Agent contract test for PPOAgent."
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
            "FAIL: --agent sb3 is reserved for the Path B follow-up spec "
            "and is not implemented yet. See "
            "specs/001-ppo-skeleton/spec.md → 'Out of scope'.",
            file=sys.stderr,
        )
        return 2

    # args.agent == "ppo"
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
