"""Per-TODO test runner for ppo_skeleton.py.

Run all five tests:

    uv run python workshop-1/1-ppo/test_ppo.py

Run only one TODO test:

    uv run python workshop-1/1-ppo/test_ppo.py --step 1

Exit codes:
    0  every test that ran reported PASS
    1  at least one test reported FAIL or NOT_IMPLEMENTED
    2  CLI argument error (e.g. --step 99)

Each test does a LOCAL import of only the symbol it needs, so
``--step 1`` works even when TODOs 2–5 are still raising
``NotImplementedError``. Do not add a top-level
``from ppo_skeleton import *`` here — it would break that isolation.

Spec: specs/001-ppo-skeleton/spec.md
Contract: specs/001-ppo-skeleton/contracts/test-runner-cli.md
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
# Step registry
# ---------------------------------------------------------------------------

STEPS: dict[int, tuple[str, Callable[[], None]]] = {}


def step(n: int, name: str):
    """Register a test function as the test for TODO ``n``."""

    def decorator(fn: Callable[[], None]) -> Callable[[], None]:
        if n in STEPS:
            raise RuntimeError(f"step {n} already registered as {STEPS[n][0]!r}")
        STEPS[n] = (name, fn)
        return fn

    return decorator


def _run_step(n: int) -> str:
    """Execute test step ``n`` and return PASS / NOT_IMPLEMENTED / FAIL."""
    name, fn = STEPS[n]
    try:
        fn()
    except NotImplementedError as e:
        print(f"TODO {n} not yet implemented ({name}): {e}")
        return "NOT_IMPLEMENTED"
    except AssertionError as e:
        print(f"FAIL: TODO {n} ({name}): {e}")
        return "FAIL"
    print(f"TODO {n} OK!")
    return "PASS"


# ===========================================================================
# TODO 1 — GAE
# ===========================================================================


@step(1, "GAE")
def test_step_1() -> None:
    """Validate compute_gae against a hand-computed reference."""
    import torch

    from ppo_skeleton import compute_gae

    # Case 1 — no done flags. Hand-computed closed-form reference.
    rewards = torch.tensor([1.0, 1.0, 1.0, 1.0], dtype=torch.float32)
    values = torch.tensor([0.5, 0.6, 0.7, 0.8, 0.9], dtype=torch.float32)
    dones = torch.tensor([0.0, 0.0, 0.0, 0.0], dtype=torch.float32)
    gamma, lam = 0.99, 0.95

    out = compute_gae(rewards, values, dones, gamma=gamma, lam=lam)

    # Shape and dtype.
    assert out.shape == (4,), f"expected shape (4,), got {tuple(out.shape)}"
    assert out.dtype == torch.float32, f"expected dtype float32, got {out.dtype}"

    # Hand-computed reference: iterate from t=3 down to t=0.
    expected = torch.zeros(4, dtype=torch.float32)
    gae = 0.0
    for t in reversed(range(4)):
        delta = rewards[t] + gamma * values[t + 1] * (1.0 - dones[t]) - values[t]
        gae = float(delta) + gamma * lam * (1.0 - float(dones[t])) * gae
        expected[t] = gae

    diff = (out - expected).abs().max().item()
    assert diff < 1e-5, (
        f"output[t] does not match closed-form reference within atol=1e-5 "
        f"(max abs diff {diff:.6e}; expected={expected.tolist()}, got={out.tolist()})"
    )

    # Case 2 — done in the middle. The reset MUST cut the bootstrap so that
    # advantages at t<=2 are not contaminated by values[3].
    rewards2 = torch.tensor([1.0, 1.0, 1.0, 1.0], dtype=torch.float32)
    values2 = torch.tensor([0.5, 0.6, 0.7, 0.8, 0.9], dtype=torch.float32)
    dones2 = torch.tensor([0.0, 0.0, 1.0, 0.0], dtype=torch.float32)

    out2 = compute_gae(rewards2, values2, dones2, gamma=gamma, lam=lam)

    expected2 = torch.zeros(4, dtype=torch.float32)
    gae = 0.0
    for t in reversed(range(4)):
        delta = rewards2[t] + gamma * values2[t + 1] * (1.0 - dones2[t]) - values2[t]
        gae = float(delta) + gamma * lam * (1.0 - float(dones2[t])) * gae
        expected2[t] = gae

    diff2 = (out2 - expected2).abs().max().item()
    assert diff2 < 1e-5, (
        f"done-in-the-middle case: output does not match reference within "
        f"atol=1e-5 (max abs diff {diff2:.6e}; expected={expected2.tolist()}, "
        f"got={out2.tolist()}). Did you forget the (1 - dones[t]) factor?"
    )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Per-TODO test runner for ppo_skeleton.py."
    )
    parser.add_argument(
        "--step",
        type=int,
        choices=sorted(STEPS.keys()) if STEPS else None,
        default=None,
        help="Run only the test for TODO N. Omit to run all five.",
    )
    return parser.parse_args()


def _main() -> int:
    args = _parse_args()
    if args.step is not None:
        if args.step not in STEPS:
            print(
                f"FAIL: step {args.step} is not registered. "
                f"Valid steps: {sorted(STEPS)}.",
                file=sys.stderr,
            )
            return 2
        result = _run_step(args.step)
        return 0 if result == "PASS" else 1

    # Run all steps in numeric order, then print a summary.
    results: dict[int, str] = {}
    for n in sorted(STEPS):
        results[n] = _run_step(n)

    total = len(results)
    passed = sum(1 for r in results.values() if r == "PASS")
    print(f"\n=== Summary: {passed} / {total} passed ===")
    for n in sorted(results):
        print(f"  TODO {n}: {results[n]}")

    return 0 if passed == total else 1


if __name__ == "__main__":
    sys.exit(_main())
