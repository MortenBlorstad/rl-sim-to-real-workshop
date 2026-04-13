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


# ===========================================================================
# TODO 2 — sample action
# ===========================================================================


@step(2, "sample action")
def test_step_2() -> None:
    """Validate sample_action shape, dtype, range, and stochastic behavior."""
    import torch

    from ppo_skeleton import ActorNetwork, sample_action

    torch.manual_seed(0)
    obs_dim, action_dim = 2, 1
    actor = ActorNetwork(obs_dim, action_dim)
    log_std = torch.zeros(action_dim)
    obs = torch.tensor([0.1, -0.2], dtype=torch.float32)

    # Single-obs shape and dtype.
    action, log_prob = sample_action(actor, obs, log_std, deterministic=False)
    assert action.shape == (action_dim,), (
        f"single-obs action shape: expected ({action_dim},), got {tuple(action.shape)}"
    )
    assert action.dtype == torch.float32, (
        f"action dtype: expected float32, got {action.dtype}"
    )
    assert log_prob.shape == (), (
        f"single-obs log_prob shape: expected scalar (), got {tuple(log_prob.shape)}"
    )
    assert torch.isfinite(log_prob), f"log_prob is not finite: {log_prob.item()}"
    # Range check on a single sample (cheap; the 1000-sample loop below covers
    # this more robustly).
    assert -1.0 <= float(action.item()) <= 1.0, (
        f"action {action.item()} not in [-1, 1]"
    )

    # Batched shape.
    obs_batch = torch.tensor(
        [[0.1, -0.2], [-0.3, 0.4], [0.0, 0.0]], dtype=torch.float32
    )
    a_batch, lp_batch = sample_action(actor, obs_batch, log_std, deterministic=False)
    assert a_batch.shape == (3, action_dim), (
        f"batched action shape: expected (3, {action_dim}), got {tuple(a_batch.shape)}"
    )
    assert lp_batch.shape == (3,), (
        f"batched log_prob shape: expected (3,), got {tuple(lp_batch.shape)}"
    )

    # Stochastic sampling has nonzero variance over many calls.
    samples = torch.stack(
        [sample_action(actor, obs, log_std, deterministic=False)[0] for _ in range(1000)]
    )
    var = float(samples.var().item())
    assert var > 1e-6, (
        f"stochastic sampling has near-zero variance ({var:.2e}). "
        f"Did you ignore the deterministic=False branch?"
    )

    # Range check across all samples.
    assert float(samples.min().item()) >= -1.0 and float(samples.max().item()) <= 1.0, (
        f"some sampled actions are outside [-1, 1] "
        f"(min={float(samples.min().item()):.3f}, max={float(samples.max().item()):.3f}). "
        f"Did you forget to clamp?"
    )

    # Deterministic mode produces (essentially) identical outputs across
    # calls. A tiny epsilon is allowed because PyTorch ops are not always
    # bit-for-bit reproducible across calls, but the variance MUST be many
    # orders of magnitude below the stochastic case (which is around 1.0
    # for log_std=0).
    det_samples = torch.stack(
        [sample_action(actor, obs, log_std, deterministic=True)[0] for _ in range(100)]
    )
    det_var = float(det_samples.var().item())
    assert det_var < 1e-10, (
        f"deterministic=True must give effectively zero variance across "
        f"calls (< 1e-10), got {det_var:.2e}. Did you sample from the "
        f"distribution instead of returning the mean when deterministic=True?"
    )


# ===========================================================================
# TODO 3 — evaluate actions
# ===========================================================================


@step(3, "evaluate actions")
def test_step_3() -> None:
    """Validate evaluate_actions against an independent Normal reference."""
    import torch
    from torch.distributions import Normal

    from ppo_skeleton import ActorNetwork, evaluate_actions

    torch.manual_seed(0)
    obs_dim, action_dim = 2, 1
    actor = ActorNetwork(obs_dim, action_dim)
    log_std = torch.zeros(action_dim)

    B = 32
    obs = torch.randn(B, obs_dim)
    actions = torch.randn(B, action_dim).clamp(-1.0, 1.0)

    log_probs, entropy = evaluate_actions(actor, obs, actions, log_std)

    assert log_probs.shape == (B,), (
        f"log_probs shape: expected ({B},), got {tuple(log_probs.shape)}"
    )
    assert entropy.shape == (B,), (
        f"entropy shape: expected ({B},), got {tuple(entropy.shape)}"
    )

    # Independent reference using torch.distributions.Normal directly.
    with torch.no_grad():
        mean = actor(obs)
    ref_dist = Normal(mean, log_std.exp())
    ref_log_probs = ref_dist.log_prob(actions).sum(dim=-1)
    ref_entropy = ref_dist.entropy().sum(dim=-1)

    diff_lp = (log_probs - ref_log_probs).abs().max().item()
    assert diff_lp < 1e-5, (
        f"log_probs do not match Normal(mean, log_std.exp()).log_prob(...).sum(-1) "
        f"within atol=1e-5 (max abs diff {diff_lp:.6e}). "
        f"Did you forget the .sum(dim=-1) over action dims?"
    )

    diff_ent = (entropy - ref_entropy).abs().max().item()
    assert diff_ent < 1e-5, (
        f"entropy does not match Normal(...).entropy().sum(-1) within "
        f"atol=1e-5 (max abs diff {diff_ent:.6e})."
    )

    assert torch.isfinite(log_probs).all(), "log_probs contains non-finite values"
    assert torch.isfinite(entropy).all(), "entropy contains non-finite values"


# ===========================================================================
# TODO 4 — PPO loss
# ===========================================================================


@step(4, "PPO loss")
def test_step_4() -> None:
    """Validate ppo_loss unclipped branch, clipped branch, and scalar/grad."""
    import torch

    from ppo_skeleton import ppo_loss

    B = 16

    # 1) Unclipped branch at ratio == 1.0: new_log_probs == old_log_probs.
    #    The result must equal -advantages.mean() exactly (within tolerance).
    new_lp = torch.zeros(B, requires_grad=True)
    old_lp = torch.zeros(B)
    advantages = torch.tensor(
        [0.5, -0.5, 1.0, -1.0, 0.2, -0.2, 0.7, -0.7, 0.3, -0.3, 0.9, -0.9, 0.1, -0.1, 0.6, -0.6]
    )
    loss = ppo_loss(new_lp, old_lp, advantages, clip_eps=0.2)

    assert loss.shape == torch.Size([]), (
        f"loss must be a scalar tensor, got shape {tuple(loss.shape)}"
    )
    assert loss.requires_grad, "loss must carry gradient (requires_grad=True)"

    expected = -advantages.mean()
    diff = (loss - expected).abs().item()
    assert diff < 1e-6, (
        f"unclipped branch (ratio=1.0): expected loss == -advantages.mean() "
        f"= {float(expected):.6f}, got {float(loss):.6f} (diff {diff:.2e})"
    )

    # 2) Clipped branch (positive advantage): force ratio = exp(1.0) ~= 2.718,
    #    well above 1 + clip_eps = 1.2. With advantage > 0 this hits the
    #    clipped branch, and the gradient w.r.t. new_log_probs must be 0.
    new_lp_pos = torch.full((B,), 1.0, requires_grad=True)
    old_lp_zero = torch.zeros(B)
    pos_adv = torch.full((B,), 0.5)
    loss_pos = ppo_loss(new_lp_pos, old_lp_zero, pos_adv, clip_eps=0.2)
    loss_pos.backward()
    grad_pos_max = new_lp_pos.grad.abs().max().item()
    assert grad_pos_max < 1e-6, (
        f"clipped branch (ratio>1+eps, adv>0): expected zero gradient w.r.t. "
        f"new_log_probs, got max abs grad {grad_pos_max:.6e}. "
        f"Did you forget to clamp the ratio with .clamp(1-eps, 1+eps)?"
    )

    # 3) Clipped branch (negative advantage): force ratio = exp(-1.0) ~= 0.368,
    #    well below 1 - clip_eps = 0.8. With advantage < 0 this hits the
    #    clipped branch, and the gradient must again be 0.
    new_lp_neg = torch.full((B,), -1.0, requires_grad=True)
    neg_adv = torch.full((B,), -0.5)
    loss_neg = ppo_loss(new_lp_neg, old_lp_zero, neg_adv, clip_eps=0.2)
    loss_neg.backward()
    grad_neg_max = new_lp_neg.grad.abs().max().item()
    assert grad_neg_max < 1e-6, (
        f"clipped branch (ratio<1-eps, adv<0): expected zero gradient w.r.t. "
        f"new_log_probs, got max abs grad {grad_neg_max:.6e}."
    )


# ===========================================================================
# TODO 5 — training loop smoke test
# ===========================================================================


@step(5, "training loop")
def test_step_5() -> None:
    """Smoke test: a short MountainCarContinuous-v0 training run.

    Checks that train() runs end-to-end on a tiny budget, returns the
    expected dict, produces no NaN losses, and finishes under 10 seconds.
    Does NOT assert learning — convergence on MountainCarContinuous is
    verified in stage 2-mountaincar/, not here.
    """
    import math
    import time

    import gymnasium as gym
    import torch
    import torch.nn as nn

    from ppo_skeleton import ActorNetwork, ValueNetwork, train

    env = gym.make("MountainCarContinuous-v0")
    env.reset(seed=0)

    obs_dim = int(env.observation_space.shape[0])
    action_dim = int(env.action_space.shape[0])

    torch.manual_seed(0)
    actor = ActorNetwork(obs_dim, action_dim)
    value = ValueNetwork(obs_dim)
    log_std = nn.Parameter(torch.zeros(action_dim))

    captured: list[float] = []

    def capture(line):
        # Tolerant capture: line may be a string or already-formatted text.
        if isinstance(line, str):
            for token in line.split():
                if token.startswith("policy_loss="):
                    try:
                        captured.append(float(token.split("=", 1)[1]))
                    except ValueError:
                        pass

    t0 = time.perf_counter()
    stats = train(
        env,
        actor,
        value,
        log_std,
        total_timesteps=512,
        rollout_size=256,
        n_epochs=2,
        batch_size=64,
        seed=0,
        log_fn=capture,
    )
    elapsed = time.perf_counter() - t0

    assert elapsed < 10.0, (
        f"smoke test exceeded 10s budget (took {elapsed:.2f}s). "
        f"Use total_timesteps=512 in tests; production hyperparameters "
        f"should not leak into the test fixture."
    )

    assert isinstance(stats, dict), f"train() must return a dict, got {type(stats).__name__}"
    required_keys = {"mean_reward", "policy_loss", "value_loss", "entropy", "n_updates"}
    missing = required_keys - set(stats.keys())
    assert not missing, (
        f"stats dict missing required keys: {sorted(missing)}. "
        f"Got keys: {sorted(stats.keys())}."
    )

    for key in ("policy_loss", "value_loss", "entropy", "mean_reward"):
        v = stats[key]
        assert isinstance(v, (int, float)), (
            f"stats[{key!r}] must be a number, got {type(v).__name__}"
        )
        assert not math.isnan(float(v)), f"stats[{key!r}] is NaN"
    assert isinstance(stats["n_updates"], int) and stats["n_updates"] > 0, (
        f"stats['n_updates'] must be a positive int, got {stats['n_updates']!r}"
    )

    assert all(not math.isnan(loss) for loss in captured), (
        f"captured a NaN policy_loss in the printed log lines: {captured}"
    )

    env.close()


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
