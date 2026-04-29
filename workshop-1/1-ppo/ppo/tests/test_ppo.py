"""Per-TODO test runner for the refactored ``ppo`` package.

Usage::

    uv run python workshop-1/1-ppo/ppo/tests/test_ppo.py            # all 5 steps
    uv run python workshop-1/1-ppo/ppo/tests/test_ppo.py --step 1   # just GAE

Exit codes:
    0  every step that ran reported PASS
    1  at least one step reported FAIL or NOT_IMPLEMENTED
    2  CLI argument error (e.g. --step 99)

Each test does a LOCAL import of only the symbols it needs, so
``--step 1`` works even when other TODOs are still raising
``NotImplementedError``.

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
    """Validate ``RolloutBuffer.compute_gae`` against a hand-computed reference."""
    import torch

    from ppo import RolloutBuffer

    buf = RolloutBuffer(size=4, obs_dim=2, action_dim=1)

    # Case 1 — no done flags. Hand-computed closed-form reference.
    rewards = torch.tensor([1.0, 1.0, 1.0, 1.0], dtype=torch.float32)
    values = torch.tensor([0.5, 0.6, 0.7, 0.8, 0.9], dtype=torch.float32)
    dones = torch.tensor([0.0, 0.0, 0.0, 0.0], dtype=torch.float32)
    gamma, lam = 0.99, 0.95

    out = buf.compute_gae(rewards, values, dones, gamma=gamma, lam=lam)

    assert out.shape == (4,), f"expected shape (4,), got {tuple(out.shape)}"
    assert out.dtype == torch.float32, f"expected dtype float32, got {out.dtype}"

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

    out2 = buf.compute_gae(rewards2, values2, dones2, gamma=gamma, lam=lam)

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
    """Validate ``PPOAgent.sample_action`` shape, dtype, range, stochasticity."""
    import gymnasium as gym
    import torch

    from ppo import PPOAgent

    env = gym.make("MountainCarContinuous-v0")
    try:
        agent = PPOAgent(env, hyperparameters={"random_state": 0})
        obs_dim, action_dim = agent.obs_dim, agent.action_dim
        device = agent.device
        obs = torch.tensor([0.1, -0.2], dtype=torch.float32, device=device)

        action, log_prob = agent.sample_action(obs, deterministic=False)
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
        amin = float(agent.action_min.min().item())
        amax = float(agent.action_max.max().item())
        assert amin <= float(action.item()) <= amax, (
            f"action {action.item()} not in [{amin}, {amax}]"
        )

        # Batched shape.
        obs_batch = torch.tensor(
            [[0.1, -0.2], [-0.3, 0.4], [0.0, 0.0]],
            dtype=torch.float32,
            device=device,
        )
        a_batch, lp_batch = agent.sample_action(obs_batch, deterministic=False)
        assert a_batch.shape == (3, action_dim), (
            f"batched action shape: expected (3, {action_dim}), got {tuple(a_batch.shape)}"
        )
        assert lp_batch.shape == (3,), (
            f"batched log_prob shape: expected (3,), got {tuple(lp_batch.shape)}"
        )

        # Stochastic sampling has nonzero variance over many calls.
        samples = torch.stack(
            [agent.sample_action(obs, deterministic=False)[0] for _ in range(1000)]
        )
        var = float(samples.var().item())
        assert var > 1e-6, (
            f"stochastic sampling has near-zero variance ({var:.2e}). "
            f"Did you ignore the deterministic=False branch?"
        )

        # Range across all samples.
        smin = float(samples.min().item())
        smax = float(samples.max().item())
        assert smin >= amin and smax <= amax, (
            f"some sampled actions outside [{amin}, {amax}] "
            f"(min={smin:.3f}, max={smax:.3f}). Did you forget to clamp?"
        )

        # Deterministic mode → effectively zero variance.
        det_samples = torch.stack(
            [agent.sample_action(obs, deterministic=True)[0] for _ in range(100)]
        )
        det_var = float(det_samples.var().item())
        assert det_var < 1e-10, (
            f"deterministic=True must give effectively zero variance "
            f"(< 1e-10), got {det_var:.2e}. Did you sample from the "
            f"distribution instead of returning the mean?"
        )
    finally:
        env.close()


# ===========================================================================
# TODO 3 — evaluate actions
# ===========================================================================


@step(3, "evaluate actions")
def test_step_3() -> None:
    """Validate ``PPOAgent.evaluate_actions`` against an independent Normal reference."""
    import gymnasium as gym
    import torch
    from torch.distributions import Normal

    from ppo import PPOAgent

    env = gym.make("MountainCarContinuous-v0")
    try:
        agent = PPOAgent(env, hyperparameters={"random_state": 0})
        device = agent.device
        B, action_dim = 32, agent.action_dim
        obs = torch.randn(B, agent.obs_dim, device=device)
        actions = torch.randn(B, action_dim, device=device).clamp(
            agent.action_min, agent.action_max
        )

        log_probs, entropy = agent.evaluate_actions(obs, actions)

        assert log_probs.shape == (B,), (
            f"log_probs shape: expected ({B},), got {tuple(log_probs.shape)}"
        )
        assert entropy.shape == (B,), (
            f"entropy shape: expected ({B},), got {tuple(entropy.shape)}"
        )

        with torch.no_grad():
            mean = agent.actor(obs)
        ref_dist = Normal(mean, agent.log_std.exp())
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
    finally:
        env.close()


# ===========================================================================
# TODO 4 — PPO loss
# ===========================================================================


@step(4, "PPO loss")
def test_step_4() -> None:
    """Validate ``PPOAgent.ppo_loss``: ratio=1 reduction, clipped-branch grads."""
    import gymnasium as gym
    import torch

    from ppo import PPOAgent

    env = gym.make("MountainCarContinuous-v0")
    try:
        agent = PPOAgent(env, hyperparameters={"random_state": 0})
        B = 16

        # 1) Ratio = 1.0: loss must equal -advantages.mean() exactly.
        new_lp = torch.zeros(B, requires_grad=True)
        old_lp = torch.zeros(B)
        advantages = torch.tensor(
            [0.5, -0.5, 1.0, -1.0, 0.2, -0.2, 0.7, -0.7,
             0.3, -0.3, 0.9, -0.9, 0.1, -0.1, 0.6, -0.6]
        )
        loss = agent.ppo_loss(new_lp, old_lp, advantages, clip_eps=0.2)

        assert loss.shape == torch.Size([]), (
            f"loss must be scalar, got shape {tuple(loss.shape)}"
        )
        assert loss.requires_grad, "loss must carry gradient (requires_grad=True)"

        expected = -advantages.mean()
        diff = (loss - expected).abs().item()
        assert diff < 1e-6, (
            f"unclipped branch (ratio=1.0): expected loss == -advantages.mean() "
            f"= {float(expected):.6f}, got {float(loss):.6f} (diff {diff:.2e})"
        )

        # 2) Clipped branch (positive advantage): ratio = exp(1) ~ 2.718 > 1+eps.
        new_lp_pos = torch.full((B,), 1.0, requires_grad=True)
        old_lp_zero = torch.zeros(B)
        pos_adv = torch.full((B,), 0.5)
        loss_pos = agent.ppo_loss(new_lp_pos, old_lp_zero, pos_adv, clip_eps=0.2)
        loss_pos.backward()
        grad_pos_max = new_lp_pos.grad.abs().max().item()
        assert grad_pos_max < 1e-6, (
            f"clipped branch (ratio>1+eps, adv>0): expected zero gradient, "
            f"got max abs grad {grad_pos_max:.6e}. "
            f"Did you forget to clamp the ratio with .clamp(1-eps, 1+eps)?"
        )

        # 3) Clipped branch (negative advantage): ratio = exp(-1) ~ 0.368 < 1-eps.
        new_lp_neg = torch.full((B,), -1.0, requires_grad=True)
        neg_adv = torch.full((B,), -0.5)
        loss_neg = agent.ppo_loss(new_lp_neg, old_lp_zero, neg_adv, clip_eps=0.2)
        loss_neg.backward()
        grad_neg_max = new_lp_neg.grad.abs().max().item()
        assert grad_neg_max < 1e-6, (
            f"clipped branch (ratio<1-eps, adv<0): expected zero gradient, "
            f"got max abs grad {grad_neg_max:.6e}."
        )
    finally:
        env.close()


# ===========================================================================
# TODO 5 — training loop smoke test
# ===========================================================================


@step(5, "training loop")
def test_step_5() -> None:
    """Smoke test: a short MountainCarContinuous-v0 training run.

    Checks that ``train()`` runs end-to-end on a tiny budget, returns the
    expected dict, produces no NaN losses, and finishes under 10 seconds.
    Does NOT assert learning — that is verified by the stage-2 driver.
    """
    import math
    import time

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

        captured: list[float] = []

        def capture(line: str) -> None:
            for token in line.split():
                if token.startswith("policy_loss="):
                    try:
                        captured.append(float(token.split("=", 1)[1]))
                    except ValueError:
                        pass

        t0 = time.perf_counter()
        stats = agent.train(env, total_timesteps=512, random_state=0, log_fn=capture)
        elapsed = time.perf_counter() - t0

        assert elapsed < 10.0, (
            f"smoke test exceeded 10s budget (took {elapsed:.2f}s). "
            f"Use total_timesteps=512 in tests; production hyperparameters "
            f"should not leak into the test fixture."
        )

        assert isinstance(stats, dict), (
            f"train() must return a dict, got {type(stats).__name__}"
        )
        required = {"mean_reward", "policy_loss", "value_loss", "entropy", "n_updates"}
        missing = required - set(stats.keys())
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
    finally:
        env.close()


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Per-TODO test runner for the refactored ppo package."
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
