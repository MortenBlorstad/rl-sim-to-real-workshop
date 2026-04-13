# Brainstorm: PPO Skeleton with Per-TODO Tests

**Date:** 2026-04-13
**Related issues:** none

## Chosen Direction

Build a single-file `workshop-1/1-ppo/ppo_skeleton.py` that exposes five top-level symbols corresponding to the five TODOs — `compute_gae`, `sample_action`, `evaluate_actions`, `ppo_loss`, and `train` — plus a fully provided `PolicyNetwork`, `RolloutBuffer`, and a `PPOAgent` class that implements the Constitution Article II `Agent` interface by delegating to those functions. The actual training entry point lives behind `if __name__ == "__main__":` so that importing the module never executes the training loop and never trips on another TODO's `NotImplementedError`.

Tests live in a sibling `test_ppo.py` that uses a small custom runner (~50 lines) built around a `@step(n, name)` decorator. Each test imports only the symbol under test via a local `from ppo_skeleton import ...`, so an unfinished TODO elsewhere cannot break an earlier `--step N` invocation. The runner catches `NotImplementedError` explicitly and prints beginner-facing output matching Article IV (`"TODO 1 OK!"` / `"FAIL: expected shape (5,), got (5,1)"`). Every test must run in under 10 seconds; TODO 5 is a smoke test by default, with an opt-in `--step 5 --full` for a weak CartPole learning assertion.

Reference solutions live on the `solutions` branch, and per-TODO checkpoint tags (`ws1-todo1-done`, …, `ws1-todo5-done`) let stuck participants jump directly to the state where TODO N is complete via `git checkout ws1-todoN-done -- workshop-1/1-ppo/ppo_skeleton.py`.

## Key Decisions

- **Single file, top-level symbols.** Preserves the beginner mental model while giving tests clean import targets. Matches Article V.
- **Training loop guarded by `__main__`.** Importing `ppo_skeleton` for any `--step N` has zero side effects and is immune to unfinished downstream TODOs.
- **Local imports inside each test.** Isolates `--step N` from `NotImplementedError`s in other TODOs — the defining requirement of Article IV independent testing.
- **Custom 50-line runner with `@step(n, name)`.** Article IV's prescribed output format and CLI do not match pytest conventions; a tiny purpose-built runner is cheaper than adapting pytest.
- **Per-TODO test strategy:**
  - *TODO 1 (GAE)*: hand-computed toy trajectory, `atol=1e-5`, plus done-in-the-middle edge case.
  - *TODO 2 (Sample action)*: seeded tiny policy, 1000 samples, check shape/dtype/variance/`deterministic=True` constancy.
  - *TODO 3 (Evaluate actions)*: element-wise comparison against `torch.distributions` reference for `log_prob` and `entropy`.
  - *TODO 4 (PPO loss)*: synthetic advantages and ratios exercising clipped vs. unclipped branches; check scalar tensor with `requires_grad=True`.
  - *TODO 5 (Training loop)*: CartPole-v1, 256 steps, smoke test that `train()` returns the expected dict and produces finite losses. Opt-in `--full` runs 2k steps and asserts mean reward > 30.
- **`PPOAgent` lives in the same file.** Wrapping the five TODO functions in the Article II `Agent` interface inside `ppo_skeleton.py` means the Workshop 2 handoff and `test_agent_interface.py --agent ppo` work without a second module. `PPOAgent.preprocess` is a no-op for vector envs and is overridden for pixel envs in Workshop 2.
- **`solutions` branch + per-TODO git tags.** Article V and Article IX catch-up mechanisms are used directly, no inline solution comments.

## Constraints & Trade-offs

- **10-second test cap (Article IV)** forces TODO 5 to be a smoke test by default. Acknowledged; the `--full` flag provides a weak learning check for participants who want more confidence, but it is not part of the default workshop flow.
- **Local imports inside test functions** look unusual to experienced Python testers. The trade-off is worth it: it's the only clean way to make `--step 1` genuinely independent of whether TODO 2–5 are implemented.
- **Toy-trajectory GAE test uses `atol=1e-5`**, not exact equality, per Article IV.
- **Weak learning assertion is inherently flaky** if pushed tight. Mean-reward > 30 on CartPole after 2k steps is loose enough to be reliable while still proving learning happened. Tighter assertions are rejected.
- **Single file grows large** (~300–400 lines with docstrings and `PPOAgent`). Acceptable: participants scroll, but they never lose context.

## Discarded Alternatives

- **Multi-file split (`gae.py`, `policy.py`, `loss.py`, `train.py`).** Cleaner unit boundaries, but forces 3-hour-beginner participants to juggle 4–5 files. See `decisions.md`.
- **pytest-based runner.** Standard tooling but fights the Article IV output format and CLI contract. See `decisions.md`.
- **Plain asserts without a runner.** Cannot distinguish "not implemented" from "implemented incorrectly", and stops at the first failure. See `decisions.md`.
- **Inline `# SOLUTION:` comments.** Spoils the exercise on first file open. See `decisions.md`.
- **Full-training learning assertion as the default `--step 5`.** Exceeds the 10-second test cap. See `decisions.md`.
