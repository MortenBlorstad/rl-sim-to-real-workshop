# Contract: Test runners

**Date**: 2026-04-29
**Branch**: `003-fix-mountaincar-train`

This contract defines the public surface and behavior of the two test files under `workshop-1/1-ppo/ppo/tests/`. Both follow the pre-refactor convention from commit `60321eb` — custom `@step(n, name)` registry, no pytest, no auto-discovery.

## File 1: `test_ppo.py`

### CLI

```bash
uv run python workshop-1/1-ppo/ppo/tests/test_ppo.py            # run all 5 steps
uv run python workshop-1/1-ppo/ppo/tests/test_ppo.py --step 1   # run only step 1
```

### Steps

| Step | Name | Targets | Imports |
|---|---|---|---|
| 1 | GAE | `RolloutBuffer.compute_gae` (TODO 1) | `from ppo import RolloutBuffer; import torch` |
| 2 | sample action | `PPOAgent.sample_action` (TODO 2) | `from ppo import PPOAgent` + a tiny env (MountainCar) |
| 3 | evaluate actions | `PPOAgent.evaluate_actions` (TODO 3) | `from ppo import PPOAgent`; reference: `torch.distributions.Normal` |
| 4 | PPO loss | `PPOAgent.ppo_loss` (TODO 4) | `from ppo import PPOAgent` (instance is constructed against a tiny env) |
| 5 | training loop smoke | `PPOAgent.train` (TODO 5) | `import gymnasium as gym; from ppo import PPOAgent` |

### Per-step behavior

- **Result classification**: `NotImplementedError` → `"NOT_IMPLEMENTED"`, `AssertionError` → `"FAIL"`, otherwise → `"PASS"`.
- Each step does **local** imports inside the function so unfilled TODOs in unrelated steps do not cascade.
- Numeric tolerance: `atol = 1e-5` for closed-form references, `atol = 1e-6` for ratio=1 reductions.

### Step 1 specifics (GAE)

- Hand-computed reference iterating `delta_t = r_t + γ V_{t+1}(1-d_t) - V_t; gae_t = delta_t + γλ(1-d_t) gae_{t+1}` from `t=T-1` down to `0`.
- Two cases: no done flags; done in the middle.
- Asserts shape `(4,)`, dtype `torch.float32`, max-abs-diff `< 1e-5`.

### Step 2 specifics (sample_action)

- Constructs a tiny env (`gym.make("MountainCarContinuous-v0")`) and a `PPOAgent` over it.
- Single-obs and batched-obs shape/dtype.
- Stochastic sampling: 1000 calls, asserts variance `> 1e-6` and all samples in `[action_min, action_max]`.
- Deterministic mode: 100 calls, asserts variance `< 1e-10`.

### Step 3 specifics (evaluate_actions)

- Independent reference via `torch.distributions.Normal(actor(obs), log_std.exp())`.
- Asserts shapes `(B,)`, finite values, max-abs-diff `< 1e-5` for both `log_probs` and `entropy`.

### Step 4 specifics (ppo_loss)

- Three sub-checks:
  1. Ratio = 1 (`new_log_probs == old_log_probs`): loss equals `-advantages.mean()` within `1e-6`; `requires_grad=True`; scalar shape.
  2. Clipped branch with positive advantage (force `ratio = exp(1.0) ≈ 2.718`, `adv > 0`): gradient w.r.t. `new_log_probs` must be ≈ 0.
  3. Clipped branch with negative advantage (force `ratio = exp(-1.0) ≈ 0.368`, `adv < 0`): gradient w.r.t. `new_log_probs` must be ≈ 0.

### Step 5 specifics (train smoke)

- `gym.make("MountainCarContinuous-v0")`, `total_timesteps=512`, `rollout_size=256`, `n_epochs=2`, `batch_size=64` (override via `agent.hyperparameters[...]` before calling `train`).
- Wall-clock budget: `< 10 s`.
- Asserts the returned dict contains `{"mean_reward", "policy_loss", "value_loss", "entropy", "n_updates"}`, all non-NaN; `n_updates > 0`.
- Captures the printed log lines and asserts no NaN in any `policy_loss=` token.

### Exit codes

- `0` — every run step returned `PASS`.
- `1` — at least one step returned `FAIL` or `NOT_IMPLEMENTED`.
- `2` — invalid `--step N`.

## File 2: `test_agent_interface.py`

### CLI

```bash
uv run python workshop-1/1-ppo/ppo/tests/test_agent_interface.py --agent ppo
uv run python workshop-1/1-ppo/ppo/tests/test_agent_interface.py --agent sb3   # exits 2 (out of scope)
```

### Steps

| Step | Code | Targets | Notes |
|---|---|---|---|
| 1 | C1 | `_AGENT_REGISTRY` contains `"PPOAgent"` | Structural; no construction |
| 2 | C3 | `predict(raw_obs).shape == (action_dim,); dtype float32; in [-1, 1]` | Constructs against a tiny env |
| 3 | C3-det | `predict(raw, deterministic=True)` is identical across calls | |
| 4 | C4 | `train(total_timesteps=512)` smoke returns dict with documented keys | < 10 s |
| 5 | C5-base | `save → load → predict(raw, deterministic=True)` round-trip equality | uses `tempfile.mkstemp(suffix=".pt")` |
| 6 | C5-subclass | A locally-defined `@register_agent` subclass survives save/load via `_AGENT_REGISTRY` | tests the registry path |
| 7 | C7 | `evaluate(env, n_episodes=2, record_video=False)` returns `list[float]` of length 2 with finite values; `record_video=True` writes `eval.mp4` (or `eval.mp4.skipped`) under a `tempfile.TemporaryDirectory()` | NEW |

### Removed steps (vs. pre-refactor)

- ~~Old step 2/3/4 (C2 preprocess identity, determinism, subclass override)~~ — `preprocess()` was removed in the refactor.
- ~~Old step 10 (C6 `_get/_set_preprocess_state` no-op)~~ — methods removed.
- The new step numbering is contiguous (1..7); the C-codes (C1, C3, C4, C5, C7) come from the constitution but are renumbered locally to keep the runner output clean.

### Exit codes

- `0` — every step returned `PASS`.
- `1` — at least one step returned `FAIL` or `NOT_IMPLEMENTED`.
- `2` — `--agent sb3` or unrecognized agent (matches pre-refactor convention).

## Acceptance criteria (cross-references)

| Spec FR | Met by |
|---|---|
| FR-012 | `test_ppo.py` steps 1..5 |
| FR-013 | `test_agent_interface.py` steps 1..7 |
| FR-014 | Both files are runnable from repo root via `uv run python ...`; combined budget `< 60 s` |
| FR-015 | All file output via `tempfile`/`TemporaryDirectory`; no network |
| User Story 2 acceptance #1 | All steps `PASS` when TODOs are filled |
| User Story 2 acceptance #2 | When TODO 1 is unfilled, `test_ppo.py` step 1 reports `NOT_IMPLEMENTED`; steps 2/3/4 still run independently |
| User Story 2 acceptance #3 | `test_agent_interface.py` step 5 (predict equality across save/load) |
| User Story 2 acceptance #4 | `test_ppo.py` step 5 (train smoke returns documented keys) |
