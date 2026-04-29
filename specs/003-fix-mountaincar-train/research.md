# Phase 0 Research: Fix MountainCar Training Driver After PPO Refactor

**Date**: 2026-04-29
**Branch**: `003-fix-mountaincar-train`
**Spec**: [spec.md](./spec.md)

This document resolves the items the spec deferred to plan-level ("Misc / Placeholders" in the clarification coverage table), captures the agent-bug audit, and records the SB3-driver follow-up.

## R1. Refactor-bug audit (the latent bugs FR-011 must fix)

**Decision**: Fix the following six concrete defects in `workshop-1/1-ppo/ppo/ppo.py` as part of FR-011. All identified by reading the file end-to-end against the new module surface.

| # | Where | Symptom | Fix |
|---|---|---|---|
| 1 | `PPOAgent.__init__` (`log_std` line) | `nn.Parameter(torch.ones(...) * log_std_init, device=self.device)` — `nn.Parameter` does not accept a `device=` kwarg. | `self.log_std = nn.Parameter(torch.ones(self.action_dim, device=self.device) * self.hyperparameters["log_std_init"])` |
| 2 | `PPOAgent.train()` | `optimizer = torch.optim.Adam(... + list(self.value.parameters()) + ...)` — `self.value` does not exist; the attribute is `self.critic`. | Replace `self.value` with `self.critic` (one occurrence near line 256, one near 282). |
| 3 | `PPOAgent.train()` rollout loop | `self.sample_action(self.actor, obs_t, self.log_std, deterministic=False)` — old signature; the refactored signature is `sample_action(obs, deterministic=False)`. | Replace with `self.sample_action(obs_t, deterministic=False)`. |
| 4 | `PPOAgent.train()` update loop | `self.evaluate_actions(batch["obs"], batch["actions"])` — refactored signature still requires `(actor, obs, actions, log_std)`. | Either (a) update the call to `self.evaluate_actions(self.actor, batch["obs"], batch["actions"], self.log_std)`, or (b) change the method signature to `evaluate_actions(self, obs, actions)` and read `self.actor` / `self.log_std` internally. **Plan picks (b)** — it matches `sample_action`'s OO style and removes the redundant arguments. The test in step 3 of `test_ppo.py` will use the new (b) signature. |
| 5 | `PPOAgent.predict()` | Calls `self.sample_action(obs, ...)` with whatever the caller passed — but the docstring says `obs: np.ndarray`, and `sample_action` expects `torch.Tensor`. | Convert at the boundary: `obs_t = torch.as_tensor(obs, dtype=torch.float32, device=self.device)`. |
| 6 | `PPOAgent.load()` | `target_cls(hyperparameters=state["hyperparameters"])` — `__init__` requires `env`. | Change `load(cls, path)` to `load(cls, path, env)` and pass `env` through. The test in step 8 of `test_agent_interface.py` exercises this path. |

**Rationale**: All six are observable from a static read. Each has exactly one minimal fix. None require a redesign of the public API.

**Alternatives considered**: Reverting to the flat-file `ppo.py` (rejected — the user has invested effort in the package layout). Adding compatibility shims for the old call signatures (rejected — adds dead code with no value to participants).

## R2. Pre-flight check on unfilled TODOs

**Decision**: Do not pre-flight. Allocate the run directory, then let `NotImplementedError` propagate naturally; `RunLogger.__exit__` writes `status: "error"` into `meta.json`. Print a clean two-line message ("Looks like TODO N is still raising NotImplementedError. Fill it in and re-run.") in the driver's outer `except NotImplementedError` block before re-raising for the exit code.

**Rationale**: Pre-flighting GAE alone wouldn't catch unfilled `sample_action` (which fires first). Pre-flighting all five TODOs requires synthetic inputs that mirror the real shapes — extra glue code for marginal UX gain. The status-tagged run dir is acceptable: it shows up in `runs/mountaincar/` as `status: "error"` and the participant deletes it on the next attempt by passing `--force`.

**Alternatives considered**:
- (a) Pre-flight call to all five TODOs with synthetic inputs — rejected: adds ~30 lines of fixture code for an edge case that occurs once per participant.
- (b) Wrap the run-dir allocation inside a `try` that cleans up on `NotImplementedError` — rejected: hides the artifact that the participant might want to inspect (e.g. whether `meta.json` even got written).

## R3. Video recording mechanism for `PPOAgent.evaluate()`

**Decision**: Use `gymnasium.wrappers.RecordVideo` with `name_prefix="eval"` and `video_length=1000` (long enough for a single MountainCar episode), wrapping `gym.make(env_id, render_mode="rgb_array")`. After episodes finish, rename the produced `eval-episode-0.mp4` to `eval.mp4` so the run directory matches the contract from feature 002. If `imageio[ffmpeg]` is unavailable, catch `ImportError` (or `gymnasium.error.DependencyNotInstalled`), log one line via `print(..., file=sys.stderr)`, write a sentinel `eval.mp4.skipped` (per the run-format contract), and return the per-episode returns without a video.

**Rationale**:
- `RecordVideo` is the documented Gymnasium standard and the workshop already declares `imageio[ffmpeg]` in its dependency group.
- The rename step keeps the on-disk schema stable for `analyze.ipynb`, which already expects `eval.mp4`.
- The `eval.mp4.skipped` sentinel matches the format in `specs/002-training-and-visualization/contracts/run-format.md` ("`eval.mp4` | `eval.mp4.skipped`") — no schema change.

**Alternatives considered**:
- Manual `env.render()` + `imageio.mimsave(...)` — rejected: ~15 lines of glue, manual frame-buffer management, and we'd still need `imageio` so the dependency story is identical.
- Saving `.gif` instead of `.mp4` — rejected: violates the existing schema; analyze.ipynb does not consume `.gif`.

## R4. Package import path / `__init__.py` strategy

**Decision**: Add three `__init__.py` files:

- `workshop-1/1-ppo/ppo/__init__.py` — re-exports `PPOAgent`, `RolloutBuffer`, `ActorNetwork`, `CriticNetwork`, `register_agent`, `_AGENT_REGISTRY`, and re-exports the utilities from `ppo.utils`.
- `workshop-1/1-ppo/ppo/utils/__init__.py` — re-exports `seed_everything`, `format_update_line`, `get_device`, `RunLogger`, `RunDirectoryExistsError`, `make_log_fn`, `parse_update_line`.
- `workshop-1/1-ppo/ppo/tests/__init__.py` — empty (presence makes `ppo.tests` a regular package so the test files can be referenced unambiguously).

The driver gains a single `sys.path.insert(0, str(_HERE.parent / "1-ppo"))` line (the same pattern `train_sb3.py` uses today). After that, the imports are flat: `from ppo import PPOAgent`, `from ppo.utils import RunLogger, RunDirectoryExistsError, make_log_fn, seed_everything`.

**Rationale**: A single, focused `sys.path.insert` is the workshop convention (pyproject.toml does not install workshop modules as packages — `dependencies = []`). It is *not* a "hack" in the sense FR-002 prohibits: that requirement bans the *three-line* path-soup currently in `train.py` (one for `_HERE`, one for `_WORKSHOP1`, one for `1-ppo`) and references to deleted top-level helper modules. A single insert at the package root is acceptable.

**Alternatives considered**:
- Add `[tool.setuptools]` entries to `pyproject.toml` and `uv sync` to install — rejected: increases install friction for participants; out of scope.
- Restructure `workshop-1/` into `workshop_1/` (importable name) — rejected: invasive directory rename, breaks every existing path in READMEs and CLAUDE.md.
- Use absolute path with `runpy` — rejected: workshop participants will not understand it.

## R5. Test environment for the `train()` smoke step

**Decision**: Use `gym.make("MountainCarContinuous-v0")` exactly as the pre-refactor `test_ppo.py` did. Limit `total_timesteps=512`, `rollout_size=256`, `n_epochs=2`, `batch_size=64`, asserting `< 10 s` wall clock. No stub env.

**Rationale**: The pre-refactor convention used MountainCarContinuous (commit `60321eb` step 5). Pendulum is also continuous-action `Box(1,)` and would work, but switching environments would diverge from the pre-refactor reference and require a separate justification. MountainCar has no rendering in the test path, so it is fast enough.

**Alternatives considered**: Pendulum-v1 (rejected — no benefit), custom 1D stub env (rejected — drifts from the canonical environment matrix in the constitution).

## R6. SB3 driver follow-up (out of scope, surfaced for visibility)

**Status**: Not in scope for this feature. Documented here so the next planning cycle picks it up.

`workshop-1/2-mountaincar/train_sb3.py` currently imports three deleted modules:

```python
from _runlog import RunLogger, RunDirectoryExistsError
from _eval import record_eval_episode
from _sb3_jsonl_callback import Sb3JsonlCallback
```

Those modules used to live at `workshop-1/_runlog.py`, `workshop-1/_eval.py`, `workshop-1/_sb3_jsonl_callback.py` and were deleted in the refactor. After this feature lands, the SB3 driver will still be broken. Recommended follow-up:

- Move `_eval.py` (or the `record_eval_episode` function) into `workshop-1/1-ppo/ppo/utils/_eval.py` and re-export from `ppo.utils`.
- Update `train_sb3.py` to use the new import paths — same `sys.path.insert(0, str(_HERE.parent / "1-ppo"))` pattern.
- Keep `_sb3_jsonl_callback.py` where it is (`ppo/utils/`); it already moved correctly.

This is a small follow-up spec ("`004-fix-sb3-driver`" or merged into the next feature) — it does not need its own clarification cycle since the changes mirror this feature's driver fix.

## R7. Constitution amendment recommendation (follow-up)

**Status**: Recommended follow-up. Does not block this feature.

The plan's Complexity Tracking documents three Article II / IV / VII deviations driven by the spec's Q4 choice. Per the constitution's own Governance section (*"Participant-experience override … open an amendment PR updating the principle"*), the right next step is a constitution amendment that:

1. Either removes Article II's `preprocess()` requirement, or scopes it as image-observation-only and explicitly permits driver-level `gym.ObservationWrapper` chains for vector envs.
2. Updates Article IV to drop C2/C6 from the `test_agent_interface.py` contract and add a C7 `evaluate()` clause.
3. Updates Article VII to require Workshop 2 drivers to expose a shared `_make_env(env_id, ...)` helper that the Pi loader can re-invoke (so the wrapper chain still travels with the model, just in a different way).

This amendment ships with the SB3 follow-up (R6) so all three pieces land together.
