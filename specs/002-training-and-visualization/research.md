# Research: Stage Training Drivers, Structured Logging, and Analysis Notebooks

**Date**: 2026-04-29
**Goal**: Resolve open implementation questions for plan.md before writing tasks.

## R1 — Eval-video recording: `gymnasium.wrappers.RecordVideo`

**Decision**: Use `gymnasium.wrappers.RecordVideo` on a freshly-constructed evaluation env (separate from the training env) with a fixed `episode_trigger=lambda _: True` recording all (and there's only one) eval episodes. Add `imageio[ffmpeg]` to `workshop1` dependencies. Wrap the recording in a `try/except` that on failure writes `eval.mp4.skipped` containing the exception message and returns successfully — training artifacts are not blocked.

**Rationale**: `RecordVideo` is the Gymnasium-canonical path; backed by `moviepy`/`imageio-ffmpeg`. Headless macOS/Linux work fine when `imageio-ffmpeg` is present (it bundles a portable ffmpeg binary). The `try/except` fallback is required by FR-015. CarRacing already constructs cleanly with `render_mode="rgb_array"`; MountainCarContinuous-v0 likewise.

**Alternatives considered**:
- Hand-rolled frame collection + `imageio.mimsave(...)` writing GIF/MP4 — works without ffmpeg, but produces larger files and lower quality, and means re-implementing what `RecordVideo` already does. Acceptable as the *fallback* path inside the `except`, not as the primary.
- Skip video entirely — rejected, US1 / US3 acceptance scenarios require it, and inline `eval.mp4` playback is the highest-impact piece of the notebook.

## R2 — SB3 per-rollout metric extraction

**Decision**: Subclass `stable_baselines3.common.callbacks.BaseCallback` overriding `_on_rollout_end()`. Read scalars from `self.logger.name_to_value` (SB3 publishes `train/loss`, `train/policy_gradient_loss`, `train/value_loss`, `train/entropy_loss`, `train/clip_fraction`, `train/approx_kl`, `rollout/ep_rew_mean`, `rollout/ep_len_mean`, `time/total_timesteps`, `time/iterations`). Map into the canonical schema:
- `update` ← `time/iterations`
- `timesteps` ← `time/total_timesteps`
- `policy_loss` ← `train/policy_gradient_loss`
- `value_loss` ← `train/value_loss`
- `entropy` ← `-train/entropy_loss` (SB3's "entropy_loss" is the negative of entropy; flip the sign to match the canonical positive-entropy convention)
- `mean_return` ← `rollout/ep_rew_mean`
- `log_std_mean` ← `np.exp(model.policy.log_std).mean().item()` (reach into the policy directly; SB3 doesn't log it)
- `grad_norm` ← not exposed by SB3; emit `null` and document under `metric_definitions` in `meta.json`
- `wall_time_seconds` ← `time.monotonic() - start_time` (callback owns its own clock)

**Rationale**: `_on_rollout_end()` is the canonical hook point and fires exactly once per PPO rollout, matching the "one record per update" cadence. `name_to_value` is SB3's documented logger surface.

**Alternatives considered**:
- Override `_on_step()` and emit per-step records — wrong cadence (much noisier), incompatible with stage-2 schema.
- Use a custom logger via `model.set_logger(...)` — deeper SB3 internals, more code, no real benefit over `name_to_value`.

## R3 — Default hyperparameters for CarRacing CNN PPO

**Decision**: For the **custom PPO** CarRacing path, override `PPOAgent.DEFAULT_HYPERPARAMS` in `CarRacingPPOAgent.__init__` with: `rollout_size=2048`, `batch_size=128`, `n_epochs=10`, `lr=2.5e-4`, `gamma=0.99`, `gae_lambda=0.95`, `clip_eps=0.1`, `value_coef=0.5`, `entropy_coef=0.0`, `max_grad_norm=0.5`. For the **SB3** path, use SB3 defaults — they're already tuned for `CnnPolicy` on Atari-like envs (`n_steps=128`, `batch_size=256`, `n_epochs=4`, `lr=2.5e-4`, `clip_range=0.1`, etc.) and modifying them in workshop-time is out of scope for this feature.

**Rationale**: CleanRL's reference PPO-Atari hyperparameters and SB3's `ppo_atari.yaml` zoo entry both converge on `lr=2.5e-4`, `clip_range=0.1` (smaller than the 0.2 default), and 10 epochs for image-based PPO. Per-rollout size of 2048 fits comfortably in laptop RAM at `(4, 84, 84)` float32 (≈ 26 MiB / rollout). Entropy coef 0 is the standard CarRacing setting (the policy std handles exploration).

**Alternatives considered**:
- Reuse stage-1 defaults (`lr=3e-4`, `clip_eps=0.2`, etc.) — these are tuned for vector envs and are too aggressive for pixel inputs; entropy collapses and the policy degenerates.
- Expose all hyperparameters via CLI — explicitly rejected during clarify (Q3, Option A).

## R4 — `RolloutBuffer` shape generalization

**Decision**: Replace `np.zeros((size, obs_dim), dtype=np.float32)` with `np.zeros((size, *obs_shape), dtype=np.float32)`. Constructor signature changes from `RolloutBuffer(size, obs_dim, action_dim)` to `RolloutBuffer(size, obs_shape, action_dim)` where `obs_shape: tuple[int, ...]`. For backward compatibility, accept an `int` and treat it as `(int,)` — that preserves the existing call site `RolloutBuffer(rollout_size, obs_dim, action_dim)` in `ppo.train()` exactly when `obs_dim` is a Python int. `RolloutBuffer.add(obs, ...)` already accepts an ndarray for `obs`; no change needed there. `get_batches()` already preserves shape via fancy indexing.

**Rationale**: Backward-compat shim keeps existing tests passing (FR-018, SC-003). The `*obs_shape` star-unpack is the obvious idiom and adds zero compute overhead.

**Alternatives considered**:
- Force the new tuple-only signature — breaks `test_ppo.py --step 5` and the `_main()` CartPole runner; rejected per FR-018.
- Use a list of arrays (per-step variable shape) — overkill; observations are uniformly-shaped within a rollout.

## R5 — Notebook output policy

**Decision**: Strip outputs manually before commit (run "Restart Kernel and Clear Outputs" then save). Do NOT add a tooling dependency on `nbstripout` or a `pre-commit` hook for v1 — the workshop's `pyproject.toml` is intentionally minimal and the maintainer commits notebooks rarely (≈ once per design change). Document the convention in `workshop-1/README.md`.

**Rationale**: A pre-commit hook is a long-term workflow artifact, not v1 scope. Manual clear-on-save is a 2-second discipline that suffices for a teaching repo with a small maintainer team.

**Alternatives considered**:
- `nbstripout --install` per-clone — adds a per-participant install step, hostile to beginners.
- Commit notebooks WITH outputs — diffs become unreadable, repo bloats.

## R6 — `imageio[ffmpeg]` dependency

**Decision**: Add `imageio[ffmpeg]>=2.31` to the `workshop1` dependency group in `pyproject.toml`. Refresh `uv.lock` in the same change.

**Rationale**: `gymnasium.wrappers.RecordVideo` requires either `moviepy` or a working ffmpeg binary; `imageio-ffmpeg` is the most portable option (bundles a static ffmpeg binary across macOS/Linux/Windows). Adding it costs ~30 MB on disk per install — acceptable.

**Alternatives considered**:
- System `ffmpeg` (apt/brew) — unreliable across participant machines.
- `moviepy` — heavier dependency tree (PIL, decorator, proglog, requests) for less benefit.

## R7 — `agent` kwarg in `ppo.train()` — backward compatibility surface

**Decision**: Add `agent: PPOAgent | None = None` as a keyword-only argument with default `None`. When `None`, `train()` runs in "raw observations" mode (current behavior). When provided, `train()`:
1. Calls `agent.reset_preprocess_state()` immediately before `env.reset(seed=seed)` and after every termination.
2. Calls `agent.preprocess(raw_obs)` on every observation from `env.reset` / `env.step` before storing in the rollout buffer or feeding into networks.
3. Replaces the old `obs_dim = int(np.prod(env.observation_space.shape))` with `sample_obs = agent.preprocess(env.observation_space.sample())` then `obs_shape = sample_obs.shape` to size the rollout buffer correctly. (When `agent` is `None`, the old path is taken.)

`PPOAgent.train()` is updated to pass `agent=self` plus `metrics_fn` through.

**Rationale**: Keyword-only with `None` default is the lowest-disruption way to introduce the agent dependency: existing callers (the `_main()` CartPole runner, `test_ppo.py --step 5`) don't pass `agent` and so see no behavior change. The shape-detection shim handles the new "preprocessed shape != raw env shape" case cleanly.

**Alternatives considered**:
- Pass `(preprocess_fn, reset_preprocess_fn)` callables instead of the agent — slightly more decoupled, but two args instead of one and `ppo.train()` already conceptually "knows" about an agent through `metrics_fn` and the actor/value/log_std it receives. Rejected for ergonomics.
- Refactor to `train(env, agent, total_timesteps, ...)` and break the `(env, actor, value, log_std)` signature — breaks tests and skeleton; rejected.

## R8 — Auto-discovery semantics in `analyze.ipynb`

**Decision**: Default `RUN_DIR` cell sets `RUN_DIR = None`. The next cell, if `RUN_DIR is None`, picks `runs/<stage>/*` directories sorted by mtime descending and selects the first; on empty list, raises `RuntimeError` with a copy-pasteable hint:
> `no runs found under runs/<stage>/. Train one first via 'uv run python workshop-1/<stage>/train.py', or set RUN_DIR = 'pretrained/sample-runs/<stage>/<name>'.`

`pretrained/sample-runs/` is **not** part of auto-discovery; the participant must explicitly set `RUN_DIR` to use a sample. This makes the auto-discovery contract simple and predictable.

**Rationale**: Hiding sample runs behind an explicit override prevents the surprise of "I trained but the notebook keeps showing the lecturer's pretrained run." Erroring instead of silently using the sample matches FR-019/FR-020.

**Alternatives considered**:
- Auto-discover both `runs/` and `pretrained/sample-runs/`, prefer `runs/` if non-empty — works, but the heuristic surprises participants when they don't realize the sample took precedence.
- Symlink or copy a sample into `runs/` — pollutes participant `git status` if they ever run the driver after.

## Recommendations

All findings inform plan.md's chosen direction directly:
- R1 → eval video story works on stock laptops with one new dep
- R2 → SB3 callback path is straightforward; canonical-schema mapping documented
- R3 → CarRacing CNN PPO has a starting point that's actually likely to learn
- R4 → buffer change is trivial; backward-compat shim preserves all existing tests
- R5 → no new tooling deps for notebook hygiene
- R6 → one new line in `pyproject.toml`; lock refreshed
- R7 → `agent` kwarg surface is decided
- R8 → notebook discovery semantics decided

**No NEEDS CLARIFICATION items remain.** Proceed to Phase 1.
