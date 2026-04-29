# Feature Specification: Stage Training Drivers, Structured Logging, and Analysis Notebooks

**Feature Branch**: `002-training-and-visualization`
**Created**: 2026-04-29
**Status**: Draft
**Input**: Add per-stage training drivers and analysis notebooks to `workshop-1/2-mountaincar/` and `workshop-1/3-car-racing/`. Each `train.py` is a thin driver writing a self-contained run directory (`runs/<stage>/<timestamp>/{meta.json, metrics.jsonl, model.pt, eval.mp4}`); each `analyze.ipynb` reads it and renders training curves, an inline eval video, and a value/policy heatmap (MountainCar). Plumb structured metrics out of `ppo.train()` via an additive `metrics_fn` kwarg. Custom PPO is primary on both stages; SB3 stays as an alternative on CarRacing using the same JSONL schema. In-scope enabling work for the CarRacing custom path: CNN actor + value, Gymnasium preprocess wrappers, and a `RolloutBuffer` shape generalization.

## Clarifications

### Session 2026-04-29

- Q: Default `total_timesteps` for each stage's `train.py`? → A: MountainCar 50k, CarRacing 200k
- Q: Run directory location + git-tracking? → A: Repo-root `runs/` fully gitignored; sample runs under tracked `pretrained/sample-runs/<stage>/`
- Q: CLI hyperparameter overrides on `train.py`? → A: None — only `--timesteps`, `--seed`, `--run-name`, `--no-eval`; participants edit `train.py` to vary other hyperparameters
- Q: Does MountainCar also ship an SB3 alternative driver? → A: Yes — `workshop-1/2-mountaincar/train_sb3.py` using `PPO("MlpPolicy", ...)` with the same JSONL callback as CarRacing
- Q: `--run-name` collision behavior when the directory already exists? → A: Fail with a clear error and non-zero exit; opt-in `--force` flag overwrites
- Q: How is CarRacing pixel preprocessing wired into training (post-clarify constitution check)? → A: For **custom PPO**, `ppo.train()` gains an optional `agent` kwarg; when provided, it calls `agent.preprocess(obs)` and `agent.reset_preprocess_state()` directly in the rollout loop (Article II: `agent.preprocess()` is the single source of truth — no wrapper, no separate transform code). For **SB3**, standalone Gymnasium observation wrappers are permitted as a **scoped exemption to Article II**, on the grounds that SB3's training loop is closed and cannot accept `agent.preprocess()` injection.

## User Scenarios & Testing *(mandatory)*

### User Story 1 — Beginner trains a MountainCar agent and inspects results (Priority: P1)

A workshop participant has just finished the PPO TODOs in stage 1. In their ~15-minute MountainCar slot, they run a single command to train a custom PPO agent on `MountainCarContinuous-v0`, watch fixed-width metric lines stream to the console, then open a notebook that loads the latest run and shows them training curves, a 2-D value/policy heatmap, and a video of one greedy evaluation episode.

**Why this priority**: This is the primary workshop flow for stage 2. Without it, stage 2 has no entry point — the existing `MountainCarPPOAgent` stub has no driver script and the existing `mountaincar.ipynb` is empty. P1 because every other story builds on the run-directory contract this story establishes.

**Independent Test**: From a clean checkout, run `uv run python workshop-1/2-mountaincar/train.py` to completion (≤ 5 min on a laptop CPU), then open `workshop-1/2-mountaincar/analyze.ipynb` and run all cells. Success means: a `runs/mountaincar/<timestamp>/` directory exists with the four expected files, the notebook plots without manual editing, and the eval video plays inline.

**Acceptance Scenarios**:

1. **Given** the participant has implemented all five PPO TODOs and is on a fresh repo checkout, **When** they run `uv run python workshop-1/2-mountaincar/train.py`, **Then** training proceeds to completion, one fixed-width metric line is printed per PPO update, and a `runs/mountaincar/<timestamp>/` directory is written containing exactly `meta.json`, `metrics.jsonl`, `model.pt`, and `eval.mp4`.
2. **Given** at least one completed MountainCar run exists on disk, **When** the participant runs all cells of `workshop-1/2-mountaincar/analyze.ipynb`, **Then** the notebook auto-discovers the latest run, plots episode return / policy_loss / value_loss / entropy curves from `metrics.jsonl`, renders a value-function heatmap and a policy-mean quiver over the position×velocity grid, and embeds `eval.mp4` for inline playback — all without the participant editing any cell.
3. **Given** training has been interrupted with Ctrl+C after a few PPO updates, **When** the participant inspects `runs/mountaincar/<timestamp>/`, **Then** `metrics.jsonl` contains the partial set of updates already completed (one record per update, written atomically per line) and `meta.json` exists with the start-time and run hyperparameters; `model.pt` and `eval.mp4` may or may not exist.

---

### User Story 2 — Participant pastes a training log into Claude / Copilot to debug (Priority: P1)

A participant's training is misbehaving (e.g., entropy not decreasing, returns flat, NaN losses). They `tail` `metrics.jsonl` and `meta.json`, paste both into a Claude / Copilot chat, and get useful diagnostic suggestions because the log is fully self-describing — every line is a complete dict including update index, timesteps, all loss components, current `log_std.exp()`, and gradient norm.

**Why this priority**: This is the explicit reason the user prefers JSONL over TensorBoard. Workshop participants are beginners; their first debugging move *is* asking an LLM. P1 because the format choice is load-bearing for the entire feature.

**Independent Test**: Run a training to completion, then `head -1 metrics.jsonl | python -m json.tool` and confirm every key documented in the spec is present. Pipe the file plus `meta.json` into a fresh Claude session asking "is anything obviously wrong?" — the question must be answerable from the log alone (no other context needed).

**Acceptance Scenarios**:

1. **Given** a completed training run, **When** the participant runs `cat runs/<stage>/<timestamp>/meta.json`, **Then** they see a single JSON object containing at minimum: `stage`, `env_id`, `agent_class`, `seed`, `total_timesteps`, `hyperparameters` (full dict), `git_sha`, `started_at` (ISO 8601), `finished_at` (ISO 8601 or `null` if still running), `python_version`, `torch_version`, and `gymnasium_version`.
2. **Given** a completed training run, **When** the participant runs `head -1 runs/<stage>/<timestamp>/metrics.jsonl | python -m json.tool`, **Then** they see a single JSON object containing at minimum: `update`, `timesteps`, `policy_loss`, `value_loss`, `entropy`, `mean_return` (rolling mean of the last 10 episode returns or current partial return if no episode finished), `log_std_mean` (current `log_std.exp().mean().item()`), `grad_norm` (the post-clip gradient norm; or pre-clip if simpler — see Assumptions), and `wall_time_seconds` (seconds since training start).
3. **Given** a `metrics.jsonl` file from a broken run pasted into Claude, **When** the LLM is asked "what's wrong?", **Then** the file alone provides enough information to diagnose common failure modes (NaN loss, entropy stuck, returns not improving) without the LLM needing to read source code — i.e., the per-line dict carries its own schema and the values are interpretable.

---

### User Story 3 — Participant trains and analyzes CarRacing with custom PPO (Priority: P1)

A participant moves to the CarRacing slot. They run `uv run python workshop-1/3-car-racing/train.py`, which trains a CNN-equipped `CarRacingPPOAgent` on a wrapped `CarRacing-v3` env (grayscale → 84×84 → frame-stack 4) using the same `ppo.train()` function from stage 1. After training, they open `workshop-1/3-car-racing/analyze.ipynb`, which auto-loads the latest run, plots curves, and embeds `eval.mp4` of one greedy episode.

**Why this priority**: P1 because CLAUDE.md and `workshop-1/README.md` both promise CarRacing is reachable end-to-end, and the existing `CarRacingPPOAgent` stub is explicitly not runnable (`train()` will crash on the stage-1 MLP per its own docstring). Without this story, stage 3 has no working primary path.

**Independent Test**: From a clean checkout (with stage 1 TODOs complete), run `uv run python workshop-1/3-car-racing/train.py --timesteps 50000` to completion, then open `workshop-1/3-car-racing/analyze.ipynb` and run all cells. Success means: training completes without shape errors, the run directory has all four files, and the notebook plots and plays the eval video.

**Acceptance Scenarios**:

1. **Given** the participant has completed stage 1 TODOs, **When** they run `uv run python workshop-1/3-car-racing/train.py`, **Then** the env is `gymnasium.make("CarRacing-v3", render_mode="rgb_array")` with **no observation wrappers**; pixel preprocessing happens inside `ppo.train()` via `CarRacingPPOAgent.preprocess()` (the single source of truth per Article II); a CNN-equipped `CarRacingPPOAgent` trains end-to-end so observations enter `RolloutBuffer` with shape `(4, 84, 84)`; and a `runs/car-racing/<timestamp>/` directory is written with the same four files as stage 2.
2. **Given** training of CarRacing is in progress, **When** the participant inspects the running process, **Then** they see the same fixed-width per-update log lines as stage 2 (same `format_update_line` format) and the JSONL records share the same schema (no stage-specific divergence).
3. **Given** a completed CarRacing run, **When** the participant runs all cells of `workshop-1/3-car-racing/analyze.ipynb`, **Then** training curves are plotted from `metrics.jsonl` and `eval.mp4` plays inline. (The MountainCar-specific value/policy heatmap is *not* shown because the state space is high-dimensional pixels.)

---

### User Story 4 — Participant uses Stable-Baselines3 escape hatch (either stage) (Priority: P2)

A participant who could not complete the stage 1 TODOs (or is stuck on the CarRacing CNN refactor, or is short on time) can run `uv run python workshop-1/2-mountaincar/train_sb3.py` or `workshop-1/3-car-racing/train_sb3.py` and get a working SB3 PPO baseline that writes the *same* `runs/<stage>/<timestamp>/*` directory layout. The same `analyze.ipynb` per stage works against either custom-PPO runs or SB3 runs.

**Why this priority**: P2 because CLAUDE.md establishes SB3 as the documented escape hatch ("SB3 is the escape hatch — nobody should get stuck and miss the fun parts"). Without an SB3 path on stage 2, a participant who failed any of the stage 1 TODOs is locked out of the entire workshop. With one on each stage, there is always at least one working path forward.

**Independent Test**: With the custom-PPO agents *intentionally broken* (e.g., raise `NotImplementedError` somewhere in `ppo.train()`), run each `train_sb3.py` to completion and confirm both produce a complete run directory and both stage notebooks render them.

**Acceptance Scenarios**:

1. **Given** the participant has the workshop1 dependency group installed, **When** they run `uv run python workshop-1/3-car-racing/train_sb3.py`, **Then** SB3's `PPO("CnnPolicy", ...)` trains on a wrapped `CarRacing-v3` env, an SB3 callback writes `metrics.jsonl` records matching the stage-2 schema (same keys, same units), and a `runs/car-racing/<timestamp>/` directory is produced with `meta.json`, `metrics.jsonl`, `model.pt` (or `model.zip` — see Assumptions), and `eval.mp4`.
2. **Given** the participant has the workshop1 dependency group installed, **When** they run `uv run python workshop-1/2-mountaincar/train_sb3.py`, **Then** SB3's `PPO("MlpPolicy", ...)` trains on `MountainCarContinuous-v0`, the SB3 callback writes `metrics.jsonl` records with the same schema as the CarRacing SB3 driver, and a `runs/mountaincar/<timestamp>/` directory is produced with the four expected files.
3. **Given** an SB3-produced run directory in either stage, **When** the participant runs the corresponding `analyze.ipynb`, **Then** the notebook works without modification — the analysis is path-agnostic with respect to which driver produced the run.

---

### User Story 5 — Lecturer ships a sample run for offline demos (Priority: P3)

The workshop maintainer commits a `runs/<stage>/sample/` directory (or pretrained equivalent) so the lecturer can demo `analyze.ipynb` on a real, completed run without needing to wait for live training during a 15-minute slot.

**Why this priority**: P3 because it's a workshop-logistics affordance, not a participant-facing feature. Tracked here so the run-directory layout decision accommodates it (path-agnostic notebook + non-timestamp directory names like `sample/` must still load).

**Independent Test**: Drop a `runs/mountaincar/sample/` directory containing the four expected files into the repo, open `analyze.ipynb`, and confirm it can load the sample run via either auto-discovery (latest by mtime) or by an explicit `RUN_DIR = "..."` cell variable that participants can edit.

**Acceptance Scenarios**:

1. **Given** a `pretrained/sample-runs/<stage>/<run-name>/` directory exists, **When** the participant edits the `RUN_DIR` cell variable in `analyze.ipynb` to point at the sample, **Then** the notebook loads and renders that run.
2. **Given** auto-discovery is restricted to `runs/<stage>/` (which is gitignored and may be empty on a fresh clone), **When** no participant runs exist yet, **Then** the notebook MUST surface a clear, non-cryptic message ("no runs yet — train one with `train.py`, or set `RUN_DIR = 'pretrained/sample-runs/<stage>/<name>'`") rather than silently auto-loading a sample.

---

### Edge Cases

- **Training interrupted by Ctrl+C** before the first PPO update completes: the run directory and `meta.json` already exist, but `metrics.jsonl` may be empty and `model.pt` / `eval.mp4` are absent. `analyze.ipynb` must show a clear, non-cryptic error in that single case ("this run has no metric records yet — train longer") rather than crashing on an empty-DataFrame plot.
- **Disk full while writing JSONL**: the `RunLogger.__call__` is best-effort; an `OSError` from a write must not crash the training loop. Log a single warning and continue training. Other failure modes (model save failing at end) follow the same best-effort policy at the end-of-training boundary only.
- **Two `train.py` runs starting in the same second**: the timestamp-suffixed directory name must be unique. If a collision is possible (sub-second start), append a short uniquifier or use `YYYYMMDD-HHMMSS-<pid>`.
- **`eval.mp4` fails to record** (no codec, headless display issue): training succeeded; do not fail the whole run. Write a `eval.mp4.skipped` marker file with the recorded reason and let the notebook show a graceful "no video for this run" message.
- **Custom-PPO MountainCar fails to learn within `total_timesteps`**: the run completes successfully and the artifacts are produced. Solving MountainCar is not a success criterion of this feature; producing analyzable artifacts is.
- **`mountaincar.ipynb` already exists empty (0 bytes)** in the working tree: it is replaced by `analyze.ipynb`. The empty file is removed so two notebooks do not coexist.
- **GitHub Actions / headless CI**: video recording requires `imageio-ffmpeg`; if unavailable, the `.skipped` marker path applies.

## Requirements *(mandatory)*

### Functional Requirements

- **FR-001**: `ppo.train()` MUST accept an optional `metrics_fn: Callable[[dict], None] | None = None` kwarg that, when provided, is invoked exactly once per PPO update with a structured dict containing all metrics enumerated in US2 acceptance scenario 2. When `None`, behavior MUST be identical to the current implementation (backward-compatible; no caller change required).
- **FR-001b**: `ppo.train()` MUST also accept an optional `agent: PPOAgent | None = None` kwarg. When provided, `train()` MUST call `agent.reset_preprocess_state()` immediately before each `env.reset()` (the initial reset and every post-`done` reset) and MUST call `agent.preprocess(raw_obs)` on every observation returned from `env.reset()` / `env.step()` before storing it in the rollout buffer or passing it to the actor / value networks. When `agent` is `None`, behavior MUST be identical to the current implementation (backward-compatible). This is the constitutionally-required path for Article II compliance: `agent.preprocess()` is the single source of truth for observation transforms; no wrapper exists.
- **FR-002**: `PPOAgent.train(env, total_timesteps, **kwargs)` MUST pass `agent=self` to the underlying `ppo.train()` (so that `preprocess` and `reset_preprocess_state` run automatically during training) and forward `metrics_fn` (and only known kwargs). Stage drivers MUST use this path; they MUST NOT contain an env-step / PPO-update loop of their own.
- **FR-003**: A shared `workshop-1/_runlog.py` MUST provide a `RunLogger(stage: str, hyperparameters: dict, env_id: str, agent_class: str, seed: int)` class. Its constructor MUST create `runs/<stage>/<YYYYMMDD-HHMMSS>/` (with a uniquifier in the case of sub-second collision), open a `metrics.jsonl` file, write a `meta.json`, and expose a `__call__(metrics: dict) -> None` that appends one JSON line per call. It MUST also expose a `close(status: str = "ok") -> None` that updates `meta.json` with `finished_at` and `status`.
- **FR-004**: `RunLogger` writes MUST be best-effort — an `OSError` during `__call__` MUST NOT propagate; it MUST be logged once via `print(... file=sys.stderr)` and subsequent updates SHOULD continue to attempt writes (with a small backoff to avoid log spam).
- **FR-005**: `meta.json` MUST contain at minimum the keys listed in US2 acceptance scenario 1.
- **FR-006**: `metrics.jsonl` MUST contain at minimum the keys listed in US2 acceptance scenario 2 on every line, with stable types across lines (so `pd.read_json(..., lines=True)` produces a homogeneously-typed DataFrame).
- **FR-007**: `workshop-1/2-mountaincar/train.py` MUST: build `gymnasium.make("MountainCarContinuous-v0")`, instantiate `MountainCarPPOAgent`, call `agent.train(env, total_timesteps, metrics_fn=runlog)`, save the trained model to `runs/<stage>/<timestamp>/model.pt` via `agent.save(...)`, run **one** greedy evaluation episode via `gymnasium.wrappers.RecordVideo` writing to `runs/<stage>/<timestamp>/eval.mp4`, and call `runlog.close("ok")` (or `close("interrupted")` on KeyboardInterrupt).
- **FR-008**: `workshop-1/3-car-racing/train.py` MUST do the same as FR-007 with two differences: (a) the env is `gymnasium.make("CarRacing-v3", render_mode="rgb_array")` with **no standalone observation wrappers** — pixel preprocessing happens inside `ppo.train()` via `agent.preprocess()` per FR-001b; (b) the agent is `CarRacingPPOAgent` whose `__init__` constructs a CNN actor and CNN value network compatible with the `(4, 84, 84)` *post-preprocess* observation shape.
- **FR-009**: `CarRacingPPOAgent` MUST replace the inherited `ActorNetwork` / `ValueNetwork` MLPs with CNN equivalents (e.g., NatureCNN-style: 3 conv layers + linear head producing the action mean for the actor, scalar for the value). The CNN classes MUST live alongside the agent in `workshop-1/3-car-racing/agent.py`. The existing `CarRacingPPOAgent.preprocess()` (crop sky → grayscale → resize 84×84 → normalize → framestack 4 with internal `_frame_buffer`) MUST be preserved unchanged as the single source of truth per Article II.
- **FR-009b**: `PPOAgent` (base class in `workshop-1/1-ppo/ppo.py`) MUST gain a `reset_preprocess_state(self) -> None` method that defaults to a no-op. `CarRacingPPOAgent` MUST override it to clear `self._frame_buffer` (and reset any other stateful preprocess state any subclass introduces). `ppo.train()` calls this method at every `env.reset()` per FR-001b so the frame buffer cannot leak across episodes.
- **FR-010**: `RolloutBuffer` MUST generalize from flat `(size, obs_dim)` storage to `(size, *obs_shape)` storage. The shape stored is the **post-preprocess** shape (e.g., `(4, 84, 84)` for CarRacing, `(2,)` for MountainCar). The change MUST be backward-compatible: passing a 1-D obs space MUST produce the same buffer behavior as today (so existing stage 1 tests continue to pass).
- **FR-011**: `workshop-1/2-mountaincar/analyze.ipynb` MUST: (a) auto-discover the latest run under `runs/mountaincar/` by directory mtime if no `RUN_DIR` override is set, (b) plot four curves (mean_return, policy_loss, value_loss, entropy) from `metrics.jsonl`, (c) render a value-function heatmap over the (position, velocity) grid, (d) render a policy-mean quiver over the same grid, (e) embed `eval.mp4` for inline playback. The notebook MUST run end-to-end via "Run All" without any cell editing, given a valid run directory.
- **FR-012**: `workshop-1/3-car-racing/analyze.ipynb` MUST: (a) auto-discover the latest run under `runs/car-racing/` by directory mtime if no `RUN_DIR` override is set, (b) plot the same four curves, (c) embed `eval.mp4`. It MUST NOT include the heatmap / quiver cells (they are MountainCar-specific).
- **FR-013**: Both stages MUST ship an SB3 alternative driver: `workshop-1/2-mountaincar/train_sb3.py` (using `PPO("MlpPolicy", env, ...)` on `MountainCarContinuous-v0` with **no observation wrappers**) and `workshop-1/3-car-racing/train_sb3.py` (using `PPO("CnnPolicy", env, ...)` on `gymnasium.make("CarRacing-v3", render_mode="rgb_array")` wrapped through `GrayScaleObservation` → `ResizeObservation(84, 84)` → `FrameStackObservation(4)`). The CarRacing SB3 path is hereby granted a **scoped exemption to Article II's prohibition on standalone observation wrappers**, on the grounds that SB3's training loop is closed and cannot accept `agent.preprocess()` injection — the corresponding constitution-compliant path for custom PPO is FR-001b. Both SB3 drivers MUST share a single SB3 `BaseCallback` subclass (e.g., `workshop-1/_sb3_jsonl_callback.py`) that emits one record per rollout completion to `metrics.jsonl` with the canonical schema, mapping SB3's `logger.name_to_value` keys into the canonical names. Both drivers MUST produce the same `runs/<stage>/<timestamp>/{meta.json, metrics.jsonl, model.*, eval.mp4}` layout as the custom-PPO drivers. SB3's native model file extension (`.zip`) is acceptable in lieu of `.pt` (see Assumptions).
- **FR-014**: All driver scripts MUST accept exactly these CLI flags: `--timesteps`, `--seed`, `--run-name` (override the timestamp directory), `--no-eval` (skip the eval episode + video), and `--force` (allow overwriting an existing run directory). PPO hyperparameters (`lr`, `rollout_size`, `batch_size`, `n_epochs`, `gamma`, `gae_lambda`, `clip_eps`, `value_coef`, `entropy_coef`, `max_grad_norm`) MUST NOT be exposed as CLI flags — participants who want to experiment edit the driver script directly. Default `--timesteps` MUST be **50000 for `2-mountaincar/train.py`** and **200000 for `3-car-racing/train.py`** (and `train_sb3.py`). When `--run-name <name>` is given and `runs/<stage>/<name>/` already exists, the driver MUST exit non-zero with a clear error message (e.g., `"run directory <path> already exists; pick a different --run-name or pass --force to overwrite"`) UNLESS `--force` is also passed, in which case the existing directory is removed and re-created. The scripts MUST complete cleanly at any user-supplied `--timesteps` value (no minimum-steps requirement enforced beyond what `ppo.train()` already imposes).
- **FR-015**: Eval-episode video recording MUST gracefully degrade. If the underlying recorder raises (no codec, headless), training and the run-dir contents MUST still complete; an `eval.mp4.skipped` text file noting the reason replaces the missing video, and the notebook MUST handle this case without crashing.
- **FR-016**: `workshop-1/README.md` MUST be updated to document: how to run each `train.py`, how to open each `analyze.ipynb`, where runs are written, and the SB3 alternative for stage 3. The pre-existing per-TODO checkpoint section MUST be preserved.
- **FR-017**: The empty `workshop-1/2-mountaincar/mountaincar.ipynb` MUST be removed (replaced by the new `analyze.ipynb`). Notebooks MUST be committed with cleared outputs (`nbstripout` or manual clear) so diffs stay readable.
- **FR-018**: Existing tests MUST continue to pass: `uv run python workshop-1/1-ppo/test_ppo.py --step {1..5}` and `uv run python workshop-1/1-ppo/ppo.py` MUST behave identically to before the `metrics_fn` kwarg was added (the kwarg defaults to `None` and is a no-op).
- **FR-019**: All run directories MUST be written under the **repository root** at `runs/<stage>/<timestamp>/`, regardless of the directory from which `train.py` was invoked. The `runs/` directory MUST be added to `.gitignore` (with no exceptions) so participants' `git status` stays clean throughout the workshop.
- **FR-020**: Sample runs shipped by the workshop maintainer (US5) MUST live at the **separate, tracked** path `pretrained/sample-runs/<stage>/<run-name>/` — not inside `runs/`. Each stage's `analyze.ipynb` MUST accept an explicit `RUN_DIR` cell override that can point at either a `runs/...` directory or a `pretrained/sample-runs/...` directory; auto-discovery (latest by mtime) is restricted to `runs/<stage>/`.

### Key Entities

- **Run directory** — `runs/<stage>/<run-name>/`. The unit of "one training execution". Self-contained (no symlinks, no external state), portable (zip + share), time-ordered by directory name.
- **Run metadata (`meta.json`)** — One JSON object per run capturing everything an LLM would need to interpret the metrics file: env id, agent class, seed, hyperparameters, total_timesteps, git SHA, library versions, start/finish timestamps, run status (`ok`, `interrupted`, `error`).
- **Metric record (one line of `metrics.jsonl`)** — One JSON dict per PPO update. Self-describing per line (no header dependency), stable schema, scalar values only.
- **`RunLogger`** — In-process bridge from `metrics_fn` callbacks to the on-disk JSONL. Owns the open file handle and the `meta.json` lifecycle.
- **Stage driver (`train.py`)** — Thin ~50–80 line script per stage. Builds env + agent, instantiates `RunLogger`, calls `agent.train(..., metrics_fn=runlog)`, runs one eval episode, saves the model. Contains no env-step / PPO-update loop.
- **Analysis notebook (`analyze.ipynb`)** — Per-stage notebook that loads a run directory and renders artifacts. No training code. Path-agnostic with respect to which driver produced the run.

## Success Criteria *(mandatory)*

### Measurable Outcomes

- **SC-001**: A workshop participant on a fresh laptop checkout can go from "stage 1 TODOs done" to "MountainCar training plotted in a notebook" in **under 10 minutes wall time** (training + opening + running the notebook), with zero file edits, on default settings.
- **SC-002**: Pasting `meta.json` plus the last 50 lines of `metrics.jsonl` into Claude with the prompt "is anything obviously wrong?" produces a useful diagnostic in **at least 80% of common failure modes** (NaN loss, entropy stuck, returns flat, learning rate too high) — verified by the maintainer running ≥ 5 representative broken runs through the loop pre-workshop.
- **SC-003**: 100% of existing stage 1 tests (`test_ppo.py --step 1..5`, `ppo.py` script-mode invariants, `test_agent_interface.py`) continue to pass after the `metrics_fn` kwarg is added and `RolloutBuffer` is shape-generalized.
- **SC-004**: Stage 3 has **at least one working end-to-end path** for every workshop attendee — measured as: either the custom-PPO CNN driver completes a run *or* the SB3 driver does, on a stock laptop with the workshop1 dependency group installed.
- **SC-005**: A new `analyze.ipynb` cell runs in **under 30 seconds** end-to-end (excluding video playback) on a typical run with ≤ 500 PPO updates, so participants can iterate quickly.
- **SC-006**: Both `analyze.ipynb` notebooks render correctly against either a custom-PPO run or an SB3 run (CarRacing only, for SB3) — verified by running the notebook against both run types pre-workshop.
- **SC-007**: `Ctrl+C` during training leaves an inspectable run directory in **100% of cases** — `meta.json` and any `metrics.jsonl` records written so far are preserved; no corruption (final line in JSONL is either complete or absent, never a half-written record).
- **SC-008**: All training drivers fit in **≤ 100 lines each** (excluding imports and docstrings), enforcing the thin-driver rule.

## Assumptions

- **JSONL line atomicity**: assumed adequate via flushing after each `__call__`. We do not need cross-process atomicity; only one writer per file. If the workshop ever runs two trainings into the same file (which is not the design), this assumption breaks — out of scope.
- **Gradient norm direction**: assumed to be the **post-clip** norm (i.e., the norm after `clip_grad_norm_`). If post-clip is awkward to obtain from the existing `ppo.train()` structure, **pre-clip is acceptable** and MUST be documented in `meta.json` under a `metric_definitions` field.
- **SB3 model file extension**: SB3's native serialization is `.zip`. We accept `model.zip` (or `model.pt` for the custom path) — `analyze.ipynb` does not need to load the model, only the run-dir metadata + JSONL + video, so the extension is informational only.
- **Eval video recorder**: `gymnasium.wrappers.RecordVideo` (which depends on `moviepy`/`imageio-ffmpeg`) is the default. If `imageio-ffmpeg` is not in `workshop1` deps, this spec assumes adding it. If adding it is undesirable, an `imageio.mimsave` fallback writing GIF is acceptable.
- **Run discovery in notebook**: "latest" is determined by directory mtime, not by name. This naturally handles the `sample/` directory (which won't be the latest unless intentionally `touch`ed).
- **CarRacing stage-3 SB3 path takes precedence in the README** for participants who didn't finish all stage 1 TODOs: the README already implies SB3 is the documented path. The custom-PPO CNN path is documented as the "primary" path for participants who finished TODOs comfortably and want to see their own PPO eat pixels.
- **No live plots in v1**: per `decisions.md`. Console + post-hoc notebook is sufficient.
- **No TensorBoard mirror in v1**: rejected as primary; not added as a secondary writer to keep the v1 surface small. Can be added later behind a `--tensorboard` flag.
- **No multi-episode eval in v1**: per user decision. One greedy episode, recorded.
- **`mountaincar.ipynb` is empty and disposable**: confirmed by file inspection (0 bytes, no commits with content). Replacing it is safe.
- **Custom PPO on CarRacing is in scope**: per user decision, with the named enabling work (CNN agent, env wrappers, `RolloutBuffer` shape generalization).
- **Stage 1 reference solutions stay on `solutions` branch**: this feature is being developed on `002-training-and-visualization` (branched off `solutions`) and intended to merge back into `solutions`. Per-TODO tags `ws1-todoN-done` are unaffected.
- **Brainstorm and decision log are co-located** at `specs/002-training-and-visualization/{brainstorm.md, decisions.md}` (moved from the original `specs/training-and-visualization/` to align with the speckit feature directory).
