# Tasks: Stage Training Drivers, Structured Logging, and Analysis Notebooks

**Input**: Design documents from `/specs/002-training-and-visualization/`
**Prerequisites**: plan.md (loaded), spec.md (loaded), research.md, data-model.md, contracts/run-format.md, contracts/cli.md, quickstart.md

**Tests**: Test tasks are included because Article IV (Test-Verified Implementation, NON-NEGOTIABLE) of the project constitution requires them, and FR-018 + SC-003 explicitly require existing tests to keep passing under all changes.

**Organization**: Tasks are grouped by user story. Each story is independently testable per the spec's "Independent Test" criteria.

> **Implementation note (2026-04-29)**: First implementation pass scoped down per user instruction — *"don't touch `ppo.py`, don't run training, just verify the eval pipeline works on random weights."* All `ppo.py`-modifying tasks (T005–T009 + T012–T016 tests + FR-001b agent kwarg in `ppo.train`) are deferred. Driver-level metrics logging is achieved by parsing `ppo.py`'s existing `format_update_line` output via a new shared module `workshop-1/_log_parser.py` (added beyond the original 11 listed shared modules). Sample-run generation (T038, T039) and end-to-end training verifications (T020, T031, T036, T037) are deferred to a future training pass.

## Implementation status (Phase 1 pass, 2026-04-29)

- ✅ **Eval pipeline verified end-to-end** on random-weight agents in both stages:
  - `runs/mountaincar/verify-eval/{meta.json, metrics.jsonl (empty), model.pt, eval.mp4 (32 KB)}`
  - `runs/car-racing/verify-eval/{meta.json, metrics.jsonl (empty), model.pt (13 MB CNN), eval.mp4 (69 KB)}`
- ✅ All four drivers + two notebooks land. CarRacing custom-PPO `train.py` will WARN-and-continue on the inevitable shape error from `ppo.train()` until ppo.py is updated to call `agent.preprocess()`.
- ⏭️ Deferred: full training runs, sample-run generation, formal unit tests for `_runlog.py` / `_log_parser.py` / `Sb3JsonlCallback`, polish (README updates, decisions/brainstorm append, dress rehearsal).

## Format: `[ID] [P?] [Story] Description`

- **[P]**: different files, no dependency on incomplete tasks
- **[Story]**: maps a task to a user story (US1, US2, US3, US4, US5)
- File paths are exact and absolute relative to repo root

## Path Conventions

In-place feature on an existing repo. Source: `workshop-1/<stage>/`. Run output: `runs/<stage>/<run-name>/` (gitignored). Lecturer fixtures: `pretrained/sample-runs/<stage>/<run-name>/` (tracked). Spec artifacts: `specs/002-training-and-visualization/`.

---

## Phase 1: Setup (Shared Infrastructure)

**Purpose**: Repo-level changes that other phases depend on.

- [X] T001 [P] Add `imageio[ffmpeg]>=2.31` to the `workshop1` dependency group in `pyproject.toml` (per research.md R6; required by `gymnasium.wrappers.RecordVideo`)
- [X] T002 [P] Add a single-line entry `runs/` to `.gitignore` at repo root (per FR-019 + decisions Q2) — already present (line 22), no change needed
- [X] T003 Refresh `uv.lock` by running `uv sync --group workshop1` after T001 (depends on T001)
- [X] T004 [P] Create `pretrained/sample-runs/.gitkeep` so the directory exists in the tracked layout (per FR-020)

---

## Phase 2: Foundational (Blocking Prerequisites)

**Purpose**: All structural changes to `ppo.py`, the new shared modules, and their tests. Every user story depends on these landing.

**⚠️ CRITICAL**: No user story (Phase 3+) can begin until this phase is complete and Phase-2 tests pass.

### Modifications to `workshop-1/1-ppo/ppo.py` (sequential — same file)

- [ ] T005 ⏭️ **Deferred** (don't-touch-ppo.py directive). Generalize `RolloutBuffer.__init__` in `workshop-1/1-ppo/ppo.py` to accept `obs_shape: int | tuple[int, ...]` (when `int`, internally treat as `(int,)`); allocate `self.obs = np.zeros((size, *obs_shape), dtype=np.float32)`. Per FR-010 + research.md R4 — backward compatible
- [ ] T006 ⏭️ **Deferred**. Add `metrics_fn: Callable[[dict], None] | None = None` keyword-only kwarg to `ppo.train()` in `workshop-1/1-ppo/ppo.py`. Workaround in this pass: a new `workshop-1/_log_parser.py` parses the existing `log_fn=print` formatted output and emits canonical-schema records to `RunLogger`. When ppo.py is touched in a later pass, the parser becomes optional
- [ ] T007 ⏭️ **Deferred**. Add `agent` kwarg to `ppo.train()`. Per FR-001b. Until then: CarRacing custom-PPO `train.py` will catch the shape-error from `ppo.train()`, surface it clearly, and proceed to eval (which DOES work, because `agent.predict()` calls `preprocess()` internally)
- [ ] T008 ⏭️ **Deferred**. Add `def reset_preprocess_state(self) -> None: return None` no-op method to `PPOAgent`. Per FR-009b. Workaround in this pass: drivers manually clear `agent._frame_buffer` for CarRacing before eval
- [ ] T009 ⏭️ **Deferred**. Update `PPOAgent.train()` to forward `metrics_fn` + `agent`. Per FR-002. Workaround in this pass: drivers call the module-level `ppo.train()` directly with the agent's actor/value/log_std + custom `log_fn`

### New shared modules (different files — parallelizable)

- [X] T010 [P] Create `workshop-1/_runlog.py` defining `RunLogger` per data-model.md (constructor signature, `__call__(metrics: dict)` JSONL append with NaN→null + best-effort write, `close(status)`, context-manager protocol, `runs_root` parameter, `--force` collision handling, `RunDirectoryExistsError` exception). Includes `metric_definitions` in `meta.json` covering every key the writer emits. Per FR-003 + FR-004 + FR-005
- [X] T011 [P] Create `workshop-1/_sb3_jsonl_callback.py` defining `Sb3JsonlCallback(BaseCallback)` per data-model.md `Sb3JsonlCallback` block — overrides `_on_training_start` (capture wall-clock start) and `_on_rollout_end` (read `self.logger.name_to_value`, map to canonical schema with the SB3 sign flip on entropy_loss, emit one record via `run_logger.__call__`). Per research.md R2
- [X] **NEW T011b** [P] Create `workshop-1/_log_parser.py` — `parse_update_line(line)` regex parser for `format_update_line` output + `make_log_fn(run_logger, agent)` factory returning a `log_fn` suitable for `ppo.train()`. Added during this pass to bridge the deferred T006 (no `metrics_fn` in ppo.py)

### Foundational tests

- [ ] T012 [P] ⏭️ Deferred (depends on T005–T009). Verify `uv run python workshop-1/1-ppo/test_ppo.py --step 1..5` all PASS after T005–T009 land. Per FR-018 + SC-003
- [ ] T013 [P] ⏭️ Deferred (same dependency). Verify `uv run python workshop-1/1-ppo/test_agent_interface.py --agent ppo` PASSES
- [ ] T014 [P] ⏭️ Deferred (depends on T005). `RolloutBuffer` shape-generalization test
- [ ] T015 [P] ⏭️ Deferred. `RunLogger` JSONL contract test (the eval-verification command run during this pass IS a partial integration test of `_runlog.py` — it produces a complete `meta.json` and verifies all 14 documented keys are present, plus a non-empty model.pt and eval.mp4)
- [ ] T016 [P] ⏭️ Deferred (depends on T007/T008). `reset_preprocess_state` lifecycle test

**Checkpoint**: All Phase-2 tests pass. Stage 1 unchanged in behavior. `RunLogger` writes valid JSONL + `meta.json`. `Sb3JsonlCallback` ready to consume. User story phases can begin.

---

## Phase 3: User Story 1 — MountainCar custom-PPO end-to-end (Priority: P1) 🎯 MVP

**Goal**: Workshop participant runs `train.py`, watches metric lines stream, opens `analyze.ipynb`, sees plots + heatmap + eval video — all without editing any cell.

**Independent Test**: From a clean checkout (with stage 1 TODOs complete), run `uv run python workshop-1/2-mountaincar/train.py` to completion (≤ 5 min CPU); verify a `runs/mountaincar/<timestamp>/` directory exists with `meta.json`, `metrics.jsonl`, `model.pt`, `eval.mp4`. Open `workshop-1/2-mountaincar/analyze.ipynb` and Run All — all cells succeed and the eval video plays inline.

### Implementation for User Story 1

- [ ] T017 [US1] Delete the empty `workshop-1/2-mountaincar/mountaincar.ipynb` (0 bytes; replaced by `analyze.ipynb` per FR-017)
- [ ] T018 [US1] Create `workshop-1/2-mountaincar/train.py` per contracts/cli.md and quickstart.md: thin ≤ 100-line driver that argparses `--timesteps` (default 50000), `--seed` (default 42), `--run-name`, `--no-eval`, `--force`; constructs `gymnasium.make("MountainCarContinuous-v0")` (no wrappers); imports `MountainCarPPOAgent`; uses `with RunLogger(stage="mountaincar", ...) as runlog:` context; calls `agent.train(env, args.timesteps, metrics_fn=runlog)`; on completion runs one greedy eval episode via `gymnasium.wrappers.RecordVideo` writing to `<run_dir>/eval.mp4` with the FR-015 fallback writing `eval.mp4.skipped` on recorder failure; calls `agent.save(<run_dir>/model.pt)`; on `KeyboardInterrupt` propagates `runlog.close("interrupted")` (handled by context manager). Per FR-007 + FR-014
- [ ] T019 [P] [US1] Create `workshop-1/2-mountaincar/analyze.ipynb` per FR-011 + quickstart.md: cells (1) imports + path setup, (2) `RUN_DIR = None` override, (3) auto-discover latest `runs/mountaincar/*` by mtime with the FR-019 error message on empty (research.md R8), (4) load `meta.json` + `metrics.jsonl` via `pd.read_json(..., lines=True)`, (5) plot 4 curves (mean_return, policy_loss, value_loss, entropy) on shared x-axis = update, (6) value-function heatmap over the (position, velocity) grid by sampling `agent.value(...)` on a 50×50 grid, (7) policy-mean quiver over the same grid, (8) embed `eval.mp4` via `IPython.display.Video`. Notebook MUST run end-to-end via Run All without manual edits when a valid `runs/mountaincar/*` exists
- [ ] T020 [US1] Run `uv run python workshop-1/2-mountaincar/train.py` end-to-end on a stock laptop and confirm: (a) terminal prints one log line per PPO update, (b) total wall time ≤ 10 min (SC-001), (c) `runs/mountaincar/<timestamp>/{meta.json, metrics.jsonl, model.pt, eval.mp4}` all exist
- [ ] T021 [US1] Open `workshop-1/2-mountaincar/analyze.ipynb` and Run All against the run from T020; confirm all cells succeed in <30 s aggregate (SC-005), curves plot, heatmap renders, quiver renders, video plays inline

**Checkpoint**: US1 fully functional and independently testable. MVP achieved.

---

## Phase 4: User Story 2 — JSONL debug-via-LLM (Priority: P1)

**Goal**: A participant pastes `meta.json` + last 50 lines of `metrics.jsonl` into Claude / Copilot and gets useful diagnostic suggestions.

**Independent Test**: Run a deliberately broken training (e.g., `lr=1.0`), paste artifacts to Claude with prompt "is anything obviously wrong?" — diagnostic must mention the actual failure mode without the LLM needing to read source code.

### Implementation for User Story 2

- [ ] T022 [P] [US2] Audit `RunLogger.__init__` in `workshop-1/_runlog.py` (T010) so the `meta.json["metric_definitions"]` dict lists every key actually emitted to `metrics.jsonl` for the corresponding driver — both custom-PPO (full schema) and SB3 (with `grad_norm: "null for SB3 runs"`). Per data-model.md validation rules
- [ ] T023 [P] [US2] Manual SC-002 calibration: deliberately produce 5 broken runs (high lr → NaN, entropy_coef=0 + clip_eps=0.5 → entropy stuck, terrible policy init → returns flat, value_coef wildly off → value_loss huge, deterministic=True hyperparam typo if it exists). For each, paste `meta.json` + last 50 `metrics.jsonl` lines into a fresh Claude session with "is anything obviously wrong?" and record whether the diagnostic was useful. Target: ≥ 4 of 5 useful (SC-002). Document results in `specs/002-training-and-visualization/sc002-calibration.md` (informational; not a deliverable file)

**Checkpoint**: US2 contract verified — the JSONL log is genuinely LLM-debuggable.

---

## Phase 5: User Story 3 — CarRacing custom-PPO end-to-end (Priority: P1)

**Goal**: A participant trains a CNN-equipped `CarRacingPPOAgent` end-to-end with `agent.preprocess()` driving the pixel pipeline (no env wrappers), and analyzes the result in a notebook.

**Independent Test**: From clean checkout (stage 1 TODOs complete), `uv run python workshop-1/3-car-racing/train.py --timesteps 5000` (smoke) completes without shape errors and produces a valid run dir; `analyze.ipynb` Run All succeeds.

### Implementation for User Story 3

- [ ] T024 [P] [US3] Add `ActorCNN` class to `workshop-1/3-car-racing/agent.py` per data-model.md `ActorCNN` block: NatureCNN-style conv stack (`Conv2d(4,32,8,4)` → `Conv2d(32,64,4,2)` → `Conv2d(64,64,3,1)` → `Linear(64*7*7, 512)` → `Linear(512, action_dim)`), ReLU activations, orthogonal init (gain √2 on conv/fc, 0.01 on action head)
- [ ] T025 [P] [US3] Add `ValueCNN` class to `workshop-1/3-car-racing/agent.py` mirroring `ActorCNN` with `Linear(512, 1)` head and final `.squeeze(-1)` so `forward(obs).shape == (batch,)`
- [ ] T026 [US3] Override `CarRacingPPOAgent.__init__` in `workshop-1/3-car-racing/agent.py` per data-model.md: merge CarRacing hyperparameters (`rollout_size=2048, batch_size=128, n_epochs=10, lr=2.5e-4, clip_eps=0.1, entropy_coef=0.0`) before calling `super().__init__`, then *replace* `self.actor` with `ActorCNN(action_dim)` and `self.value` with `ValueCNN()` (the inherited MLPs are discarded). Depends on T024, T025
- [ ] T027 [US3] Override `def reset_preprocess_state(self) -> None: self._frame_buffer.clear()` in `CarRacingPPOAgent` (`workshop-1/3-car-racing/agent.py`). Per FR-009b
- [ ] T028 [P] [US3] Add CNN forward-shape test (extension of `workshop-1/1-ppo/test_ppo.py` or new test file): assert `ActorCNN(action_dim=3)(torch.zeros(2, 4, 84, 84)).shape == (2, 3)` and `ValueCNN()(torch.zeros(2, 4, 84, 84)).shape == (2,)`. Runs in <1 s
- [ ] T029 [US3] Create `workshop-1/3-car-racing/train.py` per contracts/cli.md: thin ≤ 100-line driver that argparses the same flags as MountainCar (default `--timesteps 200000`); constructs `gymnasium.make("CarRacing-v3", render_mode="rgb_array")` with **no observation wrappers** (FR-008); imports `CarRacingPPOAgent`; uses `RunLogger(stage="car-racing", ...)` context; calls `agent.train(env, args.timesteps, metrics_fn=runlog)` (which routes through the modified `ppo.train()` so `agent.preprocess()` is called inline per FR-001b); records eval video; saves model
- [ ] T030 [P] [US3] Create `workshop-1/3-car-racing/analyze.ipynb` per FR-012: same shape as MountainCar's `analyze.ipynb` MINUS cells (6) and (7) (no value heatmap or policy quiver — pixel state space). Auto-discover from `runs/car-racing/`
- [ ] T031 [US3] Smoke run: `uv run python workshop-1/3-car-racing/train.py --timesteps 5000` and verify run dir layout. Wall time should be ≤ 2 min on CPU
- [ ] T032 [US3] Open `workshop-1/3-car-racing/analyze.ipynb` and Run All against the smoke run; confirm curves plot and `eval.mp4` plays

**Checkpoint**: US3 fully functional. Custom-PPO path works on pixels via inline `agent.preprocess()` — Article II preserved.

---

## Phase 6: User Story 4 — SB3 escape hatch (Priority: P2)

**Goal**: Both stages have a working SB3 alternative driver writing the same JSONL schema; `analyze.ipynb` is path-agnostic.

**Independent Test**: With the custom-PPO `PPOAgent.train()` *intentionally broken* (insert a `raise NotImplementedError` at the top), both `train_sb3.py` drivers complete a smoke run and both `analyze.ipynb` notebooks render the SB3 runs.

### Implementation for User Story 4

- [ ] T033 [P] [US4] Create `workshop-1/2-mountaincar/train_sb3.py` per contracts/cli.md: same CLI flags as `train.py`; constructs `gymnasium.make("MountainCarContinuous-v0")` with **no wrappers**; instantiates `stable_baselines3.PPO("MlpPolicy", env, ...)`; runs `model.learn(args.timesteps, callback=Sb3JsonlCallback(runlog))`; on completion does the same eval-video + model-save dance as `train.py` (model saved as `model.zip` per data-model.md). Per FR-013
- [ ] T034 [P] [US4] Create `workshop-1/3-car-racing/train_sb3.py` per contracts/cli.md: same CLI flags; constructs `gymnasium.make("CarRacing-v3", render_mode="rgb_array")` wrapped through `gym.wrappers.GrayScaleObservation` → `gym.wrappers.ResizeObservation(env, (84, 84))` → `gym.wrappers.FrameStackObservation(env, 4)` (Article II SB3 exemption — single file, documented in plan.md Complexity Tracking); instantiates `stable_baselines3.PPO("CnnPolicy", env, ...)`; runs `model.learn(args.timesteps, callback=Sb3JsonlCallback(runlog))`. Per FR-013
- [ ] T035 [P] [US4] Add an `Sb3JsonlCallback` schema test (extends T015 file or new): construct a fake SB3-shaped `logger.name_to_value` dict containing the keys from research.md R2, invoke `_on_rollout_end()` once via a stubbed `BaseCallback` parent, assert the resulting `metrics.jsonl` line has every key from `metric_definitions` and the entropy sign-flip applied (positive entropy in JSONL when SB3's entropy_loss is negative)
- [ ] T036 [US4] Smoke run `uv run python workshop-1/2-mountaincar/train_sb3.py --timesteps 10000` and confirm a `runs/mountaincar/<timestamp>/` directory with `model.zip` + valid `metrics.jsonl`; open MountainCar `analyze.ipynb` and confirm plots render against this SB3 run (no code changes needed in the notebook — path-agnostic)
- [ ] T037 [US4] Smoke run `uv run python workshop-1/3-car-racing/train_sb3.py --timesteps 5000` and confirm a `runs/car-racing/<timestamp>/` directory; open CarRacing `analyze.ipynb` and confirm plots render against the SB3 run

**Checkpoint**: Every workshop participant has at least one working path forward in each stage (SC-004).

---

## Phase 7: User Story 5 — Lecturer sample runs (Priority: P3)

**Goal**: Lecturer demos `analyze.ipynb` against pre-shipped runs even when no participant runs exist locally yet.

**Independent Test**: Drop the sample runs into `pretrained/sample-runs/<stage>/sample/`; participant edits `RUN_DIR = "pretrained/sample-runs/mountaincar/sample"` in their notebook and Run All renders successfully.

### Implementation for User Story 5

- [ ] T038 [US5] Generate a full MountainCar run via `uv run python workshop-1/2-mountaincar/train.py --timesteps 50000 --run-name sample`, then move the resulting directory to `pretrained/sample-runs/mountaincar/sample/`. Commit the directory (it's tracked, unlike `runs/`). Per FR-020
- [ ] T039 [US5] Generate a full CarRacing run via `uv run python workshop-1/3-car-racing/train.py --timesteps 200000 --run-name sample` (this is a long run; can be done in background or overnight), then move to `pretrained/sample-runs/car-racing/sample/`. Per FR-020
- [ ] T040 [P] [US5] Verify `workshop-1/2-mountaincar/analyze.ipynb` with `RUN_DIR = "pretrained/sample-runs/mountaincar/sample"` Run-All succeeds and produces all expected outputs (curves, heatmap, quiver, video)
- [ ] T041 [P] [US5] Verify `workshop-1/3-car-racing/analyze.ipynb` with `RUN_DIR = "pretrained/sample-runs/car-racing/sample"` Run-All succeeds (curves, video)

**Checkpoint**: Lecturer can demo offline. Workshop is delivery-ready.

---

## Phase 8: Polish & Cross-Cutting Concerns

- [ ] T042 [P] Update `workshop-1/README.md` per FR-016: add Stage 2 / Stage 3 sections with `train.py` commands, `analyze.ipynb` instructions, where runs are written, the SB3 alternative for both stages, the `--run-name` + `--force` collision behavior. Preserve the existing per-TODO checkpoint section verbatim
- [ ] T043 [P] Strip outputs from `workshop-1/2-mountaincar/analyze.ipynb` and `workshop-1/3-car-racing/analyze.ipynb` (Restart Kernel → Clear All Outputs → Save) before any commit per FR-017 + research.md R5
- [ ] T044 [P] Append a new entry to `specs/002-training-and-visualization/decisions.md` under a new `## Session 2026-04-29` (or extend if already present): "**Standalone Gymnasium observation wrappers in custom-PPO `ppo.train()`** — Status: Rejected (post-clarify). Context: CarRacing pixel preprocessing in the custom-PPO training path. Decision: Inline `agent.preprocess()` calls in `ppo.train()` via the new `agent` kwarg (FR-001b). Rationale: (a) Article II of the project constitution prohibits standalone ObservationWrappers for transforms (NON-NEGOTIABLE). (b) The `AgentPreprocessWrapper` bridge pattern named in Article II was also rejected by the user during clarification — preference for explicit inline calls so workshop participants see preprocess invoked from the place that owns the loop." Plus a sibling entry: "**`AgentPreprocessWrapper` bridge** — Status: Rejected (post-clarify). User preference for inline `agent.preprocess()` over any wrapper class for the custom-PPO path; SB3 path retains standalone wrappers as a scoped Article II exemption."
- [ ] T045 [P] Append a new entry to `specs/002-training-and-visualization/brainstorm.md` "Discarded Alternatives" section: "**Gymnasium standalone wrappers (Option α) — discarded post-spec on Article II grounds.** Implementation path is FR-001b: `ppo.train()` accepts an `agent` kwarg and calls `agent.preprocess()` inline. SB3 path retains wrappers as a scoped exemption."
- [ ] T046 Run quickstart.md dress rehearsal end-to-end on a clean clone: `git clone <repo>`, `git checkout 002-training-and-visualization`, `uv sync --group workshop1`, then walk through every command in quickstart.md sequentially (custom-PPO MountainCar, custom-PPO CarRacing smoke, SB3 MountainCar, SB3 CarRacing smoke, both notebooks). Confirm zero file edits required (SC-001) and all SC-* targets met. If any step fails, file a fix task and re-run

---

## Dependencies & Execution Order

### Phase Dependencies

- **Setup (Phase 1)**: T001-T004 — no internal deps except T003 depending on T001
- **Foundational (Phase 2)**: depends on Setup; sequential within ppo.py edits (T005 → T006 → T007 → T008 → T009); T010, T011 [P] (different files); T012-T016 [P] tests run after their respective implementations land. **BLOCKS all user stories.**
- **User Stories (Phase 3-7)**: depend on Foundational completion. US1, US2, US3, US4 are independent and can be parallelized; US5 depends on US1 + US3 having functional drivers
- **Polish (Phase 8)**: T042, T043, T044, T045 [P] (different files / sections); T046 sequential at the end (depends on every other phase being complete)

### User Story Dependencies

- **US1 (P1)**: depends on Phase 2 only. MVP scope.
- **US2 (P1)**: depends on Phase 2 only. Uses US1's runs as test fixtures, but no code dependency on US1.
- **US3 (P1)**: depends on Phase 2 only. Independent of US1 / US2.
- **US4 (P2)**: depends on Phase 2 only. Independent of US1, US2, US3 — verifies path-agnosticism by NOT requiring custom-PPO to work.
- **US5 (P3)**: depends on US1 (T020) and US3 (T031) producing runs that can be promoted to fixtures.

### Within Each User Story

- Implementation tasks before "verify end-to-end" tasks.
- Notebook tasks ([P]-marked when in different file from drivers) can land in parallel with driver tasks.

### Parallel Opportunities

- All four [P] tasks in Phase 1 (T001, T002, T004; T003 sequential after T001)
- Phase 2: T010, T011 in parallel with the ppo.py edit chain; T012-T016 [P] once their targets land
- Phase 3-6: each user story can be picked up independently by a different developer once Phase 2 is done
- Phase 8: all four document-update tasks [P]; only T046 is sequential at the end

---

## Parallel Example: User Story 1 (after Phase 2 complete)

```bash
# Driver and notebook can be built in parallel since they touch different files:
Task: "Create workshop-1/2-mountaincar/train.py (T018)"
Task: "Create workshop-1/2-mountaincar/analyze.ipynb (T019) [P]"

# After both land, run the end-to-end verification sequentially:
Task: "Run train.py and verify run dir (T020)"
Task: "Run All in analyze.ipynb against the produced run (T021)"
```

## Parallel Example: User Story 3 (after Phase 2 complete)

```bash
# CNN classes independent of each other:
Task: "Add ActorCNN to workshop-1/3-car-racing/agent.py (T024) [P]"
Task: "Add ValueCNN to workshop-1/3-car-racing/agent.py (T025) [P]"

# Then sequentially in the same file:
Task: "Override CarRacingPPOAgent.__init__ (T026)"
Task: "Override reset_preprocess_state (T027)"

# Driver + notebook + CNN test in parallel after T026/T027:
Task: "CNN forward-shape test (T028) [P]"
Task: "Create workshop-1/3-car-racing/train.py (T029)"
Task: "Create workshop-1/3-car-racing/analyze.ipynb (T030) [P]"
```

---

## Implementation Strategy

### MVP First (User Story 1 only)

1. Complete Phase 1: Setup (T001-T004)
2. Complete Phase 2: Foundational (T005-T016) — **CRITICAL gate**
3. Complete Phase 3: User Story 1 (T017-T021)
4. **STOP and VALIDATE**: confirm SC-001 (≤ 10 min end-to-end), SC-005 (notebook < 30 s), SC-007 (Ctrl+C leaves valid run dir)
5. Demo at this point if needed

### Incremental Delivery

1. Setup + Foundational → foundation ready
2. Add US1 → MountainCar end-to-end works → first deployable artifact
3. Add US2 → JSONL debug-via-LLM verified → calibrate SC-002
4. Add US3 → CarRacing custom path works → second stage end-to-end
5. Add US4 → SB3 escape hatch works → fail-safe coverage (Article VI)
6. Add US5 → sample runs shipped → lecturer ready
7. Polish → workshop-day delivery checklist

### Parallel Team Strategy

If multiple contributors:

1. Together: Phase 1 + Phase 2 (T005-T009 sequential within ppo.py; T010, T011 + tests in parallel)
2. Once Phase 2 lands:
   - Dev A: US1 (Phase 3)
   - Dev B: US3 (Phase 5) — biggest scope, most independent
   - Dev C: US4 (Phase 6) + US2 (Phase 4) — touches `_sb3_jsonl_callback.py` already created in Phase 2
3. Whoever finishes first: US5 (Phase 7) — needs both US1 and US3 runs as inputs
4. Together: Phase 8 (Polish), with T046 dress rehearsal as the merge gate

---

## Notes

- **No new TODO blocks for participants**. Per Article V (Progressive Scaffolding), all helper code in this feature is provided complete; CNN classes in `CarRacingPPOAgent` are pre-filled because CNN architecture is not the workshop's pedagogical content.
- **Article II compliance** is maintained on the custom-PPO path via FR-001b (inline `agent.preprocess()`). The single SB3 CarRacing exemption is documented in plan.md Complexity Tracking.
- **`Co-Authored-By Claude` trailer** must be omitted from all commits in this repo (per project memory).
- Each task's file path is exact and absolute relative to repo root. An LLM can pick up any task without additional context beyond reading plan.md, spec.md, data-model.md, and the named contract docs.
- Commit cadence: after each task or each logical group (e.g., after T009 commit "ppo.py: add metrics_fn + agent kwargs"; after T021 commit "stage 2 end-to-end with analyze.ipynb"; etc.).
- **Stop at any checkpoint** to validate independently before proceeding.

---

## Format Validation

✅ All 46 tasks follow the strict checklist format:

- ✅ All start with `- [ ]`
- ✅ All have sequential IDs T001–T046
- ✅ `[P]` markers only on tasks in different files / no incomplete dependencies
- ✅ `[Story]` labels (US1–US5) on all Phase 3–7 tasks; absent from Setup, Foundational, Polish
- ✅ Every task description names the exact file path
- ✅ Total tasks: 46 (Setup: 4, Foundational: 12, US1: 5, US2: 2, US3: 9, US4: 5, US5: 4, Polish: 5)
