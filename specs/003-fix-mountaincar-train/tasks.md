# Tasks: Fix MountainCar Training Driver After PPO Refactor

**Input**: Design documents from `/Users/mortenblorstad/projects/phd/RL-workshop/specs/003-fix-mountaincar-train/`
**Prerequisites**: plan.md, spec.md, research.md, data-model.md, contracts/, quickstart.md

**Tests**: Test creation IS a user story in this feature (US2). The five `test_ppo.py` steps and seven `test_agent_interface.py` steps are deliverables, not optional.

**Organization**: Tasks are grouped by user story so US1 (training driver works) and US2 (test suite passes) can be worked in parallel after Phase 2 completes.

## Format: `[ID] [P?] [Story] Description`

- **[P]**: Can run in parallel (different files, no dependencies on incomplete tasks)
- **[Story]**: User-story label (US1, US2, US3) for tasks inside a story phase
- All paths are absolute under `/Users/mortenblorstad/projects/phd/RL-workshop/`

## Path Conventions

This is a workshop training-script package, not a `src/tests/` layout. Source lives under `workshop-1/1-ppo/ppo/` and `workshop-1/2-mountaincar/`. Tests live alongside the package they test (`workshop-1/1-ppo/ppo/tests/`).

---

## Phase 1: Setup (Shared Infrastructure)

**Purpose**: Pre-flight checks. No new dependencies (`imageio[ffmpeg]` is already in `[dependency-groups].workshop1`).

- [X] T001 Confirm `uv sync --group workshop1` runs cleanly on the working machine and `python -c "import imageio_ffmpeg, gymnasium.wrappers.RecordVideo"` succeeds. No edits — sanity check only.

---

## Phase 2: Foundational (Blocking Prerequisites)

**Purpose**: Make `ppo` importable as a package, fix the six refactor bugs in `ppo.py`, and implement `PPOAgent.evaluate()`. Both US1 and US2 are blocked until this phase completes.

**⚠️ CRITICAL**: No user-story work begins until T002–T006 are all done.

- [X] T002 [P] Create `/Users/mortenblorstad/projects/phd/RL-workshop/workshop-1/1-ppo/ppo/__init__.py` re-exporting `PPOAgent`, `RolloutBuffer`, `ActorNetwork`, `CriticNetwork`, `register_agent`, and `_AGENT_REGISTRY` from their submodules per `contracts/agent-api.md`.
- [X] T003 [P] Create `/Users/mortenblorstad/projects/phd/RL-workshop/workshop-1/1-ppo/ppo/utils/__init__.py` re-exporting `seed_everything`, `format_update_line`, `get_device` (from `utils.py`), `RunLogger`, `RunDirectoryExistsError` (from `_runlog.py`), and `make_log_fn`, `parse_update_line` (from `_log_parser.py`) per `contracts/agent-api.md`.
- [X] T004 [P] Create `/Users/mortenblorstad/projects/phd/RL-workshop/workshop-1/1-ppo/ppo/tests/__init__.py` as an empty file so `ppo.tests` is a regular Python package.
- [X] T005 Fix the six refactor bugs in `/Users/mortenblorstad/projects/phd/RL-workshop/workshop-1/1-ppo/ppo/ppo.py` per `research.md` R1. Specifically: (1) replace `nn.Parameter(... , device=self.device)` with `nn.Parameter(torch.ones(self.action_dim, device=self.device) * self.hyperparameters["log_std_init"])`; (2) replace `self.value` with `self.critic` in `train()` (two occurrences); (3) update the `sample_action(...)` call inside `train()` to the new signature `self.sample_action(obs_t, deterministic=False)`; (4) change `evaluate_actions` to `def evaluate_actions(self, obs, actions)` (drop `actor`, `log_std` from the public signature; read them from `self.actor` and `self.log_std`) and update the call site in `train()` accordingly; (5) make `predict(obs: np.ndarray, ...)` convert via `torch.as_tensor(obs, dtype=torch.float32, device=self.device)` before calling `sample_action`; (6) change `PPOAgent.load(cls, path)` to `PPOAgent.load(cls, path, env)` and pass `env` into the constructor inside `load()`. Also restore the five `# -- YOUR CODE HERE --` blocks to `raise NotImplementedError("TODO N: ...")` defaults so participants can still discover them (the maintainer-side solution code stays only on the `solutions` branch — confirm this against the existing solutions branch before stripping).
- [X] T006 Implement `PPOAgent.evaluate(self, env, n_episodes=10, record_video=True, video_dir=None) -> list[float]` in `/Users/mortenblorstad/projects/phd/RL-workshop/workshop-1/1-ppo/ppo/ppo.py` per `contracts/agent-api.md` and `research.md` R3. When `record_video=True`: build a fresh `gym.make(env_id, render_mode="rgb_array")`, wrap with `gymnasium.wrappers.RecordVideo(env, video_folder=str(video_dir), name_prefix="eval")`, run `n_episodes` greedy episodes via `predict(obs, deterministic=True)`, then rename the produced `eval-episode-0.mp4` to `eval.mp4`. Catch `ImportError`/`gymnasium.error.DependencyNotInstalled`: write `eval.mp4.skipped`, print one stderr warning, continue without video. Return the list of per-episode returns. The function must derive `env_id` from the input env's `spec.id` so the recorded env is independent of any wrappers the caller may have applied.

**Checkpoint**: Foundation ready. `from ppo import PPOAgent` works; `PPOAgent(env).evaluate(env, n_episodes=1, record_video=False)` returns a list. US1 and US2 can now proceed in parallel.

---

## Phase 3: User Story 1 — Workshop participant trains a MountainCar agent end-to-end (Priority: P1) 🎯 MVP

**Goal**: A workshop participant runs `train.py` with default arguments, watches PPO update lines stream past, and gets a populated `runs/mountaincar/<run-name>/` directory with `meta.json`, `metrics.jsonl`, `model.pt`, and (unless `--no-eval`) `eval.mp4`.

**Independent Test**: `uv run python workshop-1/2-mountaincar/train.py --timesteps 4096 --run-name smoke --force` exits 0, prints ~2 update lines, and produces a populated run directory whose `meta.json` ends with `status: "ok"`.

### Implementation for User Story 1

- [X] T007 [US1] Replace the contents of `/Users/mortenblorstad/projects/phd/RL-workshop/workshop-1/2-mountaincar/train.py` with a clean driver that follows `contracts/cli.md` exactly. Specifically: argparse for `--timesteps`, `--run-name`, `--no-eval`, `--force` (NO `--seed`); module-top `hyperparameters: dict = {...}` with `random_state: 42` (per spec Q2); single `sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "1-ppo"))` (per `research.md` R4); imports `from ppo import PPOAgent` and `from ppo.utils import RunLogger, RunDirectoryExistsError, make_log_fn, seed_everything`; keeps the `NormalizeObs(gym.ObservationWrapper)` class and applies it to the env before constructing `PPOAgent` (per spec Q4); `runs_root = Path(__file__).resolve().parents[2] / "runs"`; opens `RunLogger(stage="mountaincar", hyperparameters=hyperparameters, env_id=ENV_ID, agent_class=type(agent).__name__, seed=hyperparameters["random_state"], total_timesteps=args.timesteps, run_name=args.run_name, force=args.force, runs_root=runs_root)` (catch `RunDirectoryExistsError` → exit 1); inside `with runlog:` builds `log_fn = make_log_fn(runlog, agent)` and calls `agent.train(env, total_timesteps=args.timesteps, random_state=hyperparameters["random_state"], log_fn=log_fn)`; saves via `agent.save(str(runlog.run_dir / "model.pt"))`; if not `args.no_eval` calls `agent.evaluate(env, n_episodes=1, record_video=True, video_dir=runlog.run_dir)`; outer `try/except` maps `KeyboardInterrupt → exit 130`, `NotImplementedError → exit 3` with friendly message, other `Exception → exit 2`; `finally: env.close()`. Delete the broken `# import` comments and the `## Claude implement training loop ...` placeholder comment block.
- [X] T008 [US1] Run the smoke validation: `cd /Users/mortenblorstad/projects/phd/RL-workshop && uv run python workshop-1/2-mountaincar/train.py --timesteps 4096 --run-name smoke --force`. Confirm exit code 0; confirm `runs/mountaincar/smoke/{meta.json,metrics.jsonl,model.pt,eval.mp4}` all exist; confirm `meta.json` ends with `"status": "ok"`; confirm `metrics.jsonl` has at least one line containing all of `update`, `timesteps`, `policy_loss`, `value_loss`, `entropy`, `mean_return`, `log_std_mean`, `grad_norm`, `wall_time_seconds`. Document any deviation as a follow-up task.

**Checkpoint**: At this point, US1 is fully functional and testable independently — a participant who has filled the five PPO TODOs can train MountainCar end-to-end.

---

## Phase 4: User Story 2 — Refactored PPO modules covered by automated tests (Priority: P1)

**Goal**: A maintainer runs the two test files at `workshop-1/1-ppo/ppo/tests/` and gets `5/5` and `7/7` `PASS` summaries. Unfilled TODOs report `NOT_IMPLEMENTED` without cascading into unrelated steps.

**Independent Test**: `uv run python workshop-1/1-ppo/ppo/tests/test_ppo.py` prints `=== Summary: 5 / 5 passed ===`. `uv run python workshop-1/1-ppo/ppo/tests/test_agent_interface.py --agent ppo` prints `=== Summary: 7 / 7 passed ===`. Combined wall clock < 60 s.

### Implementation for User Story 2

- [X] T009 [P] [US2] Create `/Users/mortenblorstad/projects/phd/RL-workshop/workshop-1/1-ppo/ppo/tests/test_ppo.py` following the convention in `contracts/tests.md` and the pre-refactor reference in commit `60321eb`. Five `@step(n, name)` registered functions: (1) GAE — closed-form 4-step reference with and without mid-trajectory `done`, `atol=1e-5`, calls `RolloutBuffer().compute_gae(...)`; (2) sample_action — single-obs and batched shape/dtype, 1000-sample variance > 1e-6 in `[action_min, action_max]`, deterministic-mode variance < 1e-10, constructed against `gym.make("MountainCarContinuous-v0")` then `PPOAgent(env)`; (3) evaluate_actions — `(B,)` shapes, `Normal(actor(obs), log_std.exp())` reference equality `atol=1e-5` for both `log_probs` and `entropy`; (4) ppo_loss — ratio=1 reduction to `-adv.mean()` `atol=1e-6` + scalar shape + `requires_grad=True`, plus two clipped-branch gradient probes (positive-adv with `new_log_probs=1`, negative-adv with `new_log_probs=-1`) that must yield gradient ≈ 0; (5) train smoke — `gym.make("MountainCarContinuous-v0")`, override `agent.hyperparameters` for `rollout_size=256, n_epochs=2, batch_size=64`, call `agent.train(env, total_timesteps=512, random_state=0, log_fn=capture)`, assert returned dict has `{mean_reward, policy_loss, value_loss, entropy, n_updates}`, no NaN, wall-clock `< 10 s`. Include the `_main()` CLI runner with `--step N` plus exit codes 0/1/2 per `contracts/tests.md`. Each step does a LOCAL `from ppo import PPOAgent`/`RolloutBuffer` so unrelated unfilled TODOs don't cascade.
- [X] T010 [P] [US2] Create `/Users/mortenblorstad/projects/phd/RL-workshop/workshop-1/1-ppo/ppo/tests/test_agent_interface.py` following `contracts/tests.md`. Seven steps: (1) C1 — `from ppo import _AGENT_REGISTRY; assert "PPOAgent" in _AGENT_REGISTRY`; (2) C3 — predict shape/dtype/range against `MountainCarContinuous-v0`; (3) C3-det — `predict(raw, deterministic=True)` equality across calls; (4) C4 — `train(total_timesteps=512)` smoke returns dict with documented keys; (5) C5-base — `tempfile.mkstemp(".pt")` + save → `PPOAgent.load(path, env)` → `predict(raw, deterministic=True)` equality with original; (6) C5-subclass — locally-defined `@register_agent class _RoundTrip(PPOAgent)` survives save/`PPOAgent.load(path, env)` and the loaded instance reports `type(loaded).__name__ == "_RoundTrip"`; (7) C7 — `agent.evaluate(env, n_episodes=2, record_video=False)` returns `list[float]` of length 2 with finite values, then `agent.evaluate(env, n_episodes=1, record_video=True, video_dir=tmp_dir)` produces either `eval.mp4` or `eval.mp4.skipped` under a `tempfile.TemporaryDirectory()`. Include the `--agent {ppo,sb3}` CLI; `--agent sb3` exits 2 with the documented message. Do NOT port the pre-refactor C2 (preprocess identity/determinism/override) or C6 (`_get/_set_preprocess_state`) steps — those APIs were removed in the refactor.
- [X] T011 [US2] Run both test files end-to-end and confirm: `uv run python workshop-1/1-ppo/ppo/tests/test_ppo.py` → exit 0 with `5 / 5 passed`; `uv run python workshop-1/1-ppo/ppo/tests/test_agent_interface.py --agent ppo` → exit 0 with `7 / 7 passed`; combined wall clock measured via `time` < 60 s. Also run `uv run python workshop-1/1-ppo/ppo/tests/test_ppo.py --step 3` to confirm the per-step CLI works in isolation. If any step's assertion needs tightening or loosening, update the test file (not the agent) and document the change in a comment.

**Checkpoint**: US1 and US2 are both fully functional. Either can be demoed independently.

---

## Phase 5: User Story 3 — Training metrics analyzed in existing notebook (Priority: P2)

**Goal**: The existing `analyze.ipynb` opens a run produced by the new driver and renders all standard plots without `KeyError`, `JSONDecodeError`, or schema mismatches.

**Independent Test**: Open `workshop-1/2-mountaincar/analyze.ipynb`, point its run-dir cell at `runs/mountaincar/smoke/` (from T008), run-all, confirm no exceptions and that loss/return/`log_std_mean` curves render.

### Implementation for User Story 3

- [X] T012 [US3] Open `/Users/mortenblorstad/projects/phd/RL-workshop/workshop-1/2-mountaincar/analyze.ipynb`, set its run-dir to the smoke run produced in T008, and run all cells. Verify: no exceptions; `meta.json` parses; `metrics.jsonl` has the expected columns; loss/return curves render. If any cell fails because the schema diverged or a column is missing, update the notebook cell (not the JSONL schema — the schema is frozen by feature 002's `run-format.md` contract). Save the notebook with cleared outputs (so the diff stays small).

**Checkpoint**: All three user stories deliver their independently-verifiable outcomes.

---

## Phase 6: Polish & Cross-Cutting Concerns

**Purpose**: Surface follow-ups, validate the full quickstart, and verify SC-005 (no dead-import references survive).

- [X] T013 [P] Grep audit: from `/Users/mortenblorstad/projects/phd/RL-workshop/`, run `grep -rn "from _runlog\|from _eval\|from _log_parser\|from _sb3_jsonl_callback" workshop-1/ specs/003-fix-mountaincar-train/ 2>/dev/null` and confirm zero hits in the custom-PPO path. Hits in `train_sb3.py` are expected (R6 follow-up — surfaced, not fixed here).
- [X] T014 [P] Grep audit: from `/Users/mortenblorstad/projects/phd/RL-workshop/`, run `grep -n "self.value" workshop-1/1-ppo/ppo/ppo.py` and confirm zero hits (the bug-fix migration to `self.critic` is complete).
- [X] T015 Run the full quickstart from `/Users/mortenblorstad/projects/phd/RL-workshop/specs/003-fix-mountaincar-train/quickstart.md` end-to-end on a clean shell. Each numbered section's commands must produce the documented outputs. Note any deviation; do not silently accept partial success.
- [ ] T016 Optional SC-004 sanity (single-seed, not the full 5-seed sweep): edit `hyperparameters["random_state"]` in `train.py` to 7, run `uv run python workshop-1/2-mountaincar/train.py --timesteps 200000 --run-name sc004-seed7 --force`, confirm `mean_return` in the last `metrics.jsonl` line is `≥ 90`. This validates one of the 3-of-5 seeds required by SC-004; full sweep is out of scope for this feature unless trivially scriptable.
- [ ] T017 Document the constitution-amendment follow-up: append a single bullet to `/Users/mortenblorstad/projects/phd/RL-workshop/specs/003-fix-mountaincar-train/research.md` Section R7 confirming the recommendation has been read by a maintainer (no code change — just a checkbox / signature line).

---

## Dependencies & Execution Order

### Phase Dependencies

- **Phase 1 (Setup)**: No dependencies. T001 is a sanity check.
- **Phase 2 (Foundational)**: Depends on Phase 1. **BLOCKS** all user-story phases.
  - Within Phase 2: T002, T003, T004 are mutually parallel ([P]). T005 and T006 both touch `ppo.py` and must run sequentially after T002–T004 (T002 makes `from ppo import ...` work for both subsequent fix verification and the test files later).
- **Phase 3 (US1)** and **Phase 4 (US2)**: Both depend on Phase 2 completion. **Independent of each other** — can run in parallel.
- **Phase 5 (US3)**: Depends on Phase 3 (needs a run dir from T008) and on T010's existence is irrelevant.
- **Phase 6 (Polish)**: Depends on Phase 3 + Phase 4 + Phase 5 completion.

### Within Each User Story

- US1: T007 → T008 (T008 verifies T007).
- US2: T009 ∥ T010 (different files, parallel) → T011 (verifies both).
- US3: T012 (single task).

### Parallel Opportunities

- Phase 2: T002 ∥ T003 ∥ T004.
- After Phase 2: Phase 3 (US1) ∥ Phase 4 (US2) — different developers can pick one each.
- Phase 4: T009 ∥ T010 — both test files written in parallel.
- Phase 6: T013 ∥ T014 — independent grep audits.

---

## Parallel Example: Phase 2 init-files burst

```bash
# Three __init__.py files — different paths, no shared state — can be edited together:
Task: "Create workshop-1/1-ppo/ppo/__init__.py with re-exports per contracts/agent-api.md"
Task: "Create workshop-1/1-ppo/ppo/utils/__init__.py with re-exports per contracts/agent-api.md"
Task: "Create workshop-1/1-ppo/ppo/tests/__init__.py (empty)"
```

## Parallel Example: Phase 4 test-files burst

```bash
# Two test files — different paths — can be written in parallel:
Task: "Write workshop-1/1-ppo/ppo/tests/test_ppo.py per contracts/tests.md (5 @step functions + CLI runner)"
Task: "Write workshop-1/1-ppo/ppo/tests/test_agent_interface.py per contracts/tests.md (7 @step functions + --agent CLI)"
```

---

## Implementation Strategy

### MVP (US1 only)

1. T001 (Setup).
2. T002–T006 (Foundational — package imports + agent fixes + `evaluate()`).
3. T007–T008 (US1 driver + smoke run).
4. **Stop and validate**: train 4096 steps, confirm `runs/mountaincar/smoke/` is populated.
5. Optional: ship as branch checkpoint.

### Recommended path (US1 + US2 — both P1)

1. T001 (Setup).
2. T002–T006 (Foundational).
3. T007 ∥ T009 ∥ T010 (driver + both test files in parallel).
4. T008 + T011 (smoke + test verification).
5. T012 (US3 — notebook compat).
6. Phase 6 polish.

### Solo developer (sequential)

T001 → T002 → T003 → T004 → T005 → T006 → T007 → T008 → T009 → T010 → T011 → T012 → T013 → T014 → T015 → T016 → T017.

---

## Notes

- [P] = different files, no dependencies on incomplete tasks.
- [Story] tag (US1, US2, US3) only appears on tasks inside a user-story phase.
- All paths are absolute under `/Users/mortenblorstad/projects/phd/RL-workshop/`.
- Constitution violations (Article II / IV / VII) are tracked in `plan.md` Complexity Tracking and `research.md` R7 — no task in this list addresses them; that is a follow-up amendment outside this feature.
- The SB3 driver fix (`research.md` R6) is **not** in this task list. It is a separate follow-up spec.
- T005's "restore `# -- YOUR CODE HERE --` blocks to `raise NotImplementedError`" step assumes the participant-facing `main` branch should ship with TODOs. If you are working directly on the `solutions` branch, skip that sub-step and keep the maintainer solution code in place.
