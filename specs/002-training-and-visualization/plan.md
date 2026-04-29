# Implementation Plan: Stage Training Drivers, Structured Logging, and Analysis Notebooks

**Branch**: `002-training-and-visualization` | **Date**: 2026-04-29 | **Spec**: [spec.md](./spec.md)
**Input**: Feature specification from `/specs/002-training-and-visualization/spec.md`

## Summary

Add per-stage training drivers and analysis notebooks to `workshop-1/2-mountaincar/` and `workshop-1/3-car-racing/`. Drivers are thin scripts that call `PPOAgent.train(env, total_timesteps, metrics_fn=runlog)` and write a self-contained `runs/<stage>/<timestamp>/{meta.json, metrics.jsonl, model.*, eval.mp4}` directory. Notebooks load any run directory and render training curves, an inline `eval.mp4`, and (for MountainCar) a 2-D state-space value/policy heatmap. The JSONL log is the load-bearing format choice — workshop participants paste it into Claude / Copilot to debug.

The non-trivial enabling work happens inside `workshop-1/1-ppo/ppo.py`:

1. `ppo.train()` gains two additive kwargs: `metrics_fn` (callback per PPO update) and `agent` (when provided, `train()` invokes `agent.preprocess(obs)` and `agent.reset_preprocess_state()` directly in the rollout loop — Article II compliance, no wrapper).
2. `PPOAgent` gains a `reset_preprocess_state()` no-op hook that subclasses (`CarRacingPPOAgent`) override to clear stateful preprocess buffers.
3. `RolloutBuffer` generalizes from flat `(size, obs_dim)` to `(size, *obs_shape)` so it can store post-preprocess shapes like `(4, 84, 84)`.
4. `CarRacingPPOAgent` gains a CNN actor + CNN value network alongside its existing (preserved) `preprocess()` pipeline.

Both stages also ship an SB3 alternative driver (`train_sb3.py`) writing the same JSONL schema, so `analyze.ipynb` is path-agnostic. The CarRacing SB3 path is the one place where standalone Gymnasium observation wrappers are used — a documented, scoped Article II exemption (SB3's training loop is closed and cannot accept `agent.preprocess()` injection).

## Technical Context

**Language/Version**: Python 3.10–3.12 (per Constitution; uv-managed)
**Primary Dependencies**: PyTorch ≥ 2.1, Gymnasium ≥ 0.29, NumPy, OpenCV (cv2), Stable-Baselines3, matplotlib, `imageio[ffmpeg]` (newly required for `gymnasium.wrappers.RecordVideo`)
**Storage**: Single-file artifacts on local disk under repo-root `runs/<stage>/<timestamp>/` (gitignored) or `pretrained/sample-runs/<stage>/<run-name>/` (tracked). No DB. JSON Lines for per-update metrics, JSON for run metadata, `.pt` for custom PPO models, `.zip` for SB3 models, `.mp4` for eval video.
**Testing**: Existing per-step custom runner in `workshop-1/1-ppo/test_ppo.py --step {1..5}`; existing `test_agent_interface.py`. New tests added inline (Article IV).
**Target Platform**: Workshop participant laptops (macOS, Linux, Windows). Headless CI for sample-run regeneration.
**Project Type**: Python CLI tooling + Jupyter notebooks within a teaching repo.
**Performance Goals**: SC-001 (≤ 10 min wall-time MountainCar end-to-end on a laptop CPU), SC-005 (analyze.ipynb cells render in ≤ 30 s), no degradation in `ppo.train()` step rate vs. current implementation.
**Constraints**: Driver scripts ≤ 100 lines each (SC-008); JSONL writes best-effort (FR-004); `Ctrl+C` leaves an inspectable partial run dir (SC-007); existing stage 1 tests keep passing unchanged (FR-018, SC-003).
**Scale/Scope**: Per-run JSONL files cap at low hundreds of records (≤ 500 PPO updates typical). Workshop cohort 15–30 participants × 1–3 runs each = ~50–100 runs/workshop, all local.

## Constitution Check

*GATE: Must pass before Phase 0 research. Re-check after Phase 1 design.*

| Article | Check | Result |
|---------|-------|--------|
| I. Participant-First Design | English text in all new code/docs; visible artifacts (plots, mp4); actionable error messages on `--run-name` collision and missing-run cases | ✅ PASS |
| II. Two Paths, One Agent API (NON-NEGOTIABLE) | Custom PPO: `agent.preprocess()` called directly inline in `ppo.train()` per FR-001b; SoT preserved. `Agent.predict()` / `Agent.save()` / `Agent.load()` unchanged. SB3 CarRacing: scoped exemption (see Complexity Tracking) | ⚠️ PASS WITH DOCUMENTED EXEMPTION |
| III. Gymnasium-Conformant Environments | All envs constructed via `gymnasium.make`; 5-tuple `step` API; observation transforms live in `Agent.preprocess()` for custom path; SB3 wrappers covered by Article II exemption | ✅ PASS |
| IV. Test-Verified Implementation (NON-NEGOTIABLE) | Existing `test_ppo.py --step {1..5}` continues to pass (FR-018, SC-003); new tests added for `RunLogger` JSONL contract, `RolloutBuffer` shape generalization, `PPOAgent.reset_preprocess_state` lifecycle, `CarRacingPPOAgent` CNN forward shapes, SB3 callback record schema; all under 10 s | ✅ PASS |
| V. Progressive Scaffolding | New helper code (`_runlog.py`, `_sb3_jsonl_callback.py`, drivers, notebooks) is provided complete; CNN classes in `CarRacingPPOAgent` are pre-filled (CNN architecture is not the workshop's pedagogical content); no new TODO blocks introduced | ✅ PASS |
| VI. Fail-Safe Workshop Design | `train_sb3.py` per stage is the documented escape hatch for participants who can't complete TODOs; `pretrained/sample-runs/` covers the "training too slow" failure mode for `analyze.ipynb` demos | ✅ PASS |
| VII. Sim-to-Real Pipeline Integrity | `PPOAgent.save/load` semantics unchanged; preprocess-state round-trip already correct in current code; `reset_preprocess_state` is for fresh episodes (not persisted across save/load); deploy chain unaffected | ✅ PASS |

**Gate**: PASS. Proceeding to Phase 0.

## Project Structure

### Documentation (this feature)

```text
specs/002-training-and-visualization/
├── plan.md              # This file
├── research.md          # Phase 0 output
├── data-model.md        # Phase 1 output
├── quickstart.md        # Phase 1 output
├── contracts/
│   ├── run-format.md    # meta.json + metrics.jsonl schema
│   └── cli.md           # train.py + train_sb3.py CLI surface
├── brainstorm.md        # pre-spec brainstorm (historical)
├── decisions.md         # append-only decision log
├── checklists/
│   └── requirements.md  # spec-quality checklist (passed)
└── tasks.md             # Phase 2 output (NOT created by /speckit.plan)
```

### Source Code (repository root)

```text
workshop-1/
├── README.md            # MODIFIED: document train.py / analyze.ipynb / SB3 alternatives
├── _runlog.py           # NEW: shared RunLogger writing meta.json + metrics.jsonl
├── _sb3_jsonl_callback.py  # NEW: shared SB3 BaseCallback emitting canonical-schema records
├── 1-ppo/
│   ├── ppo.py           # MODIFIED: metrics_fn + agent kwargs in train(), PPOAgent.reset_preprocess_state(), RolloutBuffer shape generalization
│   ├── test_ppo.py      # PRESERVED behavior; possibly extended with shape-generalization assertion
│   ├── test_agent_interface.py  # PRESERVED behavior
│   └── __init__.py      # PRESERVED
├── 2-mountaincar/
│   ├── agent.py         # PRESERVED: MountainCarPPOAgent (identity preprocess)
│   ├── train.py         # NEW: thin custom-PPO driver
│   ├── train_sb3.py     # NEW: SB3 MlpPolicy alternative
│   └── analyze.ipynb    # NEW (replaces empty mountaincar.ipynb): plots + heatmap + video
└── 3-car-racing/
    ├── agent.py         # MODIFIED: ActorCNN + ValueCNN added; preprocess() preserved; reset_preprocess_state override
    ├── train.py         # NEW: thin custom-PPO driver (no env wrappers)
    ├── train_sb3.py     # NEW: SB3 CnnPolicy + Gymnasium preprocess wrappers (Article II exemption)
    └── analyze.ipynb    # NEW: plots + video (no heatmap)

runs/                    # NEW directory; gitignored; populated by drivers
pretrained/sample-runs/  # NEW directory; tracked; lecturer fixtures
.gitignore               # MODIFIED: add `runs/`
pyproject.toml           # MODIFIED: add `imageio[ffmpeg]` to workshop1 group
```

**Structure Decision**: This is an in-place feature on an existing teaching repo, not a new project. The structure follows the established `workshop-1/<stage>/` pattern. Two new shared modules (`_runlog.py`, `_sb3_jsonl_callback.py`) live at `workshop-1/` root with leading-underscore naming to signal "internal helper, not a participant TODO file." Run output (`runs/`) sits at repo root for cross-stage consistency and global gitignore.

## Complexity Tracking

> Filled because Constitution Check has one documented scoped exemption.

| Violation | Why Needed | Simpler Alternative Rejected Because |
|-----------|------------|-------------------------------------|
| Standalone Gymnasium observation wrappers (`GrayScaleObservation` → `ResizeObservation(84, 84)` → `FrameStackObservation(4)`) used inside `workshop-1/3-car-racing/train_sb3.py` (Article II prohibits standalone wrappers for transforms) | SB3's `PPO("CnnPolicy", env)` consumes the env directly via its own internal rollout loop, which the workshop's custom code cannot reach. The only way to deliver post-preprocess `(4, 84, 84)` observations to SB3 is via Gymnasium wrappers on the env. The exemption is constrained to *one file* (`3-car-racing/train_sb3.py`) and applies only to the SB3-alternative path. The constitution-compliant alternative for custom PPO (FR-001b: `agent.preprocess()` called inline) is implemented in parallel, and `MountainCarPPOAgent.preprocess()` is identity so no exemption is needed there. | (a) Forking / monkey-patching SB3 to inject `agent.preprocess()` calls into its rollout loop — far higher complexity and maintenance cost than one documented file. (b) Dropping SB3 from CarRacing — leaves participants who can't write a CNN actor with no fallback path, violating Article VI (Fail-Safe Workshop Design). (c) Implementing the preprocessing inside an `AgentPreprocessWrapper(gym.ObservationWrapper)` that delegates to `agent.preprocess()` — the constitution explicitly permits this pattern, but the user rejected it in favor of pure inline calls during clarification. |

This exemption is recorded in:
- spec.md FR-013 (the SB3 driver requirement)
- spec.md Clarifications Q6 (the design decision)
- decisions.md Session 2026-04-29 (audit trail — to be appended in tasks phase if not already)

No other Constitution violations.

## Phase 0 Output

See [research.md](./research.md). Resolves:

- `imageio-ffmpeg` and `gymnasium.wrappers.RecordVideo` behavior on macOS/Linux/headless
- SB3 `BaseCallback` hook point for per-rollout metric extraction (mapping `logger.name_to_value` keys → canonical schema)
- Default PPO hyperparameters for the CNN-equipped CarRacing custom path
- `RolloutBuffer` shape-generalization PyTorch idioms
- Notebook output stripping policy (`nbstripout` vs manual)

## Phase 1 Output

- [data-model.md](./data-model.md) — entities: `RunDirectory`, `MetaJson`, `MetricRecord`, `RunLogger`, `PPOAgent` extension hook, generalized `RolloutBuffer`, `CarRacingPPOAgent` CNN classes, sample-run layout
- [contracts/run-format.md](./contracts/run-format.md) — JSON schema for `meta.json` and per-line schema for `metrics.jsonl`
- [contracts/cli.md](./contracts/cli.md) — `train.py` and `train_sb3.py` argparse surface, exit codes, error-message contracts
- [quickstart.md](./quickstart.md) — workshop-day step-by-step
- Agent context update: `.specify/scripts/bash/update-agent-context.sh claude` will be run at end of Phase 1.

**Post-design Constitution re-check**: PASS (no new violations introduced; the SB3 exemption is the only one and is scoped to a single file).

**Stop**: `/speckit.plan` ends here. Tasks generation is `/speckit.tasks`.
