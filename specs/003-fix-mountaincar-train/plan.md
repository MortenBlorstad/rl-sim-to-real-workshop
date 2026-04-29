# Implementation Plan: Fix MountainCar Training Driver After PPO Refactor

**Branch**: `003-fix-mountaincar-train` | **Date**: 2026-04-29 | **Spec**: [spec.md](./spec.md)
**Input**: Feature specification from `specs/003-fix-mountaincar-train/spec.md`

## Summary

Repair the stage-2 MountainCar custom-PPO training pipeline after the in-progress refactor of `workshop-1/1-ppo/` from a flat-file `ppo.py` to a package layout under `workshop-1/1-ppo/ppo/`. Three concrete deliverables:

1. **Driver**: rewrite `workshop-1/2-mountaincar/train.py` to use the new package, keep the `NormalizeObs` env wrapper (per Q4), wire `RunLogger` + `make_log_fn`, call `agent.train()` then `agent.evaluate()` (per Q1), save `model.pt`, and exit cleanly. No `--seed` flag (per Q2). No references to deleted top-level helpers.
2. **Agent fixes**: repair the latent bugs in the refactored `PPOAgent.train()` (wrong attribute names, wrong call signatures, missing `device` handling, broken `load()`), implement `PPOAgent.evaluate()` (currently `NotImplementedError`), and add the package `__init__.py` files needed to import `ppo` cleanly.
3. **Tests**: port the pre-refactor `test_ppo.py` and `test_agent_interface.py` runners (commit `60321eb`) into `workshop-1/1-ppo/ppo/tests/` adapted to the refactored API — keep the `@step` registry pattern (no pytest), drop the C2/C6 preprocess tests, add a C7 `evaluate()` test.

The plan documents three Article II / IV violations introduced by the spec's clarifications (preprocess removed, driver-level wrapper, dropped C2/C6 tests). Article II is NON-NEGOTIABLE in the constitution; the violations are tracked in Complexity Tracking with a recommendation to open a constitution amendment as a follow-up.

## Technical Context

**Language/Version**: Python 3.10–3.12 (per Constitution; `pyproject.toml` pins `>=3.10,<3.13`)
**Primary Dependencies**: PyTorch ≥ 2.1, Gymnasium ≥ 0.29, NumPy, `imageio[ffmpeg] >= 2.31` (for `RecordVideo`). No new dependencies.
**Storage**: Local disk under `<repo-root>/runs/mountaincar/<run-name>/` (gitignored). Schema fixed by feature 002's `run-format.md` contract — unchanged.
**Testing**: Custom step-runner (per spec Q3, matching commit `60321eb` convention). No pytest. Files: `workshop-1/1-ppo/ppo/tests/test_ppo.py`, `workshop-1/1-ppo/ppo/tests/test_agent_interface.py`. Combined wall-clock budget: < 60 s.
**Target Platform**: Developer laptop (macOS/Linux); device auto-detection via `get_device()` (CUDA → MPS → CPU).
**Project Type**: Workshop training scripts (Python package + CLI driver). No web/mobile.
**Performance Goals**: Train smoke test < 10 s on `MountainCarContinuous-v0` with `total_timesteps=512`. Full training reaches `mean_return ≥ 90` within 200 000 timesteps for ≥ 3 of 5 distinct `random_state` values.
**Constraints**: Tests must not touch the network, must not write into the participant's `runs/` (use `tempfile`/`tmp_path`), must not require a display. RunLogger writes are best-effort; training never blocks on disk errors.
**Scale/Scope**: Single user, single machine. ~5 source files modified, ~3 new (`__init__.py` × 3). Two test files containing ~10 step functions each.

## Constitution Check

*GATE: Must pass before Phase 0 research. Re-checked after Phase 1 design (see end of section).*

Constitution: `.specify/memory/constitution.md` v1.0.0 (ratified 2026-04-13).

| Article | Status | Notes |
|---|---|---|
| I. Participant-First Design | ✅ Pass | Driver runs end-to-end; clear `RunDirectoryExistsError` message; English code/identifiers/docstrings; READMEs unchanged in this feature. |
| II. Two Paths, One Agent API (NON-NEGOTIABLE) | ❌ **VIOLATION** | Spec Q4 keeps `NormalizeObs(gym.ObservationWrapper)` at the driver level; spec Q3 ratifies the refactor's removal of `Agent.preprocess()`. Article II requires `Agent.preprocess()` as the single source of truth and prohibits standalone observation wrappers for transforms. See Complexity Tracking. |
| III. Gymnasium-Conformant Environments | ✅ Pass | `gym.make(...).reset()`/`.step(...)` with the 5-tuple step API throughout. `NormalizeObs` extends `gym.ObservationWrapper` correctly. |
| IV. Test-Verified Implementation (NON-NEGOTIABLE) | ⚠ **PARTIAL VIOLATION** | The `@step` runner pattern, per-step `NotImplementedError` detection, `< 10 s` budget, and shape/dtype/range checks are honored (FR-012/013/014). But Article IV requires `test_agent_interface.py` to cover "preprocess for vector and pixel inputs" + "preprocess determinism" (C2) and the constitution lists `--agent sb3` as required — both are out of scope here because preprocess was removed (C2/C6 dropped per spec Q3). See Complexity Tracking. |
| V. Progressive Scaffolding | ✅ Pass | The five `# -- YOUR CODE HERE --` TODO blocks remain in `ppo.py`; their `raise NotImplementedError(...)` defaults are restored where the refactor accidentally inlined the solutions. Helper code (`RunLogger`, `make_log_fn`, `format_update_line`) ships complete. |
| VI. Fail-Safe Workshop Design | ✅ Pass | The SB3 escape hatch (`train_sb3.py`) is acknowledged. RunLogger writes are best-effort (FR-009 covers ffmpeg-missing graceful-degrade). `KeyboardInterrupt` → `status: "interrupted"`. |
| VII. Sim-to-Real Pipeline Integrity | ⚠ **DEFERRED** | This feature only covers Workshop 1 / stage 2. The implication of removing `preprocess()` for Workshop 2's Pi deployment (`Agent.load()` no longer reconstructs the preprocessing pipeline; the Pi's driving loop will need to apply driver-side wrappers explicitly) is a Workshop-2 concern. Tracked in Complexity Tracking and surfaced as a follow-up. |

**Gate decision**: Proceed with documented violations. Article II's NON-NEGOTIABLE clause is the strongest constraint, but the constitution itself defines the override mechanism (Governance section: *"Participant-experience override. If any principle conflicts with the lived experience of workshop participants, participant experience wins — open an amendment PR updating the principle."*). The user explicitly chose Q4 Option A knowing it diverges from prior guidance. The plan recommends a constitution amendment as a follow-up rather than blocking on this feature.

## Project Structure

### Documentation (this feature)

```text
specs/003-fix-mountaincar-train/
├── plan.md              # This file
├── research.md          # Phase 0 output (decisions on the deferred items + agent fixes)
├── data-model.md        # Phase 1: PPOAgent + RolloutBuffer + RunLogger surfaces
├── contracts/
│   ├── agent-api.md     # Public API surface of the refactored ppo package
│   ├── cli.md           # train.py CLI surface
│   └── tests.md         # Step runners' CLI + step IDs + acceptance assertions
├── quickstart.md        # Phase 1: how to run training + tests on a fresh checkout
└── tasks.md             # Phase 2 output (created by /speckit.tasks, NOT by this command)
```

### Source Code (repository root)

```text
workshop-1/
├── 1-ppo/
│   ├── __init__.py                          # already empty; left as-is
│   └── ppo/
│       ├── __init__.py                      # NEW — re-exports PPOAgent, RolloutBuffer, etc.
│       ├── ppo.py                           # MODIFIED — bug fixes + evaluate() impl
│       ├── networks.py                      # unchanged
│       ├── rollout_buffer.py                # unchanged
│       ├── utils/
│       │   ├── __init__.py                  # NEW — re-exports seed_everything, format_update_line, get_device, RunLogger, RunDirectoryExistsError, make_log_fn, parse_update_line
│       │   ├── utils.py                     # unchanged
│       │   ├── _runlog.py                   # unchanged
│       │   ├── _log_parser.py               # unchanged
│       │   └── _sb3_jsonl_callback.py       # unchanged (out of scope)
│       └── tests/
│           ├── __init__.py                  # NEW — empty (just makes the dir importable)
│           ├── test_ppo.py                  # NEW — five @step tests + CLI runner
│           └── test_agent_interface.py      # NEW — agent-contract @step tests + --agent ppo CLI
└── 2-mountaincar/
    ├── train.py                             # REWRITTEN — driver per FR-001..FR-009
    ├── train_sb3.py                         # NOT TOUCHED — its broken imports (_runlog, _eval, _sb3_jsonl_callback at workshop-1 top level) are a follow-up; surfaced in research.md
    └── analyze.ipynb                        # unchanged
```

**Structure Decision**: Single Python package (`workshop-1/1-ppo/ppo`) consumed by a sibling driver script (`workshop-1/2-mountaincar/train.py`) via a single `sys.path.insert(0, str(_HERE.parent / "1-ppo"))` line — the same pattern `train_sb3.py` already uses. This is intentionally not a "src/tests" layout: the workshop's pedagogy depends on participants seeing the agent code physically next to the training driver. Tests live alongside the package they test (`ppo/tests/`).

## Complexity Tracking

| Violation | Why Needed | Simpler Alternative Rejected Because |
|-----------|------------|--------------------------------------|
| **Article II — `preprocess()` removed; driver-level `NormalizeObs` wrapper** | Spec Q4 (user-explicit): "Keep `NormalizeObs` in `train.py` and apply before constructing PPOAgent." The pre-refactor `preprocess()` method has been deleted from the `PPOAgent` source. Re-adding it would (a) reverse the refactor the user just performed, (b) require participants to implement another method as part of TODO 5, and (c) couple agent-level state to env-level transforms that don't need it for vector envs. | (a) Restoring `preprocess()` and forbidding the wrapper would force MountainCar normalization into the agent for no functional benefit (vector obs, no state). (b) Inline normalization in `PPOAgent.train()` would smuggle logic into the participant's TODO 5 block. (c) Dropping normalization entirely changes the convergence baseline. **Follow-up**: open a constitution amendment to either remove Article II's `preprocess()` requirement or re-scope it as image-only (stage 3 / Workshop 2). |
| **Article IV — `test_agent_interface.py` drops C2 (preprocess identity/determinism) and C6 (`_get/_set_preprocess_state`)** | The methods these tests exercise no longer exist on `PPOAgent` after the refactor. Keeping the tests as-is would make them fail on import (`ImportError: cannot import name 'preprocess' from 'ppo'`). | Restoring `preprocess()` solely to satisfy these tests would re-violate Article II per the previous row. The test file gains a new C7 test for `evaluate()` to keep the contract surface fully covered. |
| **Article IV — no `--agent sb3` test invocation in this feature** | Out of scope per spec Assumptions: "Stable-Baselines3 is not exercised by this feature — the SB3 path lives in `train_sb3.py` and has its own callback. This feature is the custom-PPO path only." | Adding SB3-side tests to this feature would expand it past the user's stated boundary and pull in `train_sb3.py`'s broken imports (separately tracked in research.md as a known follow-up). |
| **Article VII — preprocessing pipeline no longer inside `Agent.save/load`** | Same root cause as the first row. For stage 2 (vector obs), there is no preprocessing pipeline to persist beyond what the env wrapper does at runtime. | The Pi-deployment implication is real but lives in Workshop 2 territory. The spec for Workshop 2 will need a story for replaying the driver's wrapper chain on the Pi (likely: bake the wrapper chain into a `_make_env(...)` helper that both `train.py` and the Pi loader call). Tracked as a Workshop-2 follow-up; does not block this feature. |
