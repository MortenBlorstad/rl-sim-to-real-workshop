# Implementation Plan: PPO Skeleton with Per-TODO Tests

**Branch**: `001-ppo-skeleton` | **Date**: 2026-04-13 | **Spec**: [spec.md](./spec.md)
**Input**: Feature specification from `specs/001-ppo-skeleton/spec.md`

## Summary

Build a single Python file `workshop-1/1-ppo/ppo_skeleton.py` that exposes five top-level TODO functions (`compute_gae`, `sample_action`, `evaluate_actions`, `ppo_loss`, `train`), a fully-provided `ActorNetwork` + `ValueNetwork` + `RolloutBuffer`, and a `PPOAgent` class implementing the Constitution Article II `Agent` interface with an identity `preprocess()` overridable by subclasses. The training entry point sits behind `if __name__ == "__main__":` so module import has no side effects. A custom ~80-line test runner at `workshop-1/1-ppo/test_ppo.py` exposes per-TODO tests via `--step N`, with each test using local imports so unfinished TODOs elsewhere never break a working step. A shared `workshop-1/1-ppo/test_agent_interface.py --agent ppo` verifies the Article II contract for `PPOAgent`. Two stub subclass files (`workshop-1/2-mountaincar/agent.py` and `workshop-1/3-car-racing/agent.py`) demonstrate the override contract; their full implementations belong to separate stage features. Per-TODO recovery uses git tags on the `solutions` branch (`ws1-todo1-done` … `ws1-todo5-done`).

## Technical Context

**Language/Version**: Python ≥ 3.10 (per Constitution).
**Primary Dependencies**: PyTorch (≥ 2.1), Gymnasium (≥ 0.29), NumPy. No Stable-Baselines3 in this spec — that is the SB3Agent follow-up. No `gym-donkeycar` (Workshop 2 only).
**Storage**: Single-file model artifacts on local disk: `*.pt` (PyTorch) saved/loaded by `PPOAgent.save` / `PPOAgent.load`. No database.
**Testing**: Custom ~80-line runner in `workshop-1/1-ppo/test_ppo.py` driven by a `@step(n, name)` decorator, plus `workshop-1/1-ppo/test_agent_interface.py --agent ppo`. No pytest dependency.
**Target Platform**: macOS / Linux / Windows participant laptops, CPU-only baseline. Same code must later load on a Raspberry Pi 4 (Workshop 2), but no Pi-specific code lives in this spec.
**Project Type**: Python teaching package — multi-stage workshop layout under `workshop-1/`.
**Performance Goals**:
- Each `--step N` test completes in < 10 s on a standard laptop CPU (Constitution Article IV).
- Full `test_ppo.py` (no flags) completes in < 60 s (SC-004).
- `uv run python ppo_skeleton.py` completes its training run in roughly 1–3 minutes on CPU and shows a downward loss trend (FR-026, US2 AS2).
**Constraints**:
- All user-facing strings in English (Constitution Article I, FR-050).
- `ppo_skeleton.py` must be importable with zero side effects (FR-005).
- Each TODO must raise `NotImplementedError("TODO N: <description>")` until implemented (FR-002).
- No `pip install` / `requirements.txt` / `conda` — only `uv` and `pyproject.toml` `[dependency-groups]` (Constitution Article VIII).
**Scale/Scope**: 15–30 workshop participants per session, single-process training, vector observation environment (`Box(2,)`), continuous action environment (`Box(1,)`, `[-1, 1]`).

## Constitution Check

*GATE: Must pass before Phase 0 research. Re-check after Phase 1 design.*

Evaluated against `.specify/memory/constitution.md` v1.0.0:

| Article | Compliance | Notes |
|---|---|---|
| I. Participant-First Design | **PASS** | Every print/comment/docstring is English; the test runner emits the prescribed `TODO N OK!` / `FAIL: ...` messages; the script exit prints actionable errors on NaN or non-decreasing loss (FR-027). |
| II. Two Paths, One Agent API | **PASS (Path A only)** | `PPOAgent` implements the full `preprocess/predict/train/save/load` contract. SB3Agent (Path B) is explicitly deferred to a follow-up spec — clarification Q3. The current spec pins the contract in FR-007/008/030/031 so the deferred Path B can plug in without reshaping Path A. |
| III. Gymnasium-Conformant Environments | **PASS** | `MountainCarContinuous-v0` is the default training env. The skeleton uses the modern `obs, info = env.reset(); obs, reward, terminated, truncated, info = env.step(...)` API. No standalone `ObservationWrapper` for transforms — preprocessing lives in `Agent.preprocess()`. |
| IV. Test-Verified Implementation | **PASS** | Every TODO has an independently-runnable test with the prescribed output format and time budget. The shared `test_agent_interface.py --agent ppo` is delivered. Note: full SB3 sibling test (`--agent sb3`) is out of scope (Q3). |
| V. Progressive Scaffolding | **PASS** | Single working file with 5 numbered TODOs; helper code (Actor/Value/RolloutBuffer/env setup/logging) is provided complete. Solutions branch + per-TODO tags are part of the catch-up plan. |
| VI. Fail-Safe Workshop Design | **PASS** | Per-TODO checkpoint tags (`ws1-todo1-done` … `ws1-todo5-done`) on the `solutions` branch, plus the `solutions`-branch full file recovery, satisfy the catch-up matrix for stage 1. Pretrained models are not yet required at the stage-1 milestone. |
| VII. Sim-to-Real Pipeline Integrity | **PASS (interface only)** | `PPOAgent.save` / `.load` round-trip persists model + preprocessing config; `predict(raw_obs)` works on a loaded agent without external setup (FR-031). Workshop 2 will consume this. The `export_model.py` and `deploy.sh` scripts are out of scope for this spec — they belong to Workshop 2. |
| Dependencies & Repo Hygiene | **PASS** | `pyproject.toml` `[dependency-groups]` adds `torch` and `gymnasium` to the existing `workshop1` group. `uv.lock` is committed. `.gitignore` already excludes training outputs. No large files added. |
| Workshop Delivery Workflow | **PASS** | The spec's success criteria require dry-running every README command, all `test_ppo.py` steps passing, and `test_agent_interface.py --agent ppo` passing before merging to `main`. |

**Result**: All gates pass with two explicit deferrals (SB3Agent, the export/deploy scripts), both already documented as out of scope in the spec. **No Complexity Tracking entries needed** — there are no constitution violations.

### Post-design re-check (after Phase 1 artifacts written)

Re-evaluated all nine articles after writing `research.md`, `data-model.md`, `contracts/agent-interface.md`, `contracts/test-runner-cli.md`, `contracts/todo-functions.md`, and `quickstart.md`. **No new violations introduced.** Specific cross-checks:

- **Article II** — the data model and `agent-interface.md` together pin the `_AGENT_REGISTRY` + `_get_preprocess_state` / `_set_preprocess_state` extension hooks needed for FR-031. The single-file `.pt` save format from research R5 satisfies the "single file" requirement.
- **Article IV** — `test-runner-cli.md` documents the local-import isolation pattern that makes per-step independence (FR-011) actually achievable. The CLI contract pins exit codes, output strings, and the three result categories.
- **Article V** — the data-model classes (`ActorNetwork`, `ValueNetwork`, `RolloutBuffer`) are listed as fully-provided helpers; only the five TODO functions are participant-facing gaps.
- **Article VII** — the `_get_preprocess_state` extension hook means a Workshop 2 pixel agent can serialize its frame-buffer config without touching the base class. Pi-side loading still uses `PPOAgent.load(path)` unchanged.
- **Workshop Delivery Workflow** — `quickstart.md` doubles as the dress-rehearsal script for the workshop leader.

## Project Structure

### Documentation (this feature)

```text
specs/001-ppo-skeleton/
├── plan.md              # This file
├── spec.md              # Feature specification (already written)
├── brainstorm.md        # Brainstorm notes (carried over)
├── decisions.md         # Decision log (carried over)
├── research.md          # Phase 0 output (this command)
├── data-model.md        # Phase 1 output (this command)
├── quickstart.md        # Phase 1 output (this command)
├── contracts/           # Phase 1 output (this command)
│   ├── agent-interface.md   # PPOAgent's Article II contract
│   ├── test-runner-cli.md   # test_ppo.py CLI contract
│   └── todo-functions.md    # Signatures of the 5 TODO functions
├── checklists/
│   └── requirements.md  # Spec quality checklist (already passing)
└── tasks.md             # Phase 2 output (/speckit.tasks command, NOT created here)
```

### Source Code (repository root)

```text
workshop-1/
├── 1-ppo/                          # ← THIS SPEC delivers everything in this dir
│   ├── ppo_skeleton.py             # The single-file skeleton (TODOs + helpers + PPOAgent)
│   ├── test_ppo.py                 # Custom @step runner with --step N
│   └── test_agent_interface.py     # Article II contract test (--agent ppo)
│
├── 2-mountaincar/                  # ← THIS SPEC delivers ONLY the subclass stub
│   └── agent.py                    # MountainCarPPOAgent(PPOAgent): vector preprocess (identity / normalization stub)
│
└── 3-car-racing/                   # ← THIS SPEC delivers ONLY the subclass stub
    └── agent.py                    # CarRacingPPOAgent(PPOAgent): pixel preprocess pipeline (functional skeleton, full training is a separate stage spec)
```

The two `agent.py` files in `2-mountaincar/` and `3-car-racing/` exist solely to demonstrate and exercise the override contract from FR-009. Their full training scripts, hyperparameter sweeps, and per-stage tests belong to separate stage features and are explicitly out of scope (see spec Assumptions → Out of scope). The `2-mountaincar/agent.py` may legitimately be a thin pass-through (preprocess returns `obs`); the `3-car-racing/agent.py` ships a working pixel preprocessing pipeline so that `test_agent_interface.py` can exercise the override path even though the stage-3 training script is deferred.

**Structure Decision**: Single-file skeleton (`ppo_skeleton.py`) plus a sibling test runner and Agent-contract test, with two minimal subclass demonstrations in adjacent stage directories. This matches the brainstorm's "Direction 1" (single file + custom runner) and the constitution's progressive-scaffolding rule. No `src/`, `tests/`, or package layout — the workshop directories ARE the layout, and that is intentional so participants can `cd workshop-1/1-ppo/` and stay there for the entire stage.

## Complexity Tracking

*Not applicable — Constitution Check passed with no violations.*
