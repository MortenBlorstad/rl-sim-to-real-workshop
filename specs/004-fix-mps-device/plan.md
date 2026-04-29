# Implementation Plan: Fix Device Selection (MPS Should Not Be Slower Than CPU)

**Branch**: `004-fix-mps-device` | **Date**: 2026-04-29 | **Spec**: [spec.md](./spec.md)
**Input**: Feature specification from `specs/004-fix-mps-device/spec.md`

## Summary

Restore honest device auto-selection in the custom PPO implementation under `workshop-1/1-ppo/ppo/`, then eliminate the per-env-step CPU↔MPS round-trips in `PPOAgent.train()` that are the actual root cause of MPS being slower than CPU.

Three concrete deliverables:

1. **`get_device()` policy** (`workshop-1/1-ppo/ppo/utils/utils.py`): remove the unconditional `return torch.device("cpu")` and re-enable CUDA → MPS → CPU auto-selection. Honour an environment-variable override (`RL_WORKSHOP_DEVICE` ∈ {`cpu`, `cuda`, `mps`, `auto`}) and fail fast when the requested device is unavailable. When MPS is selected, set `PYTORCH_ENABLE_MPS_FALLBACK=1` in the process environment before any tensor is allocated (per spec clarification Q1).

2. **Hot-path device routing** (`workshop-1/1-ppo/ppo/ppo.py`): split rollout collection from the update phase. Rollout runs on CPU (no per-step `.to(device)` and no `.item()` syncs from MPS); only the full `RolloutBuffer` batch is moved to `self.device` once per epoch in the update phase, where the cost is amortised over `batch_size`. Networks live on `self.device`; a small CPU-resident shadow copy of the actor (and critic for value bootstrap) is kept in sync via `actor_cpu.load_state_dict(actor.state_dict())` once per update boundary. This is transparent to user code (`predict`, `sample_action`, `evaluate` keep the same signatures and tensor types).

3. **Run metadata + tests + benchmark**: extend the per-run `metadata.json` with an authoritative `device` field reflecting the actual backend used (FR-010). Parameterise `test_ppo.py` over `{cpu, mps}` (and `cuda` where present) using `agent.device` injection. Add a `bench_device.py` script under `workshop-1/1-ppo/ppo/tests/` (NOT in `tests/` runtime; it is opt-in) that measures wall-clock per update on Pendulum (small MLP) and CarRacing (CNN) for CPU vs MPS and reports the ratios used by SC-001 and SC-002. Update READMEs (FR-012).

The plan introduces no new dependencies and touches no Workshop 2 code. The Article II / VII surface (`Agent` interface, sim-to-real pipeline) is unchanged: only the *device* on which existing operations run changes, and `save`/`load` already moves tensors via `state_dict()` which is device-agnostic.

## Technical Context

**Language/Version**: Python 3.10–3.12 (per Constitution; `pyproject.toml` pins `>=3.10,<3.13`)
**Primary Dependencies**: PyTorch ≥ 2.1 (MPS backend used on Apple Silicon), Gymnasium ≥ 0.29, NumPy. No new dependencies.
**Storage**: Per-run metadata under `runs/<stage>/<run-name>/metadata.json` — schema fixed by feature 002's `run-format.md` contract; this feature only *populates* the existing `device` field rather than redefining the schema.
**Testing**: Existing custom `@step` runner at `workshop-1/1-ppo/ppo/tests/test_ppo.py`. Tests are extended to parameterise over devices via the `RL_WORKSHOP_DEVICE` env var the runner reads. No pytest, no new test framework.
**Target Platform**: Developer laptop. Three concrete device classes are in scope:
- macOS / Apple Silicon (M-series) — primary target for the MPS fix.
- Linux + CUDA — must keep working (auto-selects CUDA).
- macOS / Intel + headless CI — CPU fallback, must keep working without warnings.
**Project Type**: Workshop training package (`workshop-1/1-ppo/ppo/`) consumed by stage drivers (`workshop-1/2-pendulum/train.py`, `workshop-1/3-car-racing/train.py`). No web/mobile.
**Performance Goals** (per spec SC-001 / SC-002):
- Pendulum stage: mean wall-clock per PPO update on MPS ≤ 1.10 × CPU over ≥ 20 updates after 2-update warm-up.
- CarRacing stage: mean wall-clock per PPO update on MPS ≤ 0.80 × CPU over ≥ 10 updates after 2-update warm-up.
**Constraints**:
- The unit test suite must run in < 60 s wall-clock on each device (the existing < 10 s per step budget from the constitution still applies; device parameterisation does not multiply test count by 3).
- Existing `pretrained/` `.pt` artefacts must keep loading (FR-008). They were saved with `torch.save(state_dict)` — device-agnostic — so the load path can be preserved by routing `torch.load(..., map_location="cpu")` followed by `.to(self.device)`.
- The override mechanism must work without modifying any source under `workshop-1/`.
**Scale/Scope**: ~3 source files modified (`utils/utils.py`, `ppo.py`, `_runlog.py` for the metadata field), 1 new file (`tests/bench_device.py`), 1 test file extended (`tests/test_ppo.py`). No new package boundaries.

## Constitution Check

*GATE: Must pass before Phase 0 research. Re-checked after Phase 1 design (see end of section).*

Constitution: `.specify/memory/constitution.md` v1.0.0 (ratified 2026-04-13).

| Article | Status | Notes |
|---|---|---|
| I. Participant-First Design | ✅ Pass | Override is a single env var with clear error messages on unavailable devices. README updates spell out the default per-platform. All identifiers/docstrings in English. No silent successes (the chosen device is logged and recorded in `metadata.json`). |
| II. Two Paths, One Agent API (NON-NEGOTIABLE) | ✅ Pass | The `Agent` interface (`preprocess`, `predict`, `train`, `save`, `load`) is unchanged. The CPU rollout / device update split lives entirely inside `PPOAgent.train()`; callers see no API change. `predict()` keeps accepting raw NumPy obs and returning NumPy actions. SB3 path is untouched. |
| III. Gymnasium-Conformant Environments | ✅ Pass | Env step/reset signatures unchanged. No new `ObservationWrapper`s. |
| IV. Test-Verified Implementation (NON-NEGOTIABLE) | ✅ Pass | All 5 PPO `@step` tests + the agent-interface tests must pass on each available device. Test parameterisation goes through `RL_WORKSHOP_DEVICE` (the same override participants will use), keeping the contract uniform. The < 10 s per-test budget is preserved — device parameterisation runs are CI-loop-level, not multiplied per test. A new `bench_device.py` script measures the SC-001/SC-002 ratios but is **not** in the test runner; running it is opt-in (workshop-leader smoke test). |
| V. Progressive Scaffolding | ✅ Pass | The five `# TODO N: …` blocks in `ppo.py` are unchanged in structure. The CPU-rollout / device-update split lives outside the TODO blocks (around them, in `train()`'s scaffolding) so participants still write the same code in the same place. `raise NotImplementedError("TODO N: …")` defaults preserved on `solutions`-branch merge. |
| VI. Fail-Safe Workshop Design | ✅ Pass | `PYTORCH_ENABLE_MPS_FALLBACK=1` is auto-set on MPS so unsupported ops never block a participant (clarification Q1). `RL_WORKSHOP_DEVICE=cpu` is the documented escape hatch when MPS misbehaves on a specific PyTorch build (User Story 2). Pretrained `.pt` artefacts keep loading via `map_location="cpu"`-then-`.to(device)` (FR-008). |
| VII. Sim-to-Real Pipeline Integrity | ✅ Pass | `Agent.save()` continues to write a device-agnostic `state_dict`. `Agent.load()` already reconstructs the agent on the local `get_device()`, so a CPU-saved Mac model loads on MPS and a Pi-side CPU load is unaffected. No change to `export_model.py` / `deploy.sh` is required: the Pi has no MPS, `get_device()` returns CPU there, and the save/load contract is preserved. |

**Gate decision**: ✅ Proceed. All seven articles are satisfied; no violations to track. The change is intentionally scoped narrowly: device policy + a transparent rollout/update split + metadata + benchmark.

## Project Structure

### Documentation (this feature)

```text
specs/004-fix-mps-device/
├── plan.md              # This file
├── research.md          # Phase 0 — root-cause analysis + decisions on the rollout/update split, override mechanism shape, MPS_FALLBACK timing
├── data-model.md        # Phase 1 — DeviceSelection policy + RunMetadata.device field
├── contracts/
│   ├── device-policy.md # Public contract of get_device(): inputs, outputs, errors
│   └── run-metadata.md  # Delta against feature 002's run-format.md (just the `device` field shape)
├── quickstart.md        # Phase 1 — how to verify the fix on a fresh checkout (Apple Silicon + CUDA + CPU-only paths)
├── checklists/
│   └── requirements.md  # already created by /speckit.specify
└── tasks.md             # Phase 2 output (created by /speckit.tasks, NOT by this command)
```

### Source Code (repository root)

```text
workshop-1/
├── 1-ppo/
│   └── ppo/
│       ├── ppo.py                              # MODIFIED — split rollout (CPU) from update (self.device); shadow-copy actor/critic state for rollout; preserve TODO 5 structure
│       ├── networks.py                         # unchanged
│       ├── rollout_buffer.py                   # unchanged (already device-agnostic; stores NumPy/CPU and yields tensors)
│       ├── utils/
│       │   ├── utils.py                        # MODIFIED — get_device() policy: env-var override, CUDA→MPS→CPU auto, fail-fast, set PYTORCH_ENABLE_MPS_FALLBACK on MPS
│       │   ├── _runlog.py                      # MODIFIED — write `device` field into metadata.json
│       │   └── _log_parser.py                  # unchanged
│       └── tests/
│           ├── test_ppo.py                     # EXTENDED — read RL_WORKSHOP_DEVICE, parameterise tensor construction; existing assertions unchanged
│           ├── test_agent_interface.py         # EXTENDED — add cross-device save/load round-trip test (CPU↔MPS, max-abs-diff ≤ 1e-3 per spec SC-004)
│           └── bench_device.py                 # NEW — opt-in micro-benchmark for SC-001 / SC-002 (not in the @step runner)
├── 2-pendulum/
│   └── train.py                                # unchanged — already constructs PPOAgent which calls get_device() internally
└── 3-car-racing/
    └── train.py                                # unchanged — same reason
```

**Structure Decision**: Single Python package (`workshop-1/1-ppo/ppo/`) with the device fix localised to two files (`utils/utils.py`, `ppo.py`) plus one metadata field (`utils/_runlog.py`). The benchmark script lives next to the tests but is explicitly out of the runtime test suite — it requires both an env on the machine *and* user opt-in (workshop leaders run it before delivery; participants do not). No new package boundaries, no new dependencies.

## Complexity Tracking

> *Filled only when Constitution Check has violations.* All gates pass; this section is intentionally empty.

| Violation | Why Needed | Simpler Alternative Rejected Because |
|-----------|------------|--------------------------------------|
| *(none)* | — | — |

---

## Phase 0 — Outline & Research

Resolved decisions (full detail in `research.md`):

1. **Why MPS is currently slower than CPU.** Confirmed by code reading (`workshop-1/1-ppo/ppo/ppo.py:279–317`): the rollout loop performs (a) `torch.as_tensor(obs, device=self.device)` per env step, (b) `self.critic(obs_t).item()` per step (forces device→host sync), (c) `log_prob.item()` per step, and (d) `action.detach().cpu().numpy()` per step. On MPS each of these is a synchronous round-trip over the unified-memory boundary; on CPU these are no-ops. With a 2×64 MLP forward, the round-trip cost dominates the compute. CarRacing (CNN) would behave better but is currently capped by the same per-step pattern.

2. **Strategy choice — CPU rollout / device update.** Decision: rollout collection runs on CPU (cheap tensors), `RolloutBuffer` continues to store NumPy/CPU data, and only the full batch is moved to `self.device` inside the update phase (where line 328 `batch = {k: v.to(self.device) ...}` already happens). For per-step inference (sample_action, value bootstrap), keep a CPU-resident shadow copy of actor + critic and refresh from the device weights once per update via `actor_cpu.load_state_dict(actor.state_dict())`. Alternatives considered: (i) leave everything on device — fails because the per-step transfers stay; (ii) eliminate `.item()` calls only — partial fix, still pays per-step `.to(device)`; (iii) pure CPU agent — defeats the user's goal. The shadow-copy pattern keeps the public API unchanged.

3. **Override mechanism shape.** Decision: environment variable `RL_WORKSHOP_DEVICE` with values `cpu | cuda | mps | auto` (default `auto`). Rationale: env vars are zero-friction (works for `train.py`, `test_ppo.py`, the bench script, and any participant script without per-driver flag plumbing); they survive across stages without re-passing CLI args; they live next to `PYTORCH_ENABLE_MPS_FALLBACK` semantically. Alternatives considered: CLI flag on each driver (rejected — multiplies surfaces, leaks into TODO blocks), hyperparameter on `PPOAgent` (rejected — pollutes the participant API, doesn't help test runners or the bench).

4. **MPS_FALLBACK timing.** Decision: `os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")` is set inside `get_device()` *before* it returns `torch.device("mps")`, and `get_device()` is called early — the first `PPOAgent.__init__()` triggers it before any tensor is allocated on MPS. Rationale: PyTorch reads this env var at op-dispatch time per call, so the timing requirement is "set it before the first MPS op," which the location above guarantees. `setdefault` preserves any user-set value.

5. **Cross-device load.** Decision: change `torch.load(path, weights_only=False)` to `torch.load(path, map_location="cpu", weights_only=False)` then let `agent = target_cls(env, hyperparameters=...)` place tensors on the new local device (it already does, via `.to(self.device)` in `__init__`). Rationale: Avoids "tried to load CUDA tensor on Mac" failures with `pretrained/` artefacts. The `state_dict` on disk is device-tagged in PyTorch; `map_location` strips that.

6. **Test parameterisation.** Decision: tests read `RL_WORKSHOP_DEVICE` exactly like production code; the `@step` runner adds a one-line "device: …" header to its output. CI / workshop-leader pre-flight runs the suite once per available device by setting the env var. Total wall-clock under the constitution's < 10 s per step is preserved because each device run sees the same individual test budget.

7. **Numerical tolerance for cross-device load (SC-004).** Decision: a new `@step` test in `test_agent_interface.py` saves a freshly-trained PPOAgent on one device, loads on another, runs a deterministic single-episode rollout, and asserts max-abs-diff ≤ 1e-3 on actions for CPU↔MPS and ≤ 1e-5 for CPU↔CUDA (from spec clarification Q2). Where the alternate device is unavailable, the test self-skips with a clear message.

8. **Pretrained artefact regeneration.** Decision: not required. `pretrained/*.pt` files are state-dicts produced by `Agent.save`; the load path's `map_location="cpu"` change handles them. A spot-check load on each available device is added to the workshop-leader pre-flight (referenced from `quickstart.md`) but is not a test gate.

**Output**: `research.md` (consolidates the eight items above with code-line references and the alternatives-considered chain).

## Phase 1 — Design & Contracts

**Prerequisites:** `research.md` complete.

1. **Data model** (`data-model.md`):
   - `DevicePolicy` — pure function `(host_caps, env_var) → torch.device | DeviceUnavailableError`. Inputs: `torch.cuda.is_available()`, `torch.backends.mps.is_available()`, `os.environ.get("RL_WORKSHOP_DEVICE", "auto")`. Output: `torch.device` and a side-effect (set `PYTORCH_ENABLE_MPS_FALLBACK=1` when result is MPS). Errors: `DeviceUnavailableError("requested 'cuda' but no CUDA backend; available: cpu, mps")` (illustrative; the message lists what *is* available so participants can self-correct).
   - `RunMetadata.device` — string, one of `"cpu" | "cuda" | "mps"`, the actual backend used (not the requested one). Lives in the existing `metadata.json` per feature 002's `run-format.md` — this feature only ensures the field is populated for custom-PPO runs.

2. **Contracts** (`contracts/`):
   - `device-policy.md` — table of (env var value × cuda available × mps available) → resolved device or error, plus the `PYTORCH_ENABLE_MPS_FALLBACK` side-effect rule. This is the single source of truth that `tests/test_ppo.py` mirrors.
   - `run-metadata.md` — points at feature 002's existing `run-format.md` and pins down the `device` field's allowed values + when it is written (at run start, never mutated).

3. **Quickstart** (`quickstart.md`): three short flows — (a) "On Apple Silicon, verify MPS is now used and at parity," (b) "Override to CPU for debugging," (c) "Run the cross-device load test." Each ends with a single visible-outcome assertion line.

4. **Agent context update**: Run `.specify/scripts/bash/update-agent-context.sh claude` to refresh `CLAUDE.md`'s "Active Technologies" / "Recent Changes" sections. No new dependencies are added; the script's "preserve manual additions" markers will keep this feature's entry minimal.

**Output**: `data-model.md`, `contracts/device-policy.md`, `contracts/run-metadata.md`, `quickstart.md`, refreshed `CLAUDE.md`.

## Constitution Check (Post-Design)

Re-evaluated after Phase 1 design — no change in posture. All seven articles still pass; the design choices made in research/data-model/contracts (env-var override, transparent rollout/update split, no new tests in the runtime path beyond cross-device load, device-agnostic state_dict on save/load) keep Articles II / IV / VII intact and reinforce VI (escape hatch documented).

**Gate decision**: ✅ Proceed to `/speckit.tasks`.
