# Feature Specification: Fix Device Selection (MPS Should Not Be Slower Than CPU)

**Feature Branch**: `004-fix-mps-device`
**Created**: 2026-04-29
**Status**: Draft
**Input**: User description: "fix device. device on cpu is faster than mps now. it should not be that."

## Background *(context — non-mandatory)*

The custom PPO implementation exposes a `get_device()` helper that is intended to pick the fastest available accelerator (CUDA → MPS → CPU). At present that helper unconditionally returns CPU; the accelerator-selection logic is dead code sitting after an early `return`. The reason this workaround was introduced is that, on Apple Silicon, training was empirically *slower* on MPS than on CPU for the workshop's small networks. The user wants this fixed: MPS should provide a real speedup (or at least parity), and the device helper should once again select the best backend honestly instead of being hard-coded to CPU.

## Clarifications

### Session 2026-04-29

- Q: How should the framework handle MPS ops that PyTorch doesn't support natively? → A: Framework auto-sets `PYTORCH_ENABLE_MPS_FALLBACK=1` when MPS is selected, so unsupported ops silently fall back to CPU per-op (no warning required).
- Q: What max-absolute-difference bound should hold when a `.pt` artefact is loaded across CPU↔MPS and a deterministic rollout is replayed? → A: max-abs-diff ≤ 1e-3 (matches typical fp32 reduction noise on MPS). The CPU↔CUDA bound stays at 1e-5.

## User Scenarios & Testing *(mandatory)*

### User Story 1 — MPS speedup on Apple Silicon is real, not a regression (Priority: P1)

A workshop participant on a Mac with Apple Silicon runs `train.py` for the Pendulum stage (or any custom-PPO stage) on a fresh checkout. They expect that the framework will use MPS automatically because their machine has it, and they expect this to be at worst on par with — and ideally faster than — running the same training on CPU. Today the framework silently falls back to CPU and pretends MPS does not exist; this story restores the original promise of `get_device()`.

**Why this priority**: This is the whole point of the bug report. Without this, the workshop cannot truthfully claim "we use the best available accelerator," and Apple Silicon participants get no benefit from their hardware. It also unblocks the deeper investigation of *why* MPS was slower in the first place.

**Independent Test**: On an Apple Silicon Mac, run a short training session (≤ 30 PPO updates) for the Pendulum stage twice — once with the device forced to CPU and once with default device selection — and observe that (a) the default selection actually picks MPS, and (b) wall-clock time per update on MPS is no worse than on CPU.

**Acceptance Scenarios**:

1. **Given** an Apple Silicon Mac where `torch.backends.mps.is_available()` returns true, **When** training is started without overriding device selection, **Then** the run logs "device=mps" (or equivalent) and tensors are placed on the MPS device.
2. **Given** the same machine and the same short training run, **When** wall-clock time per PPO update is measured on MPS and on CPU, **Then** MPS is no slower than CPU within a small tolerance (e.g., MPS time ≤ 1.10 × CPU time on the Pendulum stage).
3. **Given** a machine without MPS but with CUDA, **When** training starts, **Then** CUDA is selected.
4. **Given** a machine with neither CUDA nor MPS, **When** training starts, **Then** CPU is selected and there is no failure.

---

### User Story 2 — Participant can override device selection (Priority: P2)

A participant who is debugging a numerical issue, profiling, or running on a machine where MPS happens to misbehave for their specific PyTorch build wants to force CPU (or any other device) without editing library code. They expect a clearly documented, supported way to do this — for example a CLI flag on `train.py`, an environment variable, or a hyperparameter — so they can fall back without forking the repo.

**Why this priority**: Even after MPS parity is restored, edge cases will exist (driver bugs, transient PyTorch nightlies, ops that fall back to CPU silently). Giving participants an escape hatch prevents the workshop from grinding to a halt for one person, which is the same design philosophy that motivates the "SB3 escape hatch" elsewhere in the workshop.

**Independent Test**: On any machine, start training with the override set to `cpu` and verify both that the run logs report CPU and that the run completes successfully — without modifying source files.

**Acceptance Scenarios**:

1. **Given** a machine where MPS is available, **When** the participant sets the documented override to `cpu`, **Then** training runs on CPU and the override is reflected in the run log/metadata.
2. **Given** the override is left unset, **When** training starts, **Then** automatic selection (User Story 1) takes effect.
3. **Given** the override is set to a value the local machine cannot honour (e.g. `cuda` on a Mac), **When** training starts, **Then** the system fails fast with a clear error message rather than silently falling back.

---

### User Story 3 — Existing tests and saved artefacts keep working across devices (Priority: P3)

The PPO test suite, the rollout/eval pipeline, and any pretrained `*.pt` artefacts in `pretrained/` must keep working after this change. In particular, a model saved on one device must be loadable on another (e.g., a participant downloads a CPU-trained pretrained model and runs it on their MPS Mac, or vice versa), and all `torch.as_tensor(..., device=self.device)` boundaries inside the agent must continue to function on every supported backend.

**Why this priority**: Workshop participants rely on `pretrained/` as a safety net (per `CLAUDE.md`). Breaking cross-device load would silently make those safety nets unusable for half the audience. The PPO unit tests already construct tensors on `agent.device`; restoring real MPS selection must not break them.

**Independent Test**: Run the existing `test_ppo.py` suite on each available device, then load a CPU-saved checkpoint on MPS (and vice versa where hardware permits) and confirm that one rollout produces sensible (finite, in-bounds) actions.

**Acceptance Scenarios**:

1. **Given** the existing PPO unit tests, **When** they are executed on a machine where `get_device()` returns MPS, **Then** they all pass with no `dtype`/`device` mismatch errors.
2. **Given** a `.pt` file produced on one device, **When** it is loaded on a different device via `PPOAgent.load`, **Then** it loads successfully and `predict`/rollout returns finite actions inside the action bounds.

---

### Edge Cases

- **MPS is "available" but the running PyTorch build silently falls back to CPU for some op** (e.g., a distribution sample). The auto-selected MPS run must still complete; if any op hard-fails, the error must be clear enough for a participant to understand they should override to CPU.
- **MPS unsupported ops**: when MPS is selected, the framework MUST automatically set `PYTORCH_ENABLE_MPS_FALLBACK=1` in the process environment so that any op PyTorch does not support on MPS (e.g., certain `Normal` sampling paths, certain reductions) silently falls back to CPU per-op. Setting must happen early enough to take effect (i.e., before any tensor is allocated on MPS). No warning is required when a fallback occurs; the goal is a frictionless workshop experience.
- **Per-step CPU↔MPS transfer overhead**: rollout collection currently moves a single observation to the device per env step (`torch.as_tensor(obs, ..., device=self.device)`). On small networks this transfer can dominate the forward pass and is the most likely cause of MPS being slower than CPU. The fix must address (or explicitly document an exemption for) this hot path.
- **CartPole / Pendulum / MountainCar networks are tiny (2×64 MLP)**: even with transfers eliminated, the GPU may be no faster than CPU. The success criterion is *parity*, not unconditional speedup, on these stages.
- **CarRacing has a CNN input**: this stage is the one most likely to show real MPS speedup; correctness on it is non-negotiable.
- **CI / headless machines** without MPS or CUDA: behaviour must be unchanged (CPU, no warnings, no crashes).
- **`pretrained/` artefacts** were produced under the current CPU-only behaviour. They must continue to load under the new selection logic.

## Requirements *(mandatory)*

### Functional Requirements

- **FR-001**: The framework MUST select the fastest available backend automatically by default, in the order CUDA → MPS → CPU, with no hard-coded forced-CPU fallback in `get_device()`.
- **FR-002**: On Apple Silicon, automatic selection MUST result in MPS being used whenever `torch.backends.mps.is_available()` is true.
- **FR-003**: The framework MUST provide a documented, source-edit-free way for a participant to override device selection (e.g., environment variable and/or CLI flag), and the chosen device MUST be reflected in run metadata/logs.
- **FR-004**: Per-PPO-update wall-clock time on MPS MUST be no worse than on CPU on the Pendulum stage (small MLP), within a tolerance of 10%, on a representative Apple Silicon machine.
- **FR-005**: Per-PPO-update wall-clock time on MPS MUST be no worse than on CPU on the CarRacing stage (CNN input), and SHOULD show a measurable speedup, on a representative Apple Silicon machine.
- **FR-006**: The PPO unit test suite (`workshop-1/1-ppo/ppo/tests/test_ppo.py`) MUST pass on each supported device (CPU, MPS, and CUDA where present) without modification of test logic — only fixture/parameterisation changes are acceptable.
- **FR-007**: Saved model artefacts (`.pt` files produced by `PPOAgent.save`) MUST be loadable across devices: a model saved on CPU must load on MPS and vice versa, and a single rollout after load MUST produce finite, in-bounds actions.
- **FR-008**: Existing `pretrained/` artefacts MUST continue to load and run on all supported backends after this change.
- **FR-009**: When the user-requested device is unavailable on the local machine, the system MUST fail fast with a clear, actionable error message (naming the requested device and the available alternatives) instead of silently substituting another device.
- **FR-010**: The selected device MUST be recorded in the per-run metadata file under `runs/<stage>/<run-name>/` so that retrospective benchmarking and debugging can attribute timings to a backend.
- **FR-011**: If the fix involves keeping rollout collection on CPU and only moving full batches to the accelerator for the update phase (the most common cause of MPS slowness in this codebase), that behaviour MUST be transparent to user code — `PPOAgent.predict`, `sample_action`, and the rollout buffer interfaces must continue to work without callers being aware of where tensors live mid-rollout.
- **FR-012**: Documentation (`README.md` for the affected stages and/or top-level CLAUDE.md entries) MUST be updated to describe the override mechanism, the expected default behaviour per platform, and the auto-fallback behaviour on MPS.
- **FR-013**: When MPS is the selected device, the framework MUST set `PYTORCH_ENABLE_MPS_FALLBACK=1` in the process environment before any tensor is allocated on MPS, so that ops not implemented on MPS transparently fall back to CPU per-op. No user-visible warning is required for individual fallback events.

### Key Entities *(include if feature involves data)*

- **Device selection policy**: The decision function that maps `(host capabilities, user override)` to the active `torch.device`. Currently degenerate (always CPU); after this feature, a small but well-defined function with documented inputs and a logged decision.
- **Run metadata record**: The existing per-run metadata JSON gains an authoritative `device` field reflecting the actual backend used (not just what was requested), so cross-run comparisons of wall-clock time are interpretable.

## Success Criteria *(mandatory)*

### Measurable Outcomes

- **SC-001**: On a representative Apple Silicon Mac running the Pendulum stage, mean wall-clock time per PPO update on MPS is within 10% of CPU (i.e., MPS is at parity or faster) over a window of at least 20 updates after warm-up.
- **SC-002**: On the same Mac running the CarRacing stage (CNN), mean wall-clock time per PPO update on MPS is at least 20% faster than on CPU over a window of at least 10 updates after warm-up.
- **SC-003**: 100% of the existing `test_ppo.py` tests pass on each of {CPU, MPS} on an Apple Silicon Mac, and on {CPU, CUDA} on a Linux+CUDA host.
- **SC-004**: A `.pt` artefact produced on one device loads on every other device with zero edits to user code, and a single deterministic rollout after load produces actions matching the source-device rollout within: max-abs-difference ≤ 1e-5 for CPU↔CUDA, and ≤ 1e-3 for CPU↔MPS (looser bound accommodates fp32 reduction-order differences on MPS).
- **SC-005**: A participant can switch the active device via the documented override mechanism in under 30 seconds and without editing any source file under `workshop-1/`.
- **SC-006**: `runs/<stage>/<run-name>/metadata.json` (or the equivalent existing field defined in feature 002's `run-format.md`) reports the actual device used; spot-checking five recent runs across CPU/MPS shows correct attribution in 5/5 cases.

## Assumptions

- The investigation will confirm — and the implementation will address — that the dominant cause of "MPS slower than CPU" is per-env-step Python→MPS tensor placement during rollout collection, combined with very small (2×64) network forward passes that do not amortise the transfer cost. If profiling reveals a different root cause, the fix scope adjusts but the success criteria above still apply.
- "Representative Apple Silicon Mac" means an M-series Mac (M1/M2/M3-class) with a recent stable PyTorch release that the workshop's `pyproject.toml` already pins. We are not measuring across every Mac generation.
- The override mechanism may be introduced as an environment variable, a CLI flag on the stage `train.py` drivers, a hyperparameter in `PPOAgent`, or some combination — the exact surface is a planning-phase decision; the requirement is only that *some* documented, source-edit-free override exists.
- The `pretrained/` artefacts under version control are still usable after the fix; if profiling reveals that some artefacts must be regenerated for cross-device load to work, that regeneration is in scope for the implementation but the artefacts themselves are not redesigned here.
- This feature touches only the custom PPO implementation under `workshop-1/1-ppo/`. The Stable-Baselines3 path already does its own device handling and is out of scope.
- Workshop 2 (DonkeyCar) device handling is out of scope for this feature.
