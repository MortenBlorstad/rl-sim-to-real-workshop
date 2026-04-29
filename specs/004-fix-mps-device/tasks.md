---

description: "Task list for feature 004-fix-mps-device"
---

# Tasks: Fix Device Selection (MPS Should Not Be Slower Than CPU)

**Input**: Design documents from `/specs/004-fix-mps-device/`
**Prerequisites**: plan.md, spec.md, research.md, data-model.md, contracts/device-policy.md, contracts/run-metadata.md, quickstart.md

**Tests**: Tests are required for this feature — User Story 3 *is* the test-portability story, and Constitution Article IV (NON-NEGOTIABLE) requires `test_ppo.py` and `test_agent_interface.py` to keep passing on every supported device.

**Organization**: Tasks are grouped by user story so each story can be implemented and validated independently against the spec's acceptance scenarios.

## Format: `[ID] [P?] [Story] Description`

- **[P]**: Can run in parallel (different files, no dependencies)
- **[Story]**: Which user story this task belongs to (US1, US2, US3)
- File paths are absolute and reference real files in the repo unless flagged NEW

## Path Conventions

This is a workshop training package, not a "src/tests" project:

- Library code: `workshop-1/1-ppo/ppo/`
- Library tests (custom step-runner, no pytest): `workshop-1/1-ppo/ppo/tests/`
- Stage drivers: `workshop-1/{2-pendulum,3-car-racing}/train.py`
- Notebooks: `workshop-1/{2-pendulum,3-car-racing}/analyze.ipynb`

---

## Phase 1: Setup (Shared Infrastructure)

**Purpose**: No new project structure or dependencies. Just verify the working state.

- [ ] T001 Verify `pyproject.toml` `[dependency-groups].workshop1` already pins PyTorch ≥ 2.1, Gymnasium ≥ 0.29, NumPy and that `uv sync --group workshop1` succeeds on a fresh checkout. No edits unless something is missing.
- [ ] T002 Verify branch `004-fix-mps-device` is checked out and `specs/004-fix-mps-device/{spec.md,plan.md,research.md,data-model.md,contracts/,quickstart.md}` are all present.

---

## Phase 2: Foundational (Blocking Prerequisites)

**Purpose**: Implement the device-selection policy. Every user story uses this — US1 reads `get_device()` to pick MPS, US2 exercises the env-var override and the fail-fast error, US3 parameterises tests via the same env var.

**⚠️ CRITICAL**: No user story work can begin until this phase is complete.

- [ ] T003 Add `DeviceUnavailableError(RuntimeError)` class in `workshop-1/1-ppo/ppo/utils/utils.py` per `contracts/device-policy.md` § "Error type". Docstring must say it is raised when `RL_WORKSHOP_DEVICE` names an unavailable device or an unrecognised value, and include the three-part message format (quoted requested value, why, available devices + actionable next step).
- [ ] T004 Replace the body of `get_device()` in `workshop-1/1-ppo/ppo/utils/utils.py` with the decision table from `contracts/device-policy.md`. Remove the unconditional `return torch.device("cpu")` at line 52. Read `RL_WORKSHOP_DEVICE` (case-insensitive, whitespace-stripped, default `auto`); call `os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")` only when the resolved device's type is `mps`; raise `DeviceUnavailableError` with the contract-mandated message format on the four error rows of the table. No logging from inside this function — caller logs.
- [ ] T005 Re-export `get_device` and `DeviceUnavailableError` from `workshop-1/1-ppo/ppo/utils/__init__.py`. Append `DeviceUnavailableError` to `__all__`.

**Checkpoint**: `from ppo.utils import get_device, DeviceUnavailableError` works; `RL_WORKSHOP_DEVICE=cpu python -c "from ppo.utils import get_device; print(get_device())"` prints `cpu`; `RL_WORKSHOP_DEVICE=gpu python -c "..."` raises `DeviceUnavailableError` with the expected message.

---

## Phase 3: User Story 1 — MPS speedup on Apple Silicon is real (Priority: P1) 🎯 MVP

**Goal**: Restore auto-selected MPS on Apple Silicon and make it actually faster than (or at parity with) CPU by eliminating per-env-step CPU↔MPS round-trips.

**Independent Test**: Run `bench_device.py --stage pendulum` and `--stage carracing` on an Apple Silicon Mac. Pendulum line ends `PASS_SC001=true` (MPS within 10% of CPU). CarRacing line ends `PASS_SC002=true` (MPS at least 20% faster than CPU).

### Implementation for User Story 1

- [ ] T006 [US1] In `workshop-1/1-ppo/ppo/ppo.py:50-77` (`PPOAgent.__init__`), after `self.actor = ActorNetwork(...).to(self.device)` and `self.critic = CriticNetwork(...).to(self.device)`, allocate CPU shadow copies: `self._actor_cpu = ActorNetwork(self.obs_dim, self.action_dim)` (no `.to(...)` — defaults to CPU), `self._critic_cpu = CriticNetwork(self.obs_dim)`, `self._log_std_cpu = torch.zeros(self.action_dim) + self.hyperparameters["log_std_init"]`. Initialise them by copying current state-dict from the device-resident networks (so first rollout uses the same weights). Add a `_refresh_cpu_shadow()` method that does `self._actor_cpu.load_state_dict(self.actor.state_dict()); self._critic_cpu.load_state_dict(self.critic.state_dict()); self._log_std_cpu.copy_(self.log_std.detach().cpu())`.
- [ ] T007 [US1] In `workshop-1/1-ppo/ppo/ppo.py` `train()` method (TODO 5 block), rewrite the rollout collection loop (currently lines 269–308) to use the CPU shadow networks. Concrete edits:
  - Replace `obs_t = torch.as_tensor(obs, dtype=torch.float32, device=self.device)` with `obs_t = torch.as_tensor(obs, dtype=torch.float32)` (CPU).
  - Replace the `with torch.no_grad(): action, log_prob = self.sample_action(obs_t, ...)` and `value_t = self.critic(obs_t).item()` block with a CPU-side equivalent: `action, log_prob = self._sample_action_cpu(obs_t, deterministic=False); value_t = self._critic_cpu(obs_t).item()`. Add a private `_sample_action_cpu` method that mirrors `sample_action` but uses `self._actor_cpu` and `self._log_std_cpu`. Public `sample_action` is unchanged (still uses device-resident weights for the update phase).
  - Replace the truncation-bootstrap block (lines 290–294) with the CPU-shadow critic: `terminal_value = self._critic_cpu(torch.as_tensor(next_obs, dtype=torch.float32)).item()`.
  - Replace the post-rollout last-value bootstrap (lines 313–317) with the CPU-shadow critic.
  - At the END of each update iteration (after `buffer.reset()`, line 369), call `self._refresh_cpu_shadow()` so the next rollout sees the freshly-trained weights.
  - Keep the line `batch = {k: v.to(self.device) for k, v in batch.items()}` at line 328 untouched — the update phase stays on `self.device`.
  - Preserve the `# -- YOUR CODE HERE --` / `# -- END YOUR CODE --` markers and the TODO 5 structure (Constitution Article V: no inlining solutions inside the markers; the shadow-copy plumbing lives outside the markers).
- [ ] T008 [P] [US1] Create NEW file `workshop-1/1-ppo/ppo/tests/bench_device.py` — opt-in micro-benchmark for SC-001/SC-002 and the workshop-leader pretrained smoke. Required entry points (CLI):
  - `--stage pendulum --updates N --warmup K`: trains `PPOAgent` on `Pendulum-v1` for `N` updates after `K` warm-up updates, on `cpu` then on the auto-selected device, measures mean wall-clock per update, prints exactly one final line: `[bench] pendulum: cpu=<X>s/upd  <auto_device>=<Y>s/upd  ratio=<R>  PASS_SC001=<bool>` where `PASS_SC001 = (R <= 1.10)` if the auto device is MPS, otherwise `PASS_SC001=N/A` (skipped).
  - `--stage carracing --updates N --warmup K`: same shape with `CarRacing-v3`, asserting `PASS_SC002 = (R <= 0.80)`.
  - `--pretrained-smoke`: iterate over `pretrained/*.pt`, load each via `PPOAgent.load`, run one greedy episode on the corresponding env, assert finite + within action bounds, print `[bench] pretrained smoke: <N> files OK on device=<name>`.
  - The script honours `RL_WORKSHOP_DEVICE` only via `get_device()` (not its own override). It uses `tempfile.TemporaryDirectory()` for scratch runs (no writes into participant's `runs/`).
- [ ] T009 [US1] On an Apple Silicon Mac, run `uv run python workshop-1/1-ppo/ppo/tests/bench_device.py --stage pendulum --updates 22 --warmup 2`. Confirm the final line ends `PASS_SC001=true`. If it fails, profile the rollout loop (likely a missed `.item()` or `.to(device)` left in `train()`) and iterate on T007 — do **NOT** widen the tolerance. Capture the bench output verbatim into a comment in the implementation PR description.
- [ ] T010 [US1] On the same Mac, run `uv run python workshop-1/1-ppo/ppo/tests/bench_device.py --stage carracing --updates 12 --warmup 2`. Confirm `PASS_SC002=true`. Same iterate-don't-relax rule. (CarRacing requires `swig` and `gymnasium[box2d]`; if those are not installed, run `uv sync --group workshop1` and re-attempt.)
- [ ] T011 [P] [US1] Sanity-check on a non-Mac, non-CUDA host (or by setting `RL_WORKSHOP_DEVICE=cpu`): the same bench script runs to completion and reports `PASS_SC001=N/A` cleanly (no crash, no division-by-zero on missing device).

**Checkpoint**: User Story 1 fully functional. On Apple Silicon, MPS is now auto-selected by default and is at parity-or-faster than CPU on Pendulum and ≥ 20 % faster on CarRacing. The full custom-PPO training driver `workshop-1/2-pendulum/train.py` runs unchanged because the rollout/update split lives entirely inside `PPOAgent.train()`.

---

## Phase 4: User Story 2 — Participant can override device selection (Priority: P2)

**Goal**: A documented, source-edit-free override (`RL_WORKSHOP_DEVICE`) with fail-fast behaviour on impossible requests, and an authoritative record of the actual device in the per-run metadata.

**Independent Test**: `RL_WORKSHOP_DEVICE=cpu uv run python workshop-1/2-pendulum/train.py` runs on CPU even on a Mac with MPS; the run's `metadata.json` contains `"device": "cpu"`. `RL_WORKSHOP_DEVICE=cuda uv run python workshop-1/2-pendulum/train.py` (on a Mac) fails before training starts with the contract-mandated error message.

### Implementation for User Story 2

- [ ] T012 [US2] In `workshop-1/1-ppo/ppo/ppo.py:59` (after `self.device = get_device()`), add a single startup log line: `print(f"[PPOAgent] device={self.device.type} (RL_WORKSHOP_DEVICE={os.environ.get('RL_WORKSHOP_DEVICE', 'auto')})", file=sys.stderr)`. (Use `sys.stderr` so it never disturbs stdout-parsing test runners.) Add `import os` if not already present.
- [ ] T013 [P] [US2] In `workshop-1/1-ppo/ppo/utils/_runlog.py`, when the metadata file is being prepared at run start, add the field `metadata["device"] = agent.device.type if agent is not None else get_device().type` per `contracts/run-metadata.md`. The field is written once and never updated. Find the call site by grepping `_runlog.py` for the metadata-dict construction; add the field next to existing run-start fields (run_name, env_id, timestamp, etc.).
- [ ] T014 [P] [US2] Update `workshop-1/2-pendulum/README.md` and `workshop-1/3-car-racing/README.md` (Norwegian, per repo convention): add a short subsection (≤ 5 lines) titled `Override av enhet` explaining the env var: allowed values `cpu | cuda | mps | auto`, default `auto`, behaviour when the requested device is unavailable, and the auto-fallback for unsupported MPS ops. Each README is currently in Norwegian; keep the new section in Norwegian.
- [ ] T015 [P] [US2] Update root `CLAUDE.md` § "Active Technologies" / "Conventions": add one line under Conventions noting the `RL_WORKSHOP_DEVICE` env var and the default policy. Do not duplicate the per-stage README content; this is for assistant context only.
- [ ] T016 [US2] Smoke-test fail-fast: on a Mac, run `RL_WORKSHOP_DEVICE=cuda uv run python workshop-1/2-pendulum/train.py --total-timesteps 4096`. Confirm it raises `DeviceUnavailableError` *before* the first env step, the message contains the literal `'cuda'` (quoted), the phrase `not available`, and the line `Available: cpu, mps`. Smoke-test invalid value: `RL_WORKSHOP_DEVICE=gpu` → message contains `not a recognised value` and `Allowed: cpu, cuda, mps, auto`.

**Checkpoint**: User Story 2 fully functional. The override mechanism is documented in stage READMEs, every PPO run prints its resolved device once at startup, the run's `metadata.json` records the actual backend used, and impossible requests fail with the contract-mandated message.

---

## Phase 5: User Story 3 — Existing tests and saved artefacts keep working across devices (Priority: P3)

**Goal**: The PPO test suite passes on each available device with no logic edits, `.pt` artefacts load across devices, and notebooks read the new metadata field defensively.

**Independent Test**: `RL_WORKSHOP_DEVICE=mps uv run python workshop-1/1-ppo/ppo/tests/test_ppo.py` and `RL_WORKSHOP_DEVICE=cpu uv run python ...` both end with `ALL STEPS OK!`. `uv run python workshop-1/1-ppo/ppo/tests/test_agent_interface.py --step 8` (the new C8 cross-device load step) ends `C8 cross-device load: max-abs-diff=<value>  bound=1e-3  OK!` on Apple Silicon.

### Implementation for User Story 3

- [ ] T017 [US3] In `workshop-1/1-ppo/ppo/ppo.py:504` (`PPOAgent.load`), change `state = torch.load(path, weights_only=False)` to `state = torch.load(path, map_location="cpu", weights_only=False)`. The subsequent `target_cls(env, hyperparameters=...)` already calls `.to(self.device)` in `__init__`, so the loaded state-dict is placed on the local device when `actor.load_state_dict(...)` etc. happen. No other changes to `load`.
- [ ] T018 [P] [US3] In `workshop-1/1-ppo/ppo/tests/test_ppo.py`, add a header line at the start of the `__main__` runner: `print(f"device: {os.environ.get('RL_WORKSHOP_DEVICE', 'auto')} -> {get_device().type}", file=sys.stderr)`. No changes to individual `@step` functions — they already use `agent.device` (lines 145–146, 228–231) which now varies with the env var. Verify by running the suite under `RL_WORKSHOP_DEVICE=cpu`, then `RL_WORKSHOP_DEVICE=mps` (Mac only), then `RL_WORKSHOP_DEVICE=cuda` (Linux only); each must end with `ALL STEPS OK!`.
- [ ] T019 [P] [US3] In `workshop-1/1-ppo/ppo/tests/test_agent_interface.py`, add a new `@step("C8")` function `test_cross_device_load` per `contracts/run-metadata.md` § "Test mapping" and `research.md` R7. The step:
  1. Determines available devices: `available = ["cpu"]` plus `cuda` if `torch.cuda.is_available()` else nothing, plus `mps` if `torch.backends.mps.is_available()` else nothing.
  2. If `len(available) < 2`: print `[skipped: only one device available (<name>)]` and return — counts as pass.
  3. Picks `source = available[0]`, builds `PPOAgent(env)` with `RL_WORKSHOP_DEVICE=<source>`, runs 64 update steps with a fixed seed, saves to `tmp_path/agent.pt`.
  4. Replays a deterministic 1-episode rollout from a fixed seed, recording the actions array `actions_src`.
  5. For each `target` in `available[1:]`: temporarily set `os.environ["RL_WORKSHOP_DEVICE"] = target`, call `PPOAgent.load(tmp_path/agent.pt, env)`, replay the same deterministic 1-episode rollout, compute `max-abs-diff` against `actions_src`. Bound: `1e-3` if any of `{source, target}` is `mps`, else `1e-5`.
  6. Print the final-line contract: `C8 cross-device load: max-abs-diff=<v>  bound=<b>  OK!` (or `FAIL: ...`).
- [ ] T020 [P] [US3] In `workshop-1/2-pendulum/analyze.ipynb` and `workshop-1/3-car-racing/analyze.ipynb`, find the cell(s) that read `metadata.json` and replace any direct `meta["device"]` access with `meta.get("device", "unknown")` per `contracts/run-metadata.md` § "Backwards compatibility". If neither notebook currently reads `device`, add a one-line display in an existing summary cell so the new metadata field is visible to participants.
- [ ] T021 [US3] Run the workshop-leader pretrained-smoke flow on each available device: `RL_WORKSHOP_DEVICE=cpu uv run python workshop-1/1-ppo/ppo/tests/bench_device.py --pretrained-smoke`, then `RL_WORKSHOP_DEVICE=mps ...` (Mac), then `RL_WORKSHOP_DEVICE=cuda ...` (Linux). Each must end with `[bench] pretrained smoke: <N> files OK on device=<name>` for some `N ≥ 1` (assuming at least one `.pt` exists under `pretrained/`). If any file fails, regenerate it from the `solutions` branch — do **NOT** silently delete failing artefacts.

**Checkpoint**: All three user stories now pass their independent tests. The PPO test suite is green on every available device; cross-device load is exercised by C8; pretrained artefacts load correctly on CPU, MPS, and CUDA where each is available.

---

## Phase 6: Polish & Cross-Cutting Concerns

**Purpose**: Clean up, validate end-to-end against `quickstart.md`, and confirm no regressions in adjacent areas.

- [ ] T022 [P] Audit `workshop-1/1-ppo/ppo/ppo.py` for any lingering hard-coded device references (e.g., `.cpu()`, `device="cpu"`, `device="cuda"`) that should now go through `self.device` or stay on CPU per the rollout/update split. Excludes intentional `.cpu().numpy()` boundary calls in `predict()` and rollout (those are correct). One-line comment justifying each remaining intentional reference.
- [ ] T023 Walk through `quickstart.md` end-to-end on an Apple Silicon Mac (Flow A → Flow B → Flow C → Flow D plus the "Run the full PPO test suite on each available device" section). Each flow's documented `Expected last line` MUST match verbatim. Capture any deviation as a quickstart-only edit (the implementation should already match — if it doesn't, fix the implementation, not the quickstart).
- [ ] T024 [P] Update `specs/004-fix-mps-device/checklists/requirements.md` — re-run the spec quality checklist against the now-implemented spec. Tick any items that were already Pass; flag any that became invalid (none expected).

---

## Dependencies & Execution Order

### Phase Dependencies

- **Phase 1 (Setup)**: No dependencies; trivial verification only.
- **Phase 2 (Foundational)**: Depends on Phase 1. **Blocks all user stories.** T003 → T004 → T005 (sequential within `utils.py`/`__init__.py`).
- **Phase 3 (US1)**: Depends on Phase 2. T006 (shadow-copy plumbing) → T007 (rollout rewrite) → T009/T010 (bench measurements). T008 (bench script) is parallel to T006/T007 (different file). T011 is parallel to T009/T010 (different machine class).
- **Phase 4 (US2)**: Depends on Phase 2. T012 (startup log) and T013 (metadata field) are independent of US1's `ppo.py` rewrite (they touch different code paths). Can start as soon as Phase 2 is done. T014/T015 (docs) are parallel to T012/T013. T016 (smoke test) depends on T012.
- **Phase 5 (US3)**: Depends on Phase 2. T017 (load `map_location`) is independent of US1/US2. T018, T019, T020 are parallel to each other (different files). T021 depends on T008 (bench script) and T017 (load fix).
- **Phase 6 (Polish)**: Depends on all user-story phases.

### User-Story Independence

- **US1 ↔ US2**: Independent at the file level (US1 lives in `ppo.py` rollout; US2 lives in `ppo.py` `__init__` + `_runlog.py` + READMEs). Both depend on Phase 2.
- **US1 ↔ US3**: Independent at the file level (US3 touches `ppo.py:load`, tests, notebooks). T021 touches `bench_device.py` (US1 deliverable) — so T021 must come after T008.
- **US2 ↔ US3**: Independent. The `metadata.device` field is owned by US2; US3's notebook update reads it but does not write it.

If staffed in parallel: one person on US1 (T006–T011), one on US2 (T012–T016), one on US3 (T017–T020) once Phase 2 is in. T021 is a final integration step.

### Parallel Opportunities

```bash
# Within Phase 2 — must run sequentially (single file edits compound):
T003 → T004 → T005

# Within Phase 3 (US1) — T008 in parallel with T006 → T007:
T006 → T007                # PPOAgent shadow-copy + rollout rewrite (same file, sequential)
T008 [P]                   # bench_device.py (NEW file, parallel)
T009, T010                 # depend on T006/T007 + T008

# Within Phase 4 (US2):
T013 [P]                   # _runlog.py
T014 [P]                   # pendulum README
T015 [P]                   # car-racing README + CLAUDE.md
T012 → T016                # ppo.py log line, then smoke test

# Within Phase 5 (US3):
T017 [P]                   # ppo.py:load (independent)
T018 [P]                   # test_ppo.py runner header
T019 [P]                   # test_agent_interface.py C8
T020 [P]                   # analyze.ipynb (both notebooks)
T021                       # depends on T008 + T017
```

---

## Implementation Strategy

### MVP First (User Story 1 only)

1. Phase 1 (T001, T002) — verify state.
2. Phase 2 (T003 → T004 → T005) — `get_device()` policy.
3. Phase 3 (T006 → T007 → T008 [P] → T009 → T010 → T011 [P]) — rollout/update split + bench.
4. **STOP and validate**: SC-001 and SC-002 both PASS. The user's stated goal — "MPS should not be slower than CPU" — is met. This is shippable on its own.

### Incremental Delivery

After MVP:

1. Add US2 (T012–T016) — override docs + run-metadata field. Now participants can debug-override and `analyze.ipynb` has device attribution.
2. Add US3 (T017–T021) — cross-device load + multi-device tests. Now `pretrained/` artefacts are device-portable and CI-style multi-device checks exist.
3. Polish (T022–T024).

### Constitution-Compliance Notes

- **Article II (Two Paths, One Agent API):** the rollout/update split lives entirely inside `PPOAgent.train()`; the public `Agent` interface (`preprocess`, `predict`, `train`, `save`, `load`) is unchanged. SB3 path is untouched.
- **Article IV (Test-Verified Implementation):** every code change has a corresponding test or extends an existing one (T009, T010, T011, T016, T018, T019, T021).
- **Article V (Progressive Scaffolding):** the `# TODO 5` markers in `ppo.py:train()` and the `raise NotImplementedError` defaults on the `solutions`-branch merge are preserved; T007's edits live around the markers, not inside them.
- **Article VI (Fail-Safe):** `RL_WORKSHOP_DEVICE=cpu` is the documented escape hatch; auto-fallback via `PYTORCH_ENABLE_MPS_FALLBACK=1` (T004) handles unsupported MPS ops without participant intervention.
- **Article VII (Sim-to-Real):** `Agent.save` writes a device-agnostic `state_dict`; `Agent.load` (T017) reads with `map_location="cpu"` then constructs the agent on the local device. The Pi pipeline is unaffected.

---

## Notes

- T009 and T010 require an Apple Silicon Mac. If the implementer is not on one, mark the tasks as "delegated to a workshop leader for verification" and proceed; do not skip silently.
- T020 touches notebook JSON. Use `nbformat` or hand-edit the JSON cells; do not corrupt the notebook structure.
- The `solutions` branch must be updated in lockstep with `main` for this feature, per Article V — the `# -- YOUR CODE HERE --` blocks on `main` keep `raise NotImplementedError`, and the matching solution lines on `solutions` get the same shadow-copy plumbing as T006/T007. Track this as a follow-up commit on `solutions` after merge to `main`.
