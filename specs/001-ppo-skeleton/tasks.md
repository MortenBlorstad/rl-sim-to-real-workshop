---

description: "Task list for PPO Skeleton with Per-TODO Tests"
---

# Tasks: PPO Skeleton with Per-TODO Tests

**Input**: Design documents from `specs/001-ppo-skeleton/`
**Prerequisites**: plan.md, spec.md, research.md, data-model.md, contracts/, quickstart.md

**Tests**: This feature explicitly requires tests (Constitution Article IV + spec FR-010 through FR-024). Test tasks are mandatory, not optional.

**Organization**: Tasks are grouped by user story (US1–US4) so each story can land as an independent increment.

## Format: `[ID] [P?] [Story] Description`

- **[P]**: Can run in parallel (different files, no dependencies on incomplete tasks)
- **[Story]**: Which user story this task serves (US1, US2, US3, US4); omitted for Setup, Foundational, and Polish phases
- All file paths are repo-relative

## Path conventions

- Skeleton lives in `workshop-1/1-ppo/` (see plan.md → Project Structure)
- Subclass stubs live in `workshop-1/2-mountaincar/` and `workshop-1/3-car-racing/`
- Solutions for the per-TODO recovery checkpoints live on the `solutions` git branch (NOT in this branch's working tree)

---

## Phase 1: Setup (Shared Infrastructure)

**Purpose**: Project initialization and dependency wiring.

- [ ] T001 Update `pyproject.toml` so the `[dependency-groups]` `workshop1` group declares `torch >= 2.1`, `gymnasium >= 0.29`, `numpy`, and `opencv-python` (the last only because the stage-3 stub in `workshop-1/3-car-racing/agent.py` imports `cv2`)
- [ ] T002 Run `uv sync --group workshop1` from the repo root and commit the resulting `uv.lock` so reproducible installs work for all participants (Constitution Article VIII)
- [ ] T003 [P] Verify `.gitignore` already excludes `*.pt`, `__pycache__/`, and `.venv/` — add any missing entries; this prevents stray model artifacts from landing on `main`

---

## Phase 2: Foundational (Blocking Prerequisites)

**Purpose**: Build the non-TODO portions of `ppo_skeleton.py` so every later phase has a working foundation. These tasks MUST complete before any user-story work begins.

**⚠️ CRITICAL**: At the end of Phase 2 the file imports cleanly, every TODO raises `NotImplementedError`, the `__main__` block crashes loudly with `NotImplementedError: TODO 1: ...`, and importing the module has zero side effects.

- [ ] T004 Create `workshop-1/1-ppo/ppo_skeleton.py` with module-level imports (`numpy`, `torch`, `torch.nn`, `torch.nn.functional as F`, `torch.distributions`, `gymnasium as gym`, `random`), the `DEFAULT_SEED = 42` constant, the `_AGENT_REGISTRY: dict[str, type] = {}` dict, and the `register_agent(cls)` decorator described in `data-model.md` §3
- [ ] T005 [P] Implement the `ActorNetwork(nn.Module)` class in `workshop-1/1-ppo/ppo_skeleton.py` per `data-model.md` §2.1 and `research.md` R2 — a 2-layer MLP with hidden size 64, Tanh activations, orthogonal init (gain `sqrt(2)` for hidden layers, `0.01` for the output layer)
- [ ] T006 [P] Implement the `ValueNetwork(nn.Module)` class in `workshop-1/1-ppo/ppo_skeleton.py` per `data-model.md` §2.2 — same 2×64 Tanh MLP shape as `ActorNetwork` but with a single scalar output and orthogonal init gain `1.0` on the output layer
- [ ] T007 Implement the `RolloutBuffer` class in `workshop-1/1-ppo/ppo_skeleton.py` per `data-model.md` §2.3 — pre-allocated NumPy arrays of length `size`, `add(...)`, `compute_returns_and_advantages(last_value, gamma, gae_lambda)` (which calls `compute_gae` — fine to reference the TODO function name even though it raises until TODO 1 is implemented), `get_batches(batch_size)` yielding shuffled minibatches as dicts, and `reset()`
- [ ] T008 Implement the `PPOAgent` class in `workshop-1/1-ppo/ppo_skeleton.py` per `data-model.md` §3 and `contracts/agent-interface.md` — decorated with `@register_agent`; constructor builds owned `ActorNetwork`, `ValueNetwork`, and `log_std = nn.Parameter(torch.zeros(action_dim))`; implements `preprocess` (identity), `predict` (calls `self.preprocess` then `sample_action`), `train` (delegates to module-level `train`), `save`/`load` per the schema in `research.md` R5, and the `_get_preprocess_state` / `_set_preprocess_state` extension hooks (base returns `{}` / no-op)
- [ ] T009 Add the five TODO function stubs at module top level in `workshop-1/1-ppo/ppo_skeleton.py` per `contracts/todo-functions.md`: `compute_gae`, `sample_action`, `evaluate_actions`, `ppo_loss`, `train`. Each stub MUST have the exact signature from the contract, a docstring describing the required behavior, `# Hint:` comments with formulas and expected shapes, the `# -- YOUR CODE HERE --` / `# -- END YOUR CODE --` markers, and `raise NotImplementedError("TODO N: <description>")` as the body. Order strictly by dependency (1 → 5)
- [ ] T010 Add the `if __name__ == "__main__":` block at the bottom of `workshop-1/1-ppo/ppo_skeleton.py` that seeds `random` / `numpy` / `torch` / the env with `DEFAULT_SEED`, builds `MountainCarContinuous-v0`, instantiates `ActorNetwork` / `ValueNetwork` / `log_std`, calls `train(...)` with the defaults from `research.md` R3 (`total_timesteps=8192`, `rollout_size=1024`, etc.), captures the printed losses (or the returned stats dict), and on exit verifies FR-027 invariants (no NaN, last `policy_loss` < first `policy_loss`); on failure prints `FAIL: <reason>` and exits non-zero, on success prints `✓ Training complete: loss trending down (X → Y), no NaN losses.`

**Checkpoint**: `uv run python workshop-1/1-ppo/ppo_skeleton.py` crashes with `NotImplementedError: TODO 1: compute generalized advantage estimation`. `python -c "import workshop_1.one_ppo.ppo_skeleton"` (or equivalent) succeeds with zero side effects.

---

## Phase 3: User Story 1 — Single TODO + per-step test (Priority: P1) 🎯 MVP

**Goal**: A participant can implement TODO 1 (GAE) in `ppo_skeleton.py`, run `test_ppo.py --step 1`, and get either a clear `TODO 1 OK!` or an actionable `FAIL: ...` line within 10 seconds. Other TODOs being unimplemented MUST NOT break this loop.

**Independent Test**: Run `uv run python workshop-1/1-ppo/test_ppo.py --step 1` against (a) the unmodified skeleton, (b) a deliberately-broken `compute_gae` (e.g., wrong shape), (c) a known-good `compute_gae`. Observe `NOT_IMPLEMENTED`, `FAIL: ...`, `TODO 1 OK!` respectively, with the exit code matching `contracts/test-runner-cli.md`.

### Implementation for User Story 1

- [ ] T011 [US1] Create `workshop-1/1-ppo/test_ppo.py` with the runner infrastructure per `contracts/test-runner-cli.md` and `research.md` R4: the `STEPS: dict[int, tuple[str, callable]] = {}` registry, the `step(n, name)` decorator, the `_run_step(n)` wrapper that distinguishes `PASS` / `NOT_IMPLEMENTED` / `FAIL` by exception type and prints the prescribed lines, the summary printer (`=== Summary: X / 5 passed ===` block), and the `argparse`-based CLI with optional `--step N` and the exit-code rules from the contract (0 if all PASS, 1 otherwise, 2 on argparse error)
- [ ] T012 [US1] Implement `test_step_1` (registered via `@step(1, "GAE")`) in `workshop-1/1-ppo/test_ppo.py` per `contracts/todo-functions.md` TODO 1 invariants: local import of `compute_gae`, hand-computed reference for `rewards = [1, 1, 1, 1]`, `values = [0.5, 0.6, 0.7, 0.8, 0.9]`, `dones = [0, 0, 0, 0]` with `gamma=0.99, lam=0.95`, assert each output element matches within `atol=1e-5`, plus a second assertion for `dones = [0, 0, 1, 0]` verifying the bootstrap is cut at the reset boundary, plus shape and dtype assertions. Use the local-import isolation pattern from `research.md` R4
- [ ] T013 [P] [US1] Verify `workshop-1/README.md` already documents `uv run python 1-ppo/test_ppo.py --step 1` for stage 1 (it does); leave the README otherwise untouched

**Checkpoint**: Phase 3 deliverable is independently shippable. A participant who only solves TODO 1 has a complete feedback loop. The other four `--step N` invocations report `NOT_IMPLEMENTED` until US2 lands their tests.

---

## Phase 4: User Story 2 — All five TODOs + script run (Priority: P1)

**Goal**: A participant can complete TODOs 1–5 in order, run `test_ppo.py` with no flags to see all five steps pass, and run `ppo_skeleton.py` to see the training loop execute on MountainCarContinuous-v0 with a downward loss trend printed across 8 update lines.

**Independent Test**: Against a known-good reference implementation of all 5 TODOs, run `uv run python workshop-1/1-ppo/test_ppo.py` (expect `5 / 5 passed`) and `uv run python workshop-1/1-ppo/ppo_skeleton.py` (expect 8 update lines, `policy_loss` strictly decreasing from first to last, exit code 0).

### Implementation for User Story 2

- [ ] T014 [P] [US2] Implement `test_step_2` (registered via `@step(2, "sample action")`) in `workshop-1/1-ppo/test_ppo.py` per `contracts/todo-functions.md` TODO 2 invariants: local import of `sample_action` and `ActorNetwork`, build a tiny seeded actor + `log_std`, sample 1000 stochastic actions and assert variance > 0, sample 100 deterministic actions and assert variance == 0, assert single-obs and batched-obs shapes, dtype, action range `[-1, 1]`, and finite `log_prob`
- [ ] T015 [P] [US2] Implement `test_step_3` (registered via `@step(3, "evaluate actions")`) in `workshop-1/1-ppo/test_ppo.py` per `contracts/todo-functions.md` TODO 3 invariants: local import of `evaluate_actions` and `ActorNetwork`, build an actor + `log_std`, build a batch of `(obs, action)`, compute the participant's `(log_probs, entropy)`, and compare element-wise within `atol=1e-5` against `torch.distributions.Normal(mean, log_std.exp()).log_prob(actions).sum(-1)` and `.entropy().sum(-1)`. Assert shapes `(B,)`
- [ ] T016 [P] [US2] Implement `test_step_4` (registered via `@step(4, "PPO loss")`) in `workshop-1/1-ppo/test_ppo.py` per `contracts/todo-functions.md` TODO 4 invariants: local import of `ppo_loss`, scalar-tensor and `requires_grad` assertions, the unclipped-branch test at `ratio = 1.0` asserting `out == -advantages.mean()`, and the clipped-branch tests for both positive and negative advantages forcing the ratio outside `[1 - clip_eps, 1 + clip_eps]` and asserting that the gradient with respect to `new_log_probs` is 0 on the clipped side
- [ ] T017 [US2] Implement `test_step_5` (registered via `@step(5, "training loop")`) in `workshop-1/1-ppo/test_ppo.py` per `contracts/todo-functions.md` TODO 5 invariants: local import of `train`, `ActorNetwork`, `ValueNetwork`, build a small actor/value/`log_std`, build `gym.make("MountainCarContinuous-v0")`, call `train(env, actor, value, log_std, total_timesteps=512, rollout_size=256, ...)`, assert the returned dict contains `mean_reward`, `policy_loss`, `value_loss`, `entropy`, `n_updates`, assert no captured loss is NaN, assert wall-clock runtime under 10 seconds (use `time.perf_counter()`)

**Checkpoint**: `uv run python workshop-1/1-ppo/test_ppo.py` runs all five steps (against the reference impl on the `solutions` branch from Phase 6, OR by manually filling TODOs locally during development) and prints `=== Summary: 5 / 5 passed ===`. `uv run python workshop-1/1-ppo/ppo_skeleton.py` prints 8 update lines and exits 0.

---

## Phase 5: User Story 3 — PPOAgent Article II contract test (Priority: P1)

**Goal**: A participant who saved their `PPOAgent` at the end of Workshop 1 can `PPOAgent.load(path)` on a different machine and call `predict(raw_obs)` immediately. The Constitution Article II contract is verified by an automated test.

**Independent Test**: Run `uv run python workshop-1/1-ppo/test_agent_interface.py --agent ppo` against a complete reference implementation. Expect every contract test to PASS. Then deliberately break `_set_preprocess_state` to ignore its input and re-run; expect `test_save_load_round_trip` (or the override-restored variant) to FAIL with an actionable message.

### Implementation for User Story 3

- [ ] T018 [US3] Create `workshop-1/1-ppo/test_agent_interface.py` reusing the `@step` runner machinery (either `from test_ppo import step, _run_step, ...` or copy the ~20 lines locally), with an `--agent {ppo,sb3}` CLI argument that for this spec only accepts `ppo` (raise `SystemExit(2)` with a friendly message for `sb3` saying "Path B follow-up spec"). Wire CLI parsing and the summary printer the same way as `test_ppo.py`
- [ ] T019 [US3] Implement contract tests C1, C2, C3, C4, and C6 in `workshop-1/1-ppo/test_agent_interface.py` per `contracts/agent-interface.md`: `test_registry_contains_PPOAgent`, `test_preprocess_identity`, `test_preprocess_deterministic`, `test_predict_raw_obs_shape_dtype_range`, `test_predict_deterministic_flag`, `test_train_method_smoke` (build a tiny env, call `agent.train(env, total_timesteps=512)`, assert returned dict contains the required keys), and `test_get_set_preprocess_state_base_no_op`. Each registered as a `@step(n, name)` entry
- [ ] T020 [US3] Implement contract test C5 (save/load round-trip) in `workshop-1/1-ppo/test_agent_interface.py`: `test_save_load_roundtrip_base` saves a fresh `PPOAgent`, loads it via `PPOAgent.load(path)`, asserts `type(loaded).__name__ == "PPOAgent"` and asserts `loaded.predict(raw_obs, deterministic=True)` equals the original's. Then `test_save_load_roundtrip_subclass_class_restored` defines a local `_RoundTripAgent(PPOAgent)` decorated with `@register_agent` overriding `preprocess` with `obs.clip(-0.5, 0.5)`, saves it, loads via base `PPOAgent.load(path)`, asserts the loaded class name is `_RoundTripAgent` and that `loaded.preprocess(np.array([2.0, -2.0], dtype=np.float32))` returns the clipped values

**Checkpoint**: `uv run python workshop-1/1-ppo/test_agent_interface.py --agent ppo` reports all PASS against a reference implementation. `--agent sb3` exits 2 with the deferred-spec message.

---

## Phase 6: User Story 4 — Per-TODO recovery checkpoints (Priority: P2)

**Goal**: A participant stuck on TODO N can run `git checkout ws1-todoN-done -- workshop-1/1-ppo/ppo_skeleton.py` and recover a version where TODOs 1..N are solved and TODOs N+1..5 still raise `NotImplementedError`, so they can keep moving without losing work on unrelated TODOs.

**Independent Test**: From a clean checkout of `001-ppo-skeleton`, deliberately break TODO 3 in `ppo_skeleton.py`, run `git checkout ws1-todo3-done -- workshop-1/1-ppo/ppo_skeleton.py`, re-run `uv run python workshop-1/1-ppo/test_ppo.py`, and observe `TODO 1: PASS`, `TODO 2: PASS`, `TODO 3: PASS`, `TODO 4: NOT_IMPLEMENTED`, `TODO 5: NOT_IMPLEMENTED`.

### Implementation for User Story 4

> **Branch context**: tasks T022–T026 happen on the `solutions` branch, NOT on `001-ppo-skeleton`. Create the branch from the current `001-ppo-skeleton` HEAD (where Phase 2 has produced a fully working file with `NotImplementedError` bodies). Each commit fills in exactly one TODO body and is tagged before moving to the next.

- [ ] T021 [US4] Create the `solutions` branch from the current `001-ppo-skeleton` HEAD: `git checkout -b solutions` (locally; pushed in Polish T030)
- [ ] T022 [US4] On the `solutions` branch, replace ONLY the body of TODO 1 (`compute_gae`) in `workshop-1/1-ppo/ppo_skeleton.py` with the reference GAE implementation per `contracts/todo-functions.md`; keep TODOs 2–5 raising `NotImplementedError`. Commit (`git commit -m "Solution for TODO 1: GAE"`) and tag: `git tag ws1-todo1-done`. Verify with `uv run python workshop-1/1-ppo/test_ppo.py --step 1` → `TODO 1 OK!`
- [ ] T023 [US4] On the `solutions` branch, replace the body of TODO 2 (`sample_action`) in `workshop-1/1-ppo/ppo_skeleton.py` with the reference implementation; keep TODOs 3–5 raising `NotImplementedError`. Commit and tag `ws1-todo2-done`. Verify `--step 1` and `--step 2` both pass
- [ ] T024 [US4] On the `solutions` branch, replace the body of TODO 3 (`evaluate_actions`) in `workshop-1/1-ppo/ppo_skeleton.py` with the reference implementation; keep TODOs 4–5 raising `NotImplementedError`. Commit and tag `ws1-todo3-done`. Verify `--step 1`, `--step 2`, `--step 3` all pass
- [ ] T025 [US4] On the `solutions` branch, replace the body of TODO 4 (`ppo_loss`) in `workshop-1/1-ppo/ppo_skeleton.py` with the reference implementation; keep TODO 5 raising `NotImplementedError`. Commit and tag `ws1-todo4-done`. Verify `--step 1` through `--step 4` all pass
- [ ] T026 [US4] On the `solutions` branch, replace the body of TODO 5 (`train`) in `workshop-1/1-ppo/ppo_skeleton.py` with the reference implementation. Commit and tag `ws1-todo5-done`. Verify `uv run python workshop-1/1-ppo/test_ppo.py` reports `5 / 5 passed`. Verify `uv run python workshop-1/1-ppo/ppo_skeleton.py` runs ~1–3 minutes, prints 8 update lines, `policy_loss` is strictly decreasing, and exits 0
- [ ] T027 [US4] Update `workshop-1/README.md` "Stuck?" section to document the per-TODO recovery command pattern (`git checkout ws1-todoN-done -- workshop-1/1-ppo/ppo_skeleton.py`), with an explicit warning that participants should commit any unrelated work first because the path-scoped checkout will overwrite their copy of `ppo_skeleton.py`

**Checkpoint**: All five `ws1-todoN-done` tags exist locally on the `solutions` branch. The recovery dress rehearsal in `quickstart.md` Step 5 succeeds end-to-end.

---

## Phase 7: Polish & Cross-Cutting Concerns

**Purpose**: Stage 2/3 subclass stubs, end-to-end dress rehearsal, and pushing the new branch + tags.

- [ ] T028 [P] Create `workshop-1/2-mountaincar/agent.py` per `data-model.md` §4.1 — a minimal `MountainCarPPOAgent(PPOAgent)` subclass decorated with `@register_agent`, importing `PPOAgent` and `register_agent` from the stage-1 module, with `preprocess` returning `obs` unchanged. Add a 5-line module docstring noting the file's purpose (override-contract demo + extension point for stage 2)
- [ ] T029 [P] Create `workshop-1/3-car-racing/agent.py` per `data-model.md` §4.2 — `CarRacingPPOAgent(PPOAgent)` decorated with `@register_agent`, with the full pixel preprocess pipeline (crop sky, grayscale, resize to 84×84, normalize, frame-stack 4) and the `_get_preprocess_state` / `_set_preprocess_state` overrides that persist `_frame_buffer`. Document explicitly that the stage-1 MLP `ActorNetwork` is NOT compatible with `(4, 84, 84)` observations and that end-to-end CarRacing training is the stage-3 spec's responsibility
- [ ] T030 Run the full `quickstart.md` dress rehearsal end-to-end on the `solutions` branch (Steps 1 through 7) on a clean machine state. Document any discrepancy as a defect to fix before merging. Required by Constitution Workshop Delivery Workflow ("dry-run every command in every README before each workshop release")
- [ ] T031 Push to GitHub: `git push -u origin solutions` followed by `git push origin ws1-todo1-done ws1-todo2-done ws1-todo3-done ws1-todo4-done ws1-todo5-done`. Open a PR from `001-ppo-skeleton` → `main` referencing this spec; do NOT merge automatically — wait for review

---

## Dependencies & Execution Order

### Phase dependencies

- **Setup (Phase 1)**: no dependencies — start immediately
- **Foundational (Phase 2)**: depends on Setup; **blocks every user story**
- **US1 (Phase 3)**: depends on Foundational; produces the MVP
- **US2 (Phase 4)**: depends on Foundational + US1's `test_ppo.py` runner skeleton (T011); the four new step tests can be developed in parallel
- **US3 (Phase 5)**: depends on Foundational + US1's runner machinery (T011) for the `@step` decorator and CLI plumbing; otherwise independent of US2
- **US4 (Phase 6)**: depends on Foundational (Phase 2 produces the file with `NotImplementedError` bodies that the `solutions` branch will progressively fill in). T022–T026 are sequential because each commits incrementally on the same branch and each tag must point at the right commit. Independent of US1–US3 in code, but the verification at each step relies on the test runner from T011/T012/T014/T015/T016/T017
- **Polish (Phase 7)**: depends on US1, US2, US3, US4 all being complete

### Within each user story

- US1: T011 (runner) → T012 (step 1 test); T013 can run any time
- US2: T014, T015, T016 in parallel; T017 last
- US3: T018 → T019, T020 in any order
- US4: T021 → T022 → T023 → T024 → T025 → T026 → T027 (strictly sequential because of the incremental tagging)
- Polish: T028, T029 in parallel; T030 after T026 (needs the full reference impl); T031 last

### Parallel opportunities

```bash
# Phase 2 — actor and critic networks are independent
T005 [P]  # ActorNetwork
T006 [P]  # ValueNetwork

# Phase 4 — independent test functions in the same file (different functions, no shared mutable state)
T014 [P]  # test_step_2 (sample action)
T015 [P]  # test_step_3 (evaluate actions)
T016 [P]  # test_step_4 (PPO loss)

# Phase 7 — independent stub files
T028 [P]  # 2-mountaincar/agent.py
T029 [P]  # 3-car-racing/agent.py
```

---

## Implementation strategy

### MVP first — US1 only

1. Phase 1: Setup
2. Phase 2: Foundational (the file imports cleanly with all TODOs as `NotImplementedError`)
3. Phase 3: US1 (test_ppo.py runner + step 1 test)
4. **Stop and validate**: `uv run python workshop-1/1-ppo/test_ppo.py --step 1` correctly reports `NOT_IMPLEMENTED` on a fresh skeleton, `FAIL: ...` on a deliberately-broken `compute_gae`, and `TODO 1 OK!` on a known-good `compute_gae`
5. **Workshop-shippable**: at this point a participant could already begin TODO 1 with a working feedback loop. The remaining four TODOs would be follow-up work, but the core teaching loop is live

### Incremental delivery

1. MVP: Phases 1–3 → ship US1
2. + Phase 4 → all five TODOs, full stage 1 PPO experience
3. + Phase 5 → Article II contract verified, Workshop 2 handoff path proven
4. + Phase 6 → catch-up mechanism live, the workshop is fail-safe
5. + Phase 7 → stage 2/3 subclass stubs in place, ready to PR to `main`

### Validation between phases

After each phase, run the corresponding section of `quickstart.md`:

- After Phase 2: Step 1 (the unmodified-skeleton failure messages)
- After Phase 3: Step 1 + Step 3 (TODO 1 only)
- After Phase 4: Steps 1–4 (full TODO loop + script run)
- After Phase 5: Step 2 + Step 6 (Article II + save/load round-trip)
- After Phase 6: Step 5 (per-TODO recovery)
- After Phase 7: Step 7 + the full Smoke-test checklist

---

## Notes

- [P] tasks live in different files OR in disjoint regions of the same file with no shared mutable state.
- [Story] labels map tasks to user stories for traceability and for `/speckit.taskstoissues` GitHub issue tagging.
- Each user story's checkpoint is independently verifiable using `quickstart.md`.
- Tests are mandatory in this spec (Constitution Article IV).
- Reference solutions live exclusively on the `solutions` branch — they MUST NOT land on `001-ppo-skeleton` or `main`.
- Do not commit `*.pt` model artifacts; `.gitignore` already excludes them.
- All user-facing strings (comments, prints, docstrings, error messages) MUST be in English (Constitution Article I, FR-050).
