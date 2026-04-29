# Feature Specification: Fix MountainCar Training Driver After PPO Refactor

**Feature Branch**: `003-fix-mountaincar-train`
**Created**: 2026-04-29
**Status**: Draft
**Input**: User description: "want to fix train.py for 2-mountaincar. I refactoring the ppo implementation. see 1-ppo/ppo. so make train.py, logging, imports fit with the refactoring. Also need to implement tests for the refactoring version in the /tests folder."

## Clarifications

### Session 2026-04-29

- Q: Where does the evaluation logic live? → A: Implement `PPOAgent.evaluate(env, n_episodes, record_video)` in `ppo.py` (replace the current `NotImplementedError`). The training driver calls it; stage-3 (CarRacing) and the SB3-equivalent path reuse it.
- Q: How does `--seed` propagate from the CLI? → A: No `--seed` CLI flag. The seed is set in code via `hyperparameters["random_state"]` defined at the top of `train.py`; the driver passes that dict into `PPOAgent` and forwards `random_state` to `agent.train(...)`. Multi-seed sweeps are run by editing the value (or by a small loop script).
- Q: How rigorous should the unit tests be? → A: Match the pre-refactor convention used by `workshop-1/1-ppo/test_ppo.py` and `test_agent_interface.py` (see commit `60321eb`): a custom `@step(n, name)` registry runner (no pytest); each test does local imports of just the symbols it needs so a single unfilled TODO doesn't cascade; `NotImplementedError` → `NOT_IMPLEMENTED`, `AssertionError` → `FAIL`; numeric rigor is "contract + hand-computed reference" (e.g. GAE matches a closed-form recurrence on a 4-step trajectory, `evaluate_actions` matches an independent `torch.distributions.Normal` reference, `ppo_loss` is checked at ratio=1 and on the clipped branches via gradient probes); `train()` is a < 10 s smoke run on `MountainCarContinuous-v0` with `total_timesteps=512`. Adapt the agent-contract test (`test_agent_interface.py`) to the refactored API: keep C1 (registry membership), C3 (predict shape/dtype/range, deterministic flag), C4 (train smoke), and C5 (save/load round-trip including subclass class restoration via `_AGENT_REGISTRY`). Drop C2 and C6 — `preprocess()` and `_get/_set_preprocess_state` no longer exist in the refactored API. Add a new test for `PPOAgent.evaluate()` (returns list of episode returns; with `record_video=True` writes a non-empty `.mp4` under `tmp_path`).
- Q: Where does observation normalization live for MountainCar? → A: Keep the `NormalizeObs(gym.ObservationWrapper)` defined at the top of `train.py` and apply it to the env before constructing `PPOAgent`. Driver-level wrappers are now allowed for the custom-PPO path (this supersedes the older "no wrappers" guidance, which was written when the agent had a `preprocess()` method). `PPOAgent` itself stays wrapper-agnostic; the agent's `obs_min`/`obs_max` fields are populated from the wrapped env (so they describe the post-normalization range).

## User Scenarios & Testing *(mandatory)*

### User Story 1 - Workshop participant trains a MountainCar agent end-to-end (Priority: P1)

A workshop participant has finished the PPO TODOs in stage 1 and now wants to apply their custom PPO implementation to MountainCarContinuous. They run the stage-2 training driver from the command line, watch the fixed-width log lines stream past, find the run directory under `runs/mountaincar/<run-name>/`, and confirm that `meta.json`, `metrics.jsonl`, and `model.pt` are all present.

**Why this priority**: This is the entire purpose of stage 2. If the training driver is broken (as it is today — incomplete `main()`, missing imports, wrong hyperparameter assignment), no participant can complete the stage. Everything else depends on this working.

**Independent Test**: Run `uv run python workshop-1/2-mountaincar/train.py --timesteps 4096 --run-name smoke` from the repo root. Training should complete without exceptions, print one update line per PPO iteration, and produce a `runs/mountaincar/smoke/` directory containing `meta.json`, `metrics.jsonl` (one line per update), and `model.pt`.

**Acceptance Scenarios**:

1. **Given** the participant has filled in the five PPO TODOs in `workshop-1/1-ppo/ppo/`, **When** they run `train.py` with default arguments, **Then** training proceeds to completion and produces a populated run directory under `runs/mountaincar/`.
2. **Given** a run directory with the chosen `--run-name` already exists, **When** they run `train.py` again without `--force`, **Then** the script exits with a clear error pointing them at `--force` or a different name (no data is overwritten).
3. **Given** they pass `--force`, **When** they run `train.py` with the same `--run-name`, **Then** the previous run directory is replaced and a fresh one is written.
4. **Given** training has completed, **When** the participant inspects `metrics.jsonl`, **Then** every line contains at least `update`, `timesteps`, `policy_loss`, `value_loss`, `entropy`, `mean_return`, `log_std_mean`, `grad_norm`, and `wall_time_seconds`.

---

### User Story 2 - Refactored PPO modules are covered by automated tests (Priority: P1)

A workshop maintainer (or a participant verifying their work) wants to run the test suite for the refactored PPO package and see that the public API surface — `RolloutBuffer`, `ActorNetwork`, `CriticNetwork`, and `PPOAgent` (`sample_action`, `evaluate_actions`, `ppo_loss`, `predict`, `evaluate`, `save`, `load`, `train`) — behaves correctly. The tests live alongside the package at `workshop-1/1-ppo/ppo/tests/` and run under `pytest`.

**Why this priority**: Stage 1 ships with two test files (`test_ppo.py`, `test_agent_interface.py`) that are currently empty (0 lines). Without tests, neither participants nor maintainers can detect regressions introduced by the refactor (and there are several already — see Edge Cases). Tests double as the regression net for fixing those bugs.

**Independent Test**: Run `uv run python workshop-1/1-ppo/ppo/tests/test_ppo.py` (all five TODOs) and `uv run python workshop-1/1-ppo/ppo/tests/test_agent_interface.py --agent ppo` (agent contract) and confirm that every step for a completed TODO reports `PASS`, and that any TODO still raising `NotImplementedError` reports `NOT_IMPLEMENTED` (not `FAIL`) without cascading into unrelated steps.

**Acceptance Scenarios**:

1. **Given** all five PPO TODOs are filled in correctly, **When** the maintainer runs both test files, **Then** every step reports `PASS` and the summary line shows `N / N passed`.
2. **Given** TODO 1 (`compute_gae`) is unfilled, **When** the maintainer runs `test_ppo.py`, **Then** step 1 reports `NOT_IMPLEMENTED` (not `FAIL`) and the other steps continue to run independently (TODO 5's smoke test may also report `NOT_IMPLEMENTED` because `train()` calls `compute_gae` — that cascade is acceptable; TODOs 2/3/4 must still run).
3. **Given** the agent's `save()` is called and then `PPOAgent.load()` is invoked on the resulting file, **When** a `predict()` is run on a fixed observation with `deterministic=True` before and after round-tripping, **Then** the two outputs are equal element-wise.
4. **Given** a short integration smoke test that calls `PPOAgent(env).train(env, total_timesteps=2048)`, **When** it runs, **Then** it returns a stats dict containing the documented keys (`mean_reward`, `policy_loss`, `value_loss`, `entropy`, `n_updates`) and does not raise.

---

### User Story 3 - Training metrics can be analysed in the existing notebook (Priority: P2)

After training, the participant opens `workshop-1/2-mountaincar/analyze.ipynb` (or any plotting workflow that reads `runs/mountaincar/<run-name>/metrics.jsonl`). The JSONL lines must conform to the schema documented in `specs/002-training-and-visualization/contracts/run-format.md` so existing downstream tooling keeps working without changes.

**Why this priority**: This isn't blocking the participant's first training run, but it determines whether the analysis half of the workshop continues to work. Lower priority because the analysis flow can be debugged after a successful training run.

**Independent Test**: Open the existing analysis notebook with the new run directory and confirm that loss curves, mean-return curves, and `log_std_mean` curves all render without `KeyError` or schema mismatches.

**Acceptance Scenarios**:

1. **Given** a finished run, **When** the analysis notebook reads `metrics.jsonl`, **Then** all expected metric columns are present and finite (no NaN/Inf — substituted with null on disk per the existing logger contract).
2. **Given** a finished run, **When** the notebook reads `meta.json`, **Then** it contains the same fields previously emitted by `RunLogger` (env_id, agent_class, seed, hyperparameters, git_sha, started_at, finished_at, status, library versions, metric_definitions).

---

### Edge Cases

- The PPO refactor introduced several latent bugs that this feature must surface and fix as part of making the driver work. Any tests written must catch them (now or in a regression sense):
  - `PPOAgent.train()` references `self.value` instead of `self.critic` (NameError at first update).
  - `PPOAgent.train()` calls `self.sample_action(self.actor, obs_t, self.log_std, ...)` and `self.evaluate_actions(batch["obs"], batch["actions"])`, but the refactored signatures are `sample_action(obs, deterministic=False)` and `evaluate_actions(actor, obs, actions, log_std)`.
  - `PPOAgent.__init__` constructs `nn.Parameter(..., device=self.device)`, but `nn.Parameter` does not accept a `device` keyword argument.
  - `PPOAgent.load()` calls `target_cls(hyperparameters=state["hyperparameters"])` without passing `env`, so loading any saved model raises `TypeError`.
  - `predict()` accepts a NumPy array per its docstring but forwards it to `sample_action`, which expects a `torch.Tensor`.
  - `train.py` has `hyperparameters = dict = {...}` (rebinds the builtin `dict` and is not what the author intended) and an undefined `exit_code`.
  - The PPO package has no `__init__.py` files at `ppo/`, `ppo/utils/`, or `ppo/tests/`, so `from ppo import PPOAgent` and `from ppo.utils import ...` are not yet importable as a package.
- A participant who has not yet completed TODO 1 (`compute_gae`) tries to run `train.py`. The driver should fail fast at update 1 with the original `NotImplementedError("TODO 1: ...")`, not after silently allocating an empty run directory.
- The participant runs `train.py` on a machine without a writable `runs/` directory or with the disk full. Per the existing `RunLogger` contract, JSONL writes are best-effort and the training loop must not crash.
- The participant interrupts training with Ctrl+C. The run directory must be left in a state where `meta.json` records `status: "interrupted"`.
- An evaluation episode is requested but the environment cannot be rendered (no display, missing ffmpeg). Behaviour for video recording when prerequisites are missing is captured by FR-009 below.

## Requirements *(mandatory)*

### Functional Requirements

- **FR-001**: `workshop-1/2-mountaincar/train.py` MUST run end-to-end with default arguments and produce a complete run directory (`meta.json`, `metrics.jsonl`, `model.pt`) without manual intervention beyond the documented CLI flags.
- **FR-002**: The training driver MUST import `PPOAgent`, `RolloutBuffer`, `seed_everything`, `RunLogger`, `RunDirectoryExistsError`, and `make_log_fn` from the refactored package layout under `workshop-1/1-ppo/ppo/` (no `sys.path` hacks, no copy-pasted helpers, no references to deleted top-level files like `_runlog.py` or `_log_parser.py`).
- **FR-003**: The training driver MUST instantiate `PPOAgent` with the configured environment and the documented hyperparameter dictionary, then call `agent.train(env, total_timesteps=..., log_fn=...)` exactly once.
- **FR-004**: The training driver MUST wrap training in a `RunLogger` context so that `meta.json` records `status: "ok"` on success, `status: "interrupted"` on `KeyboardInterrupt`, and `status: "error"` on uncaught exception.
- **FR-005**: The training driver MUST attach a `log_fn` (constructed via `make_log_fn`) so each PPO update produces one JSONL record with the metric keys listed in `specs/002-training-and-visualization/contracts/run-format.md` (`update`, `timesteps`, `policy_loss`, `value_loss`, `entropy`, `mean_return`, `log_std_mean`, `grad_norm`, `wall_time_seconds`).
- **FR-006**: The training driver MUST save the trained model via `agent.save(<run_dir>/model.pt)` after training completes, before closing the `RunLogger` and the environment.
- **FR-007**: The CLI MUST accept `--timesteps`, `--run-name`, `--no-eval`, and `--force` and behave per the existing stage-2 contract (force-overwrite, default env `MountainCarContinuous-v0`). It MUST NOT expose a `--seed` flag — the seed is configured in source via `hyperparameters["random_state"]` defined at the top of `train.py` (default `42`).
- **FR-007a**: The driver MUST pass the hyperparameter dict (including `random_state`) into `PPOAgent(env, hyperparameters=...)` and MUST forward `random_state=hyperparameters["random_state"]` to `agent.train(...)` so that network initialisation and the training loop use the same effective seed.
- **FR-007b**: The driver MUST keep the `NormalizeObs(gym.ObservationWrapper)` class defined at the top of `train.py` and apply it to `gym.make(ENV_ID)` before passing the env into `PPOAgent`. The agent does not normalize observations itself; normalization is the driver's responsibility for the custom-PPO path.
- **FR-008**: When `--no-eval` is not set, the training driver MUST call `agent.evaluate(env, n_episodes=1, record_video=True)` (with the eval target directory pointed at the active run directory) so that `eval.mp4` and a returned list of episode returns become available. The driver MUST NOT duplicate the evaluation loop inline — `PPOAgent.evaluate()` is the single source of truth.
- **FR-009**: If video recording prerequisites (e.g. ffmpeg) are missing, the driver MUST log a warning and continue rather than crashing — the training run itself is the artifact.
- **FR-010**: The refactored PPO package MUST be importable as `ppo` from `workshop-1/1-ppo/` (with `__init__.py` files at `ppo/`, `ppo/utils/`, and `ppo/tests/` as needed) and re-export the public symbols `PPOAgent`, `RolloutBuffer`, `ActorNetwork`, `CriticNetwork`, `seed_everything`, `format_update_line`, `RunLogger`, `RunDirectoryExistsError`, and `make_log_fn`.
- **FR-011**: The bugs listed in the Edge Cases section MUST be fixed in `ppo.py` so that `train()`, `predict()`, `evaluate()`, `save()`, and `load()` work without modification by participants. `PPOAgent.evaluate()` MUST be implemented (replacing the current `NotImplementedError`) and accept `(env, n_episodes: int = 10, record_video: bool = True, video_dir: str | Path | None = None)`, returning the list of per-episode returns. The five `# -- YOUR CODE HERE --` blocks remain the only intentional gaps.
- **FR-012**: `workshop-1/1-ppo/ppo/tests/test_ppo.py` MUST contain one registered step per PPO TODO (1: GAE, 2: sample_action, 3: evaluate_actions, 4: ppo_loss, 5: train smoke). It MUST follow the pre-refactor runner convention from commit `60321eb`: a `@step(n, name)` decorator + `STEPS` registry, local imports inside each test, `NotImplementedError → NOT_IMPLEMENTED`, `AssertionError → FAIL`, and a `--step N` CLI for running a single step. Numeric rigor matches the pre-refactor file: closed-form GAE reference (with and without a mid-trajectory `done`), `torch.distributions.Normal` reference for `evaluate_actions`, `ratio=1` and clipped-branch gradient probes for `ppo_loss`, and a `total_timesteps=512`, `< 10 s` smoke run for `train()` that asserts the returned stats dict shape and that no NaNs appear in the captured loss lines.
- **FR-013**: `workshop-1/1-ppo/ppo/tests/test_agent_interface.py` MUST follow the same step-runner convention with `--agent ppo` CLI, adapted to the refactored API. Required steps: C1 — `_AGENT_REGISTRY` contains `PPOAgent`; C3 — `predict` returns the right shape/dtype/range and is deterministic when `deterministic=True`; C4 — `train(total_timesteps=512)` smoke returns a stats dict with the documented keys; C5 — save/load round-trip on `PPOAgent` and on a registered subclass restores the class via `_AGENT_REGISTRY` lookup (use `tempfile.mkstemp(suffix=".pt")`); C7 (new) — `evaluate()` returns a list of length `n_episodes` of finite floats and, with `record_video=True`, writes a non-empty `.mp4` under `tmp_path`. C2 (preprocess identity) and C6 (`_get/_set_preprocess_state`) MUST NOT be ported — those entry points no longer exist after the refactor.
- **FR-014**: The two test files MUST each be runnable with a single command from the repo root: `uv run python workshop-1/1-ppo/ppo/tests/test_ppo.py [--step N]` and `uv run python workshop-1/1-ppo/ppo/tests/test_agent_interface.py --agent ppo`. Combined wall-clock for both (all steps) MUST be under 60 seconds on a developer laptop.
- **FR-015**: Test artifacts MUST NOT touch network, MUST NOT write into the participant's `runs/` directory (use `tempfile`/`tmp_path` for any file output), and MUST NOT depend on display/render (`render_mode="rgb_array"` only when video is being captured).

### Key Entities *(include if feature involves data)*

- **PPOAgent** — The refactored agent class living in `workshop-1/1-ppo/ppo/ppo.py`. Owns actor, critic, and `log_std`; provides `train`, `predict`, `evaluate`, `save`, `load`. Public surface is what `train.py` and the tests consume.
- **RolloutBuffer** — Pre-allocated trajectory buffer in `workshop-1/1-ppo/ppo/rollout_buffer.py`. Holds obs/actions/log_probs/rewards/dones/values and computes GAE.
- **RunLogger** — Per-run directory and JSONL writer in `workshop-1/1-ppo/ppo/utils/_runlog.py`. Owner of `meta.json` and `metrics.jsonl`. Schema is fixed by the existing run-format contract from feature 002.
- **Run directory** — On-disk artifact under `runs/mountaincar/<run-name>/` containing `meta.json`, `metrics.jsonl`, `model.pt`, and (optionally) `eval.mp4`.

## Success Criteria *(mandatory)*

### Measurable Outcomes

- **SC-001**: A workshop participant who has filled in the five PPO TODOs can run `workshop-1/2-mountaincar/train.py` with default arguments and obtain a populated `runs/mountaincar/<run-name>/` directory with no code changes outside the TODO blocks. Measured by: green smoke run on a fresh checkout of this branch.
- **SC-002**: The full test suite at `workshop-1/1-ppo/ppo/tests/` passes in under 60 seconds when all TODOs are completed correctly. Measured by: `uv run pytest` exit code and wall-clock time.
- **SC-003**: When any single TODO is reset to its `raise NotImplementedError(...)` state, the corresponding test(s) fail with a message that names the TODO and no other tests fail with unrelated cascades. Measured by: targeted regression script that re-introduces each `NotImplementedError` one at a time.
- **SC-004**: Trained agent reaches a `mean_return` ≥ 90 on `MountainCarContinuous-v0` within 200 000 timesteps using the default hyperparameters for at least 3 of 5 distinct values of `hyperparameters["random_state"]` (the documented stage-2 baseline). Measured by: a small wrapper script that edits the in-code `random_state` and re-invokes `train.py`, since there is no CLI seed flag.
- **SC-005**: Zero references in the stage-2 driver to any module path that no longer exists after the refactor (no `_runlog`, `_eval`, `_log_parser` at `workshop-1/` top level; no `ppp.py`, no flat-file `ppo.py` next to `2-mountaincar/`). Measured by: grep audit on the diff.
- **SC-006**: Stage-2 `analyze.ipynb` opens an existing run produced by the new driver and renders all standard plots without `KeyError`, `JSONDecodeError`, or schema mismatches. Measured by: notebook-level smoke (run-all on a freshly produced run directory).

## Assumptions

- Participants run all commands from the repo root using `uv run`, as documented in CLAUDE.md and the workshop READMEs.
- The on-disk schema for `meta.json` and `metrics.jsonl` is unchanged from feature 002 (`specs/002-training-and-visualization/contracts/run-format.md`); this feature only updates the import paths, not the schema.
- `MountainCarContinuous-v0` remains the default environment for stage 2; CartPole and CarRacing keep their own drivers in stages 2/3 (out of scope here).
- `imageio[ffmpeg]` is already installed via the workshop1 dependency group; if missing on a given machine, FR-009 covers the graceful-degrade path.
- Stable-Baselines3 is not exercised by this feature — the SB3 path lives in `train_sb3.py` and has its own callback. This feature is the custom-PPO path only.
- Tests use lightweight environments (`gym.make("MountainCarContinuous-v0")` or `Pendulum-v1`) and short rollouts so the suite stays under 60 seconds without GPU.
- The five `# -- YOUR CODE HERE --` blocks in `ppo.py` are the only intentional gaps; everything else (init, train scaffolding, save/load) must work out of the box. Bug fixes that restore that invariant are in scope; redesigning the public API is not.
