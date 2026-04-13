# Feature Specification: PPO Skeleton with Per-TODO Tests

**Feature Branch**: `001-ppo-skeleton`
**Created**: 2026-04-13
**Status**: Draft
**Input**: Build a single-file, beginner-facing PPO scaffold for Workshop 1, Stage 1, where participants implement five specific pieces of PPO (GAE, sample action, evaluate actions, clipped surrogate loss, training loop) as numbered TODO blocks. Each TODO must ship with one or more independent tests that tell the participant whether their implementation is correct. A companion `PPOAgent` class must implement the Constitution Article II `Agent` interface so the resulting model is usable in Workshop 2 without rewriting code.

## Clarifications

### Session 2026-04-13

- Q: Should `ppo_skeleton.py` target discrete actions, continuous actions, or both? → A: Continuous only (MountainCarContinuous-v0 / Normal distribution)
- Q: How much preprocessing machinery does Workshop 1's `PPOAgent` carry? → A: Base `PPOAgent` in `1-ppo/` has identity `preprocess()`; stages `2-mountaincar/` and `3-car-racing/` each define a subclass of `PPOAgent` that overrides `preprocess()` with environment-specific logic. Participants may further override preprocessing if they want different behavior.
- Q: Is `SB3Agent` (Path B) in scope for this spec? → A: No. Defer `SB3Agent` and `test_agent_interface.py --agent sb3` to a separate follow-up spec. This spec ships only `PPOAgent` and the `--agent ppo` test path.
- Q: What does running `ppo_skeleton.py` need to demonstrate? → A: Only that the training loop runs end-to-end without errors / NaN, and that the loss is trending downward across a handful of update steps. Convergence on MountainCarContinuous (reaching the flag, beating a random baseline) is explicitly the responsibility of stage `2-mountaincar/`, not this spec. The stage-1 default training run is short (a few minutes at most) and the `--step 5 --full` learning-assertion mode is dropped from this spec.
- Q: Does the skeleton ship a shared actor-critic trunk or two separate networks? → A: Two separate networks — `ActorNetwork` (outputs policy distribution parameters) and `ValueNetwork` (outputs state value). Both provided as complete helper code; no parameter sharing.

## User Scenarios & Testing *(mandatory)*

### User Story 1 - A participant implements one TODO and verifies it in under a minute (Priority: P1)

A workshop participant opens `ppo_skeleton.py`, reads the description and hints for TODO 1 (GAE), writes their implementation in the marked YOUR-CODE block, runs the dedicated test for that step, and either sees a clear success message or a concrete failure message that tells them what to fix.

**Why this priority**: This is the tightest loop in the workshop. If this loop is slow, noisy, or unreliable, the entire 3-hour session collapses. It is also the smallest viable slice of the feature: if only TODO 1 + its test works, the workshop can still begin on time.

**Independent Test**: Can be fully tested by running `uv run python workshop-1/1-ppo/test_ppo.py --step 1` against a fresh skeleton (where every TODO raises `NotImplementedError`), then against a correct reference implementation, then against several common buggy implementations, and observing the prescribed output in each case.

**Acceptance Scenarios**:

1. **Given** an unmodified skeleton (all TODOs raise `NotImplementedError`), **When** the participant runs `--step 1`, **Then** the runner reports "TODO 1 not yet implemented" and exits non-zero.
2. **Given** a correct TODO 1 implementation, **When** the participant runs `--step 1`, **Then** the runner prints "TODO 1 OK!" and exits zero — regardless of whether TODO 2–5 are still unfinished.
3. **Given** a buggy TODO 1 implementation that returns the wrong shape, **When** the participant runs `--step 1`, **Then** the runner prints a message naming the symbol, the expected shape, and the observed shape, and exits non-zero.
4. **Given** a correct TODO 1 implementation and `--step 1`, **Then** the test completes in under 10 seconds on a standard laptop CPU.

---

### User Story 2 - A participant completes all five TODOs and sees the training loop run end-to-end (Priority: P1)

A participant works through TODOs 1–5 in order, running `--step N` after each one, then runs the full skeleton (`uv run python ppo_skeleton.py`) and watches the training loop execute on MountainCarContinuous-v0: episodes roll out, updates fire, and the printed losses trend downward across updates without any errors or NaN values. Whether the agent actually solves the environment is *not* tested here — that is what stage `2-mountaincar/` is for.

**Why this priority**: Reaching a working training loop is the headline moment of stage 1. It is the proof that all five TODOs fit together. Solving MountainCarContinuous is a separate, later milestone in stage 2.

**Independent Test**: Can be fully tested by starting from the skeleton, filling in a known-good reference solution for all five TODOs, running `test_ppo.py` (no flags) and observing all five steps pass, then running `ppo_skeleton.py` and observing that the training loop completes its configured timestep budget, that no loss value is NaN, and that the printed loss values show a downward trend across updates.

**Acceptance Scenarios**:

1. **Given** a complete reference implementation, **When** the participant runs `test_ppo.py` with no flags, **Then** all five steps report OK and the runner exits zero.
2. **Given** a complete reference implementation, **When** the participant runs `ppo_skeleton.py` on MountainCarContinuous-v0, **Then** training completes without error, no printed loss is NaN, and the printed loss values show a clear downward trend (the last printed loss is lower than the first).
3. **Given** a complete reference implementation, **When** the participant runs `test_ppo.py --step 5` (smoke mode), **Then** the smoke test completes in under 10 seconds and reports the expected training-stats dictionary shape, asserting only that the loop ran without producing NaN losses (not that the agent learned).

---

### User Story 3 - A participant who finishes Workshop 1 with PPOAgent reuses their saved model in Workshop 2 (Priority: P1)

A participant saves their trained `PPOAgent` to disk at the end of Workshop 1, and in Workshop 2 they load it via `Agent.load(path)` and call `agent.predict(raw_obs)` on a DonkeyCar observation without writing any additional preprocessing code.

**Why this priority**: Constitution Article II makes this interoperability non-negotiable. Without it, Workshop 2 cannot hand off from Workshop 1.

**Independent Test**: Can be fully tested by running `test_agent_interface.py --agent ppo` against the skeleton's `PPOAgent` and confirming that save/load round-trip and `predict(raw_obs)` contract tests pass.

**Acceptance Scenarios**:

1. **Given** a trained `PPOAgent`, **When** `agent.save(path)` is called followed by `PPOAgent.load(path)`, **Then** the loaded agent produces the same action as the original for an identical raw observation.
2. **Given** a loaded `PPOAgent`, **When** `agent.predict(raw_obs)` is called with a raw environment observation, **Then** the agent returns an action without the caller having invoked any preprocessing code.
3. **Given** the skeleton, **When** `test_agent_interface.py --agent ppo` is run, **Then** all contract tests pass.

---

### User Story 4 - A participant falls behind and recovers via the solutions checkpoint for TODO N (Priority: P2)

A participant gets stuck on TODO 3, runs out of time, and needs to jump past it to keep up with the rest of the room. They check out the per-TODO solutions checkpoint for TODO 3 and continue with TODO 4 on top of the recovered code.

**Why this priority**: Required by Constitution Article VI (fail-safe design). Lower priority than US1–US3 because the workshop can still run with a coarser recovery mechanism, but without it a stuck participant is blocked entirely.

**Independent Test**: Can be fully tested by deliberately leaving TODO 3 broken, checking out the TODO 3 solution via the documented git command, and confirming that `test_ppo.py --step 3` now passes.

**Acceptance Scenarios**:

1. **Given** an incorrect or missing TODO 3 implementation, **When** the participant runs the documented solutions-branch checkout command for TODO 3, **Then** the file is updated and `test_ppo.py --step 3` passes.
2. **Given** a recovered skeleton, **When** the participant continues with TODO 4, **Then** their TODO 4 work is not overwritten by the TODO 3 recovery.

---

### Edge Cases

- What happens when a participant runs `--step 2` while TODO 1 still raises `NotImplementedError`? The runner must report TODO 2's result without tripping on TODO 1 — every `--step N` must be independent of the state of any other TODO.
- What happens when the participant's Python environment does not have the required dependencies installed? The runner must print an actionable error pointing at the documented `uv sync --group workshop1` command.
- What happens when a participant's TODO 4 implementation silently returns a non-scalar loss? The TODO 4 test must catch this and report it as a shape failure, not fail cryptically inside the training loop.
- What happens when a participant's TODO 5 training loop diverges (NaN loss)? The smoke test must detect NaN and report "loss is NaN" as the failure reason, not let the test silently hang or crash.
- What happens when tests are run on a machine with no GPU? All tests must run on CPU within the 10-second cap.
- What happens when a participant runs `test_ppo.py` with no arguments? All five steps must run and a summary must report each step's pass/fail status and the total count.

## Requirements *(mandatory)*

### Functional Requirements

#### Skeleton file

- **FR-001**: The feature MUST deliver a single file at `workshop-1/1-ppo/ppo_skeleton.py` containing exactly five numbered TODO blocks corresponding to GAE, sample action, evaluate actions, PPO clipped surrogate loss, and the training loop.
- **FR-002**: Each TODO block MUST include a numbered label (`# TODO N: <what to implement>`), a description of expected behavior, hints containing formulas or variable names or expected shapes, a default body that raises `NotImplementedError("TODO N: <description>")`, and explicit `# -- YOUR CODE HERE --` and `# -- END YOUR CODE --` markers.
- **FR-003**: The skeleton MUST provide fully working helper code so that participants only write RL logic inside the five TODO blocks. The provided helpers MUST include, at minimum: an `ActorNetwork` class (outputs the parameters of a `Normal` distribution over actions for a continuous action space), a separate `ValueNetwork` class (outputs a scalar state value), a `RolloutBuffer` for collecting trajectories, environment construction, and per-update loss/return logging. Actor and critic MUST be separate networks with no parameter sharing.
- **FR-004**: Every symbol that a TODO defines MUST be exported at module top level so tests can import it directly by name without traversing private helpers.
- **FR-005**: Any execution of the training loop or any other side-effectful code MUST be guarded by `if __name__ == "__main__":` so that importing the module for testing has no side effects.
- **FR-006**: TODOs MUST be ordered by dependency: TODO N may call code from TODO < N, but never the reverse.
- **FR-007**: The skeleton MUST also define a `PPOAgent` class that implements the Constitution Article II `Agent` contract (`preprocess`, `predict`, `train`, `save`, `load`) by delegating to the five TODO functions, such that a trained `PPOAgent` can be saved and later loaded on a different machine and immediately accept raw observations via `predict()`.
- **FR-008**: `PPOAgent.preprocess()` MUST be the identity function (returns `obs` unchanged) in the base class. The method MUST be a regular instance method that subclasses can override without touching any other part of `PPOAgent`. `predict()` and the training rollout loop MUST call `self.preprocess()` (not a bare identity) so that overrides in subclasses take effect automatically.
- **FR-009**: The repository MUST ship two `PPOAgent` subclasses to demonstrate and exercise the override pattern: one in `workshop-1/2-mountaincar/` for MountainCarContinuous-v0 (vector observations; preprocess MAY remain identity or apply normalization) and one in `workshop-1/3-car-racing/` for CarRacing-v3 (pixel observations; preprocess MUST apply the cropping/grayscale/resize/normalize/frame-stack pipeline described by the constitution). Each subclass MUST be runnable via its own stage entry point. The detailed implementation of these subclasses is the subject of separate stage features and is out of scope for this spec; this requirement only fixes their existence and the override contract they MUST honor.

#### Test runner

- **FR-010**: The feature MUST deliver a test runner at `workshop-1/1-ppo/test_ppo.py` that supports running one specific TODO step via `--step N` and running all steps when invoked with no arguments.
- **FR-011**: Each `--step N` invocation MUST be independent: running `--step 1` MUST succeed when TODOs 2–5 are unfinished, and likewise for every other step.
- **FR-012**: Each test step MUST complete in under 10 seconds on a standard laptop CPU. No full training runs are permitted inside `--step N` in default mode.
- **FR-013**: On success, each step MUST print a human-readable success line in the form `TODO N OK!`. On failure, each step MUST print an actionable failure line that names the expected and observed values (e.g., `FAIL: expected shape (5,), got (5, 1)`).
- **FR-014**: When a TODO step is still unfinished (the step's `NotImplementedError` fires), the runner MUST report "TODO N not yet implemented" rather than re-raising the exception as an unhandled crash.
- **FR-015**: The runner MUST return a non-zero exit code when any step invoked in the current run fails, and zero when every invoked step passes.
- **FR-016**: When invoked with no arguments, the runner MUST print a summary at the end containing the pass/fail status of every step and the total pass count.

#### Per-TODO test coverage

- **FR-020**: The TODO 1 (GAE) test MUST validate the returned advantages against a hand-computed reference for a small toy trajectory, using an absolute tolerance rather than exact float equality, and MUST cover at least one trajectory with a mid-trajectory `done=True`.
- **FR-021**: The TODO 2 (sample action) test MUST validate that the action returned has the same shape and dtype as a `Box(1,)` continuous action in the `[-1, 1]` range, MUST confirm that stochastic sampling produces variability across repeated calls with the same input, and MUST confirm that a `deterministic=True` flag produces the same (mean) action on repeated calls for the same input.
- **FR-022**: The TODO 3 (evaluate actions) test MUST validate that the returned `log_prob` and `entropy` values match an independent Normal-distribution reference implementation element-wise within tolerance, for a batch of continuous actions.
- **FR-023**: The TODO 4 (PPO loss) test MUST validate the unclipped branch at ratio 1.0, the clipped branch at a ratio outside the clip window, and MUST confirm the loss is a scalar tensor that carries gradient information.
- **FR-024**: The TODO 5 (training loop) test in default mode MUST be a smoke test that runs a short MountainCarContinuous-v0 training run (on the order of a few hundred steps) and asserts that the returned training-statistics dictionary has the expected keys, that no loss is NaN, and that the run completes in under 10 seconds. The smoke test does NOT assert learning — convergence on MountainCarContinuous is verified in stage `2-mountaincar/`, not here.
- **FR-025**: The PPO skeleton MUST target a **continuous** action space using the `Normal` distribution. The default training environment in `if __name__ == "__main__":` MUST be `MountainCarContinuous-v0`. Discrete-action support (Categorical distribution) is explicitly out of scope.
- **FR-026**: When `ppo_skeleton.py` is run as a script (`uv run python ppo_skeleton.py`), it MUST execute a short training loop (a few minutes at most on a standard laptop CPU), print the training loss after each update, and exit cleanly. The default budget MUST be small enough that participants can run it during the 60-minute PPO block but large enough that the printed losses span at least a handful of update steps so a trend is visible. The default budget MAY be exposed as a CLI flag (`--timesteps`) but a sensible hard-coded default that meets these constraints is sufficient.
- **FR-027**: When run as a script, the skeleton MUST verify on exit that (a) no printed `policy_loss` is NaN, (b) no printed `entropy` is NaN, and (c) the last printed `entropy` is strictly less than the first printed `entropy`. If any of these checks fails it MUST print an actionable error message naming the failure. This is the script-level equivalent of the smoke test. Note: PPO's clipped surrogate `policy_loss` is not a supervised loss — it is not expected to decrease monotonically — so the trend check uses `entropy` instead, which is the canonical monotonic signal in PPO (entropy goes down as the policy commits to actions).

#### Agent interface integration

- **FR-030**: The repository MUST include a shared `workshop-1/1-ppo/test_agent_interface.py` that verifies the Article II contract for `PPOAgent`. For the base class (`--agent ppo`) the test MUST cover: preprocess behaves as identity for a vector observation, preprocess is deterministic (same input → same output), `predict(raw_obs)` works end-to-end without external preprocessing, save/load round-trip equality (the loaded agent produces the same action as the original for an identical raw observation), and the override contract (a trivial subclass overriding `preprocess()` with a known transform causes `predict()` to use that transform). Pixel-pipeline behavior is verified in the per-stage tests for the `2-mountaincar/` and `3-car-racing/` subclasses, not here.
- **FR-031**: `PPOAgent.save` MUST persist both model weights and any preprocessing configuration that subclasses register, into a single file. `PPOAgent.load` MUST return an agent of the correct (sub)class ready to call `predict(raw_obs)` with no additional setup. The save/load mechanism MUST be designed so that a subclass can declare additional state to persist (e.g., a frame-stack buffer config or normalization statistics) without modifying the base class — for example, by serializing `type(self).__name__` and a subclass-provided state dict alongside the model weights.

#### Catch-up mechanism

- **FR-040**: The repository MUST provide a documented per-TODO recovery path so that a participant stuck on TODO N can check out a reference implementation scoped to that TODO without losing their work on other TODOs. The exact mechanism (solutions branch + per-TODO git tags, or equivalent) MUST be described in `workshop-1/README.md`.

#### Documentation

- **FR-050**: Every user-facing string in `ppo_skeleton.py` and `test_ppo.py` (comments, print statements, docstrings, error messages) MUST be in English (Constitution Article I).
- **FR-051**: The `workshop-1/README.md` step-by-step runbook for Stage 1 MUST tell the participant how to run each `--step N` command and how to run `test_ppo.py` with no arguments. (This is already in place from the current README and only needs verification during implementation.)

### Key Entities

- **TODO Block**: A numbered, self-contained gap in the skeleton. Has a number (1–5), a description, hints, markers, and a matching step in the test runner.
- **Test Step**: A named, independently runnable unit in the test runner, keyed by TODO number, with a success message, one or more failure messages, and a time budget (≤ 10 s in default mode).
- **ActorNetwork**: Provided helper. Maps an observation to the parameters (mean and standard deviation) of a `Normal` distribution over actions in `Box(1,)`, `[-1, 1]`. No parameter sharing with the value network.
- **ValueNetwork**: Provided helper. Maps an observation to a scalar state-value estimate. Separate from `ActorNetwork`.
- **PPOAgent**: A class that wraps the five TODO functions in the Article II `Agent` interface, persistable to a single file, loadable on any machine including the Raspberry Pi. Owns one `ActorNetwork` and one `ValueNetwork`. `preprocess()` is identity in the base class; subclasses override it.
- **Solutions Checkpoint**: A recoverable state (e.g., a git tag and branch combination) that represents a known-good implementation of TODO N, reachable via a single documented command.

## Success Criteria *(mandatory)*

### Measurable Outcomes

- **SC-001**: In a live workshop of 15–30 participants, at least 90 percent of participants who complete TODO 1 and run `--step 1` see either the success message or an actionable failure message within 10 seconds of running the command.
- **SC-002**: At least 80 percent of participants reach a running training loop (TODO 5 smoke test passing AND `ppo_skeleton.py` executing end-to-end with a downward loss trend visible in the printed output) by the end of the 60-minute PPO-implementation block. Solving MountainCarContinuous is NOT a stage-1 success criterion.
- **SC-003**: A participant who gets stuck on exactly one TODO can recover using the documented per-TODO checkpoint mechanism in under 2 minutes, without losing their work on any other TODO, in at least 95 percent of attempts.
- **SC-004**: Running the full `test_ppo.py` (all five steps) against a correct reference implementation takes under 60 seconds total on a standard laptop CPU.
- **SC-005**: A `PPOAgent` saved at the end of Workshop 1 can be loaded in Workshop 2 and produce an action from a raw observation with zero lines of participant-written preprocessing code.
- **SC-006**: Error messages produced by the test runner are rated "clear enough to act on without asking the workshop leader" by at least 90 percent of participants in post-workshop feedback.
- **SC-007**: Zero user-facing strings in `ppo_skeleton.py` or `test_ppo.py` are in a language other than English at the time of workshop delivery.

## Assumptions

- Participants have completed the root README setup and have a working `uv sync --group workshop1` environment before touching `ppo_skeleton.py`.
- Participants have Python and basic numpy/PyTorch familiarity, consistent with the stated workshop audience.
- A standard laptop CPU (no GPU) is the baseline target for test runtime.
- `MountainCarContinuous-v0` from Gymnasium is available and the baseline hyperparameters shipped with the skeleton allow the printed loss to show a clear downward trend within the default training budget. Because MountainCarContinuous has sparse rewards, neither the default-mode smoke test nor the `__main__` script run requires the agent to actually solve the environment — that is verified in stage `2-mountaincar/`.
- The `solutions` branch and per-TODO checkpoint tags are maintained by workshop leaders in line with Constitution Articles V and IX. This spec requires their existence at workshop time; creating and pushing those tags is operational work outside this spec.
- The Workshop 2 DonkeyCar stage will consume `PPOAgent` (or one of its subclasses) via `Agent.load(path)` and is responsible for its own preprocessing override if needed. This spec only requires that `PPOAgent` save/load correctly and respect the Article II contract.
- **Out of scope for this spec**: `SB3Agent` (Path B), the `test_agent_interface.py --agent sb3` invocation, and any Stable-Baselines3 dependency wiring. These are deferred to a separate follow-up spec. Path A (custom `PPOAgent`) MUST be runnable end-to-end without `SB3Agent` existing.
- **Out of scope for this spec**: the full implementations of the `2-mountaincar/` and `3-car-racing/` stage subclasses themselves. This spec only fixes the override contract that those subclasses MUST honor; their interior logic, training scripts, and stage-specific tests belong to separate stage features.
