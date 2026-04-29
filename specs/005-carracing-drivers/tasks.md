---

description: "Task list for feature 005-carracing-drivers"
---

# Tasks: CarRacing Training Drivers (Custom PPO + SB3 + HuggingFace Fine-Tune)

**Input**: Design documents from `/specs/005-carracing-drivers/`
**Prerequisites**: plan.md, spec.md, research.md, data-model.md, contracts/cli.md, contracts/meta-fields.md, quickstart.md

**Tests**: Required. Constitution Article IV (NON-NEGOTIABLE) demands `test_agent_interface.py` covers the new CNN path; Article II demands the existing `Agent` interface tests keep passing for both MLP and CNN agents.

**Organization**: Tasks are grouped by user story so each story can be implemented and validated independently against the spec's acceptance scenarios.

## Format: `[ID] [P?] [Story] Description`

- **[P]**: Can run in parallel (different files, no dependencies)
- **[Story]**: Which user story this task belongs to (US1, US2, US3)
- File paths are absolute and reference real files in the repo unless flagged NEW

## Path Conventions

This is a workshop training package, not a "src/tests" project:

- Library code: `workshop-1/1-ppo/ppo/`
- Library tests (custom step-runner, no pytest): `workshop-1/1-ppo/ppo/tests/`
- Stage drivers: `workshop-1/3-car-racing/{train.py, train_sb3.py, README.md}`
- Notebooks: `workshop-1/3-car-racing/analyze.ipynb` (if it exists; if not, defer to a follow-up)

---

## Phase 1: Setup (Shared Infrastructure)

**Purpose**: Add the new dependency and verify the workspace is ready.

- [ ] T001 Add `huggingface_hub>=0.20` to `[dependency-groups].workshop1` in `/Users/mortenblorstad/projects/phd/RL-workshop/pyproject.toml`. Place it next to the existing `stable-baselines3` line, matching the existing alphabetical-ish ordering.
- [ ] T002 Run `uv sync --group workshop1` from the repo root to install `huggingface_hub`. Confirm `uv run python -c "import huggingface_hub; print(huggingface_hub.__version__)"` prints a version ≥ 0.20. The `uv.lock` file must update — commit it together with the `pyproject.toml` change.
- [ ] T003 Verify the branch is `005-carracing-drivers` and that `specs/005-carracing-drivers/{spec.md,plan.md,research.md,data-model.md,contracts/,quickstart.md}` are all present.

---

## Phase 2: Foundational (Blocking Prerequisites)

**Purpose**: Remove the broken pre-refactor CarRacing file and extend `RunLogger` so all three stories can write the new `meta.json` fields. Every user story depends on this.

**⚠️ CRITICAL**: No user story work can begin until this phase is complete.

- [ ] T004 Delete `/Users/mortenblorstad/projects/phd/RL-workshop/workshop-1/3-car-racing/agent.py`. This is the pre-refactor `CarRacingPPOAgent` class that the new `train.py` no longer needs (CNN selection moved into `PPOAgent.__init__`). Confirm via `git rm` that no other file in the repo imports from it.
- [ ] T005 Extend `/Users/mortenblorstad/projects/phd/RL-workshop/workshop-1/1-ppo/ppo/utils/_runlog.py` `RunLogger.__init__` to accept three new optional kwargs: `network_arch: str | None = None`, `hf_repo_id: str | None = None`, `hf_filename: str | None = None`. When provided (non-None), write them into `self._meta` so they land in `meta.json` per `contracts/meta-fields.md`. Maintain backwards compatibility — existing callers that don't pass these kwargs must continue to work without `meta.json` schema changes (the fields just won't appear).

**Checkpoint**: `agent.py` is gone, `RunLogger(...)` accepts the three new kwargs without breaking any of the existing Pendulum / MountainCar drivers. Run a smoke `RL_WORKSHOP_DEVICE=cpu uv run python workshop-1/1-ppo/ppo/tests/test_agent_interface.py --agent ppo` to confirm the existing 7 tests still pass.

---

## Phase 3: User Story 1 — SB3 path trains CarRacing end-to-end (Priority: P1) 🎯 MVP

**Goal**: Rewrite `train_sb3.py` so a participant can train CarRacing from scratch with `PPO("CnnPolicy", ...)` and produce the canonical run directory.

**Independent Test**: `uv run python workshop-1/3-car-racing/train_sb3.py --timesteps 10000 --run-name smoke-sb3 --force` exits 0; `runs/car-racing/smoke-sb3/` contains `meta.json`, `metrics.jsonl` (non-empty), `model.zip`, and `eval.mp4` (or `eval.mp4.skipped`); `meta.json["agent_class"]` identifies this as the SB3 CNN path.

### Implementation for User Story 1

- [ ] T006 [US1] Rewrite `/Users/mortenblorstad/projects/phd/RL-workshop/workshop-1/3-car-racing/train_sb3.py` per `contracts/cli.md` § "train_sb3.py". Concrete shape:
  - Imports: `gymnasium as gym`, `gymnasium.vector.AutoresetMode`, `gymnasium.wrappers.{GrayscaleObservation, ResizeObservation, FrameStackObservation}`, `stable_baselines3.PPO`, and from `ppo.utils`: `RunDirectoryExistsError, RunLogger, Sb3JsonlCallback, record_eval_episode, get_device`.
  - Build `env = gym.make_vec("CarRacing-v3", num_envs=4, vectorization_mode="sync", wrappers=[...], vector_kwargs={"autoreset_mode": AutoresetMode.SAME_STEP})`. The `wrappers=[...]` list uses lambdas to apply `ResizeObservation((84, 84))` and `FrameStackObservation(4)` since they need extra args; `GrayscaleObservation(keep_dim=False)` can be passed directly.
  - Construct `model = PPO("CnnPolicy", env, seed=args.seed, device=str(get_device()), verbose=0)`.
  - Open `RunLogger(stage="car-racing", ..., network_arch="cnn", hf_repo_id=None, hf_filename=None, ...)` (the latter two anchor the always-present-with-null contract from `contracts/meta-fields.md`).
  - Inside `with runlog:` call `model.learn(total_timesteps=args.timesteps, callback=Sb3JsonlCallback(runlog), progress_bar=False)`.
  - Save `model.save(str(runlog.run_dir / "model.zip"))`.
  - If not `--no-eval`: call `record_eval_episode(ENV_ID, model.predict, runlog.run_dir, seed=args.seed)`. (CarRacing's `record_eval_episode` needs `env_id="CarRacing-v3"`; if that helper requires a callable that takes single-obs and returns single-action, adapt with a `lambda obs: model.predict(obs, deterministic=True)[0]`.)
  - Handle `RunDirectoryExistsError` and `KeyboardInterrupt` per the existing Pendulum `train_sb3.py` pattern.
- [ ] T007 [US1] Run `uv run python workshop-1/3-car-racing/train_sb3.py --timesteps 10000 --run-name smoke-sb3 --force` and confirm: (a) exit code 0, (b) `runs/car-racing/smoke-sb3/meta.json` exists with `status: "ok"`, `network_arch: "cnn"`, `device` populated, `hf_repo_id: null`, `hf_filename: null`, (c) `metrics.jsonl` non-empty, (d) `model.zip` non-empty, (e) `eval.mp4` or `eval.mp4.skipped` present, (f) wall-clock under 60s on Apple Silicon (SC-001).

**Checkpoint**: User Story 1 functional. Workshop participants on the SB3 path can train CarRacing end-to-end with one command. The constitutional escape hatch is unblocked.

---

## Phase 4: User Story 2 — Custom-PPO path trains CarRacing with a CNN policy (Priority: P2)

**Goal**: Add CNN actor/critic networks to the `ppo` package, teach `PPOAgent.__init__` to auto-detect image observations, rewrite `train.py`, and add a CNN smoke test.

**Independent Test**: `uv run python workshop-1/3-car-racing/train.py --timesteps 10000 --run-name smoke-custom --force` completes without errors, writes the canonical run dir; `meta.json["network_arch"] == "cnn"`; `agent.predict(raw_pixel_obs)` after `PPOAgent.load("model.pt", env)` returns a `(3,)` float32 action in bounds; on Apple Silicon, MPS time/update is no slower than CPU (SC-005).

### Implementation for User Story 2

- [ ] T008 [P] [US2] Add `CnnActorNetwork` and `CnnCriticNetwork` to `/Users/mortenblorstad/projects/phd/RL-workshop/workshop-1/1-ppo/ppo/networks.py` per `data-model.md` § 3. Backbone: `Conv2d(in_ch, 32, 8, 4) → ReLU → Conv2d(32, 64, 4, 2) → ReLU → Conv2d(64, 64, 3, 1) → ReLU → flatten → Linear(64*7*7, 512) → ReLU`. Actor head: `Linear(512, action_dim)` with orthogonal init gain `0.01`; critic head: `Linear(512, 1)` with orthogonal init gain `1.0`. Conv layers and FC bottleneck use orthogonal init gain `sqrt(2)`. Forward must accept `(B, in_channels, 84, 84)` AND `(in_channels, 84, 84)` (single obs); critic returns `(B,)` after `.squeeze(-1)` (or `()` for single obs). Existing MLP `ActorNetwork`/`CriticNetwork` are not touched.
- [ ] T009 [US2] In `/Users/mortenblorstad/projects/phd/RL-workshop/workshop-1/1-ppo/ppo/ppo.py` `PPOAgent.__init__` (currently lines 50-77 plus the device-fix block from feature 004), after computing `single_obs_space` add:
  ```python
  obs_shape = tuple(single_obs_space.shape)
  if len(obs_shape) == 1:
      self.network_arch = "mlp"
      self.actor = ActorNetwork(self.obs_dim, self.action_dim).to(self.device)
      self.critic = CriticNetwork(self.obs_dim).to(self.device)
  elif len(obs_shape) == 3:
      self.network_arch = "cnn"
      in_channels = obs_shape[0]
      self.actor = CnnActorNetwork(in_channels, self.action_dim).to(self.device)
      self.critic = CnnCriticNetwork(in_channels).to(self.device)
  else:
      raise ValueError(
          f"PPOAgent: unsupported observation shape {obs_shape}. "
          f"Expected 1D (vector) or 3D (image) observations after env wrappers."
      )
  ```
  Replace the existing unconditional MLP allocation with this branch. Update the imports at the top of the file: `from .networks import ActorNetwork, CriticNetwork, CnnActorNetwork, CnnCriticNetwork`. The `[PPOAgent] device=...` log line added in feature 004 grows a sibling `[PPOAgent] network_arch={self.network_arch}` line printed to stderr.
- [ ] T010 [US2] In the same file, update `PPOAgent.save` (around line 487) to include `"network_arch": self.network_arch` in the saved `state` dict. Update `PPOAgent.load` (classmethod, around line 504) to read `state.get("network_arch")` after loading; if present and `state["network_arch"] != target_cls(env, ...).network_arch`, raise `ValueError` with a message naming both architectures and explaining the env's observation shape doesn't match the saved model. Older `.pt` files (no `network_arch` in state) skip the check — backwards-compat per `contracts/meta-fields.md`.
- [ ] T011 [US2] In the same file, update the rollout loop and `predict()` to normalise obs by 255.0 when `self.network_arch == "cnn"`. Add a private helper `def _prep_obs(self, obs):` that returns `torch.as_tensor(obs, dtype=torch.float32, device=self.device)` for MLP and `torch.as_tensor(obs, dtype=torch.float32, device=self.device) / 255.0` for CNN. Replace the existing `obs_t = torch.as_tensor(obs, ...)` lines in (a) `train()` rollout loop, (b) `train()` truncation bootstrap, (c) `train()` post-rollout `last_value` bootstrap, and (d) `predict()` with `obs_t = self._prep_obs(obs)`. The MLP path must produce identical numerics to before — Pendulum tests must keep passing bit-exactly.
- [ ] T012 [US2] Rewrite `/Users/mortenblorstad/projects/phd/RL-workshop/workshop-1/3-car-racing/train.py` mirroring `workshop-1/2-pendulum/train.py`. Concrete shape:
  - Same imports + `sys.path.insert` pattern as Pendulum.
  - `NUM_ENVS = 4`, `ENV_ID = "CarRacing-v3"`, `STAGE = "car-racing"`, `DEFAULT_TIMESTEPS = 200_000`.
  - `hyperparameters` dict tuned for CNN: keep most of Pendulum's defaults; differences worth setting are `log_std_init: -0.5` (slightly tighter exploration, partial mitigation of the asymmetric-action issue documented in `research.md` R7), `entropy_coef: 0.01`, and `n_epochs: 4` (CNN updates are heavier).
  - Build vec env: `gym.make_vec(ENV_ID, num_envs=NUM_ENVS, vectorization_mode="sync", wrappers=[GrayscaleObservation, lambda e: ResizeObservation(e, (84, 84)), lambda e: FrameStackObservation(e, 4)], vector_kwargs={"autoreset_mode": AutoresetMode.SAME_STEP})`.
  - Construct `agent = PPOAgent(env, hyperparameters=hyperparameters)`. The CNN auto-detect from T009 fires here.
  - Open `RunLogger(stage=STAGE, ..., network_arch=agent.network_arch)` (passes `"cnn"`).
  - Inside `with runlog:` call `agent.train(env, total_timesteps=args.timesteps, ..., log_fn=make_log_fn(runlog, agent))`.
  - `agent.save(str(runlog.run_dir / "model.pt"))`.
  - If not `--no-eval`: `agent.evaluate(env, n_episodes=1, record_video=True, video_dir=runlog.run_dir)`. (`evaluate` already builds its own single env via `env.spec.id`; vector env spec works.)
  - Handle `RunDirectoryExistsError` and `KeyboardInterrupt` per Pendulum's pattern. Match Pendulum's CLI flag set exactly (no `--seed`, no `--hf-repo`).
- [ ] T013 [P] [US2] Add `@step("C8")` test in `/Users/mortenblorstad/projects/phd/RL-workshop/workshop-1/1-ppo/ppo/tests/test_agent_interface.py`. Title: "C8 CNN smoke on (4, 84, 84) vec env". The step:
  1. Build `env = gym.make_vec("CarRacing-v3", num_envs=2, vectorization_mode="sync", wrappers=[GrayscaleObservation, lambda e: ResizeObservation(e, (84, 84)), lambda e: FrameStackObservation(e, 4)], vector_kwargs={"autoreset_mode": AutoresetMode.SAME_STEP})`.
  2. `agent = PPOAgent(env, hyperparameters={"rollout_size": 32, "n_epochs": 1, "batch_size": 16, "random_state": 0, "log_std_init": -0.5})`.
  3. Assert `agent.network_arch == "cnn"`.
  4. `t0 = time.perf_counter(); stats = agent.train(env, total_timesteps=64, random_state=0, log_fn=lambda _: None); elapsed = time.perf_counter() - t0`. (`total_timesteps=64`, `rollout_size=32`, `num_envs=2` ⇒ 1 update of 32 inner-loop iterations; cheap.)
  5. Assert `elapsed < 10.0` (Article IV per-step budget).
  6. Assert `stats["mean_reward"]`, `stats["policy_loss"]`, `stats["value_loss"]`, `stats["entropy"]` are all finite (not NaN, not inf).
  7. `env.close()` in a `finally` block.
- [ ] T014 [US2] Run `uv run python workshop-1/3-car-racing/train.py --timesteps 10000 --run-name smoke-custom --force` on Apple Silicon. Confirm: (a) exit code 0, (b) `meta.json["network_arch"] == "cnn"`, (c) `metrics.jsonl` contains finite (non-NaN) `policy_loss` / `value_loss` across all updates, (d) `model.pt` non-empty, (e) wall-clock < 90 s (SC-002).
- [ ] T015 [US2] Run the SC-005 retroactive validation per `quickstart.md` § Flow D step (4): `RL_WORKSHOP_DEVICE=cpu uv run python workshop-1/3-car-racing/train.py --timesteps 10000 --run-name validate-cnn-cpu --force`, then `RL_WORKSHOP_DEVICE=mps uv run python workshop-1/3-car-racing/train.py --timesteps 10000 --run-name validate-cnn-mps --force`. Compute mean wall-clock per update (excluding 2-update warm-up). Confirm `mps_per_update <= cpu_per_update * 1.10` (SC-005 — MPS no slower than CPU on the CNN, retroactive validation of feature 004's device fix). If MPS is slower than CPU on the CNN, flag in the PR description; the routing-layer fix is genuinely insufficient for the workshop's CNN scale and the asymmetric-action distribution may need to be revisited.
- [ ] T016 [P] [US2] Verify all existing PPO tests still pass on CPU and MPS, end-to-end: `RL_WORKSHOP_DEVICE=cpu uv run python workshop-1/1-ppo/ppo/tests/test_ppo.py`, `RL_WORKSHOP_DEVICE=mps uv run python workshop-1/1-ppo/ppo/tests/test_ppo.py`, `RL_WORKSHOP_DEVICE=cpu uv run python workshop-1/1-ppo/ppo/tests/test_agent_interface.py --agent ppo`, `RL_WORKSHOP_DEVICE=mps uv run python workshop-1/1-ppo/ppo/tests/test_agent_interface.py --agent ppo`. Each must end with `ALL STEPS OK!` / `Summary: 8 / 8 passed` (the C8 CNN test makes that 8). Pendulum's MLP path must keep passing bit-exactly.

**Checkpoint**: User Story 2 functional. CNN auto-detection works; custom PPO trains CarRacing; tests green on both devices; SC-005 measured (and documented if it fails). Article II compliance restored — both paths reach the CarRacing finish line.

---

## Phase 5: User Story 3 — SB3 driver fine-tunes from a HuggingFace Hub checkpoint (Priority: P3)

**Goal**: Add `--hf-repo` and `--hf-filename` flags to `train_sb3.py`. When `--hf-repo` is set, download via `huggingface_hub.hf_hub_download`, load via `PPO.load(env=env)`, fine-tune for `--timesteps` steps, evaluate. Record source identifiers in `meta.json`.

**Independent Test**: `uv run python workshop-1/3-car-racing/train_sb3.py --hf-repo sb3/ppo-CarRacing-v0 --timesteps 10000 --run-name finetune --force` downloads, loads, fine-tunes, and writes `meta.json` with `hf_repo_id: "sb3/ppo-CarRacing-v0"`, `hf_filename: "ppo-CarRacing-v0.zip"`. Eval video shows the car driving competently. Cold-cache run < 90s; warm-cache repeat has download phase < 1s.

### Implementation for User Story 3

- [ ] T017 [US3] Add a small helper module `/Users/mortenblorstad/projects/phd/RL-workshop/workshop-1/3-car-racing/_huggingface.py` (NEW) containing `HuggingFaceLoadError(RuntimeError)` plus `def download_pretrained(repo_id: str, filename: str | None = None) -> str:` that (a) auto-derives filename via `f"{repo_id.split('/', 1)[-1]}.zip"` when not provided, (b) calls `hf_hub_download(repo_id=repo_id, filename=filename)`, (c) catches `LocalEntryNotFoundError`, `RepositoryNotFoundError`, `EntryNotFoundError`, `requests.exceptions.ConnectionError`, `OSError` and re-raises `HuggingFaceLoadError` with the contract-mandated messages from `contracts/cli.md` § "HuggingFace download failure", (d) returns the local cached path on success and prints `[train_sb3] downloaded {filename} from {repo_id}` (or `(cache hit)` if the call returned in < 0.5s — measure with `time.perf_counter()` around the call). Keep this module sibling-of-driver because it is tightly coupled to `train_sb3.py` and not part of the agent contract.
- [ ] T018 [US3] Extend `train_sb3.py` argparse with two new flags per `contracts/cli.md`: `--hf-repo` (str, optional) and `--hf-filename` (str, optional). Add the mutual-exclusion check: if `--hf-filename` is set without `--hf-repo`, fail via `parser.error("--hf-filename requires --hf-repo")`. The default for `--hf-repo` is `None` (so `args.hf_repo is None` means from-scratch).
- [ ] T019 [US3] In `train_sb3.py`, branch the model construction: if `args.hf_repo is None`, do the existing `model = PPO("CnnPolicy", env, ...)`. Otherwise: `from _huggingface import download_pretrained, HuggingFaceLoadError`, then `local_path = download_pretrained(args.hf_repo, args.hf_filename)`, then `model = PPO.load(local_path, env=env, device=str(get_device()))`. Wrap the download call in a `try/except HuggingFaceLoadError as exc: print(f"Error: {exc}", file=sys.stderr); env.close(); return 1` so the driver fails fast with the contract message rather than a Python traceback. **Architecture mismatch**: the `PPO.load(...)` call may itself raise on state-dict mismatch — wrap that in a second `try/except RuntimeError` that re-raises with the contract-mandated arch-mismatch message naming the repo/filename.
- [ ] T020 [US3] Pass the resolved `repo_id` and `filename` into `RunLogger(...)`: when `args.hf_repo is None`, both kwargs are `None` (so they appear as `null` in `meta.json` per the always-present-with-null contract); otherwise pass `hf_repo_id=args.hf_repo, hf_filename=resolved_filename` (the latter is what `download_pretrained` actually used after auto-deriving). The `network_arch` kwarg stays `"cnn"` either way.
- [ ] T021 [US3] Run the cold-cache flow per `quickstart.md` § Flow C: `uv run python workshop-1/3-car-racing/train_sb3.py --hf-repo sb3/ppo-CarRacing-v0 --timesteps 10000 --run-name finetune --force`. Confirm: (a) exit code 0, (b) `meta.json["hf_repo_id"] == "sb3/ppo-CarRacing-v0"`, (c) `meta.json["hf_filename"] == "ppo-CarRacing-v0.zip"`, (d) wall-clock under 90s on Apple Silicon with internet (SC-003), (e) `eval.mp4` shows the car visibly driving (eyeball test).
- [ ] T022 [US3] Run the warm-cache flow: same command with `--run-name finetune-warm`. Confirm the `[train_sb3] (cache hit)` message appears within 1 second (SC-004) and the rest of the run takes the same time as the cold-cache run minus the download time.
- [ ] T023 [US3] Verify each negative path produces the exact contract-mandated error message from `contracts/cli.md` § "HuggingFace download failure": (a) Offline + cache empty: temporarily rename the cache dir or `unset HF_HUB_OFFLINE`; (b) Repo not found: `--hf-repo nonexistent/repo`; (c) File not in repo: `--hf-repo sb3/ppo-CarRacing-v0 --hf-filename does-not-exist.zip`. Each must print the contract message and exit 1, NOT raise a Python traceback. Skip the offline-test if it's awkward to engineer; the other two are sufficient acceptance.

**Checkpoint**: All three user stories functional. Workshop participants on the SB3 path can fine-tune from HuggingFace in ~minutes regardless of hardware.

---

## Phase 6: Polish & Cross-Cutting Concerns

**Purpose**: Documentation, notebook compatibility, workshop-leader pre-flight, `solutions`-branch lockstep merge.

- [ ] T024 [P] Update or create `/Users/mortenblorstad/projects/phd/RL-workshop/workshop-1/3-car-racing/README.md` (Norwegian, per repo convention) with three top-level sections: (1) `Trening fra null med custom PPO` (custom-PPO command from Flow B), (2) `Trening fra null med Stable-Baselines3` (SB3 command from Flow A), (3) `Fortsett trening fra HuggingFace-modell` (HF fine-tune command from Flow C, with both cold/warm-cache examples). Include a prerequisites section noting `swig` is required and showing the per-OS install command (`brew install swig`, `sudo apt-get install swig`, `choco install swig`). Mention the `RL_WORKSHOP_DEVICE=cpu` override as a one-liner.
- [ ] T025 [P] If `/Users/mortenblorstad/projects/phd/RL-workshop/workshop-1/3-car-racing/analyze.ipynb` exists: update any cell that reads `meta.json` to use the defensive pattern from `contracts/meta-fields.md` § "Reader compatibility" (`meta.get("network_arch", "unknown")`, `meta.get("hf_repo_id")`, `meta.get("hf_filename")`). Add a one-line display so the new fields are visible to participants. If the notebook does not yet exist, mark this task as "deferred to a future analyze-notebook feature" in the PR description and proceed.
- [ ] T026 Run the workshop-leader pre-flight per `quickstart.md` § Flow D end-to-end: all four sub-runs (sb3-from-scratch, custom-from-scratch, hf-cold, hf-warm), then the SC-005 cpu-vs-mps comparison. Capture timings into the PR description so the reviewer can verify the success-criteria ratios at a glance.
- [ ] T027 **`solutions`-branch lockstep merge** per `plan.md` § Complexity Tracking row 2. Cherry-pick (or rebase) the `networks.py` and `ppo.py` changes from this branch onto `solutions`. The `networks.py` additions (CNN classes) are net-new and merge cleanly. The `ppo.py` changes (CNN auto-detect in `__init__`, `_prep_obs`, save/load `network_arch` field) need to land alongside the `solutions`-branch's filled-in TODO bodies; the auto-detect block lives outside the `# -- YOUR CODE HERE --` markers so the merge is mechanical. Do NOT skip — Article V's `git checkout solutions -- <path>` recovery mechanism breaks for stage-3 participants if `solutions` lacks the CNN classes.
- [ ] T028 [P] Re-tick `/Users/mortenblorstad/projects/phd/RL-workshop/specs/005-carracing-drivers/checklists/requirements.md` against the now-implemented spec. Tick all items that pass; if anything became invalid (none expected), document it under Notes.

---

## Dependencies & Execution Order

### Phase Dependencies

- **Phase 1 (Setup)**: T001 → T002 (sequential, both touch dependency state); T003 [P] (independent).
- **Phase 2 (Foundational)**: Depends on Phase 1. T004 [P] and T005 are independent of each other; can run in parallel.
- **Phase 3 (US1)**: Depends on Phase 2. T006 → T007 (sequential, T007 verifies T006).
- **Phase 4 (US2)**: Depends on Phase 2. T008 [P] (different file from `ppo.py`) → T009 → T010 → T011 (all touch `ppo.py` sequentially) → T012 (different file, can start once T009-T011 settle) → T013 [P] (different file from `train.py`) → T014 → T015 (both verification, sequential because they re-run training) → T016 [P] (regression check, can interleave).
- **Phase 5 (US3)**: Depends on Phase 3 (US1) — modifies the `train_sb3.py` written in T006. T017 [US3] adds the helper module first, then T018 → T019 → T020 (all in `train_sb3.py`, sequential) → T021/T022/T023 (verification, sequential because they share the cache state).
- **Phase 6 (Polish)**: Depends on all user-story phases.

### User-Story Independence

- **US1 ↔ US2**: Independent at the file level (US1 in `train_sb3.py`, US2 in `train.py` + `networks.py` + `ppo.py` + tests).
- **US1 ↔ US3**: NOT independent — US3 modifies the same `train_sb3.py` US1 produces. US3 must come after US1 lands.
- **US2 ↔ US3**: Independent. US3 (SB3+HF) doesn't touch the agent or `train.py`.

If staffed in parallel: one person on US1+US3 (sequential), one on US2 (parallel from Phase 2 onward).

### Parallel Opportunities

```bash
# Phase 1:
T001 → T002    # pyproject.toml then uv sync
T003 [P]       # independent verification

# Phase 2:
T004 [P]       # delete agent.py
T005 [P]       # extend RunLogger

# Phase 4 (US2):
T008 [P]       # CNN networks (networks.py)
T009 → T010 → T011 → T012   # ppo.py + train.py edits, sequential
T013 [P]       # C8 test (test_agent_interface.py)
T014, T015     # verifications, sequential
T016 [P]       # regression check

# Phase 6:
T024 [P]       # README
T025 [P]       # notebook
T026           # pre-flight (sequential, runs commands)
T027           # solutions-branch merge
T028 [P]       # checklist re-tick
```

---

## Implementation Strategy

### MVP First (User Story 1 only)

1. Phase 1 (T001 → T002, T003 in parallel).
2. Phase 2 (T004, T005 in parallel).
3. Phase 3 (T006 → T007).
4. **STOP and validate**: SB3 from-scratch trains CarRacing end-to-end, smoke run hits SC-001. The user's stated request *"trenger å lage en train_sb3.py for race car"* is now met (the simpler half).

### Incremental Delivery

After MVP:

1. Add US2 (T008–T016) — the user's stated request *"trenger å lage en train.py … uses ActorCriticCnnPolicy"* is met. Custom PPO + CNN landed.
2. Add US3 (T017–T023) — the user's stated request *"for sb3 mulighet for å loade vekter fra hugginface"* is met.
3. Polish (T024–T028).

### Constitution-Compliance Notes

- **Article II (Two Paths, One Agent API):** the `Agent` interface (`predict`, `train`, `save`, `load`) is unchanged; CNN selection is internal to `PPOAgent.__init__`. SB3 path uses SB3's own `CnnPolicy`. Both paths produce the same canonical run-directory artefact shape.
- **Article IV (Test-Verified Implementation):** T013 adds C8 (CNN smoke) to the agent-interface tests; T016 verifies all existing tests still pass on both CPU and MPS. The HF download is **not** in the runtime test suite (network-bound, breaks the < 10 s budget); it is verified by the workshop-leader pre-flight (T026).
- **Article V (Progressive Scaffolding):** TODO 1–5 markers and `raise NotImplementedError` defaults in `ppo.py` are not touched by any task. The CNN auto-detect block in T009 lives outside the `# -- YOUR CODE HERE --` markers. T027 keeps `solutions` branch lockstep — without it the participant fallback (`git checkout solutions -- <path>`) breaks for stage 3.
- **Article VI (Fail-Safe):** SB3 path (US1) is the constitutional escape hatch. HuggingFace fine-tune (US3) is a layered safety net. `RL_WORKSHOP_DEVICE=cpu` override works for both drivers.
- **Article VII (Sim-to-Real):** Inherited deferral from feature 004 (`map_location="cpu"` in `PPOAgent.load`). This feature does not re-open it; flagged as Workshop-2 follow-up in `plan.md` § Complexity Tracking row 1.

---

## Notes

- T002 modifies `uv.lock`. Stage that file together with the `pyproject.toml` change in the same commit.
- T013 (C8 test) constructs a `CarRacing-v3` env, which requires `swig` to be installed system-wide. If running these tasks on a clean machine without `swig`, install it first (`brew install swig` / `sudo apt-get install swig`) or T013 will fail with the contract-mandated error from T024's README pointer.
- T015 (SC-005 measurement) requires Apple Silicon hardware. If the implementer is on Linux+CUDA, the task becomes "verify CUDA is auto-selected and training completes; SC-005 is meaningless without MPS"; mark accordingly in the PR.
- T021–T023 (HuggingFace flow verification) require internet access. If the implementer is offline at task time, T023 (negative paths) can be partially tested without internet; the cold-cache T021 must be re-run later.
- T026 (workshop-leader pre-flight) is a release-gate task; do not skip it before merging the feature.
