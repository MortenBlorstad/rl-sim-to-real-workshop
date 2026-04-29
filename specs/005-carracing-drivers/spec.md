# Feature Specification: CarRacing Training Drivers (Custom PPO + SB3 + HuggingFace)

**Feature Branch**: `005-carracing-drivers`
**Created**: 2026-04-29
**Status**: Draft
**Input**: User description: "trenger å lage en train.py og en train_sb3.py for race car similar to I have done for pendulum. uses ActorCriticCnnPolicy architecture. for sb3 mulighet for å loade vekter fra hugginface"

## Background *(context — non-mandatory)*

Stage 3 of Workshop 1 is the CarRacing-v3 challenge — a pixel-input continuous-control task where participants train an agent to drive around a procedurally-generated track. The stage already has skeleton driver files at `workshop-1/3-car-racing/train.py` (custom PPO) and `workshop-1/3-car-racing/train_sb3.py` (SB3 escape hatch), but both are broken after recent refactors of the `ppo` package — they reference symbols (`ppo.train`, `ppo._seed_everything`, top-level `_runlog`/`_eval`/`_sb3_jsonl_callback` modules) that no longer exist. Pendulum's stage-2 drivers were repaired in feature 003 and now follow a clean pattern: thin scripts that build a vector env, construct the agent, run training through `RunLogger`, save the model, and evaluate. CarRacing should follow the same pattern, adapted for pixel observations and a CNN-based policy/value architecture, plus an SB3-only option to **initialise from a HuggingFace Hub checkpoint and fine-tune from there** for participants who can't afford a full from-scratch training run during the workshop.

## Clarifications

### Session 2026-04-29

- Q: When the participant invokes the HuggingFace path, what happens? → A: Load + fine-tune. `--hf-repo X` downloads the HF checkpoint, initialises SB3's `PPO` from those weights instead of random init, and continues training for the configured `--timesteps` steps before evaluating. Skip-training-entirely is **not** a supported mode.

## User Scenarios & Testing *(mandatory)*

### User Story 1 — SB3 path trains CarRacing end-to-end (Priority: P1)

A workshop participant who chose the Stable-Baselines3 path (or fell back to it after struggling with the custom PPO TODOs) wants to train a CarRacing agent using the same workflow they already used for Pendulum: one command, one progress log, one run directory at the end with everything `analyze.ipynb` expects (`meta.json`, `metrics.jsonl`, `model.zip`, `eval.mp4`). The SB3 driver uses SB3's CNN-based actor-critic policy (`ActorCriticCnnPolicy` / `"CnnPolicy"`) on stacked grayscale frames so the same training command works on a participant's laptop.

**Why this priority**: SB3 is the constitutional escape hatch (Article VI). If only one CarRacing path works, it must be this one — it has to deliver a trained model whether or not anyone in the room finished the custom PPO. It is also the simplest path to land because SB3 ships the CNN policy out of the box.

**Independent Test**: From a clean checkout, run `uv run python workshop-1/3-car-racing/train_sb3.py --timesteps 10000 --run-name smoke --force` and confirm: (a) the script exits with status 0, (b) `runs/car-racing/smoke/` contains `meta.json`, `metrics.jsonl` (non-empty), `model.zip`, and either `eval.mp4` or `eval.mp4.skipped`, and (c) `meta.json["agent_class"]` clearly identifies this as the SB3 CNN path.

**Acceptance Scenarios**:

1. **Given** a fresh checkout with `uv sync --group workshop1` complete, **When** the participant runs `train_sb3.py` with default arguments, **Then** training starts, prints SB3 progress lines, writes `metrics.jsonl` records via the JSONL callback, and on completion produces `model.zip`, `eval.mp4`/`.skipped`, and `meta.json` with `status: "ok"`.
2. **Given** the same setup, **When** the participant interrupts with Ctrl-C mid-training, **Then** the run directory still exists, `meta.json["status"]` is `"interrupted"`, and any partial `model.zip` is either present or omitted cleanly (no half-written file that crashes `analyze.ipynb`).
3. **Given** the run directory already exists from a previous attempt, **When** the participant re-runs without `--force`, **Then** the driver fails fast with a clear `RunDirectoryExistsError` message naming the conflicting path and the `--force` / `--run-name` options.

---

### User Story 2 — Custom PPO path trains CarRacing with a CNN policy (Priority: P2)

A workshop participant who finished all five PPO TODOs in stage 1 wants to apply *their* PPO implementation to CarRacing — same `PPOAgent` class, same `train()` method, but with a CNN-based actor and critic appropriate for pixel observations. The custom-PPO `train.py` builds a vector env (with grayscale + resize + frame-stack wrappers per env), constructs `PPOAgent` (which auto-selects a CNN architecture when the observation is 3D image data), and runs through the same `RunLogger` flow as Pendulum.

**Why this priority**: Article II of the constitution (NON-NEGOTIABLE) requires both paths — custom PPO and SB3 — to work for every workshop challenge, including CarRacing. Without this, participants who picked Path A have nowhere to apply their work in stage 3. P2 (not P1) because Path B already delivers a working car; this is what gives Path A graduates a finishing line.

**Independent Test**: Run `uv run python workshop-1/3-car-racing/train.py --timesteps 10000 --run-name smoke --force` on a clean checkout. The script must complete without `NotImplementedError` or shape-mismatch errors, write the canonical run-directory artefacts, and `meta.json["agent_class"]` must clearly identify this as the custom-PPO CNN path.

**Acceptance Scenarios**:

1. **Given** a working `PPOAgent` (all five TODOs implemented) and pixel observations from `CarRacing-v3` after grayscale + resize + frame-stack preprocessing, **When** the driver runs `agent.train(env, total_timesteps=10_000)` on a vector env, **Then** training completes, `model.pt` is written via `agent.save(...)`, and the rolled-up `metrics.jsonl` contains finite (non-NaN) `policy_loss` / `value_loss` values across all updates.
2. **Given** the same training run, **When** `agent.predict(raw_pixel_obs)` is called on a Pi-shaped single observation after `agent = PPOAgent.load("model.pt", env)`, **Then** the call returns a float32 action of shape `(3,)` within the action bounds `[-1, 0, 0] … [1, 1, 1]` — i.e., the saved model is portable to the same single-env inference path Pendulum uses.
3. **Given** an Apple Silicon Mac, **When** the driver runs with `RL_WORKSHOP_DEVICE=mps` (the default on that hardware), **Then** training uses MPS and is at least as fast as the CPU run on the same machine — i.e., the device fix from feature 004 actually pays off here, where the CNN forward dominates per-step overhead.

---

### User Story 3 — SB3 driver fine-tunes from a HuggingFace Hub checkpoint (Priority: P3)

A workshop participant who is behind, on slow hardware, or wants to skip the slow early-training phase invokes the SB3 driver with `--hf-repo`. The driver downloads a community checkpoint from HuggingFace Hub, initialises SB3's `PPO` from those weights instead of random init, and **fine-tunes for the configured `--timesteps`** before evaluating. Because the starting weights already drive competently, even a short fine-tune produces a working car — letting every participant see a competent car drive in workshop time, regardless of their machine.

**Why this priority**: Workshop hardware is heterogeneous. CarRacing's pixel + CNN combo is the most compute-hungry stage in Workshop 1; training from scratch to "drives the track competently" takes hours of wall-clock time. Fine-tuning from a HF checkpoint compresses that into ~minutes while keeping the same training loop, the same `meta.json`/`metrics.jsonl` artefacts, and the same teaching value (the participant *does* see PPO updates running on their machine). P3 because it is additive — US1 and US2 already deliver the workshop's core value (from-scratch training).

**Independent Test**: Run `uv run python workshop-1/3-car-racing/train_sb3.py --hf-repo <some-public-sb3-CarRacing-repo> --timesteps 10000 --run-name finetune --force`. The driver must (a) download the named checkpoint from HuggingFace Hub, (b) load it into SB3's `PPO`, (c) fine-tune for `--timesteps` env steps, (d) produce `eval.mp4` showing the car driving competently, and (e) write `meta.json` recording the source `repo_id` and `filename`.

**Acceptance Scenarios**:

1. **Given** an internet connection and a valid HuggingFace repo identifier (e.g. `sb3/ppo-CarRacing-v0`), **When** the participant runs `train_sb3.py --hf-repo X --timesteps 10000`, **Then** the model is downloaded, loaded, fine-tuned for 10 000 env steps, evaluated, and the resulting `meta.json` records the source `repo_id` and `filename`, distinguishing the run from a from-scratch one.
2. **Given** no internet connection, **When** the participant invokes the HuggingFace path, **Then** the driver fails fast with a clear error message that names the URL it tried to reach and suggests the offline alternative (use a `pretrained/` artefact).
3. **Given** a HuggingFace repo identifier that does not exist or has no SB3-compatible artefact, **When** the participant runs the command, **Then** the driver surfaces the underlying error message in a readable form, names the repo it tried, and suggests checking the spelling — it does not silently produce a corrupt run directory.
4. **Given** a HuggingFace checkpoint whose architecture / hyperparameters do not match the local SB3 `PPO` config (e.g. different network width, different `n_steps`), **When** the participant invokes the HF path, **Then** the underlying SB3 mismatch error is surfaced with a hint that the HF checkpoint was trained with a different config and the participant should pick a compatible repo.

---

### Edge Cases

- **CarRacing requires `swig` for Box2D**. If the system dependency is missing, `gymnasium[box2d]` fails to import. Both drivers must surface this with an actionable hint (install `swig` per the README), not a Python traceback.
- **Pixel observation pipeline**: the per-env wrapper chain `Grayscale → Resize(84, 84) → FrameStack(4)` produces `(4, 84, 84)` uint8 observations. The custom PPO needs to consume this shape natively; SB3's `CnnPolicy` already does.
- **Action space asymmetry**: CarRacing's actions are `Box([-1, 0, 0], [1, 1, 1])` — steer is symmetric, gas and brake are non-negative. The PPO `Normal` sampling + `clamp` already handles asymmetric bounds; only worth listing here so the implementer remembers to confirm the existing clamp logic behaves correctly when tested.
- **`evaluate()` rendering**: video recording requires `render_mode="rgb_array"` and a working `imageio[ffmpeg]`. Both drivers must write `eval.mp4.skipped` (sentinel) on ffmpeg failure and continue, matching the Pendulum pattern.
- **HuggingFace download caching**: repeated invocations with the same `repo_id`/filename should reuse the local cache (don't re-download each run).
- **HuggingFace artefact device tags**: a CUDA-saved SB3 zip must load on a Mac without crashing — SB3 handles this internally with `device="auto"` or `device="cpu"`, but the driver should set `device` consistent with the `RL_WORKSHOP_DEVICE` mechanism shipped in feature 004.
- **HuggingFace checkpoint architecture mismatch**: an HF checkpoint whose policy network or training-loop hyperparameters (e.g. `n_steps`, network arch) differ from the local SB3 `PPO` config will fail to load via `PPO.load(env=...)`. The driver MUST surface SB3's underlying error with an actionable hint that the HF repo was trained with a different config and the participant should pick a compatible one.
- **Run directory schema** is fixed by feature 002's `run-format.md` contract — any new fields written by these drivers must extend, not redefine, the schema.

## Requirements *(mandatory)*

### Functional Requirements

- **FR-001**: `workshop-1/3-car-racing/train_sb3.py` MUST run end-to-end on a fresh checkout with `uv sync --group workshop1` and produce the canonical run directory (`meta.json` + `metrics.jsonl` + `model.zip` + `eval.mp4` or `eval.mp4.skipped`) under `runs/car-racing/<run-name>/`.
- **FR-002**: `workshop-1/3-car-racing/train.py` MUST run end-to-end with the refactored `PPOAgent` and produce the canonical run directory (`meta.json` + `metrics.jsonl` + `model.pt` + `eval.mp4` or `eval.mp4.skipped`) under `runs/car-racing/<run-name>/`.
- **FR-003**: Both drivers MUST use the same canonical preprocessing pipeline at the env-wrapper level: per-underlying-env `Grayscale → Resize(84, 84) → FrameStack(4)`, applied via `gym.make_vec(..., wrappers=[...])` so the agent sees `(4, 84, 84)` observations.
- **FR-004**: The custom-PPO `train.py` MUST construct a vector env with the same `SyncVectorEnv` + `SAME_STEP` autoreset configuration introduced in feature 004 for Pendulum, so `PPOAgent.train()` sees the API it expects.
- **FR-005**: `PPOAgent` MUST be able to consume CarRacing's `(4, 84, 84)` observations and produce a `(3,)` continuous action — i.e., the agent's network architecture choice MUST adapt to image observations (CNN-based actor and critic), automatically or via a configuration knob, without breaking the existing MLP path used for Pendulum.
- **FR-006**: `train_sb3.py` MUST use SB3's `ActorCriticCnnPolicy` (passed as the `"CnnPolicy"` string to `stable_baselines3.PPO`).
- **FR-007**: `train_sb3.py` MUST support a documented command-line option that loads pretrained weights from HuggingFace Hub by `repo_id` (and `filename` if multiple checkpoints exist in the repo), initialises the SB3 `PPO` model from those weights, and continues training for the configured `--timesteps` steps before evaluating. The source identifiers MUST be recorded in `meta.json`. Skip-training-entirely is **not** a supported mode; participants who only want to view a pretrained agent driving should use a local `pretrained/` artefact instead.
- **FR-008**: When the HuggingFace path is invoked, the driver MUST cache downloaded artefacts so a second run with the same `repo_id`/filename reuses the local file rather than re-downloading.
- **FR-009**: When the HuggingFace path is invoked and no internet connection is available, or the repo/filename is invalid, the driver MUST fail fast with a message that names the repo+filename and suggests an alternative (e.g. fall back to a local `pretrained/` artefact).
- **FR-010**: Both drivers MUST honour `RL_WORKSHOP_DEVICE` (cpu / cuda / mps / auto) introduced in feature 004 — for SB3 by passing the resolved device to `PPO(...)`/`PPO.load(...)`, for custom PPO automatically through `PPOAgent.__init__` calling `get_device()`.
- **FR-011**: Both drivers MUST handle `KeyboardInterrupt` cleanly: `meta.json["status"] = "interrupted"`, no half-written `model.{pt,zip}` files (write atomically or omit on interrupt).
- **FR-012**: Both drivers MUST handle `RunDirectoryExistsError` with a clear message identifying the conflicting path and pointing the participant at `--force` or `--run-name`.
- **FR-013**: `meta.json` for the SB3-with-HuggingFace runs MUST record the source identifiers (`hf_repo_id`, `hf_filename`) so the run is unambiguously distinguishable from a self-trained run when reading `metrics.jsonl` later.
- **FR-014**: The Stage-3 `README.md` MUST document the three commands (custom training, SB3 training, SB3 from HuggingFace), the required system dependency (`swig`), and the override env vars (`RL_WORKSHOP_DEVICE`).
- **FR-015**: Both drivers MUST write data through the existing `RunLogger` from `ppo.utils` so the `meta.json` schema (including the `device` field added in feature 004's run-metadata contract) is consistent with stages 1 and 2.

### Key Entities *(include if feature involves data)*

- **CarRacing run directory**: `runs/car-racing/<run-name>/{meta.json, metrics.jsonl, model.{pt|zip}, eval.mp4|eval.mp4.skipped}` — extends feature 002's `run-format.md`. New fields for the HuggingFace path: `hf_repo_id` (string), `hf_filename` (string), both nullable.
- **Pretrained-from-Hub artefact**: a downloaded SB3 zip cached in HuggingFace's local cache directory (default `~/.cache/huggingface/hub/...`). Lifetime is OS-managed; the driver does not own its cleanup.
- **Stage-3 wrapper chain**: per-env `Grayscale → Resize(84, 84) → FrameStack(4)` — an environment-level transformation, not an agent-level one (consistent with the project's settled "no `Agent.preprocess()`, driver wraps env" pattern).

## Success Criteria *(mandatory)*

### Measurable Outcomes

- **SC-001**: A workshop participant on a default-configured Apple Silicon Mac with `swig` installed can run `uv run python workshop-1/3-car-racing/train_sb3.py --timesteps 10000 --run-name smoke --force` and reach the green "training complete" log line in under 60 seconds, with the canonical run directory written successfully.
- **SC-002**: A workshop participant who has finished the five PPO TODOs can run `uv run python workshop-1/3-car-racing/train.py --timesteps 10000 --run-name smoke --force` and reach the same green "training complete" line in under 90 seconds, also writing the canonical run directory.
- **SC-003**: From a fresh machine without prior HuggingFace cache and with internet, a participant can run `train_sb3.py --hf-repo <public-CarRacing-repo> --timesteps 10000 --run-name finetune --force` and complete in under 90 seconds end-to-end (download + fine-tune + evaluate), with `eval.mp4` showing the car visibly driving (not random behaviour).
- **SC-004**: A second invocation of the same HuggingFace command on the same machine completes the *download* in under 1 second (cache hit, no network usage observable). The fine-tune + evaluate portion may take its full normal time.
- **SC-005**: On the same Apple Silicon Mac running `train.py` (custom PPO), `RL_WORKSHOP_DEVICE=mps` produces wall-clock per-update time *no slower* than `RL_WORKSHOP_DEVICE=cpu` over a 5-update window after warm-up — i.e., feature 004's device fix actually wins on CarRacing's CNN, validating the entire device-policy investment retroactively.
- **SC-006**: Both drivers' produced run directories pass the existing `analyze.ipynb` (or a stage-3-specific notebook) without modification: the notebook reads `meta.json`, plots `metrics.jsonl`, and shows `eval.mp4` for each run shape.
- **SC-007**: A participant who omits `swig` system-wide gets a clear actionable error message naming the missing dependency, install command per OS, and a link to the README — not a raw Python traceback.

## Assumptions

- The existing `pretrained/` directory will gain a CarRacing entry as a follow-up (separate feature). The HuggingFace flag is a *complementary* online safety net, not a replacement.
- `huggingface_sb3` (or equivalent helper) is acceptable to add to the `workshop1` dependency group; it is a small dependency with no transitive surprises. If the planner determines a manual `huggingface_hub` + `PPO.load` flow is cleaner, that is also acceptable; the requirement is functional, not API-shaped.
- The custom PPO's CNN architecture follows the standard "Nature DQN" backbone (three conv layers + FC head) that SB3's `CnnPolicy` already uses. The exact channel counts / hidden sizes are a planning-phase decision.
- The custom-PPO observation handling stays at the env-wrapper level (per the settled feedback "driver wraps env, no `Agent.preprocess()`"). The agent only needs to support a 3D obs in its forward pass.
- This feature does not retrain or regenerate `pretrained/` artefacts. Any required `pretrained/` updates are a follow-up.
- This feature does not modify Workshop 2 (DonkeyCar / PiRacer) code.
- The HuggingFace flow is read-only: this feature does NOT include uploading models to HuggingFace from the workshop, only downloading.
- Test scope: like feature 003, smoke-test step at the agent level (PPO `train()` smoke on a 4×84×84 vector env), plus a dry-run HuggingFace test that monkey-patches the download (or uses a tiny test-only repo). Full convergence runs are out of scope for the test suite.
