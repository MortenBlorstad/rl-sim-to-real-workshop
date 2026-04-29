# Implementation Plan: CarRacing Training Drivers (Custom PPO + SB3 + HuggingFace Fine-Tune)

**Branch**: `005-carracing-drivers` | **Date**: 2026-04-29 | **Spec**: [spec.md](./spec.md)
**Input**: Feature specification from `specs/005-carracing-drivers/spec.md`

## Summary

Replace both CarRacing driver scripts (`workshop-1/3-car-racing/train.py` for custom PPO, `workshop-1/3-car-racing/train_sb3.py` for the SB3 escape hatch), add CNN actor/critic networks to the `ppo` package, and wire a HuggingFace-Hub `--hf-repo` flag into the SB3 driver that loads pretrained weights and continues training from there.

Three concrete deliverables:

1. **CNN networks in the `ppo` package** (`workshop-1/1-ppo/ppo/networks.py`): two new classes `CnnActorNetwork` and `CnnCriticNetwork`, each built on the standard "Nature DQN" backbone (3 conv layers + FC head). `PPOAgent.__init__` auto-detects 3D image observations (`len(single_obs_space.shape) == 3`) and constructs CNN networks instead of the existing 2×64 MLPs. The MLP path is unchanged for Pendulum.

2. **Custom-PPO `train.py`** (`workshop-1/3-car-racing/train.py`): a thin driver mirroring the Pendulum pattern. Builds a `gym.make_vec` with per-env wrappers `[GrayscaleObservation, ResizeObservation((84, 84)), FrameStackObservation(4)]` and `SAME_STEP` autoreset, constructs `PPOAgent`, runs through `RunLogger`, saves `model.pt`, evaluates with video. No `CarRacingPPOAgent` subclass — CNN selection is automatic from the obs shape.

3. **SB3 `train_sb3.py`** (`workshop-1/3-car-racing/train_sb3.py`): a thin driver using `PPO("CnnPolicy", env, ...)`. Adds `--hf-repo REPO_ID [--hf-filename FILE]` flags. When `--hf-repo` is set, the driver downloads the checkpoint via `huggingface_hub.hf_hub_download`, calls `PPO.load(path, env=env, device=...)` (which initialises from those weights), then `model.learn(total_timesteps=...)` to fine-tune from there. The `meta.json` records `hf_repo_id` and `hf_filename` for retrospective attribution.

The plan adds **one new dependency** (`huggingface_hub`) to the `workshop1` group. No changes to Workshop 2 code. No changes to the `Agent` interface. Pendulum drivers and the MLP path are not touched.

## Technical Context

**Language/Version**: Python 3.10–3.12 (per Constitution; `pyproject.toml` pins `>=3.10,<3.13`)
**Primary Dependencies**: PyTorch ≥ 2.1, Gymnasium ≥ 0.29 (with `[box2d]` extra for CarRacing — already installed transitively), Stable-Baselines3, NumPy, OpenCV (cv2 — required by `ResizeObservation`), `imageio[ffmpeg] >= 2.31`. **NEW**: `huggingface_hub >= 0.20` (for `hf_hub_download`).
**Storage**: Per-run dirs under `runs/car-racing/<run-name>/{meta.json, metrics.jsonl, model.{pt,zip}, eval.mp4|.skipped}` — extends feature 002's `run-format.md` contract with two new optional fields (`hf_repo_id`, `hf_filename`). HuggingFace cache lives at `~/.cache/huggingface/hub/...` (OS-managed, not the project's concern).
**Testing**: Existing custom `@step` runners (`workshop-1/1-ppo/ppo/tests/{test_ppo.py, test_agent_interface.py}`). New step C8 in `test_agent_interface.py`: smoke-test `PPOAgent` on a `(4, 84, 84)` vector env to confirm the CNN auto-detection + train loop end-to-end. No HF network test in the runtime suite (a network-bound test is not appropriate for the constitution's < 10 s per step budget); the HF path is verified by the workshop-leader pre-flight in `quickstart.md`.
**Target Platform**: Developer laptop (macOS Apple Silicon primary, Linux+CUDA secondary, headless CI tertiary). Inference on Raspberry Pi 4 (CPU-only) is the eventual Workshop-2 destination — the saved `model.pt` must round-trip CPU loadable; this is currently *not* enforced (see Article VII complexity tracking below).
**Project Type**: Workshop training package (`workshop-1/1-ppo/ppo/`) consumed by stage drivers (`workshop-1/3-car-racing/`). No web/mobile.
**Performance Goals** (per spec SC-001 / SC-002 / SC-003 / SC-005):
- SB3 from-scratch smoke (10 000 timesteps) < 60 s on Apple Silicon.
- Custom-PPO from-scratch smoke (10 000 timesteps) < 90 s on Apple Silicon.
- HuggingFace fine-tune (download + 10 000-step fine-tune + evaluate) < 90 s on a fresh cache.
- Custom-PPO MPS time per update **no worse than CPU** on the CarRacing CNN — the retroactive validation that the 004 device fix actually wins on the workload it was meant for.
**Constraints**:
- Memory budget: a `(rollout_size_per_env, num_envs, 4, 84, 84)` float32 buffer must fit comfortably in laptop RAM. With `num_envs=4`, `rollout_size=2048` → `size_per_env=512`, that's `512 * 4 * 4 * 84 * 84 * 4 bytes ≈ 230 MB`. Acceptable. Larger combinations need explicit attention.
- The CNN forward pass must work on CPU, MPS, **and** CUDA — no MPS-only ops.
- Constitution Article V: `# TODO N: …` markers and `raise NotImplementedError` defaults in `ppo.py` are preserved; CNN classes live in `networks.py`, fully outside any TODO block.
- Test suite per-step budget: < 10 s per step (constitution Article IV). The CNN training smoke (C8) must hit this on the auto-selected device of the runner's machine.
**Scale/Scope**: ~5 source files modified or new (`networks.py`, `ppo.py` __init__, `train.py`, `train_sb3.py`, `pyproject.toml`), ~1 new test step, ~1 README updated. No new package boundaries. No changes to the `Agent` interface, `RolloutBuffer`, or `RunLogger`.

## Constitution Check

*GATE: Must pass before Phase 0 research. Re-checked after Phase 1 design (see end of section).*

Constitution: `.specify/memory/constitution.md` v1.0.0 (ratified 2026-04-13).

| Article | Status | Notes |
|---|---|---|
| I. Participant-First Design | ✅ Pass | Three commands, all single-line copy-pasteable. Stage-3 README will document `swig` as the install prerequisite, the `--hf-repo` flag, and the `RL_WORKSHOP_DEVICE` override. All identifiers/docstrings in English. The `swig` missing case (SC-007) gets a clear error. |
| II. Two Paths, One Agent API (NON-NEGOTIABLE) | ✅ Pass | Both `train.py` and `train_sb3.py` end with the same artefact shape (`runs/car-racing/<run>/{meta.json, metrics.jsonl, model.{pt,zip}, eval.mp4}`). The `Agent` interface (`predict`, `train`, `save`, `load`) is unchanged: CNN selection is internal to `PPOAgent.__init__`. SB3's interaction with the Pi pipeline is unchanged. |
| III. Gymnasium-Conformant Environments | ✅ Pass | `gym.make_vec(..., wrappers=[Grayscale, Resize, FrameStack])` produces 5-tuple `step()` results across all envs; obs are `(num_envs, 4, 84, 84)`, action is `(num_envs, 3)`. The wrappers are stock Gymnasium 1.x — no homemade `ObservationWrapper`s. |
| IV. Test-Verified Implementation (NON-NEGOTIABLE) | ✅ Pass | New `@step("C8")` test in `test_agent_interface.py`: build `gym.make_vec("CarRacing-v3", num_envs=2, wrappers=[…], SAME_STEP)`, construct `PPOAgent` (auto-CNN), train for 256 timesteps, assert `stats` dict shape + finite losses + `< 10 s` budget. The SB3+HF path is **not** in the test suite (network-bound; covered by quickstart pre-flight). All five existing PPO TODO tests + 7 agent-interface tests must keep passing. |
| V. Progressive Scaffolding | ✅ Pass | `# TODO 1..5` markers in `ppo.py` are not touched. The CNN auto-detection lives in `__init__` — outside any TODO block. New `CnnActorNetwork` / `CnnCriticNetwork` are siblings of the existing MLP networks in `networks.py`. The `solutions` branch will need lockstep updates to keep the same five-TODO contract intact (tracked as a Complexity Tracking row). |
| VI. Fail-Safe Workshop Design | ✅ Pass | SB3 path (US1, P1) is the constitutional escape hatch — even if the custom PPO has bugs, participants reach a trained car. HuggingFace path (US3, P3) layers a second safety net on top. `RL_WORKSHOP_DEVICE=cpu` works for both. `pretrained/` directory remains the offline fallback. |
| VII. Sim-to-Real Pipeline Integrity | ⚠ **DEFERRED** | The `model.pt` saved by `train.py` on a Mac (MPS-tagged tensors in `state_dict`) will fail to load on a Raspberry Pi (CPU-only) without `map_location="cpu"` in `PPOAgent.load`. Feature 004 explicitly chose "no fix" for that line (user direction). This feature does not re-open that decision; the issue is tracked in Complexity Tracking and surfaced as a Workshop-2 follow-up — Workshop 1 stage 3 finishes at "model.pt on disk", not "model.pt running on the Pi". |

**Gate decision**: ✅ Proceed with one tracked deferral (Article VII, inherited from feature 004's "no fix" decision, surfaced explicitly here).

## Project Structure

### Documentation (this feature)

```text
specs/005-carracing-drivers/
├── plan.md              # This file
├── research.md          # Phase 0 — CNN architecture choice, HuggingFace integration shape, vec-env memory math, asymmetric-action-space caveat
├── data-model.md        # Phase 1 — meta.json delta (hf_repo_id, hf_filename, network_arch), CNN-network public surface
├── contracts/
│   ├── cli.md           # train.py + train_sb3.py CLI surfaces (flags, defaults, error messages)
│   └── meta-fields.md   # Delta against feature 002's run-format.md: the new optional HF + network_arch fields
├── quickstart.md        # Phase 1 — four flows: SB3 from-scratch smoke, custom-PPO from-scratch smoke, SB3 + HuggingFace fine-tune, workshop-leader pre-flight
├── checklists/
│   └── requirements.md  # already created by /speckit.specify
└── tasks.md             # Phase 2 output (created by /speckit.tasks, NOT by this command)
```

### Source Code (repository root)

```text
workshop-1/
├── 1-ppo/
│   └── ppo/
│       ├── networks.py                         # MODIFIED — add CnnActorNetwork, CnnCriticNetwork (Nature DQN backbone). MLP classes unchanged.
│       ├── ppo.py                              # MODIFIED — PPOAgent.__init__ auto-selects CNN vs MLP based on len(single_obs_space.shape). New `network_arch` field on the agent for introspection / metadata. No changes to TODO blocks 1–5.
│       └── tests/
│           └── test_agent_interface.py         # MODIFIED — add @step("C8") CNN smoke test on a (4, 84, 84) vector env.
├── 3-car-racing/
│   ├── train.py                                # REWRITTEN — thin custom-PPO driver mirroring 2-pendulum/train.py. Removes the broken CarRacingPPOAgent / agent.py path.
│   ├── train_sb3.py                            # REWRITTEN — thin SB3 driver with --hf-repo / --hf-filename flags. Removes the broken workshop-1 root imports.
│   ├── agent.py                                # DELETED — pre-refactor CarRacingPPOAgent class no longer needed (CNN selection is internal to PPOAgent).
│   └── README.md                               # MODIFIED (or NEW) — three commands documented, swig dep, RL_WORKSHOP_DEVICE override, --hf-repo usage.
pyproject.toml                                  # MODIFIED — add huggingface_hub>=0.20 to [dependency-groups].workshop1.
```

**Structure Decision**: Keep the existing single-package layout (`workshop-1/1-ppo/ppo/`) consumed by sibling driver scripts (`workshop-1/3-car-racing/{train.py, train_sb3.py}`). The CNN vs MLP selection is a private detail inside `PPOAgent.__init__` triggered by the observation-space shape; participants reading the agent see one `PPOAgent` class, not a hierarchy. The two CarRacing driver scripts are direct siblings of the Pendulum drivers and follow the same pattern, which keeps the workshop's mental model uniform across stages.

## Complexity Tracking

| Violation | Why Needed | Simpler Alternative Rejected Because |
|-----------|------------|--------------------------------------|
| **Article VII — `model.pt` not guaranteed to load cross-device** | Inherited from feature 004's explicit "no fix" decision on `map_location="cpu"` in `PPOAgent.load`. Adding it would touch a single line but reopen a decision already made. | This feature could just add the `map_location="cpu"` line and silently fix it. We don't because (a) the user's reasoning in 004 still applies — Workshop 1 doesn't deploy to the Pi, Workshop 2 does; (b) adding device-portability shims piecemeal across features fragments the decision. **Follow-up**: Workshop 2's first feature should re-open this decision explicitly, with a CarRacing-saved `.pt` round-tripping CPU on the Pi as the acceptance test. |
| **Article V — `solutions` branch lockstep update for CNN networks** | The `main` branch `ppo.py` keeps `# TODO 1..5` markers and `raise NotImplementedError` defaults; `solutions` has the filled-in versions. CNN classes added to `networks.py` on `main` need to merge cleanly onto `solutions` without duplicating or conflicting with the inlined solutions. | Skipping the `solutions` merge would break the participant fallback mechanism (`git checkout solutions -- <path>`). The merge itself is mechanical (CNN classes are net-new, not edits to existing code), but the implementer must remember to do it. **Follow-up**: same-PR `solutions` merge is part of the implementation task list. |

---

## Phase 0 — Outline & Research

Resolved decisions (full detail in `research.md`):

1. **CNN auto-detect heuristic.** Decision: `PPOAgent.__init__` inspects `single_obs_space.shape` — if length 3 (interpreted as `(channels, height, width)` after the FrameStack wrapper), instantiate `CnnActorNetwork` / `CnnCriticNetwork`; otherwise instantiate the existing MLP `ActorNetwork` / `CriticNetwork`. The chosen architecture is recorded as `self.network_arch ∈ {"mlp", "cnn"}` for introspection and `meta.json` attribution. Alternatives considered: a `network_arch="cnn"` hyperparameter (rejected — boilerplate, easy to mis-set), a `CnnPPOAgent` subclass (rejected — Article II prefers a single `PPOAgent` class).

2. **CNN architecture: Nature DQN backbone, separate actor and critic.** Decision: each of `CnnActorNetwork` / `CnnCriticNetwork` carries its own backbone (`Conv(32, 8×8, s=4) → Conv(64, 4×4, s=2) → Conv(64, 3×3, s=1) → flatten → Linear(512)`) followed by a small head. This doubles parameter count vs SB3's shared-trunk `ActorCriticCnnPolicy`, but matches the existing PPO package's separate-actor-separate-critic pattern (Article II compatibility, no special-casing in `PPOAgent`). Alternatives considered: shared trunk (cleaner but requires restructuring the agent), pretrained backbone from torchvision (out of scope — adds a download step).

3. **Input dtype: float32 normalised to [0, 1].** Decision: the rollout stores observations as float32 in `[0, 1]` (i.e., `obs / 255.0`). The wrapper chain produces uint8 `(4, 84, 84)`; `PPOAgent.train()` divides by 255 before feeding to the CNN, mirroring SB3's `CnnPolicy` behaviour. Alternative considered: keep uint8 in the rollout buffer (saves 4× memory), convert to float on each forward — rejected as a premature optimisation that complicates the buffer; the 230 MB budget at our default sizes is fine.

4. **HuggingFace integration via plain `huggingface_hub`.** Decision: add `huggingface_hub>=0.20` to `[dependency-groups].workshop1`. The driver calls `huggingface_hub.hf_hub_download(repo_id=…, filename=…)` and passes the resulting path to `stable_baselines3.PPO.load(path, env=env, device=…)`. Alternatives considered: `huggingface_sb3` package (rejected — a thin wrapper around the same `hf_hub_download` call, one extra dependency for one fewer line of code). Caching is the library default (`~/.cache/huggingface/hub/...`), which satisfies SC-004 (cache hit < 1 s).

5. **Default HuggingFace filename.** Decision: when `--hf-filename` is not provided, derive it from the repo: `<basename(repo_id)>.zip` (e.g., `sb3/ppo-CarRacing-v0` → `ppo-CarRacing-v0.zip`). This matches the SB3 / HuggingFace community convention. The `--hf-filename` flag overrides for repos that follow a different convention. No default `--hf-repo` value: a participant must explicitly opt in.

6. **Vector env shape.** Decision: `num_envs=4` for both drivers, matching Pendulum. Memory math (`size_per_env × num_envs × 4 × 84 × 84 × 4 bytes`) at `rollout_size=2048` → 230 MB rollout buffer, well within laptop RAM. CarRacing's per-env memory is dominated by Box2D physics state, not pixel buffers; 4 envs is the comfortable default. Alternative considered: `num_envs=2` for safety (rejected — wastes the 004 vectorisation work; 4 envs runs fine in dev tests).

7. **Asymmetric action-space caveat.** Documented (not fixed in this feature): CarRacing actions are `Box([-1, 0, 0], [1, 1, 1])`. PPO's `Normal(mean, std).sample().clamp(low, high)` wastes ~half of gas/brake samples (clamped to 0). The workshop SCs do not require convergence in test budgets, so this is acceptable. Follow-up: switch to a `TanhNormal` or rescaled `Beta` distribution for asymmetric continuous actions — out of scope here.

8. **`huggingface_hub` offline behaviour.** Decision: when offline, `hf_hub_download` raises `requests.exceptions.ConnectionError` (or `huggingface_hub.utils.LocalEntryNotFoundError` if cache empty). The driver catches the union of these and re-raises with the actionable message specified in spec FR-009. The HF cache may already contain a previously-downloaded artefact for this `repo_id`/`filename`, in which case `hf_hub_download` returns that path even with no network — that satisfies SC-004 directly without any custom logic.

**Output**: `research.md` consolidating these eight items.

## Phase 1 — Design & Contracts

**Prerequisites:** `research.md` complete.

1. **Data model** (`data-model.md`):
   - `PPOAgent.network_arch` — string field on the agent, one of `"mlp"` or `"cnn"`, set by `__init__` based on observation-space shape; included in `meta.json` for retrospective attribution.
   - `RunMetadata.{hf_repo_id, hf_filename}` — two new optional string fields on `meta.json` (extends feature 002's `run-format.md`). `null` for from-scratch runs; populated when the SB3 driver loaded weights via `--hf-repo`.

2. **Contracts** (`contracts/`):
   - `cli.md` — full CLI surface for both drivers: `--timesteps`, `--seed`, `--run-name`, `--no-eval`, `--force` (mirroring Pendulum), plus `train_sb3.py`-only `--hf-repo`, `--hf-filename`. Includes the contract-mandated error messages for each failure mode (FR-009 wording).
   - `meta-fields.md` — delta against `specs/002-training-and-visualization/contracts/run-format.md`: the two new optional HF fields, the `network_arch` field, and the same `device` field that 004 added. Reader compatibility rule: missing fields = `null` / `"unknown"`, never raise.

3. **Quickstart** (`quickstart.md`): four flows — (A) SB3 from-scratch smoke, (B) custom-PPO from-scratch smoke (verifies CNN auto-detection works), (C) SB3 + HuggingFace fine-tune, (D) workshop-leader pre-flight (re-runs all three with `RL_WORKSHOP_DEVICE=cpu` and `=mps` to validate the device-fix retroactive win on CarRacing CNN per SC-005).

4. **Agent context update**: Run `.specify/scripts/bash/update-agent-context.sh claude` to refresh `CLAUDE.md`'s "Active Technologies" / "Recent Changes" sections with the `huggingface_hub` addition.

**Output**: `data-model.md`, `contracts/cli.md`, `contracts/meta-fields.md`, `quickstart.md`, refreshed `CLAUDE.md`.

## Constitution Check (Post-Design)

Re-evaluated after Phase 1 design — no change in posture. All articles still pass except VII, which remains DEFERRED with the same Workshop-2 follow-up note. The design choices in research.md / data-model.md / contracts (auto-detect via obs shape, separate CNN actor/critic matching the existing pattern, plain `huggingface_hub` over `huggingface_sb3`) reinforce Articles II / III / IV / V / VI without introducing new violations.

**Gate decision**: ✅ Proceed to `/speckit.tasks`.
