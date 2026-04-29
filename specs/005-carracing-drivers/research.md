# Phase 0 Research: CarRacing Training Drivers

This file consolidates the eight decisions referenced in `plan.md` § Phase 0. Each item is `Decision → Rationale → Alternatives considered`.

## R1. CNN auto-detect via observation shape

**Decision**: `PPOAgent.__init__` inspects `single_obs_space.shape`:

- `len(shape) == 1` → MLP path (existing `ActorNetwork(obs_dim)` and `CriticNetwork(obs_dim)`). `self.network_arch = "mlp"`.
- `len(shape) == 3` → CNN path (new `CnnActorNetwork(in_channels, action_dim)` and `CnnCriticNetwork(in_channels)`, where `in_channels = shape[0]`). `self.network_arch = "cnn"`.
- Anything else → raise `ValueError` with a message naming the unsupported shape.

**Rationale**: The decision is unambiguous from the env's own metadata, so participants don't have to thread a config knob through. After the standard wrapper chain `Grayscale → Resize(84, 84) → FrameStackObservation(4)`, CarRacing's `single_observation_space.shape` is `(4, 84, 84)` (channels-first because that's how `FrameStackObservation` stacks), which matches PyTorch's `Conv2d(in_channels, …)` convention directly.

**Alternatives considered**:
- *`network_arch="cnn"` hyperparameter.* Rejected — adds boilerplate to every driver, easy to mis-set and hard to error on, doesn't survive `agent.load()` (would have to be persisted in `state` dict).
- *`CnnPPOAgent(PPOAgent)` subclass.* Rejected — Article II prefers a single `PPOAgent` class. Subclassing for what is essentially "different network type" multiplies the API surface for participants.
- *Inspect `dtype == uint8` instead of shape.* Rejected — couples architecture choice to dtype, which is fragile if a future preprocessing wrapper produces float32 pixel obs. Shape is the more durable signal.

## R2. CNN architecture: Nature DQN backbone, separate actor and critic

**Decision**: Add two classes to `workshop-1/1-ppo/ppo/networks.py`:

```python
class CnnActorNetwork(nn.Module):
    """Nature DQN backbone (3 conv + FC) + linear policy-mean head."""
    def __init__(self, in_channels: int, action_dim: int, hidden: int = 512): ...

class CnnCriticNetwork(nn.Module):
    """Nature DQN backbone (3 conv + FC) + linear scalar value head."""
    def __init__(self, in_channels: int, hidden: int = 512): ...
```

Backbone: `Conv(in_channels, 32, kernel=8, stride=4) → ReLU → Conv(32, 64, kernel=4, stride=2) → ReLU → Conv(64, 64, kernel=3, stride=1) → ReLU → flatten → Linear(64*7*7, 512) → ReLU`. This is the canonical 84×84 → 512-dim feature extractor used by SB3's `CnnPolicy`, DeepMind's Atari DQN, and most PPO-on-pixels papers. Actor head: `Linear(512, action_dim)`. Critic head: `Linear(512, 1)`.

Each of `CnnActorNetwork` and `CnnCriticNetwork` carries its own backbone (no shared trunk). `log_std` stays a separate `nn.Parameter` owned by `PPOAgent`, exactly as in the MLP path.

**Rationale**: Matches the existing PPO package's "separate actor, separate critic" pattern (see `ActorNetwork` / `CriticNetwork` in the same file). This avoids special-casing `PPOAgent` to know about a shared trunk, keeps the optimizer construction (`list(actor.parameters()) + list(critic.parameters()) + [log_std]`) identical between MLP and CNN paths, and keeps the `state_dict` shape simple (separate actor and critic state dicts, exactly like Pendulum). Doubles the parameter count vs SB3's shared-trunk policy, but the workshop is teaching, not optimising — readability wins over parameter efficiency.

**Alternatives considered**:
- *Shared trunk* (matching SB3's `ActorCriticCnnPolicy`). Rejected — would require restructuring `PPOAgent.__init__`, splitting the optimizer parameter list, and threading a "feature extractor" abstraction through `train()` / `predict()`. Big change for a teaching codebase.
- *Pretrained backbone from torchvision.* Rejected — adds a download step at first construction, complicates the `solutions` branch, doesn't match the SB3 path (which trains from scratch).
- *Smaller IMPALA-style backbone.* Rejected — Nature DQN is the best-documented baseline; switching to IMPALA would be an unjustified deviation from the well-known reference.

## R3. Input dtype: float32 normalised to [0, 1]

**Decision**: The wrapper chain produces uint8 `(4, 84, 84)` observations. `PPOAgent.train()` (and `predict()`) divides obs by 255.0 before feeding to the CNN, materialising float32 tensors in `[0, 1]`. The `RolloutBuffer` stores float32. The CNN's first `Conv2d` expects float32 input.

**Rationale**: SB3's `CnnPolicy` does the same normalisation internally — matching this keeps a participant's mental model consistent across paths. The 4× memory cost (uint8 → float32) at `rollout_size=2048, num_envs=4, (4, 84, 84)` is 230 MB, well within laptop RAM. Storing uint8 and converting per-forward saves 75% of buffer memory but adds a `.float() / 255.0` op in the hot path — a complication not justified at workshop scale.

**Alternatives considered**:
- *Keep uint8 in buffer, convert in-forward.* Rejected — premature optimisation; complicates the rollout buffer's TODO 1 (compute_gae) which assumes float32 throughout.
- *Normalise to [-1, 1].* Rejected — `[0, 1]` is the SB3/PyTorch convention for image obs, and the workshop is following established practice.

## R4. HuggingFace integration via plain `huggingface_hub`

**Decision**: Add `huggingface_hub>=0.20` to `[dependency-groups].workshop1` in `pyproject.toml`. The SB3 driver imports `from huggingface_hub import hf_hub_download` and calls:

```python
local_path = hf_hub_download(repo_id=args.hf_repo, filename=args.hf_filename or _default_filename(args.hf_repo))
model = stable_baselines3.PPO.load(local_path, env=env, device=device)
model.learn(total_timesteps=args.timesteps, callback=callback)
```

`hf_hub_download` handles caching automatically (default `~/.cache/huggingface/hub/`); a second call with the same `repo_id`/`filename` returns the cached path in milliseconds without network I/O.

**Rationale**: `huggingface_hub` is the official upstream dependency for everything HuggingFace. Adding `huggingface_sb3` (the SB3-specific helper) on top would mean two new packages instead of one, for a `load_from_hub` wrapper that's literally `hf_hub_download` + `PPO.load`. We use the wrapper inline in our driver instead.

**Alternatives considered**:
- *`huggingface_sb3.load_from_hub`.* Rejected — extra dependency for the same call. Saves one line.
- *Manual download via `requests`.* Rejected — reinvents caching, retry, file integrity; `huggingface_hub` already does all of this.
- *Vendor the artefact in `pretrained/`.* Rejected — that's an offline-only path; the user's spec specifically asks for online HuggingFace fetch, and the workshop benefits from "fresh community models without a repo update".

## R5. Default HuggingFace filename

**Decision**: When `--hf-filename` is omitted, derive the filename as `<basename(repo_id)>.zip`. Examples:

- `--hf-repo sb3/ppo-CarRacing-v0` → filename `ppo-CarRacing-v0.zip`
- `--hf-repo my-user/my-model` → filename `my-model.zip`

If the resulting filename is wrong for a particular repo, the participant overrides with `--hf-filename`. There is no default `--hf-repo` value: a participant must explicitly name a repo.

**Rationale**: This convention matches the published SB3 / HuggingFace community examples (`sb3/ppo-CarRacing-v0` indeed contains `ppo-CarRacing-v0.zip` as its model file, by their convention). Auto-deriving means the participant types one flag (`--hf-repo …`) instead of two for the common case. The override flag covers the long tail of repos that don't follow the convention, with a clear error message when the auto-derived filename doesn't exist.

**Alternatives considered**:
- *Single `--hf-spec` flag of the form `repo:filename`.* Rejected — less idiomatic for argparse; harder to give good error messages on malformed input.
- *Default `--hf-repo` to a canonical value (`sb3/ppo-CarRacing-v0`).* Rejected — pins us to one specific community model that we don't control. If that repo disappears or changes, our workshop breaks.
- *Glob-match in the repo for the first `*.zip`.* Rejected — would need an extra `huggingface_hub.list_repo_files()` call before download, and would silently pick something arbitrary.

## R6. Vector env shape: `num_envs=4`

**Decision**: Both `train.py` and `train_sb3.py` default to `num_envs=4`, matching Pendulum and the patterns established in feature 004. Per-env wrapper chain: `[GrayscaleObservation, ResizeObservation((84, 84)), FrameStackObservation(4)]`. Autoreset mode `SAME_STEP`.

**Rationale**: Memory math is comfortable (230 MB rollout buffer at our default sizes; CarRacing's Box2D physics state per env is also small). Vectorisation is the win that paid for itself in feature 004 — using `num_envs=4` here ensures the CarRacing CNN gets enough work per forward to amortise MPS kernel-launch overhead, which is the SC-005 retroactive validation.

**Alternatives considered**:
- *`num_envs=2`.* Rejected — wastes the 004 vectorisation work; the memory headroom is fine at 4.
- *`num_envs=8`.* Rejected — pushes rollout buffer toward 460 MB, edges the smoke test toward slower wall-clock per update on participants' less-RAM machines.

## R7. Asymmetric action-space caveat

**Documented (not fixed in this feature)**: CarRacing actions are `Box([-1, 0, 0], [1, 1, 1])` — steer is symmetric on `[-1, 1]`, gas and brake are non-negative on `[0, 1]`. PPO's standard `Normal(mean, log_std.exp()).sample().clamp(action_min, action_max)` produces samples that are symmetric around the mean, so when `mean ≈ 0` and `log_std_init = 0` (std = 1), roughly half the gas/brake samples land below 0 and clamp to 0 — wasted exploration. The clamp's gradient is zero on clamped samples, so the policy can't improve gas/brake exploration via the value gradient.

**Why we don't fix it here**: The spec's success criteria do not require convergence within smoke-test budgets — SC-001 / SC-002 / SC-003 measure wall-clock to "training complete", not "trained to a target reward". Within 10 000 timesteps, neither MLP nor CNN PPO is going to hit competent driving anyway. A proper fix (TanhNormal, rescaled Beta, or a learned action-bounds-aware distribution) is a meaningful research-side change, not a stage-3 driver concern.

**Follow-up**: A future feature could swap PPO's `Normal` distribution for `TanhNormal` (squash with tanh, then rescale to `[low, high]`). That would be a cross-cutting change across MLP + CNN paths and would also help the Pendulum case (whose actions are symmetric `[-2, 2]` so it suffers less, but not zero).

## R8. `huggingface_hub` offline behaviour

**Decision**: `hf_hub_download` raises one of:

- `huggingface_hub.utils.LocalEntryNotFoundError` — when offline AND nothing is in the local cache for this `repo_id`/`filename`.
- `huggingface_hub.utils.RepositoryNotFoundError` — when online but the repo doesn't exist.
- `huggingface_hub.utils.EntryNotFoundError` — when online and repo exists but the named file is not in it.
- Plain `requests.exceptions.ConnectionError` / `OSError` — when the call truly times out.

The SB3 driver catches the union of these and re-raises a single project-level `HuggingFaceLoadError` with the contract-mandated message (FR-009): names the `repo_id`/`filename`, names the underlying cause, suggests the offline alternative (`pretrained/`).

**Rationale**: A union-catch with a clear re-raise is the simplest pattern that hides the `huggingface_hub` API surface from the workshop participant. Reading the participant-facing error, they get one actionable line, not a Python traceback through three packages.

**Cache-hit silent-success path**: If the cache already has a fresh entry for this `repo_id`/`filename` (because the participant ran the command earlier today, or copied a cache from a peer), `hf_hub_download` returns the cached path in milliseconds with no network calls — that satisfies SC-004 directly without any project-level cache logic.

**Alternatives considered**:
- *Probe network connectivity ahead of `hf_hub_download`.* Rejected — adds latency to the happy path; the library handles this correctly.
- *Pre-validate the `repo_id`/`filename` via `huggingface_hub.HfApi`.* Rejected — extra round-trip; the download itself fails fast with an actionable message anyway.
