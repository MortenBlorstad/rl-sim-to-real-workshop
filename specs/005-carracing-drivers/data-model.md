# Phase 1 Data Model: CarRacing Training Drivers

This feature does not introduce a database, network protocol, or persistent schema. The "data model" is two named, well-typed surfaces that the implementation must honour:

1. The `PPOAgent.network_arch` field (architecture introspection).
2. The new optional fields on `meta.json` (`hf_repo_id`, `hf_filename`, `network_arch`).

The CNN network classes are public API surface in the `ppo` package; their signatures are pinned in `contracts/cli.md` (treated as a Python contract for participants reading the package).

---

## 1. `PPOAgent.network_arch`

A string field set in `__init__` and never mutated afterwards.

### Values

| Value   | Set when                                                                                  | Networks instantiated                                |
|---------|-------------------------------------------------------------------------------------------|------------------------------------------------------|
| `"mlp"` | `len(env.single_observation_space.shape) == 1` (vector obs, e.g. Pendulum's `(3,)`)        | `ActorNetwork(obs_dim, action_dim)`, `CriticNetwork(obs_dim)` |
| `"cnn"` | `len(env.single_observation_space.shape) == 3` (image obs, e.g. CarRacing's `(4, 84, 84)`) | `CnnActorNetwork(in_channels, action_dim)`, `CnnCriticNetwork(in_channels)` |

Any other shape MUST raise `ValueError` with a message that names the unsupported shape and the two supported shapes.

### Invariants

- **Set once.** Assigned during `__init__` from the env's observation space. Not a hyperparameter, not in `DEFAULT_HYPERPARAMS`, not user-settable from outside.
- **Persisted.** `Agent.save()` writes `network_arch` into the saved `state` dict so `Agent.load()` can verify the loaded weights match the architecture the env implies. If a mismatch is detected at load time (saved arch ≠ env-implied arch), `Agent.load` raises a clear error.
- **Surfaced to metadata.** `RunLogger` reads `agent.network_arch` and writes it into `meta.json` (see § 2 below).

---

## 2. `meta.json` field deltas

Lives in `runs/<stage>/<run-name>/meta.json`. Schema owned by feature 002's `run-format.md`. This feature **adds** three fields (all optional, all written at run start, none mutated afterwards):

| Key             | Type            | Required           | Allowed values                  | Source of value                                     |
|-----------------|-----------------|--------------------|---------------------------------|----------------------------------------------------|
| `network_arch`  | string          | yes (custom-PPO runs) | `"mlp"` \| `"cnn"`              | `agent.network_arch`                                |
| `hf_repo_id`    | string \| null  | yes (SB3 runs)     | any HuggingFace `repo_id`        | `args.hf_repo` (or `null` if from-scratch)          |
| `hf_filename`   | string \| null  | yes (SB3 runs)     | any HuggingFace artefact filename | `args.hf_filename` (or auto-derived; see R5)        |

For SB3 runs that did not load from HuggingFace: both `hf_repo_id` and `hf_filename` MUST be present and set to `null` (so notebooks can use a `meta["hf_repo_id"] is not None` check to distinguish fine-tune-from-HF runs).

For custom-PPO runs: `hf_*` fields MAY be omitted entirely (the custom PPO does not load from HF in this feature; a future feature could change this).

### Backwards compatibility

Older `meta.json` files (produced before this feature, e.g. those committed under `pretrained/sample-runs/`) do not have these fields. **Readers** (notebooks, ad-hoc scripts) MUST tolerate the fields being absent and treat them as `null` / `"unknown"`. They MUST NOT raise.

The `analyze.ipynb` notebook that already does `meta.get("device", "unknown")` (per feature 004's contract) extends to:

```python
arch = meta.get("network_arch", "unknown")
hf_repo = meta.get("hf_repo_id")           # None for from-scratch
hf_filename = meta.get("hf_filename")
```

---

## 3. CNN network public surface

Public classes added to `workshop-1/1-ppo/ppo/networks.py`. Pinned in this data model so the implementation does not drift.

### `CnnActorNetwork`

```python
class CnnActorNetwork(nn.Module):
    """Nature DQN backbone (3 conv + FC) + linear policy-mean head.

    Args:
        in_channels: number of input channels (typically 4 after FrameStack).
        action_dim:  continuous-action dimensionality.
        hidden:      FC bottleneck width (default 512, matches Nature DQN).

    forward(obs) accepts:
        obs: torch.Tensor of shape (B, in_channels, 84, 84), float32 in [0, 1].
    Returns:
        action_mean: torch.Tensor of shape (B, action_dim).

    Single-obs shape (in_channels, 84, 84) is also accepted and produces (action_dim,).
    """
```

### `CnnCriticNetwork`

```python
class CnnCriticNetwork(nn.Module):
    """Nature DQN backbone (3 conv + FC) + linear scalar value head.

    forward(obs) accepts:
        obs: torch.Tensor of shape (B, in_channels, 84, 84), float32 in [0, 1].
    Returns:
        value: torch.Tensor of shape (B,).
    """
```

### Initialisation

Both networks orthogonal-init their conv and FC weights with gain `sqrt(2)` (matching the existing MLP networks' `_orthogonal_init` pattern); the policy head uses gain `0.01` (small init, also matching the MLP `ActorNetwork` head); the value head uses gain `1.0`.

### Backbone (shared definition)

```text
Conv2d(in_channels, 32, kernel=8, stride=4) → ReLU
Conv2d(32, 64, kernel=4, stride=2)          → ReLU
Conv2d(64, 64, kernel=3, stride=1)          → ReLU
flatten                                       (B, 64*7*7)
Linear(64*7*7, hidden)                       → ReLU
```

Hidden output is `(B, 512)` for the default `hidden=512`. Heads consume that and produce `(B, action_dim)` (actor) or `(B,)` (critic, after `.squeeze(-1)`).

### What's NOT shared

There is no shared trunk between actor and critic — each network instantiates its own backbone. This intentionally matches the existing MLP pattern (`ActorNetwork` and `CriticNetwork` are also separate). Per `research.md` R2, this is the conscious choice that makes `PPOAgent.__init__` not need to know about a shared feature extractor.
