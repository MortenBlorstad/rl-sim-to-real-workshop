# Contract: `meta.json` field deltas for stage-3 runs

This is a *delta* against feature 002's `specs/002-training-and-visualization/contracts/run-format.md` (which owns the canonical schema) and feature 004's `specs/004-fix-mps-device/contracts/run-metadata.md` (which added the `device` field). This file pins the new fields stage-3 introduces.

## New fields

| Key             | Type            | Required for                  | Allowed values                       | Source                                |
|-----------------|-----------------|-------------------------------|--------------------------------------|---------------------------------------|
| `network_arch`  | string          | custom-PPO runs               | `"mlp"` \| `"cnn"`                   | `agent.network_arch`                  |
| `hf_repo_id`    | string \| null  | SB3 runs                      | any HuggingFace `repo_id`             | `args.hf_repo` or `null`              |
| `hf_filename`   | string \| null  | SB3 runs                      | any HF artefact filename              | `args.hf_filename` (auto-derived if omitted) or `null` |

All three fields are written **once at run start** (when the `RunLogger` constructs `meta.json`) and never mutated afterwards.

### `network_arch` rules

- For custom-PPO runs (`workshop-1/3-car-racing/train.py`, `workshop-1/2-pendulum/train.py`): the field MUST be present and equal to the agent's actual `network_arch` (`"mlp"` for Pendulum, `"cnn"` for CarRacing).
- For SB3 runs: the field MAY be present (e.g. set to `"cnn"` to match the SB3 `CnnPolicy`) or omitted. Recommended: present, with value derived from the SB3 policy class name.
- For older runs predating this feature: the field is absent. Readers MUST tolerate absence and treat as `"unknown"`.

### `hf_repo_id` and `hf_filename` rules

- For SB3 runs: BOTH fields MUST be present in every SB3-produced `meta.json`. They are `null` for from-scratch runs and concrete strings for HuggingFace fine-tune runs. Always-present-with-null lets notebooks rely on key existence.
- For custom-PPO runs: the fields are not applicable (the custom PPO does not load from HuggingFace in this feature). The fields MAY be omitted.
- Either both are non-null or both are null — never one without the other.

## Reader compatibility

```python
# Notebook-side defensive read (extends the pattern from feature 004):
device       = meta.get("device", "unknown")
arch         = meta.get("network_arch", "unknown")
hf_repo      = meta.get("hf_repo_id")          # None for from-scratch (or absent)
hf_filename  = meta.get("hf_filename")
is_finetune  = hf_repo is not None
```

The `analyze.ipynb` notebooks under `workshop-1/2-pendulum/` and `workshop-1/3-car-racing/` MUST be updated in this feature to include these defensive reads in any cell that accesses the new fields.

## Example `meta.json` snippets

### Custom-PPO from-scratch CarRacing run (this feature)

```json
{
  "stage": "car-racing",
  "env_id": "CarRacing-v3",
  "agent_class": "PPOAgent",
  "network_arch": "cnn",
  "device": "mps",
  "seed": 42,
  "total_timesteps": 10000,
  "...": "(other fields per run-format.md)",
  "started_at": "2026-04-29T...",
  "status": "ok"
}
```

### SB3 from-scratch CarRacing run (this feature)

```json
{
  "stage": "car-racing",
  "env_id": "CarRacing-v3",
  "agent_class": "sb3.PPO[CnnPolicy]",
  "network_arch": "cnn",
  "device": "mps",
  "hf_repo_id": null,
  "hf_filename": null,
  "...": "(other fields)",
  "status": "ok"
}
```

### SB3 fine-tune-from-HuggingFace CarRacing run (this feature)

```json
{
  "stage": "car-racing",
  "env_id": "CarRacing-v3",
  "agent_class": "sb3.PPO[CnnPolicy]",
  "network_arch": "cnn",
  "device": "mps",
  "hf_repo_id": "sb3/ppo-CarRacing-v0",
  "hf_filename": "ppo-CarRacing-v0.zip",
  "...": "(other fields)",
  "status": "ok"
}
```

### Older Pendulum run (pre-feature-004) — defensive reading required

```json
{
  "stage": "pendulum",
  "env_id": "Pendulum-v1",
  "agent_class": "PPOAgent",
  "...": "(no device, no network_arch, no hf_* fields)"
}
```

A notebook reading this must apply the defaults via `.get(..., "unknown")` / `.get(..., None)`. No raise on absent keys.
