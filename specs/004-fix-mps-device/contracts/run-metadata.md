# Contract: Run-metadata `device` field

This is a *delta* against feature 002's `specs/002-training-and-visualization/contracts/run-format.md` — that document owns the full schema of `runs/<stage>/<run-name>/metadata.json`. This feature only pins down the `device` field's shape and write-time guarantees.

## Field

| Key      | Type   | Required (after this feature lands) | Allowed values                              |
|----------|--------|-------------------------------------|---------------------------------------------|
| `device` | string | yes, for runs produced by custom-PPO drivers (`workshop-1/2-pendulum/train.py`, `workshop-1/3-car-racing/train.py`) | `"cpu"`, `"cuda"`, `"mps"` |

The string is the `torch.device.type` of the resolved device — i.e. the third row of `torch.device("mps").__repr__()` / `.type`, never the full repr. Specifically:

```python
metadata["device"] = get_device().type   # "cpu" | "cuda" | "mps"
```

## When written

At run start, after `PPOAgent` is constructed, before the first PPO update logs to `train.jsonl`. Once written, the `device` field is **immutable for the run** — even if a downstream step reads `get_device()` again, the recorded value is not refreshed. (Practically there is no scenario where it would change inside a run, because `get_device()` is idempotent and PyTorch capabilities don't change mid-process.)

## Source of truth

The only writer is the `RunLogger` (defined in `workshop-1/1-ppo/ppo/utils/_runlog.py`). It reads `agent.device` (preferred — captures the agent's actual device) and falls back to `get_device().type` when no agent is provided.

`agent.device` is preferred because in pathological future scenarios where `PPOAgent` accepts a manual `device=` override (not currently planned), the metadata must reflect what the agent *uses*, not what the policy returned. With current code the two are identical.

## Backwards compatibility

Older `metadata.json` files (produced before this feature, e.g. those committed under `pretrained/sample-runs/`) may not have a `device` field. **Readers** of `metadata.json` (notebooks, ad-hoc scripts) MUST tolerate the field being absent and treat that as `device: "unknown"`. They MUST NOT raise.

The `analyze.ipynb` notebooks under each stage MUST be updated in this feature to read the field defensively, e.g.:

```python
device = meta.get("device", "unknown")
```

## SB3 path

Out of scope for this feature. The Stable-Baselines3 callback (`_sb3_jsonl_callback.py`) writes its own metadata; aligning its `device` field with this contract is a follow-up. The "(after this feature lands)" qualifier in the table above is therefore restricted to custom-PPO runs.

## Test mapping

| Behaviour | Step in `test_agent_interface.py` |
|---|---|
| Run-start writes `device` matching `get_device().type` | `test_metadata_device_recorded` |
| Reader tolerates missing field on old metadata | `test_metadata_device_missing_is_unknown` (uses a fixture without the field) |
