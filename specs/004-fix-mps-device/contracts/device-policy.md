# Contract: Device-Selection Policy

This is the public, testable contract of `workshop-1/1-ppo/ppo/utils/utils.py:get_device()`. The decision table here is the single source of truth that `tests/test_ppo.py`'s device-policy step mirrors and that the override mechanism documentation in `README.md` summarises.

## Public surface

```python
# workshop-1/1-ppo/ppo/utils/utils.py
def get_device() -> torch.device: ...

class DeviceUnavailableError(RuntimeError): ...
```

Both names are re-exported from `workshop-1/1-ppo/ppo/utils/__init__.py` (alongside the existing `seed_everything`, `format_update_line`).

## Inputs

| Source | Name | Notes |
|---|---|---|
| Process env | `RL_WORKSHOP_DEVICE` | Optional. Allowed: `cpu`, `cuda`, `mps`, `auto`. Default: `auto`. Case-insensitive, whitespace stripped. |
| PyTorch | `torch.cuda.is_available()` | Sampled at call time (not memoised). |
| PyTorch | `torch.backends.mps.is_available()` | Sampled at call time. |

## Outputs

`torch.device` of type `cpu`, `cuda`, or `mps`.

## Side effects

When the resolved device's type is `mps`:

```python
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
```

`setdefault` (not `=`) so that a user who has explicitly exported `PYTORCH_ENABLE_MPS_FALLBACK=0` is honoured.

When the resolved device is `cpu` or `cuda`, no side effects.

## Decision table

| `RL_WORKSHOP_DEVICE` | `cuda?` | `mps?` | Returns | Side effect |
|---|---|---|---|---|
| (unset) / `auto` | T | T | `device(type='cuda')` | none |
| (unset) / `auto` | T | F | `device(type='cuda')` | none |
| (unset) / `auto` | F | T | `device(type='mps')` | sets `MPS_FALLBACK=1` |
| (unset) / `auto` | F | F | `device(type='cpu')` | none |
| `cpu`            | * | * | `device(type='cpu')` | none |
| `cuda`           | T | * | `device(type='cuda')` | none |
| `cuda`           | F | * | **`DeviceUnavailableError`** | none |
| `mps`            | * | T | `device(type='mps')` | sets `MPS_FALLBACK=1` |
| `mps`            | * | F | **`DeviceUnavailableError`** | none |
| any other string | * | * | **`DeviceUnavailableError`** | none |

(`*` = don't care.)

## Error message contract

`DeviceUnavailableError` messages MUST contain three things in order:

1. The literal value the user requested (quoted).
2. Why it can't be honoured (PyTorch capability).
3. The set of devices that ARE available, plus the actionable next step.

Examples:

```
RL_WORKSHOP_DEVICE='cuda' but no CUDA backend is available on this machine.
Available: cpu, mps. Either unset RL_WORKSHOP_DEVICE (auto-select) or set it
to 'cpu' or 'mps'.
```

```
RL_WORKSHOP_DEVICE='gpu' is not a recognised value. Allowed: cpu, cuda, mps,
auto. Either unset it or pick one.
```

## Invariants

- **Pure (apart from the documented env-var side effect).** No I/O, no logging from inside `get_device()`. Logging of the resolved device happens in the *caller* (`PPOAgent.__init__` or the run logger), not here.
- **Idempotent.** Two calls in the same process with the same `RL_WORKSHOP_DEVICE` MUST return equal `torch.device` objects. The env-var side effect uses `setdefault`, so the second call is a no-op.
- **Safe to call before any tensor is allocated.** This is the property that makes the `MPS_FALLBACK` side effect take effect on the first MPS op.
- **No silent fallback.** If the user requests `cuda` and CUDA is unavailable, the policy raises rather than substituting.

## Test mapping

| Decision-table row | Step in `test_ppo.py` | Notes |
|---|---|---|
| `auto` + CUDA available | `test_device_auto_cuda` | Skipped on machines without CUDA. |
| `auto` + MPS available | `test_device_auto_mps` | Skipped on machines without MPS. |
| `auto` + neither | `test_device_auto_cpu` | Always runnable. |
| `cpu` (forced) | `test_device_force_cpu` | Always runnable. |
| `cuda` on no-CUDA machine | `test_device_cuda_unavailable_raises` | Skipped on CUDA hosts. |
| `mps` on no-MPS machine | `test_device_mps_unavailable_raises` | Skipped on MPS hosts. |
| Garbage string | `test_device_invalid_value_raises` | Always runnable. |
| MPS resolved → env var set | `test_device_mps_sets_fallback_env` | Skipped on machines without MPS. |

Skips MUST print a single-line `[skipped: <reason>]` and count as passing in the runner's exit code.
