# Phase 1 Data Model: Fix Device Selection

This feature does not introduce a database, network protocol, or persistent schema. The "data model" here is two named, well-typed surfaces that the code must honour:

1. The pure decision function `DevicePolicy`.
2. The `device` field on the existing per-run metadata record.

Both are pinned here so the contracts in `contracts/` and the tasks in `tasks.md` (next phase) reference a single source of truth.

---

## 1. `DevicePolicy`

A pure function that maps `(host capabilities, user override)` to a `torch.device`, with one documented side effect.

### Signature

```python
def get_device() -> torch.device:
    """Resolve the active device per the workshop's policy.

    Inputs (read from the process environment, not parameters):
      - RL_WORKSHOP_DEVICE: one of "cpu", "cuda", "mps", "auto" (default: "auto").
        Case-insensitive. Whitespace stripped.
      - torch.cuda.is_available()
      - torch.backends.mps.is_available()

    Side effect (only when the resolved device is MPS):
      os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

    Raises:
      DeviceUnavailableError if RL_WORKSHOP_DEVICE names a device the local
      machine cannot honour (e.g., "cuda" on a Mac without CUDA).
    """
```

### Decision table

| `RL_WORKSHOP_DEVICE` | CUDA available | MPS available | Result        | Side effect              |
|----------------------|----------------|---------------|---------------|--------------------------|
| `auto` (or unset)    | yes            | (any)         | `cuda`        | —                        |
| `auto`               | no             | yes           | `mps`         | set `MPS_FALLBACK=1`     |
| `auto`               | no             | no            | `cpu`         | —                        |
| `cpu`                | (any)          | (any)         | `cpu`         | —                        |
| `cuda`               | yes            | (any)         | `cuda`        | —                        |
| `cuda`               | no             | (any)         | **raise**     | —                        |
| `mps`                | (any)          | yes           | `mps`         | set `MPS_FALLBACK=1`     |
| `mps`                | (any)          | no            | **raise**     | —                        |
| any other value      | (any)          | (any)         | **raise**     | —                        |

### Error type

```python
class DeviceUnavailableError(RuntimeError):
    """Raised when RL_WORKSHOP_DEVICE names a device the local machine cannot
    honour. The message MUST list the requested device and the set of devices
    that ARE available, so the participant can self-correct."""
```

Example message: `"RL_WORKSHOP_DEVICE='cuda' but no CUDA backend is available on this machine. Available: cpu, mps. Either unset RL_WORKSHOP_DEVICE (auto-select) or set it to 'cpu' or 'mps'."`

### Invariants

- `get_device()` is **idempotent within a process**: calling it twice with the same environment must return equal `torch.device` objects and must not double-set the env var (handled by `setdefault`).
- `get_device()` is **safe to call before any tensor is allocated**: this is the timing guarantee that lets `PYTORCH_ENABLE_MPS_FALLBACK=1` take effect on the first MPS op (per spec clarification Q1).
- `get_device()` does **not** import non-stdlib modules eagerly beyond `torch` (already a dependency). No `psutil`, no `platform`-based heuristics — the policy is purely capability-driven.

---

## 2. `RunMetadata.device` field

Lives in the existing `runs/<stage>/<run-name>/metadata.json` file (schema owned by feature 002's `run-format.md`). This feature only ensures the field is *populated* with the correct value for custom-PPO runs.

### Field

| Key      | Type   | Required | Allowed values                  | Written when     |
|----------|--------|----------|---------------------------------|------------------|
| `device` | string | yes      | `"cpu"` \| `"cuda"` \| `"mps"`  | At run start; never mutated. |

### Source of value

`metadata["device"] = str(get_device())` evaluated **after** `RolloutBuffer` and `PPOAgent` have been constructed, so it reflects the *actual* backend used (which equals the resolved device because the agent allocates tensors on `get_device()` in `__init__`). The string form drops the `torch.device(...)` wrapper — `torch.device("mps").type == "mps"` is the canonical short form.

### Why "actual not requested"

The spec's SC-006 requires that retrospective benchmarking can attribute timings to the backend that *ran* the workload, not the one the user asked for. With fail-fast on unavailable devices (R3 / FR-009), the requested and actual devices are always equal at run start, so this is a tautology — but recording the resolved value defends against any future feature where we add silent fallback (e.g., a "best-effort" mode).

### Backwards compatibility

Older `metadata.json` files in `pretrained/sample-runs/` may not have a `device` field. Readers (the `analyze.ipynb` notebooks, ad-hoc scripts) MUST treat a missing `device` as `"unknown"`, not raise. The contract guarantees the field will be present for runs produced *after* this feature lands.
