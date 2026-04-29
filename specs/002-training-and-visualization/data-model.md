# Data Model: Stage Training Drivers, Structured Logging, and Analysis Notebooks

**Date**: 2026-04-29

This document captures the entities introduced or modified by feature `002-training-and-visualization`. Field types use Python typing notation. "Stable" means the type does not vary across records in a single file or across runs.

## Entities

### `RunDirectory`

A self-contained directory representing one training execution.

- **Path**: `<repo-root>/runs/<stage>/<run-name>/` (gitignored) **or** `<repo-root>/pretrained/sample-runs/<stage>/<run-name>/` (tracked).
- **`<stage>`** ∈ {`mountaincar`, `car-racing`}.
- **`<run-name>`** = `YYYYMMDD-HHMMSS` (UTC) by default, or the user-provided `--run-name` value (FR-014).
- **Required contents** on a successful run: `meta.json`, `metrics.jsonl`, `model.pt` (custom PPO) or `model.zip` (SB3), `eval.mp4` (or `eval.mp4.skipped`).
- **Required contents** on a Ctrl+C run: at least `meta.json`. `metrics.jsonl` may be empty or partial. `model.*` and `eval.mp4` may be absent (FR-007, FR-015, SC-007).

### `MetaJson`

One JSON object per run. Single file `meta.json` at the run-directory root.

| Field | Type | Stable | Notes |
|-------|------|--------|-------|
| `stage` | `Literal["mountaincar", "car-racing"]` | yes | matches the directory layout |
| `env_id` | `str` | yes | e.g. `"MountainCarContinuous-v0"`, `"CarRacing-v3"` |
| `agent_class` | `str` | yes | e.g. `"MountainCarPPOAgent"`, `"CarRacingPPOAgent"`, `"sb3.PPO[MlpPolicy]"`, `"sb3.PPO[CnnPolicy]"` |
| `seed` | `int` | yes | the seed used for `_seed_everything` and `env.reset(seed=...)` |
| `total_timesteps` | `int` | yes | the requested `--timesteps` |
| `hyperparameters` | `dict[str, Any]` | yes | full `PPOAgent.hyperparameters` for custom path; SB3 model config for SB3 path |
| `git_sha` | `str` | yes | `git rev-parse HEAD`; `"unknown"` if outside a repo |
| `started_at` | `str` (ISO 8601 UTC) | yes | written at logger construction |
| `finished_at` | `str | null` | ends mutable | `null` while training; set on `RunLogger.close()` |
| `status` | `Literal["running", "ok", "interrupted", "error"]` | ends mutable | `"running"` initially; updated on `close()` |
| `python_version` | `str` | yes | `platform.python_version()` |
| `torch_version` | `str` | yes | `torch.__version__` |
| `gymnasium_version` | `str` | yes | `gymnasium.__version__` |
| `metric_definitions` | `dict[str, str]` | yes | maps each `metrics.jsonl` field name → human description; doubles as schema documentation for LLM consumption (per US2) |

The two "ends mutable" fields are the only post-construction writes. They are written by `RunLogger.close()` via a single `meta.json` overwrite.

### `MetricRecord`

One JSON dict per line of `metrics.jsonl`. One record per PPO update (FR-001).

| Field | Type | Notes |
|-------|------|-------|
| `update` | `int` | 1-indexed update counter |
| `timesteps` | `int` | cumulative env steps after this update |
| `policy_loss` | `float` | `last_p_loss` per `format_update_line` (last minibatch in the last epoch); finite |
| `value_loss` | `float` | `last_v_loss`; finite |
| `entropy` | `float` | mean policy entropy this update; finite |
| `mean_return` | `float` | rolling mean of last 10 episode returns (or current partial return if no episode finished); finite |
| `log_std_mean` | `float` | `agent.log_std.exp().mean().item()` — exploration scale; for SB3, read from `model.policy.log_std` |
| `grad_norm` | `float | null` | post-clip gradient L2 norm for custom path; `null` for SB3 |
| `wall_time_seconds` | `float` | seconds since `RunLogger.__init__` |

All numeric values must be finite (no NaN, no Inf) — the writer enforces this at write time and replaces NaN with `null` while logging a warning. Per US2 acceptance scenario 2, the schema is **stable across lines** so `pd.read_json(..., lines=True)` produces a homogeneously-typed DataFrame.

### `RunLogger`

In-process bridge from `metrics_fn` callbacks to on-disk JSONL. Defined in `workshop-1/_runlog.py`.

```python
class RunLogger:
    def __init__(
        self,
        stage: Literal["mountaincar", "car-racing"],
        hyperparameters: dict,
        env_id: str,
        agent_class: str,
        seed: int,
        total_timesteps: int,
        run_name: str | None = None,
        force: bool = False,
        runs_root: Path = Path("runs"),
    ): ...

    def __call__(self, metrics: dict) -> None:
        """Append one JSONL line. Best-effort: OSError logged once and swallowed."""

    def close(self, status: Literal["ok", "interrupted", "error"] = "ok") -> None:
        """Update meta.json with finished_at + status. Idempotent."""

    @property
    def run_dir(self) -> Path: ...
```

**Lifecycle**:
1. `__init__` resolves `run_dir = runs_root / stage / (run_name or now())`; if it exists and `not force`, raise `RunDirectoryExistsError` with the FR-014 error message; if `force`, `shutil.rmtree(run_dir)` then re-create.
2. Construct `meta.json` (status="running", finished_at=null) and open `metrics.jsonl` for append.
3. `__call__` filters NaN→None, appends one JSON line, flushes.
4. `close()` updates `meta.json`'s `status` and `finished_at`, closes the JSONL handle. Safe to call twice.
5. `RunLogger` is also a context manager (`__enter__/__exit__`) so drivers can use `with RunLogger(...) as runlog:` to guarantee `close()` even on exception.

### `PPOAgent.reset_preprocess_state` (new method, FR-009b)

```python
class PPOAgent:
    def reset_preprocess_state(self) -> None:
        """No-op default. Subclasses with stateful preprocess override."""
        return None
```

`CarRacingPPOAgent` override:

```python
def reset_preprocess_state(self) -> None:
    self._frame_buffer.clear()
```

**Invariants**:
- Called by `ppo.train()` immediately before each `env.reset()` when the `agent` kwarg is supplied (FR-001b).
- NOT called during inference (`predict`) — the deployment loop on the Pi calls `reset_preprocess_state()` itself before its first `predict`, or accepts that frame-buffer warmup overhead.
- Save/load semantics unchanged: `_get_preprocess_state` / `_set_preprocess_state` round-trip the buffer; `reset_preprocess_state` is a separate "fresh-episode" hook.

### `RolloutBuffer` (modified, FR-010)

```python
class RolloutBuffer:
    def __init__(
        self,
        size: int,
        obs_shape: int | tuple[int, ...],   # was: obs_dim: int
        action_dim: int,
    ):
        if isinstance(obs_shape, int):
            obs_shape = (obs_shape,)
        self.obs_shape = obs_shape
        self.obs = np.zeros((size, *obs_shape), dtype=np.float32)
        # ... rest unchanged
```

**Invariants**:
- Storage shape `(size, *obs_shape)`, where `obs_shape` is the *post-preprocess* shape: `(2,)` for MountainCar, `(4, 84, 84)` for CarRacing.
- Backward compatibility: an `int` argument still works (interpreted as `(int,)`) so existing `RolloutBuffer(rollout_size, obs_dim, action_dim)` calls in `_main()` and `test_ppo.py --step 5` keep working.
- `add()`, `compute_returns_and_advantages()`, `get_batches()`, `reset()` semantics unchanged.

### `CarRacingPPOAgent` (modified, FR-009)

Adds two CNN classes alongside the existing (preserved) `preprocess()`:

```python
class ActorCNN(nn.Module):
    """NatureCNN-style actor for (4, 84, 84) input. Outputs action mean (3-D)."""
    def __init__(self, action_dim: int):
        super().__init__()
        self.conv1 = nn.Conv2d(4, 32, 8, stride=4)
        self.conv2 = nn.Conv2d(32, 64, 4, stride=2)
        self.conv3 = nn.Conv2d(64, 64, 3, stride=1)
        self.fc = nn.Linear(64 * 7 * 7, 512)
        self.head = nn.Linear(512, action_dim)
        # orthogonal init: gain sqrt(2) for ReLU layers, 0.01 for action head
    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.conv1(obs))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = x.flatten(start_dim=1)
        x = F.relu(self.fc(x))
        return self.head(x)


class ValueCNN(nn.Module):
    """Same conv stack as ActorCNN; scalar value output."""
    # mirrors ActorCNN with head = nn.Linear(512, 1) and final .squeeze(-1)
```

`CarRacingPPOAgent.__init__` overrides the base class:

```python
def __init__(self, obs_dim, action_dim, hyperparameters=None):
    # Apply CarRacing-specific hyperparameter defaults BEFORE base init
    cr_defaults = {
        "rollout_size": 2048, "batch_size": 128, "n_epochs": 10,
        "lr": 2.5e-4, "clip_eps": 0.1, "entropy_coef": 0.0,
    }
    merged = {**cr_defaults, **(hyperparameters or {})}
    super().__init__(obs_dim, action_dim, merged)
    self._frame_buffer: list[np.ndarray] = []
    # Replace the inherited MLPs with CNNs
    self.actor = ActorCNN(action_dim)
    self.value = ValueCNN()
```

**Invariants**:
- `preprocess()`, `_get_preprocess_state()`, `_set_preprocess_state()`, `reset_preprocess_state()` are the only methods the base class needs to call for pixel handling — internal architecture is opaque to `ppo.train()`.
- `ActorCNN(obs).shape == (batch, action_dim)`; `ValueCNN(obs).shape == (batch,)`. These match `ActorNetwork` / `ValueNetwork` so `ppo.train()` is shape-agnostic.

### `Sb3JsonlCallback`

Defined in `workshop-1/_sb3_jsonl_callback.py`.

```python
class Sb3JsonlCallback(BaseCallback):
    def __init__(self, run_logger: RunLogger, model_log_std_attr_path: str = "policy.log_std"):
        super().__init__()
        self.run_logger = run_logger
        self._start_time: float | None = None

    def _on_training_start(self) -> None:
        self._start_time = time.monotonic()

    def _on_rollout_end(self) -> None:
        nv = self.logger.name_to_value
        log_std = _resolve_attr(self.model, "policy.log_std")  # tensor
        self.run_logger({
            "update": int(nv.get("time/iterations", 0)),
            "timesteps": int(nv.get("time/total_timesteps", self.num_timesteps)),
            "policy_loss": float(nv.get("train/policy_gradient_loss", float("nan"))),
            "value_loss": float(nv.get("train/value_loss", float("nan"))),
            "entropy": -float(nv.get("train/entropy_loss", float("nan"))),  # SB3 sign flip
            "mean_return": float(nv.get("rollout/ep_rew_mean", float("nan"))),
            "log_std_mean": float(log_std.exp().mean().item()) if log_std is not None else None,
            "grad_norm": None,  # SB3 doesn't expose post-clip grad norm
            "wall_time_seconds": time.monotonic() - self._start_time,
        })

    def _on_step(self) -> bool:
        return True
```

### `pretrained/sample-runs/<stage>/<name>/`

Same on-disk layout as `RunDirectory`. Tracked in git. `analyze.ipynb` accepts a hand-set `RUN_DIR = "pretrained/sample-runs/<stage>/<name>"` to load it. Auto-discovery does NOT search this path (R8 from research.md).

## Validation Rules

- All `MetricRecord` fields except `grad_norm` are required and must be finite numbers (NaN→`null` mapping at write time).
- `MetaJson.metric_definitions` MUST list every key actually written to `metrics.jsonl` for that run (so an LLM reading the meta knows what each key means without having to be told the schema).
- `RunLogger.close()` MUST be called even on exception (drivers use `with RunLogger(...) as runlog:`); on uncaught exception the status becomes `"error"`.
- `RolloutBuffer.obs_shape` MUST equal the agent's `agent.preprocess(env.observation_space.sample()).shape` when the `agent` kwarg is provided to `ppo.train()`.

## State Transitions

```text
RunLogger lifecycle:

  __init__  → status="running", started_at=now, run_dir created
       │
       │  (zero or more __call__ writes appending to metrics.jsonl)
       ▼
  close("ok"|"interrupted"|"error")  → status updated, finished_at=now, file handles closed
```

```text
PPOAgent preprocess lifecycle (when ppo.train() supplies the agent kwarg):

  agent.reset_preprocess_state()  ← initial setup
       │
  obs = agent.preprocess(env.reset()[0])
       │
  loop:
    obs = agent.preprocess(env.step(action)[0])
    if done:
      agent.reset_preprocess_state()
      obs = agent.preprocess(env.reset()[0])
```
