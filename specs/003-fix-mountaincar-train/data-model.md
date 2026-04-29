# Phase 1 Data Model: Refactored PPO Package Surfaces

**Date**: 2026-04-29
**Branch**: `003-fix-mountaincar-train`

This is the entity / type model for the refactored `ppo` package and the run directory it produces. "Entity" here covers Python classes and on-disk artifacts — the workshop has no database.

## In-process classes

### `PPOAgent` (`workshop-1/1-ppo/ppo/ppo.py`)

| Field | Type | Purpose | Source |
|---|---|---|---|
| `env` | `gym.Env` | Reference to the env passed in at construction. | `__init__` arg |
| `obs_dim` | `int` | `prod(env.observation_space.shape)` | derived |
| `action_dim` | `int` | `prod(env.action_space.shape)` | derived |
| `action_min` / `action_max` | `torch.Tensor` (float32) | Action-space bounds for clamping in `sample_action`. | derived |
| `obs_min` / `obs_max` | `np.ndarray` | Observation-space bounds (informational; not used by training). | derived |
| `hyperparameters` | `dict` | Merged with `DEFAULT_HYPERPARAMS`. Must include `random_state` (used by both `__init__` and `train`). | `__init__` arg |
| `device` | `torch.device` | Auto-selected by `get_device()`. | derived |
| `actor` | `ActorNetwork` | 2×64 tanh MLP; outputs the mean of a Normal distribution. | constructed |
| `critic` | `CriticNetwork` | 2×64 tanh MLP; outputs scalar state value. | constructed |
| `log_std` | `nn.Parameter` (shape `(action_dim,)`) | Learnable diagonal log-std. Created on `device`. | constructed |

**Public methods**:

| Method | Signature | Status |
|---|---|---|
| `sample_action` | `(obs: torch.Tensor, deterministic: bool = False) -> (action, log_prob)` | TODO 2; teaching block |
| `evaluate_actions` | `(obs: torch.Tensor, actions: torch.Tensor) -> (log_probs, entropy)` | TODO 3; teaching block (signature changed from refactor — uses `self.actor`/`self.log_std`) |
| `ppo_loss` | `(new_log_probs, old_log_probs, advantages, clip_eps=0.2) -> torch.Tensor` | TODO 4; teaching block |
| `train` | `(env, total_timesteps: int = 8192, random_state: int = DEFAULT_SEED, log_fn = print) -> dict` | TODO 5; teaching block |
| `predict` | `(obs: np.ndarray, deterministic: bool = False) -> np.ndarray` | Maintainer code; converts at the boundary (R1 bug #5) |
| `evaluate` | `(env, n_episodes: int = 10, record_video: bool = True, video_dir: str \| Path \| None = None) -> list[float]` | Maintainer code; **NEW** — replaces `NotImplementedError` per spec Q1 |
| `save` | `(path: str) -> None` | Maintainer code |
| `load` | `(cls, path: str, env: gym.Env) -> PPOAgent` | Maintainer code; signature gains `env` (R1 bug #6) |

**Module-level**:
- `_AGENT_REGISTRY: dict[str, type]` — populated by `@register_agent`
- `register_agent(cls)` — class decorator
- `DEFAULT_SEED = 42`

**State transitions**: `__init__` → (training loop, alternating between rollout and SGD updates) → `train()` returns dict → `evaluate()` returns list[float] → `save(path)` writes `.pt` → `PPOAgent.load(path, env)` reconstructs.

### `RolloutBuffer` (`workshop-1/1-ppo/ppo/rollout_buffer.py`)

| Field | Type | Purpose |
|---|---|---|
| `size`, `obs_dim`, `action_dim` | `int` | Buffer capacity + shape. |
| `obs`, `actions`, `log_probs`, `rewards`, `dones`, `values`, `advantages`, `returns` | `np.ndarray` (pre-allocated) | Per-step trajectory storage. |
| `idx` | `int` | Write head; reset by `reset()`. |

**Public methods** (unchanged by this feature):
- `add(obs, action, log_prob, reward, done, value)` — append one step.
- `compute_gae(rewards, values, dones, gamma, lam) -> torch.Tensor` — TODO 1; teaching block.
- `compute_returns_and_advantages(last_value, gamma, gae_lambda) -> None` — wraps `compute_gae`.
- `get_batches(batch_size) -> Iterator[dict]` — yields normalized advantages.
- `reset() -> None` — sets `idx = 0`.

### `ActorNetwork`, `CriticNetwork` (`workshop-1/1-ppo/ppo/networks.py`)

Unchanged by this feature. 2×64 tanh MLP with orthogonal init. Actor outputs mean (action_dim); Critic outputs scalar.

### `RunLogger`, `RunDirectoryExistsError` (`workshop-1/1-ppo/ppo/utils/_runlog.py`)

Unchanged by this feature. Schema is fixed by `specs/002-training-and-visualization/contracts/run-format.md`. Used by both `train.py` and (after R6 follow-up) `train_sb3.py`.

### Log helpers (`workshop-1/1-ppo/ppo/utils/_log_parser.py`)

- `parse_update_line(line: str) -> dict | None`
- `make_log_fn(run_logger, agent, *, also_print=True) -> Callable[[str], None]`

Unchanged. The driver calls `make_log_fn(runlog, agent)` and passes the result as `agent.train(env, ..., log_fn=log_fn)`.

## On-disk artifacts (`runs/mountaincar/<run-name>/`)

Schema is **unchanged** from feature 002 (`specs/002-training-and-visualization/contracts/run-format.md`). This feature is purely a re-wiring of the producer.

| File | Producer | Contents |
|---|---|---|
| `meta.json` | `RunLogger.__init__` + `close()` | Frozen schema (see feature 002 contract) |
| `metrics.jsonl` | `make_log_fn` callback (per-update) | One JSON object per line; keys: `update`, `timesteps`, `policy_loss`, `value_loss`, `entropy`, `mean_return`, `log_std_mean`, `grad_norm` (null for custom PPO until R8 follow-up), `wall_time_seconds` |
| `model.pt` | `agent.save(...)` | `state_dict`s + hyperparameters + `log_std` + class name |
| `eval.mp4` | `gymnasium.wrappers.RecordVideo` (renamed by `evaluate()`) | One greedy rollout from `agent.evaluate(record_video=True)` |
| `eval.mp4.skipped` | `evaluate()` if ffmpeg missing | Sentinel; FR-009 graceful-degrade |

## Test artifacts (`workshop-1/1-ppo/ppo/tests/`)

| File | Step IDs | Targets | Pre-refactor reference |
|---|---|---|---|
| `test_ppo.py` | 1, 2, 3, 4, 5 | The five PPO TODOs | commit `60321eb`'s `workshop-1/1-ppo/test_ppo.py` |
| `test_agent_interface.py` | 1 (C1), 2 (C3), 3 (C3-det), 4 (C4), 5 (C5-base), 6 (C5-subclass), 7 (C7-evaluate) | Agent contract | commit `60321eb`'s `workshop-1/1-ppo/test_agent_interface.py`, **without** former steps 2/3/4 (C2 preprocess) and 10 (C6 preprocess state) |

Both files use the `@step(n, name)` decorator and a local `STEPS` registry. Single CLI flag (`--step N` for `test_ppo.py`, `--agent ppo` for `test_agent_interface.py`). Local imports inside each test function.

## Validation rules

- `PPOAgent.train()` must return a dict with at least `{"mean_reward", "policy_loss", "value_loss", "entropy", "n_updates"}`. (Validated by `test_ppo.py` step 5 and `test_agent_interface.py` step 4.)
- `PPOAgent.evaluate()` must return `list[float]` of length `n_episodes`, all finite. With `record_video=True`, an `.mp4` (or `.mp4.skipped`) lands in `video_dir`. (Validated by `test_agent_interface.py` step 7.)
- `RolloutBuffer.compute_gae` output must match a hand-computed reference within `atol=1e-5` for the four-step canonical case (with and without a mid-trajectory `done`).
- `predict(obs, deterministic=True)` must produce identical actions across calls and across save/load round-trips.

## Out of scope

- `train_sb3.py` (R6 follow-up).
- Workshop 2 deployment changes (Article VII follow-up).
- `_AGENT_REGISTRY` discovery from disk (the registry only resolves classes already imported in the running process — same as today).
- Multi-env / vectorized `gymnasium` envs.
