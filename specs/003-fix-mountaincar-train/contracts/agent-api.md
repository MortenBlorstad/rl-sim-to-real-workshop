# Contract: `ppo` package public API

**Date**: 2026-04-29
**Branch**: `003-fix-mountaincar-train`

This contract is the **authoritative external surface** of `workshop-1/1-ppo/ppo/` after this feature lands. The driver (`workshop-1/2-mountaincar/train.py`), the test runners, and any future stage that consumes the package must use only these symbols.

## Top-level imports

```python
from ppo import (
    PPOAgent,
    RolloutBuffer,
    ActorNetwork,
    CriticNetwork,
    register_agent,
    _AGENT_REGISTRY,
)
from ppo.utils import (
    seed_everything,
    format_update_line,
    get_device,
    RunLogger,
    RunDirectoryExistsError,
    make_log_fn,
    parse_update_line,
)
```

Anything not on these two lines is implementation detail and may move.

## `PPOAgent`

```python
class PPOAgent:
    DEFAULT_HYPERPARAMS: dict   # dict copy; keys: rollout_size, n_epochs, batch_size, lr,
                                # gamma, gae_lambda, clip_eps, value_coef, entropy_coef,
                                # max_grad_norm, log_std_init, random_state

    def __init__(self, env: gym.Env, hyperparameters: dict | None = None) -> None: ...

    # Five teaching blocks (TODOs) — participants fill these in.
    def sample_action(self, obs: torch.Tensor, deterministic: bool = False
                      ) -> tuple[torch.Tensor, torch.Tensor]: ...
    def evaluate_actions(self, obs: torch.Tensor, actions: torch.Tensor
                         ) -> tuple[torch.Tensor, torch.Tensor]: ...
    def ppo_loss(self, new_log_probs: torch.Tensor, old_log_probs: torch.Tensor,
                 advantages: torch.Tensor, clip_eps: float = 0.2) -> torch.Tensor: ...
    def train(self, env: gym.Env, total_timesteps: int = 8192,
              random_state: int = DEFAULT_SEED, log_fn=print) -> dict: ...
    # (RolloutBuffer.compute_gae is the fifth teaching block.)

    # Maintainer code — fully implemented.
    def predict(self, obs: np.ndarray, deterministic: bool = False) -> np.ndarray: ...
    def evaluate(self, env: gym.Env, n_episodes: int = 10,
                 record_video: bool = True,
                 video_dir: str | Path | None = None) -> list[float]: ...
    def save(self, path: str) -> None: ...

    @classmethod
    def load(cls, path: str, env: gym.Env) -> "PPOAgent": ...
```

### Method contracts

#### `__init__(env, hyperparameters=None)`
- MUST NOT seed globally if `hyperparameters["random_state"]` is `None`.
- Otherwise MUST call `seed_everything(hyperparameters["random_state"])` exactly once before constructing `actor`, `critic`, `log_std`.
- Stores `env`, `obs_dim`, `action_dim`, `action_min`, `action_max`, `obs_min`, `obs_max`, `device` (from `get_device()`), `hyperparameters` (merged with `DEFAULT_HYPERPARAMS`).
- Constructs `actor`, `critic` on `self.device`. Constructs `log_std` as `nn.Parameter(torch.ones(action_dim, device=self.device) * hyperparameters["log_std_init"])`.

#### `sample_action(obs, deterministic=False) -> (action, log_prob)`
- `obs`: `torch.Tensor` of shape `(obs_dim,)` or `(B, obs_dim)`.
- `action`: same leading shape as `obs`, clamped to `[action_min, action_max]`, dtype `float32`.
- `log_prob`: log-prob of the **unclipped** sample summed over action dims; shape `()` or `(B,)`.
- TODO 2.

#### `evaluate_actions(obs, actions) -> (log_probs, entropy)`
- Internally uses `self.actor` and `self.log_std` (refactor: parameters dropped from the public signature).
- `log_probs`, `entropy`: shape `(B,)`.
- TODO 3.

#### `ppo_loss(new_log_probs, old_log_probs, advantages, clip_eps=0.2) -> torch.Tensor`
- Returns a scalar with `requires_grad=True`.
- Implements `-mean(min(ratio * adv, clip(ratio, 1-eps, 1+eps) * adv))`.
- TODO 4.

#### `train(env, total_timesteps=8192, random_state=DEFAULT_SEED, log_fn=print) -> dict`
- Calls `seed_everything(random_state)` once at the start.
- Builds Adam over `actor.parameters() + critic.parameters() + [log_std]` with linear-decay LR schedule.
- Allocates a `RolloutBuffer` and runs `n_updates = max(1, total_timesteps // rollout_size)` PPO updates.
- Per update, calls `log_fn(format_update_line(...))` exactly once after the update phase.
- Bootstraps through truncations (not terminations) using `self.critic`.
- Returns `{"mean_reward", "policy_loss", "value_loss", "entropy", "n_updates"}`.
- TODO 5.

#### `predict(obs: np.ndarray, deterministic=False) -> np.ndarray`
- Converts `obs` to `torch.Tensor` on `self.device` at the boundary.
- Calls `sample_action(obs_t, deterministic=deterministic)` under `torch.no_grad()`.
- Returns the action as `np.ndarray` of dtype `float32`.

#### `evaluate(env, n_episodes=10, record_video=True, video_dir=None) -> list[float]`
- If `record_video=True`:
  - Wraps a fresh env constructed with `render_mode="rgb_array"` in `gymnasium.wrappers.RecordVideo(env, video_folder=str(video_dir), name_prefix="eval")`.
  - On success, renames the produced `eval-episode-0.mp4` to `eval.mp4`.
  - On `ImportError` / `gymnasium.error.DependencyNotInstalled` (ffmpeg missing): writes `eval.mp4.skipped`, prints one warning to stderr, continues without video.
  - `video_dir=None` defaults to the current working directory; the driver always passes the run dir.
- Drives the env greedily via `predict(obs, deterministic=True)` until the episode ends (`terminated or truncated`).
- Returns a `list[float]` of length `n_episodes` of episode returns.

#### `save(path)`
- Writes a `dict` containing: `class_name`, `actor_state_dict`, `value_state_dict` (named `value_*` for back-compat with the pre-refactor format), `log_std`, `hyperparameters`. Uses `torch.save`.

#### `load(cls, path, env) -> PPOAgent`
- Reads with `torch.load(path, weights_only=False)`.
- Resolves the class via `_AGENT_REGISTRY[state["class_name"]]`. Raises `ValueError` if unknown.
- Constructs `target_cls(env, hyperparameters=state["hyperparameters"])`.
- Loads state dicts and `log_std`. Returns the instance.

## `RolloutBuffer`

Unchanged signature. See `data-model.md`. The only TODO it owns is `compute_gae(rewards, values, dones, gamma, lam)`.

## `ppo.utils` re-exports

| Symbol | From | Purpose |
|---|---|---|
| `seed_everything` | `ppo.utils.utils` | Seeds `random`, `numpy`, `torch` |
| `format_update_line` | `ppo.utils.utils` | Single source of truth for the printed log line |
| `get_device` | `ppo.utils.utils` | CUDA → MPS → CPU |
| `RunLogger`, `RunDirectoryExistsError` | `ppo.utils._runlog` | Per-run JSONL + meta.json owner |
| `make_log_fn`, `parse_update_line` | `ppo.utils._log_parser` | Adapts `print`-style log lines into JSONL records |

## Backwards-compatibility

This is the first contract for the package layout — no prior consumers to honor. The pre-refactor flat-file `ppo.py` API (`compute_gae`, `sample_action`, `ActorNetwork`, `evaluate_actions`, `ppo_loss`, `train`, `ValueNetwork`) is **not** preserved at the top level; everything goes through `PPOAgent` now. The pre-refactor test file's free-function imports (`from ppo import compute_gae`) intentionally become `from ppo import PPOAgent, RolloutBuffer; PPOAgent.evaluate_actions(...)` etc. — see `contracts/tests.md`.
