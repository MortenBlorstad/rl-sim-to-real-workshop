# Phase 1 Data Model: PPO Skeleton

**Date**: 2026-04-13
**Branch**: `001-ppo-skeleton`

This is a single-process Python library, not a database-backed application. The "data model" here describes the runtime objects (classes, functions, in-memory data containers) the skeleton ships and how they relate to one another, so that Phase 2 task generation has a concrete inventory to work from.

---

## 1. Module-level functions (the five TODOs)

These five functions are top-level symbols in `workshop-1/1-ppo/ppo_skeleton.py`. Their bodies are TODO blocks; their signatures and docstrings are fully written.

### 1.1 `compute_gae` — TODO 1

```python
def compute_gae(
    rewards: torch.Tensor,       # shape (T,)
    values: torch.Tensor,        # shape (T+1,)  — bootstrapped final value at index T
    dones: torch.Tensor,         # shape (T,) bool/float — episode termination after each step
    gamma: float = 0.99,
    lam: float = 0.95,
) -> torch.Tensor:               # shape (T,)
    """Generalized Advantage Estimation."""
```

**Validation rules** (enforced by the TODO 1 test, FR-020):
- Output shape is exactly `(T,)`.
- Output dtype matches `rewards.dtype` (float32).
- For a hand-computed toy trajectory of length 4 with no `done`, output matches the closed-form GAE values within `atol=1e-5`.
- For the same trajectory with `dones[2] = True`, the advantages at indices ≤ 2 are computed without bleeding the bootstrapped value across the episode boundary.

### 1.2 `sample_action` — TODO 2

```python
def sample_action(
    actor: ActorNetwork,
    obs: torch.Tensor,           # shape (obs_dim,) or (B, obs_dim)
    log_std: torch.Tensor,       # shape (action_dim,)
    deterministic: bool = False,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Sample an action and return (action, log_prob).

    The returned action is clipped to [-1, 1] AFTER sampling.
    The returned log_prob is computed on the UNCLIPPED sample so the
    gradient remains correct.
    """
```

**Validation rules** (FR-021):
- Output `action` has the same shape as `Box(1,)` (`shape == (1,)` for a single obs, `(B, 1)` for batched).
- Output dtype is `float32`.
- All sampled action values lie in `[-1, 1]`.
- With `deterministic=False`, repeated calls on the same `obs` produce different `action` values (variance > 0 across 1000 samples).
- With `deterministic=True`, repeated calls produce identical `action` values (variance == 0).
- `log_prob` is finite and has shape `(1,)` or `(B,)`.

### 1.3 `evaluate_actions` — TODO 3

```python
def evaluate_actions(
    actor: ActorNetwork,
    obs: torch.Tensor,           # shape (B, obs_dim)
    actions: torch.Tensor,       # shape (B, action_dim)
    log_std: torch.Tensor,       # shape (action_dim,)
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return (log_probs, entropy) for the given (obs, action) batch.

    log_probs shape: (B,)
    entropy shape:   (B,)  — per-sample entropy of the policy distribution
    """
```

**Validation rules** (FR-022):
- `log_probs` and `entropy` shapes match `(B,)` exactly.
- For each row `i`, `log_probs[i]` matches `torch.distributions.Normal(mean_i, std_i).log_prob(actions[i]).sum()` within `atol=1e-5`.
- For each row `i`, `entropy[i]` matches `torch.distributions.Normal(mean_i, std_i).entropy().sum()` within `atol=1e-5`.

### 1.4 `ppo_loss` — TODO 4

```python
def ppo_loss(
    new_log_probs: torch.Tensor, # shape (B,) — from evaluate_actions on current policy
    old_log_probs: torch.Tensor, # shape (B,) — recorded in the rollout
    advantages: torch.Tensor,    # shape (B,) — from compute_gae, normalized to mean 0 std 1
    clip_eps: float = 0.2,
) -> torch.Tensor:               # scalar tensor
    """PPO clipped surrogate objective.

    Returns the policy loss only (negative of the clipped objective).
    Value loss and entropy bonus are computed and combined OUTSIDE this
    function in the training loop.
    """
```

**Validation rules** (FR-023):
- Output is a scalar tensor (`out.shape == torch.Size([])`).
- Output `requires_grad == True` when inputs do.
- When `new_log_probs == old_log_probs` (ratio = 1.0), the unclipped branch is taken and `out == -advantages.mean()`.
- When the ratio is forced outside `[1-clip_eps, 1+clip_eps]` for every sample, the clipped branch is taken; the gradient with respect to `new_log_probs` is 0 in that branch.

### 1.5 `train` — TODO 5

```python
def train(
    env,                         # gymnasium env wrapping MountainCarContinuous-v0
    actor: ActorNetwork,
    value: ValueNetwork,
    log_std: torch.nn.Parameter,
    total_timesteps: int = 8192,
    rollout_size: int = 1024,
    n_epochs: int = 4,
    batch_size: int = 64,
    lr: float = 3e-4,
    gamma: float = 0.99,
    gae_lambda: float = 0.95,
    clip_eps: float = 0.2,
    value_coef: float = 0.5,
    entropy_coef: float = 0.01,
    max_grad_norm: float = 0.5,
    seed: int = DEFAULT_SEED,
    log_fn=print,
) -> dict:
    """Full PPO training loop.

    Wires together: rollout collection (calls sample_action), advantage
    computation (compute_gae), policy/value updates (evaluate_actions
    + ppo_loss + value MSE + entropy bonus), and per-update logging.

    Returns a stats dict with at minimum the keys:
      mean_reward (float), policy_loss (float), value_loss (float),
      entropy (float), n_updates (int).
    """
```

**Validation rules** (FR-024 smoke test):
- After running with `total_timesteps=512`, returned dict contains all five required keys.
- No printed loss is NaN.
- Function returns within 10 seconds.

---

## 2. Helper classes (provided complete, no TODOs)

### 2.1 `ActorNetwork`

```python
class ActorNetwork(nn.Module):
    def __init__(self, obs_dim: int, action_dim: int, hidden: int = 64): ...
    def forward(self, obs: torch.Tensor) -> torch.Tensor:  # returns mean, shape (B, action_dim)
```

- 2-layer MLP, hidden size 64, Tanh activations.
- Output is the **mean** of the policy distribution. The `log_std` is a separate `nn.Parameter` owned by the agent (or passed alongside), per R1.
- Orthogonal init: gain `sqrt(2)` for hidden layers, `0.01` for output layer.

### 2.2 `ValueNetwork`

```python
class ValueNetwork(nn.Module):
    def __init__(self, obs_dim: int, hidden: int = 64): ...
    def forward(self, obs: torch.Tensor) -> torch.Tensor:  # returns scalar, shape (B,)
```

- 2-layer MLP, hidden size 64, Tanh activations.
- Orthogonal init: gain `sqrt(2)` for hidden layers, `1.0` for output layer.
- No parameter sharing with `ActorNetwork`.

### 2.3 `RolloutBuffer`

```python
class RolloutBuffer:
    def __init__(self, size: int, obs_dim: int, action_dim: int): ...
    def add(self, obs, action, log_prob, reward, done, value): ...
    def compute_returns_and_advantages(self, last_value, gamma, gae_lambda) -> None: ...
        # Calls compute_gae internally — this is the integration point for TODO 1
    def get_batches(self, batch_size: int) -> Iterator[dict]: ...
        # Yields shuffled minibatches of dicts: obs, actions, old_log_probs, advantages, returns
    def reset(self) -> None: ...
```

- Pre-allocated NumPy arrays of shape `(size, ...)`.
- The buffer is the single source of truth for "what happened during the rollout"; the training loop reads from it for the update phase.

---

## 3. The `PPOAgent` class

```python
class PPOAgent:
    def __init__(self, obs_dim: int, action_dim: int,
                 hyperparameters: dict | None = None): ...

    # Article II contract
    def preprocess(self, obs: np.ndarray) -> np.ndarray: ...     # base = identity
    def predict(self, obs: np.ndarray, deterministic: bool = False) -> np.ndarray: ...
    def train(self, env, total_timesteps: int) -> dict: ...
    def save(self, path: str) -> None: ...
    @classmethod
    def load(cls, path: str) -> "PPOAgent": ...

    # Subclass extension hooks (FR-031)
    def _get_preprocess_state(self) -> dict: return {}
    def _set_preprocess_state(self, state: dict) -> None: pass
```

**Owned state**:
- `self.actor: ActorNetwork`
- `self.value: ValueNetwork`
- `self.log_std: nn.Parameter` of shape `(action_dim,)`, init 0.0
- `self.hyperparameters: dict` — the dict consumed by the module-level `train()` function
- `self.obs_dim: int`, `self.action_dim: int`

**Method contracts**:
- `preprocess(obs)` returns `obs` unchanged in the base class.
- `predict(obs)` calls `self.preprocess(obs)`, converts to a torch tensor, calls `sample_action(self.actor, obs, self.log_std, deterministic=deterministic)`, returns the action as `np.ndarray`.
- `train(env, total_timesteps)` calls the module-level `train()` with `self.actor`, `self.value`, `self.log_std`, and `self.hyperparameters`. Returns the same stats dict.
- `save(path)` calls `torch.save(state_dict, path)` where `state_dict` follows the schema in research R5.
- `load(path)` is a `@classmethod` that reads the saved dict, looks up the subclass via `_AGENT_REGISTRY`, instantiates, restores state, and returns. It is bound to the registered class, not necessarily `cls`.

**Class registry**:

```python
_AGENT_REGISTRY: dict[str, type["PPOAgent"]] = {}

def register_agent(cls):
    _AGENT_REGISTRY[cls.__name__] = cls
    return cls

@register_agent
class PPOAgent: ...
```

Subclasses (in stage 2 / stage 3 files) decorate themselves with `@register_agent`. This lets `PPOAgent.load(path)` return the correct subclass without hardcoding the list.

---

## 4. Stage subclass stubs (FR-009)

### 4.1 `MountainCarPPOAgent` — `workshop-1/2-mountaincar/agent.py`

```python
from workshop_1.one_ppo.ppo_skeleton import PPOAgent, register_agent

@register_agent
class MountainCarPPOAgent(PPOAgent):
    """PPOAgent for MountainCarContinuous-v0.

    Vector observation; preprocess is identity in this minimal stub.
    Stage 2 may override with a normalization wrapper.
    """
    def preprocess(self, obs):
        return obs
```

This file is intentionally minimal — its job is to **prove the override path works** and to give Workshop 2 a place to extend later. The stage-2 spec will replace it with a richer subclass.

### 4.2 `CarRacingPPOAgent` — `workshop-1/3-car-racing/agent.py`

```python
from workshop_1.one_ppo.ppo_skeleton import PPOAgent, register_agent
import numpy as np
import cv2  # opencv-python; added to workshop1 dependency group

@register_agent
class CarRacingPPOAgent(PPOAgent):
    """PPOAgent for CarRacing-v3.

    Pixel observation: crop sky, grayscale, resize to 84×84, normalize,
    frame-stack 4. Demonstrates the override contract for pixels.
    """
    STACK_SIZE = 4

    def __init__(self, obs_dim, action_dim, hyperparameters=None):
        super().__init__(obs_dim, action_dim, hyperparameters)
        self._frame_buffer: list[np.ndarray] = []

    def preprocess(self, obs):
        img = obs[:84, :, :]                                       # crop
        img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)                # grayscale
        img = cv2.resize(img, (84, 84), interpolation=cv2.INTER_AREA)
        img = img.astype(np.float32) / 255.0                       # normalize
        self._frame_buffer.append(img)
        if len(self._frame_buffer) > self.STACK_SIZE:
            self._frame_buffer.pop(0)
        while len(self._frame_buffer) < self.STACK_SIZE:
            self._frame_buffer.append(img)
        return np.stack(self._frame_buffer, axis=0)                # shape (4, 84, 84)

    def _get_preprocess_state(self):
        return {"frame_buffer": [f.copy() for f in self._frame_buffer]}

    def _set_preprocess_state(self, state):
        self._frame_buffer = list(state.get("frame_buffer", []))
```

**Important caveat**: this stage-3 stub ships the *preprocess pipeline* but does **not** ship a working CarRacing training script or a CarRacing-compatible CNN actor/value network — those belong to the stage-3 spec. The class is unit-tested for `preprocess()` shape and for the override contract via `test_agent_interface.py`, not for end-to-end CarRacing training. The 2D MLP `ActorNetwork` from this spec will not actually train on `(4, 84, 84)` observations; that is acceptable because end-to-end CarRacing training is out of scope here.

---

## 5. Test runner internal data structures

`test_ppo.py` keeps a module-level dict:

```python
STEPS: dict[int, tuple[str, callable]] = {}
```

populated by the `@step(n, name)` decorator. The CLI selects either one entry by `--step N` or runs all entries in numeric key order. Each entry is invoked inside this wrapper:

```python
def _run_step(n: int) -> str:
    name, fn = STEPS[n]
    try:
        fn()
        print(f"TODO {n} OK!")
        return "PASS"
    except NotImplementedError as e:
        print(f"TODO {n} not yet implemented ({name}): {e}")
        return "NOT_IMPLEMENTED"
    except AssertionError as e:
        print(f"FAIL: TODO {n} ({name}): {e}")
        return "FAIL"
```

The result categories are exactly three: `PASS`, `NOT_IMPLEMENTED`, `FAIL`. The runner's exit code is `0` iff every invoked step returned `PASS`.

---

## 6. Relationships diagram (text)

```
ppo_skeleton.py
├── compute_gae        (TODO 1) ───────► used by RolloutBuffer.compute_returns_and_advantages
├── sample_action      (TODO 2) ───────► used by RolloutBuffer collection loop in train()
│                                        used by PPOAgent.predict()
├── evaluate_actions   (TODO 3) ───────► used in train() update phase
├── ppo_loss           (TODO 4) ───────► used in train() update phase
├── train              (TODO 5) ───────► uses TODOs 1–4 + ActorNetwork + ValueNetwork + RolloutBuffer
│                                        called by PPOAgent.train()
├── ActorNetwork                ─────┐
├── ValueNetwork                ─────┼── instantiated and owned by PPOAgent
├── RolloutBuffer               ─────┘
├── PPOAgent                    ─────► extended by MountainCarPPOAgent, CarRacingPPOAgent
├── _AGENT_REGISTRY                    written by @register_agent, read by PPOAgent.load()
└── DEFAULT_SEED                       used by __main__ and train()
```

No circular dependencies. TODOs strictly ordered by dependency (FR-006): 1 ← 2,3,4 ← 5.
