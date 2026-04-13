# Contract: `PPOAgent` — Constitution Article II Agent Interface

**Spec**: 001-ppo-skeleton
**Source of truth**: Constitution Article II + spec FR-007/008/009/030/031

This document is the binding contract that `PPOAgent` and any subclass MUST satisfy. The shared test runner `workshop-1/1-ppo/test_agent_interface.py --agent ppo` is the executable form of this contract; everything below is what that test verifies.

---

## C1 — Class identity and registration

- `PPOAgent` is defined in `workshop-1/1-ppo/ppo_skeleton.py` at module top level.
- `PPOAgent` is decorated with `@register_agent`, which adds it to the module-level `_AGENT_REGISTRY: dict[str, type[PPOAgent]]`.
- Any subclass (e.g. `MountainCarPPOAgent`, `CarRacingPPOAgent`) MUST also be decorated with `@register_agent` before any instance is saved.

**Test**: `assert "PPOAgent" in _AGENT_REGISTRY`.

---

## C2 — `preprocess(obs)` — base is identity, overridable

**Signature**:

```python
def preprocess(self, obs: np.ndarray) -> np.ndarray
```

**Base-class contract**:
- Returns `obs` unchanged. Specifically: `np.array_equal(agent.preprocess(x), x)` for any `np.ndarray` `x`.
- Pure function: same input → same output. No internal state mutation in the base class.

**Subclass override contract**:
- Subclasses MAY override `preprocess()` with any deterministic transform that returns a `np.ndarray`.
- Subclasses MAY use internal state (e.g. a frame stack buffer) provided that state is exposed via `_get_preprocess_state` / `_set_preprocess_state` for save/load round-trip.
- The override MUST be picked up automatically by `predict()` (i.e. `predict()` calls `self.preprocess(...)`, not the bare base method).

**Test (vector path)**:
```python
agent = PPOAgent(obs_dim=2, action_dim=1)
x = np.random.randn(2).astype(np.float32)
assert np.array_equal(agent.preprocess(x), x)
assert np.array_equal(agent.preprocess(x), agent.preprocess(x))   # determinism
```

**Test (override path)**:
```python
@register_agent
class _ScalingAgent(PPOAgent):
    def preprocess(self, obs):
        return obs * 2.0

agent = _ScalingAgent(obs_dim=2, action_dim=1)
x = np.array([1.0, 2.0], dtype=np.float32)
assert np.array_equal(agent.preprocess(x), np.array([2.0, 4.0], dtype=np.float32))

# predict() must use the override
out = agent.predict(x)
assert isinstance(out, np.ndarray)
# And agent.preprocess was called with the raw obs, not pre-scaled
```

---

## C3 — `predict(obs, deterministic=False)` — accepts raw observations

**Signature**:

```python
def predict(self, obs: np.ndarray, deterministic: bool = False) -> np.ndarray
```

**Contract**:
- Accepts a **raw** observation (i.e. the value returned by `env.step()` or `env.reset()`), not a preprocessed one.
- Internally calls `self.preprocess(obs)` exactly once.
- Returns an action as `np.ndarray` matching the agent's action space (for this spec: `Box(1,)`, dtype `float32`, values in `[-1, 1]`).
- With `deterministic=True`, returns the distribution mean (no sampling); repeated calls on the same input return the same action.
- With `deterministic=False` (default), samples from the policy; repeated calls on the same input MAY return different actions.

**Test**:
```python
agent = PPOAgent(obs_dim=2, action_dim=1)
raw_obs = np.array([0.1, -0.2], dtype=np.float32)

a1 = agent.predict(raw_obs)
assert isinstance(a1, np.ndarray)
assert a1.shape == (1,)
assert a1.dtype == np.float32
assert -1.0 <= a1[0] <= 1.0

# determinism flag
a_det1 = agent.predict(raw_obs, deterministic=True)
a_det2 = agent.predict(raw_obs, deterministic=True)
assert np.array_equal(a_det1, a_det2)
```

---

## C4 — `train(env, total_timesteps)` — runs the training loop

**Signature**:

```python
def train(self, env, total_timesteps: int) -> dict
```

**Contract**:
- Accepts a Gymnasium-conformant environment (`reset()` returns `(obs, info)`, `step()` returns `(obs, reward, terminated, truncated, info)`).
- Calls the module-level `train()` function with `self.actor`, `self.value`, `self.log_std`, and `self.hyperparameters`.
- Returns a stats dict with at minimum these keys:
  - `mean_reward: float` — mean episode return over the last rollout
  - `policy_loss: float` — final-update policy loss
  - `value_loss: float` — final-update value loss
  - `entropy: float` — final-update mean policy entropy
  - `n_updates: int` — number of update iterations executed

**Test**: covered by FR-024 smoke test in `test_ppo.py --step 5`.

---

## C5 — `save(path)` and `load(path)` — single-file round trip

**Signatures**:

```python
def save(self, path: str) -> None
@classmethod
def load(cls, path: str) -> "PPOAgent"
```

**Contract**:
- `save(path)` writes a single `.pt` file containing a dict with at minimum these keys (per research R5):
  - `class_name: str` — `type(self).__name__`
  - `obs_dim: int`
  - `action_dim: int`
  - `actor_state_dict: dict`
  - `value_state_dict: dict`
  - `log_std: torch.Tensor`
  - `hyperparameters: dict`
  - `preprocess_state: dict` — output of `self._get_preprocess_state()`
- `load(path)` reads the file, looks up `class_name` in `_AGENT_REGISTRY`, instantiates the **registered subclass** (not necessarily `PPOAgent` itself), restores all state including `preprocess_state` via `_set_preprocess_state`, and returns the ready-to-use instance.
- The loaded instance MUST produce the same `predict(raw_obs, deterministic=True)` output as the original for an identical raw observation.

**Test**:
```python
agent = PPOAgent(obs_dim=2, action_dim=1)
raw_obs = np.array([0.3, -0.4], dtype=np.float32)
expected = agent.predict(raw_obs, deterministic=True)

with tempfile.NamedTemporaryFile(suffix=".pt", delete=False) as f:
    agent.save(f.name)
    loaded = PPOAgent.load(f.name)

assert type(loaded).__name__ == "PPOAgent"
assert np.array_equal(loaded.predict(raw_obs, deterministic=True), expected)
```

**Subclass round-trip test**:
```python
@register_agent
class _RoundTripAgent(PPOAgent):
    def preprocess(self, obs):
        return obs.clip(-0.5, 0.5)

agent = _RoundTripAgent(obs_dim=2, action_dim=1)
agent.save(path)
loaded = PPOAgent.load(path)
assert type(loaded).__name__ == "_RoundTripAgent"   # subclass is restored, not base
```

---

## C6 — `_get_preprocess_state` / `_set_preprocess_state` — subclass extension

**Signatures**:

```python
def _get_preprocess_state(self) -> dict
def _set_preprocess_state(self, state: dict) -> None
```

**Base-class contract**:
- `_get_preprocess_state()` returns `{}`.
- `_set_preprocess_state({})` is a no-op.

**Subclass contract**:
- A subclass that adds preprocessing state MUST implement both methods symmetrically: `_set_preprocess_state(self._get_preprocess_state())` MUST leave the agent's preprocessing in an equivalent state.
- These methods exist so that `save()` / `load()` can persist subclass-specific preprocessing (e.g. frame stack buffers, normalization statistics) without modifying the base class.

**Test** (in `test_agent_interface.py`):
```python
agent = PPOAgent(obs_dim=2, action_dim=1)
assert agent._get_preprocess_state() == {}
agent._set_preprocess_state({})  # no-op
```

---

## C7 — Out of scope (deferred to follow-up specs)

The following are explicitly NOT part of this contract and MUST NOT be tested by `test_agent_interface.py --agent ppo`:

- The pixel preprocessing pipeline of `CarRacingPPOAgent` end-to-end (only the override mechanism is tested here; the pipeline correctness is a stage-3 spec concern).
- `SB3Agent` and `test_agent_interface.py --agent sb3` (Path B follow-up spec).
- Loading on the Raspberry Pi via `export_model.py` / `deploy.sh` (Workshop 2 spec).
