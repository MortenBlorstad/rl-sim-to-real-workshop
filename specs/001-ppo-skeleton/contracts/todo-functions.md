# Contract: The five TODO functions

**Spec**: 001-ppo-skeleton
**Source of truth**: spec FR-001 through FR-006, FR-020 through FR-027

This document pins the **public signature** and **observable behavior** of each of the five TODO functions in `workshop-1/1-ppo/ppo_skeleton.py`. The participant fills in the body; the body is a TODO block. Everything in this contract is fixed and tested by `test_ppo.py`.

For cross-reference, signatures in this file MUST stay in sync with `data-model.md` §1.

---

## TODO 1 — `compute_gae`

```python
def compute_gae(
    rewards: torch.Tensor,   # shape (T,) float32
    values: torch.Tensor,    # shape (T+1,) float32 — bootstrapped final value at index T
    dones: torch.Tensor,     # shape (T,) float32 (0.0 / 1.0) — 1.0 means episode ended after step t
    gamma: float = 0.99,
    lam: float = 0.95,
) -> torch.Tensor:           # shape (T,) float32
```

**Required behavior**:

1. Iterate `t = T-1 ... 0`.
2. Compute `delta_t = rewards[t] + gamma * values[t+1] * (1 - dones[t]) - values[t]`.
3. Compute `gae_t = delta_t + gamma * lam * (1 - dones[t]) * gae_{t+1}` (with `gae_T = 0`).
4. Return the tensor `[gae_0, gae_1, ..., gae_{T-1}]`.

**Tested invariants** (FR-020):

- Output shape: exactly `(T,)`.
- Output dtype: `float32`.
- Hand-computed reference: for `rewards = [1, 1, 1, 1]`, `values = [0.5, 0.6, 0.7, 0.8, 0.9]`, `dones = [0, 0, 0, 0]`, `gamma = 0.99`, `lam = 0.95`, the test asserts each output element matches the closed-form result within `atol=1e-5`.
- Done in the middle: for `dones = [0, 0, 1, 0]`, the reset at index 2 MUST cut the bootstrap so `gae[0]`, `gae[1]`, `gae[2]` are computed without bleeding `values[3]` back through the reset boundary.

---

## TODO 2 — `sample_action`

```python
def sample_action(
    actor: ActorNetwork,
    obs: torch.Tensor,           # shape (obs_dim,) or (B, obs_dim), float32
    log_std: torch.Tensor,       # shape (action_dim,), float32
    deterministic: bool = False,
) -> tuple[torch.Tensor, torch.Tensor]:   # (action, log_prob)
```

**Required behavior**:

1. Compute `mean = actor(obs)`.
2. Construct `dist = torch.distributions.Normal(mean, log_std.exp())`.
3. If `deterministic`, set `action_unclipped = mean`. Else `action_unclipped = dist.rsample()` (or `dist.sample()` — gradient through the sample is not required by this contract).
4. Compute `log_prob = dist.log_prob(action_unclipped).sum(dim=-1)`.
5. Return `(action_clipped, log_prob)` where `action_clipped = action_unclipped.clamp(-1.0, 1.0)`.

**Important**: log_prob is computed on the **unclipped** sample so the gradient stays correct. Clipping happens only on the returned action.

**Tested invariants** (FR-021):

- For a single observation `(obs_dim,)`, `action.shape == (action_dim,)` and `log_prob.shape == ()`.
- For a batched observation `(B, obs_dim)`, `action.shape == (B, action_dim)` and `log_prob.shape == (B,)`.
- `action.dtype == torch.float32`.
- All action values lie in `[-1.0, 1.0]`.
- With `deterministic=False`, sampling 1000 times from the same fixed `obs` produces an action variance > 0.
- With `deterministic=True`, sampling 100 times from the same fixed `obs` produces zero variance.
- `log_prob` is finite (not `nan`, not `inf`).

---

## TODO 3 — `evaluate_actions`

```python
def evaluate_actions(
    actor: ActorNetwork,
    obs: torch.Tensor,           # shape (B, obs_dim), float32
    actions: torch.Tensor,       # shape (B, action_dim), float32
    log_std: torch.Tensor,       # shape (action_dim,), float32
) -> tuple[torch.Tensor, torch.Tensor]:   # (log_probs, entropy)
```

**Required behavior**:

1. Compute `mean = actor(obs)`. Build `dist = torch.distributions.Normal(mean, log_std.exp())`.
2. `log_probs = dist.log_prob(actions).sum(dim=-1)` — shape `(B,)`.
3. `entropy = dist.entropy().sum(dim=-1)` — shape `(B,)`.
4. Return `(log_probs, entropy)`.

**Tested invariants** (FR-022):

- `log_probs.shape == (B,)`, `entropy.shape == (B,)`.
- For each row `i`, `log_probs[i]` matches an independent Normal-distribution reference within `atol=1e-5`. The reference is constructed inline in the test from the same `mean` and `log_std`.
- For each row `i`, `entropy[i]` matches the same reference within `atol=1e-5`.
- All values are finite.

---

## TODO 4 — `ppo_loss`

```python
def ppo_loss(
    new_log_probs: torch.Tensor, # shape (B,), requires_grad=True (from evaluate_actions)
    old_log_probs: torch.Tensor, # shape (B,), no grad (recorded in rollout)
    advantages: torch.Tensor,    # shape (B,), normalized to mean 0 std 1
    clip_eps: float = 0.2,
) -> torch.Tensor:               # scalar tensor
```

**Required behavior**:

1. `ratio = (new_log_probs - old_log_probs).exp()` — shape `(B,)`.
2. `surr1 = ratio * advantages`.
3. `surr2 = ratio.clamp(1 - clip_eps, 1 + clip_eps) * advantages`.
4. Return `-torch.min(surr1, surr2).mean()` — scalar.

**Important**: This function returns the **policy loss only**. The training loop separately computes value loss (MSE between predicted values and returns) and the entropy bonus, and combines them as:

```python
total_loss = policy_loss + value_coef * value_loss - entropy_coef * mean_entropy
```

The TODO 4 test only verifies the policy loss term, not the combined total.

**Tested invariants** (FR-023):

- Return value shape: `torch.Size([])` (scalar).
- Return value `requires_grad == True` when `new_log_probs.requires_grad == True`.
- **Unclipped branch**: when `new_log_probs == old_log_probs` (ratio == 1.0) for all elements, the result equals `-advantages.mean()` exactly (within float tolerance).
- **Clipped branch (positive)**: when `new_log_probs - old_log_probs == 1.0` for all elements (ratio = e ≈ 2.718, well above `1 + clip_eps = 1.2`) AND `advantages > 0`, the gradient with respect to `new_log_probs` is zero (the clipped branch detached from `new_log_probs`).
- **Clipped branch (negative)**: when ratio is forced way below `1 - clip_eps` AND `advantages < 0`, the same gradient property holds.

---

## TODO 5 — `train`

```python
def train(
    env,                         # gymnasium env (MountainCarContinuous-v0 by default)
    actor: ActorNetwork,
    value: ValueNetwork,
    log_std: torch.nn.Parameter, # shape (action_dim,)
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
) -> dict
```

**Required behavior**: Implement the full PPO training loop:

1. Build a single Adam optimizer over `list(actor.parameters()) + list(value.parameters()) + [log_std]` with learning rate `lr`.
2. Build a `RolloutBuffer(rollout_size, obs_dim, action_dim)`.
3. Loop until `total_timesteps` steps have been collected:
   - Collect `rollout_size` transitions by repeatedly calling `sample_action(actor, obs_t, log_std)` and stepping the env. Reset the env on `terminated or truncated` and start a new episode within the same rollout.
   - Bootstrap the final value with `value(last_obs)` (zero if the rollout ended on a terminal step).
   - Call `buffer.compute_returns_and_advantages(last_value, gamma, gae_lambda)` (which calls `compute_gae`).
   - For `n_epochs` epochs, iterate over `buffer.get_batches(batch_size)`:
     - `new_log_probs, ent = evaluate_actions(actor, batch_obs, batch_actions, log_std)`
     - `pred_values = value(batch_obs).squeeze(-1)`
     - `policy_loss = ppo_loss(new_log_probs, batch_old_log_probs, batch_advantages, clip_eps)`
     - `value_loss = F.mse_loss(pred_values, batch_returns)`
     - `loss = policy_loss + value_coef * value_loss - entropy_coef * ent.mean()`
     - Backprop, clip gradients to `max_grad_norm`, optimizer step.
   - Print one line via `log_fn(...)` summarizing the iteration in the format from research R7.
   - Reset the buffer.
4. Return a dict with `mean_reward`, `policy_loss`, `value_loss`, `entropy`, `n_updates` from the final iteration.

**Tested invariants** (FR-024 smoke test):

- After running with `total_timesteps=512`:
  - The function returns a dict.
  - The dict contains all five required keys.
  - No printed loss is `nan`.
  - The function returns within 10 seconds.
  - No assertion is made about the values themselves — only that the loop ran.

**Script-mode invariants** (FR-026 / FR-027): when run from `if __name__ == "__main__":` on `MountainCarContinuous-v0` with the defaults from research R3:

- The function prints exactly `n_updates` lines (8 by default).
- No printed `policy_loss` or `value_loss` value is `nan`.
- The last printed `policy_loss` is strictly less than the first printed `policy_loss`.
- The training run completes in roughly 1–3 minutes on a standard laptop CPU.

---

## Skeleton-file structural rules

- All five TODO functions MUST be defined at module top level and importable as `from ppo_skeleton import compute_gae` (etc.). (FR-004)
- Each TODO function body MUST be:
  ```python
  # -- YOUR CODE HERE --
  raise NotImplementedError("TODO N: <description>")
  # -- END YOUR CODE --
  ```
  with hints / formulas in the docstring or in `# Hint:` comments above the marker block. (FR-002)
- `if __name__ == "__main__":` is the only place where any of these functions is invoked at module import time. (FR-005)
- TODOs MUST be defined in dependency order: TODO 1 above TODO 2 above ... above TODO 5. (FR-006)
- The `train()` function calls TODOs 1–4 internally; `PPOAgent.train()` calls module-level `train()`; `PPOAgent.predict()` calls `sample_action()` directly. None of these call paths skip preprocessing.
