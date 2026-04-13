# Phase 0 Research: PPO Skeleton with Per-TODO Tests

**Date**: 2026-04-13
**Branch**: `001-ppo-skeleton`

## Scope

The spec already resolved 5 clarifications. This document records the remaining design decisions that are well within established knowledge and do not need external research, but must be pinned down before Phase 1 contracts can be written. Each entry uses the canonical Decision / Rationale / Alternatives format.

---

## R1 — Distribution and policy parameterization

**Decision**: `ActorNetwork.forward(obs)` returns `(mean, log_std)` for an independent multivariate `Normal` distribution. `log_std` is a **learnable parameter independent of the observation** (`nn.Parameter` initialized to 0.0), not a network output. Action sampling uses `torch.distributions.Normal(mean, log_std.exp())` and the action is clipped to `[-1, 1]` after sampling.

**Rationale**:
- Observation-independent `log_std` is the standard PPO parameterization in OpenAI Spinning Up, CleanRL, and Stable-Baselines3 — it stabilizes early training and avoids participants debugging exploding standard deviations.
- Learnable as a separate parameter (not a constant) preserves the "policy learns its own exploration" intuition while removing one source of failure.
- Clipping to `[-1, 1]` post-sample is consistent with how `MountainCarContinuous-v0`'s action space is bounded; storing the unclipped sample for log-prob computation keeps the gradient honest.

**Alternatives considered**:
- *State-dependent log_std (network output)*: more flexible but harder to train and more code in `ActorNetwork`.
- *Tanh-squashed Normal (SAC-style)*: more stable for bounded actions but requires a Jacobian correction in log-prob — unnecessary complexity for a beginner workshop.
- *Beta distribution*: bounded by construction, no clipping needed, but unfamiliar to most participants and adds new vocabulary.

---

## R2 — Network architecture

**Decision**: `ActorNetwork` and `ValueNetwork` are each two-layer MLPs with hidden size **64** and `Tanh` activations. Output layer is linear. Weight init uses orthogonal init with gain `sqrt(2)` for hidden layers and `0.01` for the policy mean head, `1.0` for the value head.

**Rationale**:
- 2×64 with Tanh is the canonical PPO baseline architecture in Spinning Up and the original PPO paper for classic-control environments. It trains MountainCarContinuous fast on CPU.
- Orthogonal init with these gains reduces the chance of NaN losses and is cited in the Implementation Matters paper as one of the highest-impact PPO details.
- Tanh (vs. ReLU) gives smoother gradient flow on small classic-control envs.

**Alternatives considered**:
- *Larger MLP (256 × 256)*: slower, no benefit on `Box(2,)` observation space.
- *ReLU activations*: slightly faster but more prone to dead neurons on small networks.
- *Default PyTorch init*: works but increases the risk of early NaN losses, which would corrupt the FR-027 exit check.

---

## R3 — Default hyperparameters for the `__main__` script run

**Decision**: The `__main__` block calls `train(env, total_timesteps=8192)` with `rollout_size=1024`, `n_epochs=4`, `batch_size=64`, `lr=3e-4`, `gamma=0.99`, `gae_lambda=0.95`, `clip_eps=0.2`, `value_coef=0.5`, `entropy_coef=0.01`, `max_grad_norm=0.5`. That gives **8 update iterations**, prints loss after each, completes in roughly 30–90 seconds on CPU, and reliably produces a downward loss trend on MountainCarContinuous-v0 with the orthogonal init from R2.

**Rationale**:
- 8 update iterations is the sweet spot: enough to see a clear trend (FR-026, FR-027), few enough to fit comfortably in the 60-minute PPO block budget.
- These hyperparameters are taken directly from the CleanRL `ppo_continuous_action.py` reference, which is a known-good PPO baseline for classic-control envs.
- `total_timesteps=8192` × 8 updates is well below any plausible "agent solves MCC" threshold, which is intentional per Q4 — solving MCC is the next stage's job.

**Alternatives considered**:
- *50k timesteps*: would solve MCC sometimes; takes ~3 minutes. Risk: participants think their PPO is broken when it doesn't solve.
- *2k timesteps (1 update only)*: too short to show a trend.
- *CLI flag for `--timesteps`*: nice-to-have but adds CLI parsing surface to the script. The hard-coded default is sufficient per FR-026.

---

## R4 — Test runner architecture

**Decision**: `test_ppo.py` ships a ~80-line custom runner with this structure:

```python
STEPS = {}  # {int: (name, callable)}

def step(n, name):
    def deco(fn):
        STEPS[n] = (name, fn)
        return fn
    return deco

@step(1, "GAE")
def test_step_1():
    from ppo_skeleton import compute_gae   # local import, isolates failures
    ...
    print("TODO 1 OK!")

# ... step 2..5

if __name__ == "__main__":
    # parse --step N (optional), run requested step(s), exit code = 0/1
```

Failure handling: a top-level `try/except` around each step categorizes the result as PASS, NOT_IMPLEMENTED (catches `NotImplementedError`), or FAIL (catches `AssertionError` and prints a structured `FAIL: ...` line with expected/observed values).

**Rationale**:
- Local imports inside each test function are the only way to make `--step N` truly independent of the state of TODOs other than N (FR-011). A top-level `from ppo_skeleton import *` would import every TODO and trip on the first `NotImplementedError`.
- The custom runner is roughly 80 lines including the failure-categorization wrapper, summary printer, and CLI parsing. That is cheaper than wrestling pytest's marker system into the prescribed `--step N` interface and the prescribed output format (FR-013).
- Keeping the runner inside `test_ppo.py` (no separate `_helpers.py` module) makes every part of the test machinery readable in one file, which matches the participant-first principle.

**Alternatives considered**:
- *pytest with custom markers*: rejected in the brainstorm decision log. Adds a dependency, adds noise, fights the prescribed output format.
- *Plain asserts at top level*: cannot distinguish "not implemented" from "implemented incorrectly", and stops at the first failure (rejected in the brainstorm decision log).
- *Separate `_test_helpers.py` for fixtures*: fewer lines per file but adds a second file participants might look at and worry about. Inline fixtures inside `test_ppo.py` are simpler.

---

## R5 — Save/load file format

**Decision**: `PPOAgent.save(path)` writes a single `.pt` file via `torch.save`, containing a dict with these keys:

```python
{
    "class_name": type(self).__name__,           # e.g. "PPOAgent" or "MountainCarPPOAgent"
    "obs_dim": int,
    "action_dim": int,
    "actor_state_dict": ...,
    "value_state_dict": ...,
    "log_std": ...,
    "hyperparameters": {...},                    # the dict used by train()
    "preprocess_state": self._get_preprocess_state(),  # subclass override hook; base returns {}
}
```

`PPOAgent.load(path)` is a `@classmethod` that reads the dict, looks up the subclass by `class_name` in a small `_AGENT_REGISTRY`, instantiates it with `obs_dim`/`action_dim`, restores all state, calls `instance._set_preprocess_state(state)`, and returns the instance ready for `predict(raw_obs)`.

**Rationale**:
- Single `.pt` file matches Article II ("a single file that … `Agent.load()` can read").
- `_get_preprocess_state` / `_set_preprocess_state` is the subclass extension hook required by FR-031 — Workshop 2 pixel agents will use it to persist frame-stack buffer config or normalization statistics.
- The class registry approach lets a `MountainCarPPOAgent.save(path)` round-trip via base `PPOAgent.load(path)` returning a `MountainCarPPOAgent` instance — proven by the override-contract test in FR-030.
- `torch.save` (not pickle directly) is the idiomatic PyTorch path and survives PyTorch version upgrades better.

**Alternatives considered**:
- *ONNX export*: smaller, framework-agnostic, but Workshop 2 still wants PyTorch on the Pi. ONNX is a Workshop 2 export-stage concern, not a save-format concern.
- *Two files (weights + JSON config)*: violates Article II's "a single file" requirement.
- *cloudpickle of the whole agent*: brittle across class refactors and unsafe to load.

---

## R6 — Per-TODO recovery mechanism

**Decision**: The `solutions` branch contains the fully-implemented `ppo_skeleton.py`. Five additional git tags point at progressively-more-complete commits on `solutions`:

```
ws1-todo1-done  → solution for TODO 1 only, TODOs 2–5 still NotImplementedError
ws1-todo2-done  → solutions for TODOs 1–2,    TODOs 3–5 still NotImplementedError
ws1-todo3-done  → solutions for TODOs 1–3,    TODOs 4–5 still NotImplementedError
ws1-todo4-done  → solutions for TODOs 1–4,    TODO  5  still NotImplementedError
ws1-todo5-done  → all five TODOs solved (== `solutions` branch tip for stage 1)
```

Recovery command (added to `workshop-1/README.md`):

```bash
git checkout ws1-todo3-done -- workshop-1/1-ppo/ppo_skeleton.py
```

This overwrites only the participant's `ppo_skeleton.py` with the version where TODO 3 is solved (and TODOs 1–2, since TODO 3 depends on them); their unrelated work (e.g. notes, other files) is preserved. If they had TODO 4 work in their own copy, this *will* lose it — they should commit before recovering. The README documents this caveat.

**Rationale**:
- Constitution Article IX explicitly prescribes `ws1-todoN-done` tags. This decision pins the exact recovery command and tag semantics.
- Path-scoped checkout (`-- workshop-1/1-ppo/ppo_skeleton.py`) is the most surgical recovery: nothing else is touched.
- Building the tags incrementally rather than as five branches avoids branch sprawl and matches Article IX's "no other long-lived branches".

**Alternatives considered**:
- *Five solution files (`ppo_solution_1.py` … `ppo_solution_5.py`)*: pollutes the directory, invites participants to peek prematurely.
- *Single `solutions` branch with no per-TODO granularity*: a participant stuck on TODO 3 would have to manually delete the TODO 4–5 solutions from the recovered file. Too much surgery under workshop time pressure.

---

## R7 — Logging output format

**Decision**: The training loop prints one line per update iteration in a fixed-width format:

```
[update  1/8] timesteps=  1024  policy_loss=+0.123  value_loss=+0.456  entropy=+0.789  mean_return=-32.10
```

No tensorboard, no matplotlib, no JSON, no log file. The `train()` function returns a dict containing the final values of these stats (used by the smoke test in FR-024).

**Rationale**:
- Console-only is the lowest-friction output for beginners and works on any laptop with no extra setup.
- One line per update means the FR-027 trend check is just `last_loss < first_loss` over the printed lines, easy for participants to eyeball.
- Returning a stats dict from `train()` lets the smoke test verify the training loop's "shape" without re-parsing stdout.

**Alternatives considered**:
- *tensorboard*: adds a dependency, requires a separate viewer, overkill for stage 1.
- *matplotlib live plot*: same problem, plus blocks the script in some terminals.
- *`logging` module with handlers*: adds noise and forces participants to learn `logging` config. Plain `print()` is fine for a 60-minute exercise.

---

## R8 — Random seeding for reproducibility

**Decision**: The `__main__` block sets `seed = 42` and seeds Python's `random`, NumPy, PyTorch (CPU only — no `torch.cuda.manual_seed_all`), and the environment via `env.reset(seed=seed)`. The seed is exposed as a module constant `DEFAULT_SEED = 42` at the top of `ppo_skeleton.py`. The `train()` function takes an optional `seed` argument and re-seeds at the start.

**Rationale**:
- Same seed across participant laptops means everyone in the room sees roughly the same numbers, which makes "is this right?" debugging straightforward.
- Exposing the constant at the top of the file lets curious participants change it without hunting through code.
- CPU-only seeding is sufficient since the spec targets CPU baseline.

**Alternatives considered**:
- *No seeding (system random)*: makes the workshop room incoherent.
- *Per-test seed inside test_ppo.py*: redundant with `train()` accepting a seed argument.

---

## Summary of decisions

| ID | Topic | Decision |
|---|---|---|
| R1 | Policy distribution | Independent `Normal` with learnable observation-independent `log_std`; clip action to `[-1, 1]` after sampling |
| R2 | Network architecture | 2×64 Tanh MLPs for `ActorNetwork` and `ValueNetwork`, orthogonal init |
| R3 | Default hyperparameters | `total_timesteps=8192`, `rollout_size=1024`, 8 updates, CleanRL-style hyperparameters |
| R4 | Test runner | Custom 80-line `@step(n, name)` runner; local imports inside each test |
| R5 | Save/load format | Single `.pt` file via `torch.save` with class registry + `preprocess_state` hook |
| R6 | Recovery mechanism | Git tags `ws1-todo1-done` … `ws1-todo5-done` on `solutions` branch; path-scoped checkout |
| R7 | Logging | One fixed-width console line per update; `train()` returns final-stats dict |
| R8 | Seeding | `DEFAULT_SEED = 42`; seed Python/NumPy/PyTorch/env in `__main__` and at the top of `train()` |

All NEEDS CLARIFICATION items from Technical Context: **none**.
