# Phase 0 Research: Fix Device Selection (MPS Should Not Be Slower Than CPU)

This file consolidates the eight decisions referenced in `plan.md` § Phase 0. Each item is `Decision → Rationale → Alternatives considered`.

## R1. Root cause — why MPS is currently slower than CPU

**Decision**: The dominant cost is per-env-step **overhead at two boundaries**, not the network forward pass itself:

1. **CPU↔MPS device transfers.** Every observation is copied to MPS, every forward output is copied back, and every `.item()` is a synchronous device→host fence.
2. **NumPy↔Torch conversions.** `env.step` is numpy-typed, so each step round-trips `numpy → torch → numpy`. On CPU these are near-free (PyTorch shares memory when dtype/strides align), but combined with (1) on MPS the per-step Python + sync overhead is multiplied.

Both are addressed by the same fix (R2: keep rollout on CPU). Once rollout tensors stay on CPU, `torch.as_tensor(obs)` becomes a shared-memory view, `.item()` and `.cpu()` become no-ops, and the only remaining crossings are (a) the unavoidable env boundary (Gymnasium is numpy-typed) and (b) one `batch.to(self.device)` per minibatch in the update phase, where the cost amortises over `batch_size`.

Confirmed by reading `workshop-1/1-ppo/ppo/ppo.py`:

- `ppo.py:279` — `obs_t = torch.as_tensor(obs, dtype=torch.float32, device=self.device)` happens once per env step (`rollout_size` × `n_updates` times per training run). On MPS: `numpy → torch CPU → MPS` per step.
- `ppo.py:282` — `value_t = self.critic(obs_t).item()` forces an MPS→host sync per step.
- `ppo.py:283` — `action.detach().cpu().numpy()` is `MPS → CPU torch → numpy` per step.
- `ppo.py:293`, `ppo.py:317` — bootstrap `.item()` adds two more syncs per update.
- `ppo.py:297` — `log_prob.item()` adds yet another sync per step.

On CPU, all five "transfers" per step are free (no device boundary, and `as_tensor`/`.numpy()` share memory). On MPS each one is a synchronous fence: the CPU-side Python loop blocks until the GPU has finished and copied the scalar back. With a 2×64 MLP, the GPU work itself is microseconds; the round-trips dominate. CarRacing's CNN forward is heavier but still not enough to amortise 4–5 syncs per step at `rollout_size=2048`.

**Rationale**: Profiling-by-reading is sufficient here because the sync points are unconditional and call-counted (not data-dependent). A dedicated profiling pass would only confirm the conclusion.

**Alternatives considered**:
- *MPS kernel cold-start.* Possible secondary contributor, but `n_updates × rollout_size` calls would amortise any cold-start cost; rejected as primary cause.
- *MPS op fallback to CPU.* Some `Normal` ops do fall back; this is addressed orthogonally by R4 (MPS_FALLBACK), not by the rollout strategy.
- *Memory pressure.* The model is < 1 MB; rejected.

## R2. Strategy — CPU rollout, accelerator update

**Decision**: Inside `PPOAgent.train()`:
- Networks `self.actor`, `self.critic` and the `self.log_std` parameter remain on `self.device` (the trainable, gradient-bearing copies).
- During rollout, a CPU-resident shadow copy `self._actor_cpu`, `self._critic_cpu`, `self._log_std_cpu` is used for `sample_action` and value bootstrap. The shadow is refreshed once per update boundary via `actor_cpu.load_state_dict(actor.state_dict())` (and similarly for critic/log_std).
- The `RolloutBuffer` already stores NumPy/CPU data and yields CPU tensors (`rollout_buffer.py` unchanged).
- The existing line `batch = {k: v.to(self.device) for k, v in batch.items()}` (`ppo.py:328`) keeps moving the full minibatch to `self.device` for the update.

**Rationale**: Keeps the public `Agent` API identical and keeps `PPOAgent.predict()` working unchanged on `self.device`. Per-step rollout pays no cross-device sync. Update phase pays one transfer per minibatch (`batch_size × …`), which on MPS amortises well — exactly the regime where MPS is meant to win.

**Alternatives considered**:
- *Leave everything on device, only remove `.item()` calls.* Partial fix; the per-step `torch.as_tensor(obs, ..., device=self.device)` and `.cpu().numpy()` round-trips remain. Rejected.
- *Move `RolloutBuffer` to device.* Requires a 2048-row tensor allocation per update on the device; for tiny obs/action dims the savings are negligible and CarRacing's pixel buffer would explode device memory. Rejected.
- *Pure CPU agent (revert spec).* Defeats the user's stated goal. Rejected.
- *No shadow copy — re-`.to("cpu")` on every step.* The bandwidth cost equals what we are trying to avoid. Rejected.
- *Dispatch sample_action via `self.actor.cpu()(obs).to(device)`.* Modifies `self.actor`'s device in-place mid-train; corrupts the optimiser state and breaks gradient device-affinity. Rejected.

## R3. Override mechanism — environment variable

**Decision**: A single environment variable, `RL_WORKSHOP_DEVICE`, with allowed values `cpu`, `cuda`, `mps`, `auto` (case-insensitive). Default = `auto`. Read inside `get_device()`. The chosen device is also recorded in `runs/<stage>/<run-name>/metadata.json` under the `device` field.

**Rationale**:
- Zero plumbing across `train.py`, `test_ppo.py`, `bench_device.py`, and any participant script.
- Survives across stages (a participant who sets `RL_WORKSHOP_DEVICE=cpu` once for their session keeps that posture for Pendulum, CarRacing, and tests).
- Sits naturally next to PyTorch's own `PYTORCH_ENABLE_MPS_FALLBACK`, so the mental model is consistent.
- Honours Article I's "no source-edit-required override" expectation.

**Alternatives considered**:
- *CLI flag `--device` on each `train.py` driver.* Multiplies surface, leaks into TODO 5's argument plumbing; participants would have to thread it themselves. Rejected.
- *Hyperparameter on `PPOAgent`.* Pollutes the participant API with a non-RL concern; doesn't help test runners or the bench script. Rejected.
- *A config file `~/.rl-workshop.toml`.* Heavier than needed for a single-knob switch. Rejected.

## R4. `PYTORCH_ENABLE_MPS_FALLBACK` timing

**Decision**: Inside `get_device()`, immediately before returning `torch.device("mps")`, run `os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")`. `get_device()` is called inside `PPOAgent.__init__` *before* any tensor is moved to MPS (the `.to(self.device)` calls follow on subsequent lines).

**Rationale**: PyTorch reads this env var at op-dispatch time on each call, not at module import; so the timing requirement is "set before the first MPS op," which the placement guarantees. `setdefault` preserves any value the user has already exported (e.g., `PYTORCH_ENABLE_MPS_FALLBACK=0` for a user who specifically wants the hard error).

**Alternatives considered**:
- *Set in `workshop-1/1-ppo/ppo/__init__.py` at import time.* Works but happens even when the user is not on MPS; `setdefault` makes that benign, but conflating package import with side effects is uglier than scoping the side effect to the moment MPS is actually selected. Rejected.
- *Tell the user to set it.* Violates Article I (actionable error messages, no participant-stuck states) per spec clarification Q1. Rejected.

## R5. Cross-device load of saved artefacts

**Decision**: Replace `torch.load(path, weights_only=False)` with `torch.load(path, map_location="cpu", weights_only=False)` in `PPOAgent.load`. The subsequent `agent = target_cls(env, hyperparameters=...)` already calls `.to(self.device)` in its `__init__` for `actor` / `critic` / `log_std`, so the loaded `state_dict`'s tensors are placed on the local device by the agent constructor.

**Rationale**: Saved `.pt` files in `pretrained/` produced on CPU carry CPU device tags; on a Mac that selects MPS by default, the tags are honoured by `torch.load` unless `map_location` overrides them, leading to weird mixed-device states. `map_location="cpu"` is the canonical fix. The same fix lets a CUDA-saved model load on a Mac and vice versa.

**Alternatives considered**:
- *`map_location=lambda storage, loc: storage`.* Equivalent but more obscure. Rejected.
- *Re-save all `pretrained/` artefacts.* Would mask the underlying load-path bug; the next contributor would re-introduce it. Rejected.

## R6. Test parameterisation across devices

**Decision**: Existing `@step` tests in `workshop-1/1-ppo/ppo/tests/test_ppo.py` build their tensors via `agent.device` already (e.g. `obs = torch.tensor([...], device=device)` at lines 145–146, 228–231). With `get_device()` honouring `RL_WORKSHOP_DEVICE`, simply running the suite under different env-var values exercises each device. The runner prints a single-line header `device: <name>` so a workshop leader can scan a multi-device dry-run output. Total per-run wall-clock per device stays under the existing budget; running on three devices triples wall-clock but is a workshop-leader pre-flight task, not part of `uv run python test_ppo.py` for participants.

**Rationale**: Reuses the production override, so tests can't drift from runtime behaviour. Avoids a parametrize-style test framework (the constitution prohibits adding pytest).

**Alternatives considered**:
- *Hard-coded triple-loop inside each test.* Triples test count and breaks the < 10 s per step budget. Rejected.
- *A separate `test_device.py`.* Duplicates assertions; the value of running the *same* tests on each device is precisely catching device-specific regressions in existing logic. Rejected.

## R7. Cross-device numerical tolerance test (SC-004)

**Decision**: Add one new step to `test_agent_interface.py` (call it `C8`): construct `PPOAgent(env)`, run a small training pass (just enough to make weights non-trivial — e.g., 64 update steps), `save("/tmp/agent.pt")`, then `load("/tmp/agent.pt", env)` on each *other* available device, replay a deterministic 1-episode rollout from a fixed seed, and assert per-action `max-abs-diff` against the source-device rollout: ≤ 1e-3 for CPU↔MPS, ≤ 1e-5 for CPU↔CUDA. Skip cleanly with a `[skipped: <reason>]` marker if the alternate device is unavailable.

**Rationale**: Locks down the SC-004 acceptance criterion as an executable test. The 1e-3 / 1e-5 split is per spec clarification Q2.

**Alternatives considered**:
- *Compare per-batch logits instead of per-step actions.* Less faithful to the user-visible behaviour the spec measures. Rejected.
- *Compare distributional statistics over many rollouts.* Too slow for the < 10 s test budget. Rejected.

## R8. Pretrained artefact handling

**Decision**: No regeneration required. Existing `pretrained/*.pt` files load via the patched `PPOAgent.load` (R5). A workshop-leader pre-flight check is added to `quickstart.md` (load each pretrained file on each available device, run one greedy episode, assert finite + in-bounds actions), but it is not a CI gate.

**Rationale**: The `state_dict` format is device-agnostic; the only thing that broke cross-device load was the missing `map_location`. Once that's fixed, existing artefacts work as-is. Re-generating them would burn training time for no functional benefit.

**Alternatives considered**:
- *Regenerate `pretrained/` from `solutions` branch on every device.* Burns hours per device. Rejected.
- *Add a CI gate that loads every `pretrained/` file.* Adds CI infrastructure beyond the workshop's scope. The workshop-leader pre-flight is sufficient.
