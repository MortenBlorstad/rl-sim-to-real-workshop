# Quickstart: Fix Device Selection (MPS Should Not Be Slower Than CPU)

Three short flows. Each ends with a single visible-outcome line you can grep for in your terminal.

## Prerequisites

```bash
uv sync --group workshop1
```

## Flow A — On Apple Silicon, verify MPS is now used and at parity

This is the primary acceptance check (User Story 1, SC-001).

```bash
# 1. Run with MPS auto-selected (default behaviour after this feature lands)
unset RL_WORKSHOP_DEVICE
uv run python workshop-1/1-ppo/ppo/tests/bench_device.py --stage pendulum --updates 22 --warmup 2

# Expected last line:
# [bench] pendulum: cpu=<X.XX>s/upd  mps=<Y.YY>s/upd  ratio_mps_over_cpu=<R.RR>  PASS_SC001=<true|false>

# 2. Run on CarRacing (CNN — where MPS should clearly win)
uv run python workshop-1/1-ppo/ppo/tests/bench_device.py --stage carracing --updates 12 --warmup 2

# Expected last line:
# [bench] carracing: cpu=<X.XX>s/upd  mps=<Y.YY>s/upd  ratio_mps_over_cpu=<R.RR>  PASS_SC002=<true|false>
```

Acceptance:
- Pendulum line ends with `PASS_SC001=true` (MPS within 10% of CPU per SC-001).
- CarRacing line ends with `PASS_SC002=true` (MPS at least 20% faster per SC-002).

If either fails, do **not** revert to the hard-coded CPU; instead capture the bench output and open a bug — the SC failure is the signal that the rollout/update split (Phase 0 R2) needs adjustment.

## Flow B — Override to CPU for debugging

This verifies User Story 2 (the participant escape hatch).

```bash
RL_WORKSHOP_DEVICE=cpu uv run python workshop-1/2-pendulum/train.py --total-timesteps 4096

# Expected: training completes; one of the early printed lines reads:
#   [PPOAgent] device=cpu (RL_WORKSHOP_DEVICE=cpu)
# And:
#   runs/pendulum/<run-name>/metadata.json contains "device": "cpu"
```

Verify the metadata field:

```bash
jq -r '.device' runs/pendulum/$(ls -t runs/pendulum/ | head -1)/metadata.json
# Expected output: cpu
```

Negative path — request a device that's unavailable (on a Mac):

```bash
RL_WORKSHOP_DEVICE=cuda uv run python workshop-1/2-pendulum/train.py --total-timesteps 4096

# Expected: fails fast before training starts, with a message containing:
#   "RL_WORKSHOP_DEVICE='cuda' but no CUDA backend is available on this machine."
#   "Available: cpu, mps."
```

## Flow C — Run the cross-device load test

This verifies User Story 3 (saved-model portability across devices, SC-004).

```bash
# On Apple Silicon (CPU↔MPS round-trip):
uv run python workshop-1/1-ppo/ppo/tests/test_agent_interface.py --step 8

# Expected last line:
#   C8 cross-device load: max-abs-diff=<value>  bound=1e-3  OK!
```

The test:
1. Trains a small `PPOAgent` for 64 update steps on the auto-selected device.
2. Saves to `/tmp/agent.pt`.
3. Loads on every *other* available device.
4. Replays a deterministic 1-episode rollout from a fixed seed against the source-device rollout.
5. Asserts per-action `max-abs-diff` ≤ 1e-3 (CPU↔MPS) or ≤ 1e-5 (CPU↔CUDA).

Skipped variants print `[skipped: cuda not available]` (or similar) and count as passing.

## Flow D — Workshop-leader pre-flight (pretrained artefacts)

Run before each Workshop 1 delivery on the leader's laptop:

```bash
# Loops over each .pt under pretrained/, loads on the auto-selected device,
# runs one greedy episode, asserts finite + in-bounds actions.
uv run python workshop-1/1-ppo/ppo/tests/bench_device.py --pretrained-smoke

# Expected last line:
#   [bench] pretrained smoke: <N> files OK on device=<cpu|mps|cuda>
```

If a pretrained file fails to load, regenerate it from the `solutions` branch (per Constitution Article VI, "regenerated whenever training code changes").

## Run the full PPO test suite on each available device

```bash
RL_WORKSHOP_DEVICE=cpu uv run python workshop-1/1-ppo/ppo/tests/test_ppo.py
RL_WORKSHOP_DEVICE=mps uv run python workshop-1/1-ppo/ppo/tests/test_ppo.py    # Apple Silicon only
RL_WORKSHOP_DEVICE=cuda uv run python workshop-1/1-ppo/ppo/tests/test_ppo.py   # Linux+CUDA only
```

Each invocation prints:

```
device: <name>
TODO 1 OK!
TODO 2 OK!
…
ALL STEPS OK!
```

If any step fails, the device-name header lets you attribute the failure (most likely a real backend issue, not an algorithm bug — the same code passes on other devices).
