# Quickstart: CarRacing Training Drivers

Four flows. Each ends with a single visible-outcome line you can grep for in your terminal.

## Prerequisites

```bash
# System dependency for Box2D physics (CarRacing)
brew install swig            # macOS
sudo apt-get install swig    # Linux

# Python dependencies
uv sync --group workshop1
```

If you skip `swig`, both drivers fail with a clear actionable error message naming the missing tool. See `workshop-1/3-car-racing/README.md` for full install notes.

## Flow A — SB3 from-scratch smoke (User Story 1)

The constitutional escape hatch. Trains `PPO("CnnPolicy", ...)` for 10 000 steps and writes the canonical run directory.

```bash
uv run python workshop-1/3-car-racing/train_sb3.py --timesteps 10000 --run-name smoke-sb3 --force

# Expected last lines:
#   [train] training complete in <X.X>s
#   [eval] writing eval.mp4 (or eval.mp4.skipped if ffmpeg missing)
#
# Resulting tree:
#   runs/car-racing/smoke-sb3/
#   ├── meta.json          — status: ok, device: mps|cpu|cuda, hf_repo_id: null, hf_filename: null
#   ├── metrics.jsonl      — non-empty (one record per SB3 logger callback)
#   ├── model.zip          — SB3 PPO archive
#   └── eval.mp4 OR eval.mp4.skipped
```

Acceptance for SC-001: the wall-clock time printed on the `[train] training complete` line is under 60 seconds on Apple Silicon.

## Flow B — Custom-PPO from-scratch smoke (User Story 2)

Same shape, but using your `PPOAgent` (which auto-detects the `(4, 84, 84)` obs and constructs CNN actor + critic networks).

```bash
uv run python workshop-1/3-car-racing/train.py --timesteps 10000 --run-name smoke-custom --force

# Expected:
#   [PPOAgent] device=mps (RL_WORKSHOP_DEVICE=auto)            (or cpu)
#   [PPOAgent] network_arch=cnn
#   [update  1/N] timesteps=...  policy_loss=...  value_loss=...  ...
#   ...
#   [train] training complete in <X.X>s
#
# Resulting tree:
#   runs/car-racing/smoke-custom/
#   ├── meta.json          — status: ok, network_arch: "cnn", device: mps|cpu|cuda
#   ├── metrics.jsonl      — non-empty (one JSONL line per PPO update)
#   ├── model.pt           — PPOAgent state-dict + hyperparameters
#   └── eval.mp4 OR eval.mp4.skipped
```

Acceptance for SC-002: completes in under 90 seconds on Apple Silicon. `meta.json["network_arch"] == "cnn"` (verifies auto-detection worked).

## Flow C — SB3 fine-tune from HuggingFace (User Story 3)

Downloads a pretrained CarRacing checkpoint from HuggingFace Hub, initialises SB3's PPO from those weights, and continues training for `--timesteps` steps before evaluating. Internet required for the first run.

```bash
# First run (cache miss): downloads ~MB from huggingface.co
uv run python workshop-1/3-car-racing/train_sb3.py \
    --hf-repo sb3/ppo-CarRacing-v0 \
    --timesteps 10000 \
    --run-name finetune \
    --force

# Expected:
#   [train_sb3] downloading sb3/ppo-CarRacing-v0/ppo-CarRacing-v0.zip from HuggingFace...
#   [train_sb3] loaded pretrained weights, fine-tuning for 10000 steps
#   [train] training complete in <X.X>s
#   [eval] writing eval.mp4
#
# meta.json contains:
#   "hf_repo_id":   "sb3/ppo-CarRacing-v0",
#   "hf_filename":  "ppo-CarRacing-v0.zip"
```

Acceptance for SC-003: total wall-clock (download + fine-tune + eval) under 90 seconds with internet, and `eval.mp4` shows the car visibly driving (not random behaviour) — the pretrained weights are competent before fine-tuning starts.

```bash
# Second run (cache hit): no network usage
uv run python workshop-1/3-car-racing/train_sb3.py \
    --hf-repo sb3/ppo-CarRacing-v0 \
    --timesteps 10000 \
    --run-name finetune-cached \
    --force

# Expected the *download* portion to print "(cache hit)" and complete in < 1 second.
# Total wall-clock similar to Flow A (since fine-tune + eval still happen).
```

Acceptance for SC-004: download line prints `(cache hit)` and the cached path is reused from `~/.cache/huggingface/hub/`.

### Negative paths

```bash
# Offline + cache empty
uv run python workshop-1/3-car-racing/train_sb3.py --hf-repo sb3/ppo-CarRacing-v0 \
    --timesteps 10000 --run-name offline --force
# Expected: HuggingFaceLoadError naming the repo+filename, suggesting offline alternative.

# Repo doesn't exist
uv run python workshop-1/3-car-racing/train_sb3.py --hf-repo nonexistent/repo \
    --timesteps 10000 --run-name bad --force
# Expected: HuggingFaceLoadError naming 'nonexistent/repo', suggesting check spelling.

# File not in repo
uv run python workshop-1/3-car-racing/train_sb3.py --hf-repo sb3/ppo-CarRacing-v0 \
    --hf-filename does-not-exist.zip --timesteps 10000 --run-name bad-file --force
# Expected: HuggingFaceLoadError naming the file, pointing to the repo's tree URL.
```

## Flow D — Workshop-leader pre-flight

Run before each Workshop 1 delivery on the leader's laptop. Validates SC-001 / SC-002 / SC-003 / SC-005 in one batch.

```bash
# (1) SC-001: SB3 from-scratch
uv run python workshop-1/3-car-racing/train_sb3.py --timesteps 10000 --run-name preflight-sb3 --force

# (2) SC-002: Custom PPO from-scratch
uv run python workshop-1/3-car-racing/train.py --timesteps 10000 --run-name preflight-custom --force

# (3) SC-003 + SC-004: HuggingFace fine-tune (cold + warm cache)
uv run python workshop-1/3-car-racing/train_sb3.py --hf-repo sb3/ppo-CarRacing-v0 \
    --timesteps 10000 --run-name preflight-hf-cold --force
uv run python workshop-1/3-car-racing/train_sb3.py --hf-repo sb3/ppo-CarRacing-v0 \
    --timesteps 10000 --run-name preflight-hf-warm --force

# (4) SC-005: MPS no slower than CPU on the CNN
RL_WORKSHOP_DEVICE=cpu uv run python workshop-1/3-car-racing/train.py --timesteps 10000 --run-name preflight-cnn-cpu --force
RL_WORKSHOP_DEVICE=mps uv run python workshop-1/3-car-racing/train.py --timesteps 10000 --run-name preflight-cnn-mps --force

# Compare wall-clock per update across the two runs:
python -c "
import json
for name in ('preflight-cnn-cpu', 'preflight-cnn-mps'):
    meta = json.load(open(f'runs/car-racing/{name}/meta.json'))
    metrics = [json.loads(l) for l in open(f'runs/car-racing/{name}/metrics.jsonl')]
    upd_times = [r.get('wall_time_seconds') for r in metrics if r.get('wall_time_seconds') is not None]
    deltas = [t2 - t1 for t1, t2 in zip(upd_times, upd_times[1:])]
    avg = sum(deltas[2:]) / max(1, len(deltas[2:]))   # skip 2-update warmup
    print(f'{meta[\"device\"]:>5}: {avg:.3f}s/upd  (over {len(deltas[2:])} updates)')
"

# Expected:
#   cpu: <X.XX>s/upd
#   mps: <Y.YY>s/upd  with Y <= X (within tolerance) — SC-005 PASS
```

## Run the existing PPO test suite (sanity check, runs C8 too)

```bash
RL_WORKSHOP_DEVICE=cpu uv run python workshop-1/1-ppo/ppo/tests/test_agent_interface.py --agent ppo
RL_WORKSHOP_DEVICE=mps uv run python workshop-1/1-ppo/ppo/tests/test_agent_interface.py --agent ppo  # Apple Silicon

# Expected last lines:
#   TEST 8 OK! (C8 CNN smoke on (4, 84, 84) vec env)
#   === Summary: 8 / 8 passed ===
```
