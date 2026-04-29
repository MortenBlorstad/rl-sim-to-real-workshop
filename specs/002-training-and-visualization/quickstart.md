# Quickstart: Stage 2 + Stage 3 with Training Drivers and Analysis Notebooks

**Date**: 2026-04-29
**Audience**: workshop participants who have just finished the `1-ppo/ppo.py` TODOs.

## Prerequisites

- `uv sync --group workshop1` has been run successfully.
- All five PPO TODOs in `workshop-1/1-ppo/ppo.py` pass `uv run python workshop-1/1-ppo/test_ppo.py`.
  - If not: use the per-TODO recovery tags as documented in `workshop-1/README.md`, or fall back to the SB3 alternative path described below.

## Stage 2: MountainCar (target time: ~10 minutes)

### Train

```bash
uv run python workshop-1/2-mountaincar/train.py
```

Defaults: `--timesteps 50000`, `--seed 42`. You should see one fixed-width log line per PPO update streaming to your terminal:

```
[update  1/50] timesteps=  1024  policy_loss=-0.012  value_loss=+1.234  entropy=+1.23  mean_return=-100.42
...
```

After the loop completes, the script will run **one** greedy evaluation episode and write `eval.mp4`. Total wall time: 3–5 minutes on a typical laptop CPU.

A new directory appears at `runs/mountaincar/<timestamp>/` containing:

- `meta.json` — run-level metadata (env, agent class, seed, hyperparameters, git SHA, library versions, start/end timestamps)
- `metrics.jsonl` — one JSON dict per update; paste-this-to-Claude for debugging
- `model.pt` — trained model weights
- `eval.mp4` — recording of one greedy evaluation episode

### Analyze

```bash
uv run jupyter notebook workshop-1/2-mountaincar/analyze.ipynb
```

(Or just open it in VS Code's Jupyter integration.) The notebook auto-discovers your latest run directory, then:

1. Plots the training curves: `mean_return`, `policy_loss`, `value_loss`, `entropy`.
2. Renders a value-function heatmap over the (position, velocity) state space.
3. Renders a policy-mean quiver over the same grid.
4. Embeds `eval.mp4` for inline playback.

If you want to look at someone else's run (e.g., the lecturer's pretrained sample), edit the second cell and set:

```python
RUN_DIR = "pretrained/sample-runs/mountaincar/sample"
```

### Debugging help

If your training looks broken, paste the contents of `meta.json` and the last ~50 lines of `metrics.jsonl` into a Claude / Copilot chat with the prompt "is anything obviously wrong?" — the JSONL log carries its own schema, so the LLM has enough context to diagnose common failure modes (NaN losses, entropy stuck, returns flat).

### SB3 alternative (escape hatch)

If your stage 1 PPO is incomplete, you can train MountainCar with Stable-Baselines3 instead:

```bash
uv run python workshop-1/2-mountaincar/train_sb3.py
```

The output run directory has the same layout, and `analyze.ipynb` works against it identically.

## Stage 3: CarRacing (target time: ~30 minutes)

### Train

```bash
uv run python workshop-1/3-car-racing/train.py
```

Defaults: `--timesteps 200000`, `--seed 42`. CarRacing trains a CNN-equipped `CarRacingPPOAgent` on `(4, 84, 84)` preprocessed observations (grayscale → resize → frame-stack 4) — preprocessing is invoked **inside** the training loop via the agent's `preprocess()` method, no Gymnasium wrappers in the way. Wall time: 15–25 minutes on a typical laptop CPU.

The output `runs/car-racing/<timestamp>/` directory has the same four files as stage 2.

### Analyze

```bash
uv run jupyter notebook workshop-1/3-car-racing/analyze.ipynb
```

Auto-discovers your latest run, plots the four training curves, embeds `eval.mp4`. (No state-space heatmap — CarRacing observations are pixels.)

### SB3 alternative

```bash
uv run python workshop-1/3-car-racing/train_sb3.py
```

This path uses Gymnasium standalone wrappers (`GrayScaleObservation` → `ResizeObservation(84, 84)` → `FrameStackObservation(4)`) before SB3's `PPO("CnnPolicy", env)` because SB3's training loop is closed and can't accept the `agent.preprocess()` injection that the custom path uses. The output run directory has the same layout.

## Common CLI flags (all four drivers)

| Flag | Default | Purpose |
|------|---------|---------|
| `--timesteps N` | 50000 (MountainCar) / 200000 (CarRacing) | Total environment steps to train. |
| `--seed N` | 42 | RNG seed for reproducibility. |
| `--run-name NAME` | timestamp | Override the run-directory name. Useful for `pretrained/sample-runs/<stage>/<NAME>` lecturer fixtures. |
| `--no-eval` | (off) | Skip the eval episode and `eval.mp4` recording. |
| `--force` | (off) | Allow overwriting an existing run directory. |

PPO hyperparameters (lr, batch size, etc.) are NOT exposed as CLI flags — to experiment, edit the driver script directly. The drivers are ≤ 100 lines each, by design.

## What if `eval.mp4` fails?

If video recording fails on your machine (no codec, headless display, `imageio-ffmpeg` not installed), the script writes a `eval.mp4.skipped` text file in place of the video and continues normally. `analyze.ipynb` shows a graceful "no video for this run" message in that case. Training artifacts (`metrics.jsonl`, `model.pt`) are unaffected.

## Verification

After completing both stages, you should have on disk:

```text
runs/
├── mountaincar/
│   └── <timestamp>/{meta.json, metrics.jsonl, model.pt, eval.mp4}
└── car-racing/
    └── <timestamp>/{meta.json, metrics.jsonl, model.pt, eval.mp4}
```

And both `analyze.ipynb` notebooks render their respective runs without manual cell editing.
