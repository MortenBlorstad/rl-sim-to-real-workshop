# Contract: Driver CLI Surface

**Date**: 2026-04-29

The four driver scripts (`workshop-1/2-mountaincar/train.py`, `train_sb3.py`, `workshop-1/3-car-racing/train.py`, `train_sb3.py`) MUST share the same argparse surface, defaults differing only on the per-stage `--timesteps`.

## Flags

| Flag | Type | Default | Purpose | Source |
|------|------|---------|---------|--------|
| `--timesteps N` | int | **MountainCar: 50000; CarRacing: 200000** | Total env steps to train. | FR-014, Q1 |
| `--seed N` | int | 42 | Re-seeds Python `random`, NumPy, PyTorch, env. | FR-014 |
| `--run-name NAME` | str | `YYYYMMDD-HHMMSS` (UTC, current time at start) | Override the run-directory name. | FR-014, Q5 |
| `--no-eval` | flag (no value) | off | Skip the greedy evaluation episode and `eval.mp4` recording. | FR-014 |
| `--force` | flag (no value) | off | Overwrite an existing `runs/<stage>/<run-name>/` directory instead of erroring. | FR-014, Q5 |

No other flags are accepted. **PPO hyperparameters MUST NOT be exposed as CLI flags** (Q3) — participants edit the driver script directly to experiment.

## Behavior

1. Parse args. Resolve run name + path: `<repo-root>/runs/<stage>/<run-name>`.
2. If the run directory already exists and `--force` is not passed, exit non-zero with:
   ```
   Error: run directory '<path>' already exists. Pick a different --run-name or pass --force to overwrite.
   ```
   If `--force` is passed, `shutil.rmtree(<path>)` then re-create.
3. Construct the env, agent (or SB3 model), `RunLogger`.
4. Train via `agent.train(env, args.timesteps, metrics_fn=runlog)` (custom path) or `model.learn(args.timesteps, callback=Sb3JsonlCallback(runlog))` (SB3 path).
5. If `args.no_eval` is False: run one greedy evaluation episode via `gymnasium.wrappers.RecordVideo`, writing `eval.mp4`. On `RecordVideo` failure: write `eval.mp4.skipped` containing the exception message; continue.
6. Save the model (`agent.save(...)` or `model.save(...)`).
7. Call `runlog.close("ok")`.

The driver MUST use a `with RunLogger(...) as runlog:` context manager so that `close("interrupted")` runs on `KeyboardInterrupt` and `close("error")` runs on any other exception.

## Exit codes

| Code | Condition |
|------|-----------|
| 0 | Training completed (with or without eval video) |
| 1 | Run-directory collision without `--force` |
| 2 | Unhandled exception during training (run dir is left in place with `meta.json.status = "error"`) |
| 130 | `KeyboardInterrupt` during training (run dir preserved with `status = "interrupted"`) |

## Stage-specific differences

The four drivers differ ONLY in:

- **`workshop-1/2-mountaincar/train.py`** — env: `gymnasium.make("MountainCarContinuous-v0")` (no wrappers). Agent: `MountainCarPPOAgent`. Default `--timesteps`: 50000.
- **`workshop-1/2-mountaincar/train_sb3.py`** — env: `gymnasium.make("MountainCarContinuous-v0")` (no wrappers). Model: `stable_baselines3.PPO("MlpPolicy", env, ...)`. Default `--timesteps`: 50000.
- **`workshop-1/3-car-racing/train.py`** — env: `gymnasium.make("CarRacing-v3", render_mode="rgb_array")` (no wrappers; preprocessing happens inside `ppo.train()` via `agent.preprocess()`). Agent: `CarRacingPPOAgent`. Default `--timesteps`: 200000.
- **`workshop-1/3-car-racing/train_sb3.py`** — env: `gymnasium.make("CarRacing-v3", render_mode="rgb_array")` wrapped with `GrayScaleObservation` → `ResizeObservation(84, 84)` → `FrameStackObservation(4)` (Article II SB3 exemption). Model: `stable_baselines3.PPO("CnnPolicy", env, ...)`. Default `--timesteps`: 200000.

All four MUST share the implementation of:

- the argparse setup (factor it into a helper if needed)
- the `RunLogger` instantiation and context-manager pattern
- the eval-recording fallback to `eval.mp4.skipped`
- the model-save step

## Driver size cap

Each driver MUST be ≤ 100 lines (excluding imports and module docstring; SC-008). If you can't fit it, factor shared pieces into `workshop-1/_runlog.py` or `workshop-1/_sb3_jsonl_callback.py` (already planned).
