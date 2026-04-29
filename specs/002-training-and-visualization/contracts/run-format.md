# Contract: Run Directory Format

**Date**: 2026-04-29

This contract is the authoritative on-disk schema for `runs/<stage>/<run-name>/`. The format is the same regardless of which driver wrote it (custom-PPO `train.py` or SB3 `train_sb3.py`) so `analyze.ipynb` is path-agnostic.

## Directory Layout

```text
<run-dir>/
├── meta.json
├── metrics.jsonl
├── model.pt | model.zip
└── eval.mp4 | eval.mp4.skipped
```

- `<run-dir>` is `<repo-root>/runs/<stage>/<run-name>/` (gitignored) or `<repo-root>/pretrained/sample-runs/<stage>/<run-name>/` (tracked).
- `<stage>` ∈ {`mountaincar`, `car-racing`}.
- `<run-name>` is `YYYYMMDD-HHMMSS` (UTC) by default, or the `--run-name` value.

## `meta.json`

One JSON object. Single trailing newline. Pretty-printed with 2-space indent (so participants can read it directly in editors).

```json
{
  "stage": "mountaincar",
  "env_id": "MountainCarContinuous-v0",
  "agent_class": "MountainCarPPOAgent",
  "seed": 42,
  "total_timesteps": 50000,
  "hyperparameters": {
    "rollout_size": 1024,
    "n_epochs": 4,
    "batch_size": 64,
    "lr": 0.0003,
    "gamma": 0.99,
    "gae_lambda": 0.95,
    "clip_eps": 0.2,
    "value_coef": 0.5,
    "entropy_coef": 0.01,
    "max_grad_norm": 0.5
  },
  "git_sha": "d66e2440a1c4e7...",
  "started_at": "2026-04-29T15:42:18Z",
  "finished_at": "2026-04-29T15:46:31Z",
  "status": "ok",
  "python_version": "3.11.7",
  "torch_version": "2.4.1",
  "gymnasium_version": "0.29.1",
  "metric_definitions": {
    "update": "1-indexed PPO update counter",
    "timesteps": "cumulative environment steps after this update",
    "policy_loss": "PPO clipped surrogate loss (last minibatch of last epoch)",
    "value_loss": "MSE value loss (last minibatch of last epoch)",
    "entropy": "mean policy entropy this update (positive number; higher = more exploration)",
    "mean_return": "rolling mean of last 10 episode returns; current partial return if no episode finished",
    "log_std_mean": "exp(log_std).mean() — current exploration scale",
    "grad_norm": "post-clip gradient L2 norm; null for SB3 runs",
    "wall_time_seconds": "seconds since RunLogger initialization"
  }
}
```

### Required keys

`stage`, `env_id`, `agent_class`, `seed`, `total_timesteps`, `hyperparameters`, `git_sha`, `started_at`, `status`, `python_version`, `torch_version`, `gymnasium_version`, `metric_definitions` MUST be present at all times after `RunLogger.__init__` returns.

`finished_at` MUST be `null` while training is in progress and a UTC ISO 8601 string after `RunLogger.close()`.

### Status values

| `status` | Meaning |
|----------|---------|
| `"running"` | Training is in progress (or process died without `close()`) |
| `"ok"` | Training completed normally |
| `"interrupted"` | Caught `KeyboardInterrupt` (Ctrl+C) |
| `"error"` | Caught an unhandled exception |

## `metrics.jsonl`

One JSON object per line. UTF-8 encoded. Each line ends with `\n`. No header row.

Example (one line):

```json
{"update":12,"timesteps":12288,"policy_loss":-0.0234,"value_loss":1.4521,"entropy":1.0843,"mean_return":-87.32,"log_std_mean":0.6012,"grad_norm":0.4321,"wall_time_seconds":42.18}
```

### Required keys per line

All keys listed in `metric_definitions` MUST be present on every line of the same file. Types MUST be stable across lines (a key is either always `float`, always `int`, or always `null` — no mixing).

### NaN handling

`float("nan")` and `float("inf")` MUST be written as JSON `null`. The writer emits a single stderr warning the first time this happens per run.

### Atomicity

Each line is written via a single `file.write(json.dumps(record) + "\n"); file.flush()` pair. The final line in the file is either a complete JSON record terminated by `\n`, or absent — never a half-written record (SC-007).

## Models

| Path | Format | Loadable by |
|------|--------|-------------|
| `model.pt` | PyTorch `state_dict` saved by `PPOAgent.save()` | `PPOAgent.load(path)` (round-trips actor, value, log_std, hyperparameters, preprocess state) |
| `model.zip` | SB3 native serialization | `stable_baselines3.PPO.load(path)` |

The driver chooses the extension based on which path it is. `analyze.ipynb` does NOT load the model — it only reads `meta.json`, `metrics.jsonl`, and `eval.mp4`.

## Eval video

`eval.mp4`: H.264-encoded MP4 of one greedy evaluation episode, recorded via `gymnasium.wrappers.RecordVideo` (with the `imageio-ffmpeg` codec). Typical file size: 200 KB – 5 MB depending on episode length.

`eval.mp4.skipped`: a text file (~100 bytes) containing the exception message that prevented video recording. Written in lieu of `eval.mp4` when `RecordVideo` fails (FR-015).

Exactly one of `eval.mp4` or `eval.mp4.skipped` is present in a successful run directory; neither is present after `--no-eval` or after a Ctrl+C before the eval phase.

## Stability guarantees

- All field names (in `meta.json` and `metrics.jsonl`) MUST remain stable for the lifetime of this feature. Adding new fields is permitted; removing or renaming is a breaking change requiring a new spec.
- `analyze.ipynb` is the consumer reference. Any addition that breaks `pd.read_json(path, lines=True)` or any plot in the notebook constitutes a contract violation.
