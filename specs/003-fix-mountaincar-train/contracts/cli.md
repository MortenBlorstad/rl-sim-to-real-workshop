# Contract: `train.py` CLI

**Date**: 2026-04-29
**Branch**: `003-fix-mountaincar-train`

## Invocation

```bash
uv run python workshop-1/2-mountaincar/train.py [OPTIONS]
```

## Flags

| Flag | Type | Default | Description |
|---|---|---|---|
| `--timesteps` | `int` | `200000` | Total environment steps for training. |
| `--run-name` | `str` | `None` | Run directory name under `runs/mountaincar/`. Defaults to UTC `YYYYMMDD-HHMMSS`. |
| `--no-eval` | flag | `False` | Skip the post-training `agent.evaluate(record_video=True)` call. |
| `--force` | flag | `False` | Overwrite an existing run directory of the same name. |

**Removed flag**: `--seed` is intentionally absent (per spec Q2). Seed lives in `hyperparameters["random_state"]` defined at module top of `train.py`.

## Exit codes

| Code | Meaning |
|---|---|
| `0` | Training completed; meta.json `status: "ok"` |
| `1` | `RunDirectoryExistsError` — clear message printed pointing at `--force` |
| `2` | Uncaught exception during training — `meta.json` `status: "error"` |
| `3` | Participant TODO is still raising `NotImplementedError` — friendly two-line message printed |
| `130` | `KeyboardInterrupt` — `meta.json` `status: "interrupted"` |

## Module-top constants (edited in code, not via CLI)

```python
DEFAULT_TIMESTEPS = 200_000
ENV_ID = "MountainCarContinuous-v0"
STAGE = "mountaincar"

hyperparameters: dict = {
    "rollout_size": 2048,
    "n_epochs": 10,
    "batch_size": 64,
    "lr": 1e-3,
    "gamma": 0.98,
    "gae_lambda": 0.95,
    "clip_eps": 0.2,
    "value_coef": 0.5,
    "entropy_coef": 0.01,
    "max_grad_norm": 0.5,
    "log_std_init": 0.0,
    "random_state": 42,
}
```

## Driver flow (FR-001..FR-009)

1. Parse args.
2. `env = gym.make(ENV_ID); env = NormalizeObs(env)` (Q4).
3. `agent = PPOAgent(env, hyperparameters=hyperparameters)`.
4. Compute `runs_root = <repo-root>/runs`.
5. Open `RunLogger(stage=STAGE, hyperparameters=hyperparameters, env_id=ENV_ID, agent_class=type(agent).__name__, seed=hyperparameters["random_state"], total_timesteps=args.timesteps, run_name=args.run_name, force=args.force, runs_root=runs_root)`. On `RunDirectoryExistsError`: print message, exit 1.
6. Inside the `with runlog:` block:
   - Build `log_fn = make_log_fn(runlog, agent)`.
   - Call `agent.train(env, total_timesteps=args.timesteps, random_state=hyperparameters["random_state"], log_fn=log_fn)`.
   - `agent.save(str(runlog.run_dir / "model.pt"))`.
   - If not `args.no_eval`: `agent.evaluate(env, n_episodes=1, record_video=True, video_dir=runlog.run_dir)`.
7. Outer `try/except`:
   - `KeyboardInterrupt` → print "[train] interrupted by user" to stderr; exit 130. (`RunLogger.__exit__` will write `status: "interrupted"`.)
   - `NotImplementedError` → print "[train] Looks like a TODO is still raising NotImplementedError. Fill it in and re-run." + the original message; exit 3.
   - Any other `Exception` → print `f"[train] error: {exc!r}"` to stderr; exit 2.
8. `finally`: `env.close()`.

## On-disk effects

After a successful run, `runs/mountaincar/<run-name>/` contains:

- `meta.json` (frozen schema; `status: "ok"`)
- `metrics.jsonl` (one line per PPO update; ~`total_timesteps / rollout_size` lines)
- `model.pt`
- `eval.mp4` (or `eval.mp4.skipped` if ffmpeg unavailable; or absent if `--no-eval`)

No other files are written.
