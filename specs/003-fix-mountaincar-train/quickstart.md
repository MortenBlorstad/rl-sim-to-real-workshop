# Quickstart: Fix MountainCar Training Driver After PPO Refactor

**Date**: 2026-04-29
**Branch**: `003-fix-mountaincar-train`

This is the participant- and reviewer-facing walkthrough. After this feature lands, every command below should succeed on a fresh clone of this branch with the workshop1 dependency group installed.

## Prerequisites

```bash
uv sync --group workshop1
```

You should be on the `003-fix-mountaincar-train` branch (or its successor merge).

## 1. Run the test suite (custom step runners — no pytest)

The two test files at `workshop-1/1-ppo/ppo/tests/` validate the refactored PPO package.

```bash
# All five PPO TODOs (compute_gae, sample_action, evaluate_actions, ppo_loss, train smoke)
uv run python workshop-1/1-ppo/ppo/tests/test_ppo.py

# Only one TODO at a time
uv run python workshop-1/1-ppo/ppo/tests/test_ppo.py --step 1

# Agent contract (registry, predict, train smoke, save/load, evaluate)
uv run python workshop-1/1-ppo/ppo/tests/test_agent_interface.py --agent ppo
```

Expected output for a fully-completed checkout:

```
TODO 1 OK!
TODO 2 OK!
TODO 3 OK!
TODO 4 OK!
TODO 5 OK!

=== Summary: 5 / 5 passed ===
  TODO 1: PASS
  TODO 2: PASS
  TODO 3: PASS
  TODO 4: PASS
  TODO 5: PASS
```

If a TODO is still raising `NotImplementedError`, the runner prints `NOT_IMPLEMENTED` for that step and **other steps still run independently** (per the local-import convention).

## 2. Train a MountainCar agent

```bash
uv run python workshop-1/2-mountaincar/train.py
```

What you should see:
- One log line per PPO update (~`200000 / 2048 ≈ 97` lines), each in the fixed-width format produced by `format_update_line(...)`.
- A new directory under `runs/mountaincar/<run-name>/` containing `meta.json`, `metrics.jsonl`, `model.pt`.
- One greedy evaluation episode at the end → `eval.mp4` lands in the same directory.

A short smoke run (e.g. for CI or validation):

```bash
uv run python workshop-1/2-mountaincar/train.py --timesteps 4096 --run-name smoke --force
```

CLI flags (per `contracts/cli.md`):

| Flag | Default | Effect |
|---|---|---|
| `--timesteps INT` | `200000` | Total environment steps |
| `--run-name STR` | UTC timestamp | Run dir name under `runs/mountaincar/` |
| `--no-eval` | off | Skip the post-training video recording |
| `--force` | off | Overwrite existing run directory |

There is intentionally **no `--seed` flag**. The seed lives in `hyperparameters["random_state"]` at the top of `workshop-1/2-mountaincar/train.py` (default `42`). To run a multi-seed sweep, edit that value or write a small bash loop that does it.

## 3. Inspect the run

```bash
# meta.json (run config + status)
cat runs/mountaincar/<run-name>/meta.json

# Last few JSONL records
tail -n 5 runs/mountaincar/<run-name>/metrics.jsonl

# Open in the analysis notebook
uv run jupyter notebook workshop-1/2-mountaincar/analyze.ipynb
```

The on-disk schema is the same one feature 002 specified — `analyze.ipynb` opens new and old runs without changes.

## 4. Verify a round-trip locally

If you want a one-shot sanity check without launching a full training run:

```bash
uv run python workshop-1/1-ppo/ppo/tests/test_agent_interface.py --agent ppo
```

This exercises `__init__`, `predict`, `train(total_timesteps=512)`, `save → load → predict` equality, and `evaluate()` (with and without video recording, into a `tempfile.TemporaryDirectory()`). Wall-clock target: well under 30 seconds.

## 5. What if my training crashes mid-run?

- `Ctrl+C` → `meta.json` ends with `status: "interrupted"`. The partial `metrics.jsonl` is still readable.
- An unfilled TODO → exit code `3`, message: `"[train] Looks like a TODO is still raising NotImplementedError. Fill it in and re-run."`
- Any other exception → exit code `2`, `meta.json` ends with `status: "error"`. Re-run with `--force` to overwrite.

## 6. Known caveats / follow-ups

- `workshop-1/2-mountaincar/train_sb3.py` (the SB3 escape hatch) is **not** fixed by this feature. Its imports still reference deleted top-level helpers and will fail. Tracked in `research.md` (R6) as a follow-up spec.
- The constitution (`.specify/memory/constitution.md`) v1.0.0 mandates `Agent.preprocess()` and a corresponding test in `test_agent_interface.py`. This feature deliberately diverges from that — see `plan.md` Complexity Tracking. A constitution amendment is recommended as a follow-up (`research.md` R7).

## What changed vs. before

| Was | Now |
|---|---|
| `from ppo import PPOAgent` (flat-file `workshop-1/1-ppo/ppo.py`) | `from ppo import PPOAgent` (package `workshop-1/1-ppo/ppo/`) |
| `--seed` CLI flag | Edit `hyperparameters["random_state"]` in `train.py` |
| `from _runlog import RunLogger` (top-level helper) | `from ppo.utils import RunLogger` |
| `from _eval import record_eval_episode` | `agent.evaluate(env, ..., record_video=True)` |
| `agent.preprocess(obs)` inside `train()` | `NormalizeObs(env)` wrapper before constructing the agent |
| Tests at `workshop-1/1-ppo/test_ppo.py` (top-level) | Tests at `workshop-1/1-ppo/ppo/tests/test_ppo.py` |
