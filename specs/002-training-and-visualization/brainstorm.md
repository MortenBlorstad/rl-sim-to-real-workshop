# Brainstorm: Training Drivers, Logging, and Visualization for Workshop 1 Stages 2–3

**Date:** 2026-04-29
**Related issues:** none (the existing 31 open issues are all `001-ppo-skeleton` polish; none touch training drivers, logging, or visualization)

## Chosen Direction

Add per-stage training drivers and analysis notebooks to `workshop-1/2-mountaincar/` and `workshop-1/3-car-racing/` that follow a strict separation: **`train.py` writes a run directory; `analyze.ipynb` reads it.** Each `train.py` is a thin driver that calls `PPOAgent.train(env, total_timesteps, metrics_fn=runlog)` — no env-step / PPO-update loop is duplicated outside `ppo.train()`. Each driver writes a `runs/<stage>/<timestamp>/` directory containing `meta.json` (hyperparameters, env, seed, agent class, git SHA, start/end time), `metrics.jsonl` (one structured record per PPO update — `update`, `timesteps`, `policy_loss`, `value_loss`, `entropy`, `mean_return`, `std`, `grad_norm`), `model.pt`, and `eval.mp4` from a single greedy evaluation episode at end-of-training.

The primary log format is **JSONL**, chosen specifically so participants can `tail` the file and paste it into Claude / Copilot to debug ("here's what's happening, what's wrong"). TensorBoard is intentionally not the source of truth — its binary protobuf events lose this use case. Plumbing into the existing `ppo.train()` is one additive kwarg: `metrics_fn=None`, called with a dict per update; default no-op preserves backward compat. A small shared `workshop-1/_runlog.py` (~50 lines) provides `RunLogger` with `__call__(dict)` writing JSONL + meta.json.

Visualization lives in `analyze.ipynb` per stage. It loads the latest run directory (or any past run by path), plots return/loss/entropy curves from `metrics.jsonl`, and replays `eval.mp4` inline. MountainCar gets a bonus value-function heatmap and policy-mean quiver over the 2-D position×velocity grid — feasible because the state is genuinely 2-D and pedagogically strong for a 15-minute slot.

Both stages use the **custom PPO** as the primary path, with **SB3 as an explicit alternative** in `3-car-racing/train_sb3.py` writing the same JSONL schema so `analyze.ipynb` works for either path. CarRacing's custom path requires enabling work flagged below as in-scope.

## Key Decisions

- **`train.py` for training, `analyze.ipynb` for analysis — in both stages.** Separation prevents kernel-timeout / hidden-state / accidental-rerun failures during long training cells, lets `Ctrl+C` work cleanly, and makes the lecturer flow possible (demo a pre-shipped run's notebook while participants kick off their own).
- **Single greedy eval episode at end-of-training** (not N=10) — confirmed by user. Saves wall time and the recorded mp4 is the artifact participants actually look at.
- **JSONL as primary log format.** One self-describing dict per line. Optimizes for the "paste this to Claude" debugging path. CSV/TSV rejected because column meanings live only in the header. TensorBoard rejected as primary because the events file is binary.
- **Additive `metrics_fn=None` kwarg in `ppo.train()`.** Receives a structured dict per PPO update; default no-op. Avoids the brittle alternative of parsing `format_update_line` output in the driver.
- **Run directory layout:** `runs/<stage>/<timestamp>/{meta.json, metrics.jsonl, model.pt, eval.mp4}`. Self-contained, time-ordered, easy to `rm -rf` old runs, easy to ship a "sample run" alongside the workshop.
- **Custom PPO is primary on both stages.** SB3 stays available in `3-car-racing/train_sb3.py` as the documented escape hatch (per CLAUDE.md), writing the same JSONL schema so `analyze.ipynb` is path-agnostic.
- **Enabling work for custom PPO on CarRacing (in-scope, named explicitly):**
  - **`CarRacingPPOAgent` gets a CNN actor + CNN value network.** The current stub only has pixel preprocess; its `train()` would crash on the stage-1 MLP. The CNN replaces `ActorNetwork`/`ValueNetwork` for this subclass.
  - **Gymnasium preprocess wrappers, not agent-side preprocess at training time.** Stack `GrayScaleObservation` → `ResizeObservation(84)` → `FrameStack(4)`. The wrapped env delivers `(4, 84, 84)` obs directly to `ppo.train()`. The agent's `preprocess()` becomes inference-only (used by `predict()` at deployment).
  - **`RolloutBuffer` generalizes from flat `(size, obs_dim)` to obs-space-shape-aware storage.** Small `__init__`/`add`/`get_batches` change.
- **No live plots during training in v1.** Console prints from `format_update_line` already give participants something to watch. Live matplotlib in Jupyter is a fragility tax we don't pay yet.

## Constraints & Trade-offs

- **No new top-level dependency.** `tensorboard`, `matplotlib`, `imageio`, `stable-baselines3` are already in the `workshop1` group. Eval-video recording uses `gymnasium.wrappers.RecordVideo` (which uses `moviepy`/`imageio-ffmpeg` under the hood — verify availability during spec or add `imageio[ffmpeg]` to the dep group).
- **The CarRacing CNN agent is meaningful new code that doesn't exist yet.** Not a small driver-only change. The brainstorm names this so the spec sizes it correctly.
- **`RolloutBuffer` shape generalization touches existing tested code.** Per `001-ppo-skeleton` Article IV, this means re-running step 1–5 tests after the change. Acceptable but worth budgeting.
- **Notebook outputs are noisy in git diffs.** The notebooks should be committed *cleared of outputs* (or with `nbstripout`). The lecturer/sample run still produces the visible artifacts at workshop time.
- **JSONL grows linearly with updates.** At ~1 record per PPO update and CarRacing runs of ~hundreds of updates, file sizes stay under ~100 KB — fine.
- **`mountaincar.ipynb` currently exists empty (0 bytes).** Renaming/replacing it with `analyze.ipynb` is a minor disruption but cleaner than reusing the empty file.

## Discarded Alternatives

- **TensorBoard as primary log format** — discarded because binary event files defeat the "paste to Claude" debugging use case. May still appear as an *optional* mirror writer; not v1.
- **CSV / TSV as primary log format** — discarded because column meanings live only in the header, less self-describing per line for LLM consumption. JSONL is strictly better for this audience.
- **Parse `format_update_line` output in the driver** — discarded because brittle (silently breaks if the format string changes); the additive `metrics_fn` kwarg is ~5 lines and decoupled.
- **Training cell *inside* the notebook** — discarded because long-running Jupyter cells are a known workshop hazard (kernel disconnects, accidental re-runs, no clean Ctrl+C, hidden-state bugs). Worst for the longer CarRacing run.
- **Notebook-only (no `train.py`)** — discarded for the same reason; also makes background/headless training impossible.
- **`.py`-only (no notebook)** — discarded because the visualization payoff (state-space heatmap, inline mp4 playback, plotted curves) is significantly stronger inline in a notebook for the workshop audience.
- **N=10 evaluation episodes** — discarded per user decision; 1 greedy episode is the chosen scope.
- **SB3 everywhere** — discarded because it would skip exercising the custom PPO end-to-end on a pixel environment, undermining the workshop's pedagogical arc. SB3 stays available as the documented escape hatch.
- **Agent-side preprocessing at training time (refactor `ppo.train()` to call `agent.preprocess(obs)`)** — discarded in favor of Gymnasium wrappers. The wrapper path keeps `train()` decoupled from `PPOAgent` and is the idiomatic Gymnasium pattern.
- **Live matplotlib plots updating during training** — discarded for v1. Console metric prints are sufficient; live-plot UX in Jupyter is fragile across VS Code / Colab / classic Jupyter.
- **`tqdm` / `rich` live progress dashboard** — discarded for v1. Adds moving parts; the formatted update lines already give a clear text progress signal.
