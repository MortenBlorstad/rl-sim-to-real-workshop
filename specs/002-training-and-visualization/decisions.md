# Decision Log: Training Drivers, Logging, and Visualization

## Session 2026-04-29

### TensorBoard as primary log format

**Date**: 2026-04-29
**Status**: Rejected
**Context**: Choosing the primary log format that participants can both watch live and paste into Claude / Copilot to debug a broken run.
**Decision**: JSONL (one structured dict per PPO update) is the primary format, written by a small shared `RunLogger` in `workshop-1/_runlog.py`.
**Rationale**: TensorBoard's `events.out.tfevents.*` files are binary protobuf — useless for the "paste this log to an LLM" debugging path that the user explicitly wants. TB may still appear as an optional mirror, but is not v1 scope.

### CSV / TSV as primary log format

**Date**: 2026-04-29
**Status**: Rejected
**Context**: Same as above — primary log format choice.
**Decision**: JSONL chosen instead.
**Rationale**: Header-defined column meanings make CSV less self-describing per line. JSONL records carry their schema with them, which matters for both human inspection and LLM consumption.

### Parse `format_update_line` output in the driver

**Date**: 2026-04-29
**Status**: Rejected
**Context**: Plumbing structured metrics out of `ppo.train()`, which currently calls `log_fn(formatted_string)`.
**Decision**: Add an additive `metrics_fn=None` kwarg to `ppo.train()` that is called with a dict per update; default no-op preserves backward compat.
**Rationale**: String-parsing the formatted log line is brittle — a future change to `format_update_line` would silently break the driver. The kwarg is ~5 lines of additive code and decouples cleanly. `_main()` already does this string-parsing for its exit checks; the parser is acceptable there but should not be the model for the new drivers.

### Training cell inside the notebook

**Date**: 2026-04-29
**Status**: Rejected
**Context**: Whether `*.ipynb` should drive training itself, or only analyze a run produced by `train.py`.
**Decision**: `train.py` does the training; `analyze.ipynb` only reads run directories.
**Rationale**: Long-running Jupyter cells are a known workshop hazard — kernel disconnects, accidental re-runs, no clean `Ctrl+C`, hidden-state bugs. Worst for CarRacing's 10–30 min runs. The split also automatically honors the project rule that stage train scripts are thin drivers (no env-step / PPO-update loop duplication outside `ppo.train()`).

### Notebook-only (no `train.py`)

**Date**: 2026-04-29
**Status**: Rejected
**Context**: Whether to ship only a notebook for each stage and skip a `.py` driver.
**Decision**: Both `train.py` and `analyze.ipynb` per stage.
**Rationale**: Same as the previous entry — long-running cells fail badly under workshop conditions. Notebook-only also makes headless / backgrounded training impossible.

### `.py`-only (no notebook)

**Date**: 2026-04-29
**Status**: Rejected
**Context**: Whether to ship only `.py` and skip the notebook entirely.
**Decision**: Both `train.py` and `analyze.ipynb` per stage.
**Rationale**: The visualization payoff — state-space heatmap, inline mp4 playback, training curves — lands significantly better inline in a notebook than as a popped-out matplotlib window. For a beginner-heavy 3-hour workshop, the notebook's "see the artifact next to the explanation" pattern is worth the cost.

### N=10 evaluation episodes at end-of-training

**Date**: 2026-04-29
**Status**: Rejected
**Context**: How many greedy evaluation episodes to run at end-of-training and record.
**Decision**: 1 greedy evaluation episode, recorded to `eval.mp4`.
**Rationale**: User decision. Saves wall time in a tight workshop budget; the mp4 is what participants actually look at, and one episode suffices for that purpose. Mean/std over multiple eval episodes is not load-bearing when the visualization is qualitative.

### SB3 everywhere (both stages use Stable-Baselines3)

**Date**: 2026-04-29
**Status**: Rejected
**Context**: Whether to standardize on SB3 across both stages, since CarRacing's custom-PPO path requires meaningful enabling work (CNN agent, env wrappers, buffer generalization).
**Decision**: Custom PPO is primary on both stages; SB3 lives as an explicit alternative in `3-car-racing/train_sb3.py` writing the same JSONL schema.
**Rationale**: SB3-everywhere would skip exercising the custom PPO end-to-end on a pixel environment, undermining the workshop's pedagogical arc. The escape-hatch pattern (per CLAUDE.md) keeps SB3 available without making it the default.

### Agent-side preprocessing at training time

**Date**: 2026-04-29
**Status**: Rejected
**Context**: Making custom PPO work on CarRacing pixels — `ppo.train()` currently never calls `agent.preprocess()`, so the existing pixel-preprocess pipeline in `CarRacingPPOAgent` is unused during training.
**Decision**: Use Gymnasium wrappers (`GrayScaleObservation` → `ResizeObservation(84)` → `FrameStack(4)`) so the wrapped env delivers `(4, 84, 84)` obs to `ppo.train()` directly. The agent's `preprocess()` becomes inference-only.
**Rationale**: Refactoring `ppo.train()` to call `agent.preprocess(obs)` would couple the module-level training function to `PPOAgent`, which it currently does not depend on. The wrapper path is the idiomatic Gymnasium pattern, leaves `ppo.train()` general, and avoids per-step Python preprocessing overhead inside the rollout loop.

### Live matplotlib plots updating during training

**Date**: 2026-04-29
**Status**: Rejected
**Context**: Whether the notebook (or driver) should render live training curves that update each PPO update.
**Decision**: No live plots in v1. Console metric prints from `format_update_line` are the live signal; plots are produced post-hoc in `analyze.ipynb`.
**Rationale**: Live-matplotlib UX in Jupyter is fragile across VS Code notebooks / Colab / classic Jupyter, and adds moving parts that can fail mid-workshop. The console line per update is already informative ("mean_return going up = it's working"). Revisit if a future workshop iteration shows participants want this.

### `tqdm` / `rich` live progress dashboard

**Date**: 2026-04-29
**Status**: Rejected
**Context**: Whether to add a live text dashboard (progress bar, table) on top of the existing per-update prints.
**Decision**: Stick with the existing `format_update_line` per-update print.
**Rationale**: Adds dependencies and moving parts for a marginal UX gain. The fixed-width format string is already greppable, parseable, and visible. Revisit only if workshop feedback specifically calls for it.
