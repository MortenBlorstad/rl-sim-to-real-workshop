# Decision Log: PPO Skeleton

## Session 2026-04-13

### Multi-file module split

**Date**: 2026-04-13
**Status**: Rejected
**Context**: How to lay out the 5 TODO blocks — one file vs. separate modules per concern (`gae.py`, `policy.py`, `loss.py`, `train.py`).
**Decision**: Chose single-file `ppo_skeleton.py` with top-level functions.
**Rationale**: Beginners in a 3-hour session should not juggle 4–5 file tabs. A single file preserves the mental model while still giving `test_ppo.py` clean top-level import targets for each TODO. Matches Article V (progressive reveal, working foundations) without fragmenting the reading flow.

### pytest-based test runner

**Date**: 2026-04-13
**Status**: Rejected
**Context**: How to run per-TODO tests with the `--step N` CLI prescribed by Constitution Article IV.
**Decision**: Chose a ~50-line custom runner with a `@step(n, name)` decorator.
**Rationale**: Article IV mandates a specific output format (`"TODO 1 OK!"` / `"FAIL: expected shape ..."`) and the `--step N` interface. pytest's default output is noisy for beginners and its marker/selection system does not map cleanly to the required CLI. A tiny custom runner is cheaper to maintain than wrestling pytest into the prescribed shape.

### Plain-assert tests without a runner

**Date**: 2026-04-13
**Status**: Rejected
**Context**: Simplest possible test approach — inline `assert` statements, no framework.
**Decision**: Chose a custom runner that catches `NotImplementedError` and emits structured per-step results.
**Rationale**: Plain asserts stop at the first failure and cannot distinguish "TODO not yet implemented" from "TODO implemented incorrectly". The custom runner gives actionable beginner-facing output required by Articles I and IV.

### Inline solution snippets in `ppo_skeleton.py`

**Date**: 2026-04-13
**Status**: Rejected
**Context**: Where participants see the reference solution when stuck.
**Decision**: Chose the `solutions` branch plus per-TODO git checkpoint tags (`ws1-todo1-done`, …).
**Rationale**: Inline `# SOLUTION:` comments spoil the exercise for everyone the moment the file is opened. Article V/IX already prescribe the `solutions` branch and checkpoint tags — use them.

### Full-training assertion as the default TODO 5 test

**Date**: 2026-04-13
**Status**: Rejected
**Context**: How strictly `--step 5` should verify the training loop.
**Decision**: Chose a <10s smoke test as the default, with an opt-in `--step 5 --full` that runs a weak-learning assertion on CartPole.
**Rationale**: Article IV hard-caps each test at 10 seconds, which is not enough to prove learning. The smoke test verifies the loop runs end-to-end and returns the right dict shape; the opt-in `--full` check is available for participants who want more confidence.
