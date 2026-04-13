# Contract: `test_ppo.py` CLI

**Spec**: 001-ppo-skeleton
**Source of truth**: spec FR-010 through FR-016, FR-020 through FR-024

This is the binding contract for `workshop-1/1-ppo/test_ppo.py`. It is the participant's primary feedback loop.

---

## CLI surface

```text
uv run python workshop-1/1-ppo/test_ppo.py [--step N]
```

**Arguments**:

| Argument | Type | Default | Description |
|---|---|---|---|
| `--step N` | int (1–5) | unset | Run only the test for TODO N. If unset, run all five tests in order. |

**Exit codes**:

| Code | Meaning |
|---|---|
| 0 | Every test that ran returned `PASS`. |
| 1 | At least one test that ran returned `FAIL` or `NOT_IMPLEMENTED`. |
| 2 | CLI argument error (e.g. `--step 99`). |

---

## Output format

### Per-test lines

A test produces exactly one of these three lines after it runs:

```text
TODO N OK!
FAIL: TODO N (<name>): <expected vs observed>
TODO N not yet implemented (<name>): <message>
```

Where `<name>` is the human-readable label registered via `@step(N, "<name>")` (e.g. `"GAE"`, `"PPO loss"`).

The `FAIL: ...` line MUST always include a concrete expected-vs-observed phrasing, e.g.:

```text
FAIL: TODO 1 (GAE): expected shape (4,), got (4, 1)
FAIL: TODO 4 (PPO loss): expected scalar tensor, got shape (64,)
FAIL: TODO 3 (evaluate actions): log_prob[0] off by 0.012 (expected -1.234, got -1.246)
```

### Summary line (only when no `--step` flag)

When invoked with no `--step`, after running all five tests, the runner prints:

```text
=== Summary: 3 / 5 passed ===
  TODO 1: PASS
  TODO 2: PASS
  TODO 3: FAIL
  TODO 4: NOT_IMPLEMENTED
  TODO 5: NOT_IMPLEMENTED
```

The summary block MUST appear exactly once, at the end, regardless of how many tests passed or failed.

### Single-step output

When invoked with `--step N`, only that test's per-test line is printed (no summary block).

---

## Independence guarantee (FR-011)

Running `--step N` against a skeleton where TODOs other than `N` still raise `NotImplementedError` MUST report TODO `N`'s result correctly without crashing on the unrelated TODOs. This is achieved by having each registered step function do a **local import** of only the symbol it needs, e.g.:

```python
@step(1, "GAE")
def test_step_1():
    from ppo_skeleton import compute_gae   # local: doesn't trip TODOs 2..5
    ...
```

A top-level `from ppo_skeleton import *` is **prohibited** in `test_ppo.py`.

---

## Time budget (FR-012)

Each test function MUST complete in under 10 seconds on a standard laptop CPU. The runner does NOT enforce this with a hard timeout, but each test is expected to be inherently fast:

| TODO | Expected runtime | Why it's fast |
|---|---|---|
| 1 (GAE) | < 0.1 s | Pure tensor math on a 4-step toy trajectory |
| 2 (sample action) | < 1 s | 1000 samples from a tiny network |
| 3 (evaluate actions) | < 0.5 s | Batch of 32, single forward pass |
| 4 (PPO loss) | < 0.1 s | Pure tensor math on synthetic batch |
| 5 (training loop smoke) | 5–8 s | 512 timesteps of MountainCarContinuous-v0, ~1 update |

Total `test_ppo.py` (no flags) target: < 30 seconds (SC-004 budget is 60 s).

---

## Failure-categorization wrapper

Every registered step is invoked through this wrapper:

```python
def _run_step(n: int) -> str:
    name, fn = STEPS[n]
    try:
        fn()
    except NotImplementedError as e:
        print(f"TODO {n} not yet implemented ({name}): {e}")
        return "NOT_IMPLEMENTED"
    except AssertionError as e:
        print(f"FAIL: TODO {n} ({name}): {e}")
        return "FAIL"
    print(f"TODO {n} OK!")
    return "PASS"
```

The wrapper distinguishes the three states by exception type:

- **`NotImplementedError`** → `NOT_IMPLEMENTED`. This is what an unfinished TODO raises; treat as "step not run yet".
- **`AssertionError`** → `FAIL`. The test ran but the participant's implementation is wrong. The assertion message MUST be the expected-vs-observed phrasing.
- **No exception** → `PASS`. Print `TODO N OK!`.
- **Any other exception** propagates and the script crashes. Participants will see the traceback; this catches "the runner itself is broken" cases that we don't want to silently swallow.

---

## Examples (executable form)

### Example 1: fresh skeleton, run all

```text
$ uv run python workshop-1/1-ppo/test_ppo.py
TODO 1 not yet implemented (GAE): TODO 1: compute generalized advantage estimation
TODO 2 not yet implemented (sample action): TODO 2: sample an action from the policy
TODO 3 not yet implemented (evaluate actions): TODO 3: compute log_prob and entropy
TODO 4 not yet implemented (PPO loss): TODO 4: clipped surrogate objective
TODO 5 not yet implemented (training loop): TODO 5: wire rollout + update
=== Summary: 0 / 5 passed ===
  TODO 1: NOT_IMPLEMENTED
  TODO 2: NOT_IMPLEMENTED
  TODO 3: NOT_IMPLEMENTED
  TODO 4: NOT_IMPLEMENTED
  TODO 5: NOT_IMPLEMENTED
```

Exit code: `1`.

### Example 2: TODO 1 done, rest unfinished, run only step 1

```text
$ uv run python workshop-1/1-ppo/test_ppo.py --step 1
TODO 1 OK!
```

Exit code: `0`.

### Example 3: TODO 1 done, run all

```text
$ uv run python workshop-1/1-ppo/test_ppo.py
TODO 1 OK!
TODO 2 not yet implemented (sample action): TODO 2: sample an action from the policy
TODO 3 not yet implemented (evaluate actions): TODO 3: compute log_prob and entropy
TODO 4 not yet implemented (PPO loss): TODO 4: clipped surrogate objective
TODO 5 not yet implemented (training loop): TODO 5: wire rollout + update
=== Summary: 1 / 5 passed ===
  TODO 1: PASS
  TODO 2: NOT_IMPLEMENTED
  TODO 3: NOT_IMPLEMENTED
  TODO 4: NOT_IMPLEMENTED
  TODO 5: NOT_IMPLEMENTED
```

Exit code: `1` (only 1 / 5 passed).

### Example 4: buggy GAE implementation

```text
$ uv run python workshop-1/1-ppo/test_ppo.py --step 1
FAIL: TODO 1 (GAE): expected advantages[2]=0.234, got 0.198 (off by 0.036, atol=1e-5)
```

Exit code: `1`.

### Example 5: all five done, run all

```text
$ uv run python workshop-1/1-ppo/test_ppo.py
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

Exit code: `0`.

---

## Out of scope

- A `--full` mode for TODO 5 (rejected during clarification — that belongs to stage 2).
- A `--verbose` mode (the prescribed output is already terse and complete).
- JSON / machine-readable output (this is a participant-facing tool).
- Any test discovery beyond the `STEPS` dict (no pytest-style auto-collection).
