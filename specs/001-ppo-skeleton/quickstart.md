# Quickstart: Implementing the PPO Skeleton

**Spec**: 001-ppo-skeleton
**Audience**: someone implementing this spec, OR a workshop participant doing the exercise

This is a 5-minute walkthrough of how the artifacts in this feature fit together once they exist. Use it as the smoke-test of "did the implementation actually work end-to-end".

---

## Prerequisites

- Project root has `pyproject.toml` with a `workshop1` dependency group containing `torch`, `gymnasium`, `numpy`, and (for stage-3 stub) `opencv-python`.
- `uv sync --group workshop1` has been run successfully.
- You are on the `001-ppo-skeleton` git branch (or `solutions` for the reference implementation).

---

## Step 1: see the unmodified skeleton fail loudly

```bash
uv run python workshop-1/1-ppo/test_ppo.py
```

**Expected output**:

```text
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

Exit code is `1`. This is **success** — it proves the runner is wired up and every TODO raises the prescribed `NotImplementedError`.

---

## Step 2: verify the Article II contract test runs against `PPOAgent`

```bash
uv run python workshop-1/1-ppo/test_agent_interface.py --agent ppo
```

**Expected before any TODO is implemented**: most contract tests will report `NOT_IMPLEMENTED` (because `PPOAgent.predict` calls `sample_action`, which is TODO 2). The structural tests (registry membership, `preprocess` identity, `_get_preprocess_state` returning `{}`) MUST pass even on an unmodified skeleton.

**Expected after all TODOs are implemented**: every contract test passes, including:
- `test_preprocess_identity`
- `test_preprocess_deterministic`
- `test_predict_raw_obs_shape_and_range`
- `test_predict_deterministic_flag`
- `test_save_load_round_trip`
- `test_subclass_override_used_by_predict`
- `test_subclass_class_restored_on_load`

---

## Step 3: implement TODOs in order, verifying each

```bash
# Open ppo_skeleton.py, fill in TODO 1, then:
uv run python workshop-1/1-ppo/test_ppo.py --step 1
# Expect: TODO 1 OK!

# Fill in TODO 2:
uv run python workshop-1/1-ppo/test_ppo.py --step 2
# Expect: TODO 2 OK!

# ... and so on through TODO 5
```

Each `--step N` invocation MUST work even when later TODOs are still unfinished. If `--step 1` ever fails because TODO 2 is unfinished, that is a regression in the contract from `contracts/test-runner-cli.md` — the local-import isolation is broken.

---

## Step 4: run the script-mode training loop

After all five TODOs are complete:

```bash
uv run python workshop-1/1-ppo/ppo_skeleton.py
```

**Expected output** (numbers will vary slightly):

```text
[update  1/8] timesteps=  1024  policy_loss=+0.202  value_loss=+0.050  entropy=+1.402  mean_return=-50.18
[update  2/8] timesteps=  2048  policy_loss=+0.339  value_loss=+0.103  entropy=+1.386  mean_return=-50.44
[update  3/8] timesteps=  3072  policy_loss=+0.175  value_loss=+0.099  entropy=+1.373  mean_return=-50.63
[update  4/8] timesteps=  4096  policy_loss=+0.024  value_loss=+0.030  entropy=+1.356  mean_return=-50.58
[update  5/8] timesteps=  5120  policy_loss=+0.066  value_loss=+0.077  entropy=+1.343  mean_return=-50.08
[update  6/8] timesteps=  6144  policy_loss=+0.123  value_loss=+67.183  entropy=+1.341  mean_return=-32.99
[update  7/8] timesteps=  7168  policy_loss=+0.195  value_loss=+5.763  entropy=+1.343  mean_return=-19.13
[update  8/8] timesteps=  8192  policy_loss=+0.208  value_loss=+0.227  entropy=+1.328  mean_return=-22.92
✓ Training complete: entropy trending down (+1.4020 → +1.3280), no NaN losses.
```

**The script verifies on exit** (FR-027):
- No printed `policy_loss`, `value_loss`, or `entropy` was `nan`.
- The last `entropy` is strictly less than the first (the policy is concentrating, even when single-update `policy_loss` jitters non-monotonically).
- If either check fails, the script prints an actionable `FAIL: ...` line and exits non-zero.

**The agent does NOT need to solve MountainCarContinuous here.** The mean return stays negative; that is expected. Solving the environment is what stage `2-mountaincar/` is for.

---

## Step 5: verify the per-TODO recovery mechanism

Pick a TODO you've already solved, deliberately break it, and run the recovery command:

```bash
# Edit ppo_skeleton.py and break TODO 3
uv run python workshop-1/1-ppo/test_ppo.py --step 3
# Expect: FAIL: TODO 3 ...

# Recover just TODO 3 from the solutions branch
git checkout ws1-todo3-done -- workshop-1/1-ppo/ppo_skeleton.py
uv run python workshop-1/1-ppo/test_ppo.py --step 3
# Expect: TODO 3 OK!
```

Then check that TODOs 4 and 5 are now in their `ws1-todo3-done` state (still raising `NotImplementedError`, since the tag is "TODO 3 done, 4 and 5 not"):

```bash
uv run python workshop-1/1-ppo/test_ppo.py
# Expect: TODOs 1–3 PASS, TODOs 4–5 NOT_IMPLEMENTED
```

If you had your own work on TODOs 4 or 5 in your local copy, this will overwrite them — commit before recovering.

---

## Step 6: round-trip a trained `PPOAgent`

```bash
uv run python -c "
import sys
sys.path.insert(0, 'workshop-1/1-ppo')
from ppo_skeleton import PPOAgent
import gymnasium as gym, numpy as np

env = gym.make('MountainCarContinuous-v0')
agent = PPOAgent(obs_dim=2, action_dim=1)
agent.train(env, total_timesteps=2048)
agent.save('/tmp/ppo_quickstart.pt')

loaded = PPOAgent.load('/tmp/ppo_quickstart.pt')
obs, _ = env.reset(seed=0)
print('original action:', agent.predict(obs, deterministic=True))
print('loaded   action:', loaded.predict(obs, deterministic=True))
"
```

The `sys.path` insert is required because the stage-1 directory is `workshop-1/1-ppo/` (with hyphens and a leading digit), which is not a valid Python package path. Run the command from the repo root.

**Expected**: the two printed actions are identical. If they differ, `save` / `load` is dropping state — investigate which key in the `state_dict` is missing.

---

## Step 7: verify the subclass override stub from stage 2

```bash
uv run python -c "
import sys
sys.path.insert(0, 'workshop-1/1-ppo')
sys.path.insert(0, 'workshop-1/2-mountaincar')
from agent import MountainCarPPOAgent
agent = MountainCarPPOAgent(obs_dim=2, action_dim=1)
agent.save('/tmp/mc.pt')

from ppo_skeleton import PPOAgent
loaded = PPOAgent.load('/tmp/mc.pt')
print('loaded class:', type(loaded).__name__)
"
```

**Expected**:

```text
loaded class: MountainCarPPOAgent
```

This proves the class registry round-trip works — saving a subclass and loading via the base class yields the subclass back.

---

## Smoke-test checklist

When all of the following are true, this spec is implemented correctly:

- [ ] `uv run python workshop-1/1-ppo/test_ppo.py` runs all five steps and reports `5 / 5 passed` after solving every TODO (against the reference implementation).
- [ ] `uv run python workshop-1/1-ppo/test_ppo.py --step 1` passes when TODO 1 is solved AND TODOs 2–5 are unfinished.
- [ ] `uv run python workshop-1/1-ppo/test_agent_interface.py --agent ppo` reports all PASS against the reference implementation.
- [ ] `uv run python workshop-1/1-ppo/ppo_skeleton.py` runs in 1–3 minutes, prints 8 update lines with downward `entropy`, and exits zero.
- [ ] `git checkout ws1-todo3-done -- workshop-1/1-ppo/ppo_skeleton.py` produces a file where `--step 3` passes and `--step 4` is `NOT_IMPLEMENTED`.
- [ ] `MountainCarPPOAgent.save` followed by `PPOAgent.load` returns a `MountainCarPPOAgent`, not a plain `PPOAgent`.
- [ ] All print statements, comments, docstrings, and error messages in `ppo_skeleton.py` and `test_ppo.py` are in English.
