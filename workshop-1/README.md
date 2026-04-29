# Workshop 1 — From Theory to Sim (3 hours)

## Agenda

| Time | Block | What you do |
|------|-------|-------------|
| 0:00–0:35 | Intro to RL | Follow along with the lecture |
| 0:35–0:45 | Setup + warmup | Run `0-warmup/cartpole_random.py` |
| 0:45–1:45 | Implement PPO | Work on `1-ppo/ppo.py` |
| 1:45–2:00 | MountainCar challenge | Work on `2-mountaincar/` |
| 2:00–2:50 | CarRacing challenge | Work on `3-car-racing/` |
| 2:50–3:00 | Wrap-up | |

## Step 0: Warmup

Verify that everything is installed:

```bash
uv run python 0-warmup/cartpole_random.py
```

You should see a window with a cart and a pole. The pole falls over, and that's fine — the agent is taking random actions.

## Step 1: Implement PPO

Open `1-ppo/ppo.py`. You will find 5 `# TODO` blocks to fill in.

Work through them in order:

1. **TODO 1 — GAE:** Compute advantages with Generalized Advantage Estimation
2. **TODO 2 — Sample action:** Draw an action from the policy network
3. **TODO 3 — Evaluate actions:** Compute log-probability and entropy
4. **TODO 4 — PPO loss:** Implement the clipped surrogate objective
5. **TODO 5 — Training loop:** Wire rollout collection together with the update step

After each TODO, verify it with:

```bash
uv run python 1-ppo/test_ppo.py --step 1   # test TODO 1
uv run python 1-ppo/test_ppo.py --step 2   # test TODO 2
# etc.
```

When everything is done, train the agent:

```bash
uv run python 1-ppo/ppo.py
```

## Step 2: MountainCar challenge

The MountainCar car should be able to reach the top after roughly 200–500 episodes.

## Step 3: CarRacing challenge

Now we use Stable-Baselines3 instead of our own PPO.

```bash
uv run python 3-car-racing/train.py
```

Tasks:
1. Implement observation wrappers in `3-car-racing/wrappers.py`
2. Experiment with hyperparameters in `3-car-racing/train.py`
3. Evaluate the model: `uv run python 3-car-racing/evaluate.py`

**Competition:** Who gets the best average score over 10 episodes?

## Stuck?

- Ask the person next to you
- Ask the workshop leader
- Recover one specific TODO from the per-step solution checkpoints (see below)

### Per-TODO recovery checkpoints

Each TODO has its own git tag on the `solutions` branch. The tag points
at a state where TODOs `1..N` are solved and TODOs `N+1..5` still raise
`NotImplementedError`. To jump past TODO `N`, run:

```bash
git checkout ws1-todoN-done -- workshop-1/1-ppo/ppo.py
```

Replace `N` with `1`, `2`, `3`, `4`, or `5`. For example, to recover TODO 3:

```bash
git checkout ws1-todo3-done -- workshop-1/1-ppo/ppo.py
uv run python workshop-1/1-ppo/test_ppo.py
# TODOs 1-3: PASS, TODOs 4-5: NOT_IMPLEMENTED
```

> ⚠️ **Commit your work first.** This is a path-scoped checkout — it
> overwrites your local copy of `ppo.py`. Anything you had
> written for TODOs `N+1..5` will be lost. Run `git add` + `git commit`
> first if you want to keep that work.
