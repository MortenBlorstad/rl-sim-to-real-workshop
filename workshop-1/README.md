# Workshop 1 — From RL theory to a trained agent

Three hours, hands-on. You'll go from the RL basics through implementing
PPO from scratch to training an agent that plays CarRacing.

## Goal

By the end of the session you have a trained agent (Pendulum or CarRacing)
saved to disk, with an evaluation video to prove it works. The path to
get there is your choice — implement PPO from scratch, or use Stable-
Baselines3 — and there's a HuggingFace fine-tune route if you're short on
time.

## Agenda (3 hours)

| Time          | Block                       | What you do                                                               |
|---------------|-----------------------------|---------------------------------------------------------------------------|
| 0:00 — 0:35   | Intro to RL                 | Follow along with the lecture                                             |
| 0:35 — 0:45   | Setup + warmup              | `uv sync`, run `0-warmup/cartpole_random.py`                              |
| 0:45 — 1:45   | Implement PPO (5 TODOs)     | Work through `1-ppo/ppo/{ppo.py, rollout_buffer.py}`. Tests after each.   |
| 1:45 — 2:00   | Pendulum challenge          | Train a stage-2 agent (custom PPO or SB3)                                 |
| 2:00 — 2:50   | CarRacing challenge         | Train a stage-3 agent (custom PPO, SB3, or HF fine-tune)                  |
| 2:50 — 3:00   | Wrap-up                     | Show your eval video, brief discussion                                    |

## How PPO works — the big picture
 
PPO has two neural networks and a data buffer working together in a loop:
 
```
┌──────────────────────────────────────────────────────────────┐
│                        PPO Loop                              │
│                                                              │
│   1. COLLECT         2. SCORE           3. UPDATE            │
│                                                              │
│   Actor plays     → RolloutBuffer    → Actor & Critic        │
│   the game          stores all the     learn from the        │
│   (sample_action)   transitions        collected data        │
│                     (obs, action,      (ppo_loss +           │
│   Critic scores     reward, done,       value loss +         │
│   each state        log_prob, value)    entropy bonus)       │
│   (value network)                                            │
│                     GAE computes     → Clip the update       │
│                     advantages:        so we don't           │
│                     "was this action   change too much       │
│                     better than                              │
│                     average?"        4. REPEAT               │
│                                        Clear buffer,         │
│                                        collect again         │
└──────────────────────────────────────────────────────────────┘
```
 
**The Actor** is a neural network that decides what to do. Given a state, it outputs the mean of a probability distribution over actions. We sample from that distribution to get an action (TODO 2), and later re-evaluate stored actions under the updated policy (TODO 3). The actor is provided complete, you don't implement it, but feel free to modify it.
 
 
**The Critic** is a separate neural network that estimates "how good is this state?" (the value function V(s)). It doesn't pick actions — it just scores situations. The critic is provided complete, you don't implement it, but feel free to modify it.
 
**The Rollout Buffer** stores everything that happens during one round of data collection: observations, actions, rewards, done flags, log-probabilities, and value estimates. After collecting a full rollout, we use GAE (TODO 1) to compute advantages — "was this action better or worse than what we expected?"
 
**The PPO Loss** (TODO 4) is the key innovation. It computes how much the policy changed since the data was collected (the ratio), multiplies by the advantage, and clips the result. The clipping prevents the policy from changing too much in one update, which keeps training stable.
 
**The Training Loop** (TODO 5) wires everything together: collect data → compute advantages → run multiple epochs of gradient updates on the collected data → clear the buffer → repeat.
 
### How the TODOs connect
 
```
TODO 1 (GAE)                    Computes advantages from rewards and values
    ↓                           stored in the buffer
TODO 2 (sample_action)          Used during data collection: actor picks
    ↓                           actions, we store them with their log-probs
TODO 3 (evaluate_actions)       Used during the update: re-score the stored
    ↓                           actions under the CURRENT (updated) policy
TODO 4 (ppo_loss)               Compare old vs new log-probs → ratio → clip
    ↓
TODO 5 (training loop)          Orchestrates collection → GAE → update → repeat
```
 
---

## Tasks

### 1. Implement PPO — `workshop-1/1-ppo/ppo/`

Five `# TODO N — ...` blocks across two files:

- `rollout_buffer.py` — TODO 1 (Generalized Advantage Estimation)
- `ppo.py` — TODOs 2–5 (sample action, evaluate actions, PPO loss, training loop)

Each TODO has a heavy-hint docstring with the math, numbered steps, and
gotchas. Read it before you write anything. Each TODO has a matching
test you can run to verify your implementation.

#### TODO 1 — Generalized Advantage Estimation
**File:** `rollout_buffer.py` → `compute_gae()`
 
Compute how much better (or worse) each action was compared to what the
critic expected. GAE is a weighted average between two extremes:
1-step TD (low variance, high bias) and Monte Carlo returns (no bias,
high variance). The λ parameter (typically 0.95) controls the trade-off.
 
You implement a backwards loop over the rollout, accumulating discounted
TD-errors. The `(1 - done)` term cuts the bootstrap at episode boundaries.
 
**Key formula:**
```
δ_t = r_t + γ · V(s_{t+1}) · (1 - done_t) - V(s_t)
A_t = δ_t + (γλ) · (1 - done_t) · A_{t+1}
```
 
```bash
uv run python workshop-1/1-ppo/ppo/tests/test_ppo.py --step 1
```
 
#### TODO 2 — Sample an action from the policy
**File:** `ppo.py` → `sample_action()`
 
Given an observation, the actor outputs a mean, and we build a Normal
distribution using a learnable log_std parameter. We sample an action
from that distribution, compute its log-probability, and clip the action
to the environment's valid range.
 
Important: log-prob is computed on the **unclipped** sample. Clipping
only the returned action keeps gradients honest.
 
```bash
uv run python workshop-1/1-ppo/ppo/tests/test_ppo.py --step 2
```
 
#### TODO 3 — Evaluate actions under the current policy
**File:** `ppo.py` → `evaluate_actions()`
 
During the update phase, we re-score actions that were collected earlier
under the **current** (updated) policy. This gives us new log-probs and
entropy. The ratio between new and old log-probs is what PPO uses to
measure how much the policy changed.
 
This is almost identical to TODO 2, but without sampling — we evaluate
given actions instead of sampling new ones.
 
```bash
uv run python workshop-1/1-ppo/ppo/tests/test_ppo.py --step 3
```
 
#### TODO 4 — PPO clipped surrogate loss
**File:** `ppo.py` → `ppo_loss()`
 
The core of PPO. Compute the probability ratio (how much more or less
likely is this action under the new policy vs the old one), multiply by
the advantage, and clip to prevent too-large updates.
 
**Key formula:**
```
ratio = exp(log π_new - log π_old)
surr1 = ratio × advantage
surr2 = clip(ratio, 1-ε, 1+ε) × advantage
loss  = -mean(min(surr1, surr2))
```
 
The `min` is the conservative (pessimistic) choice — PPO takes whichever
is worse for the policy, preventing overconfident updates.
 
```bash
uv run python workshop-1/1-ppo/ppo/tests/test_ppo.py --step 4
```
 
#### TODO 5 — PPO training loop (5a, 5b, 5c)
**File:** `ppo.py` → `train()`
 
Most of the loop is provided. You fill in three small spots:
 
- **5a:** Call `buffer.compute_returns_and_advantages()` with the bootstrap value, gamma, and gae_lambda. This runs your TODO 1 implementation.
- **5b:** For each minibatch, call `evaluate_actions` (TODO 3), compute the policy loss with `ppo_loss` (TODO 4), compute the value loss with `F.mse_loss`, and combine into one scalar: `loss = p_loss + value_coef * v_loss - entropy_coef * entropy.mean()`.
- **5c:** Reset the buffer after the update phase so the next rollout starts clean.
```bash
uv run python workshop-1/1-ppo/ppo/tests/test_ppo.py --step 5
```

### 2. Pendulum — `workshop-1/2-pendulum/`

Train an agent on `Pendulum-v1`. Two paths:

- Custom PPO (uses your TODO implementations from stage 1)
- Stable-Baselines3 (drop-in alternative if you didn't finish the TODOs)

Output lands under `runs/pendulum/<run-name>/` with `model.pt` /
`model.zip`, a JSONL metrics log, and `eval.mp4`.

### 3. CarRacing — `workshop-1/3-car-racing/`

Train an agent on `CarRacing-v3`. Three paths:

- Custom PPO (`train.py`) — uses your stage-1 implementation with a CNN
  policy auto-detected from the observation shape.
- Stable-Baselines3 from scratch (`train_sb3.py`) — the same training
  budget, with `PPO("CnnPolicy", ...)`.
- SB3 fine-tune from HuggingFace (`train_sb3.py --from-hub`) — start
  from the published `sb3/ppo-CarRacing-v0` checkpoint and continue
  training. Fastest path to a driving agent.

**A note on expectations:** CarRacing is significantly harder than
Pendulum. The official SB3 baseline was trained for 4 million timesteps
with 8 parallel environments — that's hours of compute, not minutes.
Within the workshop budget, your agent will learn to
steer and stay on the road some of the time, but it won't drive clean
laps. Starting from the HuggingFace checkpoint gets you further faster,
but even fine-tuning won't produce a perfect driver in one session. The
goal here is to see your CNN policy learning from pixels and watch it
improve — not to solve the environment. If you want to push for a
high-scoring agent, keep training after the workshop with more timesteps
and experiment with hyperparameters and architectures.

---

## Resources
 
If you want to understand the theory deeper, these are the best references:
 
- [Spinning Up — Intro to RL](https://spinningup.openai.com/en/latest/spinningup/rl_intro.html) — clearest intro to RL concepts (agent, environment, policy, value)
- [Spinning Up — PPO](https://spinningup.openai.com/en/latest/algorithms/ppo.html) — concise PPO explanation with pseudocode
- [HuggingFace Deep RL Course — PPO](https://huggingface.co/learn/deep-rl-course/unit8/introduction) — visual explanation of the clipping mechanism
- [Lilian Weng — Policy Gradient Algorithms](https://lilianweng.github.io/posts/2018-04-08-policy-gradient/) — full journey from REINFORCE to PPO with math
- [The 37 Implementation Details of PPO](https://iclr-blog-track.github.io/2022/03/25/ppo-implementation-details/) — every implementation detail that matters, with ablations
- [Stable-Baselines3 PPO source](https://github.com/DLR-RM/stable-baselines3/blob/master/stable_baselines3/ppo/ppo.py) — reference implementation to compare against
---

## CLI cheat sheet

Every command below assumes you're in the repo root.

### Test your PPO implementation

```bash
# One step at a time (recommended):
uv run python workshop-1/1-ppo/ppo/tests/test_ppo.py --step 1
uv run python workshop-1/1-ppo/ppo/tests/test_ppo.py --step 2
uv run python workshop-1/1-ppo/ppo/tests/test_ppo.py --step 3
uv run python workshop-1/1-ppo/ppo/tests/test_ppo.py --step 4
uv run python workshop-1/1-ppo/ppo/tests/test_ppo.py --step 5

# All five:
uv run python workshop-1/1-ppo/ppo/tests/test_ppo.py
```

### Train custom PPO (stage 2 — Pendulum)

```bash
# Default: 200k timesteps, run name = timestamp:
uv run python workshop-1/2-pendulum/train.py

# Smaller / named run:
uv run python workshop-1/2-pendulum/train.py --timesteps 50000 --run-name my-pendulum --force # `--force` to overwrite an existing run name
```

### Train custom PPO (stage 3 — CarRacing)

```bash
uv run python workshop-1/3-car-racing/train.py --timesteps 200000 --run-name my-carracing --force
```

### Train SB3 from scratch — Pendulum

```bash
uv run python workshop-1/2-pendulum/train_sb3.py --timesteps 200000 --run-name my-pendulum-sb3 --force
```

### Train SB3 from scratch — CarRacing

```bash
uv run python workshop-1/3-car-racing/train_sb3.py --timesteps 200000 --run-name my-carracing-sb3 --force
```

### Fine-tune SB3 from a HuggingFace checkpoint

```bash
# Pendulum — load sb3/ppo-Pendulum-v1 and continue training:
uv run python workshop-1/2-pendulum/train_sb3.py --from-hub --timesteps 50000 --run-name pendulum-finetune --force

# CarRacing — load sb3/ppo-CarRacing-v0 and continue training:
uv run python workshop-1/3-car-racing/train_sb3.py --from-hub --timesteps 50000 --run-name carracing-finetune --force
```

The CarRacing checkpoint expects a specific observation shape
(64×64 grayscale, frame skip 2, frame stack 2) — the driver picks the
matching `EnvConfig.hub()` preset automatically when `--from-hub` is set.

### Evaluate without training

```bash
# Run a greedy episode against an existing checkpoint and save the video:
uv run python workshop-1/3-car-racing/train_sb3.py --from-hub --eval-only --run-name carracing-eval --force
```

Use `--eval-only` together with `--from-hub` to skip training entirely
and just record an evaluation video — great for showing the result of
a fine-tune run, or to demo the SB3 baseline before you start training
your own.

### Force re-running an existing run name

```bash
# `--force` overwrites runs/<stage>/<run-name>/ if it already exists:
uv run python workshop-1/3-car-racing/train_sb3.py --run-name my-run --timesteps 1000 --force
```

### Pick a device

The training drivers auto-select `cuda` → `mps` → `cpu`. Override with
`RL_WORKSHOP_DEVICE`:

```bash
RL_WORKSHOP_DEVICE=cpu  uv run python workshop-1/3-car-racing/train.py ...
RL_WORKSHOP_DEVICE=mps  uv run python workshop-1/3-car-racing/train.py ...
RL_WORKSHOP_DEVICE=cuda uv run python workshop-1/3-car-racing/train.py ...
```

For CarRacing's CNN, MPS is roughly 2× CPU on M-series Macs — the
default `auto` is what you want. For tiny models (Pendulum), CPU is
faster; the Pendulum driver pins `device="cpu"` in source.

## Hyperparameter reference

These are the defaults used by the custom-PPO drivers. Tune at your own
risk — the values were picked to converge inside the workshop budget.

### Pendulum (`workshop-1/2-pendulum/train.py`)

| Param            | Value      |
|------------------|------------|
| `rollout_size`   | 2048       |
| `n_epochs`       | 10         |
| `batch_size`     | 64         |
| `lr`             | 1e-3       |
| `gamma`          | 0.98       |
| `gae_lambda`     | 0.95       |
| `clip_eps`       | 0.2        |
| `value_coef`     | 0.5        |
| `entropy_coef`   | 0.0        |
| `max_grad_norm`  | 0.5        |
| `log_std_init`   | 0.0        |

### CarRacing (`workshop-1/3-car-racing/train.py`)

| Param            | Value      | Notes                                                |
|------------------|------------|------------------------------------------------------|
| `rollout_size`   | 2048       | Total across all envs (4 envs × 512 steps each)      |
| `n_epochs`       | 4          | Fewer than Pendulum — CNN updates are heavier        |
| `batch_size`     | 64         | Minibatch size                                       |
| `lr`             | 3e-4       | Standard PPO                                         |
| `gamma`          | 0.99       | Longer horizons for driving                          |
| `gae_lambda`     | 0.95       |                                                      |
| `clip_eps`       | 0.2        |                                                      |
| `value_coef`     | 0.5        |                                                      |
| `entropy_coef`   | 0.01       |                                                      |
| `max_grad_norm`  | 0.5        |                                                      |
| `log_std_init`   | -0.5       | Tighter exploration (gas/brake are asymmetric)       |

`train_sb3.py` uses Stable-Baselines3's `PPO("MlpPolicy")` /
`PPO("CnnPolicy")` defaults.

## If you get stuck

You have two ways forward without help.

### 1. Switch to Stable-Baselines3

If you can't get a TODO to pass, the SB3 driver is a complete drop-in
replacement. You skip stage 1, but you still finish stages 2 and 3 and
end the workshop with a trained agent:

```bash
uv run python workshop-1/2-pendulum/train_sb3.py --timesteps 50000 --run-name pendulum-sb3 --force
uv run python workshop-1/3-car-racing/train_sb3.py --from-hub --eval-only --run-name carracing-hub --force
```

### 2. Peek at the `solutions` branch

The `solutions` branch contains the full reference implementation. Two
ways to use it:

```bash
# Browse on GitHub — no checkout needed (open the file in your browser): https://github.com/MortenBlorstad/rl-sim-to-real-workshop/tree/solutions

# Or grab one file locally without switching branches:
git checkout solutions -- workshop-1/1-ppo/ppo/ppo.py
git checkout solutions -- workshop-1/1-ppo/ppo/rollout_buffer.py
```

The path-scoped checkout overwrites your local copy of that file —
commit any work you want to keep first.
