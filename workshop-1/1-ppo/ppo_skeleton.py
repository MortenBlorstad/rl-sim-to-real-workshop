"""PPO skeleton — Workshop 1, Stage 1.

A single-file PPO scaffold for the RL Sim-to-Real Workshop. Five TODO
blocks for participants to fill in: GAE, sample_action, evaluate_actions,
ppo_loss, training loop. Helper code (Actor / Value networks, rollout
buffer, PPOAgent class) is provided complete.

Run as a script to see the training loop execute on
MountainCarContinuous-v0:

    uv run python workshop-1/1-ppo/ppo_skeleton.py

Run the per-step tests via:

    uv run python workshop-1/1-ppo/test_ppo.py --step 1

Spec: specs/001-ppo-skeleton/spec.md
"""
from __future__ import annotations

import argparse
import math
import random
import sys
from typing import Iterator

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal


DEFAULT_SEED = 42


# ---------------------------------------------------------------------------
# Agent class registry
# ---------------------------------------------------------------------------
# Used by PPOAgent.save / .load to round-trip subclasses (e.g.
# MountainCarPPOAgent, CarRacingPPOAgent) without the base class needing to
# know about them. Subclasses must decorate themselves with @register_agent.

_AGENT_REGISTRY: dict[str, type] = {}


def register_agent(cls):
    """Class decorator: add ``cls`` to the agent registry under its name."""
    _AGENT_REGISTRY[cls.__name__] = cls
    return cls


# ---------------------------------------------------------------------------
# Helper — ActorNetwork (provided complete)
# ---------------------------------------------------------------------------


class ActorNetwork(nn.Module):
    """2x64 Tanh MLP. Outputs the mean of a Normal action distribution.

    The ``log_std`` is a separate ``nn.Parameter`` owned by the caller (or by
    PPOAgent), not a network output. See the workshop notes for why.
    """

    def __init__(self, obs_dim: int, action_dim: int, hidden: int = 64):
        super().__init__()
        self.fc1 = nn.Linear(obs_dim, hidden)
        self.fc2 = nn.Linear(hidden, hidden)
        self.head = nn.Linear(hidden, action_dim)
        self._orthogonal_init()

    def _orthogonal_init(self) -> None:
        for layer in (self.fc1, self.fc2):
            nn.init.orthogonal_(layer.weight, gain=math.sqrt(2))
            nn.init.zeros_(layer.bias)
        nn.init.orthogonal_(self.head.weight, gain=0.01)
        nn.init.zeros_(self.head.bias)

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        x = torch.tanh(self.fc1(obs))
        x = torch.tanh(self.fc2(x))
        return self.head(x)


# ---------------------------------------------------------------------------
# Helper — ValueNetwork (provided complete)
# ---------------------------------------------------------------------------


class ValueNetwork(nn.Module):
    """2x64 Tanh MLP. Outputs a scalar state value. No parameter sharing
    with ``ActorNetwork``.
    """

    def __init__(self, obs_dim: int, hidden: int = 64):
        super().__init__()
        self.fc1 = nn.Linear(obs_dim, hidden)
        self.fc2 = nn.Linear(hidden, hidden)
        self.head = nn.Linear(hidden, 1)
        self._orthogonal_init()

    def _orthogonal_init(self) -> None:
        for layer in (self.fc1, self.fc2):
            nn.init.orthogonal_(layer.weight, gain=math.sqrt(2))
            nn.init.zeros_(layer.bias)
        nn.init.orthogonal_(self.head.weight, gain=1.0)
        nn.init.zeros_(self.head.bias)

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        x = torch.tanh(self.fc1(obs))
        x = torch.tanh(self.fc2(x))
        return self.head(x).squeeze(-1)


# ---------------------------------------------------------------------------
# Helper — RolloutBuffer (provided complete)
# ---------------------------------------------------------------------------


class RolloutBuffer:
    """Pre-allocated rollout storage for a single PPO update.

    Usage:
        buf = RolloutBuffer(size, obs_dim, action_dim)
        for t in range(size):
            buf.add(obs, action, log_prob, reward, done, value)
        buf.compute_returns_and_advantages(last_value, gamma, gae_lambda)
        for batch in buf.get_batches(batch_size):
            ...  # SGD update
        buf.reset()

    ``compute_returns_and_advantages`` calls ``compute_gae`` internally,
    so this method will raise ``NotImplementedError`` until TODO 1 is done.
    """

    def __init__(self, size: int, obs_dim: int, action_dim: int):
        self.size = size
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.obs = np.zeros((size, obs_dim), dtype=np.float32)
        self.actions = np.zeros((size, action_dim), dtype=np.float32)
        self.log_probs = np.zeros(size, dtype=np.float32)
        self.rewards = np.zeros(size, dtype=np.float32)
        self.dones = np.zeros(size, dtype=np.float32)
        self.values = np.zeros(size, dtype=np.float32)
        self.advantages = np.zeros(size, dtype=np.float32)
        self.returns = np.zeros(size, dtype=np.float32)
        self.idx = 0

    def add(self, obs, action, log_prob, reward, done, value) -> None:
        i = self.idx
        self.obs[i] = obs
        self.actions[i] = action
        self.log_probs[i] = float(log_prob)
        self.rewards[i] = float(reward)
        self.dones[i] = float(done)
        self.values[i] = float(value)
        self.idx += 1

    def compute_returns_and_advantages(
        self, last_value: float, gamma: float, gae_lambda: float
    ) -> None:
        rewards_t = torch.from_numpy(self.rewards)
        dones_t = torch.from_numpy(self.dones)
        values_with_bootstrap = np.concatenate(
            [self.values, np.array([last_value], dtype=np.float32)]
        )
        values_t = torch.from_numpy(values_with_bootstrap)
        advantages_t = compute_gae(rewards_t, values_t, dones_t, gamma=gamma, lam=gae_lambda)
        self.advantages = advantages_t.detach().cpu().numpy().astype(np.float32)
        self.returns = (self.advantages + self.values).astype(np.float32)

    def get_batches(self, batch_size: int) -> Iterator[dict]:
        indices = np.random.permutation(self.size)
        adv = self.advantages.copy()
        adv = (adv - adv.mean()) / (adv.std() + 1e-8)
        for start in range(0, self.size, batch_size):
            batch_idx = indices[start : start + batch_size]
            yield {
                "obs": torch.from_numpy(self.obs[batch_idx]),
                "actions": torch.from_numpy(self.actions[batch_idx]),
                "old_log_probs": torch.from_numpy(self.log_probs[batch_idx]),
                "advantages": torch.from_numpy(adv[batch_idx]),
                "returns": torch.from_numpy(self.returns[batch_idx]),
            }

    def reset(self) -> None:
        self.idx = 0


# ---------------------------------------------------------------------------
# Helper — log line formatter (provided complete)
# ---------------------------------------------------------------------------


def format_update_line(
    update_idx: int,
    n_updates: int,
    timesteps: int,
    policy_loss: float,
    value_loss: float,
    entropy: float,
    mean_return: float,
) -> str:
    """Format one fixed-width log line for the training loop.

    Use this inside your TODO 5 ``train()`` implementation so the script-mode
    runner can parse the loss values for its exit checks.
    """
    return (
        f"[update {update_idx:2d}/{n_updates}] "
        f"timesteps={timesteps:6d}  "
        f"policy_loss={policy_loss:+.3f}  "
        f"value_loss={value_loss:+.3f}  "
        f"entropy={entropy:+.3f}  "
        f"mean_return={mean_return:+.2f}"
    )


# ===========================================================================
# TODO 1 — Generalized Advantage Estimation
# ===========================================================================


def compute_gae(
    rewards: torch.Tensor,
    values: torch.Tensor,
    dones: torch.Tensor,
    gamma: float = 0.99,
    lam: float = 0.95,
) -> torch.Tensor:
    """Compute Generalized Advantage Estimation (GAE) for a trajectory.

    Args:
        rewards: shape ``(T,)``, reward at each step.
        values:  shape ``(T+1,)``, state values including the bootstrapped
                 final value at index ``T``.
        dones:   shape ``(T,)``, ``1.0`` if the episode terminated AFTER step
                 ``t``, else ``0.0``.
        gamma:   discount factor.
        lam:     GAE lambda.

    Returns:
        ``advantages`` of shape ``(T,)`` and same dtype as ``rewards``.

    Hint — the recurrence:
        delta_t = rewards[t] + gamma * values[t+1] * (1 - dones[t]) - values[t]
        gae_t   = delta_t + gamma * lam * (1 - dones[t]) * gae_{t+1}
        with gae_T = 0.

    Iterate from ``t = T-1`` down to ``t = 0``. The ``(1 - dones[t])`` term
    is what cuts the bootstrap at episode boundaries.
    """
    # -- YOUR CODE HERE --
    T = rewards.shape[0]
    advantages = torch.zeros(T, dtype=rewards.dtype)
    gae = 0.0
    for t in reversed(range(T)):
        not_done = 1.0 - float(dones[t])
        delta = rewards[t] + gamma * values[t + 1] * not_done - values[t]
        gae = float(delta) + gamma * lam * not_done * gae
        advantages[t] = gae
    return advantages
    # -- END YOUR CODE --


# ===========================================================================
# TODO 2 — Sample an action from the policy
# ===========================================================================


def sample_action(
    actor: ActorNetwork,
    obs: torch.Tensor,
    log_std: torch.Tensor,
    deterministic: bool = False,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Sample an action from the policy and return ``(action, log_prob)``.

    Args:
        actor:         ActorNetwork that maps obs -> action mean.
        obs:           shape ``(obs_dim,)`` for a single observation, or
                       ``(B, obs_dim)`` batched.
        log_std:       shape ``(action_dim,)``, learnable log std-dev.
        deterministic: if True, return the policy mean (no sampling).

    Returns:
        action:   clipped to ``[-1, 1]``. Shape matches obs's leading dim.
        log_prob: log-prob of the UNCLIPPED sample, summed over action dims.

    Hints:
        1. mean = actor(obs)
        2. dist = torch.distributions.Normal(mean, log_std.exp())
        3. unclipped = mean if deterministic else dist.sample()
        4. log_prob  = dist.log_prob(unclipped).sum(dim=-1)
        5. action    = unclipped.clamp(-1.0, 1.0)

    Important: log_prob is computed on the UNCLIPPED sample. Clipping
    only the returned action keeps the gradient honest.
    """
    # -- YOUR CODE HERE --
    mean = actor(obs)
    dist = Normal(mean, log_std.exp())
    if deterministic:
        unclipped = mean
    else:
        unclipped = dist.sample()
    log_prob = dist.log_prob(unclipped).sum(dim=-1)
    action = unclipped.clamp(-1.0, 1.0)
    return action, log_prob
    # -- END YOUR CODE --


# ===========================================================================
# TODO 3 — Evaluate actions under the current policy
# ===========================================================================


def evaluate_actions(
    actor: ActorNetwork,
    obs: torch.Tensor,
    actions: torch.Tensor,
    log_std: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Evaluate a batch of ``(obs, action)`` under the current policy.

    Args:
        actor:    ActorNetwork.
        obs:      shape ``(B, obs_dim)``.
        actions:  shape ``(B, action_dim)``.
        log_std:  shape ``(action_dim,)``.

    Returns:
        ``(log_probs, entropy)``, both of shape ``(B,)``.

    Hints:
        1. mean      = actor(obs)
        2. dist      = Normal(mean, log_std.exp())
        3. log_probs = dist.log_prob(actions).sum(dim=-1)
        4. entropy   = dist.entropy().sum(dim=-1)
    """
    # -- YOUR CODE HERE --
    mean = actor(obs)
    dist = Normal(mean, log_std.exp())
    log_probs = dist.log_prob(actions).sum(dim=-1)
    entropy = dist.entropy().sum(dim=-1)
    return log_probs, entropy
    # -- END YOUR CODE --


# ===========================================================================
# TODO 4 — PPO clipped surrogate loss
# ===========================================================================


def ppo_loss(
    new_log_probs: torch.Tensor,
    old_log_probs: torch.Tensor,
    advantages: torch.Tensor,
    clip_eps: float = 0.2,
) -> torch.Tensor:
    """PPO clipped surrogate objective. Returns the POLICY loss only.

    The training loop combines this with the value loss (MSE) and an
    entropy bonus.

    Args:
        new_log_probs: shape ``(B,)``, log-probs under the CURRENT policy.
                       Carries gradient.
        old_log_probs: shape ``(B,)``, log-probs at rollout time. No grad.
        advantages:    shape ``(B,)``, normalized to mean 0, std 1.
        clip_eps:      clip range epsilon (typically 0.2).

    Returns:
        Scalar tensor: ``-mean(min(ratio * adv, clip(ratio, 1-eps, 1+eps) * adv))``.

    Hints:
        ratio = (new_log_probs - old_log_probs).exp()
        surr1 = ratio * advantages
        surr2 = ratio.clamp(1 - clip_eps, 1 + clip_eps) * advantages
        loss  = -torch.min(surr1, surr2).mean()
    """
    # -- YOUR CODE HERE --
    ratio = (new_log_probs - old_log_probs).exp()
    surr1 = ratio * advantages
    surr2 = ratio.clamp(1.0 - clip_eps, 1.0 + clip_eps) * advantages
    return -torch.min(surr1, surr2).mean()
    # -- END YOUR CODE --


# ===========================================================================
# TODO 5 — Full PPO training loop
# ===========================================================================


def train(
    env,
    actor: ActorNetwork,
    value: ValueNetwork,
    log_std: nn.Parameter,
    total_timesteps: int = 8192,
    rollout_size: int = 1024,
    n_epochs: int = 4,
    batch_size: int = 64,
    lr: float = 3e-4,
    gamma: float = 0.99,
    gae_lambda: float = 0.95,
    clip_eps: float = 0.2,
    value_coef: float = 0.5,
    entropy_coef: float = 0.01,
    max_grad_norm: float = 0.5,
    seed: int = DEFAULT_SEED,
    log_fn=print,
) -> dict:
    """Wire everything together into a working PPO training loop.

    Sketch:

    1. Re-seed (random / np / torch / env) with ``seed``.
    2. Build an ``Adam`` optimizer over
       ``list(actor.parameters()) + list(value.parameters()) + [log_std]``.
    3. Build a ``RolloutBuffer(rollout_size, obs_dim, action_dim)``.
    4. Loop until ``total_timesteps`` env steps have been collected:
        a. Roll out ``rollout_size`` transitions:
            - ``action, log_prob = sample_action(actor, obs_t, log_std)``
            - ``value_t = value(obs_t).item()``
            - ``next_obs, reward, terminated, truncated, _ = env.step(action.numpy())``
            - ``done = terminated or truncated``
            - ``buffer.add(obs, action.numpy(), log_prob.item(), reward, done, value_t)``
            - on done: ``next_obs, _ = env.reset()``; track episode return.
        b. Bootstrap: ``last_value = value(last_obs).item()`` (or ``0.0`` if
           the rollout ended on a terminal step).
        c. ``buffer.compute_returns_and_advantages(last_value, gamma, gae_lambda)``
        d. For ``n_epochs`` epochs, iterate ``buffer.get_batches(batch_size)``:
            - ``new_log_probs, ent = evaluate_actions(actor, batch["obs"], batch["actions"], log_std)``
            - ``pred_values = value(batch["obs"])``
            - ``p_loss = ppo_loss(new_log_probs, batch["old_log_probs"], batch["advantages"], clip_eps)``
            - ``v_loss = F.mse_loss(pred_values, batch["returns"])``
            - ``loss = p_loss + value_coef * v_loss - entropy_coef * ent.mean()``
            - ``optimizer.zero_grad(); loss.backward()``
            - ``torch.nn.utils.clip_grad_norm_(parameters, max_grad_norm)``
            - ``optimizer.step()``
        e. Print one line per iteration via:
           ``log_fn(format_update_line(update_idx, n_updates, timesteps, p_loss.item(), v_loss.item(), ent.mean().item(), mean_return))``
        f. ``buffer.reset()``

    Returns:
        Dict with at minimum: ``mean_reward``, ``policy_loss``,
        ``value_loss``, ``entropy``, ``n_updates``.

    Note:
        Use ``format_update_line(...)`` for the printed line so the
        ``__main__`` runner can parse loss values for its exit checks.
    """
    # -- YOUR CODE HERE --
    # Re-seed for reproducibility within a single train() call.
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    obs_dim = int(np.prod(env.observation_space.shape))
    action_dim = int(np.prod(env.action_space.shape))

    optimizer = torch.optim.Adam(
        list(actor.parameters()) + list(value.parameters()) + [log_std],
        lr=lr,
    )
    buffer = RolloutBuffer(rollout_size, obs_dim, action_dim)
    n_updates = max(1, total_timesteps // rollout_size)

    obs, _ = env.reset(seed=seed)
    episode_returns: list[float] = []
    current_return = 0.0
    timesteps = 0
    final_stats: dict = {}

    for update_idx in range(1, n_updates + 1):
        # ----- rollout collection -----
        last_step_done = False
        for _ in range(rollout_size):
            obs_t = torch.as_tensor(obs, dtype=torch.float32)
            with torch.no_grad():
                action, log_prob = sample_action(actor, obs_t, log_std)
                value_t = value(obs_t).item()
            action_np = action.detach().cpu().numpy().astype(np.float32)
            next_obs, reward, terminated, truncated, _ = env.step(action_np)
            done = bool(terminated or truncated)
            buffer.add(obs, action_np, log_prob.item(), reward, done, value_t)
            current_return += float(reward)
            timesteps += 1
            if done:
                episode_returns.append(current_return)
                current_return = 0.0
                obs, _ = env.reset()
                last_step_done = True
            else:
                obs = next_obs
                last_step_done = False

        # Bootstrap final value (zero if rollout ended on terminal step).
        if last_step_done:
            last_value = 0.0
        else:
            with torch.no_grad():
                last_value = value(torch.as_tensor(obs, dtype=torch.float32)).item()

        buffer.compute_returns_and_advantages(last_value, gamma, gae_lambda)

        # ----- update phase -----
        last_p_loss = 0.0
        last_v_loss = 0.0
        last_entropy = 0.0
        for _ in range(n_epochs):
            for batch in buffer.get_batches(batch_size):
                new_log_probs, ent = evaluate_actions(
                    actor, batch["obs"], batch["actions"], log_std,
                )
                pred_values = value(batch["obs"])
                p_loss = ppo_loss(
                    new_log_probs, batch["old_log_probs"], batch["advantages"], clip_eps,
                )
                v_loss = F.mse_loss(pred_values, batch["returns"])
                loss = p_loss + value_coef * v_loss - entropy_coef * ent.mean()

                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(
                    list(actor.parameters()) + list(value.parameters()) + [log_std],
                    max_grad_norm,
                )
                optimizer.step()

                last_p_loss = float(p_loss.item())
                last_v_loss = float(v_loss.item())
                last_entropy = float(ent.mean().item())

        mean_return = (
            float(np.mean(episode_returns[-10:])) if episode_returns else current_return
        )
        log_fn(
            format_update_line(
                update_idx, n_updates, timesteps,
                last_p_loss, last_v_loss, last_entropy, mean_return,
            )
        )
        buffer.reset()

        final_stats = {
            "mean_reward": mean_return,
            "policy_loss": last_p_loss,
            "value_loss": last_v_loss,
            "entropy": last_entropy,
            "n_updates": update_idx,
        }

    return final_stats
    # -- END YOUR CODE --


# ---------------------------------------------------------------------------
# PPOAgent — Constitution Article II contract (Path A)
# ---------------------------------------------------------------------------


@register_agent
class PPOAgent:
    """Custom-PPO Agent that implements the Article II Agent interface.

    Subclasses (e.g. ``MountainCarPPOAgent``, ``CarRacingPPOAgent``) should
    override ``preprocess()``, and optionally ``_get_preprocess_state`` /
    ``_set_preprocess_state`` if they have persistent preprocessing state
    that needs to round-trip through ``save`` / ``load``. Subclasses MUST
    decorate themselves with ``@register_agent`` so ``PPOAgent.load()`` can
    look them up by name.
    """

    DEFAULT_HYPERPARAMS: dict = {
        "rollout_size": 1024,
        "n_epochs": 4,
        "batch_size": 64,
        "lr": 3e-4,
        "gamma": 0.99,
        "gae_lambda": 0.95,
        "clip_eps": 0.2,
        "value_coef": 0.5,
        "entropy_coef": 0.01,
        "max_grad_norm": 0.5,
    }

    def __init__(
        self,
        obs_dim: int,
        action_dim: int,
        hyperparameters: dict | None = None,
    ):
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.hyperparameters: dict = dict(self.DEFAULT_HYPERPARAMS)
        if hyperparameters:
            self.hyperparameters.update(hyperparameters)
        self.actor = ActorNetwork(obs_dim, action_dim)
        self.value = ValueNetwork(obs_dim)
        self.log_std = nn.Parameter(torch.zeros(action_dim))

    # ------ Article II contract ------

    def preprocess(self, obs: np.ndarray) -> np.ndarray:
        """Identity preprocess in the base class. Subclasses may override."""
        return obs

    def predict(self, obs: np.ndarray, deterministic: bool = False) -> np.ndarray:
        """Take a raw observation, preprocess internally, return an action."""
        processed = self.preprocess(obs)
        obs_tensor = torch.as_tensor(np.asarray(processed, dtype=np.float32))
        with torch.no_grad():
            action, _ = sample_action(
                self.actor, obs_tensor, self.log_std, deterministic=deterministic
            )
        return action.cpu().numpy().astype(np.float32)

    def train(self, env, total_timesteps: int) -> dict:
        """Delegate to the module-level ``train()`` function."""
        return train(
            env,
            self.actor,
            self.value,
            self.log_std,
            total_timesteps=total_timesteps,
            **self.hyperparameters,
        )

    def save(self, path: str) -> None:
        """Persist model weights, hyperparameters, and preprocess state."""
        state = {
            "class_name": type(self).__name__,
            "obs_dim": self.obs_dim,
            "action_dim": self.action_dim,
            "actor_state_dict": self.actor.state_dict(),
            "value_state_dict": self.value.state_dict(),
            "log_std": self.log_std.detach().clone(),
            "hyperparameters": self.hyperparameters,
            "preprocess_state": self._get_preprocess_state(),
        }
        torch.save(state, path)

    @classmethod
    def load(cls, path: str) -> "PPOAgent":
        """Load any registered subclass by class name from a single ``.pt``."""
        state = torch.load(path, weights_only=False)
        class_name = state["class_name"]
        if class_name not in _AGENT_REGISTRY:
            raise ValueError(
                f"Unknown agent class {class_name!r}. Registered classes: "
                f"{sorted(_AGENT_REGISTRY)}. Make sure the subclass module "
                f"is imported before calling PPOAgent.load()."
            )
        target_cls = _AGENT_REGISTRY[class_name]
        agent = target_cls(
            obs_dim=state["obs_dim"],
            action_dim=state["action_dim"],
            hyperparameters=state["hyperparameters"],
        )
        agent.actor.load_state_dict(state["actor_state_dict"])
        agent.value.load_state_dict(state["value_state_dict"])
        with torch.no_grad():
            agent.log_std.copy_(state["log_std"])
        agent._set_preprocess_state(state.get("preprocess_state", {}))
        return agent

    # ------ Subclass extension hooks ------

    def _get_preprocess_state(self) -> dict:
        """Return any subclass preprocess state to persist with ``save()``."""
        return {}

    def _set_preprocess_state(self, state: dict) -> None:
        """Restore subclass preprocess state on ``load()``. Base class no-op."""
        return None


# ---------------------------------------------------------------------------
# Script entry point
# ---------------------------------------------------------------------------


def _seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def _main() -> int:
    parser = argparse.ArgumentParser(description="PPO skeleton training run.")
    parser.add_argument(
        "--timesteps",
        type=int,
        default=8192,
        help="Total environment steps for the training run.",
    )
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    args = parser.parse_args()

    _seed_everything(args.seed)
    env = gym.make("MountainCarContinuous-v0")
    env.reset(seed=args.seed)

    obs_dim = int(np.prod(env.observation_space.shape))
    action_dim = int(np.prod(env.action_space.shape))

    actor = ActorNetwork(obs_dim, action_dim)
    value = ValueNetwork(obs_dim)
    log_std = nn.Parameter(torch.zeros(action_dim))

    captured_policy_losses: list[float] = []
    captured_entropies: list[float] = []

    def _capturing_log(line: str) -> None:
        print(line)
        for token in line.split():
            if token.startswith("policy_loss="):
                try:
                    captured_policy_losses.append(float(token.split("=", 1)[1]))
                except ValueError:
                    pass
            elif token.startswith("entropy="):
                try:
                    captured_entropies.append(float(token.split("=", 1)[1]))
                except ValueError:
                    pass

    try:
        train(
            env,
            actor,
            value,
            log_std,
            total_timesteps=args.timesteps,
            seed=args.seed,
            log_fn=_capturing_log,
        )
    except NotImplementedError as exc:
        print(f"\nFAIL: TODO not yet implemented — {exc}", file=sys.stderr)
        return 1

    # FR-027 — exit-time invariants on the printed metrics.
    #
    # We check entropy (NOT policy_loss) for the trend. PPO's clipped
    # surrogate loss is not a supervised loss — it bounces around as the
    # policy improves and is not expected to decrease monotonically. The
    # canonical monotonic signal in PPO is policy entropy: as the agent
    # commits to better actions, entropy goes DOWN.
    if not captured_policy_losses:
        print(
            "FAIL: training loop printed no policy_loss lines. "
            "Did you call format_update_line(...) inside train()?",
            file=sys.stderr,
        )
        return 1
    if any(math.isnan(loss) for loss in captured_policy_losses):
        print(
            f"FAIL: at least one printed policy_loss is NaN "
            f"(saw {captured_policy_losses}).",
            file=sys.stderr,
        )
        return 1
    if not captured_entropies:
        print(
            "FAIL: training loop printed no entropy values. "
            "Did you call format_update_line(...) inside train()?",
            file=sys.stderr,
        )
        return 1
    if any(math.isnan(e) for e in captured_entropies):
        print(
            f"FAIL: at least one printed entropy is NaN "
            f"(saw {captured_entropies}).",
            file=sys.stderr,
        )
        return 1
    if captured_entropies[-1] >= captured_entropies[0]:
        print(
            f"FAIL: entropy did not trend down "
            f"(first={captured_entropies[0]:+.4f}, "
            f"last={captured_entropies[-1]:+.4f}). "
            f"Entropy should decrease as the policy commits to actions.",
            file=sys.stderr,
        )
        return 1

    print(
        f"\n✓ Training complete: entropy trending down "
        f"({captured_entropies[0]:+.4f} → {captured_entropies[-1]:+.4f}), "
        f"no NaN losses."
    )
    return 0


if __name__ == "__main__":
    sys.exit(_main())
