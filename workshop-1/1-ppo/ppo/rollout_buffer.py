import numpy as np
import torch
from typing import Iterator


class RolloutBuffer:
    """Pre-allocated rollout storage for a single PPO update.

    Stores ``size`` transitions per env, across ``num_envs`` parallel envs.
    Total transitions per update is ``size * num_envs``.

    Usage:
        buf = RolloutBuffer(size=256, num_envs=4, obs_dim=3, action_dim=1)
        for t in range(size):
            # all inputs have leading dim num_envs
            buf.add(obs, action, log_prob, reward, done, value)
        buf.compute_returns_and_advantages(last_value, gamma, gae_lambda)
        for batch in buf.get_batches(batch_size):
            ...  # SGD update; batches are flattened across (size, num_envs)
        buf.reset()

    ``compute_returns_and_advantages`` calls ``compute_gae`` internally,
    so this method will raise ``NotImplementedError`` until TODO 1 is done.
    """

    def __init__(
        self,
        size: int,
        obs_dim: int,
        action_dim: int,
        num_envs: int = 1,
        obs_shape: tuple | None = None,
    ):
        self.size = size
        self.num_envs = num_envs
        self.total = size * num_envs
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        # When obs_shape is supplied (e.g. (4, 84, 84) for CNN), store obs
        # with that full shape so the update phase can reshape batches back
        # to (B, *obs_shape) for the conv forward. When omitted (MLP), fall
        # back to the flat (size, num_envs, obs_dim) layout used by Pendulum.
        self.obs_shape = obs_shape if obs_shape is not None else (obs_dim,)
        self.obs = np.zeros((size, num_envs, *self.obs_shape), dtype=np.float32)
        self.actions = np.zeros((size, num_envs, action_dim), dtype=np.float32)
        self.log_probs = np.zeros((size, num_envs), dtype=np.float32)
        self.rewards = np.zeros((size, num_envs), dtype=np.float32)
        self.dones = np.zeros((size, num_envs), dtype=np.float32)
        self.values = np.zeros((size, num_envs), dtype=np.float32)
        self.advantages = np.zeros((size, num_envs), dtype=np.float32)
        self.returns = np.zeros((size, num_envs), dtype=np.float32)
        self.idx = 0

    def add(self, obs, action, log_prob, reward, done, value) -> None:
        """Append one timestep across all envs.

        All inputs are arrays/sequences with leading dim ``num_envs``. Scalars
        are broadcast (used by single-env tests).
        """
        i = self.idx
        self.obs[i] = obs
        self.actions[i] = action
        self.log_probs[i] = log_prob
        self.rewards[i] = reward
        self.dones[i] = done
        self.values[i] = value
        self.idx += 1

    # ===========================================================================
    # TODO 1 — Generalized Advantage Estimation
    # ===========================================================================
    def compute_gae(self, rewards: torch.Tensor, values: torch.Tensor, dones: torch.Tensor, gamma: float, lam: float) -> torch.Tensor:
        """Compute Generalized Advantage Estimation (GAE) for a trajectory.

        Args:
            rewards: shape ``(T,)`` or ``(T, N)``, reward at each step.
            values:  shape ``(T+1,)`` or ``(T+1, N)``, state values including
                     the bootstrapped final value at index ``T``.
            dones:   shape ``(T,)`` or ``(T, N)``, ``1.0`` if the episode
                     terminated AFTER step ``t``, else ``0.0``.
            gamma:   discount factor.
            lam:     GAE lambda.

        Returns:
            ``advantages`` of the same shape as ``rewards``.

        Hint — the recurrence:
            delta_t = rewards[t] + gamma * values[t+1] * (1 - dones[t]) - values[t]
            gae_t   = delta_t + gamma * lam * (1 - dones[t]) * gae_{t+1}
            with gae_T = 0.

        Iterate from ``t = T-1`` down to ``t = 0``. The ``(1 - dones[t])`` term
        is what cuts the bootstrap at episode boundaries.
        """

        # -- Solution code --
        T = rewards.shape[0]
        advantages = torch.zeros_like(rewards)
        gae = torch.zeros_like(rewards[0])
        for t in reversed(range(T)):
            not_done = 1.0 - dones[t]
            delta = rewards[t] + gamma * values[t + 1] * not_done - values[t]
            gae = delta + gamma * lam * not_done * gae
            advantages[t] = gae
        return advantages
        # -- END Solution code --

    def compute_returns_and_advantages(
        self, last_value, gamma: float, gae_lambda: float
    ) -> None:
        """Compute returns and advantages from collected rewards/values.

        ``last_value`` is the bootstrap value V(s_{T+1}) for each env; pass
        ``(num_envs,)`` array (or a scalar that broadcasts).
        """
        rewards_t = torch.from_numpy(self.rewards)  # (T, N)
        dones_t = torch.from_numpy(self.dones)  # (T, N)
        last_value_arr = np.asarray(last_value, dtype=np.float32).reshape(self.num_envs)
        values_with_bootstrap = np.concatenate(
            [self.values, last_value_arr[None, :]], axis=0
        )  # (T+1, N)
        values_t = torch.from_numpy(values_with_bootstrap)
        advantages_t = self.compute_gae(rewards_t, values_t, dones_t, gamma=gamma, lam=gae_lambda)
        self.advantages = advantages_t.detach().cpu().numpy().astype(np.float32)
        self.returns = (self.advantages + self.values).astype(np.float32)

    def get_batches(self, batch_size: int) -> Iterator[dict]:
        """Yield shuffled minibatches flattened across (size, num_envs)."""
        total = self.total
        obs_flat = self.obs.reshape(total, *self.obs_shape)
        actions_flat = self.actions.reshape(total, self.action_dim)
        log_probs_flat = self.log_probs.reshape(total)
        adv_flat = self.advantages.reshape(total)
        returns_flat = self.returns.reshape(total)

        adv_normalised = (adv_flat - adv_flat.mean()) / (adv_flat.std() + 1e-8)

        indices = np.random.permutation(total)
        for start in range(0, total, batch_size):
            batch_idx = indices[start : start + batch_size]
            yield {
                "obs": torch.from_numpy(obs_flat[batch_idx]),
                "actions": torch.from_numpy(actions_flat[batch_idx]),
                "old_log_probs": torch.from_numpy(log_probs_flat[batch_idx]),
                "advantages": torch.from_numpy(adv_normalised[batch_idx]),
                "returns": torch.from_numpy(returns_flat[batch_idx]),
            }

    def reset(self) -> None:
        self.idx = 0
