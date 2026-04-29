import numpy as np
import torch
from typing import Iterator


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

    # ===========================================================================
    # TODO 1 — Generalized Advantage Estimation
    # ===========================================================================
    def compute_gae(self, rewards: torch.Tensor, values: torch.Tensor, dones: torch.Tensor, gamma: float, lam: float) -> torch.Tensor:
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
        
        #TODO: implement GAE and return the advantages tensor. This will be called from ``compute_returns_and_advantages`` once you fill in the TODO there.
        # raise NotImplementedError("TODO 1: implement GAE in RolloutBuffer.compute_gae")
        
        # -- Solution code --
        T = rewards.shape[0]
        advantages = torch.zeros(T, dtype=rewards.dtype)
        gae = 0.0
        for t in reversed(range(T)):
            not_done = 1.0 - float(dones[t])
            delta = rewards[t] + gamma * values[t + 1] * not_done - values[t]
            gae = float(delta) + gamma * lam * not_done * gae
            advantages[t] = gae
        return advantages
        # -- END Solution code --

    def compute_returns_and_advantages(
        self, last_value: float, gamma: float, gae_lambda: float
    ) -> None:
        rewards_t = torch.from_numpy(self.rewards)
        dones_t = torch.from_numpy(self.dones)
        values_with_bootstrap = np.concatenate(
            [self.values, np.array([last_value], dtype=np.float32)]
        )
        values_t = torch.from_numpy(values_with_bootstrap)
        advantages_t = self.compute_gae(rewards_t, values_t, dones_t, gamma=gamma, lam=gae_lambda)
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