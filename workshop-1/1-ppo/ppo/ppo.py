from __future__ import annotations

import sys
from pathlib import Path

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal

from .networks import (
    ActorNetwork,
    CnnActorNetwork,
    CnnCriticNetwork,
    CriticNetwork,
    make_cnn_trunk,
)
from .rollout_buffer import RolloutBuffer
from .utils import format_update_line, get_device, seed_everything

_AGENT_REGISTRY: dict[str, type] = {}

DEFAULT_SEED = 42

def register_agent(cls):
    """Class decorator: add ``cls`` to the agent registry under its name."""
    _AGENT_REGISTRY[cls.__name__] = cls
    return cls

# ---------------------------------------------------------------------------
# PPOAgent 
# ---------------------------------------------------------------------------
@register_agent
class PPOAgent:
    """
    Custom PPO implementation. 
    See ``networks.py`` for the actor and value netqworks, and ``rollout_buffer.py`` for the rollout buffer.
    
    """
    DEFAULT_HYPERPARAMS: dict = {
        "rollout_size": 2048,
        "n_epochs": 10,
        "batch_size": 64,
        "lr": 1e-3,
        "gamma": 0.98,
        "gae_lambda": 0.95,
        "clip_eps": 0.2,
        "value_coef": 0.5,
        "entropy_coef": 0.01,
        "max_grad_norm": 0.5,
        "log_std_init": 0.0,
        "random_state": DEFAULT_SEED,
    }
    def __init__(
        self,
        env: gym.Env,
        hyperparameters: dict | None = None,
        device: str = "auto",
    ):
        self.env = env
        self.hyperparameters = dict(self.DEFAULT_HYPERPARAMS)
        if hyperparameters:
            self.hyperparameters.update(hyperparameters)
        self.device = get_device(device)
        seed_everything(self.hyperparameters.get("random_state", DEFAULT_SEED))

        # The agent reads obs/action shapes from the *single* env spaces so
        # construction works for both a vector env (used by train()) and a
        # plain single env (used by predict()/load() at inference time on
        # the Pi). Vector envs expose ``single_observation_space``; single
        # envs expose ``observation_space`` directly.
        single_obs_space = getattr(env, "single_observation_space", env.observation_space)
        single_act_space = getattr(env, "single_action_space", env.action_space)
        self.obs_dim = int(np.prod(single_obs_space.shape))
        self.action_dim = int(np.prod(single_act_space.shape))
        self.action_min = torch.as_tensor(
            single_act_space.low, dtype=torch.float32, device=self.device
        )
        self.action_max = torch.as_tensor(
            single_act_space.high, dtype=torch.float32, device=self.device
        )
        self.obs_min = single_obs_space.low
        self.obs_max = single_obs_space.high

        # Auto-select MLP vs CNN architecture from the observation shape.
        # 1D obs (e.g. Pendulum's (3,)) -> MLP. 3D obs (e.g. CarRacing's
        # (4, 84, 84) after Resize→Grayscale→FrameStack) -> shared-trunk CNN.
        # See specs/005-carracing-drivers/data-model.md.
        obs_shape = tuple(single_obs_space.shape)
        if len(obs_shape) == 1:
            self.network_arch = "mlp"
            self.actor = ActorNetwork(self.obs_dim, self.action_dim).to(self.device)
            self.critic = CriticNetwork(self.obs_dim).to(self.device)
        elif len(obs_shape) == 3:
            self.network_arch = "cnn"
            in_channels = obs_shape[0]
            trunk = make_cnn_trunk(in_channels).to(self.device)
            # Same trunk instance shared between actor and critic; PPOAgent
            # deduplicates parameters in _trainable_parameters() so the
            # optimizer doesn't update the trunk twice per step.
            self.actor = CnnActorNetwork(trunk, self.action_dim).to(self.device)
            self.critic = CnnCriticNetwork(trunk).to(self.device)
        else:
            raise ValueError(
                f"PPOAgent: unsupported observation shape {obs_shape}. "
                f"Expected 1D (vector) or 3D (image) observations after env wrappers."
            )

        self.log_std = nn.Parameter(
            torch.ones(self.action_dim, device=self.device)
            * self.hyperparameters["log_std_init"]
        )

    def _prep_obs(self, obs) -> torch.Tensor:
        """Convert a raw observation (numpy or tensor) into a float32 tensor
        on ``self.device``. For CNN agents, divides by 255 so the conv
        stack sees inputs in [0, 1] (matching SB3's CnnPolicy).
        """
        t = torch.as_tensor(obs, dtype=torch.float32, device=self.device)
        if self.network_arch == "cnn":
            t = t / 255.0
        return t

    def _trainable_parameters(self) -> list[nn.Parameter]:
        """Return the unique parameters spanning actor + critic + log_std.

        For CNN agents the trunk is shared between actor and critic; without
        this dedup the optimizer would update the trunk weights twice per
        step. The MLP path has no shared params so dedup is a no-op.
        """
        seen: set[int] = set()
        unique: list[nn.Parameter] = []
        for p in list(self.actor.parameters()) + list(self.critic.parameters()) + [self.log_std]:
            if id(p) not in seen:
                seen.add(id(p))
                unique.append(p)
        return unique

    # ===========================================================================
    # TODO 2 — Sample an action from the policy
    # ===========================================================================
    def sample_action(self, obs: torch.Tensor, deterministic: bool = False) -> tuple[torch.Tensor, torch.Tensor]:
        """Sample an action from the policy and return ``(action, log_prob)``.

        Uses ``self.actor`` (obs → action mean) and ``self.log_std``
        (per-action-dim learnable log std-dev). Action range comes from
        ``self.action_min`` / ``self.action_max``.

        Args:
            obs:           shape ``(obs_dim,)`` for a single observation, or
                           ``(B, obs_dim)`` batched. CHW pixel obs of shape
                           ``(C, H, W)`` / ``(B, C, H, W)`` also work — the
                           actor handles its own input shape.
            deterministic: if ``True``, return the policy mean (no sampling).

        Returns:
            action:   clipped to ``[self.action_min, self.action_max]``.
                      Same shape as the actor's mean output.
            log_prob: shape ``(B,)`` (or scalar for unbatched input). Log-prob
                      of the UNCLIPPED sample, summed over action dims.

        Math:
            π(a | s) = N(mean=actor(s), std=exp(log_std))
            log π(a | s) = sum_i log N_i(a_i | mean_i, std_i)

        Steps (pseudo-code):
            1. ``mean = self.actor(obs)``
            2. ``dist = torch.distributions.Normal(mean, self.log_std.exp())``
            3. ``unclipped = mean if deterministic else dist.sample()``
            4. ``log_prob = dist.log_prob(unclipped).sum(dim=-1)``
            5. ``action = unclipped.clamp(self.action_min, self.action_max)``
            6. Return ``(action, log_prob)``.

        Gotchas:
            * ``log_prob`` MUST be computed on the **unclipped** sample.
              Clipping only the returned action keeps the gradient honest;
              clipping before log-prob would assign zero gradient to
              boundary-clipped actions.
            * Sum over the **last** dim (``dim=-1``) so a multi-dim action
              produces ONE log-prob per sample. ``log_prob.sum()`` (no axis)
              would collapse the batch dim too and break the per-batch loss.
            * ``self.log_std.exp()`` not ``self.log_std`` directly — the
              learnable parameter is in log-space for unconstrained gradients.
            * ``Normal.sample()`` does not back-propagate. That's intended at
              rollout time; gradients flow through ``log_prob`` in
              ``evaluate_actions`` during the update phase.
        """
        # -- YOUR CODE HERE --
        raise NotImplementedError(
            "TODO 2: sample an action from the policy and return (action, log_prob). "
            "See docstring above for hints."
        )
        # -- END YOUR CODE --

    ###==========================================================================
    # TODO 3 — Evaluate actions under the current policy
    #============================================================================
    def evaluate_actions(
        self,
        obs: torch.Tensor,
        actions: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Evaluate a batch of ``(obs, action)`` under the current policy.

        Used inside the PPO update phase: actions sampled at rollout time are
        re-scored under the *current* policy network so we can compute the
        importance ratio and the entropy bonus.

        Uses ``self.actor`` and ``self.log_std`` internally.

        Args:
            obs:      shape ``(B, obs_dim)`` (or ``(B, C, H, W)`` for pixel
                      observations — the actor handles its own input shape).
            actions:  shape ``(B, action_dim)``. These are the actions
                      collected during rollout (post-clip values) — that's
                      fine: log-prob under a Normal is well-defined inside
                      and outside the clip range.

        Returns:
            log_probs: shape ``(B,)``. Per-sample log-prob, summed over
                       action dims.
            entropy:   shape ``(B,)``. Per-sample policy entropy, summed
                       over action dims.

        Math:
            log π_new(a | s) = sum_i log N_i(a_i | mean_i, std_i)
            H[π_new(· | s)]  = sum_i H[N_i(mean_i, std_i)]
            (Gaussian entropy: 0.5 * log(2 * π * e * std²) per dim.)

        Steps (pseudo-code):
            1. ``mean = self.actor(obs)``
            2. ``dist = Normal(mean, self.log_std.exp())``
            3. ``log_probs = dist.log_prob(actions).sum(dim=-1)``
            4. ``entropy   = dist.entropy().sum(dim=-1)``
            5. Return ``(log_probs, entropy)``.

        Gotchas:
            * Action-dim summation MUST match ``sample_action`` (also
              ``sum(dim=-1)``). If TODO 2 sums and TODO 3 doesn't (or vice
              versa), the importance ratio in TODO 4 will be off by a factor
              of ``action_dim`` and training silently diverges.
            * ``log_probs`` here CARRIES gradient through ``self.actor`` /
              ``self.log_std``; that gradient is what PPO's policy loss
              optimises. Do not call ``.detach()``.
            * ``self.log_std.exp()`` (same as TODO 2) — log-space parameter.
        """
        # -- YOUR CODE HERE --
        raise NotImplementedError(
            "TODO 3: evaluate actions under the current policy. "
            "See docstring above for hints."
        )
        # -- END YOUR CODE --


    # ===========================================================================
    # TODO 4 — PPO clipped surrogate loss
    # ===========================================================================
    def ppo_loss(
        self,
        new_log_probs: torch.Tensor,
        old_log_probs: torch.Tensor,
        advantages: torch.Tensor,
        clip_eps: float = 0.2,
    ) -> torch.Tensor:
        """PPO clipped surrogate objective. Returns the POLICY loss only.

        The training loop combines this with the value loss (MSE) and an
        entropy bonus into a single scalar to back-prop.

        Args:
            new_log_probs: shape ``(B,)``. Log-probs under the CURRENT policy
                           (from TODO 3). Carries gradient.
            old_log_probs: shape ``(B,)``. Log-probs captured at rollout time.
                           Detached — no gradient flows through this.
            advantages:    shape ``(B,)``. Buffer-normalised to mean 0, std 1
                           by ``RolloutBuffer.compute_returns_and_advantages``.
            clip_eps:      clip-range epsilon. Typically ``0.2``.

        Returns:
            Scalar tensor (zero-dim) with ``requires_grad=True``. Negative
            because the surrogate is maximised but PyTorch optimisers
            minimise.

        Math:
            r_t   = exp(log π_new(a_t|s_t) - log π_old(a_t|s_t))
            surr1 = r_t * A_t
            surr2 = clip(r_t, 1-ε, 1+ε) * A_t
            L     = -E_t[ min(surr1, surr2) ]

        Steps (pseudo-code):
            1. ``ratio = (new_log_probs - old_log_probs).exp()``
            2. ``surr1 = ratio * advantages``
            3. ``surr2 = ratio.clamp(1.0 - clip_eps, 1.0 + clip_eps) * advantages``
            4. ``loss  = -torch.min(surr1, surr2).mean()``
            5. Return ``loss``.

        Gotchas:
            * ``min`` not ``max``: PPO clips the *surrogate*, then takes the
              MIN to be conservative (pessimistic over both branches), then
              negates because we minimise.
            * ``ratio.clamp(...)`` clamps the multiplier, not the loss.
              Clamping ``surr1`` directly is wrong — it breaks the
              "clip the ratio" semantics.
            * ``advantages`` should already be normalised (zero mean, unit
              std). Don't re-normalise here.
            * ``new_log_probs - old_log_probs`` not the other way round.
              Sign flip silently destroys learning.
            * Returns the **policy loss only**. The training loop adds
              ``value_coef * value_loss - entropy_coef * entropy.mean()``.
              Don't fold those into this function.
        """
        # -- YOUR CODE HERE --
        raise NotImplementedError(
            "TODO 4: compute the PPO clipped surrogate loss. "
            "See docstring above for hints."
        )
        # -- END YOUR CODE --

    # ===========================================================================
    # TODO 5 — PPO training loop
    # ===========================================================================
    def train(
            self,
        env,
        total_timesteps: int = 8192,
        random_state: int = DEFAULT_SEED,
        log_fn=print,
    ) -> dict:
        """Wire everything together into a working PPO training loop.

        The rollout, bootstrapping, vectorised env handling, optimiser, and
        learning-rate schedule are PROVIDED. You only fill three small
        spots — TODO 5a, 5b, 5c — that connect the building blocks (TODOs
        1–4) into a working loop.

        Args:
            env:             a ``gymnasium.vector.VectorEnv`` constructed with
                             ``gym.make_vec(env_id, num_envs=N,
                             vectorization_mode='sync',
                             vector_kwargs={'autoreset_mode':
                             AutoresetMode.SAME_STEP})``. ``rollout_size``
                             must be divisible by ``num_envs``.
            total_timesteps: target env-step budget. The actual number of
                             updates is ``max(1, total_timesteps //
                             rollout_size)``.
            random_state:    seed for ``random / numpy / torch`` (the env is
                             seeded inside via ``env.reset(seed=...)``).
            log_fn:          callable receiving one formatted log line per
                             update. Defaults to ``print``; drivers pass a
                             closure that also writes to ``metrics.jsonl``.

        Returns:
            Dict with keys ``mean_reward``, ``policy_loss``, ``value_loss``,
            ``entropy``, ``n_updates`` (last-update statistics).

        Math (one update step):
            For each minibatch of size B drawn from the buffer:
                ratio        = exp(log π_new - log π_old)               (TODO 4)
                policy_loss  = -mean( min(ratio·A, clip(ratio, 1±ε)·A) )  (TODO 4)
                value_loss   = mean( (V_θ(s) - returns)² )
                loss         = policy_loss + c_v·value_loss - c_e·H[π_new]
            ∇loss → Adam step → clip-grad-norm.

        Steps (pseudo-code) — sub-TODOs:
            (provided) collect a rollout of ``rollout_size`` transitions
                       across ``num_envs`` parallel envs into ``buffer``.
            (provided) compute the bootstrap ``last_value`` for the
                       non-terminal envs.
            5a:        compute returns and advantages on the buffer using
                       the bootstrap value (one method call).
            (provided) iterate ``n_epochs`` epochs × ``buffer.get_batches``
                       minibatches. Each batch comes pre-shaped for SGD.
            5b:        for each minibatch — call ``evaluate_actions``,
                       compute ``p_loss`` (TODO 4), ``v_loss``
                       (``F.mse_loss``), then combine the components into
                       a single scalar ``loss`` (this is the line that
                       carries the gradient). The surrounding scaffold
                       handles ``optimizer.zero_grad`` → ``loss.backward``
                       → ``clip_grad_norm`` → ``optimizer.step`` and the
                       per-batch loss tracking for the log line.
            (provided) emit one ``format_update_line`` log per update and
                       step the LR scheduler.
            5c:        reset the buffer for the next rollout (one call).

        Gotchas:
            * 5a — pass ``last_value`` (the bootstrap), ``gamma``, and
              ``gae_lambda`` in that order. The buffer normalises advantages
              internally; don't re-normalise.
            * 5b — combine: ``loss = p_loss + value_coef * v_loss -
              entropy_coef * entropy.mean()``. The minus sign on entropy is
              deliberate — entropy is a bonus, not a cost.
            * 5b — call ``entropy.mean()`` explicitly. ``entropy`` is shape
              ``(B,)`` (one per sample); ``loss`` must be scalar for
              ``backward()``.
            * 5b — record ``last_p_loss / last_v_loss / last_entropy`` from
              the LAST minibatch of the LAST epoch. The format_update_line
              call below uses these.
            * 5c — must run after the update phase but before the next
              rollout starts, otherwise rollouts t and t+1 share storage
              and the buffer overflows.
            * Use ``format_update_line(...)`` for the per-update print so
              the script-mode runner can parse losses.
        """
        # Re-seed for reproducibility within a single train() call.
        seed_everything(random_state)
        lr = self.hyperparameters["lr"]
        rollout_size = self.hyperparameters["rollout_size"]
        n_epochs = self.hyperparameters["n_epochs"]
        batch_size = self.hyperparameters["batch_size"]
        gamma = self.hyperparameters["gamma"]
        gae_lambda = self.hyperparameters["gae_lambda"]
        clip_eps = self.hyperparameters["clip_eps"]
        value_coef = self.hyperparameters["value_coef"]
        entropy_coef = self.hyperparameters["entropy_coef"]
        max_grad_norm = self.hyperparameters["max_grad_norm"]

        # Vector env: amortise per-step inference cost across N parallel envs.
        # Drivers construct a vector env via gym.make_vec(..., autoreset_mode=
        # SAME_STEP), so when an env terminates/truncates the auto-reset
        # happens within the same step and the pre-reset final obs is
        # reported in info["final_obs"][i].
        if not hasattr(env, "num_envs"):
            raise ValueError(
                "PPOAgent.train() requires a vector env (gymnasium.vector). "
                "Construct one with gym.make_vec(env_id, num_envs=N, "
                "vectorization_mode='sync', vector_kwargs={'autoreset_mode': "
                "AutoresetMode.SAME_STEP}). For single-env inference use predict()."
            )
        num_envs = env.num_envs
        if rollout_size % num_envs != 0:
            raise ValueError(
                f"rollout_size ({rollout_size}) must be divisible by num_envs "
                f"({num_envs}); got remainder {rollout_size % num_envs}."
            )
        size_per_env = rollout_size // num_envs

        optimizer = torch.optim.Adam(
            self._trainable_parameters(),
            lr=lr,
            eps=1e-5,
        )
        n_updates = max(1, total_timesteps // rollout_size)

        scheduler = torch.optim.lr_scheduler.LambdaLR(
            optimizer,
            lr_lambda=lambda step: 1.0 - step / n_updates,
        )
        # Store obs with their full shape so CNN batches can be reshaped
        # back to (B, C, H, W) for the conv forward. For MLP, this is just
        # (obs_dim,) and the existing flat layout is preserved.
        obs_shape = tuple(env.single_observation_space.shape)
        buffer = RolloutBuffer(
            size_per_env,
            self.obs_dim,
            self.action_dim,
            num_envs=num_envs,
            obs_shape=obs_shape,
        )

        obs, _ = env.reset(seed=random_state)  # (N, obs_dim)
        episode_returns: list[float] = []
        current_returns = np.zeros(num_envs, dtype=np.float32)
        timesteps = 0
        final_stats: dict = {}

        for update_idx in range(1, n_updates + 1):
            # ----- rollout collection -----
            # Keep policy/value tensors on self.device for the whole rollout
            # and sync to numpy ONCE at the end. The only forced per-step
            # sync is action.cpu().numpy() (env.step needs numpy). Rewards
            # and dones come from env.step as numpy and stay on CPU until
            # the bulk transfer.
            obs_acc: list[torch.Tensor] = []
            action_acc: list[torch.Tensor] = []
            log_prob_acc: list[torch.Tensor] = []
            value_acc: list[torch.Tensor] = []
            reward_acc: list[np.ndarray] = []
            done_acc: list[np.ndarray] = []

            for _ in range(size_per_env):
                obs_t = self._prep_obs(obs)
                with torch.no_grad():
                    action_t, log_prob_t = self.sample_action(obs_t, deterministic=False)
                    value_t = self.critic(obs_t)  # (N,)

                # The one unavoidable per-step sync: env.step needs numpy.
                action_np = action_t.detach().cpu().numpy().astype(np.float32)
                next_obs, reward, terminated, truncated, info = env.step(action_np)
                reward = reward.astype(np.float32)
                done = np.logical_or(terminated, truncated).astype(np.float32)

                # Per-env episode returns, on CPU using the pre-bootstrap reward.
                current_returns += reward
                done_mask = (terminated | truncated)
                if done_mask.any():
                    for i in np.where(done_mask)[0]:
                        episode_returns.append(float(current_returns[i]))
                        current_returns[i] = 0.0

                # Bootstrap through truncations (not terminations). With
                # SAME_STEP autoreset, the true final obs is in
                # info["final_obs"][i]. The critic forward + .cpu() sync
                # only happens when at least one env truncated this step
                # (not every step), so it does not dominate.
                trunc_only = truncated & (~terminated)
                if trunc_only.any() and "final_obs" in info:
                    final_obs_arr = info["final_obs"]
                    mask = info.get("_final_obs", trunc_only)
                    trunc_idxs = np.where(trunc_only & mask)[0]
                    if len(trunc_idxs) > 0:
                        final_stack = np.stack(
                            [np.asarray(final_obs_arr[i], dtype=np.float32) for i in trunc_idxs]
                        )
                        with torch.no_grad():
                            terminal_values = self.critic(
                                self._prep_obs(final_stack)
                            ).detach().cpu().numpy()
                        for k, i in enumerate(trunc_idxs):
                            reward[i] = reward[i] + gamma * float(terminal_values[k])

                obs_acc.append(obs_t)
                action_acc.append(action_t.detach())
                log_prob_acc.append(log_prob_t.detach())
                value_acc.append(value_t.detach())
                reward_acc.append(reward)
                done_acc.append(done)

                timesteps += num_envs
                obs = next_obs

            # End-of-rollout: stack lists into (T, N, ...) device tensors and
            # transfer once per quantity. Single bulk MPS→CPU per array.
            obs_stack = torch.stack(obs_acc, dim=0)
            action_stack = torch.stack(action_acc, dim=0)
            log_prob_stack = torch.stack(log_prob_acc, dim=0)
            value_stack = torch.stack(value_acc, dim=0)

            buffer.obs[:] = obs_stack.cpu().numpy()
            buffer.actions[:] = action_stack.cpu().numpy()
            buffer.log_probs[:] = log_prob_stack.cpu().numpy()
            buffer.values[:] = value_stack.cpu().numpy()
            buffer.rewards[:] = np.stack(reward_acc, axis=0)
            buffer.dones[:] = np.stack(done_acc, axis=0)

            # Bootstrap final value for envs that did NOT just terminate.
            # (dones[T-1]=1 cuts the GAE bootstrap automatically when needed.)
            with torch.no_grad():
                last_value = self.critic(
                    self._prep_obs(obs)
                ).detach().cpu().numpy().astype(np.float32)

            # ----- TODO 5a: compute returns and advantages -----
            # See docstring section "5a". One method call on `buffer`.
            # -- YOUR CODE HERE --
            raise NotImplementedError(
                "TODO 5a: compute returns and advantages on the buffer. "
                "See docstring section 5a."
            )
            # -- END YOUR CODE --

            # ----- update phase -----
            last_p_loss = 0.0
            last_v_loss = 0.0
            last_entropy = 0.0
            for _ in range(n_epochs):
                for batch in buffer.get_batches(batch_size):
                    batch = {k: v.to(self.device) for k, v in batch.items()}
                    # ----- TODO 5b: compose the combined PPO loss -----
                    # See docstring section "5b". For each minibatch:
                    #   1. score the actions under the current policy
                    #      (TODO 3 — self.evaluate_actions(obs, actions))
                    #      → new_log_probs (B,) and entropy (B,)
                    #   2. compute the critic's value estimate
                    #      (self.critic(obs))
                    #   3. compute the policy loss (TODO 4 — self.ppo_loss)
                    #   4. compute the value loss
                    #      (F.mse_loss(pred_values, batch["returns"]))
                    #   5. combine into ONE scalar `loss`, subtracting the
                    #      entropy bonus (entropy is a bonus, not a cost).
                    # The scaffold below handles zero_grad → backward →
                    # clip_grad_norm → optimizer.step. Just produce `loss`.
                    # -- YOUR CODE HERE --
                    raise NotImplementedError(
                        "TODO 5b: compose the combined PPO loss from the components. "
                        "See docstring section 5b."
                    )
                    # -- END YOUR CODE --

                    optimizer.zero_grad()
                    loss.backward()
                    nn.utils.clip_grad_norm_(
                        self._trainable_parameters(),
                        max_grad_norm,
                    )
                    optimizer.step()

                    last_p_loss = float(p_loss.item())
                    last_v_loss = float(v_loss.item())
                    last_entropy = float(entropy.mean().item())
                   


            mean_return = (
                float(np.mean(episode_returns[-10:]))
                if episode_returns
                else float(current_returns.mean())
            )
            log_fn(
                format_update_line(
                    update_idx, n_updates, timesteps,
                    last_p_loss, last_v_loss, last_entropy, mean_return, lr=scheduler.get_last_lr()[0]
                )
            )
            scheduler.step()

            # ----- TODO 5c: reset rollout buffer for the next update -----
            # See docstring section "5c". One method call on `buffer`.
            # -- YOUR CODE HERE --
            raise NotImplementedError(
                "TODO 5c: reset the rollout buffer for the next update. "
                "See docstring section 5c."
            )
            # -- END YOUR CODE --

            final_stats = {
                "mean_reward": mean_return,
                "policy_loss": last_p_loss,
                "value_loss": last_v_loss,
                "entropy": last_entropy,
                "n_updates": update_idx,
            }

        return final_stats

    def predict(self, obs: np.ndarray, deterministic: bool = False) -> np.ndarray:
        """Take a raw observation (NumPy), return an action (NumPy).

        Converts to a tensor on ``self.device`` at the boundary (and
        normalises CNN inputs by /255) so callers never touch torch.
        """
        obs_t = self._prep_obs(obs)
        with torch.no_grad():
            action, _ = self.sample_action(obs_t, deterministic=deterministic)
        return action.cpu().numpy().astype(np.float32)
    
    def evaluate(
        self,
        env: gym.Env,
        n_episodes: int = 10,
        record_video: bool = True,
        video_dir: str | Path | None = None,
    ) -> list[float]:
        """Run greedy evaluation episodes; optionally record one video.

        Args:
            env:           the env used to derive ``env_id`` (for the video env).
                           Per-step rollouts use a fresh env so the recorder
                           can wrap it cleanly without disturbing the caller's
                           env.
            n_episodes:    number of greedy episodes.
            record_video:  if True, wrap a fresh env in
                           ``gymnasium.wrappers.RecordVideo`` and produce
                           ``<video_dir>/eval.mp4``.
            video_dir:     directory for the video. Defaults to the current
                           working directory; the driver always passes the
                           run directory.

        Returns:
            List of per-episode total returns (length ``n_episodes``).

        Notes:
            - On ``ImportError`` /
              ``gymnasium.error.DependencyNotInstalled`` (ffmpeg missing),
              writes a sentinel ``eval.mp4.skipped`` and returns the
              episode returns without a video.
            - Uses ``predict(obs, deterministic=True)`` for the policy.
        """
        env_id = env.spec.id if env.spec is not None else None
        if env_id is None:
            raise ValueError(
                "evaluate() needs env.spec.id to construct the recording env. "
                "Pass an env created via gym.make(...)."
            )
        video_dir = Path(video_dir) if video_dir is not None else Path.cwd()

        if record_video:
            video_dir.mkdir(parents=True, exist_ok=True)
            try:
                eval_env = gym.make(env_id, render_mode="rgb_array")
                eval_env = gym.wrappers.RecordVideo(
                    eval_env,
                    video_folder=str(video_dir),
                    name_prefix="eval",
                    episode_trigger=lambda episode_id: episode_id == 0,
                )
            except (
                ImportError,
                gym.error.DependencyNotInstalled,
            ) as exc:
                print(
                    f"[ppo.evaluate] ffmpeg/imageio missing ({exc!r}); "
                    f"writing eval.mp4.skipped and continuing without video.",
                    file=sys.stderr,
                )
                (video_dir / "eval.mp4.skipped").write_text("")
                eval_env = gym.make(env_id)
                record_video = False
        else:
            eval_env = gym.make(env_id)

        returns: list[float] = []
        try:
            for _ in range(n_episodes):
                obs, _ = eval_env.reset()
                ep_return = 0.0
                done = False
                while not done:
                    action = self.predict(obs, deterministic=True)
                    obs, reward, terminated, truncated, _ = eval_env.step(action)
                    ep_return += float(reward)
                    done = bool(terminated or truncated)
                returns.append(ep_return)
        finally:
            eval_env.close()

        # RecordVideo names the file ``eval-episode-0.mp4``; rename to the
        # contract-mandated ``eval.mp4`` so analyze.ipynb finds it.
        if record_video:
            produced = video_dir / "eval-episode-0.mp4"
            target = video_dir / "eval.mp4"
            if produced.exists():
                if target.exists():
                    target.unlink()
                produced.rename(target)

        return returns
    
    def save(self, path: str) -> None:
        """Persist model weights, hyperparameters, and preprocess state."""
        state = {
            "class_name": type(self).__name__,
            "network_arch": self.network_arch,
            "actor_state_dict": self.actor.state_dict(),
            "value_state_dict": self.critic.state_dict(),
            "log_std": self.log_std.detach().clone(),
            "hyperparameters": self.hyperparameters,
        }
        torch.save(state, path)

    @classmethod
    def load(cls, path: str, env: gym.Env) -> "PPOAgent":
        """Load any registered subclass by class name from a single ``.pt``.

        The ``env`` argument is required so the loaded agent can reconstruct
        ``obs_dim`` / ``action_dim`` / action bounds. Pass the same (wrapped)
        env you would use for training or evaluation.

        If the saved checkpoint records a ``network_arch`` (added in feature
        005), it is checked against the env-implied architecture. Older
        checkpoints predating this field skip the check (backwards compat).
        """
        state = torch.load(path, weights_only=False)
        class_name = state["class_name"]
        if class_name not in _AGENT_REGISTRY:
            raise ValueError(
                f"Unknown agent class {class_name!r}. Registered classes: "
                f"{sorted(_AGENT_REGISTRY)}. Make sure the subclass module "
                f"is imported before calling PPOAgent.load()."
            )
        target_cls = _AGENT_REGISTRY[class_name]
        agent = target_cls(env, hyperparameters=state["hyperparameters"])
        saved_arch = state.get("network_arch")
        if saved_arch is not None and saved_arch != agent.network_arch:
            raise ValueError(
                f"PPOAgent.load: saved checkpoint has network_arch={saved_arch!r} "
                f"but the env implies network_arch={agent.network_arch!r}. The "
                f"observation shape from the env doesn't match what the model "
                f"was trained on; pass the same wrapper chain you trained with."
            )
        agent.actor.load_state_dict(state["actor_state_dict"])
        agent.critic.load_state_dict(state["value_state_dict"])
        with torch.no_grad():
            agent.log_std.copy_(state["log_std"])
        return agent
    

    