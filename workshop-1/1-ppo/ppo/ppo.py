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
    ):
        self.env = env
        self.hyperparameters = dict(self.DEFAULT_HYPERPARAMS)
        if hyperparameters:
            self.hyperparameters.update(hyperparameters)
        self.device = get_device()
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
            5. action    = unclipped.clamp(action_low, action_high)

        Important: log_prob is computed on the UNCLIPPED sample. Clipping
        only the returned action keeps the gradient honest.
        """
        # -- YOUR CODE HERE --
        mean = self.actor(obs)
        dist = Normal(mean, self.log_std.exp())
        if deterministic:
            unclipped = mean
        else:
            unclipped = dist.sample()
        log_prob = dist.log_prob(unclipped).sum(dim=-1)
        action = unclipped.clamp(self.action_min, self.action_max)
        return action, log_prob
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

        Uses ``self.actor`` and ``self.log_std`` internally.

        Args:
            obs:      shape ``(B, obs_dim)``.
            actions:  shape ``(B, action_dim)``.

        Returns:
            ``(log_probs, entropy)``, both of shape ``(B,)``.

        Hints:
            1. mean      = self.actor(obs)
            2. dist      = Normal(mean, self.log_std.exp())
            3. log_probs = dist.log_prob(actions).sum(dim=-1)
            4. entropy   = dist.entropy().sum(dim=-1)
        """
        # -- YOUR CODE HERE --
        mean = self.actor(obs)
        dist = Normal(mean, self.log_std.exp())
        log_probs = dist.log_prob(actions).sum(dim=-1)
        entropy = dist.entropy().sum(dim=-1)
        return log_probs, entropy
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
        # TODO: Initialize rollout buffer
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

            #TODO: compute returns and advantages in buffer
            buffer.compute_returns_and_advantages(last_value, gamma, gae_lambda)

            # ----- update phase -----
            last_p_loss = 0.0
            last_v_loss = 0.0
            last_entropy = 0.0
            for _ in range(n_epochs):
                for batch in buffer.get_batches(batch_size):
                    batch = {k: v.to(self.device) for k, v in batch.items()}
                    # TODO: compute PPO policy loss, value loss, and combine with entropy bonus into a single scalar loss
                    # Hint: Use evaluate_actions to get the `new_log_probs` and `entropy`.
                    #       Use self.ppo_loss() to get the actor loss and F.mse_loss() for the critic loss.
                    #       Use use actor, critic loss and entropy to compute the final loss.
                    new_log_probs, entropy = self.evaluate_actions(
                        batch["obs"], batch["actions"],
                    )
                    pred_values = self.critic(batch["obs"])
                   
                    p_loss = self.ppo_loss(
                        new_log_probs, batch["old_log_probs"], batch["advantages"], clip_eps,
                    )
                    v_loss = F.mse_loss(pred_values, batch["returns"])
                    loss = p_loss + value_coef * v_loss - entropy_coef * entropy.mean()

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

            #TODO: reset rollout buffer for next update
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
    

    