"""CarRacing-v3 PPOAgent subclass stub.

This file demonstrates the override-contract pattern for **pixel**
observations from
``specs/001-ppo-skeleton/contracts/agent-interface.md``.

IMPORTANT — what this file is and is not:

  * IS: a working ``preprocess`` pipeline (crop sky, grayscale, resize
    to 84x84, normalize to [0, 1], frame-stack 4) that returns a
    ``(4, 84, 84)`` float32 tensor and demonstrates how subclass
    preprocessing state round-trips through ``save`` / ``load``.

  * IS NOT: a working end-to-end CarRacing training script. The
    stage-1 ``ActorNetwork`` is a 2x64 MLP that expects vector input
    of shape ``(obs_dim,)``. Calling ``train()`` on a
    ``CarRacingPPOAgent`` will fail because the MLP cannot consume
    the ``(4, 84, 84)`` pixel tensor. Stage 3
    (``workshop-1/3-car-racing/``) will replace ``ActorNetwork`` with
    a CNN that consumes pixel input and ship its own training script
    that uses this preprocess pipeline.
"""
from __future__ import annotations

import math
import os
import sys

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

_HERE = os.path.dirname(os.path.abspath(__file__))
_PPO_DIR = os.path.normpath(os.path.join(_HERE, "..", "1-ppo"))
if _PPO_DIR not in sys.path:
    sys.path.insert(0, _PPO_DIR)

from ppo import PPOAgent, register_agent  # noqa: E402


# ---------------------------------------------------------------------------
# NatureCNN-style actor / value networks for (4, 84, 84) preprocessed input.
# ---------------------------------------------------------------------------


class ActorCNN(nn.Module):
    """3 conv layers + linear head producing the action mean."""

    def __init__(self, action_dim: int):
        super().__init__()
        self.conv1 = nn.Conv2d(4, 32, kernel_size=8, stride=4)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1)
        self.fc = nn.Linear(64 * 7 * 7, 512)
        self.head = nn.Linear(512, action_dim)
        self._orthogonal_init()

    def _orthogonal_init(self) -> None:
        for layer in (self.conv1, self.conv2, self.conv3, self.fc):
            nn.init.orthogonal_(layer.weight, gain=math.sqrt(2))
            nn.init.zeros_(layer.bias)
        nn.init.orthogonal_(self.head.weight, gain=0.01)
        nn.init.zeros_(self.head.bias)

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        squeeze = (obs.dim() == 3)
        if squeeze:
            obs = obs.unsqueeze(0)
        x = F.relu(self.conv1(obs))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = x.flatten(start_dim=1)
        x = F.relu(self.fc(x))
        out = self.head(x)
        return out.squeeze(0) if squeeze else out


class ValueCNN(nn.Module):
    """Same conv stack as ActorCNN; scalar value output."""

    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(4, 32, kernel_size=8, stride=4)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1)
        self.fc = nn.Linear(64 * 7 * 7, 512)
        self.head = nn.Linear(512, 1)
        self._orthogonal_init()

    def _orthogonal_init(self) -> None:
        for layer in (self.conv1, self.conv2, self.conv3, self.fc):
            nn.init.orthogonal_(layer.weight, gain=math.sqrt(2))
            nn.init.zeros_(layer.bias)
        nn.init.orthogonal_(self.head.weight, gain=1.0)
        nn.init.zeros_(self.head.bias)

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        squeeze = (obs.dim() == 3)
        if squeeze:
            obs = obs.unsqueeze(0)
        x = F.relu(self.conv1(obs))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = x.flatten(start_dim=1)
        x = F.relu(self.fc(x))
        out = self.head(x).squeeze(-1)
        return out.squeeze(0) if squeeze else out


# CarRacing-tuned PPO defaults (see specs/.../research.md R3).
_CARRACING_HYPERPARAMS = {
    "rollout_size": 2048,
    "batch_size": 128,
    "n_epochs": 10,
    "lr": 2.5e-4,
    "clip_eps": 0.1,
    "entropy_coef": 0.0,
}


@register_agent
class CarRacingPPOAgent(PPOAgent):
    """PPOAgent for CarRacing-v3.

    Pixel observation: crop sky, grayscale, resize to 84x84, normalize
    to [0, 1], frame-stack 4. Output shape: ``(STACK_SIZE, 84, 84)``
    float32.

    The actor and value networks are CNNs (NatureCNN architecture) that
    consume the preprocessed ``(4, 84, 84)`` tensor directly.
    """

    STACK_SIZE = 4
    OUTPUT_HW = 84

    def __init__(self, obs_dim, action_dim, hyperparameters=None):
        merged: dict = dict(_CARRACING_HYPERPARAMS)
        if hyperparameters:
            merged.update(hyperparameters)
        super().__init__(obs_dim, action_dim, merged)
        self._frame_buffer: list[np.ndarray] = []
        # Replace the inherited 2x64 MLPs with CNNs sized for (4, 84, 84).
        self.actor = ActorCNN(action_dim)
        self.value = ValueCNN()

    def preprocess(self, obs):
        # Crop sky (CarRacing-v3 frames are 96x96; keep the 84 rows of road).
        img = obs[:84, :, :]
        # Grayscale.
        img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        # Resize to OUTPUT_HW square.
        img = cv2.resize(
            img, (self.OUTPUT_HW, self.OUTPUT_HW), interpolation=cv2.INTER_AREA
        )
        # Normalize to [0, 1].
        img = img.astype(np.float32) / 255.0
        # Frame stack — append, drop oldest if over capacity, pad with copies
        # of the current frame on the first call.
        self._frame_buffer.append(img)
        if len(self._frame_buffer) > self.STACK_SIZE:
            self._frame_buffer.pop(0)
        while len(self._frame_buffer) < self.STACK_SIZE:
            self._frame_buffer.append(img)
        return np.stack(self._frame_buffer, axis=0)

    def _get_preprocess_state(self) -> dict:
        return {"frame_buffer": [f.copy() for f in self._frame_buffer]}

    def _set_preprocess_state(self, state: dict) -> None:
        self._frame_buffer = list(state.get("frame_buffer", []))
