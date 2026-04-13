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

import os
import sys

import cv2
import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_PPO_DIR = os.path.normpath(os.path.join(_HERE, "..", "1-ppo"))
if _PPO_DIR not in sys.path:
    sys.path.insert(0, _PPO_DIR)

from ppo_skeleton import PPOAgent, register_agent  # noqa: E402


@register_agent
class CarRacingPPOAgent(PPOAgent):
    """PPOAgent for CarRacing-v3.

    Pixel observation: crop sky, grayscale, resize to 84x84, normalize
    to [0, 1], frame-stack 4. Output shape: ``(STACK_SIZE, 84, 84)``
    float32.
    """

    STACK_SIZE = 4
    OUTPUT_HW = 84

    def __init__(self, obs_dim, action_dim, hyperparameters=None):
        super().__init__(obs_dim, action_dim, hyperparameters)
        self._frame_buffer: list[np.ndarray] = []

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
