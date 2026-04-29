import math
import torch
import torch.nn as nn



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


class CriticNetwork(nn.Module):
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
# CNN variants for image-observation envs (e.g. CarRacing-v3 after
# Resize(84, 84) → Grayscale → FrameStack(4) wrappers, producing (4, 84, 84)).
#
# Shared-trunk Nature DQN architecture (matches SB3's ActorCriticCnnPolicy):
# one CNN backbone is constructed by the caller and passed to BOTH the actor
# and the critic. They register the same `trunk` Module so it appears in
# state_dict for either side, but PPOAgent must deduplicate parameters when
# building the optimizer (trunk params would otherwise be in the param list
# twice and get updated twice per step).
#
# See specs/005-carracing-drivers/research.md R2.
# ---------------------------------------------------------------------------


def make_cnn_trunk(in_channels: int, hidden: int = 512) -> nn.Sequential:
    """Conv(32, 8x8, s=4) → ReLU → Conv(64, 4x4, s=2) → ReLU →
    Conv(64, 3x3, s=1) → ReLU → flatten → Linear(64*7*7, hidden) → ReLU.
    Expects input (B, in_channels, 84, 84) float32 in [0, 1]. Output (B, hidden).
    Caller constructs once and passes the SAME instance to CnnActorNetwork and
    CnnCriticNetwork to share the trunk between them.
    """
    trunk = nn.Sequential(
        nn.Conv2d(in_channels, 32, kernel_size=8, stride=4),
        nn.ReLU(),
        nn.Conv2d(32, 64, kernel_size=4, stride=2),
        nn.ReLU(),
        nn.Conv2d(64, 64, kernel_size=3, stride=1),
        nn.ReLU(),
        nn.Flatten(),
        nn.Linear(64 * 7 * 7, hidden),
        nn.ReLU(),
    )
    for layer in trunk:
        if isinstance(layer, (nn.Conv2d, nn.Linear)):
            nn.init.orthogonal_(layer.weight, gain=math.sqrt(2))
            nn.init.zeros_(layer.bias)
    return trunk


class CnnActorNetwork(nn.Module):
    """Shared-trunk CNN backbone + linear policy-mean head.

    The ``trunk`` argument is the shared backbone; the same instance must
    also be passed to ``CnnCriticNetwork``. ``hidden`` must match the trunk's
    output width (default 512 from ``make_cnn_trunk``).
    """

    def __init__(self, trunk: nn.Module, action_dim: int, hidden: int = 512):
        super().__init__()
        self.trunk = trunk
        self.head = nn.Linear(hidden, action_dim)
        nn.init.orthogonal_(self.head.weight, gain=0.01)
        nn.init.zeros_(self.head.bias)

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        squeeze = obs.dim() == 3
        if squeeze:
            obs = obs.unsqueeze(0)
        feats = self.trunk(obs)
        out = self.head(feats)
        if squeeze:
            out = out.squeeze(0)
        return out


class CnnCriticNetwork(nn.Module):
    """Shared-trunk CNN backbone + linear scalar value head.

    Pass the same ``trunk`` instance you passed to ``CnnActorNetwork`` so
    both heads share the visual feature extractor.
    """

    def __init__(self, trunk: nn.Module, hidden: int = 512):
        super().__init__()
        self.trunk = trunk
        self.head = nn.Linear(hidden, 1)
        nn.init.orthogonal_(self.head.weight, gain=1.0)
        nn.init.zeros_(self.head.bias)

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        squeeze = obs.dim() == 3
        if squeeze:
            obs = obs.unsqueeze(0)
        feats = self.trunk(obs)
        out = self.head(feats).squeeze(-1)
        if squeeze:
            out = out.squeeze(0)
        return out
