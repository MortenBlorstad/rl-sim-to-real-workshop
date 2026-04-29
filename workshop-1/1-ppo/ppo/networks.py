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
