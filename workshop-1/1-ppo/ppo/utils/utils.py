
from random import random
import random
import numpy as np
import torch


# ---------------------------------------------------------------------------
# Helper — random seeding for reproducibility
# ---------------------------------------------------------------------------

def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

# ---------------------------------------------------------------------------
# Helper — log line formatter
# ---------------------------------------------------------------------------


def format_update_line(
    update_idx: int,
    n_updates: int,
    timesteps: int,
    policy_loss: float,
    value_loss: float,
    entropy: float,
    mean_return: float,
    lr: float
) -> str:
    """Format one fixed-width log line for the training loop.

    Use this inside your TODO 5 ``train()`` implementation so the script-mode
    runner can parse the loss values for its exit checks.
    """
    return (
        f"[update {update_idx:2d}/{n_updates}] "
        f"timesteps={timesteps:<6d}  "
        f"lr={lr:.3e}  "
        f"policy_loss={policy_loss:+.3f}  "
        f"value_loss={value_loss:+.3f}  "
        f"entropy={entropy:+.3f}  "
        f"mean_return={mean_return:+.2f}"
    )

# ---------------------------------------------------------------------------
# Helper — get device for PyTorch tensors
# ---------------------------------------------------------------------------
def get_device() -> torch.device:
    """Get the best available device (GPU if available, elif MPS, else CPU)."""
    return torch.device("cpu")
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif torch.backends.mps.is_available():
        return torch.device("mps")
    else:
        return torch.device("cpu")