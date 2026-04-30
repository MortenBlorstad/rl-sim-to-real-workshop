
import os
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
# Helper — silence macOS objc duplicate-class warnings on stderr
# ---------------------------------------------------------------------------

def silence_objc_dup_class_warnings() -> None:
    """Drop ``objc[PID]: Class ... is implemented in both ...`` lines from stderr.

    Emitted by the macOS Objective-C runtime when two dylibs (cv2's bundled
    libSDL2 and pygame's libSDL2) register the same classes. They are written
    directly to fd 2, so ``warnings.filterwarnings`` cannot catch them.

    Replaces fd 2 with a pipe and runs a daemon thread that forwards every
    line back to the original stderr, except lines starting with ``objc[``.
    No-op on non-macOS. Idempotent. Must be called before pygame/cv2 dylibs
    load (i.e. before ``import gymnasium`` in scripts that build CarRacing).
    """
    import sys
    if sys.platform != "darwin":
        return
    if getattr(silence_objc_dup_class_warnings, "_installed", False):
        return
    import os
    import threading

    real_fd = os.dup(2)
    r, w = os.pipe()
    os.dup2(w, 2)
    os.close(w)

    def _pump() -> None:
        buf = b""
        while True:
            try:
                chunk = os.read(r, 4096)
            except OSError:
                break
            if not chunk:
                break
            buf += chunk
            while b"\n" in buf:
                line, buf = buf.split(b"\n", 1)
                if not line.startswith(b"objc["):
                    os.write(real_fd, line + b"\n")
        if buf and not buf.startswith(b"objc["):
            os.write(real_fd, buf)

    threading.Thread(target=_pump, daemon=True).start()
    silence_objc_dup_class_warnings._installed = True  # type: ignore[attr-defined]


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

    Used inside your ``train()`` implementation so the script-mode
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
# Device-selection policy
# ---------------------------------------------------------------------------

_DEVICE_ENV_VAR = "RL_WORKSHOP_DEVICE"
_ALLOWED_VALUES = ("auto", "cpu", "cuda", "mps")


class DeviceUnavailableError(RuntimeError):
    """Raised when ``RL_WORKSHOP_DEVICE`` names a device the local machine
    cannot honour (e.g. ``cuda`` on a Mac without CUDA), or names an
    unrecognised value (e.g. ``gpu``).

    The message lists what IS available so the participant can self-correct.
    """


def _available_devices() -> list[str]:
    devices = ["cpu"]
    if torch.cuda.is_available():
        devices.append("cuda")
    if torch.backends.mps.is_available():
        devices.append("mps")
    return devices


def get_device(device: str = "auto") -> torch.device:
    """Resolve the active device per the workshop's policy.

    Reads ``RL_WORKSHOP_DEVICE`` from the process environment. Allowed values:
    ``cpu``, ``cuda``, ``mps``, ``auto`` (default ``auto``). Case-insensitive.

    Auto-selection order: ``cuda`` → ``mps`` → ``cpu``.

    Side effect: when the resolved device is MPS, sets
    ``PYTORCH_ENABLE_MPS_FALLBACK=1`` via ``os.environ.setdefault`` so that
    ops PyTorch does not implement on MPS fall back to CPU per-op (any
    user-set value is preserved).

    Raises ``DeviceUnavailableError`` when the requested device is
    unavailable or the env-var value is unrecognised.
    """
    raw = os.environ.get(_DEVICE_ENV_VAR, device)
    requested = raw.strip().lower()
    if requested not in _ALLOWED_VALUES:
        raise DeviceUnavailableError(
            f"{_DEVICE_ENV_VAR}={raw!r} is not a recognised value. "
            f"Allowed: cpu, cuda, mps, auto. Either unset it or pick one."
        )

    available = _available_devices()

    if requested == "auto":
        if "cuda" in available:
            resolved = "cuda"
        elif "mps" in available:
            resolved = "mps"
        else:
            resolved = "cpu"
    elif requested in available:
        resolved = requested
    else:
        raise DeviceUnavailableError(
            f"{_DEVICE_ENV_VAR}={requested!r} but no {requested.upper()} "
            f"backend is available on this machine. "
            f"Available: {', '.join(available)}. "
            f"Either unset {_DEVICE_ENV_VAR} (auto-select) or set it to one of "
            f"{', '.join(repr(d) for d in available)}."
        )

    if resolved == "mps":
        os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

    return torch.device(resolved)