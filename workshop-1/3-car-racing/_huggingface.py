"""HuggingFace Hub helpers for the CarRacing SB3 driver.

Sibling to ``train_sb3.py``. Tightly coupled to that driver's CLI flags
and error-message contract — not part of the agent API. See
``specs/005-carracing-drivers/contracts/cli.md``.
"""
from __future__ import annotations

import time

from huggingface_hub import hf_hub_download
from huggingface_hub.errors import (
    EntryNotFoundError,
    LocalEntryNotFoundError,
    RepositoryNotFoundError,
)


class HuggingFaceLoadError(RuntimeError):
    """Raised when a HuggingFace download fails. The message follows the
    contract from ``specs/005-carracing-drivers/contracts/cli.md`` §
    'HuggingFace download failure'."""


def _default_filename(repo_id: str) -> str:
    """Auto-derive the artefact filename from the repo basename.

    ``sb3/ppo-CarRacing-v0`` -> ``ppo-CarRacing-v0.zip``.
    """
    return f"{repo_id.split('/', 1)[-1]}.zip"


def download_pretrained(repo_id: str, filename: str | None = None) -> str:
    """Download an SB3 checkpoint from HuggingFace Hub and return the local
    cached path.

    Prints ``[train_sb3] downloaded …`` (cache miss) or ``[train_sb3] (cache
    hit)`` (round-trip < 0.5s). Wraps the underlying ``huggingface_hub``
    exceptions in ``HuggingFaceLoadError`` with actionable messages so the
    driver can surface a single readable error rather than a stack trace.
    """
    resolved_filename = filename or _default_filename(repo_id)
    t0 = time.perf_counter()
    try:
        local_path = hf_hub_download(repo_id=repo_id, filename=resolved_filename)
    except LocalEntryNotFoundError as exc:
        raise HuggingFaceLoadError(
            f"could not download {resolved_filename!r} from HuggingFace repo "
            f"{repo_id!r}.\n"
            f"You appear to be offline and the file is not in the local cache "
            f"(~/.cache/huggingface/).\n"
            f"Either connect to the internet, or use a local pretrained "
            f"artefact instead (omit --hf-repo and load from pretrained/ as "
            f"documented in the README)."
        ) from exc
    except RepositoryNotFoundError as exc:
        raise HuggingFaceLoadError(
            f"HuggingFace repo {repo_id!r} does not exist (or is private and "
            f"you are not authenticated). Check the spelling, or pick a "
            f"public CarRacing repo such as 'sb3/ppo-CarRacing-v0'."
        ) from exc
    except EntryNotFoundError as exc:
        raise HuggingFaceLoadError(
            f"HuggingFace repo {repo_id!r} exists but does not contain a file "
            f"named {resolved_filename!r}. Use --hf-filename to specify the "
            f"correct filename, or check the repo's file list at "
            f"https://huggingface.co/{repo_id}/tree/main."
        ) from exc
    except OSError as exc:
        raise HuggingFaceLoadError(
            f"could not reach HuggingFace Hub to download {resolved_filename!r} "
            f"from {repo_id!r}: {exc}.\n"
            f"Check your internet connection or fall back to a local "
            f"pretrained artefact (omit --hf-repo)."
        ) from exc

    elapsed = time.perf_counter() - t0
    if elapsed < 0.5:
        print(f"[train_sb3] (cache hit) {repo_id}/{resolved_filename}")
    else:
        print(
            f"[train_sb3] downloaded {resolved_filename} from {repo_id} "
            f"in {elapsed:.1f}s"
        )
    return local_path
