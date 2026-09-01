"""Run provenance for the raw JSON (plan §5).

A coded transcript entering research must be able to state which model, which
revision, and which parameters produced it. Cheap to record now, unrecon-
structable later -- and this project has already had three benchmarks whose
conclusions were wrong because nobody wrote down the invocation.

Nothing here may fail a run. Every lookup degrades to a recorded ``null`` with
a stated reason rather than raising.
"""

from __future__ import annotations

import hashlib
import logging
import os
import platform
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)

_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")


def sha256_file(path: Path, chunk: int = 1 << 20) -> Optional[str]:
    """Content hash of the audio, so a transcript can be tied to its source."""
    try:
        h = hashlib.sha256()
        with open(path, "rb") as fh:
            while True:
                block = fh.read(chunk)
                if not block:
                    break
                h.update(block)
        return h.hexdigest()
    except OSError as exc:
        logger.warning("Could not hash %s: %s", path, exc)
        return None


def pipeline_version() -> Dict[str, Any]:
    """Which commit of this repo produced the file, and whether it was dirty."""
    def _git(*args: str) -> Optional[str]:
        try:
            out = subprocess.run(
                ["git", "-C", str(_PROJECT_ROOT), *args],
                capture_output=True, text=True, timeout=10,
            )
            return out.stdout.strip() if out.returncode == 0 else None
        except (OSError, subprocess.SubprocessError):
            return None

    commit = _git("rev-parse", "HEAD")
    status = _git("status", "--porcelain")
    return {
        "commit": commit,
        "branch": _git("rev-parse", "--abbrev-ref", "HEAD"),
        # A dirty tree means the recorded commit does not describe the code that
        # ran. Saying so is the difference between provenance and decoration.
        "dirty": bool(status) if status is not None else None,
    }


def hf_revision(model_id: str) -> Dict[str, Optional[str]]:
    """Resolve a HuggingFace repo id to the commit sha actually on disk.

    The local cache is consulted first: it answers with the revision that was
    *used*, needs no network, and works on a compute node with no egress. The
    Hub API is only a fallback, and it reports the current head -- which is not
    necessarily what ran.
    """
    if not model_id or model_id.startswith("mock/"):
        return {"revision": None, "revision_source": "not applicable"}

    local = Path(model_id)
    if local.is_dir():
        return {"revision": None, "revision_source": f"local directory {local}"}

    try:
        from huggingface_hub import constants, scan_cache_dir
    except ImportError:
        return {"revision": None, "revision_source": "huggingface_hub not installed"}

    try:
        for repo in scan_cache_dir(constants.HF_HUB_CACHE).repos:
            if repo.repo_id.lower() != model_id.lower():
                continue
            revs = sorted(repo.revisions, key=lambda r: r.last_modified, reverse=True)
            if revs:
                return {"revision": revs[0].commit_hash, "revision_source": "local cache"}
    except Exception as exc:  # noqa: BLE001 - provenance must never fail a run
        logger.debug("cache scan failed for %s: %s", model_id, exc)

    try:
        from huggingface_hub import HfApi
        return {"revision": HfApi().model_info(model_id).sha,
                "revision_source": "hub api (current head, may differ from what ran)"}
    except Exception as exc:  # noqa: BLE001
        logger.warning("Could not resolve a revision for %s: %s", model_id, exc)
        return {"revision": None, "revision_source": f"unresolved: {type(exc).__name__}"}


def package_versions(*names: str) -> Dict[str, Optional[str]]:
    """Versions of the libraries that did the work.

    ``transformers`` 4.37 versus 5.16 changes which generation flags are
    accepted, and the package version decides whether ``mode`` exists at all.
    """
    from importlib.metadata import PackageNotFoundError, version
    out: Dict[str, Optional[str]] = {}
    for name in names:
        try:
            out[name] = version(name)
        except PackageNotFoundError:
            out[name] = None
    return out


def run_block(device: str) -> Dict[str, Any]:
    return {
        "created_utc": utc_now(),
        "device": device,
        "hostname": platform.node(),
        "python": sys.version.split()[0],
        "slurm_job_id": os.getenv("SLURM_JOB_ID"),
        "slurm_partition": os.getenv("SLURM_JOB_PARTITION"),
        "pipeline_version": pipeline_version(),
        "packages": package_versions(
            "crisperwhisper", "transformers", "torch", "ctranslate2",
            "pyannote.audio", "librosa",
        ),
    }
