"""pyannote speaker diarization.

Turns are recorded **raw** (D3): stage 1 stores what the diarizer said and
nothing more. Assigning a speaker to each word is stage 2's job, where the
assignment rule stays tunable and a misattribution is fixed without a GPU.

Settled 2026-08-31: stay on ``speaker-diarization-community-1``. DiariZen was
benchmarked against it and matched on speaker counts, speech totals and
runtime, which did not justify a second environment pinned to torch 2.1.1 with
a vendored pyannote fork.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

logger = logging.getLogger(__name__)


class Diarizer:
    def __init__(self, model_id: str, auth_token: Optional[str], device: Union[str, Any]):
        self.model_id = model_id
        self.auth_token = auth_token
        self.device = device
        self.pipeline = None

    def load(self) -> None:
        if not self.auth_token:
            logger.warning("No HF token provided; diarization will be skipped.")
            return
        try:
            import torch
            from pyannote.audio import Pipeline
        except ImportError:
            logger.error(
                "pyannote.audio is not installed; diarization unavailable. "
                "Install it, or pass --no_diarize."
            )
            return

        logger.info("Loading diarization pipeline (%s)...", self.model_id)
        safe_globals = [torch.torch_version.TorchVersion]
        try:
            from pyannote.audio.core.task import Problem, Specifications
            safe_globals += [Specifications, Problem]
        except ImportError:
            pass
        torch.serialization.add_safe_globals(safe_globals)

        original_load = torch.load

        def safe_load(*args, **kwargs):
            kwargs.pop("weights_only", None)
            return original_load(*args, **kwargs, weights_only=False)

        torch.load = safe_load
        try:
            target = torch.device(self.device) if isinstance(self.device, str) else self.device
            try:
                self.pipeline = Pipeline.from_pretrained(
                    self.model_id, token=self.auth_token
                ).to(target)
            except Exception as exc:  # noqa: BLE001
                if "403" in str(exc):
                    logger.error("HuggingFace 403: check the token's permissions.")
                    raise
                logger.info("Retrying with the legacy 'use_auth_token' argument...")
                self.pipeline = Pipeline.from_pretrained(
                    self.model_id, use_auth_token=self.auth_token
                ).to(target)
        except Exception as exc:  # noqa: BLE001
            logger.error("Failed to load the diarization pipeline: %s", exc)
            self.pipeline = None
        finally:
            torch.load = original_load

    def run(self, audio_path: Union[str, Path]) -> List[Dict[str, Any]]:
        if not self.pipeline:
            return []
        logger.info("Running speaker diarization...")
        segments: List[Dict[str, Any]] = []
        diarization = None
        try:
            diarization = self.pipeline(str(audio_path))
            for turn, speaker in diarization.speaker_diarization:
                segments.append(
                    {"start": float(turn.start), "end": float(turn.end), "speaker": speaker}
                )
            logger.info("Found %d speaker segments", len(segments))
        except Exception as exc:  # noqa: BLE001
            # `diarization` is initialised above so this block cannot raise a
            # NameError of its own. It used to (bug 2), which replaced the real
            # exception with a misleading one every single time.
            logger.error("Diarization error: %s", exc)
            if diarization is not None:
                logger.error("Diarization object type: %s", type(diarization))
        return segments

    def provenance(self) -> Dict[str, Any]:
        from pipeline.provenance import hf_revision

        block: Dict[str, Any] = {
            "model_id": self.model_id,
            "params": {"loaded": self.pipeline is not None},
        }
        block.update(hf_revision(self.model_id))
        return block
