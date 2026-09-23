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
import time
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
            # Decode once and hand pyannote the samples, not the path. Given a
            # path, its embedding step opens a fresh decoder for every chunk it
            # crops -- one per second of audio, ~340 ms each on a 145 min WAV,
            # which is most of diarization's wall time with the GPU idle. Given
            # a waveform, each crop is a slice (~0.04 ms). Audio() keeps the
            # file's own rate and channels, so the pipeline still downmixes and
            # resamples chunk by chunk exactly as before: the same samples reach
            # the models either way.
            from pyannote.audio import Audio

            started = time.perf_counter()
            waveform, sample_rate = Audio()(str(audio_path))
            logger.info(
                "Diarization audio decoded once: %d ch x %d samples at %d Hz (%.0f MB) in %.1fs",
                waveform.shape[0], waveform.shape[1], sample_rate,
                waveform.element_size() * waveform.nelement() / 1e6,
                time.perf_counter() - started,
            )
            diarization = self.pipeline(
                {"waveform": waveform, "sample_rate": sample_rate, "uri": Path(audio_path).stem}
            )
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
