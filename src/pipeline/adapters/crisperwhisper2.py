"""CrisperWhisper 2, driven through the `crisperwhisper` package (D17).

Deliberately not the transformers ASR pipeline. That pipeline has no ``mode``
parameter, so verbatim output cannot be requested through it: the same weights
return ordinary cleaned text and the model looks like any other ASR. Measured
on identical audio, the package emits 121 filled pauses where the pipeline
emitted zero, at roughly 3x the speed.

The load path *verifies* the checkpoint rather than trusting its name. The
package ships ``detect_model_version()``, which reads the tokenizer's
``[verbatim_1]`` marker without loading a model, so a v1 checkpoint is refused
here instead of running to completion with ``mode`` and ``hotwords`` silently
ignored -- which is what bug 7 did.
"""

from __future__ import annotations

import logging
import warnings
from pathlib import Path
from typing import Any, List, Optional, Union

from pipeline.adapters.base import Adapter, AdapterResult, ErrorRange, Word
from pipeline.capabilities import Capabilities

logger = logging.getLogger(__name__)


class CrisperWhisper2Adapter(Adapter):
    capabilities = Capabilities(
        word_timestamps="start_end",
        verbatim="selectable",
        speaker_labels=False,
        silence_tokens=False,
        longform="native",
        confidence=False,
        # Standard checkpoints were never trained with hotword prompts. The
        # package accepts the argument and raises only a UserWarning, so the
        # run succeeds either way -- see the class docstring of the Pro variant.
        hotwords="untrained",
    )

    def __init__(self, model_id: str, device, *, mode: str = "verbatim",
                 hotwords: Optional[List[str]] = None, backend: str = "auto",
                 language: str = "en", **options: Any):
        super().__init__(model_id, device, **options)
        self.mode = mode
        self.hotwords = list(hotwords or [])
        self.backend = backend
        self.language = language
        self.model = None
        self._resolved_backend: Optional[str] = None

    # -- load ---------------------------------------------------------------

    def load(self) -> None:
        try:
            from crisperwhisper import CrisperWhisperModel
            from crisperwhisper.version import detect_model_version
        except ImportError as exc:
            raise RuntimeError(
                "CrisperWhisper requires the 'crisperwhisper' package "
                "(pip install 'crisperwhisper[ct2]'). Running these weights "
                "through transformers.pipeline silently disables verbatim mode."
            ) from exc

        # Check before loading: cheap, and it turns a silent misroute into a
        # refusal. detect_model_version falls back to "assume v2" when it cannot
        # read the tokenizer at all, so say which way the answer was reached.
        version = detect_model_version(self.model_id)
        if version != 2:
            raise RuntimeError(
                f"{self.model_id!r} is a CrisperWhisper v{version} checkpoint, "
                "but this adapter declares verbatim='selectable' and "
                "hotword support. v1 ignores both and emits no disfluency "
                "markers while still exiting 0. Use "
                "nyralabs/CrisperWhisper2.0_large."
            )

        device_str = "cuda" if str(self.device).startswith("cuda") else "cpu"
        logger.info(
            "Loading CrisperWhisper 2: %s on %s (mode=%s, hotwords=%d, backend=%s)",
            self.model_id, device_str, self.mode, len(self.hotwords), self.backend,
        )
        kwargs = {"device": device_str}
        if self.backend and self.backend != "auto":
            kwargs["backend"] = self.backend
        self.model = CrisperWhisperModel(self.model_id, **kwargs)
        self._resolved_backend = getattr(self.model, "backend", None) or getattr(
            self.model, "_backend", None
        )
        logger.info("CrisperWhisper backend in use: %s", self._resolved_backend)

    # -- transcribe ---------------------------------------------------------

    def transcribe(self, audio_path: Union[str, Path], duration: float) -> AdapterResult:
        assert self.model is not None, "load() must be called before transcribe()"

        params = {
            "language": self.language,
            "mode": self.mode,
            "word_timestamps": True,
            "hotwords": self.hotwords or None,
            # Package defaults, recorded explicitly so the provenance block says
            # what ran rather than what the version installed that day defaulted to.
            "longform_strategy": "continuation",
            "temperature_fallback": True,
            "hallucination_mitigation": True,
        }
        logger.info("Transcribing (native long-form, %.0fs)...", duration)

        # The package signals "hotwords are untrained on this checkpoint" with a
        # UserWarning. Warnings go to stderr and vanish; capture it so it reaches
        # the log and the output record instead.
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            try:
                result = self.model.transcribe(str(audio_path), **params)
            except Exception as exc:  # noqa: BLE001 - recorded, then re-raised upward
                logger.error("CrisperWhisper failed on the whole file: %s", exc)
                return AdapterResult(
                    errors=[ErrorRange(0.0, duration, f"{type(exc).__name__}: {exc}")],
                    params=params,
                    backend=self._resolved_backend,
                )

        model_warnings = []
        for w in caught:
            text = f"{w.category.__name__}: {w.message}"
            model_warnings.append(text)
            logger.warning("crisperwhisper: %s", text)
        if model_warnings:
            params["model_warnings"] = model_warnings

        words = getattr(result, "words", None) or []
        text = (getattr(result, "text", "") or "").strip()

        if not words:
            # Text with no per-word times is still content, but the timeline is
            # gone -- record that rather than inventing one.
            logger.warning("No word timestamps returned; recording text only.")
            return AdapterResult(
                words=[], text=text, params=params, backend=self._resolved_backend,
                errors=[] if text else [
                    ErrorRange(0.0, duration, "model returned neither words nor text")
                ],
            )

        out = [
            Word(text=w.word.strip(), start=w.start, end=w.end)
            for w in words
            if getattr(w, "word", "").strip()
        ]
        logger.info("  %d word tokens over %.0fs", len(out), duration)
        return AdapterResult(
            words=out,
            text=text or " ".join(w.text for w in out),
            params=params,
            backend=self._resolved_backend,
        )


class CrisperWhisper2ProAdapter(CrisperWhisper2Adapter):
    """The Pro checkpoints, which *are* trained with hotword prompts.

    Same code path; the declaration differs, which is the whole point of the
    contract. Commercial licence only -- registered so a Pro id resolves
    correctly rather than inheriting the standard model's `untrained` warning.
    """

    capabilities = Capabilities(
        word_timestamps="start_end",
        verbatim="selectable",
        speaker_labels=False,
        silence_tokens=False,
        longform="native",
        confidence=False,
        hotwords="trained",
    )
