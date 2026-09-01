"""Fixed-window segmentation for models that declare ``longform='needs_chunking'``.

CrisperWhisper 2 does not come through here -- it declares ``longform='native'``
and its own continuation strategy carries decoder context across window edges,
which is both better and faster.

Two known defects are preserved rather than papered over, because fixing them
needs a real ``needs_chunking`` model to verify against (Phase 3):

* **bug 3** -- boundaries are fixed at ``SEGMENT_SIZE`` and cut mid-word. They
  should split on silence.
* **bug 9** -- each window calls ``librosa.load(path, offset=...)``, which
  re-decodes a compressed source from byte zero. Measured: 79 min of ``.m4a``
  took 91 min against 57 min for the same audio as ``.wav``.
"""

from __future__ import annotations

import logging
from typing import Callable, List, Tuple

from pipeline.adapters.base import ErrorRange, Word

logger = logging.getLogger(__name__)

SEGMENT_SIZE = 300.0
# Generation can fail on a window and succeed on its halves -- the trigger is
# where the model's internal 30s boundaries land, not the audio. Subdividing
# moves them, so a failed window is retried before being written off (bug 13:
# one lecture lost 12.7% of its audio to windows that were simply dropped).
MAX_RETRY_DEPTH = 3
MIN_RETRY_WINDOW = 30.0

WindowFn = Callable[[float, float], List[Word]]


def transcribe_windows(
    window_fn: WindowFn,
    duration: float,
    segment_size: float = SEGMENT_SIZE,
) -> Tuple[List[Word], List[ErrorRange]]:
    """Run ``window_fn(start, end)`` over the file, halving a window on failure.

    Returns absolute-time words and the ranges no subdivision could transcribe.
    A non-empty error list means the transcript has holes and the run must not
    report success.
    """
    words: List[Word] = []
    errors: List[ErrorRange] = []

    if duration <= segment_size:
        bounds = [(0.0, duration)]
    else:
        logger.info("Audio is %.1fs; processing in %.0fs windows.", duration, segment_size)
        bounds = []
        start = 0.0
        while start < duration:
            bounds.append((start, min(start + segment_size, duration)))
            start += segment_size

    for start, end in bounds:
        w, e = _window_with_retry(window_fn, start, end, 0)
        words.extend(w)
        errors.extend(e)

    if errors:
        lost = sum(e.end - e.start for e in errors)
        pct = (lost / duration * 100) if duration else 0.0
        logger.error(
            "INCOMPLETE TRANSCRIPT: %d range(s), %.0fs of %.0fs missing (%.1f%%)",
            len(errors), lost, duration, pct,
        )
        for e in errors:
            logger.error(
                "    missing %.0fs - %.0fs  (%.1fmin - %.1fmin)",
                e.start, e.end, e.start / 60, e.end / 60,
            )
    return words, errors


def _window_with_retry(
    window_fn: WindowFn, start: float, end: float, depth: int
) -> Tuple[List[Word], List[ErrorRange]]:
    try:
        words = window_fn(start, end)
    except Exception as exc:  # noqa: BLE001 - recorded and retried, never swallowed
        span = end - start
        if depth < MAX_RETRY_DEPTH and span / 2 >= MIN_RETRY_WINDOW:
            mid = start + span / 2
            logger.warning(
                "Window %.0fs-%.0fs failed (%s); retrying as two %.0fs halves",
                start, end, exc, span / 2,
            )
            lw, le = _window_with_retry(window_fn, start, mid, depth + 1)
            rw, re_ = _window_with_retry(window_fn, mid, end, depth + 1)
            return lw + rw, le + re_
        logger.error("Window %.0fs-%.0fs unrecoverable: %s", start, end, exc)
        return [], [ErrorRange(start, end, f"{type(exc).__name__}: {exc}")]

    # Window-relative times become absolute. A missing bound stays missing --
    # substituting the window's own end (bug 6) invents a timestamp and makes
    # the gap unfindable downstream.
    for w in words:
        if w.start is not None:
            w.start += start
        if w.end is not None:
            w.end += start
    logger.info("Window %.0fs-%.0fs completed (%d tokens)", start, end, len(words))
    return words, []
