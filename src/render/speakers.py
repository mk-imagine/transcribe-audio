"""Speaker assignment, smoothing, and the name map (plan §6, §10).

Stage 1 stores diarizer turns raw and assigns nothing (D3). Assignment happens
here, at render time, so the rule stays tunable and a misattribution is fixed
with a re-render rather than a GPU job.
"""

from __future__ import annotations

import bisect
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from render import RWord

logger = logging.getLogger(__name__)


def assign(words: List[RWord], turns: List[Dict[str, Any]]) -> None:
    """Max-overlap assignment; nearest turn for a word that falls in a gap.

    Diarizer turns do not tile the timeline -- 57 s of 600 were between turns
    in the fixture -- so containment alone leaves every pause-adjacent word
    unattributed. Nearest-turn is the fallback, and smoothing cleans up after.
    Sets ``speaker_raw`` and ``speaker`` (identical at this stage).
    """
    if not turns:
        for w in words:
            w.speaker_raw = w.speaker = None
        return

    ts = sorted(turns, key=lambda t: (t["start"], t["end"]))
    starts = [t["start"] for t in ts]
    max_len = max(t["end"] - t["start"] for t in ts)

    for w in words:
        if w.start is None or w.end is None:
            w.speaker_raw = w.speaker = None
            continue
        # Only turns starting before the word ends can overlap it; turns are
        # bounded in length, so those starting before (w.start - max_len) cannot.
        lo = bisect.bisect_left(starts, w.start - max_len)
        hi = bisect.bisect_right(starts, w.end)
        best, best_ov = None, 0.0
        for t in ts[lo:hi]:
            ov = min(w.end, t["end"]) - max(w.start, t["start"])
            if ov > best_ov:
                best, best_ov = t, ov
        if best is None:
            mid = (w.start + w.end) / 2
            best = min(ts, key=lambda t: min(abs(mid - t["start"]), abs(mid - t["end"])))
        w.speaker_raw = w.speaker = best["speaker"]


def smooth(words: List[RWord], pause_threshold: float, max_island: int = 2) -> int:
    """Flip short speaker islands inside pause-bounded runs. Returns flips made.

    Short filler tokens near a speaker change are where max-overlap is noisiest
    (§10): an "[UH]" straddling a boundary lands on whichever side it overlaps
    more, which is not necessarily who said it. Within a run of speech with no
    pause above the threshold, a stretch of at most ``max_island`` words whose
    speaker differs from *both* neighbours -- and whose neighbours agree -- is
    reassigned to them.

    Deliberately conservative. An island at a run's edge is left alone: it sits
    next to a pause, which is where a real short turn ("Yeah.") would be. A
    run that splits evenly between two speakers with no pause is left alone
    too; there is no evidence to prefer either side.
    """
    if max_island < 1 or len(words) < 3:
        return 0

    # Runs: split where the gap to the next word exceeds the threshold.
    runs: List[List[int]] = [[]]
    for k, w in enumerate(words):
        runs[-1].append(k)
        nxt = words[k + 1] if k + 1 < len(words) else None
        if nxt is not None and w.end is not None and nxt.start is not None \
                and nxt.start - w.end > pause_threshold:
            runs.append([])

    flipped = 0
    for run in runs:
        if len(run) < 3:
            continue
        # Maximal same-speaker stretches within the run.
        stretches: List[Tuple[int, int, Optional[str]]] = []   # (first, last, speaker)
        s0 = run[0]
        for a, b in zip(run, run[1:]):
            if words[b].speaker != words[a].speaker:
                stretches.append((s0, a, words[a].speaker))
                s0 = b
        stretches.append((s0, run[-1], words[run[-1]].speaker))

        for n in range(1, len(stretches) - 1):
            first, last, spk = stretches[n]
            left, right = stretches[n - 1][2], stretches[n + 1][2]
            if last - first + 1 <= max_island and left == right and left != spk:
                for k in range(first, last + 1):
                    words[k].speaker = left
                    flipped += 1
    if flipped:
        logger.info("Smoothing reassigned %d word(s) inside pause-bounded runs", flipped)
    return flipped


def load_map(path: Optional[str]) -> Dict[str, str]:
    """``SPEAKER_00: Dr. Geisler`` per line. A YAML subset, parsed without PyYAML.

    Stage 2 has zero dependencies by design (D11), and a one-key-per-line file
    is all the map needs. Blank lines and ``#`` comments are ignored.
    """
    if not path:
        return {}
    out: Dict[str, str] = {}
    for n, raw in enumerate(Path(path).read_text().splitlines(), 1):
        line = raw.split("#", 1)[0].strip()
        if not line:
            continue
        if ":" not in line:
            raise ValueError(f"{path}:{n}: expected 'SPEAKER_XX: Name', got {raw!r}")
        key, _, value = line.partition(":")
        value = value.strip().strip("\"'")
        if not key.strip() or not value:
            raise ValueError(f"{path}:{n}: empty key or name in {raw!r}")
        out[key.strip()] = value
    return out


def apply_map(words: List[RWord], name_map: Dict[str, str]) -> None:
    if not name_map:
        return
    for w in words:
        if w.speaker in name_map:
            w.speaker = name_map[w.speaker]
