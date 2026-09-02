"""The version stamp every rendered transcript carries (plan §6, Render stability).

Line numbers are stable only while the render is: re-render with a different
pause threshold and "line 47" is a different sentence. In multi-coder work
where inter-rater reliability assumes everyone coded the same text, that is a
correctness problem. So every output states what produced it -- model,
revision, render parameters, date, and a short hash of the words -- and a
printout and the current render visibly differ.
"""

from __future__ import annotations

import hashlib
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from render import RenderParams


def content_hash(words_text: List[str]) -> str:
    return hashlib.sha256(" ".join(words_text).encode("utf-8")).hexdigest()[:12]


def build(doc: Dict[str, Any], params: RenderParams, *, stream_mode: str,
          words_text: List[str], warnings: List[str], flipped: int,
          seg_stats: Dict[str, Any]) -> Dict[str, Any]:
    asr = doc.get("asr") or {}
    dia = doc.get("diarization") or {}
    src = doc.get("source") or {}
    run = doc.get("run") or {}
    excerpt = src.get("excerpt") or {}
    offset = float(excerpt.get("offset_s") or 0.0) if params.apply_offset else 0.0
    pv = run.get("pipeline_version") or {}
    return {
        "profile": params.profile.name,
        "source": src.get("audio_path"),
        "audio_sha256": (src.get("audio_sha256") or "")[:12] or None,
        "excerpt": (f"{excerpt.get('start_time')}–{excerpt.get('end_time') or 'end'}"
                    if excerpt else None),
        "offset_s": offset,
        "model": asr.get("model_id"),
        "revision": (asr.get("revision") or "")[:12] or None,
        "mode": stream_mode,
        "diarizer": dia.get("model_id"),
        "diarizer_revision": (dia.get("revision") or "")[:12] or None,
        "pipeline_commit": (pv.get("commit") or "")[:9] or None,
        "pipeline_dirty": pv.get("dirty"),
        "stage1_job": run.get("slurm_job_id"),
        "render": {
            "pause_threshold_s": params.pause_threshold,
            "anchor_interval_s": params.interval,
            "smoothing": params.smoothing,
            "max_island": params.max_island if params.smoothing else None,
            "words_reassigned": flipped,
            "speaker_map": params.speaker_map_path,
        },
        "stats": seg_stats,
        "rendered_utc": datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z"),
        "content_hash": content_hash(words_text),
        "warnings": list(warnings),
    }
