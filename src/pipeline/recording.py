"""When a recording was made, read from the file's own metadata.

Two fields are trusted, each checked against this project's own recordings on
2026-10-19:

* **BWF ``bext``** (the Tascam DR-40X set to BWF; every WAV from 2026-09-23
  on). OriginationDate and OriginationTime are the recorder's local clock, and
  on the DR-40X they mark when the file was *closed*. TimeReference is the
  first sample's offset from midnight, so it gives the start. A recording that
  runs past midnight therefore started the day before its OriginationDate.
* **MP4 ``moov/udta/date``** (iPad Voice Memos ``.m4a``). The start, in UTC:
  ``2026-08-26T21:10:25Z``. It is converted to RECORDING_TZ before the date is
  taken, because an evening recording is already tomorrow in UTC.

Nothing else is trusted. A file's modification time is reset by a copy. MP4
``mvhd`` creation_time means different things to different producers: Voice
Memos writes the save time there, 91 minutes after the start. WAVs recorded
before the DR-40X was set to BWF carry no date at all, so they get none here;
``--recorded_date`` supplies one by hand.

Pure standard library: the rename script and the contract checks run it on the
Mac, where nothing is installed.
"""

from __future__ import annotations

import re
import struct
from dataclasses import dataclass
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Any, Dict, Optional, Tuple
from zoneinfo import ZoneInfo

#: Where the recordings are made. Only used to turn a UTC timestamp (MP4) into
#: a local date; the BWF clock is already local.
RECORDING_TZ = "America/Los_Angeles"

#: EBU Tech 3285 allows any separator in OriginationDate/Time ("2026-09-29",
#: "2026:09:29"), recommending - _ : space or full stop.
_DATE = re.compile(r"(\d{4})\D(\d{2})\D(\d{2})")
_TIME = re.compile(r"(\d{2})\D(\d{2})\D(\d{2})")


@dataclass(frozen=True)
class Bwf:
    """The bext fields, plus the fmt and data sizes needed to place them in time."""
    origination_date: str           # as written, e.g. "2026-09-29"
    origination_time: str           # as written, e.g. "11:22:04"
    time_reference: int             # the first sample's offset from midnight, in samples
    sample_rate: Optional[int]
    byte_rate: Optional[int]
    data_bytes: Optional[int]


def read_bwf(path: Path) -> Optional[Bwf]:
    """The file's bext chunk, or None when it has none or is not a WAV.

    A missing file raises: "no bext" and "no file" are different answers.
    """
    found = None
    rate = byte_rate = data_bytes = None
    with open(path, "rb") as f:
        head = f.read(12)
        if len(head) < 12 or head[:4] != b"RIFF" or head[8:] != b"WAVE":
            return None
        while True:
            head = f.read(8)
            if len(head) < 8:
                break
            cid, size = struct.unpack("<4sI", head)
            body_at = f.tell()
            if cid == b"bext" and size >= 346:
                f.seek(body_at + 320)  # past Description, Originator, OriginatorReference
                raw = f.read(26)
                if len(raw) == 26:
                    found = (raw[:10], raw[10:18], struct.unpack("<Q", raw[18:])[0])
            elif cid == b"fmt " and size >= 16:
                _, _, rate, byte_rate = struct.unpack("<HHII", f.read(12))
            elif cid == b"data":
                data_bytes = size
            f.seek(body_at + size + (size & 1))
    if found is None:
        return None
    date_raw, time_raw, tref = found
    return Bwf(_text(date_raw), _text(time_raw), tref, rate, byte_rate, data_bytes)


def read_mp4_date(path: Path) -> Optional[str]:
    """The text of ``moov/udta/date``, or None when the file has none or is not MP4."""
    with open(path, "rb") as f:
        f.seek(0, 2)
        end = f.tell()
        f.seek(0)
        head = f.read(8)
        if len(head) < 8 or head[4:] != b"ftyp":
            return None
        box: Optional[Tuple[int, int]] = (0, end)
        for name in (b"moov", b"udta", b"date"):
            box = _child(f, *box, name)
            if box is None:
                return None
        f.seek(box[0])
        return _text(f.read(box[1] - box[0]))


def _child(f, start: int, end: int, want: bytes) -> Optional[Tuple[int, int]]:
    """The body (start, end) of the first `want` box between start and end."""
    pos = start
    while pos + 8 <= end:
        f.seek(pos)
        size, kind = struct.unpack(">I4s", f.read(8))
        header = 8
        if size == 1:  # 64-bit size follows
            size = struct.unpack(">Q", f.read(8))[0]
            header = 16
        elif size == 0:  # runs to the end of its parent
            size = end - pos
        if size < header:
            return None  # corrupt; stop rather than loop
        if kind == want:
            return pos + header, min(pos + size, end)
        pos += size
    return None


def _text(raw: bytes) -> str:
    return raw.split(b"\0")[0].decode("ascii", "replace").strip()


def _clock(seconds: float) -> str:
    s = int(seconds)
    return f"{s // 3600:02d}:{s // 60 % 60:02d}:{s % 60:02d}"


def _from_bwf(b: Bwf) -> Dict[str, Any]:
    m = _DATE.fullmatch(b.origination_date)
    try:
        day = date(*map(int, m.groups())) if m else None
    except ValueError:
        day = None
    if day is None:
        return {"date": None, "from": None,
                "why": f"bext OriginationDate is {b.origination_date!r}, not a date"}
    start = None
    if b.time_reference and b.sample_rate:
        start_s = b.time_reference / b.sample_rate
        start = _clock(start_s)
        t = _TIME.fullmatch(b.origination_time)
        if t:
            h, mi, s = map(int, t.groups())
            # Started late in the evening, closed after midnight. The minute of
            # slack absorbs a recorder that writes the start time here instead,
            # rounded down to the second.
            if start_s - (h * 3600 + mi * 60 + s) > 60:
                day -= timedelta(days=1)
    return {"date": day.isoformat(), "start": start, "from": "bext",
            "clock": "recorder's local time"}


def _from_mp4(text: str) -> Dict[str, Any]:
    try:
        when = datetime.fromisoformat(text.replace("Z", "+00:00"))
    except ValueError:
        when = None
    if when is None or when.tzinfo is None:
        return {"date": None, "from": None,
                "why": f"mp4 udta date is {text!r}, not a timestamp with a UTC offset"}
    local = when.astimezone(ZoneInfo(RECORDING_TZ))
    return {"date": local.date().isoformat(), "start": local.strftime("%H:%M:%S"),
            "from": "mp4 udta date", "as_written": text, "tz": RECORDING_TZ}


def from_metadata(path: Path) -> Dict[str, Any]:
    """The ``source.recorded`` block: the date and start time the recording
    began, and which field they came from. ``date`` is None when the file does
    not say, with ``why``.

    The start is the recording's, not an excerpt's: with --start_time the
    words are timed from the excerpt, but the recording began here.
    """
    bwf = read_bwf(path)
    if bwf is not None:
        return _from_bwf(bwf)
    text = read_mp4_date(path)
    if text is not None:
        return _from_mp4(text)
    return {"date": None, "from": None,
            "why": "the file has no BWF bext chunk and no MP4 udta date"}


def given_by_hand(day: str, found: Dict[str, Any]) -> Dict[str, Any]:
    """``--recorded_date``. It wins over the metadata, which is kept beside it,
    so a disagreement stays visible in the record."""
    return {"date": date.fromisoformat(day).isoformat(), "start": None,
            "from": "--recorded_date", "metadata": found}
