"""Stage 1: audio in, lossless raw JSON out.

Stage 1 is verbatim and lossless (D3). It does no cleaning, no filtering and
no speaker assignment -- anything recomputable from the JSON is recomputed in
stage 2, so changing a pause threshold or fixing a misattributed speaker never
means re-queuing a GPU job.

Model routing is the registry's job (`pipeline/registry.py`), not this file's.
Nothing here inspects a model name to decide what a model can do.

    python src/transcribe_audio.py \
        --input_path data/PSY582-S26-WK9-Thu.wav \
        --output_dir transcripts/ \
        --model nyralabs/CrisperWhisper2.0_large \
        --mode verbatim
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import re
import subprocess
import sys
from datetime import date
from pathlib import Path
from typing import List, Optional

sys.path.insert(0, str(Path(__file__).resolve().parent))

from pipeline import preview, recording, schema  # noqa: E402
from pipeline.orchestrator import CapabilityConflict, transcribe  # noqa: E402
from pipeline.registry import (  # noqa: E402
    UnknownModelError, UnsupportedModelError, known_models,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parent.parent


def _load_env() -> None:
    env_path = PROJECT_ROOT / ".env"
    if not env_path.exists():
        logger.warning("No .env file found at: %s", env_path)
        return
    try:
        from dotenv import load_dotenv
        load_dotenv(dotenv_path=env_path, override=True)
    except ImportError:
        for line in env_path.read_text().splitlines():
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                key, _, value = line.partition("=")
                os.environ.setdefault(key.strip(), value.strip().strip("'\""))
    logger.info("Loaded .env from: %s", env_path)


#: Containers soundfile reads with random access. Anything else is decoded
#: once, up front, to PCM -- bug 9: a compressed source has no random access,
#: so every reader that seeks (librosa per window, pyannote per sliding window)
#: re-decodes it from byte zero. Measured on a 79-minute .m4a: CrisperWhisper
#: took 3 minutes, and pyannote had not finished after 45.
PCM_SUFFIXES = {".wav", ".flac", ".aiff", ".aif"}


def needs_transcode(path: Path) -> bool:
    return path.suffix.lower() not in PCM_SUFFIXES


class AudioHandler:
    """Hands every later stage a PCM .wav: an excerpt when --start_time is
    given, a one-time decode when the source is compressed, the file itself
    otherwise."""

    def __init__(self, output_dir: Path):
        self.output_dir = output_dir
        self.temp_files: List[Path] = []
        self.transcoded_from: Optional[str] = None

    def prepare_segment(self, input_path: Path, start: Optional[str],
                        end: Optional[str]) -> Path:
        if input_path.is_dir():
            raise IsADirectoryError(f"Input path is a directory: {input_path}")
        if not input_path.exists():
            raise FileNotFoundError(f"Audio file not found: {input_path}")
        transcode = needs_transcode(input_path)
        if not start and not transcode:
            return input_path

        # Always a .wav container: the output is PCM s16le, and reusing the
        # source suffix wrote PCM into e.g. .m4a, which the MP4 muxer rejects.
        tag = f"segment_{start.replace(':', '')}" if start else "decoded"
        temp_path = self.output_dir / f"temp_{tag}_{input_path.stem}.wav"
        if start:
            logger.info("Creating temporary audio segment: %s to %s...", start, end or "EOF")
        else:
            logger.info("Decoding %s to PCM once, so no reader re-decodes it per window (bug 9)...",
                        input_path.suffix)
            self.transcoded_from = input_path.suffix.lower()

        command = ["ffmpeg", "-y", "-i", str(input_path)]
        if start:
            command += ["-ss", str(start)]
        if end:
            command += ["-to", str(end)]
        command += ["-c:a", "pcm_s16le", str(temp_path)]
        try:
            subprocess.run(command, check=True, stdout=subprocess.DEVNULL,
                           stderr=subprocess.PIPE, text=True)
            if not temp_path.exists() or temp_path.stat().st_size < 1000:
                raise RuntimeError("ffmpeg created an empty/invalid file.")
        except FileNotFoundError:
            logger.error("ffmpeg not found on PATH; it is required for --start_time.")
            raise
        except subprocess.CalledProcessError as exc:
            detail = (exc.stderr or "").strip().splitlines()[-8:]
            logger.error("ffmpeg failed creating the segment:\n%s", "\n".join(detail))
            raise
        self.temp_files.append(temp_path)
        return temp_path

    def cleanup(self) -> None:
        for path in self.temp_files:
            try:
                path.unlink(missing_ok=True)
                logger.info("Removed temporary file: %s", path.name)
            except OSError:
                pass


def load_hotwords(spec: Optional[str]) -> List[str]:
    """A comma-separated list, or a path to a newline-delimited file."""
    if not spec:
        return []
    path = Path(spec)
    lines = path.read_text().splitlines() if path.is_file() else spec.split(",")
    terms = [t.strip() for t in lines if t.strip()]
    if terms:
        logger.info("Loaded %d hotword term(s)", len(terms))
    return terms


def output_basename(args) -> str:
    """<audio stem>-<recorded date>[_seg_<start>][_<tag>].

    The original file's name, never the working excerpt's: with --start_time
    the excerpt is a temp file, and naming outputs after it produced
    "temp_segment_004500_geisler_seg_004500_jobverbatim_raw.json".

    The date is when the recording was made, from its own metadata, so a
    transcript can be found by the day it was recorded. The SLURM job id is no
    longer part of the name: it is in the record (``run.slurm_job_id``), and in
    a file name it said which run made a transcript, which matters less than
    what was recorded and when. A name that already holds the date, such as
    PhD-Applications-Discussion-2026-10-07, does not get it twice.
    """
    name = Path(getattr(args, "source_path", None) or args.input_path).stem
    day = (getattr(args, "recorded", None) or {}).get("date")
    if day and day not in name:
        name += f"-{day}"
    if args.start_time:
        name += f"_seg_{args.start_time.replace(':', '')}"
    if args.tag:
        name += f"_{args.tag}"
    return name


def _tag(value: str) -> str:
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]*", value):
        raise argparse.ArgumentTypeError(
            f"{value!r}: use letters, digits, '.', '_' and '-'; it goes into a file name")
    return value


def _iso_date(value: str) -> str:
    try:
        return date.fromisoformat(value).isoformat()
    except ValueError:
        raise argparse.ArgumentTypeError(f"{value!r} is not a YYYY-MM-DD date") from None


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Stage 1: transcribe audio to lossless raw JSON.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="Registered models:\n  " + "\n  ".join(known_models()),
    )
    p.add_argument("-i", "--input_path", required=True)
    p.add_argument("-o", "--output_dir", default="./transcripts")
    p.add_argument("--model", default="nyralabs/CrisperWhisper2.0_large",
                   help="A registered checkpoint id. Names are matched exactly; "
                        "an unknown id is refused rather than guessed at.")
    p.add_argument("--adapter", default=None,
                   help="Force an adapter for an unregistered checkpoint. "
                        "Explicit, and recorded in the output's provenance.")
    p.add_argument("--backend", default=None, choices=["auto", "ct2", "transformers"],
                   help="CrisperWhisper inference backend. ct2 is ~4-5x faster.")
    p.add_argument("--compute_type", default=None,
                   choices=["float16", "int8_float16", "int8", "float32"],
                   help="CrisperWhisper/ct2 numeric type. Default: float16 on "
                        "CUDA, float32 on CPU (ct2 refuses float16 on CPU).")
    p.add_argument("--device", default=None, choices=["cuda", "cpu"],
                   help="Default: cuda when available, else cpu. MPS is not a "
                        "target (D13).")
    p.add_argument("--diarizer_model", default="pyannote/speaker-diarization-community-1")
    p.add_argument("--hf_token", default=None)
    p.add_argument("--no_diarize", action="store_true")
    p.add_argument("--no_timestamps", action="store_true",
                   help="Omit timestamps from the preview text. The JSON always "
                        "keeps them.")
    p.add_argument("--timestamp_mode", choices=["word", "chunk"], default="word",
                   help="Generic Whisper only. 'chunk' is a VRAM concession for "
                        "8GB cards, where word-level alignments OOM; it coarsens "
                        "the record and is written to asr.granularity.")
    # No argparse default: the registry supplies the per-checkpoint default
    # (verbatim for CrisperWhisper 2), so a model without modes is not refused
    # over one the user never asked for.
    p.add_argument("--mode", choices=["verbatim", "intended"], default=None,
                   help="CrisperWhisper 2. 'verbatim' (its registry default) keeps "
                        "disfluencies; 'intended' returns cleaned text.")
    # Both renderings of the same audio. On the ct2 backend the two modes share
    # one encoder pass and one batched decode, so the second transcript is
    # nearly free; elsewhere it falls back to two sequential passes, which
    # produce the same thing for twice the work (D12).
    p.add_argument("--dual_stream", "--dual-stream", action="store_true",
                   help="Emit both the verbatim and intended streams, each with "
                        "word timestamps. Requires a model whose mode is "
                        "selectable.")
    p.add_argument("--hotwords", default=None,
                   help="Comma-separated terms, or a file with one per line. "
                        "Trained into the Pro checkpoints only; on a standard "
                        "checkpoint the run warns and records the warning.")
    # Retained because transcribe.slurm always passes it, and removing it would
    # break the wrapper's argument contract. It no longer does anything: stage 1
    # is lossless (D3), and disfluencies are data to be tagged in annotate.py,
    # not deleted before they are ever written to disk (bug 4).
    p.add_argument("--clean_mode", choices=["none", "basic", "intelligent"], default="none",
                   help="Accepted and ignored. Stage 1 is always verbatim; "
                        "cleaning moves to annotate.py as a tagger.")
    p.add_argument("--start_time", default=None)
    p.add_argument("--end_time", default=None)
    # Replaces --job_id, which named outputs <stem>_job<id> and was also used
    # to keep dissenter runs apart (--job_id ${M##*/}). The job id is in the
    # record now; telling runs of one recording apart is this flag's job.
    p.add_argument("--tag", type=_tag, default=None,
                   help="Appended to the output name, <name>_<tag>_raw.json, to keep "
                        "runs of one recording apart, e.g. a dissenter's model name.")
    p.add_argument("--recorded_date", type=_iso_date, default=None,
                   help="YYYY-MM-DD. When the recording was made, for a file whose "
                        "metadata does not say (WAVs from before the recorder was set "
                        "to BWF). Overrides the metadata; both are kept in the record.")
    p.add_argument("--overwrite", action="store_true",
                   help="Replace an existing record of the same name. Without it the "
                        "run is refused before any work (exit 2).")
    return p


def main() -> None:
    _load_env()
    args = build_parser().parse_args()
    if not args.hf_token:
        args.hf_token = os.getenv("HF_TOKEN")
    args.hotwords_list = load_hotwords(args.hotwords)

    if args.clean_mode != "none":
        logger.warning(
            "--clean_mode %s is accepted and ignored. Stage 1 is verbatim and "
            "lossless: cleaning here destroyed information before it was ever "
            "written to disk, and chunks that cleaned to empty were dropped, "
            "punching holes in the timeline (D3, bugs 4 and 5). Disfluencies "
            "are tagged in annotate.py, never deleted.", args.clean_mode,
        )

    # The record names the original audio; input_path becomes the working
    # excerpt or decode, which is deleted when the run ends. The date is read
    # from the original too: an excerpt or a decode carries no metadata.
    args.source_path = args.input_path
    found = recording.from_metadata(Path(args.source_path))
    if args.recorded_date:
        args.recorded = recording.given_by_hand(args.recorded_date, found)
        if found["date"] and found["date"] != args.recorded["date"]:
            logger.warning("--recorded_date %s overrides the file's own date, %s (from %s). "
                           "Both are kept in the record.", args.recorded["date"],
                           found["date"], found["from"])
    else:
        args.recorded = found
        if found["date"]:
            logger.info("Recorded %s %s (from %s)", found["date"], found.get("start") or "", found["from"])
        else:
            logger.warning("No recording date: %s. The output name will not carry one; "
                           "pass --recorded_date YYYY-MM-DD to supply it.", found["why"])

    output_dir = Path(args.output_dir)
    base = output_basename(args)
    # Checked before any work. The job id used to keep every run's outputs
    # apart; without it a second run of the same recording would overwrite
    # the first record after hours on a GPU.
    record_path = output_dir / f"{base}_raw.json"
    if record_path.exists() and not args.overwrite:
        logger.error("%s already exists. Refusing before any work: pass --overwrite to "
                     "replace it, --tag to write beside it, or another --output_dir.", record_path)
        sys.exit(2)
    output_dir.mkdir(parents=True, exist_ok=True)
    handler = AudioHandler(output_dir)

    try:
        args.input_path = str(
            handler.prepare_segment(Path(args.input_path), args.start_time, args.end_time)
        )
        args.transcoded_from = handler.transcoded_from
        try:
            doc = transcribe(args)
        except (UnknownModelError, UnsupportedModelError, CapabilityConflict) as exc:
            logger.error("%s", exc)
            sys.exit(2)
    finally:
        handler.cleanup()

    schema.write(record_path, doc)
    txt_path = output_dir / f"{base}_preview.txt"
    txt_path.write_text(preview.render(doc, timestamps=not args.no_timestamps))
    logger.info("Preview written to: %s", txt_path)

    if doc.get("secondary_stream"):
        mode = doc["secondary_stream"]["mode"]
        second = output_dir / f"{base}_{mode}_preview.txt"
        second.write_text(
            preview.render(doc, timestamps=not args.no_timestamps, secondary=True)
        )
        logger.info("Second stream (%s) written to: %s", mode, second)

    # A transcript with holes in it is not a successful run. The outputs are
    # still written -- the error ranges keep the timeline intact -- but the exit
    # status has to say so, or a scheduler reports the job clean while audio is
    # missing (bug 13: one lecture lost 12.7% this way).
    if doc["errors"]:
        logger.error(
            "Exiting non-zero: %d audio range(s) could not be transcribed.",
            len(doc["errors"]),
        )
        sys.exit(1)


if __name__ == "__main__":
    main()
