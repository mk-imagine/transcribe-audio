"""Stage 1: audio in, lossless raw JSON out.

Stage 1 is verbatim and lossless (D3). It does no cleaning, no filtering and
no speaker assignment -- anything recomputable from the JSON is recomputed in
stage 2, so changing a pause threshold or fixing a misattributed speaker never
means re-queuing a GPU job.

Model routing is the registry's job (`pipeline/registry.py`), not this file's.
Nothing here inspects a model name to decide what a model can do.

    python src/transcribe_audio.py \
        --input_path data/geisler.wav \
        --output_dir transcripts/ \
        --model nyralabs/CrisperWhisper2.0_large \
        --mode verbatim
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import List, Optional

sys.path.insert(0, str(Path(__file__).resolve().parent))

from pipeline import preview, schema  # noqa: E402
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


class AudioHandler:
    """Cuts a working excerpt with ffmpeg when --start_time is given."""

    def __init__(self, output_dir: Path):
        self.output_dir = output_dir
        self.temp_files: List[Path] = []

    def prepare_segment(self, input_path: Path, start: Optional[str],
                        end: Optional[str]) -> Path:
        if input_path.is_dir():
            raise IsADirectoryError(f"Input path is a directory: {input_path}")
        if not input_path.exists():
            raise FileNotFoundError(f"Audio file not found: {input_path}")
        if not start:
            return input_path

        # Always a .wav container: the segment is re-encoded as PCM s16le, and
        # reusing the source suffix wrote PCM into e.g. .m4a, which the MP4
        # muxer rejects outright.
        temp_path = self.output_dir / f"temp_segment_{start.replace(':', '')}_{input_path.stem}.wav"
        logger.info("Creating temporary audio segment: %s to %s...", start, end or "EOF")

        command = ["ffmpeg", "-y", "-i", str(input_path), "-ss", str(start)]
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
    # The original file's name, never the working excerpt's: with --start_time
    # the excerpt is a temp file, and naming outputs after it produced
    # "temp_segment_004500_geisler_seg_004500_jobverbatim_raw.json".
    name = Path(getattr(args, "source_path", None) or args.input_path).stem
    if args.start_time:
        name += f"_seg_{args.start_time.replace(':', '')}"
    if args.job_id:
        name += f"_job{args.job_id}"
    else:
        name += datetime.now().strftime("_%Y%m%d-%H%M%S")
    return name


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
    p.add_argument("--hotwords", default=None,
                   help="Comma-separated terms, or a file with one per line. "
                        "Trained into the Pro checkpoints only; on a standard "
                        "checkpoint the run warns and records the warning.")
    p.add_argument("--clean_mode", choices=["none", "basic", "intelligent"], default="none",
                   help="Preview text only. The JSON is always verbatim (D3); "
                        "tagging disfluencies is annotate.py's job, not this one's.")
    p.add_argument("--start_time", default=None)
    p.add_argument("--end_time", default=None)
    p.add_argument("--job_id", default=None)
    return p


def main() -> None:
    _load_env()
    args = build_parser().parse_args()
    if not args.hf_token:
        args.hf_token = os.getenv("HF_TOKEN")
    args.hotwords_list = load_hotwords(args.hotwords)

    if args.clean_mode != "none":
        logger.warning(
            "--clean_mode %s affects the preview text only. Stage 1's JSON is "
            "always verbatim: cleaning here would destroy information before it "
            "was ever written to disk (D3, bug 4).", args.clean_mode,
        )

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    handler = AudioHandler(output_dir)

    try:
        # The record names the original audio; input_path becomes the working
        # excerpt, which is deleted when the run ends.
        args.source_path = args.input_path
        args.input_path = str(
            handler.prepare_segment(Path(args.input_path), args.start_time, args.end_time)
        )
        try:
            doc = transcribe(args)
        except (UnknownModelError, UnsupportedModelError, CapabilityConflict) as exc:
            logger.error("%s", exc)
            sys.exit(2)
    finally:
        handler.cleanup()

    base = output_basename(args)
    schema.write(output_dir / f"{base}_raw.json", doc)
    txt_path = output_dir / f"{base}_preview.txt"
    txt_path.write_text(preview.render(doc, timestamps=not args.no_timestamps))
    logger.info("Preview written to: %s", txt_path)

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
