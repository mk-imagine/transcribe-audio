"""Stage 2: render a raw JSON record into printable transcripts.

    python src/render.py transcripts/x_raw.json --profile coding
    python src/render.py transcripts/x_raw.json --profile lecture --format html,plain \\
        --speaker-map transcripts/x_speakers.yaml --anchor-interval 45 --pause-threshold 0.5

Instant and model-free: everything is a pure function over the JSON (§6).
Plain text (`txt`) is always written; `--format` adds `html` (default) and
`plain`. Exit 2 means the input or the arguments were refused.
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

sys.path.insert(0, str(Path(__file__).resolve().parent))

from pipeline import schema  # noqa: E402
from render import PROFILES, RenderParams, Turn, select_stream, to_rwords  # noqa: E402
from render import segment, speakers, stamp as stampmod  # noqa: E402
from render.formats import html as fmt_html, plain as fmt_plain, txt as fmt_txt  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s",
                    datefmt="%H:%M:%S")
logger = logging.getLogger(__name__)

FORMATS = {"txt": fmt_txt, "html": fmt_html, "plain": fmt_plain}


def prepare(doc: Dict[str, Any], params: RenderParams) -> Tuple[List[Turn], Dict[str, Any]]:
    """The whole of stage 2 short of serialisation: words -> turns + stamp."""
    raw_words, mode_used, warning = select_stream(doc, params.profile.stream)
    warnings = [warning] if warning else []
    words = to_rwords(raw_words)

    speakers.assign(words, doc.get("speaker_turns") or [])
    flipped = speakers.smooth(words, params.pause_threshold, params.max_island) if params.smoothing else 0
    name_map = speakers.load_map(params.speaker_map_path)
    speakers.apply_map(words, name_map)
    if name_map:
        unmapped = sorted({w.speaker for w in words if w.speaker and w.speaker not in name_map.values()
                           and w.speaker not in name_map}, key=str)
        if unmapped:
            warnings.append(f"speaker map has no entry for: {', '.join(unmapped)}")

    turns = segment.segment(words, params.pause_threshold, params.interval)
    st = stampmod.build(
        doc, params, stream_mode=mode_used, words_text=[w.text for w in words],
        warnings=warnings, flipped=flipped, seg_stats=segment.stats(turns),
    )
    return turns, st


def render_all(doc: Dict[str, Any], params: RenderParams, formats: List[str]) -> Dict[str, str]:
    turns, st = prepare(doc, params)
    return {name: FORMATS[name].render(turns, st, params) for name in formats}


def resolve_speaker_map(src: Path, stem: str, explicit: Optional[str], enabled: bool) -> Optional[str]:
    """An explicit --speaker-map wins; otherwise the sidecar beside the input.

    Labels are numbered by first appearance within one diarizer run, so a map
    is per-recording by nature. Keeping it as ``<stem>_speakers.yaml`` next to
    ``<stem>_raw.json`` means the record and its names travel together and the
    flag is not retyped on every re-render. Whichever was used is recorded in
    the stamp, so the output says which names it carries and where they came from.
    """
    if explicit:
        return explicit
    if not enabled:
        return None
    sidecar = src.parent / f"{stem}_speakers.yaml"
    if sidecar.is_file():
        logger.info("using sidecar speaker map: %s", sidecar)
        return str(sidecar)
    return None


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Stage 2: render a raw JSON record.",
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("input", help="a schema-v1 *_raw.json from stage 1")
    p.add_argument("-o", "--output_dir", default=None, help="default: next to the input")
    p.add_argument("--profile", choices=sorted(PROFILES), default="coding")
    p.add_argument("--format", default="html",
                   help="comma-separated extras beyond txt, which is always written: html, plain")
    p.add_argument("--speaker-map", "--speaker_map", dest="speaker_map", default=None,
                   help="file of 'SPEAKER_00: Name' lines, applied at render time. Default: "
                        "<stem>_speakers.yaml beside the input, if it exists")
    p.add_argument("--no-speaker-map", "--no_speaker_map", dest="use_speaker_map",
                   action="store_false", help="ignore a sidecar map; render the anonymous labels")
    p.add_argument("--pause-threshold", "--pause_threshold", dest="pause_threshold",
                   type=float, default=0.5, help="seconds of silence that ends a sentence (default 0.5)")
    p.add_argument("--anchor-interval", "--anchor_interval", dest="anchor_interval",
                   type=float, default=None,
                   help="seconds between within-turn timestamps; default per profile (coding 30, lecture 60)")
    p.add_argument("--max-island", "--max_island", dest="max_island", type=int, default=2,
                   help="longest speaker island smoothing may reassign (default 2 words)")
    p.add_argument("--no-smoothing", "--no_smoothing", dest="smoothing", action="store_false")
    p.add_argument("--no-offset", "--no_offset", dest="apply_offset", action="store_false",
                   help="show excerpt-relative times instead of original-recording times")
    p.add_argument("--width", type=int, default=88, help="text wrap width")
    return p


def main() -> None:
    args = build_parser().parse_args()
    src = Path(args.input)
    try:
        doc = schema.read(src)
    except (OSError, ValueError) as exc:
        logger.error("%s", exc)
        sys.exit(2)

    extras = [f.strip() for f in args.format.split(",") if f.strip()]
    unknown = [f for f in extras if f not in FORMATS]
    if unknown:
        logger.error("unknown format(s): %s; choose from %s", unknown, sorted(FORMATS))
        sys.exit(2)
    formats = ["txt"] + [f for f in extras if f != "txt"]

    stem = src.stem[:-4] if src.stem.endswith("_raw") else src.stem
    speaker_map = resolve_speaker_map(src, stem, args.speaker_map, args.use_speaker_map)

    params = RenderParams(
        profile=PROFILES[args.profile], pause_threshold=args.pause_threshold,
        anchor_interval=args.anchor_interval, smoothing=args.smoothing,
        max_island=args.max_island, speaker_map_path=speaker_map,
        apply_offset=args.apply_offset, width=args.width,
    )
    try:
        outputs = render_all(doc, params, formats)
    except ValueError as exc:  # a malformed speaker map
        logger.error("%s", exc)
        sys.exit(2)

    out_dir = Path(args.output_dir) if args.output_dir else src.parent
    out_dir.mkdir(parents=True, exist_ok=True)
    suffix = {"txt": ".txt", "html": ".html", "plain": "_plain.txt"}
    for name, text in outputs.items():
        path = out_dir / f"{stem}_{args.profile}{suffix[name]}"
        path.write_text(text)
        logger.info("wrote %s (%d bytes)", path, len(text.encode()))


if __name__ == "__main__":
    main()
