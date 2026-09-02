"""Stage 1.5: enrich a raw record with proper-noun candidates from dissenter runs (plan §7b).

    python src/annotate.py transcripts/x_raw.json \\
        --dissenters transcripts/x_ark_raw.json transcripts/x_tdt_raw.json transcripts/x_ctc_raw.json

Pure data transformation over stage-1 records: the primary and each dissenter
are ordinary raw JSONs (the dissenters produced with ``--model`` set to a §7b
model). A token every dissenter disagrees with, on a fluent view of the
primary, gets the ``proper_noun_candidate`` flag. Nothing is deleted or
rewritten; masked candidates are recorded in the ``annotation`` block, so a
re-run with different rules is instant and needs no GPU (D1).

Exit 2 means an input was refused.
"""

from __future__ import annotations

import argparse
import logging
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Sequence

sys.path.insert(0, str(Path(__file__).resolve().parent))

from pipeline import schema  # noqa: E402
from pipeline.fluent import conjunction  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s",
                    datefmt="%H:%M:%S")
logger = logging.getLogger(__name__)

FLAG = "proper_noun_candidate"


def tokens_of(doc: Dict[str, Any], stream: str = "primary") -> List[str]:
    """A record's tokens for comparison: its words, silence tokens excluded."""
    words = doc["words"] if stream == "primary" else (doc.get("secondary_stream") or {}).get("words") or []
    return [w["text"] for w in words if "silence" not in (w.get("flags") or ())]


def annotate_stream(words: List[Dict[str, Any]], dissenters: Sequence[List[str]], rules: Dict[str, Any]):
    """Flag ``words`` in place; return (candidates, stats)."""
    # Compare on the spoken tokens; map back through the positions of those words.
    spoken = [k for k, w in enumerate(words) if "silence" not in (w.get("flags") or ())]
    cands, stats = conjunction([words[k]["text"] for k in spoken], dissenters, **rules)
    for c in cands:
        k = spoken[c.index]
        c.index = k                                   # report the record's word index
        if not c.masked and FLAG not in words[k]["flags"]:
            words[k]["flags"] = list(words[k]["flags"]) + [FLAG]
    return cands, stats


def annotate(doc: Dict[str, Any], dissenter_docs: Sequence[Dict[str, Any]], paths: Sequence[str],
             rules: Dict[str, Any]) -> Dict[str, Any]:
    diss_primary = [tokens_of(d) for d in dissenter_docs]
    cands, stats = annotate_stream(doc["words"], diss_primary, rules)
    block: Dict[str, Any] = {
        "created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z"),
        "flag": FLAG,
        "dissenters": [
            {"path": str(p), "model_id": d["asr"].get("model_id"), "revision": d["asr"].get("revision"),
             "adapter": d["asr"].get("adapter"), "tokens": len(t)}
            for p, d, t in zip(paths, dissenter_docs, diss_primary)
        ],
        "primary": {"stats": stats, "candidates": [c.as_dict() for c in cands]},
    }
    if doc.get("secondary_stream") and doc["secondary_stream"].get("words"):
        # The other rendering of the same audio is already fluent; the same
        # dissenters localise its garbles too, so the lecture profile sees them.
        c2, s2 = annotate_stream(doc["secondary_stream"]["words"], diss_primary, rules)
        block["secondary_stream"] = {"mode": doc["secondary_stream"]["mode"], "stats": s2,
                                     "candidates": [c.as_dict() for c in c2]}
    doc["annotation"] = block
    return doc


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Stage 1.5: flag proper-noun candidates by dissenter conjunction.")
    p.add_argument("primary", help="the primary's *_raw.json")
    p.add_argument("--dissenters", nargs="+", required=True, help="raw JSONs from the §7b models")
    p.add_argument("-o", "--output", default=None, help="default: <stem>_annotated.json beside the primary")
    p.add_argument("--min-dissenters", "--min_dissenters", dest="min_dissenters", type=int, default=None,
                   help="how many must disagree (default: all of them)")
    p.add_argument("--no-mask-adjacent", dest="mask_adjacent", action="store_false",
                   help="also flag candidates beside a dropped marker or partial")
    p.add_argument("--no-name-exempt", dest="exempt_name_like", action="store_false",
                   help="let the mask hide capitalised mid-sentence tokens too")
    p.add_argument("--no-pronoun-expansion", dest="expand_pronoun_s", action="store_false")
    return p


def main() -> None:
    args = build_parser().parse_args()
    src = Path(args.primary)
    try:
        doc = schema.read(src)
        diss = [schema.read(Path(p)) for p in args.dissenters]
    except (OSError, ValueError) as exc:
        logger.error("%s", exc)
        sys.exit(2)
    if len(diss) < 3 and args.min_dissenters is None:
        logger.error("%d dissenter(s) given; the conjunction needs three independent lineages (D18). "
                     "Pass --min-dissenters to run with fewer.", len(diss))
        sys.exit(2)
    lineages = {d["asr"].get("model_id") for d in diss}
    if len(lineages) < len(diss):
        logger.error("the same model appears twice among the dissenters: %s", sorted(lineages, key=str))
        sys.exit(2)

    rules = {"expand_pronoun_s": args.expand_pronoun_s, "mask_adjacent": args.mask_adjacent,
             "exempt_name_like": args.exempt_name_like, "min_dissenters": args.min_dissenters}
    annotate(doc, diss, args.dissenters, rules)

    st = doc["annotation"]["primary"]["stats"]
    logger.info("primary: %d fluent tokens, %d candidates, %d masked, %d flagged (%.2f%%)",
                st["fluent_tokens"], st["candidates"], st["masked"], st["flagged"], st["flagged_pct_of_fluent"])
    for c in doc["annotation"]["primary"]["candidates"]:
        logger.info("  %s %-20s adjacent=%s name_like=%s", "masked " if c["masked"] else "FLAGGED",
                    c["token"], c["adjacent_disfluency"], c["name_like"])
    stem = src.stem[:-4] if src.stem.endswith("_raw") else src.stem
    out = Path(args.output) if args.output else src.parent / f"{stem}_annotated.json"
    schema.write(out, doc)


if __name__ == "__main__":
    main()
