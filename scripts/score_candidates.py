#!/usr/bin/env python3
"""Score diarization repair candidates against a checked labeling sheet.

    python3 scripts/score_candidates.py transcripts/lectures/X_labels.json \\
        transcripts/lectures/X_speaker_candidates.json \\
        --speaker-map SPEAKER_01=L,SPEAKER_00=S
        -> recall, per-class table, volume and the bar; --json FILE also writes them

Build step 4 of docs/diarization_repair_plan.md (§1.5): the recall of the
candidate generator (``diarization_candidates.py``) against the gold, §1.4 step
2. The metrics and the bar were pre-registered in
docs/diarization_candidates_plan.md (§4, §5); this implements them, for v1's
candidates and v2's alike (v2 plan §4, §5: the same metrics and bar, plus the
high-harm return metrics, reported and not gated). ``--speaker-map auto``
derives the map from the labels (v2 plan §6.6).

The gold is read exactly as ``score_diarization.py`` reads it (its stream, its
per-word truth, its missed changes). Every error is a *transition* t, the point
between stream words t-1 and t:

* a missed change: the true speaker changes entering t, the label does not;
* a spurious boundary: the label changes entering t, and the verdict is
  ``spurious``;
* a high-harm miss: a missed change whose misattributed run, starting at t,
  puts a student's words under the instructor's label.

A candidate window [a, b] (stream positions, inclusive) **surfaces** the error
at t iff ``a < t <= b``: both words of the transition lie inside it.

Refuses candidates generated from a different source (sha256), a different
render, or with boundaries other than the ones that were labeled. The output
holds counts, word indices, times and labels, never transcript text.

Exit codes: 0 ok (whether or not the bar passes); 2 refused.
"""

from __future__ import annotations

import argparse
import json
import os
import statistics
import sys
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

import diarization_candidates as dc  # noqa: E402
import diarization_labels as dl  # noqa: E402
import score_diarization as sd  # noqa: E402
from pipeline import schema  # noqa: E402

# The pre-registered bar (plan §5).
BAR = {"high_harm_recall": 1.0, "max_sites_per_hour": 180.0, "max_words_covered_share": 0.25}
KINDS = ("missed", "missed_misattributing", "spurious", "high_harm")
_HOW_MANY = {1: "one", 2: "both", 3: "all three", 4: "all four", 5: "all five", 6: "all six"}


def surfaces(a: int, b: int, t: int) -> bool:
    """The surfacing rule (plan §4): window [a, b] holds transition t."""
    return a < t <= b


def _ratio(a: int, b: int) -> Optional[float]:
    return round(a / b, 4) if b else None


def _union(spans: List[Tuple[int, int]]) -> int:
    total, cur = 0, None
    for a, b in sorted(spans):
        if cur is None or a > cur[1]:
            if cur is not None:
                total += cur[1] - cur[0] + 1
            cur = [a, b]
        else:
            cur[1] = max(cur[1], b)
    return total + (cur[1] - cur[0] + 1 if cur else 0)


def check_match(labels: Dict[str, Any], cdoc: Dict[str, Any]) -> None:
    """Refuse candidates that describe a different source or render than the labels."""
    if cdoc.get("source_sha256") != labels["source_sha256"]:
        raise SystemExit("the candidates were generated from a different source (sha256 differs)")
    if cdoc.get("render") != labels.get("render"):
        raise SystemExit("the candidates' render parameters differ from the ones the sheet was labeled with")
    if cdoc.get("stream") != labels.get("stream"):
        raise SystemExit("the candidates judge a different stream than the one labeled")
    if cdoc.get("boundaries") != [b["word_i"] for b in labels["boundaries"]]:
        raise SystemExit("the candidates' rendered boundaries are not the labeled ones")
    if cdoc.get("params_sha256") != dc.params_sha256(cdoc.get("params") or {}):
        raise SystemExit("the candidates' params_sha256 does not match their params")


def evaluate(labels: Dict[str, Any], stream: sd.Stream, truth: List[Optional[str]],
             speaker_map: Dict[str, str], cdoc: Dict[str, Any],
             render: Optional[dc.Render] = None) -> Dict[str, Any]:
    base = sd.score(labels, stream, truth, speaker_map)
    pos = {i: k for k, i in enumerate(stream.index)}
    n = len(stream.index)
    hours = (cdoc.get("audio_duration_s") or 0) / 3600 or None

    def when(p: int) -> str:
        return dl.fmt_time(stream.start[p] or 0.0, stream.offset)

    # The errors, as transitions.
    errors: List[Dict[str, Any]] = []
    for m in base["missed_changes"]["items"]:
        t = pos[m["word_i"]]
        errors.append({"kind": "missed", "t": t, "word_i": m["word_i"], "time": m["time"],
                       "from": m["from"], "to": m["to"], "label": m["label"],
                       "misattributed_words": m["misattributed_words"], "direction": m["direction"],
                       "nearest_boundary_words": m["nearest_boundary_words"],
                       "misattributing": m["misattributed_words"] > 0,
                       "high_harm": m["direction"] == "student_as_instructor"})
    for b in labels["boundaries"]:
        if b["verdict"] == "spurious":
            errors.append({"kind": "spurious", "t": pos[b["word_i"]], "word_i": b["word_i"],
                           "time": b["time"], "boundary": b["id"], "misattributing": False,
                           "high_harm": False})
    reals = [pos[b["word_i"]] for b in labels["boundaries"] if b["verdict"] == "real"]

    def in_kind(e: Dict[str, Any], k: str) -> bool:
        return {"missed": e["kind"] == "missed",
                "missed_misattributing": e["kind"] == "missed" and e["misattributing"],
                "spurious": e["kind"] == "spurious",
                "high_harm": e["high_harm"]}[k]

    version = cdoc.get("version", 1)
    classes = dc.classes_of(version)
    cands = [(pos[c["first_word_i"]], pos[c["last_word_i"]], c) for c in cdoc["candidates"]]
    for e in errors:
        by = [c for a, b, c in cands if surfaces(a, b, e["t"])]
        e["surfaced_by"] = sorted({c["class"] for c in by}, key=classes.index)
        e["candidates"] = [c["id"] for c in by]

    def recall(es: List[Dict[str, Any]]) -> Dict[str, Any]:
        hit = sum(1 for e in es if e["surfaced_by"])
        return {"surfaced": hit, "total": len(es), "recall": _ratio(hit, len(es))}

    rec = {"overall": recall(errors), **{k: recall([e for e in errors if in_kind(e, k)]) for k in KINDS}}

    def error_in(a: int, b: int) -> bool:
        return any(surfaces(a, b, e["t"]) for e in errors)

    # Per class (and per group of classes): volume, what it surfaces, what
    # only it surfaces, and the share of its windows that hold an error.
    def stats(sel: Callable[[str], bool]) -> Dict[str, Any]:
        mine = [(a, b) for a, b, c in cands if sel(c["class"])]
        with_err = sum(1 for a, b in mine if error_in(a, b))
        return {
            "candidates": len(mine),
            "per_hour": round(len(mine) / hours, 1) if hours else None,
            "words_covered": _union(mine),
            "surfaced": {k: sum(1 for e in errors if in_kind(e, k) and any(sel(x) for x in e["surfaced_by"]))
                         for k in KINDS},
            "only_this": {k: sum(1 for e in errors if in_kind(e, k) and e["surfaced_by"]
                                 and all(sel(x) for x in e["surfaced_by"])) for k in KINDS},
            "windows_with_error": with_err,
            "windows_with_error_share": _ratio(with_err, len(mine)),
        }

    per_class = {x: {"family": dc.FAMILY[x], **stats(lambda c, x=x: c == x)} for x in classes}
    per_group = {}
    for fam in ("split", "merge", "pair"):
        n_fam = sum(1 for x in classes if dc.FAMILY[x] == fam)
        if n_fam:
            per_group[f"{fam} ({_HOW_MANY[n_fam]})"] = stats(lambda c, fam=fam: dc.FAMILY[c] == fam)
    per_group["everything but shift"] = stats(lambda c: c != "shift")

    # Sites, as the generator merged them.
    sites = [(pos[s["first_word_i"]], pos[s["last_word_i"]], s) for s in cdoc["sites"]]
    lens = sorted(b - a + 1 for a, b, _ in sites)
    covered = sum(lens)
    sites_err = sum(1 for a, b, _ in sites if error_in(a, b))
    cand_err = sum(1 for a, b, _ in cands if error_in(a, b))
    volume = {
        "audio_hours": round(hours, 4) if hours else None,
        "candidates": len(cands), "candidates_per_hour": round(len(cands) / hours, 1) if hours else None,
        "sites": len(sites), "sites_per_hour": round(len(sites) / hours, 1) if hours else None,
        "words": n, "words_covered": covered, "words_covered_share": _ratio(covered, n),
        "candidates_with_error": cand_err, "candidates_with_error_share": _ratio(cand_err, len(cands)),
        "sites_with_error": sites_err, "sites_with_error_share": _ratio(sites_err, len(sites)),
        "site_words": {"min": lens[0] if lens else None,
                       "median": statistics.median(lens) if lens else None,
                       "p90": lens[min(len(lens) - 1, int(0.9 * len(lens)))] if lens else None,
                       "max": lens[-1] if lens else None},
        "real_boundaries_in_a_window": sum(1 for t in reals if any(surfaces(a, b, t) for a, b, _ in cands)),
        "real_boundaries": len(reals),
    }

    # Every high-harm miss: its start, the transition back out, and whether one site holds both.
    high = []
    for e in errors:
        if not e["high_harm"]:
            continue
        t, m = e["t"], e["misattributed_words"]
        r = t + m
        back = [c for a, b, c in cands if r < n and surfaces(a, b, r)]
        high.append({
            "word_i": e["word_i"], "time": e["time"], "misattributed_words": m,
            "nearest_boundary_words": e["nearest_boundary_words"],
            "surfaced": bool(e["surfaced_by"]), "surfaced_by": e["surfaced_by"],
            "candidates": e["candidates"],
            "return_word_i": stream.index[r] if r < n else None,
            "return_surfaced_by": sorted({c["class"] for c in back}, key=classes.index),
            "run_inside_one_site": any(a < t and (r <= b if r < n else b == n - 1) for a, b, _ in sites),
        })

    # Reported, not gated (v2 plan §4): the returns out of the high-harm runs.
    has_return = [h for h in high if h["return_word_i"] is not None]
    ret_hit = sum(1 for h in has_return if h["return_surfaced_by"])
    both = sum(1 for h in has_return if h["surfaced"] and h["return_surfaced_by"])
    returns = {"high_harm_with_a_return": len(has_return),
               "return_surfaced": ret_hit, "return_recall": _ratio(ret_hit, len(has_return)),
               "bracketed": both, "bracketed_share": _ratio(both, len(has_return))}

    # Missed changes no candidate surfaces, with the structural facts a reason needs.
    def nearest_window(t: int) -> Tuple[Optional[int], Optional[str]]:
        best: Tuple[Optional[int], Optional[str]] = (None, None)
        for a, b, c in cands:
            d = (a + 1 - t) if t <= a else (t - b) if t > b else 0
            if best[0] is None or d < best[0]:
                best = (d, c["class"])
        return best

    unsurfaced = []
    for e in errors:
        if e["kind"] != "missed" or e["surfaced_by"]:
            continue
        t = e["t"]
        d, cls = nearest_window(t)
        item = {k: e[k] for k in ("word_i", "time", "from", "to", "label", "misattributed_words",
                                  "direction", "nearest_boundary_words")}
        item.update({"nearest_window_transitions": d, "nearest_window_class": cls})
        if render is not None:
            w = render.words
            item.update({
                "gap_s": dc._round(render.gap(t)),
                "prev_terminal": dc.segment.is_terminal(w[t - 1]),
                "sentence_start": any(f == t for f, _ in render.sentences),
                "raw_labels_before": sorted(render.hits[t - 1], key=str),
                "raw_labels_at": sorted(render.hits[t], key=str),
            })
        unsurfaced.append(item)

    hh = rec["high_harm"]
    bar = {
        "high_harm_recall": {"value": hh["recall"], "required": BAR["high_harm_recall"],
                             "pass": hh["total"] > 0 and hh["surfaced"] == hh["total"]},
        "sites_per_hour": {"value": volume["sites_per_hour"], "max": BAR["max_sites_per_hour"],
                           "pass": volume["sites_per_hour"] is not None
                           and volume["sites_per_hour"] <= BAR["max_sites_per_hour"]},
        "words_covered_share": {"value": volume["words_covered_share"], "max": BAR["max_words_covered_share"],
                                "pass": volume["words_covered_share"] is not None
                                and volume["words_covered_share"] <= BAR["max_words_covered_share"]},
    }
    bar["pass"] = all(v["pass"] for v in bar.values())

    return {
        "speaker_map": speaker_map,
        "generator_version": version, "plan": cdoc.get("plan"),
        "params_preregistered": cdoc.get("params") == dc.default_params(version),
        "params_sha256": cdoc.get("params_sha256"),
        "surfacing_rule": "window [a, b] surfaces transition t iff a < t <= b",
        "errors": {"missed": sum(1 for e in errors if e["kind"] == "missed"),
                   "spurious": sum(1 for e in errors if e["kind"] == "spurious")},
        "recall": rec,
        "per_class": per_class,
        "per_group": per_group,
        "volume": volume,
        "bar": bar,
        "high_harm": high,
        "high_harm_returns": returns,
        "unsurfaced_misses": unsurfaced,
        "errors_detail": [{k: v for k, v in e.items() if k != "t"} for e in errors],
    }


def summary(r: Dict[str, Any]) -> List[str]:
    pct = lambda x: "n/a" if x is None else f"{100 * x:.1f}%"   # noqa: E731
    rec, v, b = r["recall"], r["volume"], r["bar"]
    out = [f"generator v{r['generator_version']}, params {r['params_sha256']}, "
           f"pre-registered: {r['params_preregistered']}",
           "recall: " + "; ".join(f"{k} {x['surfaced']}/{x['total']} ({pct(x['recall'])})" for k, x in rec.items()),
           f"volume: {v['candidates']} candidates ({v['candidates_per_hour']}/h), {v['sites']} sites "
           f"({v['sites_per_hour']}/h); words covered {v['words_covered']}/{v['words']} "
           f"({pct(v['words_covered_share'])}); site words median {v['site_words']['median']}, "
           f"p90 {v['site_words']['p90']}, max {v['site_words']['max']}",
           f"windows with an error: candidates {v['candidates_with_error']}/{v['candidates']} "
           f"({pct(v['candidates_with_error_share'])}), sites {v['sites_with_error']}/{v['sites']} "
           f"({pct(v['sites_with_error_share'])}); real boundaries in a window "
           f"{v['real_boundaries_in_a_window']}/{v['real_boundaries']}",
           f"{'class':<20} {'cand':>5} {'/h':>6} {'words':>6} {'hit%':>6}  surfaced m/mm/sp/hh   only m/mm/sp/hh"]
    for x, c in list(r["per_class"].items()) + list(r["per_group"].items()):
        s, o = c["surfaced"], c["only_this"]
        out.append(f"{x:<20} {c['candidates']:>5} {str(c['per_hour']):>6} {c['words_covered']:>6} "
                   f"{pct(c['windows_with_error_share']):>6}  "
                   f"{s['missed']:>2}/{s['missed_misattributing']:>2}/{s['spurious']:>2}/{s['high_harm']:>2}"
                   f"          {o['missed']:>2}/{o['missed_misattributing']:>2}/{o['spurious']:>2}/{o['high_harm']:>2}")
    for h in r["high_harm"]:
        out.append(f"high-harm {h['word_i']} {h['time']} ({h['misattributed_words']} words, boundary "
                   f"{h['nearest_boundary_words']} away): "
                   + (f"surfaced by {', '.join(h['surfaced_by'])}" if h["surfaced"] else "NOT surfaced")
                   + f"; return by {', '.join(h['return_surfaced_by']) or 'none'}"
                   + f"; one site holds the run: {h['run_inside_one_site']}")
    hr = r["high_harm_returns"]
    out.append(f"high-harm returns (reported, not gated): surfaced {hr['return_surfaced']}/"
               f"{hr['high_harm_with_a_return']} ({pct(hr['return_recall'])}); start and return both "
               f"{hr['bracketed']}/{hr['high_harm_with_a_return']}")
    for u in r["unsurfaced_misses"]:
        out.append(f"unsurfaced miss {u['word_i']} {u['time']} {u['from']}->{u['to']} "
                   f"({u['misattributed_words']} words, boundary {u['nearest_boundary_words']} away, "
                   f"nearest window {u['nearest_window_transitions']} transitions: {u['nearest_window_class']})")
    out.append("bar: " + ("PASS" if b["pass"] else "FAIL") + " -- " + "; ".join(
        f"{k} {x['value']} ({'ok' if x['pass'] else 'fails'})" for k, x in b.items() if k != "pass"))
    return out


# ------------------------------------------------------------------ CLI -----

def main(argv: Optional[List[str]] = None) -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0],
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("labels", help="the *_labels.json that diarization_labels.py --check wrote")
    p.add_argument("candidates", help="the *_speaker_candidates.json that diarization_candidates.py wrote")
    p.add_argument("--speaker-map", required=True,
                   help="who each diarizer label is, e.g. SPEAKER_01=L,SPEAKER_00=S; or 'auto', the "
                        "map the labels imply (docs/diarization_candidates_v2_plan.md §6.6)")
    p.add_argument("--source", default=None, help="the raw JSON, if not where the labels say")
    p.add_argument("--json", default=None, help="also write the numbers here")
    a = p.parse_args(argv)
    path = Path(a.labels)
    try:
        if a.speaker_map.strip().lower() != "auto":
            sd.parse_speaker_map(a.speaker_map)         # a malformed map fails before any work
        labels = json.loads(path.read_text())
        if not labels.get("complete"):
            raise SystemExit("the labels are not complete; recall needs a finished sheet")
        cdoc = json.loads(Path(a.candidates).read_text())
        check_match(labels, cdoc)
        stream = sd.rendered_stream(labels, path, Path(a.source) if a.source else None)
        speaker_map, map_rule = sd.resolve_speaker_map(a.speaker_map, labels, stream)
        unmapped = sorted({lab for lab in stream.labels if lab is not None} - set(speaker_map))
        if unmapped:
            raise SystemExit(f"--speaker-map does not say who {', '.join(unmapped)} is")
        label = labels["source"]
        src = Path(a.source) if a.source else (Path(label) if os.path.isabs(label) else path.parent / label)
        render = dc.render(schema.read(src), cdoc["render"]["profile"])
        if [w.i for w in render.words] != stream.index or render.labels != stream.labels:
            raise SystemExit("the generator's render differs from the labeled sheet's")
    except (OSError, ValueError, KeyError) as exc:
        print(f"refused: {exc}", file=sys.stderr)
        return 2
    except SystemExit as exc:
        print(f"refused: {exc}", file=sys.stderr)
        return 2
    result = evaluate(labels, stream, sd.derive_truth(labels, stream, speaker_map), speaker_map, cdoc, render)
    result = {"labels": str(path), "labels_sha256": dl.sha256_file(path),
              "candidates": str(a.candidates), "candidates_sha256": dl.sha256_file(Path(a.candidates)),
              "source": labels["source"], "source_sha256": labels["source_sha256"],
              **result, "speaker_map_rule": map_rule}
    if map_rule != "given":
        print("speaker map (auto): " + ",".join(f"{k}={v}" for k, v in sorted(speaker_map.items())))
    for ln in summary(result):
        print(ln)
    if a.json:
        Path(a.json).write_text(json.dumps(result, indent=1) + "\n")
        print(f"wrote {a.json}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
