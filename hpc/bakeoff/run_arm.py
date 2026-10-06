#!/usr/bin/env python3
"""Run one diarization bake-off arm on one recording, on a POLARIS GPU node.

    python hpc/bakeoff/run_arm.py --arm diarizen --audio data/X.wav \\
        --out transcripts/bakeoff/X/diarizen/run1

Pre-registered in docs/diarization_bakeoff_plan.md. Each arm is called the way
its own card or README documents (D19), at its documented defaults, with **no
speaker count** passed. It writes its native output verbatim and a
``provenance.json`` that names the checkpoint and revision, the package it ran
with and that package's commit, the call and its parameters, and the runtime
and GPU memory. ``scripts/diarization_bakeoff.py`` reads that file, refuses it
if the output's sha256 or the audio's differs, and converts the output into
the schema's ``speaker_turns``.

Module scope is stdlib only (bug 15): each arm imports its packages inside its
own function, so ``--help`` and ``tests/check_bakeoff.py`` run anywhere. Every
pin is checked against what actually loaded, and a mismatch stops the run with
exit 2 rather than recording a run of something else.

Not written, on purpose: pyannote 4.0.3 also returns per-speaker embeddings
(``DiarizeOutput.speaker_embeddings``). They are voice biometrics of students
(pipeline_plan.md §10), and the bake-off does not need them.

Exit codes: 0 ok; 2 refused (a pin that does not match what loaded, a model
that did not load, an audio file that is missing).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import subprocess
import sys
import time
import wave
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

ROOT = Path(__file__).resolve().parent.parent.parent
PROVENANCE_FORMAT = 1

# The arms (plan §1). Revisions are Hugging Face commit shas; commits are git
# shas of the package's own repository. Read 2026-10-05 from the primary sources
# the plan cites.
ARMS: Dict[str, Dict[str, Any]] = {
    "community1": {
        "env": "cw2diar",
        "model": {"id": "pyannote/speaker-diarization-community-1",
                  "revision": "3533c8cf8e369892e6b79ff1bf80f7b0286a54ee"},
        "package": {"name": "pyannote.audio", "version": "4.0.3"},
        "packages": ("pyannote.audio", "pyannote.core", "torch", "torchaudio", "huggingface_hub"),
    },
    "diarizen": {
        "env": "bakeoff-diarizen",
        "model": {"id": "BUT-FIT/diarizen-wavlm-large-s80-md-v2",
                  "revision": "027b3e221b9d81dce2be794084f1dbd3ba2da403",
                  "embedding": {"id": "pyannote/wespeaker-voxceleb-resnet34-LM",
                                "revision": "837717ddb9ff5507820346191109dc79c958d614",
                                "file": "pytorch_model.bin"}},
        "package": {"name": "diarizen", "commit": "844f5555b0a98acd0931511fc641a8c5b8ba92c7",
                    "repo": "https://github.com/BUTSpeechFIT/DiariZen"},
        "packages": ("diarizen", "pyannote.audio", "torch", "torchaudio", "huggingface_hub", "numpy"),
    },
    "sortformer": {
        "env": "bakeoff-nemo",
        "model": {"id": "nvidia/Nemotron-3-Diarization",
                  "revision": "f667ed73aee57d40cc39428eb768b4fd87a0a29e",
                  "file": "Nemotron-3-Diarization.nemo"},
        "package": {"name": "nemo_toolkit", "commit": "cf724ac337d1ebc7d0dda1e23fb80916f52927a5",
                    "repo": "https://github.com/NVIDIA-NeMo/Speech"},
        # The card's "Very high latency (offline)" row: 30.4 s input buffer, in 80 ms frames.
        "streaming": {"spkcache_len": 264, "fifo_len": 40, "chunk_len": 340,
                      "chunk_right_context": 40, "spkcache_update_period": 300},
        "packages": ("nemo_toolkit", "torch", "torchaudio", "lightning", "huggingface_hub", "numpy"),
    },
}
NO_COUNT = "documented default; no num/min/max speaker count passed (plan §1)"


class Refused(Exception):
    pass


# ------------------------------------------------------------ provenance ----

def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def audio_block(path: Path) -> Dict[str, Any]:
    """What the arm was given, read from the WAV header with the stdlib."""
    block: Dict[str, Any] = {"path": str(path), "sha256": sha256_file(path)}
    try:
        with wave.open(str(path), "rb") as w:
            block.update(sample_rate=w.getframerate(), channels=w.getnchannels(),
                         sample_width_bytes=w.getsampwidth(),
                         duration_s=round(w.getnframes() / w.getframerate(), 3))
    except (wave.Error, EOFError) as exc:
        block["header"] = f"not a PCM WAV the stdlib reads: {exc}"
    return block


def _git(*args: str, cwd: Path = ROOT) -> Optional[str]:
    try:
        out = subprocess.run(["git", "-C", str(cwd), *args], capture_output=True, text=True, timeout=10)
        return out.stdout.strip() if out.returncode == 0 else None
    except (OSError, subprocess.SubprocessError):
        return None


def versions(names) -> Dict[str, Optional[str]]:
    from importlib.metadata import PackageNotFoundError, version
    out: Dict[str, Optional[str]] = {}
    for n in names:
        try:
            out[n] = version(n)
        except PackageNotFoundError:
            out[n] = None
    return out


def installed_commit(dist: str) -> Optional[str]:
    """The git commit a package was pip-installed from (PEP 610 direct_url.json), if any."""
    from importlib.metadata import PackageNotFoundError, distribution
    try:
        raw = distribution(dist).read_text("direct_url.json")
    except PackageNotFoundError:
        return None
    if not raw:
        return None
    return (json.loads(raw).get("vcs_info") or {}).get("commit_id")


def run_env() -> Dict[str, Any]:
    status = _git("status", "--porcelain")
    return {"created_utc": utc_now(), "hostname": platform.node(), "python": sys.version.split()[0],
            "slurm_job_id": os.getenv("SLURM_JOB_ID"), "slurm_partition": os.getenv("SLURM_JOB_PARTITION"),
            "cuda_visible_devices": os.getenv("CUDA_VISIBLE_DEVICES"),
            "pipeline_version": {"commit": _git("rev-parse", "HEAD"),
                                 "branch": _git("rev-parse", "--abbrev-ref", "HEAD"),
                                 "dirty": bool(status) if status is not None else None}}


def write_output(out: Path, name: str, text: str, fmt: str) -> Dict[str, Any]:
    path = out / name
    path.write_text(text)
    return {"path": name, "format": fmt, "sha256": sha256_file(path)}


def turns_json(annotation_iter) -> str:
    """pyannote turns at full float precision, as stage 1 stores them (diarize.py)."""
    return json.dumps([{"start": float(t.start), "end": float(t.end), "speaker": str(s)}
                       for t, s in annotation_iter], indent=1) + "\n"


def provenance(arm: str, run: str, cfg: Dict[str, Any], audio: Dict[str, Any], model: Dict[str, Any],
               package: Dict[str, Any], call: Dict[str, Any], native: Dict[str, Any], perf: Dict[str, Any],
               extra: List[Dict[str, Any]], warnings: List[str]) -> Dict[str, Any]:
    return {"bakeoff_format": PROVENANCE_FORMAT, "plan": "docs/diarization_bakeoff_plan.md",
            "arm": arm, "run": run, "env": cfg["env"], "audio": audio, "model": model, "package": package,
            "packages": versions(cfg["packages"]), "call": call,
            "speaker_count_policy": NO_COUNT, "native_output": native, "extra_outputs": extra,
            "performance": perf, "run_env": run_env(), "warnings": warnings}


class Timer:
    """Wall time and torch's own peak GPU memory for one call."""

    def __init__(self, torch):
        self.torch = torch

    def __enter__(self):
        if self.torch.cuda.is_available():
            self.torch.cuda.synchronize()
            self.torch.cuda.reset_peak_memory_stats()
        self.t0 = time.perf_counter()
        return self

    def __exit__(self, *exc):
        if self.torch.cuda.is_available():
            self.torch.cuda.synchronize()
        self.seconds = time.perf_counter() - self.t0
        return False

    def block(self, audio_s: Optional[float]) -> Dict[str, Any]:
        cuda = self.torch.cuda.is_available()
        mb = lambda b: round(b / 2 ** 20, 1)   # noqa: E731
        return {"diarize_s": round(self.seconds, 2),
                "realtime_factor": round(audio_s / self.seconds, 1) if audio_s and self.seconds else None,
                "device": self.torch.cuda.get_device_name(0) if cuda else "cpu",
                "gpu_peak_allocated_mb": mb(self.torch.cuda.max_memory_allocated()) if cuda else None,
                "gpu_peak_reserved_mb": mb(self.torch.cuda.max_memory_reserved()) if cuda else None}


def hf_token() -> Optional[str]:
    """HF_TOKEN from the environment, else from the repo's .env (as stage 1 reads it)."""
    if os.getenv("HF_TOKEN"):
        return os.environ["HF_TOKEN"]
    env = ROOT / ".env"
    if env.is_file():
        for line in env.read_text().splitlines():
            key, _, value = line.strip().partition("=")
            if key.strip() == "HF_TOKEN":
                return value.strip().strip("\"'")
    return None


def snapshot_revision(path: str) -> str:
    """The commit sha in a hub cache path: .../snapshots/<sha>[/file]."""
    parts = Path(path).parts
    return parts[parts.index("snapshots") + 1] if "snapshots" in parts else ""


# --------------------------------------------------------------- the arms ---

def run_community1(audio: Path, out: Path, run: str) -> None:
    """community-1 through stage 1's own loader (src/pipeline/diarize.py), called as Diarizer.run calls it.

    One call yields both of DiarizeOutput's diarizations: ``speaker_diarization``
    is what production stores and is the scored arm; ``exclusive_speaker_
    diarization`` (no overlapping turns, "adapted to downstream transcription")
    is written beside it as the optional arm the plan lists for Mark (§1.4), and
    is scored only if he adds it.
    """
    cfg = ARMS["community1"]
    sys.path.insert(0, str(ROOT / "src"))
    import torch
    from pipeline.diarize import Diarizer
    from pyannote.audio import Audio

    a = audio_block(audio)
    t0 = time.perf_counter()
    d = Diarizer(cfg["model"]["id"], hf_token(), "cuda")
    d.load()
    load_s = round(time.perf_counter() - t0, 2)
    if d.pipeline is None:
        raise Refused("community-1 did not load (see the log above); is HF_TOKEN set?")
    rev = d.provenance()
    if rev.get("revision") != cfg["model"]["revision"]:
        raise Refused(f"community-1 revision {rev.get('revision')} ({rev.get('revision_source')}) "
                      f"is not the pinned {cfg['model']['revision']}")
    installed = versions(["pyannote.audio"])["pyannote.audio"]
    if installed != cfg["package"]["version"]:
        raise Refused(f"pyannote.audio {installed}, pinned {cfg['package']['version']}")
    with Timer(torch) as tm:
        # Diarizer.run's own call: decode once, hand pyannote the samples (bug 9).
        waveform, sample_rate = Audio()(str(audio))
        output = d.pipeline({"waveform": waveform, "sample_rate": sample_rate, "uri": audio.stem})
    perf = {"load_s": load_s, **tm.block(a.get("duration_s"))}
    model = {**cfg["model"], "revision_source": rev.get("revision_source")}
    package = {**cfg["package"], "loader": "src/pipeline/diarize.py:Diarizer"}
    call = {"function": "pyannote.audio.Pipeline.__call__",
            "params": {"input": "{'waveform', 'sample_rate', 'uri'} decoded by pyannote.audio.Audio()",
                       "num_speakers": None, "min_speakers": None, "max_speakers": None}}
    for arm, field in (("community1", "speaker_diarization"),
                       ("community1-exclusive", "exclusive_speaker_diarization")):
        o = out.parent.parent / arm / run if arm != "community1" else out
        o.mkdir(parents=True, exist_ok=True)
        native = write_output(o, "turns.json", turns_json(getattr(output, field)), "turns-json")
        prov = provenance(arm, run, cfg, a, model, package, {**call, "output_field": field}, native, perf, [], [])
        (o / "provenance.json").write_text(json.dumps(prov, indent=1) + "\n")
        print(f"{arm}/{run}: {native['path']} sha256 {native['sha256'][:12]}")


def run_diarizen(audio: Path, out: Path, run: str) -> None:
    """DiariZen as its README and the v2 card call it, with the two downloads pinned.

    ``DiariZenPipeline.from_pretrained`` takes no revision (read in
    diarizen/pipelines/inference.py at the pinned commit): it is
    ``snapshot_download(repo)`` + ``hf_hub_download(wespeaker, pytorch_model.bin)``
    + the constructor. This does exactly that, with each download pinned, then
    ``pipeline(path, sess_name=...)`` as the README does, ``rttm_out_dir``
    included.
    """
    cfg = ARMS["diarizen"]
    import torch
    import diarizen
    from huggingface_hub import hf_hub_download, snapshot_download
    from diarizen.pipelines.inference import DiariZenPipeline

    src = Path(diarizen.__file__).resolve().parent.parent
    commit = _git("rev-parse", "HEAD", cwd=src)
    if commit != cfg["package"]["commit"]:
        raise Refused(f"DiariZen at {src} is commit {commit}, pinned {cfg['package']['commit']}")
    a = audio_block(audio)
    t0 = time.perf_counter()
    hub = snapshot_download(repo_id=cfg["model"]["id"], revision=cfg["model"]["revision"])
    emb_cfg = cfg["model"]["embedding"]
    emb = hf_hub_download(repo_id=emb_cfg["id"], filename=emb_cfg["file"], revision=emb_cfg["revision"])
    for got, want in ((snapshot_revision(hub), cfg["model"]["revision"]),
                      (snapshot_revision(emb), emb_cfg["revision"])):
        if got != want:
            raise Refused(f"loaded revision {got}, pinned {want}")
    out.mkdir(parents=True, exist_ok=True)
    pipe = DiariZenPipeline(diarizen_hub=Path(hub).expanduser().absolute(), embedding_model=emb,
                            rttm_out_dir=str(out))
    load_s = round(time.perf_counter() - t0, 2)
    with Timer(torch) as tm:
        result = pipe(str(audio), sess_name=audio.stem)
    native = write_output(out, "turns.json",
                          turns_json((t, s) for t, _, s in result.itertracks(yield_label=True)), "turns-json")
    rttm = out / f"{audio.stem}.rttm"
    extra = [{"path": rttm.name, "format": "rttm", "sha256": sha256_file(rttm)}] if rttm.is_file() else []
    config = (Path(hub) / "config.toml").read_text()
    package = {**cfg["package"], "source": str(src),
               "vendored_pyannote_audio": versions(["pyannote.audio"])["pyannote.audio"]}
    call = {"function": "DiariZenPipeline.__call__",
            "params": {"in_wav": "path", "sess_name": audio.stem, "rttm_out_dir": "run directory",
                       "speaker_bounds_from": "config.toml [clustering.args] min_speakers/max_speakers"},
            "config_toml": config}
    prov = provenance("diarizen", run, cfg, a, cfg["model"], package, call, native,
                      {"load_s": load_s, **tm.block(a.get("duration_s"))}, extra, [])
    (out / "provenance.json").write_text(json.dumps(prov, indent=1) + "\n")
    print(f"diarizen/{run}: {native['path']} sha256 {native['sha256'][:12]}")


def run_sortformer(audio: Path, out: Path, run: str) -> None:
    """Nemotron-3-Diarization (Sortformer) as its card's NeMo quick start calls it.

    Loaded from the pinned ``.nemo`` with ``restore_from`` (the card's
    downloaded-file route) and NeMo's default ``strict=True``: the card's example
    passes ``strict=False``, which would let a NeMo build that lacks part of the
    model load silently (PyPI 3.0.0 lacks its upsampling head). The card's
    offline-style streaming configuration is set and checked, then
    ``diarize(audio=[path], batch_size=1)``, as in the quick start.
    """
    cfg = ARMS["sortformer"]
    import torch
    from huggingface_hub import hf_hub_download
    from nemo.collections.asr.models import SortformerEncLabelModel

    commit = installed_commit("nemo_toolkit")
    if commit != cfg["package"]["commit"]:
        raise Refused(f"nemo_toolkit installed from commit {commit}, pinned {cfg['package']['commit']}")
    a = audio_block(audio)
    t0 = time.perf_counter()
    path = hf_hub_download(repo_id=cfg["model"]["id"], filename=cfg["model"]["file"],
                           revision=cfg["model"]["revision"])
    if snapshot_revision(path) != cfg["model"]["revision"]:
        raise Refused(f"loaded revision {snapshot_revision(path)}, pinned {cfg['model']['revision']}")
    model = SortformerEncLabelModel.restore_from(restore_path=path, map_location="cuda")
    model.eval()
    if not getattr(model, "streaming_mode", False):
        raise Refused("the checkpoint loaded with streaming_mode off; the card's long-form route is "
                      "streaming (chunked) inference, so the plan does not run it any other way")
    sm = model.sortformer_modules
    for key, value in cfg["streaming"].items():
        setattr(sm, key, value)
    model._check_streaming_parameters()
    load_s = round(time.perf_counter() - t0, 2)
    with Timer(torch) as tm:
        segments = model.diarize(audio=[str(audio)], batch_size=1)
    out.mkdir(parents=True, exist_ok=True)
    native = write_output(out, "segments.txt", "".join(f"{s}\n" for s in segments[0]), "nemo-segments")
    model_block = {**cfg["model"],
                   "max_num_of_spks": int(model._cfg.max_num_of_spks),
                   "output_subsampling_factor": getattr(model, "output_subsampling_factor", None),
                   "high_resolution": getattr(model, "high_resolution", None)}
    call = {"function": "SortformerEncLabelModel.diarize",
            "params": {"audio": "[path]", "batch_size": 1, "sample_rate": None,
                       "postprocessing_yaml": None, "streaming": cfg["streaming"],
                       "precision": str(next(model.parameters()).dtype)}}
    prov = provenance("sortformer", run, cfg, a, model_block, cfg["package"], call, native,
                      {"load_s": load_s, **tm.block(a.get("duration_s"))}, [], [])
    (out / "provenance.json").write_text(json.dumps(prov, indent=1) + "\n")
    print(f"sortformer/{run}: {native['path']} sha256 {native['sha256'][:12]}")


RUNNERS = {"community1": run_community1, "diarizen": run_diarizen, "sortformer": run_sortformer}


def main(argv: Optional[List[str]] = None) -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0],
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--arm", required=True, choices=sorted(RUNNERS))
    p.add_argument("--audio", required=True, help="the recording, as stage 1 was given it")
    p.add_argument("--out", required=True, help="this run's directory, e.g. transcripts/bakeoff/<stem>/<arm>/run1")
    a = p.parse_args(argv)
    audio, out = Path(a.audio), Path(a.out)
    if not audio.is_file():
        print(f"refused: {audio} not found", file=sys.stderr)
        return 2
    try:
        RUNNERS[a.arm](audio, out, out.name)
    except Refused as exc:
        print(f"refused: {exc}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
