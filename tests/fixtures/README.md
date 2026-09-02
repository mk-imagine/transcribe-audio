# `tests/fixtures/golden.json`

One real stage-1 record, committed so that every part of stage 2 — turn grouping,
within-turn anchors, speaker assignment and smoothing, line numbering, print CSS, both
profiles, every format backend — is a pure function over this file, developed and tested
in milliseconds on any machine with no model, no GPU and no audio (plan §7).

## Provenance

| | |
|---|---|
| audio | `data/251211_0009.wav`, **00:05:00–00:15:00** (a 10-minute excerpt) |
| what it is | a guest lecture in a proseminar: a visiting professor presenting their research, with host interaction. Multi-speaker. Not a research-participant interview. |
| ASR | `nyralabs/CrisperWhisper2.0_large` @ `f4334f6e8193`, ct2 backend, `--dual_stream` via `transcribe_dual()` |
| diarization | `pyannote/speaker-diarization-community-1` @ `3533c8cf8e36` |
| environment | `cw2diar` (see `docs/environments.md`), SLURM job 48738 on `gpu01` |
| pipeline | commit `1a59f8e`, clean tree |
| generated | 2026-09-01 |

## What it contains

| | |
|---|---|
| verbatim stream | 1,969 tokens, with `[UH]`/`[UM]` markers, partial words and a vocalization |
| intended stream | 1,864 tokens (`secondary_stream`), the clean rendering from the same pass |
| flags | 49 `filled_pause`, 10 `partial_word`, 2 `vocalization` |
| speaker turns | 59, across 3 labels |
| errors | none; timeline complete |

Three speaker labels on a lecture with one main speaker is real diarizer behaviour on this
material (plan §10) and is exactly what stage 2's boundary smoothing has to handle. It was
not cleaned up, because a fixture that is tidier than production output tests the wrong
thing.

Both streams carry word timestamps, which the two render profiles need (D10): the coding
profile reads the verbatim stream, the lecture profile the intended one.

## One deliberate edit

`source.audio_path` was `/Users/<cluster-id>/Repos/transcribe-audio/data/251211_0009.wav`
in the run's own record. The committed copy relativizes it to `data/251211_0009.wav` so a
public repo does not carry a cluster user id, and says so in `source.audio_path_note`.
Nothing else was touched; `audio_sha256` is the identity anchor and is unchanged.

## Regenerating

Same window, same flags, from a clean checkout of the recorded commit or later:

```bash
python src/transcribe_audio.py -i data/251211_0009.wav -o transcripts/golden \
    --start_time 00:05:00 --end_time 00:15:00 --dual_stream --job_id golden
```

Expect the same token counts; timestamps may differ on a handful of bounds by ≤0.12 s
(plan §3, batched-decode rounding). The job script that produced this one is
`hpc/jobs/golden_fixture.sbatch` on the cluster.
