# Diarization repair: plan and manual to-dos

**Status:** plan. Nothing built. Decided 2026-09-30: speaker-attribution repair lives **here**,
upstream, not in any downstream consumer (D24 in `pipeline_plan.md`).
**Part 1** is the design. **Part 2** is the manual work only Mark can do, including the
labeling this plan needs before anything can be measured.

---

## Part 1 — The plan

### 1.1 The problem

pyannote's turns, after stage 2's max-overlap assignment and island smoothing
(`src/render/speakers.py`), still misattribute speech. Two failure directions matter, and
they are not symmetric.

| Direction | What it looks like | Harm in a lecture | Harm in an interview |
|---|---|---|---|
| **Spurious split** | A snippet that is plainly the same person continuing gets a different label | Low: the content is right, the label is wrong | Real: a coder attributes a phrase to the wrong participant |
| **Missed change (false merge)** | A different person's words sit inside another speaker's turn: a student's question or wrong answer inside the lecturer's run | **High:** a downstream note presents a student's statement as lecture content, which is "teaches something false" | **High:** one participant's words coded as another's |

Mark observed the first directly: short stretches that were obviously a continuation of the
previous speech, labeled as a speaker change. The second is harder to see and is the more
dangerous, because nothing in the rendered transcript marks it.

**Why lectures care, even though "the speaker doesn't change the content."** Attribution decides
*whether a claim is lecture content at all*. The planned study-notes pipeline
(`~/.config/skillshare/drafts/study-notes-agents-plan.md`) treats the lecturer's words as the
source of truth. It also quotes "what the lecturer emphasized". Both inherit every missed change.

### 1.2 Where it sits, and why here

- **At the producer.** The errors originate in stage 1's diarization. Every consumer of a raw
  JSON: the coding profile, the lecture profile, the study-notes pipeline, future interview
  work. A fix here is one fix, and a fix downstream would be one per consumer.
- **D3 still holds.** Stage 1 stays verbatim and lossless: raw `speaker_turns` are never
  rewritten. Repair is a **stage 1.5 pass that writes a sidecar**, applied at render time,
  exactly like the speaker map:
  - `<stem>_speaker_repairs.json` beside `<stem>_raw.json`: a list of proposed reassignments,
    each with its evidence, confidence and status (`applied` | `flagged`).
  - Applied in stage 2 **after** assignment and smoothing and **before** the speaker map and
    turn grouping.
  - The render stamp records the sidecar (path and hash) and how many repairs were applied or
    flagged, as it already does for the speaker map.
- A bad repair is therefore undone by deleting or editing a file and re-rendering, never by a
  GPU job.

### 1.3 Design

**Deterministic first.** A candidate generator, extending the logic `smooth()` already has, finds
the places worth judging:

- *Split candidates* (possibly spurious boundaries):
  - islands longer than `smooth()`'s `max_island`;
  - islands at a pause-bounded run's edge, which `smooth()` deliberately leaves alone;
  - boundaries with no terminal punctuation before them and a lowercase continuation after;
  - zero-gap boundaries;
  - boundaries inside a sentence as `segment.py` defines one.
- *Merge candidates* (possible missed changes): textual turn-taking cues inside one label's
  run:
  - a question followed by what reads as an answer;
  - backchannels ("yeah", "right", "okay") mid-run;
  - second-person address followed by a reply;
  - an abrupt register change.
  These are weak signals, and the generator only *proposes* windows; it decides nothing.

**Judgment only on candidates.** A model pass reads each candidate in a bounded window of context
and returns one of: `same speaker` (repair a split), `different speaker` (repair a merge; which
side of the window, which label), or `unsure`. Rules, carried from how ASR corrections are
bounded elsewhere:

- **Evidence or flag.** A repair is `applied` only when the text makes it clear. Otherwise it is
  `flagged`, rendered visibly, and left for a human.
- **Asymmetric caution.** A repair must never *create* a false merge. Merging two labels'
  speech into one person is applied only on strong evidence; splitting out a suspected second
  speaker is preferred to leaving a possible student statement inside the lecturer's turn.
- **The original label is always kept** beside the repaired one, in the sidecar and, if the
  profile shows it, in the render.

**This is the first model-judgment step in the pipeline that is not ASR.** D19 applies in
full:

- verify on real transcripts, not on the rule logic;
- compare exact spans, not substrings;
- record which model, revision and prompt produced a sidecar, in the sidecar itself.

**Which model, run where, is an open decision** (Part 2, item 4). Stage 1.5 runs anywhere, so it
does not need the cluster.

### 1.4 Evaluation: before anything is adopted

The gold standard is Mark's labeling of one full transcript (Part 2, item 1). Until it exists,
no number in this section can be computed. Measured on it, in this order:

1. **Baseline:** the current stage 2 (`assign` plus `smooth`), scored against the labels. This says how
   much of the problem exists after the smoothing already built.
2. **Deterministic candidates:** the recall of the candidate generator. What share of true errors
   does it even surface? A candidate it never proposes can never be repaired.
3. **The model pass on candidates.**

Metrics:

- **Split repair:** precision and recall on spurious boundaries.
- **Merge repair:** recall on missed changes, and precision.
- **The gate:** **zero false merges introduced**, meaning no repair that attributes one person's
  speech to another where the labels say otherwise. This is a hard gate, not a tradeoff.
- **The flag rate:** how much is left for a human. Report it; don't gate on it.

**Adoption rule, fixed before any data:** the model pass is adopted only if it passes the gate and
improves on the deterministic baseline's error count. If deterministic rules alone get
most of the way, the model pass isn't built.

### 1.5 Build order (spike first)

1. **Labeling sheet generator** (a script, no model). It writes, for the chosen transcript:
   - one row per diarizer label change: id, time, the last ~15 words before and the first ~15
     after, current labels;
   - a section for marking missed changes by timestamp.

   It also writes a player-friendly list of timestamps for checking the audio. **Mark's labeling
   depends on this; build it first.**
2. Mark labels (Part 2, item 1).
3. Score the baseline (§1.4, step 1).
4. Candidate generator; measure its recall.
5. Model pass on candidates; score against the gate and the baseline.
6. Only if 5 passes: the sidecar writer, render-time application, stamp fields, `check_render.py`
   cases, and a real re-render of the labeled lecture.

### 1.6 Relation to existing items

- `pipeline_plan.md` §10, *Diarization at turn boundaries*: this plan is its follow-through.
- §10, *Diarization speaker count grows with duration*: a `--max_speakers` passthrough reduces
  fragmentation into many small labels. It is complementary, and worth having before labeling, since fewer
  spurious labels means fewer boundaries to label.
- §10, *Speaker identification from an enrolled embedding library*: it would also merge an
  over-split speaker, but it carries biometric-consent constraints this plan does not.

---

## Part 2 — Manual to-dos (Mark)

Things only Mark can do. Nothing so far in either project is human-verified. The lecture note
created in spike S3 of the study-notes plan was written by a Claude session, not by hand.

### Blocking this plan

1. **Label every speaker change in one transcript.** Chosen: **PSY777-F26-WK2-Mon**
   (`transcripts/lectures/PSY777-F26-WK2-Mon_job49864_raw.json`, the 2026-08-31 lecture, 93 min,
   2 diarizer labels). It is the same lecture the study-notes spikes used, so the labels serve
   both projects. Two parts:
   - **Part A — review the diarizer's boundaries.** There are **189** places where the label
     changes. Mark each `real`, `spurious` or `unsure`. The sheet shows the text either side;
     use the audio for anything unclear. Roughly an hour.
   - **Part B — find missed changes.** Read the whole transcript for places where a different
     person speaks with **no** label change: student questions, answers or comments inside the
     lecturer's turn. Record each by start and end time and who spoke. This is the slower part,
     a full read-through with audio checks, perhaps one to two hours. It is also the part that
     matters most (§1.1).
   - Waits on build step 1, the labeling sheet generator.
   - The labels stay in `transcripts/` (gitignored). They describe a class session with student
     voices, so they are not committed alongside the public fixture.

### For the study-notes benchmark (`~/.config/skillshare`)

2. **Verify the S3 lecture note before it is used as a clean base.**
   `~/Repos/personal/study-vault/Courses/PSY777/Lectures/PSY777-2026-08-31.md` was written by a Claude session. The
   planned seeded-fault benchmark for `study-notes-faultfinder` needs a note known to be free of
   faults; otherwise a faultfinder that correctly finds a real error scores as a false alarm.
   Mechanical provenance checks come first; Mark adjudicates what they can't settle.
3. **Correct the study-notes plan's "hand-written" claims** (`drafts/study-notes-agents-plan.md`
   §5.3 pedagogy line, §7 vault paragraph, the S3 row), or ask Claude to. The note is Claude-written, and the plan's
   "calibration target" needs a human reference.
4. **Later, small:** a blind judgment of style and register on a handful of generated notes. This
   is the one quality measure the benchmark can't make mechanical.
5. **Still pending from the study-notes plan:** the S3 Obsidian render check. Install Obsidian, open
   `~/Repos/personal/study-vault`, and check callouts, embeds, aliased wikilinks, Mermaid and the
   graph on desktop and mobile.

### Decisions

6. **Which model runs the repair pass, and where** (§1.3). Stage 1.5 can run locally or on the
   cluster. The choice affects cost, privacy (interview transcripts are human-subjects data;
   pipeline_plan §10, *IRB / data governance*) and reproducibility (the sidecar records the model
   and prompt either way).
