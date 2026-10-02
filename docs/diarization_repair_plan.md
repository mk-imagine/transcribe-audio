# Diarization repair: plan and manual to-dos

**Status:** plan. Build steps 1–3 are done: the labeling sheet, Mark's labels, and the baseline score (§1.7). Nothing else is built. Decided 2026-09-30: speaker-attribution repair lives **here**,
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

   **Built** (2026-10-02) as one in-place read-through: `scripts/diarization_labels.py`; format in Part 2, item 1.
2. Mark labels (Part 2, item 1). **Done** (2026-10-02); converted to the checker's grammar, §1.7.
3. Score the baseline (§1.4, step 1). **Done**: `scripts/score_diarization.py`; numbers in §1.7.
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

### 1.7 Baseline (PSY777-F26-WK2-Mon)

§1.4 step 1, scored 2026-10-02. The current stage 2 (`assign` plus `smooth` at the lecture
profile's defaults, which is the render the sheet shows) is measured against Mark's labels.
**These numbers are final under Mark's written answers to the labeling questions** (same day):
nothing in the gold is unsure, no window is excluded, and the meaning of `real` and `spurious`
is defined (below). This section gives aggregate numbers only. The labels, and everything
derived from them, stay in `transcripts/` (gitignored).

**The gold.** Mark labeled in a richer notation than the checker's grammar:
- diarizer labels as identities;
- `>> change:` at every change, including returns;
- `who:` on spurious boundaries;
- two transcript lines split to show a mid-line change;
- two verdicts written as `unknown; likely …`;
- two typos.

A converted copy beside the original (`<stem>_labels.normalized.md`) checks with 0 errors. A
conversion helper generates it; nothing in it is edited by hand. The helper logs 32 changes,
each with its line, its text before and after, and its rule. 22 of them translate the
notation. The other 10 apply Mark's written answers as operator rules, and each one quotes the
answer it applies. The original's sha256 is unchanged. The per-word speakers were rebuilt
directly from the original plus his written answers, with a separate parser. They agree with
the truth the scorer derives from the copy on all 14,270 words, and no word is unknown.

**The operator's answers.**
1. **The window after B-027/B-028.** Mark gave the speaker line by line. A one-word student
   turn falls one word after B-027, and B-028 sits inside the instructor's speech. Both
   boundaries are `spurious` under the convention in answer 3. The 243-word window that was
   excluded is gone.
2. **Three `who:` fields left empty.** He confirmed each one as the new label's speaker.
   Nothing changes.
3. **The convention.** `spurious` means the boundary is not where the change is: the speaker
   that the earlier `>> change` set continues. So `real` means a boundary at the true change.
   B-005 was labeled `real` before he adopted this convention, one word after its change. It
   is now `spurious`, logged as an operator-confirmed reinterpretation that cites his answer.
   The convention now holds on all 68 boundaries: every `real` boundary sits at a true change,
   and no `spurious` one does.
4. **A missed exchange.** One stretch of the instructor's run is a student's turn: 44 words,
   about 14 s, with the instructor answering after it. It had no markers and now does. His
   answer doesn't say which student. From context, it is the student of the exchange just
   before it (`S`): it continues that exchange and restates his question. `S` and `S2` score
   identically here.

**Speaker map.** SPEAKER_01 is the instructor (`L`) and SPEAKER_00 is a student (`S`). A
second student (`S2`) has no diarizer label of their own: the diarizer split their words
between the two labels.

```bash
python3 scripts/score_diarization.py \
    transcripts/lectures/PSY777-F26-WK2-Mon_job49864_labels.normalized.json \
    --speaker-map SPEAKER_01=L,SPEAKER_00=S
```

**How the per-word truth is read off the sheet.** The script's docstring has the full rule:
- the first word belongs to its label's speaker;
- a `real` boundary starts its `who:`, or the new label's speaker if `who:` is empty;
- a `spurious` boundary changes nothing;
- a marker starts its speaker;
- `unsure` leaves the speaker unknown until a `real` boundary or a marker names one. None
  remain in this gold.

A **missed change** is a word where the true speaker changes and the rendered label does not.
A **near miss** is a spurious boundary within N words of a missed change. The pairing is one
to one: a change already at a boundary doesn't count, and two boundaries can't share one
change. **Precision within N words** counts near misses as hits:
(real + near misses) / (real + spurious).

| Measure | Value |
|---|---|
| Diarizer boundaries | 68: real 6, spurious 62, unsure 0 |
| Boundary precision (real / (real + spurious)) | 6/68 = **8.8%**, so 91.2% are spurious splits |
| Precision within 1 word | 13/68 = **19.1%** (7 near misses) |
| Precision within 2 words | 15/68 = **22.1%** (9 near misses) |
| Precision within 3 words | 16/68 = **23.5%** (10 near misses) |
| Spurious boundaries with no true change within 3 words | 52 of 62 |
| True speaker changes | 22: 6 at a boundary, **16 missed** |
| Missed changes that misattribute words | 10, covering 78 words. In the other 6, the label is already right from that word on: a return to the speaker the label names, or the far side of a misplaced boundary |
| … that put student speech inside the instructor's turn | **5 changes, 54 words** |
| Missed changes within 3 words of a diarizer boundary | 11 of 16 |
| Words scored | 14,270 of 14,270 (none excluded) |
| Per-word attribution accuracy | **98.4%** (230 words wrong) |
| Student words attributed to the instructor | **56** of 201 student words (27.9%): 54 through missed changes, 2 through a spurious split |
| Instructor words under the student's label | 159 (135 through spurious splits, 24 through missed changes) |
| One student's words under the other's label | 15 |

**Against the provisional numbers** (scored before the answers):

| Measure | Provisional | Final | Why |
|---|---|---|---|
| Boundary precision | 7/66 = 10.6% | 6/68 = 8.8% | B-005 is `spurious` (answer 3); B-027/B-028 were `unsure` and are now `spurious` (answer 1) |
| Missed changes | 12 | 16 | Two around the one-word student turn (answer 1), two around the missed exchange (answer 4) |
| High-harm missed changes | 4, 10 words | 5, 54 words | The missed exchange (answer 4): 44 words |
| Student words attributed to the instructor | 12 of 156 (7.7%) | 56 of 201 (27.9%) | The same 44 words; the one-word turn adds a student word that is attributed correctly |
| Per-word accuracy | 98.8% over 14,027 words | 98.4% over 14,270 | The excluded window is now scored, and the exchange counts as wrong |

**What it says.**
- **Per-word accuracy hides the problem.** The instructor speaks 14,069 of the 14,270 words. A
  diarizer that printed every word as the instructor's would score 98.6%, better than the real
  98.4%. The student-side numbers are the ones to watch.
- **Spurious splits make up almost all of the boundaries:** 62 of 68. 159 of the 304 words
  rendered under the student's label are the instructor's. This is what is left *after*
  smoothing.
- **The high-harm direction is large.** 56 student words, 27.9% of what students said, print
  as lecture content. 54 of them come from 5 missed changes. This is the error §1.4's gate and
  the merge repair exist for. One missed exchange carries 44 of the 54. Its nearest diarizer
  boundary is 12 words away, so no boundary shift recovers it. The raw diarizer turns did mark
  its first word as SPEAKER_00. Under max-overlap, an overlapping SPEAKER_01 turn outvoted that
  turn, so it reached no word (the header's "dropped run"). §1.3's merge candidates are textual
  cues only. A raw-turn change that assignment dropped is acoustic evidence the generator could
  add. Step 4 should check that this exchange is proposed either way.
- **Misplaced boundaries are a minority of the spurious ones, but most of the misses.** Only 10
  of the 62 spurious boundaries are near misses within 3 words (7 within 1 word). Shifting them
  would raise precision from 8.8% to at most 23.5%. The other 52 have no change nearby and can
  only be removed. Among the misses, 11 of 16 lie within 3 words of an existing boundary, and
  the near-miss pairing matches 10 of them. §1.3's candidate list has no "shift an existing
  boundary by a few words" class. Step 4 should measure its recall alongside the listed
  candidate types.

**Caveats.**
- This is one lecture and one labeler. The high-harm count is 5 changes, and one exchange
  holds most of its words (44 of 54).
- The student in the missed exchange is a call from context, not stated in Mark's answer.
  Another student (`S2`) would give the same numbers.
- The grammar's `?` (speaker unknown) is excluded from every per-word number. A student marked
  `?` would therefore drop out of the high-harm measures entirely. The gold doesn't use `?`.
  The grammar revision below should give "a student, identity unknown" a form of its own.

**The sheet's grammar.** The conversion was needed because Mark's notation was more natural
than the checker's. A grammar revision for future sheets (a speaker map in the header, a
`>> change:` that sets the speaker in either direction, a `note:` field) is proposed in PR #26,
along with the ambiguities it would introduce. It is not adopted. Mark's answer 3 now defines
`real` and `spurious` for a boundary a few words off its change, and a revision should state
that definition in the sheet's instructions.

---

## Part 2 — Manual to-dos (Mark)

Things only Mark can do. Nothing so far in either project is human-verified. The lecture note
created in spike S3 of the study-notes plan was written by a Claude session, not by hand.

### Blocking this plan

1. **Label every speaker change in one transcript.** Chosen: **PSY777-F26-WK2-Mon**
   (`transcripts/lectures/PSY777-F26-WK2-Mon_job49864_raw.json`, the 2026-08-31 lecture, 93 min,
   2 diarizer labels). It is the same lecture the study-notes spikes used, so the labels serve
   both projects. Two parts:
   - **Part A — review the diarizer's boundaries.** Mark each place the label changes `real`,
     `spurious` or `unsure`. The raw turns change label **189** times, but the sheet shows the
     transcript as stage 2 renders it (`assign` plus `smooth`), the baseline §1.4 scores, and
     there it changes **68** times: 60 of the raw turns' 190 label runs get no word under
     max-overlap assignment (24 hold no word, 36 lose every word they overlap to the other
     label), leaving 74, and smoothing removes 6. The sheet's header carries this accounting.
   - **Part B — find missed changes.** Places where a different person speaks with **no** label
     change: student questions, answers or comments inside the lecturer's turn. This is the
     part that matters most (§1.1). A raw-turn change that assignment dropped is one of these
     if someone else really spoke there.
   - **One read-through does both.** Generate the sheet and check it as you go:

     ```bash
     python3 scripts/diarization_labels.py transcripts/lectures/PSY777-F26-WK2-Mon_job49864_raw.json
     python3 scripts/diarization_labels.py --check transcripts/lectures/PSY777-F26-WK2-Mon_job49864_labels.md
     ```

     The sheet (`<stem>_labels.md`) is the lecture profile's stream, one line per sentence as
     `segment.py` defines one, each with a stable id and a timestamp (`T0071 [00:03:26.6] …`).
     At each label change sits a boundary block:

     ```
     #### B-001  00:03:26.6  SPEAKER_01 -> SPEAKER_00
     boundary: spurious
     who:
     ```

     `boundary:` takes `real`, `spurious` or `unsure`; `who:` (optional, not on a spurious
     boundary) says who speaks next: `L` the lecturer, `S` a student, `S2`–`S9` further
     distinguishable students, `?` unknown. A missed change is a line of its own, directly above
     the line where the new speaker starts, and a matching `back` where the earlier speaker
     resumes, if they do (an illustration, not the lecture):

     ```
     >> change: S
     T0412 [00:31:02.5] is that the same as the random slope
     >> back: L "right so"
     T0413 [00:31:05.0] yes right so the slope varies by subject
     ```

     A quoted run of whole words places the marker mid-line. A change stays open until its
     `back` or the next boundary marked `real`. Open the audio only when the text is ambiguous
     (`<stem>_label-times.txt` lists every boundary as `hh:mm:ss`); mark `unsure` rather than
     guess. `--check` regenerates the sheet from the raw JSON (refusing one whose sha256
     differs), reports progress, and reports by line number anything that is not a sheet line,
     a field or a well-formed marker: an unknown verdict or `who`, a `back` with nothing open, a
     change that never returns, an edited or deleted line. With no errors it writes
     `<stem>_labels.json`, the labels §1.4 scores, positions given as word indices into the
     stream.
   - The labels stay in `transcripts/` (gitignored). They describe a class session with student
     voices, so they are not committed alongside the public fixture.
   - **Done** 2026-10-02. The sheet used a richer notation than the checker accepts. A converted
     copy (`<stem>_labels.normalized.md`) checks clean, and the original is untouched. The
     conversion and the baseline it fed are in §1.7. Mark answered the open questions in
     `<stem>_labels.questions.md` the same day, and the §1.7 numbers are final under his answers.

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
