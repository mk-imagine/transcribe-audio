# Diarization repair: plan and manual to-dos

**Status:** plan. Build steps 1–4 are done: the labeling sheet, Mark's labels, the baseline score (§1.7), and the candidate generator's recall (§1.8). **Step 4's generator (v1) fails its pre-registered bar** on one of five high-harm misses. Mark chose a held-out test over accepting it. A revised generator (v2) is pre-registered and built (§1.9). It passes on PSY777, but that lecture is its design set. **Step 5 waits on the held-out score.** That needs Mark's labels for PSY498-F26-WK3-Tue (Part 2, item 2). Nothing else is built. Decided 2026-09-30: speaker-attribution repair lives **here**,
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

**Which model, run where, is an open decision** (Part 2, item 7). Stage 1.5 runs anywhere, so it
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
4. Candidate generator; measure its recall. **Built and measured** (2026-10-02):
   `scripts/diarization_candidates.py`, pre-registered in `docs/diarization_candidates_plan.md`
   and scored by `scripts/score_candidates.py`. Results are in §1.8. The bar fails: 4 of 5
   high-harm misses are surfaced.

   **4b. v2, and a held-out test.** Pre-registered and built (2026-10-05), and design-set
   score in: `docs/diarization_candidates_v2_plan.md`, run as `--version 2`; results in §1.9.
   **Pending:** Mark's labels for PSY498-F26-WK3-Tue (Part 2, item 2). Then v1 and v2 are
   both scored on that lecture, and v2 goes to step 5 only if it passes there.
5. Model pass on candidates; score against the gate and the baseline. Waits on 4b.
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
   answer didn't say which student. Read from context as the student of the exchange just
   before it (`S`), and **confirmed by Mark on 2026-10-02**. `S` and `S2` score identically here.

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
change (operator-confirmed, 2026-10-02). **Precision within N words** counts near misses as hits:
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
- The student in the missed exchange was first a call from context; Mark confirmed it on
  2026-10-02. Another student (`S2`) would give the same numbers.
- The grammar's `?` (speaker unknown) is excluded from every per-word number. A student marked
  `?` would therefore drop out of the high-harm measures entirely. The gold doesn't use `?`.
  The grammar revision below should give "a student, identity unknown" a form of its own.

**The sheet's grammar.** The conversion was needed because Mark's notation was more natural
than the checker's. A grammar revision for future sheets (a speaker map in the header, a
`>> change:` that sets the speaker in either direction, a `note:` field) is proposed in PR #26,
along with the ambiguities it would introduce. It is not adopted. Mark's answer 3 now defines
`real` and `spurious` for a boundary a few words off its change, and a revision should state
that definition in the sheet's instructions.

### 1.8 Step 4: candidate recall (PSY777-F26-WK2-Mon)

§1.4 step 2, measured 2026-10-02 on the same gold as §1.7. **The classes, parameters, metrics
and bar were pre-registered before the generator was written or run**, in
`docs/diarization_candidates_plan.md`. That file is the spec, and it stays unedited. Order
of commits on the branch:
1. the pre-registration;
2. the generator, scorer and checks;
3. this section.

The run reported here uses the pre-registered parameters exactly (the scorer checks this, and
reports `params pre-registered: True`). **No parameter or definition was changed after the
results were seen**, so there are no post-hoc numbers to report. Aggregate numbers and word
indices only. The candidates and the score stay in `transcripts/`.

```bash
python3 scripts/diarization_candidates.py transcripts/lectures/PSY777-F26-WK2-Mon_job49864_raw.json
python3 scripts/score_candidates.py \
    transcripts/lectures/PSY777-F26-WK2-Mon_job49864_labels.normalized.json \
    transcripts/lectures/PSY777-F26-WK2-Mon_job49864_speaker_candidates.json \
    --speaker-map SPEAKER_01=L,SPEAKER_00=S
```

**What the generator does.** It regenerates the render the gold was labeled on (lecture
profile, `assign` plus `smooth`). It writes `<stem>_speaker_candidates.json`: windows of words,
each with a class and its evidence. Overlapping windows are merged into **sites**, and one
site is one model judgment in step 5. There are eleven classes:
- `shift`: every boundary ±3 words (§1.7's near misses).
- Five split classes from §1.3, all at boundaries.
- `dropped_raw_change`: a raw turn's label that no rendered word carries within 3 words
  (§1.7's dropped run).
- Four merge classes from §1.3: `question_answer`, `backchannel`, `address_reply`,
  `register_change`.

A window `[a, b]` surfaces the error at transition t iff `a < t ≤ b`. It is stdlib only,
deterministic, and writes no transcript text. Raw `speaker_turns` are only read (D3).

**Against the bar.**

| Bar (pre-registered) | Value | |
|---|---|---|
| High-harm recall = 100% | **4/5 = 80%** | **fails** |
| ≤ 180 sites per audio hour | 110.2 (172 sites over 1.56 h) | passes |
| ≤ 25% of words inside a window | 9.3% (1,323 of 14,270) | passes |

**The bar fails.** By §1.4's rule, the reason is below. Nothing was tuned to pass it.

**Recall.**

| Errors | Surfaced | Recall |
|---|---|---|
| All errors (16 missed changes + 62 spurious boundaries) | 76/78 | 97.4% |
| Missed changes | 14/16 | 87.5% |
| … that misattribute words | 9/10 | 90.0% |
| Spurious boundaries | 62/62 | 100%, by construction (`shift` has a window at every boundary) |
| High-harm misses (student speech inside the instructor's turn) | **4/5** | **80%**: 48 of the 54 words |

**Volume.**

| Measure | Value |
|---|---|
| Candidates | 384 (246.1 per audio hour) |
| Sites (one judgment each) | 172 (110.2 per audio hour) |
| Words covered | 1,323 of 14,270 (9.3%) |
| Site length, in words | median 8, p90 14, max 38 |
| Windows that hold an error | candidates 211/384 (54.9%); sites 35/172 (20.3%) |
| Sites that hold a rendered boundary | 36, and 33 of them hold an error |
| Sites with no boundary | 136, holding 795 words; **2** of them hold an error |
| `real` boundaries inside a window | 6/6: six judgments that should come back "different speaker" |

**Per class.**
- *Holds an error* is the share of the class's windows that surface at least one error.
- *Alone* counts the errors surfaced by this class and no other: missed / spurious / high-harm.
- The missed column counts 16 changes, the misattributing column 10, spurious 62 and
  high-harm 5.

| Class | Cand. | /h | Words | Holds an error | Missed | Misattr. | Spurious | High-harm | Alone |
|---|---|---|---|---|---|---|---|---|---|
| `shift` | 68 | 43.6 | 438 | 63/68 | 11 | 7 | 62 | 3 | 2 / 1 / **1** |
| `long_island` | 9 | 5.8 | 76 | 9/9 | 1 | 1 | 17 | 0 | 0 |
| `edge_island` | 23 | 14.7 | 158 | 23/23 | 3 | 3 | 43 | 1 | 0 |
| `no_punct_lowercase` | 49 | 31.4 | 97 | 49/49 | 0 | 0 | 49 | 0 | 0 |
| `zero_gap` | 15 | 9.6 | 30 | 15/15 | 0 | 0 | 15 | 0 | 0 |
| `inside_sentence` | 32 | 20.5 | 64 | 32/32 | 0 | 0 | 32 | 0 | 0 |
| `dropped_raw_change` | 80 | 51.3 | 764 | 15/80 | 10 | 6 | 14 | 3 | 0 |
| `question_answer` | 4 | 2.6 | 47 | 0/4 | 0 | 0 | 0 | 0 | 0 |
| `backchannel` | 12 | 7.7 | 31 | 5/12 | 5 | 3 | 0 | 1 | 0 |
| `address_reply` | 5 | 3.2 | 10 | 0/5 | 0 | 0 | 0 | 0 | 0 |
| `register_change` | 87 | 55.8 | 159 | 0/87 | 0 | 0 | 0 | 0 | 0 |
| *everything but `shift`* | 316 | 202.5 | 1,150 | 148/316 | 12 | 8 | 61 | 3 | 3 / 0 / **1** |

`dropped_raw_change` by kind:

| Kind | Windows | Hold an error | High-harm surfaced |
|---|---|---|---|
| `displaced` | 9 | 8 | 2 |
| `outvoted` | 50 | 6 | 1 |
| `empty` | 19 | 1 | 0 |
| `smoothed` | 2 | 0 | 0 |

**Every high-harm miss.** *Words* is the length of the misattributed run. *Boundary* is the
distance to the nearest rendered boundary.

| Word | Time | Words | Boundary | Start surfaced by | Return surfaced by | One site holds the run |
|---|---|---|---|---|---|---|
| 2277 | 00:13:53.7 | 1 | 1 | `shift` only | `shift`, `no_punct_lowercase`, `inside_sentence` | yes |
| 2330 | 00:14:12.0 | 1 | 1 | `shift`, `edge_island`, `dropped_raw_change` | six classes | yes |
| 2897 | 00:18:22.6 | 2 | 2 | `shift`, `dropped_raw_change` | six classes | yes |
| 2954 | 00:18:39.1 | 6 | 7 | **nothing** | `shift` | no |
| 4553 | 00:28:47.2 | 44 | 12 | `dropped_raw_change`, `backchannel` | **nothing** | no |

**Missed changes no class surfaces.**
- **2954 (00:18:39.1), instructor → second student, 6 words, high-harm.**
  - It starts mid-sentence. The instructor's last word before it ends in a comma, and the gap
    is 0.24 s, under the 0.5 s pause, so `segment.py` makes no sentence break.
  - Every raw turn over it is SPEAKER_01, so nothing was dropped.
  - The nearest boundary is 7 words later, outside `shift`'s ±3.
  - The text does carry two §1.3 cues: a backchannel token, then a first-person token. But
    every merge class is anchored on a sentence start or a whole sentence, so a cue
    mid-sentence fires none of them.
- **4597 (00:29:02.2), student → instructor, 0 words: the return out of the 44-word exchange.**
  - The student's turn ends in a question, and the instructor answers after a 0.68 s pause.
  - The answer opens with `so`, which the pre-registered opener list leaves out on purpose.
    It is the lecture's commonest sentence opener. So `question_answer` doesn't fire.
  - The raw turns over it are SPEAKER_01 only, and the nearest boundary is 56 words away.

**What it says.**
- **Signals (a) and (b) carry the high-harm recall, not §1.3's text cues.**
  - `shift` surfaces the three high-harm misses that sit 1–2 words from a boundary. As §1.7
    predicted, it surfaces 11 of the 16 missed changes, and one high-harm miss (2277) is
    surfaced by `shift` alone.
  - The 44-word exchange is surfaced through its first word. Two things catch it:
    - a raw SPEAKER_00 turn that lost that word to SPEAKER_01 by 5 ms of overlap (0.285 s
      against 0.290 s; `dropped_raw_change`);
    - the same word standing as a one-word mid-run sentence (`backchannel`).
  - The four merge classes surface 1 high-harm miss between them, and it is that same one.
- **The split classes add judgment evidence, not recall.** All five sit at boundaries, inside
  `shift`'s windows, so they add no site and surface nothing alone. Their windows "hold an
  error" 100% of the time only because 62 of the 68 boundaries are spurious. That says
  nothing yet about whether they tell real from spurious. That is step 5's question.
- **The off-boundary search is costly:** 136 of the 172 sites, for 2 sites that hold an error.
  - `register_change` (87 candidates), `question_answer` (4) and `address_reply` (5) surface
    nothing here.
  - `dropped_raw_change` earns its volume only in its `displaced` (8 of 9 hold an error) and
    `outvoted` kinds.
  - This is one lecture. It is not grounds to drop a class (see the caveat below).
- **A start without an end is half a candidate.** The 44-word exchange's window is 9 words
  long, and nothing surfaces where the student stops. A step-5 judgment on that window could
  split out the first word, but not the student's turn. So a merge candidate needs either
  context that reaches the run's end, or a paired candidate at the return.

**Why the bar fails** (§1.4's rule). The one unsurfaced high-harm miss has no signal of any
registered kind where it starts:
- no boundary within 3 words;
- no raw-turn dissent;
- no sentence break.

The textual cue is there, but every merge class reads whole sentences, and this one starts
after a comma. This is a gap in the class definitions. It is not a parameter at the edge of
its range, so no N or window size within reason would have surfaced it.

**What is needed to proceed** (Mark's decision; nothing here is adopted):
- **Correct path: a held-out measurement.** Label one PSY498 lecture (Part 2, item 1's
  procedure). On it, score the generator above unchanged; that is the unbiased estimate of
  this one. Score alongside it a revised generator, pre-registered before that lecture is
  labeled. The revisions this lecture suggests, all post-hoc and so all unproven here:
  - merge cues anchored on clause boundaries (a comma or a short gap) as well as sentences;
  - a paired return candidate for each merge candidate;
  - possibly dropping the classes that earned nothing.
- **Stopgap, if step 5 is to start before then:** accept the 80% high-harm recall explicitly
  as a known residual, and record the unsurfaced 6 words as the floor no repair can reach.
  That changes a pre-registered bar after seeing the result, so it is Mark's call, and it
  would be logged as such.

**Caveats.**
- **One lecture is both the design set and the evaluation set,** so these recall numbers are
  optimistic. `shift` and `dropped_raw_change` were added because of this gold, and N = 3 is
  the window it was read with. Even so, the generator fails on this lecture.
- **The high-harm count is 5,** and one exchange is 44 of the 54 words. Each high-harm miss
  moves recall by 20 points.
- **"Surfaced" is a necessary condition for a repair, not a sufficient one.** Step 5 still has
  to judge each site correctly, and the gate (zero false merges introduced) applies there.

### 1.9 Step 4 follow-up: generator v2 (design set scored, held-out pending)

**Pre-registered** 2026-10-05 in `docs/diarization_candidates_v2_plan.md`, before v2 was
written or run, and before PSY498-F26-WK3-Tue has labels. That file holds:
- the revisions, and why each was kept or dropped;
- the parameters and their hash;
- v1's metrics and bar, unrelaxed;
- the held-out protocol.

Order of commits on the branch:
1. the pre-registration;
2. two amendments, both made while building and before any v2 score. A1 corrects a parameter
   hash that had been computed before one parameter was added. A2 replaces a speaker-map rule
   that could confirm a wrong guess;
3. the generator, scorer and checks;
4. this section.

**v1 is frozen, byte for byte.** `--version 1` is the default. Regenerated on PSY777, it is
identical by `cmp` to PR #29's file (sha256 `571cab8d…`). The checks pin its parameter hash
(`fd44dbfe…`) and its whole output on two synthetic records. Rescored with the updated scorer,
every number §1.8 reports is unchanged.

**What v2 changes.** Each change is post-hoc relative to PSY777, so every v2 number below is a
**design-set number**:
- `register_change` is dropped (87 candidates, 0 hits);
- `clause_backchannel` is new: the backchannel cue inside a sentence (miss 2954);
- `question_answer` takes `so` as an opener (return 4597);
- `paired_return` and `paired_start` are new. Each proposes the transition after a question
  end: forward from a start inside a run, or back from an evaluation cue.

v2 is frozen at commit `a5b7900`, parameter hash `0c77f39d…`.

```bash
python3 scripts/diarization_candidates.py --version 2 transcripts/lectures/PSY777-F26-WK2-Mon_job49864_raw.json
python3 scripts/score_candidates.py \
    transcripts/lectures/PSY777-F26-WK2-Mon_job49864_labels.normalized.json \
    transcripts/lectures/PSY777-F26-WK2-Mon_job49864_speaker_candidates_v2.json \
    --speaker-map SPEAKER_01=L,SPEAKER_00=S
```

**Against the bar, on the design set.** Passing here proves little: v2 was built to close these
two gaps. A failure would have been informative.

| Bar (unchanged) | v1 | v2 |
|---|---|---|
| High-harm recall = 100% | 4/5, fails | **5/5, passes** |
| ≤ 180 sites per audio hour | 110.2 | 77.5 |
| ≤ 25% of words inside a window | 9.3% | 8.9% |

The pre-registered prediction was 5/5, about 78 sites/h and about 9%. It was met.

**Recall and volume.**

| Measure | v1 | v2 |
|---|---|---|
| All errors | 76/78 | 78/78 |
| Missed changes | 14/16 | 16/16 |
| … that misattribute words | 9/10 | 10/10 |
| Spurious boundaries | 62/62 | 62/62 |
| High-harm misses | 4/5 | **5/5** |
| High-harm returns surfaced (reported, not gated) | 4/5 | 5/5 |
| High-harm runs with start and return both surfaced | 3/5 | 5/5 |
| Candidates | 384 (246.1/h) | 329 (210.8/h) |
| Sites | 172 (110.2/h) | 121 (77.5/h) |
| Words covered | 1,323 (9.3%) | 1,270 (8.9%) |
| Sites that hold an error | 35/172 | 37/121 |
| Sites with no boundary: count, words, holding an error | 136, 795, 2 | 85, 742, 4 |

**The new and changed classes.** The other classes' windows are unchanged from §1.8.

| Class | Cand. | /h | Words | Holds an error | Missed surfaced | High-harm | Alone (missed / high-harm) |
|---|---|---|---|---|---|---|---|
| `question_answer` (with `so`) | 17 | 10.9 | 136 | 1/17 | 1: 4597 | 0 | 0 / 0 |
| `clause_backchannel` | 7 | 4.5 | 19 | 2/7 | 2: 2954, 4521 | 1: 2954 | **1 / 1** |
| `paired_return` | 7 | 4.5 | 14 | 1/7 | 1: 4597 | 0 | 0 / 0 |
| `paired_start` | 5 | 3.2 | 10 | 1/5 | 1: 4597 | 0 | 0 / 0 |

**What it says.**
- **`clause_backchannel` is the change the gate needed.** Miss 2954 is surfaced by it alone.
- **The 44-word exchange's return (4597) is now surfaced three ways:**
  - `question_answer`, through `so`;
  - `paired_return`, from three sources (the exchange's raw-turn and backchannel start, and the
    student's backchannel before the question);
  - `paired_start`.

  So both ends of every high-harm run are surfaced, though no single site holds the run of 2954
  or 4553.
- **`paired_start`'s hit is incidental to its rationale.** Its source is a backchannel clause
  47 words into the instructor's answer, not an evaluation of a student's answer. This is the
  one revision with no PSY777 miss behind it, and the design set says nothing about it.
- **Volume falls by 30%.** Dropping `register_change` alone saves 43.6 sites/h, and the new and
  changed classes add back 10.9. Each one's marginal cost is in the plan's §1.

**Volume on eight unlabeled lectures** (no labels needed; the built generator):

| Lecture | v1 sites/h | v1 words | v2 sites/h | v2 words |
|---|---|---|---|---|
| PSY777-F26-WK1-Wed | 109.4 | 7.9% | 69.1 | 7.1% |
| PSY777-F26-WK2-Wed | 158.0 | 14.4% | 104.6 | 13.4% |
| PSY777-F26-WK3-Wed | 161.6 | 12.2% | 109.5 | 11.3% |
| PSY777-F26-WK4-Mon | 109.5 | 7.3% | 74.6 | 6.8% |
| PSY777-F26-WK4-Wed | 166.5 | 12.6% | 104.5 | 11.6% |
| PSY777-F26-WK5-Mon | 151.1 | 13.1% | 104.6 | 12.3% |
| PSY582-S26-WK9-Thu | 121.0 | 10.6% | 117.1 | 10.7% |
| PSY582-S26-WK12-Thu | 80.8 | 7.9% | 63.8 | 7.3% |

v2 stays under both caps on all eight, with room to spare. v1 comes within 14 sites/h of the
cap on one.

**Held-out score: pending Mark's labels** (Part 2, item 2). The protocol is the v2 plan's §6:
- v1 and v2 are both scored on PSY498-F26-WK3-Tue, with the parameter hashes above.
- **v2 goes to step 5 only if it passes the unchanged bar there.**
- The gate needs at least 3 high-harm misses.
  - Below that, an unsurfaced miss still fails it.
  - An all-surfaced result below 3 is reported as "not a test", and a further lecture is
    labeled and pooled.
- If v2 fails, nothing is tuned on PSY498. Mark chooses between a v3 with a further held-out
  lecture and §1.8's stopgap.
- The speaker map is `--speaker-map auto`: the share rule, applied to the words the labels
  state. On PSY777 it reproduces SPEAKER_01=L, SPEAKER_00=S.
- **No candidate file for PSY498 is generated until his labels check complete.**

**Caveats.**
- These are design-set numbers, and optimistic.
- The held-out test is one lecture from another course, possibly with another lecturer.
- Its high-harm count is likely to be small.

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
2. **Label PSY498-F26-WK3-Tue, the held-out lecture** (§1.9; protocol in
   `docs/diarization_candidates_v2_plan.md` §6). Step 5 waits on this.
   - **The sheet is already generated:** `transcripts/lectures/PSY498-F26-WK3-Tue_job49865_labels.md`,
     with 52 boundary blocks. The raw turns change label 181 times.
   - **The grammar and procedure are item 1's,** Parts A and B in one read-through. Check as
     you go:

     ```bash
     python3 scripts/diarization_labels.py --check transcripts/lectures/PSY498-F26-WK3-Tue_job49865_labels.md
     ```

   - **Fill `who:` on every `real` boundary.** The scorer's speaker map is computed from the
     identities the labels state (v2 plan §6.6). An empty `who:` into a label the labels never
     otherwise identify makes it refuse, and costs a round of questions.
   - **If someone other than the lecturer speaks first,** put a `>> change: WHO` above the first
     line, with its `>> back:` where the lecturer starts.
   - **Blinding.** No candidate file for this lecture is generated or shown until the labels
     check complete. The candidates' windows must not steer where you look.
   - The labels stay in `transcripts/` (gitignored), as item 1's do.

### For the study-notes benchmark (`~/.config/skillshare`)

3. **Verify the S3 lecture note before it is used as a clean base.**
   `~/Repos/personal/study-vault/Courses/PSY777/Lectures/PSY777-2026-08-31.md` was written by a Claude session. The
   planned seeded-fault benchmark for `study-notes-faultfinder` needs a note known to be free of
   faults; otherwise a faultfinder that correctly finds a real error scores as a false alarm.
   Mechanical provenance checks come first; Mark adjudicates what they can't settle.
4. **Correct the study-notes plan's "hand-written" claims** (`drafts/study-notes-agents-plan.md`
   §5.3 pedagogy line, §7 vault paragraph, the S3 row), or ask Claude to. The note is Claude-written, and the plan's
   "calibration target" needs a human reference.
5. **Later, small:** a blind judgment of style and register on a handful of generated notes. This
   is the one quality measure the benchmark can't make mechanical.
6. **Still pending from the study-notes plan:** the S3 Obsidian render check. Install Obsidian, open
   `~/Repos/personal/study-vault`, and check callouts, embeds, aliased wikilinks, Mermaid and the
   graph on desktop and mobile.

### Decisions

7. **Which model runs the repair pass, and where** (§1.3). Stage 1.5 can run locally or on the
   cluster. The choice affects cost, privacy (interview transcripts are human-subjects data;
   pipeline_plan §10, *IRB / data governance*) and reproducibility (the sidecar records the model
   and prompt either way).
8. **The recording setup.** It decides what the diarizer has to work with in every lecture
   after this one.
   - **The setup today:** a Tascam DR-40X with its internal X/Y microphones, near the front of
     the class. The lecturer is not individually mic'd. Every WAV in `data/` is mono,
     48 kHz/16-bit.
   - **The questions:**
     1. Is the recorder set to mono, or are the SD card's originals stereo and downmixed later?
        If they are stereo, keep the originals. The X/Y pair's two channels carry direction,
        which could separate the lecturer from the students.
     2. Which way does the recorder face?
     3. Optional, if the lecturer agrees: a lavalier on one of the recorder's XLR inputs. That
        gives a channel that is almost only the lecturer.
