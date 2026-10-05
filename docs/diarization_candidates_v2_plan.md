# Diarization repair, step 4 follow-up: candidate generator v2 (pre-registration)

**Status:** pre-registered 2026-10-05, **before v2 was written or run, and before the held-out
lecture (PSY498-F26-WK3-Tue) has labels.** This file fixes v2's revisions, its parameters, the
metrics, the bar and the held-out protocol. The results go in `docs/diarization_repair_plan.md`
§1.9, not here. This file is not edited after the measurement. A change found necessary while
building, before v2 is scored on PSY777, is logged in §8 as a dated amendment.

**v1 is unchanged and stays frozen.** Its spec is `docs/diarization_candidates_plan.md`
(unedited), its parameter hash is `fd44dbfe718e24b8dd9b6c8afbc0b1ed1c0a26c453228148d1d26a2d702730b6`,
and its output on PSY777-F26-WK2-Mon must stay byte-identical to PR #29's. Everything this file
does not change is v1's definition, by reference: the terms (v1 §1), the classes v2 keeps (v1
§2), their parameters (v1 §3) and the metrics (v1 §4).

Why v2 exists (repair plan §1.8): v1 surfaced 4 of 5 high-harm misses on PSY777, and its bar
is 100%. Mark chose the held-out path over accepting 80%: he labels a second lecture, and both
generators are scored on it, v1 unchanged and v2 as fixed here.

## 0. How v2 was designed (disclosure)

- **From §1.8's diagnosis, with PSY777's gold in view.** Every revision below is **post-hoc
  relative to PSY777**, so v2's PSY777 numbers are design-set numbers (§7).
- **Scratch code, not committed, measured volume** for the candidate definitions on PSY777 and
  on eight unlabeled lectures: PSY777-F26 WK1-Wed, WK2-Wed, WK3-Wed, WK4-Mon, WK4-Wed and
  WK5-Mon, and PSY582-S26 WK9-Thu and WK12-Thu. Volume needs no labels. On PSY777 it also
  checked that the two transitions §1.8 found unsurfaced (words 2954 and 4597) fall inside the
  windows proposed here. **No other recall was computed** before this file was written.
- **No PSY498 file was opened, read, searched or run.** Nothing below depends on a fact about
  that lecture.

Scratch volume, v1 (as built) against the v2 defined here, in sites per audio hour and share of
words inside a window:

| Lecture | v1 sites/h | v1 words | v2 sites/h | v2 words |
|---|---|---|---|---|
| PSY777-F26-WK2-Mon (design set) | 110.2 | 9.3% | 77.5 | 8.9% |
| PSY777-F26-WK1-Wed | 109.4 | 7.9% | 69.1 | 7.1% |
| PSY777-F26-WK2-Wed | 158.0 | 14.4% | 104.6 | 13.4% |
| PSY777-F26-WK3-Wed | 161.6 | 12.2% | 109.5 | 11.3% |
| PSY777-F26-WK4-Mon | 109.5 | 7.3% | 74.6 | 6.8% |
| PSY777-F26-WK4-Wed | 166.5 | 12.6% | 104.5 | 11.6% |
| PSY777-F26-WK5-Mon | 151.1 | 13.1% | 104.6 | 12.3% |
| PSY582-S26-WK9-Thu | 121.0 | 10.6% | 117.1 | 10.7% |
| PSY582-S26-WK12-Thu | 80.8 | 7.9% | 63.8 | 7.3% |

The marginal costs quoted below are the drop in v2's sites per hour when that one revision is
taken out.

## 1. The revisions

| # | Revision | Decision | PSY777 finding behind it (post-hoc) |
|---|---|---|---|
| R1 | `register_change` | **dropped** | 87 candidates, 0 hold an error; the costliest class |
| R2 | `question_answer`, `address_reply` | **kept** (R4 changes `question_answer`'s openers) | 0 hits, but 9 candidates in all |
| R3 | `clause_backchannel` | **new** | miss 2954 starts after a comma; every merge cue was sentence-anchored |
| R4 | `so` as a `question_answer` opener | **adopted**, for `question_answer` only | return 4597: a question, then an answer opening with `so` |
| R5 | `paired_return` | **new** | the 44-word exchange's return (4597) is 44 words from its start; nothing surfaced it |
| R6 | `paired_start` | **new** | no miss needs it; the mirror of R5 (below) |

Unchanged: `shift`, the five split classes, `dropped_raw_change` and `backchannel`.

**R1. Drop `register_change`.** On PSY777 it proposed 87 windows, and none holds an error. It
surfaced nothing alone. Its proxy, a first-person onset after twenty words without one, fires on
a lecturer's own "I". It is also the costliest class: kept in v2, it would add 23 to 72 sites per
audio hour across the nine lectures. With it, v1's total reaches 151–167 sites/h on four of the
eight unlabeled lectures, close to the 180 cap. Keeping it would spend most of the
headroom the new classes need on a class with no evidence for it. The risk: a held-out
high-harm miss that only this class would surface. On PSY777 there was none.

**R2. Keep `question_answer` and `address_reply`.** Neither held an error on PSY777. They are
textbook turn-taking cues, and they cost 9 candidates (5.8 per hour) together. Zero hits on a
lecture with 5 high-harm misses is weak evidence against a class that cheap. `address_reply`
is unchanged; `question_answer` changes only by R4.

**R3. Anchor the backchannel cue at clauses, not only sentences.** Miss 2954 (instructor to a
second student, 6 words) starts mid-sentence: the instructor's last word ends in a comma, and
the gap is 0.24 s, under the 0.5 s sentence pause. Its first word is a backchannel token that
stands as its own comma-bounded clause. v1's `backchannel` reads whole sentences only, so it
cannot fire there. `clause_backchannel` is the same cue one level down: a short clause made only
of backchannel tokens, with clauses on both sides in the same label stretch. It fires only where
v1's `backchannel` cannot, inside a sentence, so the two classes' per-class numbers stay
separable. Of the other merge cues:
- `question_answer` and `address_reply` end their cue at terminal punctuation, so they are
  sentence-level by nature;
- a clause-level first-person onset was rejected, because the sentence-level one scored 0 of 87.

Clause boundaries are comma-like punctuation or a pause over 0.3 s (§2). 0.3 s is §6's
provisional sentence threshold, which `segment.py` records as fragmenting sentences (a boundary
every 2.3 s): too fine for sentences, about right for clauses. Miss 2954's gap (0.24 s) is under
it, so the pause rule is not fitted to that miss. The comma is what catches it. The fillers `uh`
and `um` are excluded at the clause level: a filler between commas or pauses is a hesitation
inside one speaker's sentence. The intended stream carries none (0 on all nine lectures), so
the exclusion only guards the volume if a stream ever does. Scratch cost: 4 to 36 candidates
per lecture. That is +1.9 to +8.5 sites/h on the PSY777 lectures, and +4.5 and +13.8 on the two
PSY582 ones, whose lecturer ends many sentences with a tag "right?".

**R4. Add `so` to `question_answer`'s openers only.** The return out of the 44-word exchange
(word 4597) is a student's question answered by the instructor with "So", after a 0.68 s pause.
v1 left `so` out of `ANSWER_OPENERS` on purpose: it opens 282 of PSY777's 2,181 sentences
(13%). That reason holds for `address_reply`, which needs only a short address before the
opener, so its list is unchanged. In `question_answer` the opener must follow a sentence ending
in `?`, and that conjunction keeps the cost small: +0 to +5.7 sites/h in scratch. The gain is
also gate-relevant, beyond 4597. A one-sentence student question answered with "So" has its
*start* inside the window too, because the window holds the transition into the question.

**R5. Pair every start inside a run with a return.** §1.8: "a start without an end is half a
candidate". The 44-word exchange's start is surfaced, but its return is not, so a step-5
judgment could split out the first word but not the turn. `paired_return` searches forward from
every candidate that proposes a speaker change inside a run (§2 lists the classes). It stops at
the first question end, within 60 words and without crossing a rendered boundary, and proposes
the transition after it. The reasoning: a student's turn inside a lecture most often ends in a
question to the instructor, and the instructor resumes after it. Other return cues were weighed
and rejected:
- **the next sentence opening with `so`:** it opens 13% of the lecturer's sentences, so it is
  mostly the lecturer continuing;
- **the next pause over 0.5 s:** on the exchange, the first one after its start falls 15 words
  before the student stops;
- **the next sentence start:** that is almost always the student's own next sentence.

60 words is about 24 s at this corpus's 2.5 words per second. PSY777's longest student run
inside the lecturer's label is 44 words.

R5 cannot move the gate. The gate counts a high-harm miss at its start (v1 §4), and that rule is
not changed. A surfaced return is reported as an additional metric (§4). Scratch cost: +0 to
+5.5 sites/h.

**R6. Pair every evaluation cue with a start.** This one has **no PSY777 miss behind it**, and
it is labeled as such. It is the mirror of R5, grounded in classroom discourse's commonest
exchange: the instructor asks (initiation), a student answers (response), the instructor
evaluates ("Right.", "Exactly."). A mid-run backchannel is often that evaluation, and the
answer it evaluates began after the instructor's last question. `paired_start` searches back
from each `backchannel` and `clause_backchannel` window to the last question end, under the
same limits as R5. PSY777 has an instance at words 4517–4520, under the student's label, so it
is not a miss there. A student's answer that opens with a content word, inside the lecturer's
label, would be surfaced by no other class. R6 is the only revision aimed at a pattern the
design set does not contain. It was kept because it costs at most +2.7 sites/h in scratch, and
because the gate is all-or-nothing on a small count. Its PSY777 contribution is expected to be
nil.

**Considered, not adopted.**
- **Widening `shift`'s N to reach 2954** (7 words from a boundary). That fits N to one miss,
  more than doubles `shift`'s words, and adds no signal.
- **Crediting a surfaced return toward the gate.** It is reported instead (§4). The gated rule
  stays v1's.
- **Dropping the split classes.** They add no recall, but they also add no site: all five sit
  inside `shift`'s windows. They carry evidence for the step-5 judgment.
- **A one-word question mid-run as a call on a student.** PSY777's instance of that pattern
  (high-harm miss 2277) is surfaced by `shift`. Tag questions ("right?") would dominate the
  class's volume on some lecturers (PSY582).

## 2. Definitions

v1 §1's terms hold: the stream, the rendered labels (`assign` plus `smooth`, lecture profile
defaults), boundary, transition, gap, pause, terminal, sentence, label stretch, token and window.

**Clause.** Clauses partition each sentence. A clause boundary falls before word p, where p is
not its sentence's first word, iff either:
- word p-1's text, with closing quotes and brackets stripped (`segment._CLOSERS`), ends in one
  of `CLAUSE_PUNCT`; or
- gap(p-1, p) > `CLAUSE_PAUSE_S`.

Every sentence boundary is a clause boundary.

**Question end.** Word q ends a question iff it is terminal and its text, closers stripped, ends
in `?`. This is v1's `question_answer` test.

**Classes**, in output order:

| Class | Family | Definition | Window |
|---|---|---|---|
| `shift` | shift | v1, unchanged | v1 |
| `long_island`, `edge_island`, `no_punct_lowercase`, `zero_gap`, `inside_sentence` | split | v1, unchanged | v1 |
| `dropped_raw_change` | raw | v1, unchanged | v1 |
| `question_answer` | merge | v1's rule, with R's first token in `QUESTION_ANSWER_OPENERS` (`ANSWER_OPENERS` plus `so`) | v1: `[first(Q)-1, first(R)]` |
| `backchannel` | merge | v1, unchanged: a whole *sentence* of 1 to 3 backchannel tokens, mid-run | v1: `[first-1, last+1]` |
| `clause_backchannel` | merge | a clause `[f, l]` of 1 to `BACKCHANNEL_MAX_WORDS` words, every token in `BACKCHANNEL` and none in `FILLERS`. The clause is **not a whole sentence**, and the clause before it and the clause after it both exist in the same label stretch (`stretch[f-1] = stretch[f]`, `stretch[l+1] = stretch[l]`) | `[f-1, l+1]` |
| `address_reply` | merge | v1, unchanged (v1's `ANSWER_OPENERS`) | v1 |
| `paired_return` | pair | for each candidate of a class in `PAIR_RETURN_FROM`, window `[a, b]` after clamping: q runs from b up to min(b + `PAIR_MAX_WORDS` - 1, n - 2). The search stops with no result at the first q where `stretch[q+1] ≠ stretch[b]`; otherwise it takes the first q that ends a question | `[q, q+1]` |
| `paired_start` | pair | for each candidate of a class in `PAIR_START_FROM`, window `[a, b]`: q runs from a down to max(0, a - `PAIR_MAX_WORDS` + 1). The search stops with no result at the first q where `stretch[q] ≠ stretch[a]`; otherwise it takes the first q that ends a question, which is the last one before the cue | `[q, q+1]` |

`register_change` is gone (R1).

**Pairs, further rules.**
- A pair window is proposed from the candidates the other classes emit, never from another pair
  window.
- Two sources can find the same question end. The windows are then **one candidate**, whose
  evidence lists every source: its class, its `at_word_i`, and the words between the source
  window's edge and q.
- A pair window can lie inside its source's window: a backchannel straight after a question,
  for one. It is still emitted. It adds a candidate, not a site.

**Unchanged from v1:**
- sites: connected components of windows sharing a word;
- the clamping, and "a window must hold one transition";
- candidate ids in window order;
- no transcript text in the output.

**The output** records:
- `version: 2`;
- `plan: docs/diarization_candidates_v2_plan.md`;
- the parameters below, and their hash.

v1's output keeps its exact bytes, so it carries no version field. Its version is its `plan`
and its parameter hash. The CLI selects the version with `--version` (default 1, the version
that is registered and unreplaced). v2 writes `<stem>_speaker_candidates_v2.json` by default,
so the v1 file is never overwritten.

## 3. Parameters (fixed now)

| Parameter | Value | Why |
|---|---|---|
| v1's `N`, `ZERO_GAP_S`, `ISLAND_MAX_WORDS`, `BACKCHANNEL_MAX_WORDS`, `ADDRESS_MAX_WORDS` | 3, 0.005 s, 15, 3, 10 | v1 §3, unchanged |
| v1's `ANSWER_OPENERS`, `BACKCHANNEL`, `SECOND_PERSON`, `SOLICIT`, `go ahead` | v1 §3 | unchanged |
| `QUESTION_ANSWER_OPENERS` | `ANSWER_OPENERS` plus `so` | R4 |
| `CLAUSE_PUNCT` | `,` `:` `;` `–` (en dash) `—` (em dash) | punctuation that ends a clause without ending a sentence |
| `CLAUSE_PAUSE_S` | 0.3 s | R3 |
| `FILLERS` | `uh` `um` | R3: a hesitation, not a backchannel, inside a sentence |
| `PAIR_MAX_WORDS` | 60 | R5: about 24 s of speech |
| `PAIR_RETURN_FROM` | `dropped_raw_change`, `question_answer`, `backchannel`, `clause_backchannel`, `address_reply` | every class that proposes a change inside a run |
| `PAIR_START_FROM` | `backchannel`, `clause_backchannel` | R6: the evaluation cues |
| v1's register parameters and `FIRST_PERSON` | removed | R1 |

The canonical parameter JSON (`json.dumps(params, sort_keys=True, separators=(",", ":"))`, the
same hash v1 uses) holds, beside the values above:
- `"version": 2`;
- `"classes"`, the thirteen classes in output order.

Its keys are:
- v1's `near_words`, `zero_gap_s`, `island_max_words`, `backchannel_max_words`,
  `address_max_words`, `answer_openers`, `backchannel`, `second_person`, `solicit` and
  `solicit_bigrams`;
- `question_answer_openers`, `clause_punct`, `clause_pause_s`, `fillers`, `pair_max_words`,
  `pair_return_from` and `pair_start_from`;
- `version` and `classes`.

So the class set is part of the hash. With the lexicons exactly as v1's, sorted, and the lists
in the orders above, **v2's `params_sha256` is
`0c77f39da15c71b6ac5c51a75ea5aaa3dda8aa8edf76be9035a6ce58dc267fa6`** (corrected by
amendment A1, §8: the hash first registered here was computed without `FILLERS`). The scorer
reports whether a candidate file carries its version's registered parameters.

## 4. Metrics

**Identical to v1 §4**:
- the gold read as `score_diarization.py` reads it;
- the error locations (missed change, spurious boundary, high-harm miss at the transition
  **into** the student's speech);
- the surfacing rule, window `[a, b]` surfaces transition t iff `a < t ≤ b`;
- recall overall, missed, misattributing, spurious and high-harm;
- the per-class table (candidates, per hour, words, errors by kind, errors only this class
  surfaces, share of windows holding an error), sites, words covered, the site-length
  distribution;
- the `real` boundaries inside a window, every high-harm miss with its return and whether one
  site holds its run, and the unsurfaced misses with their structural reasons.

The per-family groups are reported for split, merge and pair, along with "everything but
`shift`".

**Additional, reported and not gated.** For each high-harm miss, t is its start and m its
misattributed words. Where the run ends before the stream does, its **return** is transition
t + m:
- **high-harm return recall:** surfaced returns over high-harm misses that have a return;
- **high-harm runs bracketed:** high-harm misses whose start and return are both surfaced.

The scorer reports both for any candidate file, v1's included.

## 5. The bar (identical to v1 §5, not relaxed)

1. **High-harm recall = 100%.**
2. **At most 180 judgment sites per audio hour, and at most 25% of the stream's words inside
   some window.**

Overall recall is reported, not gated. If the bar fails, v1 §5 applies: report why, do not tune.

**Predicted on PSY777** (scratch code, stated so that §1.9 can be read against it):
- high-harm 5/5, with 2954 by `clause_backchannel`;
- about 78 sites/h;
- about 9% of words.

## 6. The held-out protocol (PSY498-F26-WK3-Tue)

### 6.1 Blinding and order

1. **v2 is frozen** at the commit that adds it to this branch, recorded in §1.9. Two kinds of
   change after that:
   - one made before the PSY777 score is a dated amendment (§8);
   - one made after the PSY777 score is a new version with its own hash and its own
     pre-registration. It is pre-registered relative to PSY498 only if it is made before
     PSY498's labels exist, by someone who has not seen them or that lecture's text.
2. **Mark labels without candidates.** No candidate file for PSY498 is generated or shown
   until his labels check complete (`complete: true`, 0 errors), so no window can steer his
   attention. The sheet is `transcripts/lectures/PSY498-F26-WK3-Tue_job49865_labels.md`,
   with the PSY777 sheet's grammar. If he writes a richer notation again, it is converted as
   §1.7 did: by a logged helper beside an untouched original, with operator answers quoted.
   That happens before any candidate exists.
3. **Then, in this order:**
   1. Regenerate v1 on PSY777 and `cmp` it against PR #29's file (sha256
      `571cab8d986324ba4ec3499324c37f1d50921905f9fcb6307f113174793284b9`). If they differ,
      stop: v1 is no longer v1.
   2. Resolve the speaker map from the labels (6.6), and record it.
   3. Score the baseline with `score_diarization.py`. Its high-harm count is what 6.4 reads.
   4. Generate v1 (`--version 1`) and v2 (`--version 2`). Check the parameter hashes and
      refuse anything else:
      - v1 `fd44dbfe718e24b8dd9b6c8afbc0b1ed1c0a26c453228148d1d26a2d702730b6`, PR #29's;
      - v2 `0c77f39da15c71b6ac5c51a75ea5aaa3dda8aa8edf76be9035a6ce58dc267fa6`, §3's (as
        amended, A1).
   5. Score both with `score_candidates.py`, on the same labels and the same map. **Both are
      reported.**

### 6.2 What each score means

- **v1 on PSY498** is the unbiased estimate of v1: its parameters were frozen before either
  label set existed for that lecture. It is reported for information.
- **v2 on PSY498** is the test of v2.
- **v2 on PSY777** is a design-set number (§7).

### 6.3 Adoption rule

**v2 goes forward to step 5 only if it passes the bar on PSY498** (§5, all three parts),
subject to 6.4. v1 does not go forward on any held-out result: it already failed its own bar on
the lecture it was registered for, and a held-out pass would not undo that.

**v2 must also pass on PSY777.** It is designed to. If it fails there, that failure is reported
in §1.9 and v2 is not taken to PSY498 as is. Mark decides between a new version (6.1) and the
stopgap (6.5).

### 6.4 Minimum count

The gate needs at least **3 high-harm misses** on the held-out lecture.
- **At 3 or more**, the gate is tested as registered.
- **Below 3, an unsurfaced high-harm miss still fails the bar.** One unrepairable student
  statement is enough, at any count.
- **Below 3 with every one surfaced, 0 included,** the result is reported as **"below the
  minimum: not a test of the gate"**. The report gives:
  - k/k;
  - each miss's word index, time, run length and surfacing classes;
  - both volume parts of the bar;
  - the one-sided 95% lower bound on recall, 0.05^(1/k).

  v2 is then **not adopted on this lecture.** One more lecture is labeled (Mark's choice of
  lecture, made before its candidates exist), and the gate is evaluated on the pooled held-out
  lectures, with v1 and v2 unchanged. Pooling continues until the pooled count reaches 3. Each
  lecture must also pass both volume caps on its own.

Why 3: when every miss of k is surfaced, the one-sided 95% lower bound on true recall is
0.05^(1/k):
- 22% at k = 2;
- 37% at k = 3;
- 55% at k = 5.

No small k is strong evidence. Three is the least at which "all surfaced" rules out a recall
below one in three. Step 5 is itself gated on the same labeled lectures, so a pass at small k
proceeds to a further test, not to adoption. The count is per missed change, as the scorer
counts it: a long exchange counts once.

### 6.5 If neither passes

- **v2 fails on PSY498:**
  - an unsurfaced high-harm miss, at any count; or
  - a volume cap exceeded; or
  - with the pooled count reached, any part of the bar failing.

  Neither generator goes to step 5. The report gives the structural reason for every
  unsurfaced miss, as §1.8 did, with word indices and times only.
- **Nothing is tuned on PSY498.** It is spent as a held-out set. Any v3 is post-hoc relative to
  both lectures. It needs its own pre-registration and a further held-out lecture, labeled
  after v3 is frozen.
- **The stopgap stays Mark's call,** as in §1.8: accept a residual high-harm recall
  explicitly, record the unsurfaced words as the floor no repair can reach, and log it as a
  change to the bar made after the result was seen.

### 6.6 The speaker map, from the labels alone

`score_diarization.py` reads the map in three places:
- the first word's speaker;
- a `real` boundary whose `who:` is empty, which starts the new label's identity;
- the label shown for every word, which decides what is an error and its direction. A
  high-harm error is a student's word under a label that maps to `L`.

Every rendered label must be mapped. For PSY498 the map is `--speaker-map auto`. It is computed
from Mark's labels and the rendered stream, never from candidates:

1. **Start:** the rendered label with the most words maps to `L`. Ties go to the label name
   that sorts first. Every other label maps to `S`.
2. **Truth:** derive the per-word truth with the current map (`derive_truth`). Count only
   known words: `L`, `S` or `S2`–`S9`, not unknown and not `?`. For each label X, nL(X) is the
   instructor's words under X and nS(X) the students' words under X; NL and NS are the totals.
3. **Map:**
   - If NS = 0, every label maps to `L`.
   - Otherwise X maps to `L` iff nL(X)·NS ≥ nS(X)·NL: X holds at least as large a share of all
     the instructor's words as of all the students' words. Ties go to `L`, the direction that
     counts more high-harm errors.
   - Otherwise X maps to the student identity with the most words under X. Ties go to `S`, then
     `S2` … `S9`.
4. **Repeat** steps 2–3 until the map stops changing. If it has not settled after 10 rounds,
   the scorer refuses. Mark then states the map in writing, logged as an operator answer, before
   any candidate file for the lecture is generated.

Why this rule:
- **Not plurality.** On PSY777, SPEAKER_00's rendered words are 159 instructor and 145
  student, so a plurality rule would map the student's label to `L`. The share rule gives
  SPEAKER_01=L, SPEAKER_00=S, the map §1.7 used, and it settles in one round. Verified in scratch
  against the normalized gold.
- **Not one to one,** as diarization error rate maps labels. A lecturer split across two
  labels is one person, and a render's speaker map would name both as the lecturer. Mapping one
  of them to a student would hide every student word under it from the high-harm count.
- **Why a loop:** the truth reads the map at the first word and at an empty `who:`. A
  provisional map, checked against the truth it produces, settles that without asking anyone.

It works for any number of labels. The resolved map and each label's counts are written into
the score JSON.

### 6.7 Commands

```bash
L=transcripts/lectures/PSY498-F26-WK3-Tue_job49865
python3 scripts/diarization_labels.py --check ${L}_labels.md
python3 scripts/score_diarization.py ${L}_labels.json --speaker-map auto --json ${L}_baseline-score.json
python3 scripts/diarization_candidates.py --version 1 ${L}_raw.json
python3 scripts/diarization_candidates.py --version 2 ${L}_raw.json
python3 scripts/score_candidates.py ${L}_labels.json ${L}_speaker_candidates.json \
    --speaker-map auto --json ${L}_candidates-score.json
python3 scripts/score_candidates.py ${L}_labels.json ${L}_speaker_candidates_v2.json \
    --speaker-map auto --json ${L}_candidates-v2-score.json
```

If the labels need converting (6.1), the scorers take the converted `…_labels.normalized.json`
instead. Aggregate numbers and word indices go into the repair plan. Everything else stays in
`transcripts/`.

## 7. Caveats, stated in advance

- **v2's PSY777 numbers are design-set numbers, and optimistic.** R3 and R5 exist because of
  two PSY777 transitions, and the scratch check confirmed both fall inside the new windows. A
  PSY777 pass is expected, and it proves little. A PSY777 failure would be informative.
- **One held-out lecture,** from a different course, possibly with a different lecturer and a
  different volume profile. The eight unlabeled lectures measured v1 at 81 to 167 sites/h.
  v1 may fail on volume alone.
- **The high-harm count will be small,** which 6.4 addresses and cannot fix.
- **Surfaced is necessary for a repair, not sufficient.** Step 5 still judges every site, under
  the zero-false-merges gate.
- **R6 targets a pattern PSY777 does not contain,** so the design set says nothing about its
  recall.

## 8. Amendments

**A1 (2026-10-05, before any v2 code was written or run).** The hash in §3 was wrong.
- **What was wrong:** it was computed from a parameter JSON that left out `FILLERS`, which was
  added to R3 and to §3's table while this file was being written, after the hash had been
  computed.
- **What it should be:** the JSON that §3 describes, with `"fillers": ["uh", "um"]`, hashes to
  `0c77f39da15c71b6ac5c51a75ea5aaa3dda8aa8edf76be9035a6ce58dc267fa6`.
- **What was first registered:**
  `09c06bfa4e74058b9b3f55dd0525804b506a714ba9649ae5a704a233b8685405`.

No definition, parameter or rule changes. §3 and 6.1 now carry the corrected hash, and §3 also
lists the JSON's keys, which the hash depends on.
