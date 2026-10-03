# Diarization repair, step 4: the candidate generator (pre-registration)

**Status:** pre-registered 2026-10-02, **before the generator was written or run**. This file
fixes the candidate classes, their parameters, the metrics and the bar for build step 4 of
`docs/diarization_repair_plan.md` (§1.5). The results go in that plan's step 4 section
(§1.8), not here. This file is not edited after the measurement. Anything changed after the
results were seen is logged there as **post-hoc**, with the numbers before and after.

What the generator is for (§1.3): it proposes the places a model will judge in step 5. A
repair can only happen inside a candidate, so an error no candidate surfaces can never be
repaired. Every candidate also costs a model judgment, and every judgment is a chance to
introduce a false merge. So recall and volume are measured together.

## 1. Input and terms

- **Input:** one raw JSON. **Stream:** the lecture profile's stream (`intended` here), as
  `select_stream` and `to_rwords` give it. Positions `0..n-1` in stream order. Output
  windows are given as `words[].i` of that stream, the same index the gold uses.
- **Rendered labels:** `assign` then `smooth` at the lecture render's defaults:
  `pause_threshold` 0.5 s, `max_island` 2. This is the render the gold was labeled on.
  `speaker_raw` is `assign`'s label before smoothing.
- **Boundary at p:** the rendered label of word p differs from word p-1's.
- **Transition t** (1 ≤ t ≤ n-1): the point between words t-1 and t. Every error has one.
- **Gap(p-1, p):** `start[p] - end[p-1]`. A **pause** is a gap above `pause_threshold`.
- **Terminal:** `segment.is_terminal` (terminal punctuation, abbreviations excepted).
- **Sentence:** `segment.sentences` on the rendered words: punctuation, pause or label change.
- **Label stretch:** a maximal run of words with one rendered label.
- **Token:** a word's text, lowercased, curly apostrophes made straight, and stripped of
  leading and trailing characters other than letters, digits, `'` and `-`.
- **Window `[a, b]`:** inclusive stream positions, clamped to `[0, n-1]`. It holds the
  transitions `a < t ≤ b`, where both words of the transition are inside it.

## 2. Candidate classes

Each candidate is a window with one class and its evidence fields. One place can carry
several candidates of different classes. Merge candidates are never emitted across a
rendered boundary: the window's cue sentences share one label.

### Shift (signal a, §1.7)

| Class | Fires at | Window |
|---|---|---|
| `shift` | every boundary p | `[p-N-1, p+N]`: transitions p-N to p+N, where the true change may lie |

### Split candidates (possibly spurious boundaries, §1.3)

| Class | Fires at | Window |
|---|---|---|
| `long_island` | a label stretch `[s, e]` of at most `ISLAND_MAX_WORDS` words whose neighbors on both sides exist and share a label other than its own, with no pause at either edge (inside one pause-bounded run). After smoothing these are longer than `max_island` | `[s-1, e+1]` |
| `edge_island` | the same, but with a pause at one edge or both (at a pause-bounded run's edge, which `smooth` leaves alone) | `[s-1, e+1]` |
| `no_punct_lowercase` | a boundary p where word p-1 is not terminal and word p's first letter is lowercase | `[p-1, p]` |
| `zero_gap` | a boundary p with gap(p-1, p) ≤ `ZERO_GAP_S` | `[p-1, p]` |
| `inside_sentence` | a boundary p that `segment.py` would not split without the label change: word p-1 not terminal, and the gap at most `pause_threshold` (or unknown) | `[p-1, p]` |

### Dropped raw change (signal b, §1.7)

The raw turns are matched to words by the hit rule the labeling sheet's accounting uses. A
turn hits a word if they overlap by more than zero, or the word's midpoint lies in the
turn. A word's **raw labels** are the labels of the turns that hit it, plus its
`speaker_raw`, so a word that smoothing flipped still counts. A word is **contested for X**
if X is one of its raw labels and its rendered label is not X.

| Class | Fires at | Window |
|---|---|---|
| `dropped_raw_change` | a maximal stretch `[a, b]` of words contested for one label X. One exception: the stretch borders a rendered X word on either side **and** is at most N words long. Then a rendered boundary lies within N words of both its transitions, so stage 2 contains the change, displaced by at most N, and the `shift` window holds it | `[a-N-1, b+N+1]`: both raw transitions, each ±N |
| `dropped_raw_change` (kind `empty`) | a raw label run (consecutive turns, sorted by start, with one label, as the sheet's accounting groups them) that hits no word. g is the first word starting at or after the run's start. It fires unless a rendered X word lies within `[g-N, g+N-1]`, and is skipped when g is 0 or there is no such word | `[g-N-1, g+N]` |

Evidence: X, the rendered label, the stretch length, whether it borders rendered X, how
many of its words `assign` gave X before smoothing, and `kind`:
- `displaced`: it borders rendered X and is longer than N;
- `smoothed`: otherwise, if `assign` gave any of its words X;
- `outvoted`: otherwise;
- `empty`: the no-word runs above.

### Merge candidates (possible missed changes, §1.3)

| Class | Fires at | Window |
|---|---|---|
| `question_answer` | a sentence Q whose last word is terminal and ends in `?`, followed in the same label stretch by a sentence R whose first token is in `ANSWER_OPENERS` | `[first(Q)-1, first(R)]`: holds the transitions into Q (a student asked) and into R (someone answered) |
| `backchannel` | a sentence of 1 to `BACKCHANNEL_MAX_WORDS` words, every token in `BACKCHANNEL`, with a sentence before and after it in the same label stretch (mid-run) | `[first-1, last+1]` |
| `address_reply` | a sentence A of at most `ADDRESS_MAX_WORDS` words, last word terminal, holding a `SECOND_PERSON` or `SOLICIT` token or the bigram `go ahead`. A sentence R follows in the same label stretch, and R's first token is in `ANSWER_OPENERS` | `[last(A), first(R)]` |
| `register_change` | a sentence start s that is not the first in its label stretch. Take up to `REGISTER_WINDOW_WORDS` words before s and from s, within the stretch. Each side has at least `REGISTER_MIN_WORDS` words. The words before have no `FIRST_PERSON` token, and the words from s have at least `REGISTER_ONSET`. This proxy reads a switch from the lecture's *we/you* voice into a first-person *I* voice. It is crude, and it is the only stdlib register signal that does not need a model | `[s-1, s]` |

## 3. Parameters (fixed now)

| Parameter | Value | Why |
|---|---|---|
| render | lecture profile, `intended` stream, `pause_threshold` 0.5 s, `max_island` 2 | the render the gold was labeled on |
| `N` (shift and raw tolerance) | 3 words | the baseline scorer's near window (`NEAR_WORDS`). §1.7 reports near misses within 1–3 words |
| `ZERO_GAP_S` | 0.005 s | half the timestamps' 0.01 s resolution, so "zero" survives float subtraction |
| `ISLAND_MAX_WORDS` | 15 | about two sentences at the sheet's 6.5 words per line: short enough to read as an interjection |
| `BACKCHANNEL_MAX_WORDS` | 3 | "yeah", "okay okay", "mm-hmm right" |
| `ADDRESS_MAX_WORDS` | 10 | a short address: "what do you think", "go ahead" |
| `REGISTER_WINDOW_WORDS` / `REGISTER_MIN_WORDS` / `REGISTER_ONSET` | 20 / 5 / 2 | two first-person tokens within 20 words, none in the 20 before |

Lexicons (tokens, as normalized above):

- `ANSWER_OPENERS`: yes yeah yep yup no nope nah right exactly correct sure okay ok well um
  uh hmm mm mhm mm-hmm uh-huh i i'm i've i'd i'll maybe probably because good great true
- `BACKCHANNEL`: yeah yes yep yup okay ok right sure exactly correct true mm mhm mm-hmm
  uh-huh hmm uh um cool gotcha
- `SECOND_PERSON`: you your you're you've you'd you'll yours yourself yourselves y'all
- `SOLICIT`: anyone anybody someone somebody question questions (plus the bigram `go ahead`)
- `FIRST_PERSON`: i i'm i've i'd i'll me my mine myself

The generator writes all of these into its output, with a hash of the canonical parameter
JSON.

## 4. Metrics

All metrics are measured against the gold (`<stem>_labels.normalized.json`, speaker map
`SPEAKER_01=L,SPEAKER_00=S`), with the per-word truth derived exactly as
`score_diarization.py` derives it.

**Error locations.** Each is a transition t:

- **missed change:** the true speaker changes entering word t and the rendered label does
  not. There are 16, including returns that misattribute no word.
- **spurious boundary:** a boundary at t whose verdict is `spurious`. There are 62.
- **high-harm miss:** a missed change at t whose misattributed run (starting at t) puts
  student speech under the instructor's label. There are 5. Its location is the transition
  **into** the student's speech.

The error set is missed ∪ spurious: 78 transitions. The two never coincide, because one sits
at a label change and the other does not.

**The surfacing rule.** A candidate window `[a, b]` surfaces the error at transition t
**iff `a < t ≤ b`**: both words on either side of the transition lie inside the window. An
error is surfaced if any candidate surfaces it.

**Recall** = surfaced / total, for:
- the error set overall;
- missed changes, also restricted to the 10 that misattribute words;
- spurious boundaries;
- high-harm misses.

**Per class:**
- candidates, and candidates per audio hour;
- words covered (the union of the class's windows);
- errors surfaced, by kind;
- errors surfaced by this class **only** (its marginal contribution);
- the share of its windows that surface at least one error.

**Judgment sites.** Step 5 judges a place once, with every candidate's evidence on it. So the
candidates are merged into **sites**: connected components of windows that share at least one
word. Volume is reported both ways:
- candidates per audio hour, the upper bound if each candidate is judged alone;
- sites per audio hour.

Also reported:
- **words covered:** the union of all windows, as a count and a share of the stream;
- the share of windows, and of sites, that surface at least one error;
- the site-length distribution.

Audio hours are `source.duration_s / 3600`.

**Reported for information, not gated:**
- how many of the 6 `real` boundaries fall inside a window: each one is a judgment that
  should come back "different speaker";
- for every high-harm miss: whether the transition back out of the student's run is
  surfaced too, and whether the whole run lies inside one site.

**Unsurfaced misses:** every missed change no candidate surfaces, by word index and time,
with a one-line structural reason. No transcript text.

## 5. The bar

1. **High-harm recall = 100%** (5 of 5). This is the gate direction: a student's words left
   inside the instructor's turn are the error §1.4's gate exists for.
2. **At most 180 judgment sites per audio hour** (3 per minute; 281 for this 93.6-minute
   lecture), **and at most 25% of the stream's words inside some window.** A repair can only
   happen inside a window, and each judgment is a chance to introduce a false merge, so the
   generator has to narrow the text, not hand all of it to the model. Raw candidates per
   hour are reported, not capped.

Overall recall is reported, not gated. Recall on spurious boundaries is **100% by
construction**: `shift` puts a window on every boundary. Its per-class breakdown is the
informative part.

If the bar fails, §1.4's rule applies: report why. Do not tune to pass.

## 6. Caveats, stated in advance

- **One lecture is both the design set and the evaluation set.** These recall numbers are
  optimistic. Two of the classes, `shift` and `dropped_raw_change`, exist *because* of what
  this gold showed (§1.7), and N = 3 matches the window the baseline was read with. §1.7
  already says 11 of the 16 missed changes lie within 3 words of a boundary. `shift` is
  therefore expected to surface at least those 11, and that is not a test.
- **The unbiased estimate needs a held-out labeled lecture.** PSY498 is the planned second
  course. The parameters above are frozen for that run.
- **Post-hoc changes:** any parameter or definition changed after the first measurement is
  logged in §1.8 as post-hoc, with the before and after numbers. The pre-registered run stays
  the headline.
- **Five high-harm misses** make 100% a coarse bar. One exchange carries 44 of their 54 words.
