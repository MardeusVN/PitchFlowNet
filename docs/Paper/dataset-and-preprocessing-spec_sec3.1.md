# Spec: Paper Subsection — Dataset and Preprocessing (3.1)

**Date:** 2026-06-22
**Status:** Draft

---

## Problem Statement

The Piper-Modern paper needs a concise, Springer-style Methodology subsection
(`3.1 Dataset and Preprocessing`) describing the LJSpeech-based single-speaker
preprocessing pipeline, condensed from the existing capstone report
(`docs/data_processing.md`) into academic prose, and explicitly linking the F0
frame-alignment fix to the F0Predictor contribution covered in `3.2 Proposed
Architecture`.

---

## User Stories

- **[P1]** As a paper reviewer, I want a clear, concise description of the dataset and
  cleaning pipeline so I can judge reproducibility and data quality.
  Accepted when: the subsection states dataset source, real N used (12,911), format, and
  lists each cleaning step with a one-line rationale.

- **[P1]** As a reviewer evaluating the F0Predictor contribution, I want to understand
  how F0/mel frame alignment was handled so I can judge correctness of that component.
  Accepted when: the subsection explicitly states the pyworld +1-frame offset issue and
  the crop/pad + interpolation fix, with a forward-reference to 3.2.

- **[P2]** As a reader comparing dataset scale to other LJSpeech-based TTS papers, I want
  one compact statistics table rather than full EDA.
  Accepted when: the subsection contains exactly one table (format, sample rate, N,
  total hours, duration range, transcript-length range) and zero histogram/scatter
  figures.

- **[P3]** _(out of scope)_ Related Work, Proposed Architecture (3.2), Experimental
  Setup, Results, Conclusion — separate specs needed.

---

## Functional Requirements

1. FR-01: State dataset identity — LJSpeech format, single-speaker, N=12,911 utterances
   actually used after exclusions (not 13,100 — see CSV-parsing note below). State which
   of LJSpeech's three transcript fields (ID, raw transcription, normalized
   transcription) the pipeline actually feeds to G2P: confirmed via `preprocess.py`
   lines 443/445 (`text = row[-1]`) that the pipeline always takes the last field
   regardless of row width, i.e. the normalized transcription (numbers/ordinals/
   monetary units expanded to words), not the raw one — added 2026-06-26 per user
   question.
2. FR-02: Recompute total audio duration for the actual 12,911-utterance set used in
   training (do not reuse the existing 23.92h figure, which was measured over all 13,100
   raw `.wav` files before the CSV-parsing exclusion is accounted for).
3. FR-03: List cleaning steps in pipeline order, one sentence of rationale each:
   (a) missing/empty file exclusion,
   (b) IQR-based speaking-rate outlier filtering using Silero VAD-measured speech
       duration,
   (c) VAD silence-trimming + resample to 22050 Hz,
   (d) espeak-ng grapheme-to-phoneme normalization,
   (e) max-phoneme-id length capping at training time,
   (f) corrupted tensor cache cleanup utility.
4. FR-04: Document the pyworld F0/mel frame-count mismatch (off-by-one) and the fix
   (crop/pad to match mel frame count, log-domain interpolation over unvoiced frames) as
   a deliberate implementation detail, with an explicit forward-reference to the
   F0Predictor in 3.2.
5. FR-05: Report the CSV-quoting parsing artifact as one factual sentence: default
   `csv.reader` quote handling on the `|`-delimited metadata file merges rows containing
   unescaped `"` characters, reducing the usable set from 13,100 to 12,911 utterances
   (~1.4%). Phrase as a transparency/data-integrity note, not a "difficulty overcome"
   narrative.
6. FR-06: Include exactly one table ("Table 1: Dataset statistics") with columns: format,
   sample rate, N, total duration (hours), duration range, transcript-length range.

---

## Non-Functional Requirements

- Length: target roughly 300–500 words / about half a column to one column in a Springer
  two-column layout (typical Methods-subsection budget; adjust once a candidate journal
  template is chosen).
- Citation style: Springer reference style matching the eventual target journal — not
  yet selected (see `docs/q2_publication_roadmap.md` venue-selection risk).
- Consistency: the N and total-hours figures used here must exactly match whatever is
  used later in the Experimental Setup / Results / Ablation sections, since they all
  describe the same dataset.

---

## Success Criteria

- [ ] Subsection drafted in English, fitting within roughly one Springer two-column page.
- [ ] Each cleaning step has exactly one rationale sentence — no narrative storytelling.
- [ ] F0 alignment fix is explicitly cross-referenced to 3.2's F0Predictor.
- [ ] Table 1 contains real, recomputed numbers (N=12,911 and a freshly computed total-
      hours figure) — not numbers carried over from the unfiltered 13,100-file EDA.
- [ ] CSV-parsing note is exactly one factual sentence, not a paragraph.

---

## Out of Scope

- Related Work (§2), Proposed Architecture (§3.2), Experimental Setup (§4), Results and
  Discussion (§5), Conclusion (§6) — each needs its own brainstorm/spec.
- Deciding whether to fix the CSV-quoting bug in `preprocess.py` and re-preprocess/
  retrain — explicitly deferred by the user; current Piper-Modern training (epoch 361+)
  continues on the existing 12,911-row dataset.
- Literature citations for design choices (Silero VAD, IQR outlier filtering, espeak-ng)
  — not yet researched.
- Full EDA visuals (duration/text-length histograms, correlation scatter plots) — kept
  out of the paper; may still be reused for the capstone defense slides separately.

---

## Assumptions

- Baseline (roadmap 1.1) and ablation (roadmap 1.2) training, once started, will reuse
  the same 12,911-utterance `dataset.jsonl` as the current Piper-Modern run, not a
  bug-fixed 13,100-row reprocessing. If this assumption is wrong, the eventual baseline-
  vs-Piper-Modern and ablation comparisons would be confounded by different training
  data, not just architecture.
- The target journal's exact citation and section-numbering conventions are not yet
  fixed; minor reformatting of this subsection may be needed once a candidate journal is
  chosen.

---

## [NEEDS CLARIFICATION]

- [x] ~~Exact total-duration figure for the 12,911-utterance training set~~ — RESOLVED
      2026-06-26: recomputed directly from `dataset.jsonl`'s `audio_path` (raw .wav
      headers, same method as the original 23.92h/13,100 figure). Result: **23.56h**,
      duration range 1.11–10.10s (min/max/mean/median/std: 1.11s/10.10s/6.57s/6.76s/
      2.19s) — see `dataset-and-preprocessing-draft_sec3.1.md` Table 1.
- [x] ~~Transcript-length range for the 12,911 set~~ — RESOLVED 2026-06-26, with a new
      finding not previously documented: 16 of the 12,911 surviving rows are not clean
      single utterances. The same CSV-quoting bug that drops 189 rows also *corrupts*
      these 16 by folding one or more subsequent `filename|text` rows into the current
      text field (e.g. one entry's text literally contains `"...thereby.\nLJ006-0084|
      and so numerous..."`), producing transcript lengths up to **6,293 characters**
      against a single ~6s audio file — a text/audio duration mismatch, not just a
      missing-data issue. True range for the unaffected 12,895 rows is 12–187
      characters (matching the original 13,100-set range). User decision (2026-06-26):
      report the real range (12–6,293) in Table 1 with a footnote explaining the 16
      affected rows, rather than silently excluding them or switching to median/IQR.
- [ ] Explicit confirmation, before Priority 2 (baseline training) starts, that baseline/
      ablation runs will use the same 12,911-row dataset as Piper-Modern.
- [x] ~~Does the 16-row CSV-merge corruption affect the model actually reported in this
      paper?~~ — RESOLVED 2026-06-26. Checked directly against the real training run
      (`lightning_logs/version_12/hparams.yaml`): `max_phoneme_ids = 400`. Counted
      `phoneme_ids` length per row in `dataset.jsonl`: all 16 corrupted rows phonemize to
      529–17,853 ids (>> 400) and are excluded by this existing cap; zero legitimately
      long, uncorrupted rows are excluded by the same cap. So the 16 corrupted rows never
      reach the model — effective training *N* = 12,895, not 12,911. This is now stated as
      a verified fact in the draft (not an estimate), which directly defuses the
      "does this corrupted data taint your results" reviewer concern without requiring a
      re-preprocess or re-train.
