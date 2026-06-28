# Brainstorm Report: Paper Preprocessing Subsection (Piper-Modern)

**Date:** 2026-06-22
**Status:** Complete — handed off to spec

## Starting Question

How should the LJSpeech single-speaker preprocessing work be written up as part of an
academic (Springer-style) paper, distinct from the existing capstone report
`docs/data_processing.md`? Scope: only the preprocessing part, since this part of the
codebase is expected to stay stable for the rest of the project.

## Directions Explored

**Where does preprocessing sit in the overall paper outline?**

- **A — standalone early section** (`3. Dataset and Preprocessing`, before the method).
  Makes sense if data work is itself a contribution. Rejected: the project's main
  contribution is the Piper-Modern architecture, not the dataset, so a standalone section
  would be disproportionate.
- **B — subsection inside Methodology** (`3.1 Dataset and Preprocessing`, `3.2 Proposed
  Architecture`). Mirrors how VITS/VITS2 treat their datasets — data is a means to the
  method, not the headline. **Chosen.**
- **C — folded into Experimental Setup** (alongside training details and metrics).
  Common in recent TTS papers but pushes preprocessing decisions (e.g. the F0 alignment
  fix) away from the architecture section they actually justify. Not chosen, since the
  F0 fix specifically motivates the F0Predictor described in 3.2.

**How much of `data_processing.md` survives into the paper?**

- The capstone report is narrative ("4 khó khăn gặp phải", full EDA with histograms,
  step-by-step troubleshooting story). A paper subsection needs to be condensed,
  declarative, and tied to consequences for the method — not a diary of what was hard.
- Decision: keep the pipeline steps and one specific implementation detail (F0/mel frame
  alignment), drop full EDA in favor of one compact statistics table, and report the
  CSV-quoting data artifact as a transparency note rather than a "difficulty" story.

## Key Finding (not just a brainstorm idea — a verified fact)

The "CSV quoting" issue described in `data_processing.md` (`"` characters in transcripts
being treated as CSV quote characters, merging adjacent rows) was suspected to be an
artifact of the EDA analysis script only. Grepping `src/python/piper_train/preprocess.py`
(lines 437, 486) confirms `csv.reader(csv_file, delimiter="|")` is called **without**
`quoting=csv.QUOTE_NONE` in the real pipeline — i.e. it's not an EDA artifact, the actual
training pipeline has it. This is independently corroborated by the real
`dataset.jsonl` row count (12,911), which matches `13,100 − 189` exactly, the number of
rows the bug was predicted to merge.

Implication: the Piper-Modern run currently at epoch 361 has been training on 12,911
utterances, not the full 13,100. This needs to be the reported N in the paper (truth over
the originally-assumed round number), and — separately from the paper — the eventual
baseline/ablation runs (roadmap 1.1/1.2) must use the same 12,911-row dataset for a fair
comparison, not a bug-fixed reprocessed one. User decided: report 12,911 as-is, do not
fix/re-preprocess/retrain now (would waste the already-completed epoch 361 of training).

**Follow-up finding (2026-06-26, while drafting the prose):** the bug doesn't only drop
189 rows cleanly — it also *corrupts* 16 of the 12,911 surviving rows. When an unescaped
`"` opens an unterminated quote, the reader folds the next `filename|text` row(s) bodily
into the current text field (literal `\nLJ006-0084|...` substrings observed inside the
merged text). These 16 entries have transcript lengths up to 6,293 characters (vs. a true
max of 187 for the other 12,895) while still pointing at a single ~6s audio file —
i.e. a handful of training examples have grossly mismatched text/audio length, not just
"missing" data. Confirmed by recomputing real stats directly from `dataset.jsonl`'s
`audio_path` field: total duration 23.56h, duration range 1.11–10.10s (matches the
original 13,100-set distribution almost exactly, since audio itself is untouched by the
bug — only text parsing is affected). User decided to report the true transcript-length
range (12–6,293 chars) in Table 1 with an explanatory footnote, rather than excluding the
16 rows or switching to median/IQR.

**Second follow-up finding (2026-06-26, in response to user's review of the draft):**
the user flagged that reporting the 16-row corruption as a bare "transparency note"
without assessing its impact on the model is a real weakness — a Q2 reviewer would likely
ask "does this corrupted data taint your results?" rather than accept the disclosure at
face value. Checked directly rather than estimating: the real training run's
`lightning_logs/version_12/hparams.yaml` has `max_phoneme_ids = 400` (the existing
training-time length cap, step (5) of the pipeline). Counting `phoneme_ids` length per row
in `dataset.jsonl` shows all 16 corrupted rows phonemize to 529–17,853 ids, all exceeding
400, so the data loader (`vits/dataset.py:107-108`) already excludes every one of them —
and zero legitimately long, uncorrupted rows are caught by the same cap. So the 16
corrupted rows contribute zero gradient updates to the reported model; effective training
*N* is 12,895, not 12,911. This is a verified fact (not a re-train or an estimate) and is
now stated explicitly in the draft's CSV-quoting paragraph and in Table 1, which now
reports both N=12,911 (post-CSV-parsing) and N=12,895 (effective, training).

## Direction Chosen

- **Outline position:** Option B — `3. Methodology` → `3.1 Dataset and Preprocessing`,
  `3.2 Proposed Architecture`.
- **Style:** condensed academic Methods prose, not a narrative rewrite of
  `data_processing.md`.
- **Content kept:** dataset identity/size (LJSpeech, single-speaker, N=12,911 actual),
  ordered cleaning steps with one-line rationale each, the F0/mel frame-alignment fix
  (explicitly forward-referenced to 3.2's F0Predictor), one compact statistics table.
- **Content dropped:** full EDA (duration/text-length histograms, correlation plots),
  narrative "khó khăn" framing, step-by-step troubleshooting story.
- **Content reframed:** the CSV-quoting issue becomes one factual sentence (real N is
  12,911, not 13,100) rather than a "difficulty we overcame" anecdote.

## Open Items Carried Into the Spec

- Exact total-duration figure for the 12,911-row set needs to be recomputed (the existing
  23.92h figure was measured over all 13,100 raw `.wav` files, before the CSV bug's
  effect on `metadata.csv` parsing is accounted for).
- Whether baseline/ablation training will reuse the same 12,911-row `dataset.jsonl` —
  must be confirmed before Priority 2 (baseline) starts, or the comparison is invalid.
- The rest of the paper's outline (Related Work, Experimental Setup, Results, Conclusion)
  was intentionally not brainstormed here — out of scope per the user's original request
  to scope this to preprocessing only.

## Handoff

See `docs/Paper/dataset-and-preprocessing-spec_sec3.1.md` for the structured spec ready
for `/ck:plan` (or direct drafting). Note: `/ck:plan` defaults to looking under `plans/`
for spec files — if invoked later, point it explicitly at this path since paper artifacts
now live under `docs/Paper/` instead.
