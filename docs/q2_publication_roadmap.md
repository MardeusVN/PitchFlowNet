# Roadmap: What's Needed to Target a Q2 Journal

This document records what would need to be added to the current project (Piper-Modern VITS: Snake1d Generator, MultiResolutionDiscriminator, Transformer-in-Flow + DurationDiscriminator, F0Predictor — trained single-speaker on LJSpeech) in order to realistically target a Q2-ranked journal, as opposed to just finishing the graduation capstone.

Current state: one architecture, no baseline comparison, no ablations, evaluation harness (WER + UTMOS) not yet built, only 2 test sentences listened to so far across 2 checkpoints (epoch 124 vs 217).

---

## 1. Must-have (reviewers will ask immediately if missing)

### 1.1 Fair baseline
Train vanilla VITS (and/or VITS2) on the **exact same dataset, split, and training budget** as the current model. Without this, there is no basis to claim the added components ("Piper-Modern") improve on the baseline — comparing epoch 124 vs epoch 217 is two checkpoints of the *same* architecture, not a baseline comparison.

### 1.2 Ablation study
Isolate the contribution of each added component by training leave-one-out (or one-at-a-time) variants:
- Baseline VITS (no additions)
- + Snake1d generator only
- + MultiResolutionDiscriminator only
- + Transformer-in-Flow + DurationDiscriminator only
- + F0Predictor only
- Full combination (current model)

This is the single most common reason student-level TTS papers get rejected at this tier — claiming "the full stack is better" without showing which parts actually contribute.

### 1.3 Evaluation suite with statistical testing
Move beyond a handful of anecdotal test sentences:
- Run WER (Whisper) + UTMOS (UTMOSv2) across a few hundred sentences (full `etc/test_sentences/en.txt` + a held-out validation-split sample), not 2.
- Run a paired statistical test (Wilcoxon signed-rank or paired t-test) between variants/checkpoints, report effect size and confidence intervals — not a single point-estimate comparison.

### 1.4 Human evaluation (MOS study)
At least one real listening study with ~20–30 raters, randomized sample order, A/B or MOS rating. Q2 reviewers will almost always push back on naturalness claims backed only by automated metrics (UTMOS).

---

## 2. Should-have (strengthens the submission, reduces reviewer pushback)

### 2.1 Generalization test
Evaluate on at least one dataset/speaker outside LJSpeech, to preempt the "only validated on 24h, single speaker" critique.

### 2.2 Reproducibility artifacts
Fixed configs/seeds, clean repo, ideally public release of code + checkpoint. Increasingly expected by Q2-tier reviewers.

### 2.3 Clear novelty positioning in Related Work
State explicitly whether this exact combination (Snake1d + MRD + VITS2-style flow/duration-discriminator + explicit F0 conditioning) for single-speaker, low-resource TTS has been published before. If not, that gap is the paper's contribution — needs to be argued, not just implied by a components list.

---

## 3. Risks to plan around

- **Timeline:** Q2 journal review cycles commonly run 3–6+ months; this will likely extend past the graduation deadline. Treat the paper as a track that may continue after the capstone is submitted, not something to finish in parallel with it.
- **Venue selection:** SCImago quartiles (Q1/Q2/Q3) are recalculated yearly — verify the current quartile of any candidate journal directly on scimagojr.com before committing; do not rely on a journal's reputation from a prior year.
- **Team allocation:** items 1.1 (baseline) and 1.2 (ablations) are the most compute/time-expensive and block everything else (evaluation, paper writing depend on having these results first) — prioritize splitting these across the team earliest.

---

## 4. Suggested order of execution

1. Finish the WER + UTMOS eval harness (already scoped in [evaluation_methodology.md](evaluation_methodology.md)).
2. Train the baseline (vanilla VITS/VITS2) on identical data/budget.
3. Train the ablation variants.
4. Run the full evaluation suite (WER, UTMOS, statistical tests) across baseline + ablations + full model.
5. Design and run the human MOS study.
6. Write up: position novelty in Related Work, report ablation results, discuss limitations (single-speaker, data scale) honestly.
7. Select 2–3 candidate journals, verify current SCImago quartile, check scope fit before submission.
