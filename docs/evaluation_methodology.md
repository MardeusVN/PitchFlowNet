# Evaluation Methodology: RTF, WER, UTMOS

This document summarizes the evaluation approach decided on for comparing checkpoints of the current VITS-based (Piper-Modern) model, and for positioning it against other TTS systems. It also records the directions that were considered and explicitly **not** pursued, and why.

---

## 1. Why RTF / WER / UTMOS (and not RL post-training)

Modern LLM-style TTS systems (Qwen3-TTS, CosyVoice) use a 3-stage post-training pipeline: DPO (preference alignment), rule-based rewards + GSPO/GRPO (stability), and lightweight speaker fine-tuning. These techniques require a model with a **per-token policy / log-probability** (autoregressive decoding over discrete audio tokens), so that a preference or reward signal can be turned into a policy-gradient update.

The current architecture (`src/python/piper_train/vits/models.py`) is **non-autoregressive**: TextEncoder → normalizing flow (prior) → Stochastic Duration Predictor + F0 Predictor → GAN-based Generator (Snake1d, MPD/MRD discriminators). There is no token-level policy to optimize, so DPO/GSPO do not transplant onto this architecture without a fundamental redesign of the decoder. Decision: **do not implement RL-based post-training**; if mentioned in the report, frame it as a deliberately descoped direction with the architectural reason above, not as future work to actually attempt.

A lightweight "instruction control" layer (e.g. "nói giọng trầm ấm" → bias pitch/speed) was also considered. It is technically feasible as a rule-based mapper on top of existing/added scale knobs (`length_scale`, a to-be-added `f0_scale`/`energy_scale`), but it does not train the model to sound better — it only re-scales what the model already predicts, so it contributes to **controllability/demo value**, not to baseline naturalness, and carries a risk of sounding less natural if pushed outside the training distribution. Decision: **deprioritized**, not part of the current evaluation focus.

Conclusion: focus evaluation effort on **RTF, WER, UTMOS** — these map directly onto efficiency, correctness/stability, and naturalness, without requiring any architecture change up front.

---

## 2. RTF (Real-Time Factor)

- Already measured via `say.py` (`real_time_factor = infer_sec / audio_duration_sec`).
- Current CPU measurements: ~0.08–0.09 (i.e. ~11-12x faster than real time) for both the epoch-124 and epoch-217 checkpoints.
- This is a structural advantage of the non-autoregressive architecture vs. autoregressive LLM-codec TTS (which typically has much higher RTF). No further optimization needed — just report it.

---

## 3. WER (Word Error Rate) — correctness/stability gate

- **Method:** run an ASR model (Whisper) on each generated `.wav`, compare the transcription against the original input text, compute WER.
- **Role:** WER replaces the "rule-based stability" goal (catch dropped words, repetitions, mispronunciations) that the RL pipeline's stage 2 was meant to address — without needing RL. Treat it as a **gate**: a sample/checkpoint must pass an acceptable WER before its UTMOS score is meaningful to compare.
- **Specific open question to resolve with WER:** checkpoint epoch 217 reads noticeably faster than epoch 124 for the same sentence (e.g. 12.48s vs 15.21s, and 4.88s vs 6.12s on two test sentences) despite subjectively sounding clearer. WER on a larger sentence set will confirm whether the faster duration-predictor output is dropping/garbling words or is a benign change in reading rhythm.

---

## 4. UTMOS — naturalness metric

UTMOS is a neural, no-reference MOS predictor (score range 1–5, higher = more natural/human-like).

### 4.1 Which model to use

- **UTMOSv2** (`sarulab-speech/UTMOSv2`) — newer, better correlation with human MOS. Use as the primary scorer.
- **UTMOS22 "strong learner"** (`sarulab-speech/UTMOS22`, or the unofficial `pip install utmos` wrapper) — this is the version used by the Trelis Research 2026 benchmark referenced below. Needed only if direct comparison to those published numbers matters.
- **Hard rule:** use exactly one UTMOS model/version across every audio sample being compared in a given table. Scores from different UTMOS versions are not cross-comparable.

### 4.2 Reference numbers found (for context only — not directly comparable to each other)

| Model | Score | Metric / source | Test set |
|---|---|---|---|
| VITS (original paper) | ~4.43–4.54 MOS | Human MOS, Kim et al. 2021 | Clean LJSpeech held-out sentences |
| VITS2 (paper) | reported improvement over VITS1, no absolute UTMOS number found | Human MOS, Kong et al. 2023 | LJSpeech / multi-speaker sets |
| Piper (VITS/VITS2-based) | 3.3 UTMOS, 28% CER | UTMOS22, Trelis Research (2026) | "Tricky TTS" — adversarial test set (numbers, symbols, technical text) |
| Kokoro-82M | 4.5 UTMOS, 17% CER | UTMOS22, Trelis Research (2026) | Same "Tricky TTS" set |

**Caveat:** the VITS/VITS2 numbers are *human* MOS on clean, in-domain LJSpeech sentences; the Piper/Kokoro numbers are *UTMOS-predicted* scores on a deliberately adversarial test set. These two groups of numbers must not be compared directly to each other — they are included here only as separate points of reference, not a ranking.

### 4.3 Dataset / sentences to use for our own evaluation

- **Internal comparison (own checkpoints, e.g. epoch 124 vs 217 vs future):** `etc/test_sentences/en.txt` (already in repo) plus a sample of the held-out validation split (5% of LJSpeech, set via `--validation-split 0.05` in training) — the validation utterances have real ground-truth recordings, useful for WER reference too.
- **External comparison (vs. other TTS models):** the *same* fixed sentence set must be fed to every model being compared (our checkpoint + any reference model), then scored with the *same* UTMOS model. Use `en.txt` since it is generic text, not LJSpeech-specific, so it doesn't unfairly favor our model.

### 4.4 Candidate external models for comparison

Self-hostable, no API cost:
- **Kokoro-82M** — lightweight, fast, highest UTMOS in the Trelis benchmark above.
- **XTTS-v2** (Coqui) or **CosyVoice** — larger, multi-speaker/zero-shot open-source baselines.

Optional, paid/API (not required):
- ElevenLabs, OpenAI TTS — usable for context in the report, but cost money and have their own ToS; not necessary for the core evaluation.

**Framing caveat for the report:** our model is single-speaker, trained on ~24h of data; external reference models are typically multi-speaker/multilingual, trained on orders of magnitude more data. Position any comparison as an **efficiency/quality trade-off** (our RTF is dramatically better) rather than a claim of outright superiority.

---

## 5. Next steps (not yet executed)

1. Build the eval harness: Whisper-based WER + UTMOSv2 scoring, run over `etc/test_sentences/en.txt` + a validation-split sample.
2. Run it on existing checkpoints (epoch 124 "best", epoch 217 "current", and the 4 `.wav` files already generated) to get a baseline.
3. Resolve the open question above: does epoch 217's faster reading rate correlate with higher or lower WER than epoch 124?
4. Optionally extend to Kokoro-82M for an external reference point, using the same sentence set and same UTMOS model version.
