# Draft: Section 5 — Exploratory Data Analysis

**Date:** 2026-07-01
**Status:** Draft — figures finalized, text clean of preprocessing references.

---

## 5. Exploratory Data Analysis

### 5.1 Summary of Key Findings

EDA was performed across five dimensions on the full corpus of N = 13,100
utterances after applying the CSV QUOTE_NONE fix described in Section 4.1.

**5.1.1 Audio Duration (Figure 1)**
All 13,100 raw clips are present (zero missing WAV files; total raw duration
23.92 h). Durations span 1.11–10.10 s with a median of 6.76 s and a broad,
roughly unimodal distribution that peaks around 7–8 s.

**5.1.2 Speaking Rate (Figure 2)**
Computed on 13,100 clips (characters per second), speaking rate follows a
near-Gaussian distribution (median 15.25 chars/s, range ≈5–22.5 chars/s).
Standard Tukey fence at 1.5 × IQR flags 203 utterances (1.55%) at the tails.

**5.1.3 Silence Ratio (Figure 3)**
Measured on 13,100 clips. Leading silence is negligible across the corpus
(median 0.0%). Trailing silence is more variable (median 1.3%, maximum ≈13%),
with a long exponential tail. Total silence per clip extends to ≈20% for a
small number of clips.

**5.1.4 Phoneme Sequence Length (Figure 4)**
Measured on all 13,100 post-parsing utterances. Lengths follow a broad
bell-shaped distribution (mean 214.4, median 219.0 phoneme IDs). All
utterances fall within the training cap of 400 IDs (min 25, max 399); no
entries exceed the cap.

**5.1.5 Corpus-Level F0 (Figure 5)**
Measured on all 13,100 utterances (zero F0 cache failures). Mean F0 per
utterance is tightly concentrated around 206.4 Hz (corpus mean 206.7 Hz,
range 144–393 Hz), consistent with a single female speaker. Pitch range per
utterance (max − min F0) follows a right-skewed distribution (median 258.1 Hz,
extending beyond 700 Hz for the most expressive clips).

---

### 5.2 Visualizations

**Figure 1** (`01_duration_distribution.png`) — Distribution of audio clip
durations (N = 13,100; median 6.76 s).

**Figure 2** (`04_speaking_rate_distribution.png`) — Speaking rate distribution
(N = 13,100; median 15.25 chars/s).

**Figure 3** (`06_silence_ratio_distribution.png`) — Silence ratio distributions
(N = 13,100): leading (median 0.0%), trailing (median 1.3%), total (median 1.4%).

**Figure 4** (`07_phoneme_length_distribution.png`) — Phoneme sequence length
distribution (N = 13,100; all entries within 400-ID cap; max 399 IDs).

**Figure 5** (`08_f0_corpus_distribution.png`) — Corpus-level F0 (N = 13,100):
(left) mean F0 per utterance (median 206.4 Hz); (right) pitch range per
utterance (median 258.1 Hz).
