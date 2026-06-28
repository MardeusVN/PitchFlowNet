# Draft: 4.1 Datasets and 4.2 Preprocessing

**Date:** 2026-06-26
**Status:** First draft — citations verified 2026-06-26 (see References); pending
journal template pass

---

## 4.1 Datasets

We train on the LJSpeech corpus (Ito and Johnson, 2017), a public-domain,
single-speaker English audiobook-reading dataset distributed as 22,050 Hz mono WAV
recordings with a pipe-delimited transcript file. Each row provides three fields — an
identifier, a raw transcription, and a normalized transcription with numbers,
ordinals, and monetary units expanded into words (Ito and Johnson, 2017); our
pipeline reads the third (normalized-transcription) field of each row as the G2P
input text. Of the 13,100 nominal recordings, preprocessing yields *N* = 12,895
utterances totaling 23.56 h of speech used for gradient updates — §4.2 details the
189 + 16 utterances excluded along the way. We randomly split these into 12,246
training, 644 validation (5%), and 5 fixed test utterances.

**Table 1.** Dataset statistics (*N* = 12,911 after CSV parsing; *N* = 12,895
effectively used for gradient updates after the training-time phoneme-length cap, see
§4.2).

| Format | Sample rate | N (post-parsing) | N (effective, training) | Total duration | Duration range | Transcript-length range (post-parsing / effective) |
|---|---|---|---|---|---|---|
| LJSpeech (single-speaker) | 22,050 Hz | 12,911 | 12,895 | 23.56 h | 1.11–10.10 s | 12–6,293¹ / 12–187 characters |

¹ The 12–6,293-character range reflects all 12,911 post-parsing rows, including the
16 transcript-merging artifacts described in §4.2 (up to 6,293 characters each). All
16 exceed the training-time phoneme-length cap and are excluded before reaching the
model; the 12,895 effective training rows have a clean range of 12–187 characters.

## 4.2 Preprocessing

The pipeline applies five cleaning steps, in order: (1) exclusion of utterances with a
missing or zero-byte audio file; (2) Silero voice-activity-detection (VAD) (Silero
Team, 2024) trimming of leading/trailing silence, applied jointly with resampling to
the target 22,050 Hz; (3) grapheme-to-phoneme normalization via espeak-ng (espeak-ng
contributors, n.d.); (4) a training-time cap on phoneme-sequence length that discards
anomalously long utterances before batching; and (5) a cache-integrity utility that
scans cached audio/spectrogram tensors and deletes any that fail to deserialize. A
Tukey-fence speaking-rate outlier filter (keep if rate ∈ [Q1 − 2·IQR, Q3 + 2·IQR] per
speaker, rate computed as non-punctuation character count over VAD-measured speech
duration; Tukey, 1977) is implemented in the codebase but was not applied to the run
reported here; we note this because it is the step a reviewer might otherwise expect
to catch the transcript-merging artifacts described next — since it did not run, those
artifacts reach the training-time phoneme cap (step 4) instead.

A data-integrity artifact in the transcript-parsing routine of the upstream Piper
training pipeline that our preprocessing builds on* removes 189 utterances outright
and corrupts 16 surviving entries, whose transcript field contains text concatenated
from neighboring rows (up to 6,293 characters) misaligned with the single ~6 s audio
file each entry still references; for the 12,895 cleanly-parsed rows, the three-field
row structure is intact, so the field read as G2P input is reliably the normalized
transcription as intended. All 16 corrupted entries' merged transcripts phonemize to
between 529 and 17,853 phoneme ids, exceeding the training-time phoneme-length cap
(step 4, `max_phoneme_ids = 400`), so the data loader excludes every one of them
before a gradient update; no legitimately long, uncorrupted utterance is excluded by
this same cap (the longest phonemizes to 399 ids, one below the threshold). The
effective utterance count used for training is consequently *N* = 12,895, not
12,911 (Table 1, §4.1).

\* The transcript file is read with Python's default `csv.reader` quoting behavior
(`QUOTE_MINIMAL` rather than `QUOTE_NONE`); 1,052 unescaped double-quote characters
cause the reader to treat a `"` as an unterminated quote and fold subsequent
pipe-delimited lines into the current field. `git blame` confirms this behavior was
authored 2022-11-11 in the original Piper `preprocess.py` (then `larynx_train`),
predating this project — see Hansen (n.d.) in References. We leave it unpatched:
fixing it would change which rows survive parsing and would require a full
re-training for any valid before/after comparison against the Piper-Modern run
reported here.

Finally, two further tensors are derived from each utterance's waveform. A linear
spectrogram is computed via the short-time Fourier transform — FFT size, window size,
and hop size set to 1024, 1024, and 256 samples, respectively, following Kim, Kong,
and Son (2021) — and cached as the posterior encoder's input. At training time, an
80-band mel-scale spectrogram is derived from this same linear spectrogram through a
mel-filterbank projection and log compression, and used only as the reconstruction
target: the mel of the ground-truth audio segment is compared against the mel of the
generator's output, weighted by *c*<sub>mel</sub> = 45 in the total loss; it is not
itself fed to the posterior encoder. A per-frame fundamental-frequency (F0) contour is
separately extracted offline with the `pyworld` (Morise et al., 2016) DIO+StoneMask
pitch-tracking algorithm and cached; it conditions the decoder and serves as the
regression target for the F0 predictor introduced in Section 3. Because the F0 contour
must share the linear spectrogram's frame count and `pyworld` consistently returns one
more frame than the spectrogram at our hop length and sample rate, we edge-crop the
F0 contour to match, and linearly interpolate unvoiced (F0 = 0) frames in the log
domain so the predictor is not trained to regress toward zero in silence gaps.

---

## References

- Ito, K., & Johnson, L. (2017). *The LJ Speech Dataset*.
  https://keithito.com/LJ-Speech-Dataset/ — verified 2026-06-26 directly from the
  author's page, which gives this exact citation (no DOI; it's a dataset release, not
  a peer-reviewed paper).
- Kim, J., Kong, J., & Son, J. (2021). Conditional variational autoencoder with
  adversarial learning for end-to-end text-to-speech. In *Proceedings of the 38th
  International Conference on Machine Learning* (Vol. 139, pp. 5530–5540). PMLR.
  https://proceedings.mlr.press/v139/kim21f.html — verified 2026-06-29 directly from
  the PMLR proceedings page. Source of the FFT/window/hop-size convention (1024,
  1024, 256) and the linear-spectrogram-to-posterior-encoder /
  mel-spectrogram-as-reconstruction-target design our pipeline follows.
- Morise, M., Yokomori, F., & Ozawa, K. (2016). WORLD: A vocoder-based high-quality
  speech synthesis system for real-time applications. *IEICE Transactions on
  Information and Systems*, E99-D(7), 1877–1884.
  https://doi.org/10.1587/transinf.2015EDP7457 — verified 2026-06-26 via CrossRef.
- Tukey, J. W. (1977). *Exploratory Data Analysis*. Reading, MA: Addison-Wesley.
  ISBN 0-201-07616-0 — verified 2026-06-26 via WorldCat (OCLC 3058187) and Open
  Library catalogue records. Canonical source for the IQR/Tukey-fence outlier
  criterion implemented (but not applied to the reported run) in the speaking-rate
  filter described above.
- Silero Team. (2024). *Silero VAD: Pre-trained Enterprise-Grade Voice Activity
  Detector (VAD), Number Detector and Language Classifier* [Software]. GitHub
  repository. https://github.com/snakers4/silero-vad — verified 2026-06-26 directly
  from the repository's own README, which provides this exact BibTeX entry (no
  peer-reviewed paper exists; cite the repository).
- espeak-ng contributors. (n.d.). *eSpeak NG Text-to-Speech* [Software]. GitHub
  repository. https://github.com/espeak-ng/espeak-ng — checked 2026-06-26: the
  repository has **no CITATION file or formal academic citation block** (originally
  Jonathan Duddington's "speak"/eSpeak, now maintained by the espeak-ng organization
  under GPL-3.0). Cite as a software footnote/URL only; do not fabricate a year or
  author list beyond what the repo states.
- Hansen, M. (n.d.). *Piper: A Fast, Local Neural Text-to-Speech System* [Software].
  GitHub repository. https://github.com/OHF-Voice/piper1-gpl — checked 2026-06-26:
  confirmed via `git blame` on our fork that the transcript-parsing routine
  responsible for the CSV-quoting artifact above (line 437 of the original
  `preprocess.py`, then under the `larynx_train`/early `piper_train` name) predates
  our project, authored 2022-11-11. The original repository (`rhasspy/piper`) has
  since moved development to `OHF-Voice/piper1-gpl`; neither repository has a
  CITATION file or formal academic citation block. Cite as a software footnote/URL
  only.

## Notes for next pass

- [ ] Word count: **~722 words** (prose only across both subsections, excluding
      table/footnotes/references) as of 2026-06-29 — well past the 300–500 word
      target (now split across two subsections, see below). Growth this pass came
      from: (1) correcting the cleaning-step list (rate filter was never actually
      applied to the reported run — removed from the 6-step list, replaced with a
      one-sentence transparency note); (2) rewriting the mel/F0 paragraph after
      comparing against the original VITS paper (Kim et al., 2021) — the prior draft
      incorrectly implied the 80-band mel-spectrogram feeds the posterior encoder;
      verified against `lightning.py`/`norm_audio/__init__.py` that the *linear*
      spectrogram (`.spec.pt`, 1024/1024/256 FFT/window/hop) is the posterior-encoder
      input, and the mel-spectrogram is derived from it only at training time as the
      reconstruction-loss target, matching VITS exactly; (3) adding explicit
      train/val/test split counts (12,246/644/5), mirroring VITS's own transparency
      convention. Needs a trim pass before submission: candidates are the closing
      "We leave the underlying csv.reader behavior itself unpatched..." sentence
      (could move to a footnote) and the transparency note on the unapplied rate
      filter (could shorten once reviewers confirm it isn't needed).
- [x] Split into separate §4.1 Datasets / §4.2 Preprocessing subsections (2026-06-29),
      mirroring the original VITS paper's structure — §4.1 now holds only corpus
      facts, split counts, and Table 1; §4.2 holds the cleaning-step list, CSV-bug
      forensics, and the linear-spec/mel/F0 feature-extraction paragraph.
- [x] Renumbered 3.1/3.2 → 4.1/4.2 (2026-06-29): fetched VITS's actual section
      structure (ar5iv full text) and confirmed Datasets/Preprocessing are
      subsections of VITS's "3 Experiments" (with Method/architecture as their
      Section 2), not their Method section. Our paper's analogous section is
      "4. Experimental Setup," so Dataset/Preprocessing moved there; Section 3 is
      now reserved solely for the Piper-Modern architecture description. Updated the
      Introduction's roadmap paragraph (`introduction-draft_sec1.md`) to match. The
      F0-predictor forward-reference now says "introduced in Section 3" (no
      subsection decimal) since Architecture's internal subsection numbering isn't
      drafted yet — fill in the exact §3.x once it is.
- [ ] Confirm journal/citation-style template before final formatting pass (author-
      year locked in per project convention; numbered-style not used). Note: Silero
      VAD and espeak-ng are software citations without a fixed year/edition — confirm
      target venue's convention for citing software (e.g. APA software citation
      format) once the journal is chosen.
