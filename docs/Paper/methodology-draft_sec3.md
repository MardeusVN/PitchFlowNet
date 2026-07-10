# Draft: Section 3 — Methodology, Model Development, and Evaluation

---

## I. Methodology

### 1.1 Model Selection

We adopt VITS [1] as the baseline architecture, an end-to-end TTS system that jointly trains acoustic modelling and waveform synthesis within a single conditional variational autoencoder (CVAE). The text encoder produces a prior distribution over the latent space; monotonic alignment search (MAS) estimates soft phoneme-to-frame alignment without an external aligner; a normalizing flow refines the prior; a posterior encoder processes the linear spectrogram to produce a training-time approximate posterior; and a HiFi-GAN-style decoder [2] upsamples the sampled latent to a raw waveform. The adversarial objective is provided by a multi-period discriminator (MPD). All experiments are implemented on the Piper training framework [3] as the implementation foundation.

Section 2.1 introduces the architectural extensions evaluated through a factorial ablation study.


### 1.2 Data Splitting Strategy

We train on LJSpeech [8] (13,100 utterances, 23.92 h, 22,050 Hz mono). The corpus is partitioned into three non-overlapping subsets using a reproducible random permutation, with split membership held identical across all eight ablation configurations:

| Split | Count | Purpose |
|---|---|---|
| Train | 12,500 | Parameter optimisation |
| Validation | 100 | `val_loss_mel` monitoring; best-checkpoint selection |
| Test | 500 | Post-training WER, UTMOS, and RTF evaluation (held-out) |

This matches the split sizes reported in [1], though utterance membership differs because we apply an independent random permutation rather than reusing the original VITS file-ID list; our WER and UTMOS results are therefore not directly comparable to any figures published in [1] on a per-utterance basis. The checkpoint selection signal is `val_loss_mel` — the non-adversarial L1 mel reconstruction loss — rather than the composite `val_loss`, whose adversarial terms do not correlate monotonically with perceptual quality.

### 1.3 Feature Engineering and Selection

**Phoneme representation.** Text is converted offline to phoneme ID sequences via espeak-ng [9] (`en-us`). A hard cap of `max_phoneme_ids = 400` is enforced at training time; after correcting the upstream CSV parsing bug (Section 4.2), all 13,100 utterances fall within this limit (min 25, max 399 IDs).

**Silence removal.** Silero VAD [10] trims leading and trailing silence at 16 kHz; the detected speech segment is resampled to 22,050 Hz for training.

**Spectral features.** A linear-scale STFT is computed offline (FFT 1,024, window 1,024, hop 256 samples at 22,050 Hz) following [1], yielding a 513-bin magnitude spectrogram as the posterior encoder input. An 80-band mel spectrogram is derived at training time via triangular filterbank and log compression; it is used only as the reconstruction target ($c_{\text{mel}} = 45$) and is not fed to the posterior encoder.

**Fundamental frequency.** Per-frame F0 contours are extracted offline using WORLD DIO + StoneMask [11] (frame shift ≈ 11.61 ms). Unvoiced frames carry F0 = 0 by convention (DIO output); these are linearly interpolated in the log domain before caching. The contour is then truncated or edge-padded to match the spectrogram frame count (pyworld consistently returns one extra frame at our hop length). F0 features are consumed only by configurations with `use_f0 = true`.

**Waveform segments.** Random 8,192-sample segments (≈ 0.37 s at 22,050 Hz) are extracted per utterance and passed to the generator and discriminators. Segment size is fixed across all configurations.

---

## II. Model Development

### 2.1 Model Architecture

**Generator — SynthesizerTrn.** The generator consists of four modules: (1) a *text encoder* based on a relative-positional-encoding transformer (hidden 192, filter 768, 6 layers, 2 attention heads, kernel 3, dropout 0.1) that produces prior distribution parameters $(m_p, \log \sigma_p)$ and hidden state $x$; (2) a *posterior encoder* built on a dilated WaveNet stack (kernel 5, dilation 1, 16 layers, hidden 192) that maps the linear spectrogram to an approximate posterior $(m_q, \log \sigma_q)$ and samples the latent $z$; (3) a *normalizing flow* (`ResidualCouplingBlock`, 4 coupling layers, kernel 5, dilation 1, 4 WN layers per coupler) that maps between posterior and prior spaces; and (4) a *HiFi-GAN decoder* that upsamples the latent via three transposed convolutions (rates 8 × 8 × 4, initial channels 256) with multi-receptive-field (MRF) ResBlock2 branches (kernel sizes 3, 5, 7; dilation groups (1,2), (2,6), (3,12)). When `use_bigvgan = true`, each transposed convolution is preceded by a SnakeBeta activation [4] and the final non-linearity switches from LeakyReLU (slope 0.1) to SnakeBeta. SnakeBeta extends Snake [5] with decoupled per-channel frequency parameter $\alpha$ and magnitude parameter $\beta$: $\text{SnakeBeta}(x) = x + \beta^{-1} \sin^2(\alpha x)$. When `use_vits2 = true`, each residual coupling layer's conditioner network first applies a single-layer self-attention pass for global context, then feeds the result into the local WaveNet conditioner before computing the affine parameters, following the coarse-to-fine ordering in [7]. When `use_f0 = true`, a zero-initialized `f0_cond` Conv1d projects the log-F0 slice into the decoder's input channel space as an additive conditioning signal.

**Duration predictor.** The stochastic duration predictor (SDP) is a 4-flow normalizing flow with DDSConv conditioner (kernel 3, 4 layers, filter 192) that models the posterior distribution over phoneme durations during training and samples durations stochastically at inference.

**F0 predictor.** When `use_f0 = true`, a two-layer Conv1d network (hidden 256, kernel 3, dropout 0.5) regresses per-phoneme log-F0 from the text encoder's hidden state; gradients flow back through the predictor into the text encoder so that the shared representation is jointly optimised for both text encoding and pitch prediction. Ground-truth per-phoneme F0 is obtained by projecting the frame-level F0 contour through the MAS attention matrix.

**Discriminators.** The baseline MPD contains one scale sub-discriminator (`DiscriminatorS`, 6 Conv1d layers) and five period sub-discriminators (`DiscriminatorP`, periods [2, 3, 5, 7, 11], 5 Conv2d layers each), totalling 46.7 M parameters. When `use_bigvgan = true`, a `MultiResolutionDiscriminator` [4, 6] adds three `DiscriminatorR` instances operating at STFT resolutions (512/50/240), (1024/120/600), and (2048/240/1200) as 2D convolutional networks on magnitude spectrograms (280 K parameters). When `use_vits2 = true`, a `DurationDiscriminator` [7] (two Conv1d layers, hidden 192, kernel 3) scores each phoneme's predicted log-duration as real or fake, conditioned on the text encoder hidden state; it adds zero inference cost as it is discarded after training. When `use_vits2 = true`, the MAS alignment scores are additionally perturbed with adaptive Gaussian noise during training to improve alignment robustness [7]. The noise standard deviation is $\alpha_t \cdot \sigma_{\text{neg\_cent}}$, where $\sigma_{\text{neg\_cent}}$ is the standard deviation of the current batch's alignment score matrix and $\alpha_t = \max(0,\; 0.01 - t \cdot 2 \times 10^{-6})$ decays linearly with global training step $t$, reaching zero at step 5,000 (approximately epoch 13). This schedule transitions MAS from soft alignment exploration in early training to hard Viterbi alignment later, following [7].

**Ablation configurations.** The three flags define a full 2³ factorial: eight configurations covering all combinations of the three research directions.

| Config | `use_bigvgan` | `use_vits2` | `use_f0` |
|---|:---:|:---:|:---:|
| A (baseline) | ✗ | ✗ | ✗ |
| B | ✓ | ✗ | ✗ |
| C | ✗ | ✓ | ✗ |
| D | ✗ | ✗ | ✓ |
| E | ✓ | ✓ | ✗ |
| F | ✓ | ✗ | ✓ |
| G | ✗ | ✓ | ✓ |
| H | ✓ | ✓ | ✓ |

All configurations share identical model dimensions, optimiser settings, and training budget. Config A serves as the common reference for all comparisons.

### 2.2 Training Procedure

**Optimiser.** Generator and discriminators are each optimised with AdamW ($\text{lr} = 2 \times 10^{-4}$, $\beta_1 = 0.8$, $\beta_2 = 0.99$, $\varepsilon = 10^{-9}$). Both schedulers follow an exponential decay ($\gamma = 0.999875$) stepped once per epoch; at epoch 1,100 the learning rate is approximately 87% of its initial value.

**GAN training procedure.** Manual optimization is used: per batch, the generator backward pass runs first, followed by the discriminator backward pass (generator-first ordering is an explicit implementation choice, deviating from the discriminator-first convention in [1]). The generator objective is:

$$\mathcal{L}_{\text{gen}} = \mathcal{L}_{\text{adv,MPD}} + \mathcal{L}_{\text{adv,MRD}} + \mathcal{L}_{\text{fm}} + c_{\text{mel}} \cdot \mathcal{L}_{\text{mel}} + \mathcal{L}_{\text{dur}} + c_{\text{kl}} \cdot \mathcal{L}_{\text{kl}} + \mathcal{L}_{\text{dur\_gen}} + \mathcal{L}_{\text{f0}}$$

where $c_{\text{mel}} = 45$, $c_{\text{kl}} = 1.0$; $\mathcal{L}_{\text{adv,MRD}}$ is zero when `use_bigvgan = false`; $\mathcal{L}_{\text{dur\_gen}}$ is zero when `use_vits2 = false`; $\mathcal{L}_{\text{f0}}$ (per-phoneme log-F0 MSE) is zero when `use_f0 = false`. Adversarial losses follow the least-squares GAN formulation [12]: the discriminator minimises $\mathbb{E}[(1 - D(y))^2] + \mathbb{E}[D(\hat{y})^2]$ and the generator minimises $\mathbb{E}[(1 - D(\hat{y}))^2]$, applied independently to MPD and MRD. $\mathcal{L}_{\text{mel}}$ is L1 on 80-band mel spectrograms; $\mathcal{L}_{\text{fm}}$ is mean absolute difference on intermediate discriminator feature maps; $\mathcal{L}_{\text{kl}}$ is the KL divergence between posterior and prior; $\mathcal{L}_{\text{dur}}$ is the SDP negative log-likelihood.

Due to a cuFFT limitation, STFT computations within MRD and mel loss are cast to float32 even under bfloat16 mixed-precision.

**Hardware.** All runs use 2× NVIDIA RTX 4070 Ti (12 GB each), DDP (NCCL), bfloat16 mixed precision, batch size 16, global RNG seed 1234 (independent from the dedicated data-split generator described in Section 1.2), trained from scratch for 1,100 epochs with checkpoints saved every epoch.

---

## III. Model Evaluation and Fine-Tuning

### 3.1 Evaluation Metrics

All configurations are evaluated post-training on the 500-sample held-out test set via an automated harness that synthesises audio from raw text (no ground-truth audio is used) and scores the result using three core metrics applicable across all eight configurations, plus a fourth metric specific to the F0-conditioned configuration (Config I). Signal-level metrics such as PESQ and STOI are excluded: both require a time-aligned ground-truth waveform at identical duration, which TTS synthesis does not provide — the model generates audio of variable length from text alone, with no reference waveform to compare against.

**Word Error Rate (WER).** Synthesised audio is transcribed by Whisper `base.en` [13] and compared against the reference transcript using jiwer. Both hypothesis and reference are normalised identically before scoring: lowercased, then all characters outside `[a-z0-9' ]` removed, and whitespace collapsed. WER quantifies intelligibility — lower is better.

**UTMOS.** Synthesised audio is scored by UTMOSv2, a pretrained neural MOS predictor. UTMOS serves as a perceptual naturalness proxy on a 1–5 scale — higher is better. It enables large-scale comparison across ablation configurations without crowdsourced listener studies; absolute UTMOS values should be interpreted as relative indicators between configurations rather than ground-truth MOS estimates.

**Real-Time Factor (RTF).** Inference wall-clock time divided by audio duration, measured on CPU. RTF is reported as a mean over all 500 utterances and quantifies deployment cost — lower is better. RTF is not subject to per-sentence significance testing, as it reflects aggregate throughput rather than a per-utterance distribution.

**F0 Prediction Accuracy (Config I only).** For the F0-conditioned configuration, predicted per-phoneme log-F0 is evaluated against ground-truth per-phoneme F0 obtained via the same MAS-projection procedure used during training (Section 2.1). Two metrics are reported, computed only over voiced phonemes (ground-truth F0 > 0):

*F0 RMSE* — root-mean-square error in the log-F0 domain:

$$\text{F0 RMSE} = \sqrt{\frac{1}{N}\sum_{i=1}^{N}\left(\log f_{0,\text{pred},i} - \log f_{0,\text{gt},i}\right)^2}$$

*F0 Correlation* — Pearson correlation between predicted and ground-truth log-F0 sequences, measuring pitch contour shape independent of absolute offset:

$$\rho = \frac{\sum_i(\log f_{0,\text{pred},i} - \overline{\log f_{0,\text{pred}}})(\log f_{0,\text{gt},i} - \overline{\log f_{0,\text{gt}}})}{\sqrt{\sum_i(\log f_{0,\text{pred},i}-\overline{\log f_{0,\text{pred}}})^2}\,\sqrt{\sum_i(\log f_{0,\text{gt},i}-\overline{\log f_{0,\text{gt}}})^2}}$$

Both metrics are computed on the same 500-sample test set, aggregated across all phonemes rather than per-sentence, since individual sentences may contain too few voiced phonemes for a stable per-sentence estimate.

Statistical significance between any two configurations is assessed via a paired Wilcoxon signed-rank test on the 500 per-sentence WER and UTMOS score distributions ($p < 0.05$).

### 3.2 Hyperparameter Tuning

Optimiser hyperparameters ($\text{lr}$, $\beta_1$, $\beta_2$, $\gamma$, $c_{\text{mel}}$, $c_{\text{kl}}$) are adopted directly from [1] and held fixed across all configurations; modifying them per configuration would confound architectural attribution.

Architecture selection is performed through the factorial ablation design (Section 2.1) rather than a hyperparameter search. Each architectural flag is toggled independently and in combination to isolate its marginal contribution to WER and UTMOS. The final Banhmi-TTS architecture is selected as the configuration achieving the lowest WER among those showing no statistically significant UTMOS degradation relative to the baseline (Config A), per the Wilcoxon test described in Section 3.1.

---

## References

[1] J. Kim, J. Kong, and J. Son, "Conditional variational autoencoder with adversarial learning for end-to-end text-to-speech," in *Proc. ICML*, PMLR, vol. 139, pp. 5530–5540, 2021.

[2] J. Kong, J. Kim, and J. Bae, "HiFi-GAN: Generative adversarial networks for efficient and high fidelity speech synthesis," in *Advances in NeurIPS*, vol. 33, pp. 17022–17033, 2020.

[3] M. Hansen, "Piper: A fast, local neural text-to-speech system," Open Home Foundation [Software], 2023. https://github.com/OHF-Voice/piper1-gpl

[4] S. Lee, W. Ping, B. Ginsburg, B. Catanzaro, and S. Yoon, "BigVGAN: A universal neural vocoder with large-scale training," in *Proc. ICLR*, 2023.

[5] L. Ziyin, T. Hartwig, and M. Ueda, "Neural networks fail to learn periodic functions and how to fix it," in *Advances in NeurIPS*, vol. 33, pp. 1583–1594, 2020.

[6] W. Jang, D. Lim, J. Yoon, B. Kim, and J. Kim, "UnivNet: A neural vocoder with multi-resolution spectrogram discriminators for high-fidelity waveform generation," in *Proc. Interspeech 2021*, pp. 2207–2211, 2021. DOI: 10.21437/Interspeech.2021-1016

[7] J. Kong et al., "VITS2: Improving quality and efficiency of single-stage text-to-speech with adversarial learning and architecture design," in *Proc. Interspeech 2023*, pp. 4374–4378, 2023. DOI: 10.21437/Interspeech.2023-534

[8] K. Ito and L. Johnson, "The LJ Speech Dataset" [Dataset], 2017. https://keithito.com/LJ-Speech-Dataset/

[9] espeak-ng contributors, *eSpeak NG Text-to-Speech* [Software]. https://github.com/espeak-ng/espeak-ng

[10] Silero Team, "Silero VAD: pre-trained enterprise-grade voice activity detector" [Software], 2024. https://github.com/snakers4/silero-vad

[11] M. Morise, F. Yokomori, and K. Ozawa, "WORLD: A vocoder-based high-quality speech synthesis system for real-time applications," *IEICE Trans. Inf. Syst.*, vol. E99-D, no. 7, pp. 1877–1884, 2016. DOI: 10.1587/transinf.2015EDP7457

[12] X. Mao, Q. Li, H. Xie, R. Y. K. Lau, Z. Wang, and S. P. Smolley, "Least squares generative adversarial networks," in *Proc. ICCV*, pp. 2794–2802, 2017.

[13] A. Radford et al., "Robust speech recognition via large-scale weak supervision," in *Proc. ICML*, PMLR, vol. 202, pp. 28492–28518, 2023.
