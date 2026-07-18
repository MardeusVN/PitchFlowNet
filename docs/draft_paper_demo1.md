# Banhmi-TTS: Architectural Extensions to VITS/Piper for Edge-Deployable Single-Speaker TTS

---

## Abstract

We present **Banhmi-TTS**, a single-speaker text-to-speech system built on the VITS/Piper backbone that investigates whether audio quality can be improved without sacrificing real-time, CPU-only deployability. Starting from the VITS conditional variational autoencoder (Kim et al., 2021), we apply targeted architectural modifications across three independently controlled dimensions: (1) replacing the HiFi-GAN decoder's LeakyReLU activations with BigVGAN's SnakeBeta and adding a Multi-Resolution Discriminator (Lee et al., 2023); (2) augmenting the normalizing-flow coupling layers with transformer-based conditioners, introducing adversarial duration supervision, and stabilizing monotonic alignment search with Gaussian-noise perturbation, following VITS2 (Kong et al., 2023); and (3) conditioning the decoder on explicit per-phoneme log-F0 predictions, following FastPitch (Łańcucki et al., 2021). At inference, the full model is 18 M parameters (25.2 M including the training-only posterior encoder), one to two orders of magnitude smaller than current autoregressive systems. We compare the complete model with the unmodified VITS/Piper baseline on 100 held-out sentences using WER, UTMOS, and real-time factor. The complete model improves mean UTMOS from 3.550 to 3.619 ($p=0.0056$), while its WER reduction from 8.18% to 7.35% is not statistically significant ($p=0.1821$). Intermediate component combinations remain planned future ablations and are not presented as measured results.

---

## 1. Introduction

Text-to-speech (TTS) systems have advanced rapidly in recent years, in part because they are a building block for speech-to-speech (S2S) systems (Jia et al., 2019; Lee et al., 2022) that aim to let machines communicate with the prosodic and emotional nuance of human speakers rather than reading text aloud in a flat, affectless voice. Progress toward that goal has, however, split TTS systems along an axis that receives less attention than raw audio quality: deployment cost. Autoregressive language models generate expressive, controllable speech by predicting discrete acoustic tokens one at a time with a large-parameter transformer conditioned on a dedicated speech tokenizer (Zeghidour et al., 2022; Défossez et al., 2022; Kumar et al., 2023). This design gives them strong prosody, voice control, and naturalness, capable of cloning a human voice realistically (Défossez et al., 2024; Du et al., 2025; Zhang et al., 2025a). Autoregressive decoding at that parameter count, however, targets GPU-class inference; it is not built for the CPU-only, low-power hardware (single-board computers, mobile and embedded devices) that edge deployment and low-latency operation require.

VITS (Kim et al., 2021) and its lightweight derivative Piper (Hansen, n.d.) occupy the opposite corner. Rather than following the conventional two-stage pipeline, in which a first stage produces an intermediate representation such as a mel-spectrogram from text and a second stage generates the raw waveform from that representation, VITS unifies both stages into a single non-autoregressive model trained end-to-end (Kim et al., 2021). It connects the two stages through latent variables under a conditional variational autoencoder (VAE), so that the intermediate representation is learned jointly rather than predefined. To increase the expressive power of the conditional prior, it applies normalizing flows to that prior distribution, and it trains the waveform decoder with adversarial objectives in the waveform domain. A stochastic duration predictor further lets the model synthesize speech with diverse rhythms from the same text, capturing the one-to-many relationship between an input and its many plausible spoken realizations. The result is a model on the order of tens of millions of parameters, small and fast enough to synthesize speech in real time on a CPU alone, including on a Raspberry Pi (Hansen, n.d.). The cost of this efficiency is audio quality and expressiveness that trail current autoregressive systems.

This paper asks whether that quality gap can be narrowed without sacrificing the training stability and the real-time, CPU-only deployability that make VITS and Piper practical in the first place, rather than closing it by scaling up toward an autoregressive, GPU-class model. We present **Banhmi-TTS**, which keeps the VITS/Piper backbone, and therefore its real-time, CPU-only inference profile, while making modifications across three architectural dimensions, drawn from three prior systems: BigVGAN (Lee et al., 2023), VITS2 (Kong et al., 2023), and FastPitch (Łańcucki et al., 2021), that have so far only been validated in multi-speaker, high-resource, or non-flow-based settings. At inference time, Banhmi-TTS has 18 M parameters (25.2 M including the training-only posterior encoder used for the variational objective), one to two orders of magnitude smaller than autoregressive systems such as the Qwen3-TTS 0.6 B- or 1.7 B-parameter model (Hu et al., 2026). The three architectural dimensions are as follows:

1. The generator's activation and discriminator design is replaced with a BigVGAN-style SnakeBeta activation (Ziyin et al., 2020; Lee et al., 2023) paired with UnivNet's multi-resolution discriminator (Jang et al., 2021), reusing BigVGAN's published recipe rather than introducing a new combination (Section 2.2).
2. The normalizing-flow prior is augmented with VITS2-style transformer coupling layers, the MAS is stabilized with Gaussian-noise perturbation, and the duration model is supplemented with VITS2's adversarial duration discriminator (Kong et al., 2023) (Section 2.3).
3. The conditioning signal supplied to the decoder is extended with FastPitch-style per-phoneme pitch conditioning (Łańcucki et al., 2021), adapted from a feed-forward decoder to the flow-based VITS decoder (Section 2.4).

Our contribution is empirical rather than architectural: to our knowledge, no published system combines all three source lineages inside a single end-to-end VITS pipeline, and none has been evaluated for single-speaker, low-resource synthesis, where less data is available to stabilize the additional adversarial and flow-based objectives stacked on top of one another. Section 10 therefore reports a baseline-versus-full-model comparison. Because the intermediate combinations have not yet been trained, the present evidence evaluates the combined system but does not attribute its effect to any individual extension.

The rest of the paper is organized as follows. Section 2 traces each component to its source. Sections 3–5 describe the dataset, exploratory analysis, and preprocessing pipeline. Section 6 presents the experimental setup, Section 7 details the model architecture and planned ablation matrix, Section 8 describes training, and Section 9 defines the evaluation protocol. Section 10 reports the baseline-versus-full-model comparison and its limitations.

---

## 2. Related Work

### 2.1. Architecture Limitations of VITS/Piper

VITS and its lightweight implementation, Piper, are widely recognized as efficient non-autoregressive text-to-speech (TTS) architectures that achieve a favorable balance between synthesis quality and inference speed, making them particularly suitable for deployment on edge devices. Piper largely inherits the original VITS architecture, including its VAE-flow framework, GAN-based decoder, monotonic alignment mechanism, and duration modeling. Although this design has demonstrated strong performance, several architectural components of the original VITS have since been revisited and improved by subsequent studies.

The original VITS adopts the HiFi-GAN V1 generator with LeakyReLU activations together with the Multi-Period Discriminator (MPD) without modification (Kim et al., 2021). Later studies on neural vocoders have shown that both design choices exhibit inherent limitations. Ziyin et al. (2020) demonstrated that ReLU-family activations cannot effectively extrapolate periodic signals, motivating the introduction of the Snake activation function for periodic signal modeling. Meanwhile, Jang et al. (2021) observed that discriminators operating solely in the time domain are less sensitive to high-frequency over-smoothing, leading to the proposed Multi-Resolution Spectrogram Discriminator (MRD), which complements rather than replaces MPD. Building upon these advances, BigVGAN (Lee et al., 2023) combines Snake with the MPD–MRD discriminator pair and achieves state-of-the-art neural vocoding performance.

Beyond the decoder, several components of the original VITS architecture have also been identified as potential bottlenecks. Specifically, the posterior normalizing flow introduces significant computational overhead and training instability without proportional quality gains, the stochastic duration predictor is optimized solely using a likelihood objective without adversarial supervision, monotonic alignment search relies on a single deterministic alignment path, and the text encoder does not incorporate speaker conditioning in multi-speaker settings (Kim et al., 2021). VITS2 addresses these limitations by augmenting the normalizing flow's coupling layers with transformer-based conditioners, redesigning the duration predictor and text encoder, and demonstrating consistent improvements in both speech naturalness and alignment quality while preserving the overall one-stage VITS framework.

Prosody modeling represents another direction of improvement. Both VITS and Piper condition the decoder entirely on a VAE-flow latent that implicitly encodes pitch together with timbre and speaker characteristics, without explicitly providing F0 information during either training or inference (Kim et al., 2021). This implicit modeling often leads to prosody averaging and limits fine-grained controllability. In contrast, FastPitch (Łańcucki et al., 2021) demonstrated that incorporating phoneme-level pitch as an explicit conditioning signal improves both speech naturalness and pitch accuracy in feed-forward TTS architectures.

These studies collectively indicate that multiple components of the original VITS architecture have been substantially improved by subsequent research. Nevertheless, Piper continues to adopt the original VITS design with only minor modifications. Consequently, this work focuses on three research directions: (1) adapting the BigVGAN generator and discriminator design to the one-stage VITS/Piper framework while maintaining computational efficiency suitable for edge deployment; (2) incorporating the architectural improvements introduced by VITS2; and (3) augmenting the decoder with explicit phoneme-level F0 conditioning. The following subsections review each of these directions in detail.

### 2.2. Adaptation of BigVGAN for One-Stage End-to-End TTS

Piper, inheriting the original VITS architecture, employs HiFi-GAN as its default decoder. While HiFi-GAN is currently the de facto standard for high-fidelity vocoding, its core components still exhibit inherent limitations. Specifically, the LeakyReLU activation function, due to its piecewise linear nature, struggles to effectively extrapolate the smooth, periodic signals that are fundamental to audio waveforms (Ziyin et al., 2020). Furthermore, discriminators operating solely in the time domain, such as the Multi-Period Discriminator (MPD), are less sensitive to high-frequency over-smoothing, as subtle spectral artifacts often remain imperceptible in raw waveform comparisons (Jang et al., 2021).

To address these issues, BigVGAN (Lee et al., 2023) introduces the Snake activation function, which inherently models periodicity, and pairs the MPD with a Multi-Resolution Spectrogram Discriminator (MRD) to explicitly penalize high-frequency artifacts in the frequency domain. This combination achieves state-of-the-art neural vocoding performance.

Despite its superior audio quality, directly adopting the original BigVGAN into Piper is unfeasible for two fundamental reasons. First, BigVGAN is inherently designed for two-stage pipelines, receiving mel-spectrograms — which possess explicit frequency structures — as input. In contrast, Piper is a one-stage end-to-end model where the decoder must process latent representations generated by a Variational Autoencoder (VAE). These latent representations reside in a compressed, transformed latent space, meaning that forcing a vocoder designed for mel-spectrograms to interpret VAE latents poses a significant representation learning challenge. Second, the original BigVGAN architecture is computationally heavy. Deploying it in its unmodified form would drastically increase the parameter count and floating-point operations (FLOPs), violating the strict resource constraints of edge devices.

Therefore, rather than adopting alternative lightweight vocoders that often compromise high-frequency fidelity, our work focuses on integrating the core high-fidelity components of BigVGAN — specifically, the Snake activation function and the MRD — into the one-stage VAE framework of Piper. Concretely, we replace the LeakyReLU activations in the HiFi-GAN generator with SnakeBeta and add the MRD alongside the existing MPD, while retaining VITS's generator architecture and channel dimensions. This keeps parameter count compatible with edge deployment while directly inheriting BigVGAN's periodic modeling and high-frequency sensitivity. We note that BigVGAN's sinc-based anti-aliased upsampling is not adopted in this work and is left for future investigation.

### 2.3. Architectural Simplifications and Stabilization in TTS

While VITS established a strong foundation for end-to-end TTS, several architectural components limit its expressiveness and training stability. Specifically, the posterior normalizing flow relies solely on dilated-convolution conditioners in each coupling layer, restricting the conditioner's receptive field to a local window and preventing it from capturing long-range dependencies across the latent sequence. Additionally, the stochastic duration predictor (SDP) relies solely on a likelihood-based objective, which provides a limited supervision signal for learning natural and diverse phoneme durations. The monotonic alignment search (MAS) mechanism, while generally effective, uses a deterministic Viterbi search that can overfit to specific alignment paths and lacks robustness in complex phonetic contexts.

To address these limitations, VITS2 (Kong et al., 2023) introduces targeted architectural refinements while preserving the core one-stage VAE framework. Most notably, it replaces the conv-only conditioner in the normalizing flow's coupling layers with a transformer-augmented conditioner, introducing a self-attention pass before the WaveNet-style convolution stack, so that global context is established prior to local convolutional refinement. The SDP is enhanced with a duration discriminator that provides adversarial supervision, improving rhythmic naturalness and diversity of predicted durations. Furthermore, VITS2 improves the MAS mechanism by injecting Gaussian noise into the log-likelihood scores used by the Viterbi alignment search, creating stochastically perturbed alignments that enhance training stability and robustness to diverse speaking styles. Finally, VITS2 refines the text encoder by incorporating explicit speaker conditioning, enabling better modeling of speaker-specific characteristics. Importantly, VITS2 retains the original MPD from VITS without modification, focusing its discriminator improvements solely on duration modeling, whereas we complement the MPD with an additional MRD as described in Section 2.2.

In this work, we incorporate all four architectural refinements from VITS2: the transformer-augmented coupling layers in the normalizing flow, the adversarially supervised duration predictor via the duration discriminator, the Gaussian-noise-perturbed MAS, and the speaker-conditioned text encoder (not instantiated in our single-speaker setup: `use_speaker_cond_enc = False` → `gin_channels = 0`, so the projection layer is never created). These additions complement our BigVGAN-inspired decoder design (MRD + MPD) described in Section 2.2, where the MRD enhances high-frequency fidelity and the duration discriminator provides adversarial duration supervision. Together, these four refinements address the expressiveness and stability bottlenecks of VITS while remaining fully compatible with the BigVGAN-inspired decoder design.

### 2.4. Explicit Prosody and Pitch Conditioning in Non-Autoregressive TTS

A critical limitation of the original VITS and Piper architectures lies in their implicit handling of prosody, particularly fundamental frequency (F0). In these models, F0 information is entirely encoded within the VAE latent representation, which is jointly optimized to capture timbre, speaker characteristics, and acoustic details alongside pitch. This implicit modeling approach often leads to prosody averaging, where the model converges to a mean pitch contour during training, resulting in monotonous speech with limited rhythmic variation. Furthermore, the absence of explicit F0 conditioning severely restricts fine-grained controllability, making it difficult to manipulate pitch patterns for expressive or application-specific synthesis without retraining the entire model.

To address this limitation, recent studies have explored explicit pitch conditioning in non-autoregressive TTS architectures. FastPitch (Łańcucki et al., 2021) demonstrated that incorporating phoneme-level pitch contours as an explicit conditioning signal significantly improves both speech naturalness and pitch accuracy in feed-forward models. By providing F0 information directly to the decoder, FastPitch enables more precise control over prosodic features and mitigates the prosody averaging problem. Similarly, other works have shown that injecting pitch information at various stages of the decoding process enhances the model's ability to reproduce diverse speaking styles and emotional tones.

In this work, we augment the Piper decoder with explicit F0 conditioning to overcome the limitations of implicit prosody modeling. We introduce a lightweight F0 predictor that generates phoneme-level pitch contours from the text encoder's hidden states, which are then upsampled to frame-level and injected into the decoder as an additional conditioning signal. This design allows the decoder to leverage both the rich acoustic information from the latent space and explicit pitch guidance, with minimal additional computational cost at inference. Importantly, this augmentation is compatible with our BigVGAN-inspired architecture (MRD + MPD) and VITS2 refinements described in Sections 2.2 and 2.3, as the F0 conditioning is integrated at the decoder input stage without modifying the flow or duration modules.

---

## 3. Data Description

### 3.1. Sources

The LJSpeech dataset is a publicly available English speech corpus released by Keith Ito [1]. The textual content is derived from seven public-domain non-fiction books obtained from Project Gutenberg, published between 1884 and 1964. All audio recordings were produced by a single volunteer narrator through the LibriVox audiobook project, ensuring speaker consistency across the entire corpus.

### 3.2. Size and Format

The dataset comprises two primary components:

- **Audio data.** A total of 13,100 WAV files are stored in the `wavs/` directory, corresponding to approximately 24 hours of speech. Each file is encoded in 16-bit Pulse Code Modulation (PCM), sampled at 22,050 Hz, and recorded in mono-channel format.
- **Metadata file.** A single `metadata.csv` file contains 13,100 entries, where each row maps an audio file to its corresponding text transcription. The file is encoded in UTF-8 format and uses the pipe character (`|`) as the field delimiter.

### 3.3. Features

Each entry in `metadata.csv` consists of three fields, as summarized in Table 1.

**Table 1.** Fields in `metadata.csv`.

| Feature | Data type | Description |
|---|---|---|
| `file_id` | String | Unique identifier of each audio clip (e.g., `LJ001-0001`) |
| `original_text` | String | Raw transcription extracted from the source books, including numbers, abbreviations, and special symbols |
| `normalized_text` | String | Text after normalization, where numbers, abbreviations, and symbols are expanded into word-level forms |

**References**

[1] K. Ito, "The LJ Speech Dataset," 2017. [Online]. Available: https://keithito.com/LJ-Speech-Dataset/

---

## 4. Exploratory Data Analysis

### 4.1. Audio Duration

All 13,100 raw clips are available (no missing WAV files were observed; total raw duration 23.92 h). Durations range from 1.11 to 10.10 s, with a median of 6.76 s and a broad, roughly unimodal distribution peaking around 7–8 s.

### 4.2. Speaking Rate

Computed on 13,100 clips (characters per second), the speaking rate follows an approximately Gaussian distribution with a median of 15.25 chars/s and a range of approximately 5–22.5 chars/s. A standard Tukey fence at 1.5 × IQR identifies 203 utterances (1.55%) as outliers at the distribution tails.

### 4.3. Silence Ratio

Measured on 13,100 clips. Leading silence is negligible across the corpus (median 0.0%). Trailing silence shows higher variability (median 1.3%, maximum around 13%), resulting in a long-tailed distribution. Total silence per clip reaches approximately 20% for a small subset of utterances.

### 4.4. Phoneme Sequence Length

Measured on all 13,100 post-parsing utterances. Lengths follow a broad bell-shaped distribution (mean 214.4, median 219.0 phoneme IDs). All utterances fall within the training cap of 400 IDs (min 25, max 399); no entries exceed the cap.

### 4.5. Corpus-Level F0

Measured on all 13,100 utterances (zero F0 cache failures). Mean F0 per utterance is tightly concentrated around 206.4 Hz (corpus mean 206.7 Hz, range 144–393 Hz), reflecting the corpus's single-speaker composition. Pitch range per utterance (max − min F0) follows a right-skewed distribution (median 258.1 Hz, extending beyond 700 Hz for the most expressive clips).

---

## 5. Data Cleaning and Preprocessing

### 5.1. Missing and Empty File Exclusion

Before preprocessing, each metadata entry is verified against the filesystem to ensure that its corresponding audio file exists and is non-empty. Entries associated with missing or zero-byte WAV files are excluded from further processing.

For the LJSpeech corpus, all 13,100 audio files are present and valid; therefore, no samples are removed at this stage.

### 5.2. Transcript Parsing and G2P Normalization

The metadata file is parsed using Python's `csv.reader` with the vertical bar (`|`) as the field delimiter. Initial parsing with the default `QUOTE_MINIMAL` mode resulted in incorrect transcript parsing because the LJSpeech metadata contains 1,052 unescaped double-quote characters. These quotation marks were interpreted as field boundary markers, causing the parser to span multiple rows into a single quoted field. Consequently, only 12,911 rows were parsed correctly, while 16 corrupted entries contained transcripts of up to 6,293 characters that no longer corresponded to their associated audio files.

To address this issue, the parser was configured to use `QUOTE_NONE`, which disables quote-character interpretation and preserves the original metadata structure. This modification recovered all 13,100 utterances without transcript corruption. All subsequent preprocessing stages operate on the complete dataset.

Each transcript is then converted into a phoneme sequence using an espeak-ng-based grapheme-to-phoneme (G2P) system [1]. The resulting phoneme IDs are stored in the intermediate dataset representation.

### 5.3. Silence Removal and Resampling via VAD

Each waveform is processed using a voice activity detection (VAD)-based trimming pipeline. Audio is first loaded at 16 kHz for VAD inference using Silero VAD [2] and segmented into 30 ms frames. Speech regions are identified using a probability threshold of 0.2, while several context frames are retained before and after detected speech boundaries to avoid abrupt truncation.

The detected speech segment is extracted from the original waveform, resampled to 22,050 Hz, converted to a floating-point tensor, and normalized to the range [−1, 1] before being stored for subsequent processing.

### 5.4. Spectrogram Computation

A linear-scale short-time Fourier transform (STFT) is computed from each normalized waveform using an FFT size of 1,024, a window length of 1,024 samples, and a hop size of 256 samples, following the preprocessing configuration of Kim et al. (2021) [4]. The resulting linear spectrogram is cached as a tensor file.

During model training, an 80-channel mel spectrogram is generated on-the-fly by applying a mel filterbank followed by logarithmic compression. The mel spectrogram serves as the acoustic reconstruction target during model optimization.

### 5.5. Fundamental Frequency (F0) Extraction

Fundamental frequency (F0) contours are extracted offline using the WORLD vocoder's DIO algorithm followed by StoneMask refinement [3]. The frame shift is configured to match the spectrogram hop size (approximately 11.61 ms).

Because `pyworld` consistently returns one extra frame at the configured hop length, each contour is aligned through truncation or edge-padding to match the spectrogram frame count. Unvoiced frames are assigned an F0 value of zero, while interpolation is performed in the log-F0 domain to produce a continuous pitch contour suitable for model training. The aligned F0 sequence is then cached as a tensor.

### 5.6. Cache Integrity Verification

After preprocessing, all cached tensor files are validated by attempting deserialization using `torch.load()`. Any unreadable or corrupted cache files are removed to ensure the integrity and robustness of the subsequent training pipeline.

### 5.7. Phoneme Length Constraint

During training, a maximum phoneme sequence length of 400 (`max_phoneme_ids = 400`) is enforced to satisfy the model's batching constraints. Utterances exceeding this limit are excluded before batch construction.

Following the transcript parsing correction described in Section 5.2, all 13,100 utterances satisfy this constraint, with phoneme sequence lengths ranging from 25 to 399 tokens. Consequently, no utterances are removed at this stage.

---

## 6. Experimental Setup

### 6.1. Model Selection

We adopt VITS [1] as the baseline architecture, an end-to-end TTS system that jointly trains acoustic modelling and waveform synthesis within a single conditional variational autoencoder (CVAE). The text encoder produces a prior distribution over the latent space; monotonic alignment search (MAS) estimates soft phoneme-to-frame alignment without an external aligner; a normalizing flow refines the prior; a posterior encoder processes the linear spectrogram to produce a training-time approximate posterior; and a HiFi-GAN-style decoder [2] upsamples the sampled latent to a raw waveform. The adversarial objective is provided by a multi-period discriminator (MPD). All experiments are implemented on the Piper training framework [3] as the implementation foundation.

Section 7.1 introduces the architectural extensions evaluated jointly in the full model. A factorial design for isolating their individual contributions is retained as planned future work (Section 7.6).

### 6.2. Data Splitting

We train on LJSpeech [8] (13,100 utterances, 23.92 h, 22,050 Hz mono). The corpus is partitioned into three non-overlapping subsets using a reproducible random permutation. The baseline and complete model use identical split membership:

| Split | Count | Purpose |
|---|---|---|
| Train | 12,500 | Parameter optimisation |
| Validation | 100 | `val_loss_mel` monitoring; best-checkpoint selection |
| Test | 500 | Post-training WER, UTMOS, and RTF evaluation (held-out) |

This matches the split sizes reported in [1], although utterance membership differs because we use an independent random permutation instead of the original VITS file-ID list. Therefore, our WER and UTMOS results are not directly comparable to those reported in [1] on a per-utterance basis. Checkpoints are selected using `val_loss_mel` (the non-adversarial L1 mel reconstruction loss) instead of the composite `val_loss`, since its adversarial terms do not correlate monotonically with perceptual quality.

### 6.3. Feature Engineering

**Phoneme representation.** Text is converted to phoneme ID sequences via espeak-ng [9] (en-us). A hard cap of `max_phoneme_ids = 400` is enforced at training time; all 13,100 utterances fall within this limit (min 25, max 399 IDs).

**Silence removal.** Silero VAD [10] trims leading and trailing silence at 16 kHz; the detected speech segment is resampled to 22,050 Hz for training.

**Spectral features.** A linear-scale STFT is computed (FFT 1,024, window 1,024, hop 256 samples at 22,050 Hz) following [1], yielding a 513-bin magnitude spectrogram as the posterior encoder input. An 80-band mel spectrogram is derived at training time via triangular filterbank and log compression; it is used only as the reconstruction target ($c_\text{mel} = 45$) and is not fed to the posterior encoder.

**Fundamental frequency.** Per-frame F0 contours are extracted using WORLD DIO + StoneMask [11] (frame shift ≈ 11.61 ms). Unvoiced frames carry F0 = 0 by convention (DIO output); these are linearly interpolated in the log domain before caching. The contour is then truncated or edge-padded to match the spectrogram frame count (`pyworld` consistently returns one extra frame at our hop length). F0 features are consumed only by configurations with `use_f0 = true`.

**Waveform segments.** Random 8,192-sample segments (approximately 0.37 s at 22,050 Hz, corresponding to 32 latent frames) are extracted per utterance and passed to the generator and discriminators during windowed generator training (Kim et al., 2021). Segment size is fixed for Config A and Config H.

---

## 7. Model Architecture

### 7.1. Generator

**Text encoder.** A relative positional encoding Transformer (hidden 192, filter 768, 6 layers, 2 attention heads, kernel 3, dropout 0.1) that produces prior distribution parameters $(\mu_p, \log\sigma_p)$ and hidden state $\mathbf{x}$. Following VITS2, the text encoder supports a speaker-embedding input slot; in our single-speaker setup, `use_speaker_cond_enc = False` so `gin_channels = 0` is passed to the text encoder and the projection layer is not instantiated.

**Posterior encoder.** A dilated WaveNet stack (kernel 5, dilation 1, 16 residual blocks, hidden 192) that maps the linear spectrogram to an approximate posterior $(\mu_q, \log\sigma_q)$ and samples the latent $\mathbf{z}$. The posterior encoder is used only during training.

**Normalizing flow.** A ResidualCouplingBlock comprising 4 volume-preserving affine coupling layers (Jacobian determinant 1, mean-only transform), each conditioned by a WaveNet stack (kernel 5, dilation 1, 4 residual layers). The volume-preserving constraint follows Kim et al. (2021). When `use_vits2 = true`, each coupling layer's conditioner gains an additional single-layer self-attention pass before the WaveNet convolution stack, so that global context is established before local refinement.

**HiFi-GAN decoder.** Upsamples the latent via three transposed convolutions (rates 8 × 8 × 4, initial channels 256) with multi-receptive-field (MRF) ResBlock2 branches (kernel sizes 3, 5, 7; dilation groups (1,2), (2,6), (3,12)). When `use_bigvgan = true`, each transposed convolution is preceded by a SnakeBeta activation [5] and the final non-linearity switches from LeakyReLU (slope 0.1) to SnakeBeta.

**F0 conditioning.** When `use_f0 = true`, a zero-initialized `f0_cond` Conv1d projects the Hz-scale F0 frame slice into the decoder's input channel space as an additive conditioning signal, following the approach of Łańcucki et al. (2021) adapted from a feed-forward to a flow-based decoder.

### 7.2. Duration Predictor

The stochastic duration predictor (SDP) is a 4-flow normalizing flow — using rational-quadratic neural spline flows (Durkan et al., 2019) in its coupling layers rather than affine coupling — conditioned by a DDSConv network (dilated depth-wise separable convolutions, kernel 3, 3 layers, GELU activation, LayerNorm, filter 192). The SDP models the posterior distribution over phoneme durations during training and samples durations stochastically at inference.

### 7.3. F0 Predictor

When `use_f0 = true`, a two-layer Conv1d network (hidden 256, kernel 3, dropout 0.5) regresses per-phoneme log-F0 from the text encoder's hidden state $\mathbf{x}$. Ground-truth per-phoneme F0 is obtained by projecting the frame-level F0 contour through the MAS attention matrix.

### 7.4. Discriminators

The baseline MPD contains one scale sub-discriminator (DiscriminatorS, 6 Conv1d layers) and five period sub-discriminators (DiscriminatorP, periods [2, 3, 5, 7, 11], 5 Conv2d layers each), totalling 46.7 M parameters. As noted in Kim et al. (2021), this formulation is equivalent to the Multi-Period Discriminator with periods [1, 2, 3, 5, 7, 11], where DiscriminatorS serves as the period-1 sub-discriminator operating on raw waveforms.

When `use_bigvgan = true`, a MultiResolutionDiscriminator [4, 6] adds three DiscriminatorR instances operating at STFT resolutions (512/50/240), (1024/120/600), and (2048/240/1200) as 2D convolutional networks on magnitude spectrograms (280 K parameters).

When `use_vits2 = true`, a DurationDiscriminator [7] (5 Conv1d layers, hidden 192, kernel 3) scores each phoneme's predicted log-duration as real or fake, conditioned on the text encoder hidden state; it adds zero inference cost as it is discarded after training.

### 7.5. MAS Noise Annealing

When `use_vits2 = true`, the MAS alignment scores are additionally perturbed with adaptive Gaussian noise during training to improve alignment robustness [7]. The noise standard deviation is $\tau \cdot \sigma(\mathbf{S})$, where $\sigma(\mathbf{S})$ is the standard deviation of the current batch's alignment score matrix and

$$\tau = \max\!\left(0,\ 0.01 - t \times 2 \times 10^{-6}\right)$$

decays linearly with global training step $t$, reaching zero at step 5,000 (approximately epoch 13). This schedule transitions MAS from soft alignment exploration in early training to hard Viterbi alignment later, following [7].

### 7.6. Planned Ablation Configurations

The three flags define a possible full $2^3$ factorial with eight configurations. In the present study, only Config A (baseline) and Config H (complete model) have been trained and evaluated. Configurations B–G are listed to document the planned ablation design; they are not treated as completed experiments or used to attribute gains to individual components.

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

For a future controlled ablation, all configurations should share identical model dimensions, optimiser settings, data splits, random seeds, and training budgets. The reported comparison in this paper is limited to Config A versus Config H.

---

## 8. Training Procedure

### 8.1. Optimiser

Generator and discriminators are each optimised with AdamW ($\text{lr} = 2 \times 10^{-4}$, $\beta_1 = 0.8$, $\beta_2 = 0.99$, $\varepsilon = 10^{-9}$), following Kim et al. (2021). Gradient norms are clipped to 1.0 for both generator and discriminators. Both schedulers follow an exponential decay ($\gamma = 0.999875 = 0.999^{1/8}$) stepped once per epoch, matching the schedule in Kim et al. (2021); at epoch 1,100 the learning rate is approximately 87% of its initial value.

### 8.2. GAN Training Procedure

Manual optimization is used: per batch, the generator backward pass runs first, followed by the discriminator backward pass (generator-first ordering is an explicit implementation choice, deviating from the discriminator-first convention in [1]). The generator objective is:

$$\mathcal{L}_\text{gen} = \mathcal{L}_\text{adv,MPD} + \mathcal{L}_\text{adv,MRD} + \mathcal{L}_\text{fm} + c_\text{mel}\mathcal{L}_\text{mel} + \mathcal{L}_\text{dur} + c_\text{kl}\mathcal{L}_\text{kl} + \mathcal{L}_\text{dur,gen} + \mathcal{L}_\text{f0}$$

where $c_\text{mel} = 45$, $c_\text{kl} = 1.0$; $\mathcal{L}_\text{adv,MRD}$ is zero when `use_bigvgan = false`; $\mathcal{L}_\text{dur,gen}$ is zero when `use_vits2 = false`; $\mathcal{L}_\text{f0}$ (per-phoneme log-F0 MSE) is zero when `use_f0 = false`.

Adversarial losses follow the least-squares GAN formulation [12]: the discriminator minimises $\mathbb{E}[(1 - D(y))^2] + \mathbb{E}[D(\hat{y})^2]$ and the generator minimises $\mathbb{E}[(1 - D(\hat{y}))^2]$, applied independently to MPD and MRD. $\mathcal{L}_\text{mel} = \|\text{Mel}(y) - \text{Mel}(\hat{y})\|_1$ is L1 on 80-band mel spectrograms; $\mathcal{L}_\text{fm}$ is mean absolute difference on intermediate discriminator feature maps; $\mathcal{L}_\text{kl}$ is the KL divergence between posterior and prior; $\mathcal{L}_\text{dur}$ is the SDP negative log-likelihood.

Due to a cuFFT limitation, STFT computations within MRD and mel loss are cast to float32 even under bfloat16 mixed-precision.

### 8.3. Hardware and Schedule

Config A and Config H use 2 × NVIDIA RTX 4070 Ti (12 GB each), DDP (NCCL), bfloat16 mixed precision, batch size 16 per GPU (32 total), and global RNG seed 1234 (independent from the dedicated data-split generator described in Section 6.2). The selected checkpoints are epoch 861 for Config A and epoch 863 for Config H, chosen by validation mel loss. Checkpoints are saved every epoch.

---

## 9. Model Evaluation

### 9.1. Evaluation Metrics

Config A and Config H are evaluated post-training on the same 100 held-out sentences via an automated harness that synthesises audio from raw text (no ground-truth audio is used for WER, UTMOS, or RTF scoring). The present comparison reports three metrics: WER, UTMOS, and RTF.

**Word Error Rate (WER).** Synthesised audio is transcribed by Whisper [13] and compared against the reference transcript using `jiwer`. Both hypothesis and reference are normalised identically before scoring: lowercased, then all characters outside `[a-z0-9' ]` removed, and whitespace collapsed.

**UTMOS.** Synthesised audio is scored using UTMOSv2, a pretrained neural MOS predictor. It serves as an automated perceptual-naturalness proxy on a 1–5 scale where higher values indicate better predicted quality. UTMOS is not a substitute for a human MOS listening study.

**Real-Time Factor (RTF).** Inference wall-clock time divided by audio duration. RTF is reported as a mean over the same 100 sentences and quantifies synthesis cost, where lower values are better.

**Planned F0 analysis.** The evaluation harness supports log-F0 RMSE and Pearson correlation between predicted and ground-truth log-F0. These values were not produced by the completed A–H evaluation and are therefore not reported. They should be included when the F0-enabled planned configurations D, F, G, and H are evaluated under a common protocol.

Statistical significance between Config A and Config H is assessed via a paired Wilcoxon signed-rank test on the 100 paired per-sentence WER and UTMOS score distributions ($p < 0.05$).

### 9.2. Hyperparameter Tuning

Optimiser hyperparameters ($\text{lr}$, $\beta_1$, $\beta_2$, $\varepsilon$, $\gamma$, $\lambda$, $c_\text{mel}$, $c_\text{kl}$) are adopted directly from [1] and held fixed between Config A and Config H. They should likewise remain fixed when the planned B–G experiments are conducted.

No architecture selection claim is made from the planned factorial design because configurations B–G have not been evaluated. The present experiment tests only whether the complete architecture differs from the baseline; it cannot identify which component causes any observed difference.

---

## 10. Baseline versus Full-Model Comparison

### 10.1. Quantitative Results

Both systems were evaluated on the same 100 sentences. Config H obtains a lower mean WER and higher mean UTMOS than Config A, with a modest increase in mean RTF.

| Configuration | Mean WER ↓ | Median WER ↓ | Mean UTMOS ↑ | Median UTMOS ↑ | Mean RTF ↓ |
|---|---:|---:|---:|---:|---:|
| A — VITS/Piper baseline | 0.0818 | 0.0541 | 3.5497 | 3.5533 | 0.0658 |
| H — complete model | 0.0735 | 0.0000 | 3.6193 | 3.6493 | 0.0721 |

The paired Wilcoxon test does not establish a statistically significant WER difference ($W=278.5$, $p=0.1821$). The UTMOS distributions differ significantly ($W=1719.0$, $p=0.0056$), favouring Config H. Accordingly, the results support an improvement in predicted naturalness under UTMOSv2, but they do not establish an intelligibility improvement at the 0.05 significance level. Mean RTF increases from 0.0658 to 0.0721 (approximately 9.6%), although both measurements remain below 1.0 in the recorded evaluation environment.

### 10.2. Discussion and Limitations

The comparison evaluates the architectural package as a whole. Because configurations B–G were not trained, the observed UTMOS improvement cannot be assigned separately to the BigVGAN-style decoder/discriminator changes, VITS2-style flow and alignment changes, or explicit F0 conditioning. Describing the experiment as a completed factorial ablation would therefore overstate the evidence.

The evaluation also has three scope limitations. First, it uses 100 sentences rather than the originally planned 500-sample test set. Second, UTMOS is an automated predictor and does not replace a blinded human MOS study. Third, F0 RMSE and correlation were not emitted by the completed evaluation, so no claim is made about pitch-prediction accuracy. The planned B–G configurations and F0-specific evaluation are deferred to future work using the matrix in Section 7.6.

---

## References

- Durkan, C., Bekasov, A., Murray, I., & Papamakarios, G. (2019). Neural spline flows. *Advances in Neural Information Processing Systems (NeurIPS)*, 32.
- Hansen, D. (n.d.). Piper: A fast, local neural text-to-speech system. GitHub. https://github.com/rhasspy/piper
- Jang, W., Lim, D., Yoon, J., Kim, B., & Kim, J. (2021). UnivNet: A neural vocoder with multi-resolution spectrogram discriminators for high-fidelity waveform generation. *Proc. Interspeech*, 2207–2211.
- Jia, Y., Zhang, Y., Weiss, R., Wang, Q., Shen, J., Ren, F., ... & Wu, Y. (2019). Transfer learning from speaker verification to multispeaker text-to-speech synthesis. *Advances in Neural Information Processing Systems (NeurIPS)*, 32.
- Kim, J., Kong, J., & Son, J. (2020). Glow-TTS: A generative flow for text-to-speech via monotonic alignment search. *Advances in Neural Information Processing Systems (NeurIPS)*, 33, 8067–8077.
- Kim, J., Kong, J., & Son, J. (2021). Conditional variational autoencoder with adversarial learning for end-to-end text-to-speech. *Proceedings of the 38th International Conference on Machine Learning (ICML)*, PMLR 139, 5530–5540.
- Kong, J., Kim, J., & Bae, J. (2020). HiFi-GAN: Generative adversarial networks for efficient and high fidelity speech synthesis. *Advances in Neural Information Processing Systems (NeurIPS)*, 33, 17022–17033.
- Kong, J., Park, C., Kim, B., Kim, J., Kong, D., & Kim, S. (2023). VITS2: Improving quality and efficiency of single-stage text-to-speech with adversarial learning and architecture design. *Proc. Interspeech*, 4374–4378.
- Łańcucki, A. (2021). FastPitch: Parallel text-to-speech with pitch prediction. *Proc. ICASSP*, 6588–6592.
- Lee, J., Han, S., Chang, J. S., & Han, H. (2023). BigVGAN: A universal neural vocoder with large-scale training. *Proceedings of the 11th International Conference on Learning Representations (ICLR)*.
- Mao, X., Li, Q., Xie, H., Lau, R. Y., Wang, Z., & Paul Smolley, S. (2017). Least squares generative adversarial networks. *Proceedings of the IEEE International Conference on Computer Vision (ICCV)*, 2794–2802.
- Morise, M., Yokomori, F., & Ozawa, K. (2016). WORLD: A vocoder-based high-quality speech synthesis system for real-time applications. *IEICE Transactions on Information and Systems*, E99-D(7), 1877–1884.
- Radford, A., Kim, J. W., Xu, T., Brockman, G., McLeavey, C., & Sutskever, I. (2023). Robust speech recognition via large-scale weak supervision. *Proceedings of the 40th International Conference on Machine Learning (ICML)*, PMLR 202, 28492–28518.
- Silero Team. (2021). Silero VAD: Pre-trained enterprise-grade voice activity detector. GitHub. https://github.com/snakers4/silero-vad
- Ziyin, L., Hartwig, T., & Ueda, M. (2020). Neural networks fail to learn periodic functions and how to fix it. *Advances in Neural Information Processing Systems (NeurIPS)*, 33, 1583–1594.
