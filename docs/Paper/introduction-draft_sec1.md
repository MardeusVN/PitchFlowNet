# 1. Introduction

Text-to-speech (TTS) synthesis has advanced rapidly in recent years, in part because it
is a building block for speech-to-speech (S2S) systems that aim to let machines
communicate with the prosodic and emotional nuance of human speakers rather than reading
text aloud in a flat, affectless voice. Progress toward that goal has, however, split
TTS systems along an axis that receives less attention than raw audio quality:
deployment cost. Autoregressive, language-model-style systems such as Qwen3-TTS (Hu et
al., 2026) generate expressive, controllable speech by predicting discrete acoustic
tokens one at a time with a 0.6B- or 1.7B-parameter transformer conditioned on a
dedicated 12 Hz speech tokenizer. This design gives them strong prosody and voice
control, but autoregressive decoding at that parameter count targets GPU-class
inference; it is not built for the CPU-only, low-power hardware — single-board
computers, mobile and embedded devices — that edge deployment requires. VITS (Kim, Kong,
& Son, 2021) and its lightweight derivative Piper (Hansen, n.d.) occupy the opposite corner: a
non-autoregressive conditional-VAE-plus-flow-plus-GAN-decoder architecture with on the
order of tens of millions of parameters, small and fast enough to synthesize speech in
real time on CPU alone, including on a Raspberry Pi. The cost of this efficiency is
audio quality and expressiveness that trail current autoregressive systems.

This paper asks whether that quality gap can be narrowed without abandoning the
non-autoregressive, edge-deployable profile that makes VITS and Piper practical in the
first place, rather than closing it by scaling up toward an autoregressive, GPU-class
model. We present Banhmi-TTS, which keeps the VITS/Piper backbone — and therefore its
real-time, CPU-only inference profile — unchanged, while replacing four of its
components, drawn from three prior systems — BigVGAN, VITS2, and FastPitch — that have
so far only been validated in multi-speaker, high-resource, or non-flow-based settings.
At inference time Banhmi-TTS has 17.9M parameters (25.2M including the
training-only posterior encoder used for the variational objective) — two orders of
magnitude smaller than Qwen3-TTS's 0.6B–1.7B. The four replaced components are:

- the generator's activation and discriminator design, replaced with a BigVGAN-style
  Snake activation paired with UnivNet's multi-resolution discriminator, reusing
  BigVGAN's published recipe rather than introducing a new combination (Section 2.2);
- the normalizing-flow prior, augmented with VITS2-style transformer coupling layers
  (Section 2.3);
- the duration model, replaced with VITS2's adversarial duration discriminator
  (Section 2.3);
- the conditioning signal supplied to the decoder, replaced with FastPitch-style
  per-phoneme pitch conditioning, adapted from a feed-forward decoder to the flow-based
  VITS decoder (Section 2.4).

We modify none of these four components with a new technique. Our contribution is
empirical: to our knowledge, no published system combines all three source lineages
inside a single end-to-end VITS pipeline, and none has been evaluated for
single-speaker, low-resource synthesis, where less data is available to stabilize the
additional adversarial and flow-based objectives stacked on top of one another. Section
5 reports the ablations needed to show whether this combination composes cleanly in
that regime, rather than assuming composability as a foregone conclusion.

The rest of the paper is organized as follows. Section 2 traces each of these four
components to its source and states precisely what combination has not, to our
knowledge, been published before. Section 3 describes the resulting Banhmi-TTS
architecture. Section 4 details the dataset, preprocessing pipeline, and experimental
setup, and Section 5 reports results and ablations. Section 6 concludes and discusses
limitations.
