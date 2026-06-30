# Related Work §2.2 — Vocoder activation + discriminator (Snake + MRD)

Status: gap statement finalized and verified (theory + code). Ablation design decided, not yet run.
Scope: this file covers ONLY the Snake/MRD (generator activation + discriminator) line. The other two
architecture lines (transformer-flow + duration discriminator; F0 conditioning) are separate, not yet drafted.

## Final gap statement — English (Related Work, condensed)

> VITS adopts the HiFi-GAN V1 generator (LeakyReLU) and multi-period discriminator unchanged (Kim et al.,
> 2021). Subsequent vocoder work has shown both choices are suboptimal: Ziyin et al. (2020) prove that
> ReLU-family activations cannot extrapolate periodic signals, motivating the periodic Snake activation;
> Jang et al. (2021) show that purely time-domain discriminators miss high-frequency over-smoothing,
> motivating the multi-resolution spectrogram discriminator (MRD), used alongside — not in place of — the
> multi-period discriminator (MPD). BigVGAN (Lee et al., 2023) adopts Snake together with this MPD+MRD
> discriminator pair and achieves state-of-the-art vocoding — but only for a two-stage, mel-conditioned generator
> trained with a single adversarial objective. Whether these gains transfer to a single-stage generator
> conditioned on a VAE-flow latent and jointly optimized against KL-divergence and duration losses, as in
> VITS/Piper, remains untested.

## Final gap statement — Vietnamese (bản dịch)

> VITS sử dụng nguyên generator HiFi-GAN V1 (LeakyReLU) và multi-period discriminator, không thay đổi gì
> (Kim et al., 2021). Các nghiên cứu vocoder sau đó đã chỉ ra cả hai lựa chọn này đều chưa tối ưu: Ziyin et
> al. (2020) chứng minh rằng các activation thuộc họ ReLU không thể extrapolate tín hiệu tuần hoàn, từ đó
> đề xuất Snake activation mang tính tuần hoàn; Jang et al. (2021) chỉ ra rằng discriminator chỉ xử lý
> miền thời gian bỏ lọt lỗi over-smoothing ở dải tần cao, từ đó đề xuất multi-resolution spectrogram
> discriminator (MRD), dùng **thêm cạnh** — không phải thay thế — multi-period discriminator (MPD). BigVGAN
> (Lee et al., 2023) áp dụng Snake cùng với cặp discriminator MPD+MRD này và đạt state-of-the-art về chất
> lượng vocoding — nhưng chỉ trong bối cảnh một generator hai giai đoạn (two-stage), nhận mel-spectrogram làm
> điều kiện, huấn luyện với một mục tiêu adversarial duy nhất. Liệu những cải thiện này có giữ nguyên khi
> áp dụng cho một generator một giai đoạn (single-stage), nhận điều kiện từ một latent của VAE-flow, và
> được tối ưu đồng thời với KL-divergence và duration loss, như trong VITS/Piper — vẫn chưa được kiểm chứng.

## Expanded version (for Architecture/design-rationale section, not Related Work)

The condensed paragraph above is for §2 (Related Work). The full theoretical/empirical justification below
belongs in the Architecture section when justifying why Banhmi-TTS's generator/discriminator design
changed — not in Related Work, to keep that section from ballooning across all 3 architecture lines.

- **Snake vs LeakyReLU (Ziyin, Hartwig & Ueda, 2020, NeurIPS — verified via ar5iv arXiv:2006.08195 +
  NeurIPS proceedings PDF):**
  - Theorem 1 (ReLU): a feedforward ReLU network asymptotically converges to an **affine** transformation
    as input scales (z→∞): lim_{z→∞}‖f_ReLU(zu) − zW_u·u − b_u‖₂ = 0. An affine function cannot be
    periodic ⇒ structurally cannot extrapolate periodicity.
  - Explicit extension to LeakyReLU (exact quote): "one can prove the same theorem for any continuous
    activation function that asymptotically converges to a tanh or ReLU; for example, this would include
    Swish and **Leaky-ReLU** (and almost all the other ReLU-based variants), which converge to ReLU."
  - Theorem 3 (Universal Extrapolation Theorem): a Snake network can converge uniformly to any
    piecewise-C¹ periodic function as width N→∞.
  - Snake definition (corrected): Snake_a(x) = x + (1/a)·sin²(ax). The "+x" term preserves monotonicity
    (avoids the infinite-local-minima problem of pure sin(x), which is non-monotonic and 2π-periodic in
    its minima).
  - Caveat: Ziyin et al. test on CIFAR-10, temperature/financial time series, and synthetic periodic
    regression — **no audio/waveform task**. BigVGAN is the first to apply this to vocoding.

- **MRD vs time-domain-only discriminator (Jang et al., 2021, Interspeech, UnivNet — verified via ar5iv
  arXiv:2106.07889):**
  - UnivNet's discriminator = MRSD (multi-resolution spectrogram discriminator) + MPWD (multi-period
    waveform discriminator). UnivNet does **not** use MSD anywhere in its main architecture.
  - Ablation row "without MRSD" → remaining config is **MPWD-only**, MOS drops 3.92→3.38 (largest drop in
    their table). Exact quote: "if MRSD is removed, an over-smoothing problem occurs... most notable in
    the high-frequency band of the generated waveform" → described as an "audible metallic artifact"
    elsewhere in the paper.
  - BigVGAN (Lee et al., 2023, ICLR — verified via ar5iv arXiv:2206.04658) independently confirms by
    **replacing MSD with MRD** (keeps MPD): "replacing MSD with MRD improves audio quality with reduced
    pitch and periodicity artifacts" (§3.1).
  - Caveat: BigVGAN's published ablation table (Table 4, MUSDB18-HQ SMOS) only isolates Snake +
    anti-aliasing filter — **no joint ablation row isolating MRD's contribution alongside Snake**. So
    "Snake+MRD are complementary" is not proven by a BigVGAN ablation number; only each component's
    individual benefit is separately verified (Ziyin et al. for Snake, Jang et al. for MRD). BigVGAN's
    full system (using both) achieves SOTA, but that's not the same as a factorial ablation.

- **VITS's own baseline, verified directly against the VITS paper text (ar5iv arXiv:2106.06103) and our
  code:**
  - §2.5.3 (exact quote): "The decoder is essentially the HiFi-GAN V1 generator."
  - §2.5.4 + Appendix B.2 (exact quote): "We follow the discriminator architecture of the multi-period
    discriminator proposed in HiFi-GAN... we leave only the first sub-discriminator of the multi-scale
    discriminator that operates on raw waveforms and discard two sub-discriminators operating on
    average-pooled waveforms."
  - Code match: `src/python/piper_train/vits/models.py:660` —
    `discs = [DiscriminatorS(...)] + [DiscriminatorP(i,...) for i in periods]` — exactly 1 raw-waveform
    MSD-style branch + 5 period branches, matching the paper's Appendix B.2 description exactly.
  - LeakyReLU confirmed in code (not explicit in VITS paper text): `models.py:511,524` —
    `F.leaky_relu(x, self.LRELU_SLOPE)` in the `use_snake=False` branch of `Generator`.
  - Snake/MRD implementation: `modules.py` `Snake1d` class (docstring cites BigVGAN); `models.py`
    `DiscriminatorR`/`MultiResolutionDiscriminator` (757–851, docstring cites UnivNet-style design);
    flags `use_snake`/`use_mrd` wired in `lightning.py` (model_d_mrd conditionally instantiated).

- **Caveats found during literature stress-test (not fully resolved, flagged for honesty):**
  - A non-peer-reviewed GitHub repo (`shigabeev/vits2_pytorch_bigvgan`) attempts combining VITS2 +
    BigVGAN decoder — weakens (does not invalidate) a "no prior work" framing. Mitigated by phrasing the
    gap as "to the best of our knowledge, no peer-reviewed work..." rather than an absolute claim.
    Not yet investigated whether this repo does true single-stage integration or just swaps in BigVGAN as
    an external two-stage vocoder.
  - "Period VITS" (Shirahata et al., 2023, IEEE ICASSP, arXiv:2210.15964) does explicit pitch modeling
    inside an end-to-end VAE+adversarial framework — relevant to the **F0 conditioning** gap (separate
    subsection), not to this Snake/MRD gap. Must be addressed when drafting the F0 subsection.

## Ablation design (for §5 Results) — this is where the gap actually gets answered

The gap statement is a question ("does the benefit transfer?"), not a claim — it can only be answered by
training, not by more literature/theory. Minimum rigorous design: 2×2 factorial isolating Snake and MRD,
holding the other three Banhmi-TTS flags (`use_transformer_flows`, `use_dur_disc`, `use_f0`) **off** in
all four runs to avoid confounding with the other two architecture lines (which get their own ablation
block later).

| Config | use_snake | use_mrd | Role |
|---|---|---|---|
| A (baseline) | False | False | ≈ original VITS/Piper (LeakyReLU + MPD only) |
| B | True | False | Isolates Snake's contribution |
| C | False | True | Isolates MRD's contribution |
| D (full vocoder) | True | True | Combined — compare against B+C−A to test complementary vs. additive vs. redundant |

This directly fixes the methodological gap we identified in BigVGAN itself (no joint ablation isolating
MRD's contribution alongside Snake) — our own ablation, if run this way, would be stronger than BigVGAN's
published one on this specific point.

Status: design decided, **no training run yet**. Compute estimate not yet done. Eval via existing
WER+UTMOS+Wilcoxon harness (see memory: eval_harness_ready).
