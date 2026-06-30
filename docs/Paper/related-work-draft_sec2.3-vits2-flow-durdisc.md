# Related Work §2.3 — Flow + duration model (VITS2-style transformer flow + duration discriminator)

Status: gap statement drafted and verified (theory + code), same rigor level as §2.2 (Snake+MRD).
Ablation design decided, not yet run.
Scope: this file covers ONLY the flow/duration line (`use_transformer_flows`, `use_dur_disc`). F0
conditioning (FastPitch-style) is a separate, not-yet-drafted subsection.

## VITS's baseline (verified against Kim, Kong & Son, 2021, PMLR v139, and our code)

- Flow: `ResidualCouplingLayer`, conditioner = WaveNet-style dilated convolutions (`modules.py` `WN`
  class) only — no self-attention.
- Duration: `StochasticDurationPredictor` (flow-based), trained purely by the flow's own
  negative-log-likelihood — no adversarial signal on the predicted durations.
- Alignment: monotonic alignment search (MAS) is a deterministic Viterbi-style search — code-verified:
  `models.py:990` calls `monotonic_align.maximum_path(neg_cent, attn_mask.squeeze(1))` with no noise added
  to `neg_cent` beforehand.

## VITS2's four changes (Kong, Park, Kim, Kim, Kong & Kim, 2023, Interspeech — arXiv:2307.16430)

VITS2 bundles **four** changes into one release, only two of which Banhmi-TTS adopts:

1. **Transformer block in the flow's coupling-layer conditioner** (adopted, `use_transformer_flows`).
   Stated mechanism (exact quote): "Although a convolution block captures adjacent patterns effectively,
   it has a disadvantage in capturing long-term dependencies owing to the limitations of its receptive
   field." Self-attention is added to give the flow conditioner a global receptive field over the latent
   sequence.
2. **Adversarial duration discriminator** (adopted, `use_dur_disc`). The paper states the goal ("more
   natural speech with higher efficiency... than the previous work") but gives **no theoretical mechanism**
   for why adversarial training on durations beats pure likelihood training — unlike Snake (Ziyin et al.,
   2020), this is an empirically-motivated change, not a proven one. State this honestly in the paper; do
   not invent a mechanism VITS2 itself doesn't claim.
3. **Gaussian-noise-augmented alignment search** ("alignment noise") — **not adopted**. Our MAS stays
   deterministic (see code citation above).
4. **Speaker-conditioned text encoder** — **not adopted** (not applicable; single-speaker corpus).

## The gap

VITS2's own ablation (Table 1, MOS on LJSpeech — same single-speaker corpus we use) never isolates any of
these four changes starting from the original VITS baseline, and never tests any 2-of-4 subset. It only
removes one change at a time from the **full four-component system**:

| Row | MOS | Δ from full VITS2 (4.47) |
|---|---|---|
| VITS2 (full, all 4 changes) | 4.47 | — |
| w/o adversarial duration discriminator | 4.33 | −0.14 |
| w/o alignment noise | 4.32 | −0.15 |
| w/o transformer block | 4.41 | −0.06 |
| VITS (baseline, 0 changes) | 4.38 | −0.09 |

**Honesty note on why #3/#4 were left out:** this is a scope/implementation fact, not a principled
architectural decision — code-verified (grep across `piper_train/`) confirms alignment-noise MAS was never
implemented in this codebase; there is no `use_noised_mas`-style flag, and `models.py:990`'s
`monotonic_align.maximum_path` call has no noise injection path at all to toggle. Do not retroactively
invent an architectural justification for excluding it. State this plainly in the paper's Limitations
section (Section 6): Banhmi-TTS's flow/duration line covers 2 of VITS2's 4 changes by scope choice, and
alignment noise (the single largest individual contributor in VITS2's own ablation, −0.15 MOS) is left for
future work rather than shown to be unnecessary. Speaker-conditioned encoding (#4) is excluded for a
different, valid reason — not applicable to a single-speaker corpus — and does not need the same caveat.

Banhmi-TTS adopts exactly the two changes (transformer flow + duration discriminator) whose combined
individual-removal cost (0.14 + 0.06 = 0.20) is smaller than what's left out: alignment noise alone costs
0.15 when removed from the full system — comparable to, or larger than, either of the two components we
keep. VITS2 never reports a row removing alignment noise *and* speaker conditioning while keeping the
other two (i.e., never reports our exact subset), so whether transformer flow + duration discriminator
retain their benefit without alignment noise is untested, not merely under-cited.

## Final gap statement — English (Related Work, condensed)

> VITS's normalizing flow uses only dilated-convolution conditioners, and its stochastic duration
> predictor is trained by likelihood alone, with deterministic monotonic alignment search (Kim, Kong &
> Son, 2021). VITS2 (Kong et al., 2023) replaces both: a transformer block is added to the flow conditioner
> because convolution's limited receptive field cannot capture long-range dependencies across the latent
> sequence, and an adversarial discriminator is added on top of the predicted durations. VITS2 bundles
> these two changes with two more — Gaussian-noise-augmented alignment search and a speaker-conditioned
> text encoder — and its own ablation only removes one change at a time from the full four-component
> system, never isolating any subset against the original VITS baseline. Banhmi-TTS adopts only the
> transformer flow and the duration discriminator, while keeping VITS's original deterministic alignment
> search; whether these two changes retain VITS2's reported gains in that exact two-of-four combination,
> never reported in VITS2's own ablation, remains untested.

## Final gap statement — Vietnamese (bản dịch)

> Normalizing flow của VITS chỉ dùng dilated-convolution làm conditioner, và stochastic duration predictor
> của nó chỉ được huấn luyện bằng likelihood, với monotonic alignment search xác định kiểu Viterbi (Kim,
> Kong & Son, 2021). VITS2 (Kong et al., 2023) thay đổi cả hai: thêm một transformer block vào conditioner
> của flow vì convolution có receptive field hạn chế, không nắm được long-range dependency trong latent
> sequence; và thêm một adversarial discriminator lên duration được dự đoán. VITS2 gộp hai thay đổi này với
> hai thay đổi khác — alignment search có thêm Gaussian noise, và text encoder có điều kiện theo speaker —
> và ablation của chính VITS2 chỉ bỏ từng thay đổi một khỏi hệ thống đầy đủ 4 thành phần, chưa từng tách
> riêng bất kỳ tổ hợp con nào so với baseline VITS gốc. Banhmi-TTS chỉ áp dụng transformer flow và duration
> discriminator, vẫn giữ nguyên deterministic alignment search gốc của VITS; liệu hai thay đổi này có giữ
> được mức cải thiện mà VITS2 báo cáo trong đúng tổ hợp hai-trên-bốn này — một tổ hợp chưa từng được báo
> cáo trong ablation của chính VITS2 — vẫn chưa được kiểm chứng.

## Expanded version (for Architecture/design-rationale section, not Related Work)

- **Transformer-in-flow mechanism (Kong et al., 2023):** exact receptive-field quote above; paper's
  Figure 2 shows attention spans "various positions" vs. convolution's narrow local window. This is the
  only one of VITS2's four changes with a stated theoretical rationale.
- **Duration discriminator:** empirically motivated only, no theorem — state this as an honest limit on
  the citable justification, mirroring the same honesty standard applied to Snake/MRD in §2.2.
- **Ablation-table numbers and the partial-bundle gap:** the full table above, plus the explicit
  observation that the one component we drop (alignment noise, −0.15) is not smaller than either component
  we keep (transformer −0.06, duration discriminator −0.14) — so dropping it is not obviously "safe" by
  VITS2's own data.
- **Code locations:** `models.py` `ResidualCouplingLayer` vs. `TransformerCouplingLayer` (flow), `dp`
  (`StochasticDurationPredictor`) vs. `DurationDiscriminator` (duration), `models.py:990`
  (`monotonic_align.maximum_path`, deterministic, confirms no alignment-noise adoption), flags wired in
  `lightning.py`.

## Ablation design (for §5 Results)

Same 2×2-factorial pattern as §2.2, holding `use_snake`/`use_mrd`/`use_f0` off in all four runs to isolate
this line from the other two architecture lines:

| Config | use_transformer_flows | use_dur_disc | Role |
|---|---|---|---|
| E (baseline) | False | False | ≈ original VITS/Piper flow+duration |
| F | True | False | Isolates transformer-flow's contribution |
| G | False | True | Isolates duration-discriminator's contribution |
| H (full subset) | True | True | Combined — compare against F+G−E |

This directly fixes the same kind of methodological gap identified in VITS2 itself (no ablation row
isolating either change from baseline, nor any 2-of-4 subset) — consistent with the same fix already
planned for the Snake/MRD ablation in §2.2.

Status: design decided, **no training run yet**. Eval via existing WER+UTMOS+Wilcoxon harness (see
memory: eval_harness_ready). Can likely be combined with the A/B/C/D Snake/MRD runs into one larger
factorial sweep if compute allows — not yet decided.
