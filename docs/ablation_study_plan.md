# Ablation Study Plan (Roadmap 1.2)

Status: **planning only — no training started yet.** This document scopes what
needs to be built and run; it does not implement the toggle flags or launch
any training job.

Goal: isolate which of the four components added on top of vanilla VITS
(commit `73c04d8`) actually contribute to the gains measured between
Piper-Modern checkpoints — without this, "the full stack is better" is not a
defensible claim (see [q2_publication_roadmap.md](q2_publication_roadmap.md) §1.2).

---

## 1. Variant matrix

| # | Variant | Snake1d | MultiResolutionDiscriminator | Transformer-Flow + DurationDiscriminator | F0Predictor |
|---|---|---|---|---|---|
| 0 | Baseline (vanilla VITS, `73c04d8`) | – | – | – | – |
| 1 | + Snake1d only | yes | – | – | – |
| 2 | + MRD only | – | yes | – | – |
| 3 | + Transformer-Flow + DurationDiscriminator only | – | – | yes | – |
| 4 | + F0Predictor only | – | – | – | yes |
| 5 | Full combination (current Piper-Modern) | yes | yes | yes | yes |

Variant 0 doubles as the fair baseline from roadmap 1.1 — no separate training
run needed for that item once this one exists.

Transformer-Flow and DurationDiscriminator are grouped into a single arm
because that's how roadmap 1.2 lists them (VITS2-style flow change and its
matching discriminator are introduced together), not because they're
mechanically coupled in code.

---

## 2. What needs a toggle flag (not yet implemented)

Current code in `src/python/piper_train/vits/`:

- **Transformer-in-Flow** — already toggleable: `use_transformer_flows` in
  `models.py:338` (`ResidualCouplingBlock.__init__`). Reusable as-is.
- **DurationDiscriminator** — hardcoded instantiation in
  `lightning.py:124-130`. Needs a flag to skip creating `model_d_dur` and skip
  its loss term in `training_step_g`/`training_step_d`. Should share one
  on/off switch with `use_transformer_flows` per the grouping above.
- **Snake1d** — hardcoded in `Generator.__init__` (`models.py:454-478`,
  `pre_up_snakes` / `final_snake`). Needs a flag to fall back to the vanilla
  VITS generator activation (plain LReLU, no pre-activation) when disabled.
- **MultiResolutionDiscriminator** — hardcoded instantiation in
  `lightning.py:121-123`. Needs a flag to skip creating `model_d_mrd` and skip
  its loss term.
- **F0Predictor** — hardcoded in `SynthesizerTrn.__init__` (`models.py:932`)
  plus the `f0_cond` conditioning path in `Generator.forward`. Needs a flag to
  skip the predictor, its loss term, and the `f0_cond` injection.

All five flags should default to their **current** (`True`/enabled) value, so
existing Piper-Modern checkpoints and training scripts keep working unchanged
when no flag is passed — only the ablation runs pass `False` explicitly.

---

## 3. Compute cost

5 additional full training runs (variants 0–4; variant 5 already exists as
the current Piper-Modern run) at the same epoch budget as the baseline
(800–1000 epochs, plateau-based early stop — see prior agreement on matched
training budget). This is **more expensive than the single baseline run**
already planned for roadmap 1.1, and competes for the same scarce hardware
(both main GPUs at ~90% load on Piper-Modern; the weaker single-GPU machine
already earmarked for the 1.1 baseline).

Suggested sequencing once flags exist, if all 5 runs can't fit before the
deadline: run variant 0 (baseline) first since it's needed either way, then
prioritize whichever single-component arm is most central to the paper's
novelty claim — likely F0Predictor or Transformer-Flow+DurationDiscriminator,
since Snake1d/MRD are smaller, well-established generator/discriminator swaps
with precedent in BigVGAN/UnivNet.

---

## 4. Next action items (not started)

- [ ] Add the four missing toggle flags above to `models.py` / `lightning.py`.
- [ ] Confirm flags default to current behavior (no regression for ongoing training).
- [ ] Decide where each ablation run trains (GPU contention noted in §3).
- [ ] Launch variant 0 (baseline) — doubles as roadmap 1.1 deliverable.
- [ ] Launch variants 1–4 as hardware frees up.
- [ ] Re-run `eval_harness.py` against all 6 checkpoints once trained, with paired Wilcoxon tests across all pairs.
