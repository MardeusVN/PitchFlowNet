# Banhmi-TTS

A single-speaker neural TTS system built as an FPT University graduation project. Banhmi-TTS extends the [Piper](https://github.com/OHF-Voice/piper1-gpl) / VITS training pipeline with three architectural improvements: a BigVGAN-style decoder, VITS2 flow and duration improvements, and explicit F0 conditioning. The project conducts a 2³ ablation study (Configs A–H) to measure each component's contribution independently.

> Built on top of [Piper](https://github.com/rhasspy/piper) (MIT License, © 2022 Michael Hansen).

---

## Architecture

Banhmi-TTS (Config H) adds three orthogonal changes on top of the VITS baseline:

| Component | Flag | What changes |
|---|---|---|
| **BigVGAN decoder** | `use_bigvgan` | SnakeBeta activation replaces LeakyReLU; Multi-Resolution Discriminator (MRD) added alongside MPD |
| **VITS2 flow + duration** | `use_vits2` | TransformerCouplingLayer replaces WaveNet-only coupling; DurationDiscriminator added; MAS gets Gaussian noise perturbation |
| **F0 conditioning** | `use_f0` | Per-phoneme F0 Predictor trained with MSE loss; frame-level F0 injected into decoder via zero-init `f0_cond` Conv1d |

### Key module connections (Config H)

```
Phoneme IDs ──► Text Encoder ──► h ──► SDP ──► duration w
                             └──► F0 Predictor ──► log_f0_pred
                             └──► Normalizing Flow (VITS2) ◄── z (Posterior, train)
                                        │
                                   MAS Alignment (train)
                                        │ attn
                             ┌──────────┴──────────┐
                             ▼                     ▼
                    phone_f0 (GT target)     expand log_f0_pred (infer)
                             │                     │ f0_frame
                          L_f0 MSE            ┌────┘
                                              ▼
               z_slice (train) ──► Generator (BigVGAN) ──► ŷ ──► MPD + MRD
               f0_slice GT (train) ──►    │                        │
               z flow⁻¹ (infer) ──►       │                   DurDisc (VITS2)
               f0_frame (infer) ──►        └──────────────────────────────────►
```

**Inference parameters:** ~18 M (generator + predictors). Posterior Encoder (7.2 M), MRD (280 K), and DurationDiscriminator (120 K) are training-only.

---

## Ablation Study — 2³ Factorial Design

| Config | `use_bigvgan` | `use_vits2` | `use_f0` | Description |
|:---:|:---:|:---:|:---:|---|
| **A** | ✗ | ✗ | ✗ | Baseline VITS (Piper-compatible) |
| **B** | ✓ | ✗ | ✗ | +BigVGAN only |
| **C** | ✗ | ✓ | ✗ | +VITS2 only |
| **D** | ✗ | ✗ | ✓ | +F0 only |
| **E** | ✓ | ✓ | ✗ | +BigVGAN +VITS2 |
| **F** | ✓ | ✗ | ✓ | +BigVGAN +F0 |
| **G** | ✗ | ✓ | ✓ | +VITS2 +F0 |
| **H** | ✓ | ✓ | ✓ | Full model (Banhmi-TTS) |

Config files: [`configs/`](configs/)

---

## Requirements

- Python 3.9+
- `espeak-ng` (`sudo apt-get install espeak-ng`)
- CUDA-capable GPU (training validated on 2× RTX 4070 Ti, 12 GB VRAM each)
- Multi-GPU training requires **WSL2 or Linux** (NCCL; Windows DDP with CUDA is not supported)

---

## Installation

```bash
cd src/python
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip setuptools wheel
pip install -e .

# Build the Cython monotonic alignment extension (required for training)
bash build_monotonic_align.sh
```

---

## Dataset Preparation

Banhmi-TTS is trained on [LJSpeech](https://keithito.com/LJ-Speech-Dataset/) (single speaker, 22 050 Hz, ~24 h).

**Expected directory layout:**

```
/path/to/dataset/
├── metadata.csv      # pipe-delimited: id|text  (no header)
└── wav/
    ├── LJ001-0001.wav
    └── ...
```

**Run preprocessing:**

```bash
python3 -m piper_train.preprocess \
    --language en-us \
    --input-dir /path/to/dataset/ \
    --output-dir /path/to/data_preprocessed/ \
    --dataset-format ljspeech \
    --single-speaker \
    --sample-rate 22050
```

This produces `config.json`, `dataset.jsonl`, and `.pt` audio/spectrogram tensors.

> **Note:** Preprocessing silently drops ~189 rows (~1.4%) due to a known CSV quoting edge case. All ablation configs are trained on the same 12,911-row split to keep comparisons fair.

---

## Training

Each config has a corresponding YAML file. Select the one you want:

```bash
# Config A — baseline
python3 -m piper_train --config configs/baseline-vits.yaml

# Config H — full model (BigVGAN + VITS2 + F0)
python3 -m piper_train --config configs/01_Baseline_BigVgan_VITS2_FO.yaml
```

Checkpoints are saved under `lightning_logs/version_0/checkpoints/` (excluded from git).

**Monitor training:**

```bash
tensorboard --logdir lightning_logs/
```

Watch `loss_disc_all` and `loss_gen_all`. Training runs to ~1100 epochs from scratch.

---

## Inference

Generate audio from a JSONL file of phoneme IDs:

```bash
cat test_sentences.jsonl | \
    python3 -m piper_train.infer \
        --sample-rate 22050 \
        --checkpoint lightning_logs/version_0/checkpoints/epoch=*.ckpt \
        --output-dir ./output/
```

---

## Evaluation

The eval harness computes WER (Whisper-large-v3), UTMOS, and RTF on 500 held-out utterances:

```bash
python3 -m piper_train.eval_harness \
    --checkpoint /path/to/checkpoint.ckpt \
    --test-file /path/to/data_preprocessed/test_entries_500.jsonl \
    --output-dir ./eval_results/
```

**Known results (Config A vs Config H best checkpoint):**

| Metric | Config A (ep. 773) | Config H (ep. 1079) |
|---|:---:|:---:|
| WER ↓ | 9.17% | TBD |
| UTMOS ↑ | 3.366 | TBD |
| CPU RTF ↓ | 0.0559 | 0.0568 (+1.6%) |

---

## Export

Export a trained checkpoint to ONNX for Piper-compatible inference:

```bash
python3 -m piper_train.export_onnx \
    /path/to/model.ckpt \
    /path/to/model.onnx

cp /path/to/data_preprocessed/config.json /path/to/model.onnx.json
```

---

## Project Structure

```
configs/                  # Training YAML for each ablation config (A–H)
docs/                     # Training guides and data processing notes
src/
└── python/
    └── piper_train/
        ├── vits/
        │   ├── models.py       # SynthesizerTrn — main model, F0 path
        │   ├── modules.py      # SnakeBeta, ResBlock2, MRF, BigVGAN layers
        │   ├── attentions.py   # TransformerCouplingLayer (VITS2)
        │   ├── losses.py       # Generator and discriminator losses
        │   └── lightning.py    # PyTorch Lightning training module
        ├── preprocess.py       # Dataset preprocessing
        ├── eval_harness.py     # WER + UTMOS + Wilcoxon evaluation
        ├── measure_rtf.py      # CPU real-time factor measurement
        ├── export_onnx.py      # ONNX export
        └── infer.py            # Waveform generation from phoneme IDs
```

---

## Citation

If you use this code, please cite:

```bibtex
@misc{banhmitts2025,
  author = {Nguyen, Duy},
  title  = {Banhmi-TTS: Integrating Modern Vocoder, Flow, and Pitch-Conditioning
             Advances into a Lightweight Production Text-to-Speech System},
  year   = {2025},
  note   = {FPT University Graduation Project}
}
```

This project builds on:
- [Piper](https://github.com/rhasspy/piper) — Hansen (2023)
- [VITS](https://arxiv.org/abs/2106.06103) — Kim et al. (2021)
- [VITS2](https://arxiv.org/abs/2307.16430) — Kong et al. (2023)
- [BigVGAN](https://arxiv.org/abs/2206.04658) — Lee et al. (2023)
- [FastPitch](https://arxiv.org/abs/2006.04577) — Łańcucki (2021)

---

## License

This project inherits the MIT License from upstream Piper. See [LICENSE.md](LICENSE.md).
