#!/bin/bash
# Ablation training: configs B through G, from scratch, sequential.
# Each config uses both GPUs (devices=2) — cannot parallelize on a 2-GPU machine.
# Expected time: ~800-900 epochs per config = ~6× config H training time total.
#
# To run a single config, comment out the others below.
# Logs go to: /home/dev/<config_dir>/train.log

set -e
source /home/dev/miniconda3/bin/activate edgetts

echo "--- syncing code ---"
rsync -a --exclude='.git' --exclude='data' --exclude='notebooks' \
    /mnt/c/Users/duyng/OneDrive/Documents/Repository/FPT-Graduation-Project/EdgeTTS/ /home/dev/EdgeTTS/

cd /home/dev/EdgeTTS/src/python
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# ── Config B: BigVGAN only ────────────────────────────────────────────────────
echo "======= CONFIG B: +BigVGAN ======="
python -m piper_train \
    --dataset-dir /home/dev/data_preprocessed \
    --default_root_dir /home/dev/03_Config_B_BigVGAN \
    --accelerator gpu \
    --devices 2 \
    --strategy ddp_find_unused_parameters_true \
    --precision bf16-mixed \
    --batch-size 16 \
    --num-val-examples 100 \
    --num-test-examples 500 \
    --max_epochs 1100 \
    --checkpoint-epochs 1 \
    --seed 1234 \
    --use-bigvgan true \
    --use-vits2 false \
    --use-f0 false \
    --grad-clip 1.0
echo "======= CONFIG B DONE ======="

# ── Config C: VITS2 only ──────────────────────────────────────────────────────
echo "======= CONFIG C: +VITS2 ======="
python -m piper_train \
    --dataset-dir /home/dev/data_preprocessed \
    --default_root_dir /home/dev/04_Config_C_VITS2 \
    --accelerator gpu \
    --devices 2 \
    --strategy ddp_find_unused_parameters_true \
    --precision bf16-mixed \
    --batch-size 16 \
    --num-val-examples 100 \
    --num-test-examples 500 \
    --max_epochs 1100 \
    --checkpoint-epochs 1 \
    --seed 1234 \
    --use-bigvgan false \
    --use-vits2 true \
    --use-f0 false \
    --grad-clip 1.0
echo "======= CONFIG C DONE ======="

# ── Config D: F0 only ─────────────────────────────────────────────────────────
echo "======= CONFIG D: +F0 ======="
python -m piper_train \
    --dataset-dir /home/dev/data_preprocessed \
    --default_root_dir /home/dev/05_Config_D_F0 \
    --accelerator gpu \
    --devices 2 \
    --strategy ddp_find_unused_parameters_true \
    --precision bf16-mixed \
    --batch-size 16 \
    --num-val-examples 100 \
    --num-test-examples 500 \
    --max_epochs 1100 \
    --checkpoint-epochs 1 \
    --seed 1234 \
    --use-bigvgan false \
    --use-vits2 false \
    --use-f0 true \
    --grad-clip 1.0
echo "======= CONFIG D DONE ======="

# ── Config E: BigVGAN + VITS2 ─────────────────────────────────────────────────
echo "======= CONFIG E: +BigVGAN+VITS2 ======="
python -m piper_train \
    --dataset-dir /home/dev/data_preprocessed \
    --default_root_dir /home/dev/06_Config_E_BigVGAN_VITS2 \
    --accelerator gpu \
    --devices 2 \
    --strategy ddp_find_unused_parameters_true \
    --precision bf16-mixed \
    --batch-size 16 \
    --num-val-examples 100 \
    --num-test-examples 500 \
    --max_epochs 1100 \
    --checkpoint-epochs 1 \
    --seed 1234 \
    --use-bigvgan true \
    --use-vits2 true \
    --use-f0 false \
    --grad-clip 1.0
echo "======= CONFIG E DONE ======="

# ── Config F: BigVGAN + F0 ────────────────────────────────────────────────────
echo "======= CONFIG F: +BigVGAN+F0 ======="
python -m piper_train \
    --dataset-dir /home/dev/data_preprocessed \
    --default_root_dir /home/dev/07_Config_F_BigVGAN_F0 \
    --accelerator gpu \
    --devices 2 \
    --strategy ddp_find_unused_parameters_true \
    --precision bf16-mixed \
    --batch-size 16 \
    --num-val-examples 100 \
    --num-test-examples 500 \
    --max_epochs 1100 \
    --checkpoint-epochs 1 \
    --seed 1234 \
    --use-bigvgan true \
    --use-vits2 false \
    --use-f0 true \
    --grad-clip 1.0
echo "======= CONFIG F DONE ======="

# ── Config G: VITS2 + F0 ─────────────────────────────────────────────────────
echo "======= CONFIG G: +VITS2+F0 ======="
python -m piper_train \
    --dataset-dir /home/dev/data_preprocessed \
    --default_root_dir /home/dev/08_Config_G_VITS2_F0 \
    --accelerator gpu \
    --devices 2 \
    --strategy ddp_find_unused_parameters_true \
    --precision bf16-mixed \
    --batch-size 16 \
    --num-val-examples 100 \
    --num-test-examples 500 \
    --max_epochs 1100 \
    --checkpoint-epochs 1 \
    --seed 1234 \
    --use-bigvgan false \
    --use-vits2 true \
    --use-f0 true \
    --grad-clip 1.0
echo "======= CONFIG G DONE ======="

echo "======= ALL ABLATION CONFIGS DONE ======="
