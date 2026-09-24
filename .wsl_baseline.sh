#!/bin/bash
set -e
source /home/dev/miniconda3/bin/activate edgetts

echo "--- syncing code ---"
rsync -a --exclude='.git' --exclude='data' --exclude='notebooks' \
    /mnt/c/Users/duyng/OneDrive/Documents/Repository/FPT-Graduation-Project/PitchFlowNet/ /home/dev/PitchFlowNet/

cd /home/dev/PitchFlowNet/src/python

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

python -m piper_train \
    --dataset-dir /home/dev/data_preprocessed \
    --default_root_dir /home/dev/02_Baseline_VanillaVITS \
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
    --use-f0 false \
    --grad-clip 1.0
