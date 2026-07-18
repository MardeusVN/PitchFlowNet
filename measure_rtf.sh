#!/bin/bash
# Measure CPU real-time factor for a checkpoint.
# Usage: bash measure_rtf.sh [checkpoint_path]
set -e
source /home/dev/miniconda3/bin/activate edgetts

CKPT="${1:-/home/dev/01_Baseline_BigVgan_VITS2_FO/lightning_logs/version_2/checkpoints/best-epoch=1079-val_loss_mel=19.2943.ckpt}"
SENTENCES="/home/dev/test_sentences_500.txt"

echo "Checkpoint : $CKPT"
echo "Sentences  : $SENTENCES"

rsync -a --exclude='.git' --exclude='data' --exclude='notebooks' \
    /mnt/c/Users/duyng/OneDrive/Documents/Repository/FPT-Graduation-Project/EdgeTTS/ /home/dev/EdgeTTS/

cd /home/dev/EdgeTTS/src/python

python -m piper_train.measure_rtf \
    --checkpoint "$CKPT" \
    --sentences-file "$SENTENCES" \
    --num-sentences 20 \
    --language vi
