#!/bin/bash
set -e
source /home/dev/miniconda3/bin/activate edgetts

echo "--- syncing code ---"
rsync -a --exclude='.git' --exclude='data' --exclude='notebooks' \
    /mnt/c/Users/duyng/OneDrive/Documents/Repository/FPT-Graduation-Project/EdgeTTS/ /home/dev/EdgeTTS/

echo "--- installing tensorboard ---"
pip install -q "tensorboard>=2.13,<3"

echo "--- launching training ---"
cd /home/dev/EdgeTTS/src/python
python -m piper_train \
    --dataset-dir /home/dev/data_preprocessed \
    --batch-size 8 \
    --validation-split 0.05 \
    --num-test-examples 5 \
    --max_epochs 10000 \
    --checkpoint-epochs 1
