#!/bin/bash
set -e
source /home/dev/miniconda3/bin/activate edgetts
REPO=/home/dev/EdgeTTS

echo "--- syncing code from Windows ---"
rsync -a --exclude='.git' --exclude='data' --exclude='notebooks' \
    /mnt/c/Users/duyng/OneDrive/Documents/Repository/FPT-Graduation-Project/EdgeTTS/ "$REPO/"

echo "--- syncing dataset from Windows (this can take a while, 3.6GB) ---"
mkdir -p /home/dev/data
rsync -a --info=progress2 \
    /mnt/c/Users/duyng/OneDrive/Documents/Repository/FPT-Graduation-Project/EdgeTTS/data/ /home/dev/data/

echo "--- running preprocess ---"
cd "$REPO/src/python"
python -m piper_train.preprocess \
    --input-dir /home/dev/data \
    --output-dir /home/dev/data_preprocessed \
    --language en-us \
    --sample-rate 22050 \
    --dataset-format ljspeech \
    --single-speaker \
    --max-workers 16

echo "--- done ---"
ls -la /home/dev/data_preprocessed
wc -l /home/dev/data_preprocessed/dataset.jsonl
