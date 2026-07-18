# Banhmi-TTS preprocessing

This is the new English-only offline preprocessing pipeline. It preserves the
useful parts of Piper's dataset preparation while intentionally removing VAD
and waveform trimming.

## Data contract

Input uses the LJSpeech layout:

```text
dataset/
|-- metadata.csv       # utterance_id|text, no header
`-- wavs/
    `-- utterance_id.wav
```

The pipeline performs:

```text
metadata validation
-> eSpeak English phonemization
-> audio decode, mono conversion, and conditional resampling
-> non-destructive audio quality measurements
-> VITS-compatible linear spectrogram
-> WORLD F0 plus an explicit voiced/unvoiced mask
-> deterministic train/validation/test split
-> portable relative-path manifests and an audit report
```

No VAD, silence trimming, or peak normalization is applied. Leading/trailing
silence is measured for audit purposes only.

## Run

Install eSpeak-ng as a system dependency. Then create/install the isolated
environment from `src/python`:

```bash
conda create -n banhmi-tts python=3.10 pip -y
conda activate banhmi-tts
pip install -r requirements-banhmi-preprocess.txt
pip install -e . --no-deps
```

`--no-deps` is intentional: the legacy Piper package metadata depends on the
archived `piper-phonemize` wheel, while the new Banhmi frontend calls eSpeak-ng
directly and does not need that package.

Run from the repository root:

```bash
python -m banhmi_tts.preprocessing \
  --input-dir /path/to/dataset \
  --output-dir /path/to/banhmi_preprocessed \
  --language en-us \
  --sample-rate 22050 \
  --num-validation 100 \
  --num-test 500 \
  --seed 1234 \
  --max-workers 8
```

For a smoke test, use a separate output directory and leave enough records for
the requested split:

```bash
python -m banhmi_tts.preprocessing \
  --input-dir /path/to/dataset \
  --output-dir /tmp/banhmi_smoke \
  --limit 10 \
  --num-validation 1 \
  --num-test 1 \
  --max-workers 1
```

## Output

```text
banhmi_preprocessed/
|-- config.json
|-- report.json
|-- manifest.jsonl
|-- failures.jsonl
|-- splits/
|   |-- train.jsonl
|   |-- validation.jsonl
|   `-- test.jsonl
|-- records/             # resumable cache sidecars
`-- artifacts/
    |-- audio/           # FloatTensor [1, samples]
    |-- spectrogram/     # FloatTensor [n_fft/2+1, frames]
    `-- pitch/           # f0_hz, log_f0, voiced; all [frames]
```

Artifact paths in manifests are relative to the output directory. Cache keys
cover the source audio SHA-256, utterance id, text, and the complete canonical
preprocessing configuration. Changing audio, text, or feature parameters
therefore creates a new cache entry instead of silently reusing stale tensors.
