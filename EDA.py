"""
Exploratory Data Analysis (EDA) for LJ-Speech Dataset
=====================================================
This script performs comprehensive EDA including:
- Audio duration distribution
- Text length distribution  
- Audio-text alignment (speaking rate)
- Sample waveform & spectrogram visualization
- Data quality verification
"""

import csv
import json
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import librosa
import librosa.display
import scipy.io.wavfile as wavfile
import torch
from tqdm import tqdm
from scipy import stats

# ============ CONFIGURATION ============
DATA_DIR = "."                    # Thư mục chứa wavs/ và metadata.csv
METADATA_FILE = os.path.join(DATA_DIR, "metadata.csv")
DATASET_JSONL = "/home/dev/data_preprocessed/dataset.jsonl"
OUTPUT_DIR = "eda_outputs"        # Thư mục lưu biểu đồ
SAMPLE_RATE = 22050               # LJ-Speech sample rate
MAX_PHONEME_IDS = 400             # Training-time phoneme-length cap

# Create output directory
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Set visualization style
sns.set_style("whitegrid")
plt.rcParams["figure.dpi"] = 150
plt.rcParams["font.size"] = 10


# ============ 1. LOAD METADATA ============
print("=" * 60)
print("STEP 1: Loading metadata...")
print("=" * 60)

df = pd.read_csv(
    METADATA_FILE,
    sep="|",
    header=None,
    names=["file_id", "text"],
    usecols=[0, 1],
    quoting=csv.QUOTE_NONE,
    dtype=str,
)
print(f"✅ Loaded {len(df)} entries from metadata.csv")
print(df.head())


# ============ 2. AUDIO DURATION ANALYSIS ============
print("\n" + "=" * 60)
print("STEP 2: Analyzing audio durations...")
print("=" * 60)

durations = []
missing_files = []

for _, row in tqdm(df.iterrows(), total=len(df), desc="Processing audio"):
    wav_path = os.path.join(DATA_DIR, "wavs", f"{row['file_id']}.wav")
    if not os.path.exists(wav_path):
        missing_files.append(row["file_id"])
        durations.append(np.nan)
        continue
    try:
        y, sr = librosa.load(wav_path, sr=SAMPLE_RATE)
        durations.append(len(y) / sr)
    except Exception as e:
        print(f"⚠️  Error loading {wav_path}: {e}")
        durations.append(np.nan)

df["duration"] = durations
valid_df = df.dropna(subset=["duration"])

print(f"\n📊 Duration Statistics:")
print(f"   Total clips       : {len(df)}")
print(f"   Valid clips       : {len(valid_df)}")
print(f"   Missing files     : {len(missing_files)}")
print(f"   Total duration    : {valid_df['duration'].sum() / 3600:.2f} hours")
print(f"   Mean duration     : {valid_df['duration'].mean():.2f}s")
print(f"   Median duration   : {valid_df['duration'].median():.2f}s")
print(f"   Std deviation     : {valid_df['duration'].std():.2f}s")
print(f"   Min / Max         : {valid_df['duration'].min():.2f}s / {valid_df['duration'].max():.2f}s")

# Plot duration distribution
plt.figure(figsize=(10, 6))
sns.histplot(valid_df["duration"], bins=50, kde=True, color="steelblue")
plt.axvline(valid_df["duration"].median(), color="red", linestyle="--",
            label=f"Median: {valid_df['duration'].median():.2f}s")
plt.xlabel("Duration (seconds)")
plt.ylabel("Number of clips")
plt.title("Distribution of Audio Clip Durations")
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "01_duration_distribution.png"))
print(f"✅ Saved: {OUTPUT_DIR}/01_duration_distribution.png")


# ============ 3. TEXT LENGTH ANALYSIS ============
print("\n" + "=" * 60)
print("STEP 3: Analyzing text properties...")
print("=" * 60)

df["text_length_chars"] = df["text"].str.len()
df["text_length_words"] = df["text"].str.split().str.len()

print(f"\n📊 Text Length Statistics (characters):")
print(f"   Mean   : {df['text_length_chars'].mean():.1f}")
print(f"   Median : {df['text_length_chars'].median():.1f}")
print(f"   Min    : {df['text_length_chars'].min()}")
print(f"   Max    : {df['text_length_chars'].max()}")

# Plot text length distribution
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

sns.histplot(df["text_length_chars"], bins=50, kde=True, ax=axes[0], color="seagreen")
axes[0].set_xlabel("Number of characters")
axes[0].set_ylabel("Count")
axes[0].set_title("Text Length (Characters)")

sns.histplot(df["text_length_words"], bins=50, kde=True, ax=axes[1], color="coral")
axes[1].set_xlabel("Number of words")
axes[1].set_ylabel("Count")
axes[1].set_title("Text Length (Words)")

plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "02_text_length_distribution.png"))
print(f"✅ Saved: {OUTPUT_DIR}/02_text_length_distribution.png")


# ============ 4. AUDIO-TEXT CORRELATION ============
print("\n" + "=" * 60)
print("STEP 4: Analyzing audio-text alignment...")
print("=" * 60)

valid_df = df.dropna(subset=["duration"]).copy()
valid_df["speaking_rate"] = valid_df["text_length_chars"] / valid_df["duration"]

correlation, p_value = stats.pearsonr(valid_df["text_length_chars"], valid_df["duration"])
print(f"\n📊 Correlation Analysis:")
print(f"   Pearson r (chars vs duration) : {correlation:.4f}")
print(f"   p-value                       : {p_value:.2e}")
print(f"   Median speaking rate          : {valid_df['speaking_rate'].median():.2f} chars/sec")

# Scatter plot
plt.figure(figsize=(10, 6))
plt.scatter(
    valid_df["text_length_chars"],
    valid_df["duration"],
    alpha=0.3, s=8, c="steelblue"
)
plt.xlabel("Text length (characters)")
plt.ylabel("Audio duration (seconds)")
plt.title(f"Audio Duration vs Text Length (r = {correlation:.3f})")

# Add regression line
z = np.polyfit(valid_df["text_length_chars"], valid_df["duration"], 1)
p = np.poly1d(z)
x_line = np.linspace(valid_df["text_length_chars"].min(),
                     valid_df["text_length_chars"].max(), 100)
plt.plot(x_line, p(x_line), "r--", linewidth=2, label="Linear fit")
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "03_duration_vs_text_length.png"))
print(f"✅ Saved: {OUTPUT_DIR}/03_duration_vs_text_length.png")

# Speaking rate distribution
plt.figure(figsize=(10, 6))
sns.histplot(valid_df["speaking_rate"], bins=50, kde=True, color="purple")
plt.xlabel("Speaking rate (characters/second)")
plt.ylabel("Count")
plt.title("Distribution of Speaking Rate")
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "04_speaking_rate_distribution.png"))
print(f"✅ Saved: {OUTPUT_DIR}/04_speaking_rate_distribution.png")


# ============ 5. SAMPLE WAVEFORM & SPECTROGRAM ============
print("\n" + "=" * 60)
print("STEP 5: Visualizing sample audio...")
print("=" * 60)

# Pick a representative sample (median duration)
sample_idx = (valid_df["duration"] - valid_df["duration"].median()).abs().idxmin()
sample_id = valid_df.loc[sample_idx, "file_id"]
sample_path = os.path.join(DATA_DIR, "wavs", f"{sample_id}.wav")

y, sr = librosa.load(sample_path, sr=SAMPLE_RATE)

fig, axes = plt.subplots(3, 1, figsize=(12, 10))

# Waveform
librosa.display.waveshow(y, sr=sr, ax=axes[0], color="steelblue")
axes[0].set_title(f"Waveform — Sample: {sample_id}")
axes[0].set_xlabel("Time (s)")
axes[0].set_ylabel("Amplitude")

# Mel-spectrogram
mel_spec = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128)
mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)
img = librosa.display.specshow(mel_spec_db, sr=sr, x_axis="time", y_axis="mel",
                                ax=axes[1], cmap="magma")
axes[1].set_title("Mel-Spectrogram")
axes[1].set_xlabel("Time (s)")
fig.colorbar(img, ax=axes[1], format="%+2.0f dB")

# Pitch (F0) via pyin
f0, voiced_flag, voiced_probs = librosa.pyin(y, fmin=50, fmax=500, sr=sr)
times = librosa.times_like(f0, sr=sr)
axes[2].plot(times, f0, color="darkgreen", linewidth=1)
axes[2].set_title("Fundamental Frequency (F0) — Pitch Contour")
axes[2].set_xlabel("Time (s)")
axes[2].set_ylabel("Frequency (Hz)")
axes[2].set_ylim(0, 500)

plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "05_sample_audio_visualization.png"))
print(f"✅ Saved: {OUTPUT_DIR}/05_sample_audio_visualization.png")


# ============ 6. OUTLIER DETECTION ============
print("\n" + "=" * 60)
print("STEP 6: Detecting outliers...")
print("=" * 60)

Q1 = valid_df["speaking_rate"].quantile(0.25)
Q3 = valid_df["speaking_rate"].quantile(0.75)
IQR = Q3 - Q1
lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR

outliers = valid_df[(valid_df["speaking_rate"] < lower_bound) |
                    (valid_df["speaking_rate"] > upper_bound)]

print(f"\n📊 Outlier Detection (speaking rate):")
print(f"   IQR bounds        : [{lower_bound:.2f}, {upper_bound:.2f}] chars/sec")
print(f"   Number of outliers: {len(outliers)} ({len(outliers)/len(valid_df)*100:.2f}%)")

if len(outliers) > 0:
    print(f"\n   Top 5 outliers:")
    print(outliers.nlargest(5, "speaking_rate")[["file_id", "duration",
                                                  "text_length_chars", "speaking_rate"]])


# ============ 7. SILENCE RATIO ANALYSIS ============
print("\n" + "=" * 60)
print("STEP 7: Analyzing silence ratio (pre-VAD)...")
print("=" * 60)

leading_silence_ratios = []
trailing_silence_ratios = []
total_silence_ratios = []

for _, row in tqdm(valid_df.iterrows(), total=len(valid_df), desc="Silence analysis"):
    wav_path = os.path.join(DATA_DIR, "wavs", f"{row['file_id']}.wav")
    try:
        sr, data = wavfile.read(wav_path)
        if data.dtype != np.float32:
            data = data.astype(np.float32) / np.iinfo(data.dtype).max
        # Find voiced region using energy-based trim
        _, intervals = librosa.effects.trim(data, top_db=30)
        trim_start, trim_end = intervals[0], intervals[1]
        total_samples = len(data)
        leading = trim_start / total_samples
        trailing = (total_samples - trim_end) / total_samples
        total_sil = leading + trailing
        leading_silence_ratios.append(leading)
        trailing_silence_ratios.append(trailing)
        total_silence_ratios.append(total_sil)
    except Exception:
        leading_silence_ratios.append(np.nan)
        trailing_silence_ratios.append(np.nan)
        total_silence_ratios.append(np.nan)

valid_df = valid_df.copy()
valid_df["leading_silence"] = leading_silence_ratios
valid_df["trailing_silence"] = trailing_silence_ratios
valid_df["total_silence"] = total_silence_ratios

print(f"\n📊 Silence Ratio Statistics (proportion of clip):")
print(f"   Median leading silence  : {np.nanmedian(leading_silence_ratios)*100:.1f}%")
print(f"   Median trailing silence : {np.nanmedian(trailing_silence_ratios)*100:.1f}%")
print(f"   Median total silence    : {np.nanmedian(total_silence_ratios)*100:.1f}%")
print(f"   Clips with >20% silence : {(np.array(total_silence_ratios) > 0.2).sum()}")

fig, axes = plt.subplots(1, 3, figsize=(15, 5))
for ax, col, label, color in zip(
    axes,
    ["leading_silence", "trailing_silence", "total_silence"],
    ["Leading Silence Ratio", "Trailing Silence Ratio", "Total Silence Ratio"],
    ["steelblue", "coral", "seagreen"]
):
    data_col = valid_df[col].dropna() * 100
    sns.histplot(data_col, bins=50, kde=True, ax=ax, color=color)
    ax.axvline(data_col.median(), color="red", linestyle="--",
               label=f"Median: {data_col.median():.1f}%")
    ax.set_xlabel("Silence ratio (%)")
    ax.set_ylabel("Count")
    ax.set_title(label)
    ax.legend()
plt.suptitle("Silence Ratio Distribution", fontsize=13)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "06_silence_ratio_distribution.png"))
print(f"✅ Saved: {OUTPUT_DIR}/06_silence_ratio_distribution.png")


# ============ 8. PHONEME LENGTH DISTRIBUTION ============
print("\n" + "=" * 60)
print("STEP 8: Analyzing phoneme length distribution...")
print("=" * 60)

phoneme_lengths = []
with open(DATASET_JSONL, "r") as f:
    for line in tqdm(f, desc="Reading dataset.jsonl"):
        d = json.loads(line)
        phoneme_lengths.append(len(d["phoneme_ids"]))

phoneme_lengths = np.array(phoneme_lengths)
n_corrupted = (phoneme_lengths > MAX_PHONEME_IDS).sum()
n_normal = (phoneme_lengths <= MAX_PHONEME_IDS).sum()

print(f"\n📊 Phoneme Length Statistics:")
print(f"   Total utterances         : {len(phoneme_lengths):,}")
print(f"   Mean phoneme ids         : {phoneme_lengths.mean():.1f}")
print(f"   Median phoneme ids       : {np.median(phoneme_lengths):.1f}")
print(f"   Max (uncorrupted)        : {phoneme_lengths[phoneme_lengths <= MAX_PHONEME_IDS].max()}")
print(f"   Above cap ({MAX_PHONEME_IDS})         : {n_corrupted} utterances (corrupted)")
print(f"   Min corrupted            : {phoneme_lengths[phoneme_lengths > MAX_PHONEME_IDS].min() if n_corrupted else 'N/A'}")

plt.figure(figsize=(12, 6))
main_lengths = phoneme_lengths[phoneme_lengths <= 600]
sns.histplot(main_lengths, bins=60, kde=True, color="steelblue")
plt.xlabel("Number of phoneme IDs")
plt.ylabel("Count")
plt.title("Phoneme Sequence Length Distribution")
n_beyond = (phoneme_lengths > 600).sum()
if n_beyond > 0:
    plt.annotate(f"{n_beyond} entries beyond x-axis\n(up to {phoneme_lengths.max():,} IDs)",
                 xy=(0.98, 0.92), xycoords="axes fraction", ha="right", fontsize=9, color="gray")
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "07_phoneme_length_distribution.png"))
print(f"✅ Saved: {OUTPUT_DIR}/07_phoneme_length_distribution.png")


# ============ 9. F0 CORPUS-LEVEL ANALYSIS ============
print("\n" + "=" * 60)
print("STEP 9: Analyzing F0 distribution (corpus-level)...")
print("=" * 60)

mean_f0_list = []
min_f0_list = []
max_f0_list = []
f0_range_list = []

with open(DATASET_JSONL, "r") as f:
    lines = f.readlines()

for line in tqdm(lines, desc="Loading F0 tensors"):
    d = json.loads(line)
    try:
        f0 = torch.load(d["audio_f0_path"], weights_only=False).numpy()
        voiced = f0[f0 > 0]
        if len(voiced) > 0:
            mean_f0_list.append(voiced.mean())
            min_f0_list.append(voiced.min())
            max_f0_list.append(voiced.max())
            f0_range_list.append(voiced.max() - voiced.min())
    except Exception:
        pass

mean_f0_arr = np.array(mean_f0_list)
f0_range_arr = np.array(f0_range_list)

print(f"\n📊 F0 Statistics (corpus-level):")
print(f"   Utterances with F0 data : {len(mean_f0_arr):,}")
print(f"   Mean F0 (corpus)        : {mean_f0_arr.mean():.1f} Hz")
print(f"   Median mean F0          : {np.median(mean_f0_arr):.1f} Hz")
print(f"   F0 range (corpus)       : {mean_f0_arr.min():.1f} – {mean_f0_arr.max():.1f} Hz")
print(f"   Median pitch range/utt  : {np.median(f0_range_arr):.1f} Hz")

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

sns.histplot(mean_f0_arr, bins=60, kde=True, ax=axes[0], color="darkgreen")
axes[0].axvline(np.median(mean_f0_arr), color="red", linestyle="--",
                label=f"Median: {np.median(mean_f0_arr):.1f} Hz")
axes[0].set_xlabel("Mean F0 per utterance (Hz)")
axes[0].set_ylabel("Count")
axes[0].set_title("Distribution of Mean F0 per Utterance")
axes[0].legend()

sns.histplot(f0_range_arr, bins=60, kde=True, ax=axes[1], color="darkorange")
axes[1].axvline(np.median(f0_range_arr), color="red", linestyle="--",
                label=f"Median: {np.median(f0_range_arr):.1f} Hz")
axes[1].set_xlabel("F0 range per utterance (Hz)")
axes[1].set_ylabel("Count")
axes[1].set_title("Distribution of Pitch Range per Utterance")
axes[1].legend()

plt.suptitle("Corpus-Level F0 Analysis", fontsize=13)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "08_f0_corpus_distribution.png"))
print(f"✅ Saved: {OUTPUT_DIR}/08_f0_corpus_distribution.png")


# ============ 10. SUMMARY ============
print("\n" + "=" * 60)
print("📋 EDA SUMMARY")
print("=" * 60)
summary = f"""
Dataset: LJ-Speech
==================
Total clips          : {len(df):,}
Valid clips          : {len(valid_df):,}
Missing files        : {len(missing_files)}
Total duration       : {valid_df['duration'].sum()/3600:.2f} hours
Median duration      : {valid_df['duration'].median():.2f}s
Median text length   : {df['text_length_chars'].median():.0f} characters
Median speaking rate : {valid_df['speaking_rate'].median():.2f} chars/sec
Duration-text corr.  : r = {correlation:.3f}
Outliers detected    : {len(outliers)}

Generated visualizations:
  - {OUTPUT_DIR}/01_duration_distribution.png
  - {OUTPUT_DIR}/02_text_length_distribution.png
  - {OUTPUT_DIR}/03_duration_vs_text_length.png
  - {OUTPUT_DIR}/04_speaking_rate_distribution.png
  - {OUTPUT_DIR}/05_sample_audio_visualization.png
"""
print(summary)

# Save summary to file
with open(os.path.join(OUTPUT_DIR, "eda_summary.txt"), "w") as f:
    f.write(summary)
print(f"✅ Summary saved to: {OUTPUT_DIR}/eda_summary.txt")
print("\n🎉 EDA completed successfully!")