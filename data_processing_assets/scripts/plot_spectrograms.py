import textwrap
import librosa
import librosa.display
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

plt.rcParams["font.size"] = 10
plt.rcParams["figure.dpi"] = 150

pairs = [
    ("representative", "Dr. Clark did not see any other hole or wound on the President's head.",
     "WER 0.071 / 0.071   UTMOS 3.42 -> 3.85"),
    ("best", "At different points on the coast where I often visit they build great seagoing ships.",
     "WER 0.067 / 0.067   UTMOS 3.01 -> 4.36"),
]

fig, axes = plt.subplots(2, 2, figsize=(11, 7.5), sharey=True)

for col, (tag, text, metrics) in enumerate(pairs):
    for row, (model, fname) in enumerate([("Piper (baseline, epoch 773)", f"piper_{tag}.wav"),
                                            ("Banhmi-TTS (full, epoch 1079)", f"banhmi_{tag}.wav")]):
        y, sr = librosa.load(f"/tmp/spec_wavs/{fname}", sr=None)
        mel = librosa.feature.melspectrogram(y=y, sr=sr, n_fft=1024, hop_length=256, n_mels=80)
        mel_db = librosa.power_to_db(mel, ref=np.max)
        ax = axes[row, col]
        img = librosa.display.specshow(mel_db, sr=sr, hop_length=256, x_axis="time", y_axis="mel",
                                        ax=ax, cmap="magma", fmax=8000)
        ax.set_title(model, fontsize=10.5, fontweight="bold", pad=6)
        if col == 0:
            ax.set_ylabel("Mel freq (Hz)")
        else:
            ax.set_ylabel("")
        ax.set_xlabel("Time (s)" if row == 1 else "")

    pos_top = axes[0, col].get_position()
    col_x = (pos_top.x0 + pos_top.x1) / 2
    wrapped = textwrap.fill(f'"{text}"', width=42)
    fig.text(col_x, 0.895, wrapped, ha="center", va="top", fontsize=9, style="italic")
    fig.text(col_x, 0.825, metrics, ha="center", va="top", fontsize=9)

fig.colorbar(img, ax=axes, format="%+2.0f dB", fraction=0.02, pad=0.02, label="Power (dB)")
fig.suptitle("Matched-sentence mel-spectrogram comparison (500-sentence shared test set)",
             fontsize=13, fontweight="bold", y=0.98)

plt.subplots_adjust(top=0.78)
plt.savefig("/tmp/spec_wavs/fig_spectrogram_comparison.png", bbox_inches="tight", dpi=200)
plt.savefig("/tmp/spec_wavs/fig_spectrogram_comparison.pdf", bbox_inches="tight")
print("saved")
