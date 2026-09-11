import csv
import json
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

plt.rcParams["font.size"] = 11
plt.rcParams["figure.dpi"] = 150

piper_csv = "/home/dev/eval_h_vs_baseline/best-epoch=773-val_loss_mel=20.2158_per_sentence.csv"
banhmi_csv = "/home/dev/eval_h_vs_baseline/best-epoch=1079-val_loss_mel=19.2943_per_sentence.csv"
summary_path = "/home/dev/eval_h_vs_baseline/summary.json"

def load(path):
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))

p = load(piper_csv)
b = load(banhmi_csv)
piper_wer = np.array([float(r["wer"]) for r in p])
banhmi_wer = np.array([float(r["wer"]) for r in b])
piper_utmos = np.array([float(r["utmos"]) for r in p])
banhmi_utmos = np.array([float(r["utmos"]) for r in b])

with open(summary_path) as f:
    summary = json.load(f)
banhmi_stats, piper_stats = summary["checkpoints"]
wer_p = summary["paired_tests"][0]["wer_wilcoxon_p"]
utmos_p = summary["paired_tests"][0]["utmos_wilcoxon_p"]

fig, axes = plt.subplots(1, 2, figsize=(9.5, 5.2))

colors = {"Piper": "#2c5aa0", "Banhmi-TTS": "#0f7a72"}

# WER panel
ax = axes[0]
bp = ax.boxplot([piper_wer, banhmi_wer], tick_labels=["Piper\n(baseline)", "Banhmi-TTS\n(full)"],
                 patch_artist=True, showfliers=False, widths=0.55, medianprops=dict(color="black"))
for patch, c in zip(bp["boxes"], [colors["Piper"], colors["Banhmi-TTS"]]):
    patch.set_facecolor(c)
    patch.set_alpha(0.55)
rng = np.random.default_rng(0)
ax.scatter(1 + rng.uniform(-0.12, 0.12, len(piper_wer)), piper_wer, s=6, color=colors["Piper"], alpha=0.35, zorder=3)
ax.scatter(2 + rng.uniform(-0.12, 0.12, len(banhmi_wer)), banhmi_wer, s=6, color=colors["Banhmi-TTS"], alpha=0.35, zorder=3)
ax.set_ylabel("WER (lower is better)")
ax.set_title("Word Error Rate", fontweight="bold")
sig = "n.s." if wer_p >= 0.05 else ("*" if wer_p >= 0.01 else ("**" if wer_p >= 0.001 else "***"))
ax.text(0.5, 0.97, f"Wilcoxon p = {wer_p:.3f} ({sig})", transform=ax.transAxes,
        ha="center", va="top", fontsize=9.5)
ax.text(0.02, 0.02, f"mean {piper_stats['mean_wer']:.3f}", transform=ax.transAxes, fontsize=8.5, color=colors["Piper"])
ax.text(0.98, 0.02, f"mean {banhmi_stats['mean_wer']:.3f}", transform=ax.transAxes, fontsize=8.5,
        color=colors["Banhmi-TTS"], ha="right")

# UTMOS panel
ax = axes[1]
bp = ax.boxplot([piper_utmos, banhmi_utmos], tick_labels=["Piper\n(baseline)", "Banhmi-TTS\n(full)"],
                 patch_artist=True, showfliers=False, widths=0.55, medianprops=dict(color="black"))
for patch, c in zip(bp["boxes"], [colors["Piper"], colors["Banhmi-TTS"]]):
    patch.set_facecolor(c)
    patch.set_alpha(0.55)
ax.scatter(1 + rng.uniform(-0.12, 0.12, len(piper_utmos)), piper_utmos, s=6, color=colors["Piper"], alpha=0.35, zorder=3)
ax.scatter(2 + rng.uniform(-0.12, 0.12, len(banhmi_utmos)), banhmi_utmos, s=6, color=colors["Banhmi-TTS"], alpha=0.35, zorder=3)
ax.set_ylabel("UTMOS (higher is better)")
ax.set_title("Predicted Naturalness (UTMOSv2)", fontweight="bold")
sig = "n.s." if utmos_p >= 0.05 else ("*" if utmos_p >= 0.01 else ("**" if utmos_p >= 0.001 else "***"))
ax.text(0.5, 0.03, f"Wilcoxon p = {utmos_p:.1e} ({sig})", transform=ax.transAxes,
        ha="center", va="bottom", fontsize=9.5)
ax.text(0.02, 0.97, f"mean {piper_stats['mean_utmos']:.3f}", transform=ax.transAxes, fontsize=8.5,
        color=colors["Piper"], va="top")
ax.text(0.98, 0.97, f"mean {banhmi_stats['mean_utmos']:.3f}", transform=ax.transAxes, fontsize=8.5,
        color=colors["Banhmi-TTS"], ha="right", va="top")

fig.suptitle("Baseline vs. full model: automatic quality metrics (n = 500, matched sentences)",
             fontweight="bold", fontsize=13)
plt.tight_layout(rect=[0, 0, 1, 0.94])
plt.savefig("/tmp/spec_wavs/fig_wer_utmos_comparison.png", dpi=200, bbox_inches="tight")
plt.savefig("/tmp/spec_wavs/fig_wer_utmos_comparison.pdf", bbox_inches="tight")
print("saved")
print("piper mean_rtf", piper_stats["mean_rtf"], "banhmi mean_rtf", banhmi_stats["mean_rtf"])
