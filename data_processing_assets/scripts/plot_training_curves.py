import csv
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams["font.size"] = 11
plt.rcParams["figure.dpi"] = 150


def load_dedup(path):
    with open(path, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    seen = {}
    for r in rows:
        step = int(r["step"])
        seen[step] = float(r["val_loss_mel"])  # later occurrence (resumed run) wins
    steps = sorted(seen)
    vals = [seen[s] for s in steps]
    epochs = list(range(0, len(steps)))  # Lightning epoch counter is 0-indexed
    return epochs, vals


piper_epochs, piper_vals = load_dedup("/tmp/spec_wavs/piper_baseline_scalars.csv")
banhmi_epochs, banhmi_vals = load_dedup("/tmp/spec_wavs/banhmi_full_scalars.csv")

print("piper points:", len(piper_epochs), "banhmi points:", len(banhmi_epochs))
print("piper @ epoch773 (sanity check, should be 20.2158):", piper_vals[773])
print("banhmi @ epoch1079 (sanity check, should be 19.2943):", banhmi_vals[1079])

fig, ax = plt.subplots(figsize=(9, 5.5))
ax.plot(piper_epochs, piper_vals, color="#2c5aa0", linewidth=1.1, alpha=0.9, label="Piper (baseline)")
ax.plot(banhmi_epochs, banhmi_vals, color="#0f7a72", linewidth=1.1, alpha=0.9, label="Banhmi-TTS (full)")

ax.scatter([773], [piper_vals[773]], color="#2c5aa0", s=70, zorder=5, edgecolor="white", linewidth=1.2)
ax.annotate(f"best checkpoint\nepoch 773, val_loss_mel={piper_vals[773]:.4f}",
            xy=(773, piper_vals[773]), xytext=(773 - 250, piper_vals[773] + 6),
            fontsize=9, color="#14304f",
            arrowprops=dict(arrowstyle="->", color="#2c5aa0", lw=1.2))

ax.scatter([1079], [banhmi_vals[1079]], color="#0f7a72", s=70, zorder=5, edgecolor="white", linewidth=1.2)
ax.annotate(f"best checkpoint\nepoch 1079, val_loss_mel={banhmi_vals[1079]:.4f}",
            xy=(1079, banhmi_vals[1079]), xytext=(1079 - 420, banhmi_vals[1079] - 9),
            fontsize=9, color="#0b4a44",
            arrowprops=dict(arrowstyle="->", color="#0f7a72", lw=1.2))

ax.set_xlabel("Epoch")
ax.set_ylabel("val_loss_mel")
ax.set_xlim(0, 1100)
ax.set_title("Validation mel loss over training: Piper vs. Banhmi-TTS", fontweight="bold", fontsize=13)
ax.legend(loc="upper right", frameon=True)
ax.grid(alpha=0.25)

plt.tight_layout()
plt.savefig("/tmp/spec_wavs/fig_training_curves.png", dpi=200, bbox_inches="tight")
plt.savefig("/tmp/spec_wavs/fig_training_curves.pdf", bbox_inches="tight")

# zoomed inset-style second panel: last 400 epochs, since early epochs dwarf the interesting convergence tail
fig2, ax2 = plt.subplots(figsize=(9, 5.5))
ax2.plot(piper_epochs, piper_vals, color="#2c5aa0", linewidth=1.3, alpha=0.9, label="Piper (baseline)")
ax2.plot(banhmi_epochs, banhmi_vals, color="#0f7a72", linewidth=1.3, alpha=0.9, label="Banhmi-TTS (full)")
ax2.scatter([773], [piper_vals[773]], color="#2c5aa0", s=70, zorder=5, edgecolor="white", linewidth=1.2)
ax2.scatter([1079], [banhmi_vals[1079]], color="#0f7a72", s=70, zorder=5, edgecolor="white", linewidth=1.2)
ax2.set_xlim(650, 1100)
ax2.set_ylim(min(min(piper_vals[649:]), min(banhmi_vals[649:])) - 0.3,
             max(max(piper_vals[649:]), max(banhmi_vals[649:])) + 0.3)
ax2.set_xlabel("Epoch")
ax2.set_ylabel("val_loss_mel")
ax2.set_title("Convergence tail (epoch 650-1100)", fontweight="bold", fontsize=13)
ax2.legend(loc="upper right", frameon=True)
ax2.grid(alpha=0.25)
plt.tight_layout()
plt.savefig("/tmp/spec_wavs/fig_training_curves_zoom.png", dpi=200, bbox_inches="tight")
plt.savefig("/tmp/spec_wavs/fig_training_curves_zoom.pdf", bbox_inches="tight")

print("saved")
