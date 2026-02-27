import matplotlib.pyplot as plt
import matplotlib

matplotlib.rcParams.update({"font.size": 13})

# Data from eval.txt files (ideal_latency = StreamLAAL)
# Baseline: seg640, seg960, seg1280
baseline_laal = [2.202, 3.162, 4.191]
baseline_bleu = [45.16, 46.95, 48.07]
baseline_comet = [72.41, 76.28, 77.50]

# Abstract: seg640, seg960, seg1280
abstract_laal = [2.151, 3.113, 4.139]
abstract_bleu = [44.86, 47.87, 48.44]
abstract_comet = [73.18, 76.95, 78.05]

chunk_sizes = ["0.64", "0.96", "1.28"]

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

marker_kw = dict(markersize=8, linewidth=2)

# BLEU subplot
ax1.plot(baseline_laal, baseline_bleu, "o-", color="#2196F3", label="Baseline", **marker_kw)
ax1.plot(abstract_laal, abstract_bleu, "s-", color="#FF5722", label="+ Abstract", **marker_kw)
for i, cs in enumerate(chunk_sizes):
    ax1.annotate(f"{cs}s", (baseline_laal[i], baseline_bleu[i]),
                 textcoords="offset points", xytext=(8, -12), fontsize=10, color="#2196F3")
    ax1.annotate(f"{cs}s", (abstract_laal[i], abstract_bleu[i]),
                 textcoords="offset points", xytext=(8, 6), fontsize=10, color="#FF5722")
ax1.set_xlabel("StreamLAAL (s)")
ax1.set_ylabel("BLEU")
ax1.set_title("Quality-Latency Tradeoff: BLEU")
ax1.legend()
ax1.grid(True, alpha=0.3)

# COMET subplot
ax2.plot(baseline_laal, baseline_comet, "o-", color="#2196F3", label="Baseline", **marker_kw)
ax2.plot(abstract_laal, abstract_comet, "s-", color="#FF5722", label="+ Abstract", **marker_kw)
for i, cs in enumerate(chunk_sizes):
    ax2.annotate(f"{cs}s", (baseline_laal[i], baseline_comet[i]),
                 textcoords="offset points", xytext=(8, -12), fontsize=10, color="#2196F3")
    ax2.annotate(f"{cs}s", (abstract_laal[i], abstract_comet[i]),
                 textcoords="offset points", xytext=(8, 6), fontsize=10, color="#FF5722")
ax2.set_xlabel("StreamLAAL (s)")
ax2.set_ylabel("COMET")
ax2.set_title("Quality-Latency Tradeoff: COMET")
ax2.legend()
ax2.grid(True, alpha=0.3)

fig.tight_layout()
fig.savefig("quality_latency_tradeoff.png", dpi=150, bbox_inches="tight")
print("Saved to quality_latency_tradeoff.png")
