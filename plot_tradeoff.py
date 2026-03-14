import glob
import os
import re

import matplotlib.pyplot as plt
import matplotlib

matplotlib.rcParams.update({"font.size": 13})


def parse_eval(path):
    """Parse an eval.txt file, return (longyaal_cu, bleu, comet)."""
    text = open(path).read()
    longyaal = float(re.search(r"^\s+LongYAAL \(CU\)\s+([\d.]+)", text, re.MULTILINE).group(1))
    bleu = float(re.search(r"^\s+BLEU\s+([\d.]+)", text, re.MULTILINE).group(1))
    comet = float(re.search(r"^\s+COMET\s+([\d.]+)", text, re.MULTILINE).group(1)) * 100
    return longyaal, bleu, comet


# Discover all eval.txt files and group by (lang, approach)
lang_names = {"zh": "En→Zh", "de": "En→De", "it": "En→It"}
results = {}  # (lang_code, approach) -> sorted list of (laal, bleu, comet, seg)

for path in sorted(glob.glob("outputs/en-*/*/seg*_mss5.0_h0/eval.txt")):
    parts = path.split(os.sep)
    lang_code = parts[1].split("-")[1]  # en-zh -> zh
    approach = parts[2]                  # baseline or with-context
    seg = int(re.search(r"seg(\d+)", parts[3]).group(1))
    laal, bleu, comet = parse_eval(path)
    key = (lang_code, approach)
    results.setdefault(key, []).append((laal, bleu, comet, seg))

# Sort each group by segment size (latency)
for key in results:
    results[key].sort(key=lambda x: x[3])

# Get unique language codes in order
lang_codes = sorted(set(k[0] for k in results))
n_langs = len(lang_codes)

fig, axes = plt.subplots(1, n_langs, figsize=(6 * n_langs, 5), squeeze=False)

marker_kw = dict(markersize=8, linewidth=2)
styles = {
    "baseline": ("o-", "#2196F3", "Baseline"),
    "with-context": ("s-", "#FF5722", "+ Context"),
}

for col, lang in enumerate(lang_codes):
    ax = axes[0, col]
    for approach, (fmt, color, label) in styles.items():
        key = (lang, approach)
        if key not in results:
            continue
        data = results[key]
        laals = [d[0] for d in data]
        comets = [d[2] for d in data]
        segs = [d[3] for d in data]
        ax.plot(laals, comets, fmt, color=color, label=label, **marker_kw)
        for i, seg in enumerate(segs):
            offset_y = 6 if approach == "with-context" else -12
            ax.annotate(f"{seg/1000:.2f}s", (laals[i], comets[i]),
                        textcoords="offset points", xytext=(8, offset_y),
                        fontsize=10, color=color)

    ax.set_xlabel("LongYAAL (CU)")
    ax.set_ylabel("XCOMET-XL")
    ax.set_title(f"Quality-Latency: {lang_names.get(lang, lang)}")
    ax.legend()
    ax.grid(True, alpha=0.3)

fig.tight_layout()
fig.savefig("quality_latency_tradeoff.png", dpi=150, bbox_inches="tight")
print("Saved to quality_latency_tradeoff.png")
