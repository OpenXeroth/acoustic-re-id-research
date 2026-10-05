"""Render manuscript Figures 1 and 3 from the public review aggregates.

Run with: uv run --with matplotlib python scripts/plot_review_sensitivity.py
"""

import json
import pathlib

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

r = pathlib.Path(__file__).resolve().parents[1] / "docs/review-sensitivity"
d = json.loads((r / "split-and-time.json").read_text())
out = r / "figures"
out.mkdir(exist_ok=True)
plt.rcParams.update(
    {
        "font.family": "DejaVu Sans",
        "font.size": 10,
        "axes.spines.top": False,
        "axes.spines.right": False,
    }
)
heads = [
    ("kernel ridge to one-hot", "Kernel ridge"),
    ("mean cosine similarity to each identity", "Class mean"),
    ("single nearest enrollment clip, cosine", "Nearest clip"),
]
fig, axs = plt.subplots(1, 3, figsize=(10, 6.64), sharex=True, sharey=True)
for ax, sp, title in zip(
    axs,
    ["littleowl", "chiffchaff", "pipit"],
    ["Little owl", "Chiffchaff", "Tree pipit"],
    strict=True,
):
    for model, y0, c in [("birdnet", 0, "#276684"), ("google-perch", 3.5, "#738878")]:
        vals = d[f"matched-{sp}-{model}.json"]["summary"]
        for j, (key, _label) in enumerate(heads):
            v = vals[key]
            a = v["difference_mean"]
            lo, hi = v["difference_95_individual_bootstrap"]
            ax.errorbar(a, y0 + j, xerr=[[a - lo], [hi - a]], fmt="o", color=c, capsize=3)
    ax.set_title(title, fontweight="bold")
    ax.axvline(0, color="#999", linewidth=0.8)
    ax.set_xlim(-0.01, 0.74)
    ax.set_xticks([0, 0.2, 0.4, 0.6])
    ax.grid(axis="x", alpha=0.18)
    ax.set_xlabel("Accuracy difference")
axs[0].set_yticks([0, 1, 2, 3.5, 4.5, 5.5])
axs[0].set_yticklabels(["BirdNET / " + x[1] for x in heads] + ["Perch / " + x[1] for x in heads])
axs[0].invert_yaxis()
fig.suptitle("Random clip split − session-disjoint split", fontweight="bold", y=0.97)
fig.text(
    0.54,
    0.89,
    "Same clips · same per-bird training/test counts · ten fixed seeds",
    ha="center",
    fontsize=10,
)
fig.text(
    0.54,
    0.08,
    "Bars: paired 95% intervals over individuals, conditional on ten splits.",
    ha="center",
    fontsize=9,
)
fig.text(
    0.54,
    0.04,
    "Original rook comparison (one count-matched split): +0.126 [0.065, 0.172].",
    ha="center",
    fontsize=9,
    color="#9b5c28",
)
fig.subplots_adjust(left=0.23, right=0.98, top=0.80, bottom=0.18, wspace=0.15)
fig.savefig(out / "figure-3-count-matched.png", dpi=220)
fig.savefig(out / "figure-3-count-matched.pdf")
plt.close(fig)
fig, ax = plt.subplots(figsize=(9, 9.08))
ax.set(xlim=(0, 1), ylim=(0, 1))
ax.axis("off")


def box(x, y, w, h, text, color="#edf2ee", fs=11):
    ax.add_patch(
        FancyBboxPatch(
            (x, y),
            w,
            h,
            boxstyle="round,pad=0.009",
            facecolor=color,
            edgecolor="#738878",
            linewidth=1,
        )
    )
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fs, linespacing=1.45)


def arrow(x, y, xx, yy):
    ax.annotate(
        "",
        xy=(xx, yy),
        xytext=(x, y),
        arrowprops={"arrowstyle": "->", "color": "#465b4b", "lw": 1.4},
    )


ax.text(0.02, 0.965, "a  Identity can be tied to recording context", weight="bold", fontsize=14)
box(0.04, 0.79, 0.26, 0.11, "Bird A\nSite / recorder A")
box(0.37, 0.79, 0.26, 0.11, "Bird B\nSite / recorder B")
box(0.70, 0.79, 0.26, 0.11, "Bird C\nSite / recorder C")
ax.text(
    0.5,
    0.745,
    "A classifier may recognise the recording conditions as well as the voice.",
    ha="center",
    fontsize=11,
)
ax.text(0.02, 0.685, "b  What is divided between training and testing?", weight="bold", fontsize=14)
box(
    0.04,
    0.48,
    0.43,
    0.155,
    "Random clip split\nTrain: some Monday + Tuesday calls\n"
    "Test: other Monday + Tuesday calls\nShared recording context",
    fs=10.5,
)
box(
    0.53,
    0.48,
    0.43,
    0.155,
    "Session-disjoint split\nTrain: Monday calls\nTest: Tuesday calls\nNew recording session",
    fs=10.5,
)
ax.text(
    0.5,
    0.435,
    "Use the same birds and match training/test counts for a controlled comparison.",
    ha="center",
    fontsize=10,
)
ax.text(
    0.5,
    0.400,
    "Whole sessions may be assigned randomly while remaining separate.",
    ha="center",
    fontsize=10,
)
ax.text(0.02, 0.33, "c  Evaluate recognition on the intended task", weight="bold", fontsize=14)
box(0.04, 0.155, 0.25, 0.11, "Call clip\n(frozen encoder input)")
box(0.375, 0.155, 0.25, 0.11, "Embedding\n(acoustic features)")
box(0.71, 0.155, 0.25, 0.11, "Classifier\n(individual label)")
arrow(0.295, 0.21, 0.365, 0.21)
arrow(0.63, 0.21, 0.70, 0.21)
ax.text(
    0.5,
    0.095,
    "Fit the classifier on enrolment clips; evaluate on held-out test clips.",
    ha="center",
    fontsize=11,
)
ax.text(
    0.5,
    0.055,
    "A higher same-session test score does not establish better new-session recognition.",
    ha="center",
    fontsize=10,
    weight="bold",
)
fig.subplots_adjust(left=0.02, right=0.98, top=0.99, bottom=0.01)
fig.savefig(out / "figure-1-split-explanation.png", dpi=220)
fig.savefig(out / "figure-1-split-explanation.pdf")
plt.close(fig)
