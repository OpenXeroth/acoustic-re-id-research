"""Render open-set comparison and a graphical abstract from public aggregates.

Run: uv run --with matplotlib python scripts/plot_review_open_set.py
"""

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

root = Path(__file__).resolve().parents[1] / "docs/review-sensitivity"
out = root / "figures"
plt.rcParams.update(
    {
        "font.family": "DejaVu Sans",
        "font.size": 10,
        "axes.spines.top": False,
        "axes.spines.right": False,
    }
)
endpoints = [
    ("great-tit", "Great tit (16)"),
    ("littleowl-acrossyear", "Little owl"),
    ("rookid-full-width", "Rook"),
    ("great-tit-full", "Great tit (50)"),
    ("cockatoo-fold1", "Cockatoo*"),
]
fig, axes = plt.subplots(1, 2, figsize=(10, 4.892), sharex=True, sharey=True)
for ax, model, name in zip(
    axes, ["birdnet-v2.4", "perch-v2"], ["BirdNET v2.4", "Perch 2.0"], strict=True
):
    for y, (ep, _) in enumerate(endpoints):
        d = json.loads((root / "results" / f"open-{ep}-{model}.json").read_text())
        for scorer, offset, color, symbol in [
            ("mean-profile", -0.12, "#738878", "o"),
            ("nearest-clip", 0.12, "#28668a", "s"),
        ]:
            v = d["scorers"][scorer]["acceptance_calibrated_at_0.10"]
            ax.errorbar(
                v["median"],
                y + offset,
                xerr=[[v["median"] - v["minimum"]], [v["maximum"] - v["median"]]],
                fmt=symbol,
                color=color,
                linewidth=1,
                capsize=2,
                label=scorer if y == 0 else None,
            )
    ax.set_title(name, fontweight="bold")
    ax.set_xlim(0, 1.02)
    ax.set_xlabel("Known calls accepted and correctly named")
    ax.grid(axis="x", alpha=0.18)
axes[0].set_yticks(range(5), [name for _, name in endpoints])
axes[0].invert_yaxis()
axes[1].legend(loc="upper right", frameon=False, fontsize=9)
fig.suptitle("Open-set results depend on the scorer", fontweight="bold", y=0.98)
fig.text(
    0.55,
    0.07,
    "Points: allocation medians; bars: minima–maxima, not confidence intervals.",
    ha="center",
    fontsize=9,
)
fig.text(
    0.55,
    0.025,
    "Separate calibration per scorer targets 10% strangers; test rates can exceed it. "
    "*Random fold.",
    ha="center",
    fontsize=9,
)
fig.subplots_adjust(left=0.17, right=0.98, bottom=0.24, top=0.82, wspace=0.12)
for ext in ["png", "pdf"]:
    fig.savefig(out / f"figure-6-scorers.{ext}", dpi=220)
plt.close(fig)

fig, ax = plt.subplots(figsize=(10, 7))
ax.set(xlim=(0, 1), ylim=(0, 1))
ax.axis("off")
ax.text(
    0.04,
    0.95,
    "Does the model recognise the animal—or the recording?",
    fontsize=18,
    weight="bold",
    color="#294c38",
)
ax.text(
    0.04,
    0.90,
    "Acoustic re-identification · 13 datasets · 9 species · 36 neural networks",
    fontsize=12,
)
cards = [
    (
        0.04,
        0.52,
        "1  Split whole sessions",
        "+20–51 percentage points",
        "Random clips raised measured accuracy across\n"
        "18 count-matched comparisons (ten seeds).\n"
        "A higher score does not prove better recognition\n"
        "on a new day.",
    ),
    (
        0.53,
        0.52,
        "2  Check the background",
        "71% background · 79% calls",
        "Balanced accuracy for within-year chiffchaffs.\n"
        "Recording context carries substantial information\n"
        "about the assigned identity.",
    ),
    (
        0.04,
        0.13,
        "3  Separate identity from context",
        "Timing matters too",
        "Shared-recording tests support one zebra-finch\n"
        "dataset most clearly. BirdPark lacks same-day\n"
        "same-bird pairs across recordings; time and\n"
        "recording changes cannot be fully separated.",
    ),
    (
        0.53,
        0.13,
        "4  Let unfamiliar animals arrive",
        "37% → 46% with another scorer",
        "Highest median known-call correct acceptance\n"
        "on wild session-disjoint endpoints: mean profile\n"
        "versus nearest clip. Calibration targets 10%\n"
        "strangers; actual test rates can exceed it.",
    ),
]
for x, y, title, value, detail in cards:
    ax.add_patch(
        FancyBboxPatch(
            (x, y), 0.43, 0.31, boxstyle="round,pad=.012", facecolor="#edf2ee", edgecolor="#738878"
        )
    )
    ax.text(x + 0.012, y + 0.263, title, weight="bold", fontsize=12, color="#294c38")
    ax.text(x + 0.012, y + 0.21, value, weight="bold", fontsize=12)
    ax.text(x + 0.012, y + 0.165, detail, fontsize=10.5, va="top", linespacing=1.6)
ax.text(
    0.04,
    0.05,
    "Test the task you intend to use. Preserve untouched sessions. "
    "Calibrate and report stranger errors.",
    fontsize=11,
    weight="bold",
    color="#294c38",
)
fig.subplots_adjust(left=0.02, right=0.98, top=0.99, bottom=0.01)
for ext in ["png", "pdf"]:
    fig.savefig(out / f"graphical-abstract-v9.{ext}", dpi=200)
plt.close(fig)
