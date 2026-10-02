"""Every figure and every table figure in the paper, read from the archived results.

Run with any Python that has numpy and matplotlib:

    python scripts/paper_figures.py --results <archive mirror> --out OUT

`scripts/paper_v6_build.sh` runs it on the carried-analysis stage.

It reads result files only, by name, and writes `figures/*.png` and
`numbers.json`. No number in the manuscript is typed from another document:
each one is in `numbers.json`, which records the file and digest it came from.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import FancyBboxPatch  # noqa: E402

BLUE = "#2a78d6"
ORANGE = "#eb6834"
GREY = "#6b6b6b"
LIGHT = "#bdbdbd"
INK = "#1f1f1f"

plt.rcParams.update(
    {
        "font.family": "serif",
        "font.size": 9,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.edgecolor": INK,
        "axes.labelcolor": INK,
        "xtick.color": INK,
        "ytick.color": INK,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
    }
)

#: The headline endpoints, in the order the paper's table lists them, with the
#: label, the recording design and the file stem of the v5 headline run.
ENDPOINTS: list[tuple[str, str, str]] = [
    ("birdpark-juv01", "Zebra finch, group of four", "shared"),
    ("birdpark-juv03", "Zebra finch, group of eight", "shared"),
    ("rook-full-across-year-capped-90", "Rook, aviary, across year", "shared"),
    ("zebra-finch", "Zebra finch, one bird per recording", "single"),
    ("chiffchaff-withinyear", "Chiffchaff, within year", "territory"),
    ("chiffchaff-acrossyear", "Chiffchaff, across year", "territory"),
    ("littleowl-acrossyear", "Little owl, across year", "territory"),
    ("pipit-withinyear", "Tree pipit, within year", "territory"),
    ("pipit-acrossyear", "Tree pipit, across year", "territory"),
    ("great-tit", "Great tit, across year", "territory"),
    ("penguin-acrossnight", "Little penguin, across night", "territory"),
    ("cockatoo-fold1", "Red-tailed black cockatoo, random split", "territory"),
    ("bat-acrosstreatment", "Egyptian fruit bat, across year", "single"),
]

DESIGN_LABEL = {
    "shared": "several animals share each recording",
    "single": "one animal per recording, captive",
    "territory": "each animal at its own site, wild",
}


class Numbers:
    """Collects every value the paper cites, with the file and digest it came from."""

    def __init__(self, root: Path) -> None:
        self.root = root
        self.values: dict[str, Any] = {}
        self.sources: dict[str, str] = {}

    def load(self, name: str) -> dict[str, Any]:
        path = self.root / name
        data = path.read_bytes()
        self.sources[name] = hashlib.sha256(data).hexdigest()
        return json.loads(data)

    def put(self, key: str, value: Any, source: str) -> Any:
        self.values[key] = {"value": value, "source": source}
        return value

    def write(self, path: Path) -> None:
        path.write_text(
            json.dumps({"sources": self.sources, "values": self.values}, indent=1, sort_keys=True)
            + "\n"
        )


def r3(value: float | None) -> float | None:
    return None if value is None else round(float(value), 3)


# --- the headline table -----------------------------------------------------


def headline(numbers: Numbers) -> list[dict[str, Any]]:
    rows = []
    for stem, label, design in ENDPOINTS:
        name = f"v5-identity-{stem}.json"
        try:
            data = numbers.load(name)
        except FileNotFoundError:
            continue
        evaluation = data["evaluation"]
        within = evaluation.get("within_session_permutation") or {}
        hierarchical = evaluation.get("hierarchical_bootstrap")
        trials = data.get("stratified_trials") or {}
        animal = trials.get("animal_against_recording") or {}
        row = {
            "stem": stem,
            "label": label,
            "design": design,
            "individuals": data["identities"],
            "enrolment_clips": evaluation["enrollment_calls"],
            "query_clips": evaluation["query_calls"],
            "accuracy": r3(evaluation["classification"]["accuracy"]),
            "interval": [r3(v) for v in evaluation["identity_block_bootstrap_accuracy_95"]],
            "three_stage": [r3(v) for v in hierarchical["accuracy_95"]] if hierarchical else None,
            "uniform_chance": r3(data["uniform_chance"]),
            "majority": r3(data["majority_class_rate"]),
            "permutation_95": r3(evaluation["permutation_upper_tail"]["percentile_95"]),
            "permutation_p": evaluation["permutation_control"]["p_value_plus_one"],
            "within_p": within.get("p_value_plus_one"),
            "within_95": r3(within.get("percentile_95")),
            "within_reason": within.get("reason"),
            "roc_auc": r3(evaluation["verification"]["roc_auc"]),
            "trials_difference": animal.get("difference"),
            "trials_standardised": animal.get("standardised_difference"),
            "trials_eer": (animal.get("equal_error_rate") or {}).get("equal_error_rate"),
        }
        numbers.put(f"headline.{stem}", row, name)
        rows.append(row)
    return rows


def figure_headline(rows: list[dict[str, Any]], out: Path) -> None:
    fig, ax = plt.subplots(figsize=(6.8, 0.36 * len(rows) + 1.2))
    colours = {"shared": BLUE, "single": GREY, "territory": ORANGE}
    for y, row in enumerate(reversed(rows)):
        colour = colours[row["design"]]
        low, high = row["interval"]
        ax.plot([low, high], [y, y], color=colour, lw=1.6, solid_capstyle="round")
        cleared = row["permutation_p"] is not None and row["permutation_p"] <= 0.05
        ax.plot(
            row["accuracy"],
            y,
            "o",
            ms=6,
            color=colour if cleared else "white",
            markeredgecolor=colour,
            markeredgewidth=1.5,
            zorder=3,
        )
        ax.plot([row["permutation_95"]] * 2, [y - 0.28, y + 0.28], color=INK, lw=1.2)
        if row.get("majority") is not None:
            ax.plot(row["majority"], y, marker="v", ms=5, color=INK, lw=0, zorder=4)
        ax.text(1.02, y, f"{row['accuracy']:.3f}", va="center", fontsize=8, color=INK)
    ax.set_yticks(range(len(rows)))
    ax.set_yticklabels([f"{r['label']} ({r['individuals']})" for r in reversed(rows)])
    ax.set_xlim(0, 1)
    ax.set_xlabel("re-ID accuracy, BirdNET v2.4")
    handles = [
        plt.Line2D([], [], color=c, marker="o", lw=1.6, label=DESIGN_LABEL[k])
        for k, c in colours.items()
    ]
    handles.append(
        plt.Line2D(
            [],
            [],
            color=INK,
            marker="|",
            ms=10,
            lw=0,
            label="95th percentile of the permutation test",
        )
    )
    handles.append(
        plt.Line2D(
            [],
            [],
            color=GREY,
            marker="o",
            markerfacecolor="white",
            lw=0,
            label="does not exceed that percentile",
        )
    )
    handles.append(
        plt.Line2D([], [], color=INK, marker="v", ms=5, lw=0, label="majority-class rate")
    )
    ax.legend(
        handles=handles,
        loc="upper center",
        bbox_to_anchor=(0.4, -0.16),
        ncol=2,
        frameon=False,
        fontsize=7.5,
    )
    fig.savefig(out)
    plt.close(fig)


# --- the conceptual figure ----------------------------------------------------


def figure_concept(out: Path) -> None:
    fig = plt.figure(figsize=(7.0, 6.2))
    top = fig.add_axes([0.0, 0.70, 1.0, 0.28])
    top.set_axis_off()
    top.set_xlim(0, 4)
    top.set_ylim(0, 1)
    top.text(
        0.0,
        0.95,
        "(a)  How attribution can create a confound",
        fontsize=10,
        weight="bold",
    )
    boxes = [
        "A label is assigned using\na collar, tag, nest box,\nterritory or station",
        "The attribution method\ncan also fix where and how\na call is recorded",
        "Identity labels and\nrecording conditions\nbecome associated",
        "A classifier can score high\nby reading the channel,\nnot the animal",
    ]
    for i, text in enumerate(boxes):
        x = 0.05 + i * 0.99
        top.add_patch(
            FancyBboxPatch(
                (x, 0.12), 0.84, 0.62, boxstyle="round,pad=0.02", fc="#f3f3f3", ec=LIGHT, lw=0.8
            )
        )
        top.text(x + 0.42, 0.43, text, ha="center", va="center", fontsize=8)
        if i < 3:
            top.annotate(
                "",
                xy=(x + 0.97, 0.43),
                xytext=(x + 0.87, 0.43),
                arrowprops={"arrowstyle": "-|>", "color": GREY},
            )

    mid = fig.add_axes([0.0, 0.33, 1.0, 0.35])
    mid.set_axis_off()
    mid.set_xlim(0, 10)
    mid.set_ylim(0, 10)
    mid.text(0.0, 9.6, "(b)  Two recording designs", fontsize=10, weight="bold")
    mid.text(2.5, 8.4, "Territorial design", ha="center", fontsize=9.5, color=ORANGE, weight="bold")
    for bx, by, n in [(1.2, 5.8, 1), (3.8, 6.2, 2), (2.4, 3.2, 3)]:
        mid.add_patch(plt.Circle((bx, by), 1.0, fc="#fbe7df", ec=ORANGE, lw=0.6))
        mid.plot(bx, by, "o", color=ORANGE, ms=5)
        mid.text(bx, by + 0.45, f"bird {n}", ha="center", fontsize=8)
    mid.text(
        2.5,
        1.2,
        "one animal per site: place predicts identity",
        ha="center",
        fontsize=8,
        style="italic",
        color=ORANGE,
    )
    mid.plot([5.0, 5.0], [1.0, 8.8], color=LIGHT, lw=0.8)
    mid.text(
        7.5, 8.4, "Shared-location design", ha="center", fontsize=9.5, color=BLUE, weight="bold"
    )
    mid.add_patch(plt.Circle((7.5, 4.9), 2.0, fc="#dfeaf8", ec=BLUE, lw=0.6))
    for px, py in [(6.8, 5.6), (8.1, 5.9), (7.0, 4.2), (8.2, 4.4), (7.6, 3.6)]:
        mid.plot(px, py, "o", color=BLUE, ms=5)
    mid.text(
        7.5,
        1.2,
        "shared microphone: spatial and social cues can remain",
        ha="center",
        fontsize=8,
        style="italic",
        color=BLUE,
    )

    low = fig.add_axes([0.0, 0.0, 1.0, 0.31])
    low.set_axis_off()
    low.set_xlim(0, 10)
    low.set_ylim(0, 10)
    low.text(0.0, 9.4, "(c)  The enrol-and-score experiment", fontsize=10, weight="bold")
    steps = [
        (0.2, "Enrolment clips\nfrom sessions A", "#f3f3f3"),
        (2.7, "Frozen representation\nand one fixed head\n(kernel ridge)", "#f3f3f3"),
        (5.2, "Query clips from\ndifferent sessions B\nare named", "#f3f3f3"),
        (7.7, "Accuracy, against the\npermutation null and\nthe context controls", "#f3f3f3"),
    ]
    for x, text, fc in steps:
        low.add_patch(
            FancyBboxPatch((x, 2.2), 2.0, 5.0, boxstyle="round,pad=0.1", fc=fc, ec=LIGHT, lw=0.8)
        )
        low.text(x + 1.0, 4.7, text, ha="center", va="center", fontsize=8)
    for x in (2.25, 4.75, 7.25):
        low.annotate(
            "", xy=(x + 0.4, 4.7), xytext=(x, 4.7), arrowprops={"arrowstyle": "-|>", "color": GREY}
        )
    low.text(
        5.0,
        0.8,
        "A random split puts clips of the same session on both sides; a session split does not.",
        ha="center",
        fontsize=8,
        style="italic",
    )
    fig.savefig(out)
    plt.close(fig)


# --- the four-way gate ---------------------------------------------------------

BACKGROUND_ENDPOINTS: list[tuple[str, str, str]] = [
    ("chiffchaff-withinyear", "chiffchaff-withinyear", "Chiffchaff, within year"),
    ("littleowl-acrossyear", "littleowl", "Little owl, across year"),
    ("pipit-withinyear", "pipit-withinyear", "Tree pipit, within year"),
    ("pipit-acrossyear", "pipit-acrossyear", "Tree pipit, across year"),
    ("chiffchaff-acrossyear", "chiffchaff-acrossyear", "Chiffchaff, across year"),
]
PAIRINGS = [
    ("foreground_to_foreground", "calls on calls"),
    ("background_to_background", "backgrounds on backgrounds"),
    ("foreground_to_background", "calls scored on backgrounds"),
    ("background_to_foreground", "backgrounds scored on calls"),
]


def gate(numbers: Numbers, gate_root: Path) -> list[dict[str, Any]]:
    rows = []
    for endpoint, _, label in BACKGROUND_ENDPOINTS:
        path = gate_root / f"{endpoint}.json"
        data = json.loads(path.read_bytes())
        numbers.sources[f"gate/{endpoint}.json"] = hashlib.sha256(path.read_bytes()).hexdigest()
        row: dict[str, Any] = {
            "endpoint": endpoint,
            "label": label,
            "individuals": data["identities"],
            "chance": r3(data["chance_accuracy"]),
        }
        for key, _ in PAIRINGS:
            cell = data["pairings"][key]
            row[key] = {
                "accuracy": r3(cell["accuracy"]),
                "interval": [r3(v) for v in cell["identity_block_bootstrap_accuracy_95"]],
                "p": cell["permutation_p"],
                "roc_auc": r3(cell["roc_auc"]),
            }
        # One interval per figure: where the headline run measured the same
        # pairing, its figure, interval and p replace the gate run's, so a
        # number that appears in two tables carries one interval.
        for key, stem in (
            ("foreground_to_foreground", f"v5-identity-{endpoint}.json"),
            ("background_to_background", f"v5-identity-{endpoint}-background.json"),
        ):
            if (numbers.root / stem).exists():
                evaluation = numbers.load(stem)["evaluation"]
                row[key] = {
                    "accuracy": r3(evaluation["classification"]["accuracy"]),
                    "interval": [r3(v) for v in evaluation["identity_block_bootstrap_accuracy_95"]],
                    "p": evaluation["permutation_control"]["p_value_plus_one"],
                    "roc_auc": r3(evaluation["verification"]["roc_auc"]),
                    "source": stem,
                }
        numbers.put(f"gate.{endpoint}", row, f"gate/{endpoint}.json")
        rows.append(row)
    return rows


def figure_gate(rows: list[dict[str, Any]], out: Path) -> None:
    fig, axes = plt.subplots(1, len(rows), figsize=(7.4, 2.8), sharey=True)
    fig.subplots_adjust(wspace=0.35)
    for ax, row in zip(axes, rows, strict=True):
        for x, (key, _) in enumerate(PAIRINGS):
            cell = row[key]
            cleared = cell["p"] <= 0.05
            colour = BLUE if x == 0 else ORANGE
            ax.plot([x, x], cell["interval"], color=colour, lw=1.4)
            ax.plot(
                x,
                cell["accuracy"],
                "o",
                ms=5.5,
                color=colour if cleared else "white",
                markeredgecolor=colour,
                markeredgewidth=1.4,
                zorder=3,
            )
        ax.axhline(row["chance"], color=GREY, lw=0.8, ls=(0, (3, 3)))
        species, span = row["label"].split(", ")
        ax.set_title(f"{species}\n{span}\n{row['individuals']} birds", fontsize=7.5)
        ax.set_xticks(range(4))
        ax.set_xticklabels(["C\u2192C", "B\u2192B", "C\u2192B", "B\u2192C"], fontsize=7.5)
        ax.set_ylim(0, 1)
    axes[0].set_ylabel("identification accuracy")
    fig.savefig(out)
    plt.close(fig)


# --- added background -------------------------------------------------------------


def added_background(numbers: Numbers) -> list[dict[str, Any]]:
    rows = []
    for endpoint, stem, label in BACKGROUND_ENDPOINTS:
        name = f"background-{stem}.json"
        data = numbers.load(name)
        arms = data["representations"]["birdnet-v2.4"]["arms"]
        pairs = data["representations"]["birdnet-v2.4"]["own_vs_other"]
        levels = []
        for level, text in [("-10db", "+10 dB"), ("0db", "0 dB"), ("10db", "-10 dB")]:
            levels.append(
                {
                    "added_background_relative_to_original": text,
                    "own": r3(arms[f"own_{level}"]["classification"]["accuracy"]),
                    "other": r3(arms[f"other_{level}"]["classification"]["accuracy"]),
                    "changed": r3(pairs[level]["answer_changed_fraction"]),
                    "difference": r3(pairs[level]["own_minus_other_macro_recall"]),
                    "interval": [r3(v) for v in pairs[level]["identity_block_difference_95"]],
                }
            )
        row = {
            "endpoint": endpoint,
            "label": label,
            "original": r3(arms["original"]["classification"]["accuracy"]),
            "sham": r3(arms["sham_peak_scaled"]["classification"]["accuracy"]),
            "levels": levels,
        }
        numbers.put(f"added_background.{endpoint}", row, name)
        rows.append(row)
    return rows


def figure_added_background(rows: list[dict[str, Any]], out: Path) -> None:
    fig, ax = plt.subplots(figsize=(6.8, 2.8))
    offsets = [-0.2, 0.0, 0.2]
    shades = [ORANGE, "#f29a73", "#f7c7b1"]
    for x, row in enumerate(rows):
        for offset, shade, level in zip(offsets, shades, row["levels"], strict=True):
            low, high = level["interval"]
            excludes = low > 0 or high < 0
            ax.plot([x + offset] * 2, [low, high], color=shade, lw=1.6)
            ax.plot(
                x + offset,
                level["difference"],
                "o",
                ms=5,
                color=shade if excludes else "white",
                markeredgecolor=shade,
                markeredgewidth=1.4,
                zorder=3,
            )
    ax.axhline(0, color=GREY, lw=0.8)
    ax.set_xticks(range(len(rows)))
    ax.set_xticklabels([row["label"].replace(", ", "\n") for row in rows], fontsize=7.5)
    ax.set_ylabel("own-donor minus other-donor\nmacro recall")
    handles = [
        plt.Line2D([], [], color=shade, marker="o", lw=1.6, label=f"added background {text}")
        for shade, text in zip(
            shades, ["10 dB louder", "at equal level", "10 dB quieter"], strict=True
        )
    ]
    ax.legend(handles=handles, frameon=False, fontsize=7.5, loc="upper right")
    fig.savefig(out)
    plt.close(fig)


# --- the augmentation remedy and session compensation --------------------------------


def remedy(numbers: Numbers) -> dict[str, Any]:
    result = {}
    for stem in ("chiffchaff-withinyear", "littleowl-acrossyear"):
        name = f"stowell-remedy-{stem}.json"
        data = numbers.load(name)
        arms = {}
        for arm in ("untouched", "stratified-background-augmentation"):
            cells = data["arms"][arm]["pairings_by_head"]["kernel ridge to one-hot"]
            arms[arm] = {
                key: {
                    "accuracy": r3(cell["accuracy"]),
                    "interval": [r3(v) for v in cell["identity_block_bootstrap_accuracy_95"]],
                    "p": cell["permutation_p"],
                }
                for key, cell in cells.items()
            }
        result[stem] = arms
        numbers.put(f"remedy.{stem}", arms, name)
    return result


def compensation(numbers: Numbers, bat_path: Path, owl_name: str) -> dict[str, Any]:
    from scipy.stats import spearmanr

    raw = bat_path.read_bytes()
    numbers.sources[bat_path.name + " (reid01b)"] = hashlib.sha256(raw).hexdigest()
    bat = json.loads(raw)
    settings = []
    for treatment, rows in bat["treatments"].items():
        for row in rows:
            if row["head"].startswith("kernel"):
                settings.append(
                    {
                        "treatment": treatment,
                        "parameter": row["parameter"],
                        "identity": r3(row["accuracy"]),
                        "day": r3(row["recording_day_probe"]["accuracy"]),
                        "day_floor": r3(row["recording_day_probe"]["majority_class_rate"]),
                    }
                )
    rho_all = spearmanr([s["identity"] for s in settings], [s["day"] for s in settings])
    kept = [s for s in settings if not s["treatment"].startswith("per-session")]
    rho_kept = spearmanr([s["identity"] for s in kept], [s["day"] for s in kept])
    owl = numbers.load(owl_name)
    owl_rows = [
        {
            "treatment": treatment,
            "head": row["head"],
            "parameter": row["parameter"],
            "accuracy": r3(row["accuracy"]),
            "interval": [r3(v) for v in row["identity_block_bootstrap_accuracy_95"]],
        }
        for treatment, rows in owl["treatments"].items()
        for row in rows
    ]
    result = {
        "bat_settings": settings,
        "spearman_all": [r3(rho_all.statistic), r3(rho_all.pvalue)],
        "spearman_without_centring": [r3(rho_kept.statistic), r3(rho_kept.pvalue)],
        "settings": len(settings),
        "owl": owl_rows,
    }
    numbers.put("compensation", result, f"{bat_path.name}, {owl_name}")
    return result


# --- data division -------------------------------------------------------------------


def published_split(numbers: Numbers) -> list[dict[str, Any]]:
    rows = []
    for species in ("littleowl", "chiffchaff", "pipit"):
        for model in ("birdnet", "google-perch"):
            name = f"split-difference-{species}-{model}.json"
            data = numbers.load(name)
            for head, cell in data["heads"].items():
                rows.append(
                    {
                        "species": species,
                        "vectors": "BirdNET" if model == "birdnet" else "Perch",
                        "head": head,
                        "random_fold": r3(cell["random_fold_accuracy"]),
                        "recordists": r3(cell["recordists_accuracy"]),
                        "fall": r3(cell["fall"]),
                        "fall_interval": [r3(v) for v in cell["fall_95"]],
                        "clip_matched": species != "pipit",
                    }
                )
    numbers.put("published_split", rows, "split-difference-*.json")
    return rows


def rook_split(numbers: Numbers) -> dict[str, Any] | None:
    import numpy as np

    try:
        session = numbers.load("v5-identity-rook-full-across-year.json")
        fold = numbers.load("v5-identity-rook-full-across-year-random-fold.json")
    except FileNotFoundError:
        return None

    def by_animal(data: dict[str, Any]) -> dict[str, np.ndarray]:
        flags = np.frombuffer(data["query_correct"].encode(), dtype=np.uint8) == ord("1")
        labels = np.asarray(data["query_identities"])
        return {a: flags[labels == a] for a in sorted(set(labels))}

    a, b = by_animal(fold), by_animal(session)
    animals = sorted(set(a) & set(b))
    rng = np.random.default_rng(17)
    falls = []
    for _ in range(10000):
        draw = rng.integers(0, len(animals), len(animals))
        fa = np.concatenate([a[animals[i]] for i in draw]).mean()
        fb = np.concatenate([b[animals[i]] for i in draw]).mean()
        falls.append(fa - fb)
    result = {
        "random_fold": r3(fold["evaluation"]["classification"]["accuracy"]),
        "session": r3(session["evaluation"]["classification"]["accuracy"]),
        "fall": r3(
            fold["evaluation"]["classification"]["accuracy"]
            - session["evaluation"]["classification"]["accuracy"]
        ),
        "fall_interval": [r3(v) for v in np.quantile(falls, [0.025, 0.975])],
        "moved": fold["selection"]["random_fold"],
        "animals": len(animals),
    }
    numbers.put("rook_split", result, "v5-identity-rook-full-across-year*.json")
    return result


def figure_split(rows: list[dict[str, Any]], rook: dict[str, Any] | None, out: Path) -> None:
    fig, ax = plt.subplots(figsize=(6.8, 3.0))
    labels = []
    items = [r for r in rows if r["clip_matched"]] + [r for r in rows if not r["clip_matched"]]
    for x, row in enumerate(items):
        colour = BLUE if row["clip_matched"] else LIGHT
        ax.plot([x, x], row["fall_interval"], color=colour, lw=1.6)
        ax.plot(x, row["fall"], "o", ms=5, color=colour, zorder=3)
        head = {
            "kernel ridge to one-hot": "kernel ridge",
            "mean cosine similarity to each identity": "class mean",
            "single nearest enrollment clip, cosine": "nearest clip",
        }[row["head"]]
        species = {"littleowl": "Little owl", "chiffchaff": "Chiffchaff", "pipit": "Tree pipit"}[
            row["species"]
        ]
        labels.append(f"{species}, {row['vectors']}, {head}")
    if rook:
        x = len(items)
        ax.plot([x, x], rook["fall_interval"], color=ORANGE, lw=1.6)
        ax.plot(x, rook["fall"], "o", ms=6, color=ORANGE, zorder=3)
        labels.append("Rook, BirdNET, kernel ridge")
    ax.axhline(0, color=GREY, lw=0.8)
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=70, ha="right", fontsize=6.5)
    ax.set_ylabel("random split minus session split\n(re-ID accuracy)")
    fig.savefig(out)
    plt.close(fig)


# --- open set ------------------------------------------------------------------------


def open_set(numbers: Numbers) -> dict[str, Any]:
    result = {}
    for endpoint in ("great-tit", "littleowl-acrossyear", "cockatoo-fold1", "rookid-full-width"):
        name = f"v5-open-set-{endpoint}.json"
        try:
            data = numbers.load(name)
        except FileNotFoundError:
            continue
        reps = {}
        for rep, entry in data["representations"].items():
            s = entry["allocation_sensitivity"]
            reps[rep] = {key: {k: r3(v) for k, v in value.items()} for key, value in s.items()}
        result[endpoint] = {
            "identities": data["identities"],
            "scheme": data.get("allocation_scheme"),
            "representations": reps,
        }
    numbers.put("open_set", result, "v5-open-set-*.json")
    return result


def figure_open_set(result: dict[str, Any], out: Path) -> None:
    labels = {
        "great-tit": "Great tit (16)",
        "littleowl-acrossyear": "Little owl (16)",
        "cockatoo-fold1": "Cockatoo (16), random fold",
        "rookid-full-width": "Rook (11), across year",
    }
    fig, ax = plt.subplots(figsize=(6.4, 0.5 * len(result) + 1.2))
    for y, (_endpoint, entry) in enumerate(reversed(list(result.items()))):
        s = entry["representations"]["birdnet-v2.4"]
        ax.plot(
            [s["best_acceptance_at_0.10"]["minimum"], s["best_acceptance_at_0.10"]["maximum"]],
            [y + 0.12] * 2,
            color=BLUE,
            lw=1.2,
            alpha=0.5,
        )
        ax.plot(s["best_acceptance_at_0.10"]["median"], y + 0.12, "o", color=BLUE, ms=6)
        ax.plot(s["closed_set_accuracy"]["median"], y - 0.12, "D", color=ORANGE, ms=6)
        ax.plot(
            [s["closed_set_accuracy"]["minimum"], s["closed_set_accuracy"]["maximum"]],
            [y - 0.12] * 2,
            color=ORANGE,
            lw=1.2,
            alpha=0.5,
        )
    ax.set_yticks(range(len(result)))
    ax.set_yticklabels([labels[e] for e in reversed(list(result))])
    ax.set_xlim(0, 1)
    ax.set_xlabel("proportion of enrolled calls (median and range over 16 allocations)")
    handles = [
        plt.Line2D(
            [], [], color=ORANGE, marker="D", lw=1.2, label="names the right enrolled individual"
        ),
        plt.Line2D(
            [],
            [],
            color=BLUE,
            marker="o",
            lw=1.2,
            label="accepted and named right, strangers held to 10%",
        ),
    ]
    ax.legend(
        handles=handles,
        frameon=False,
        fontsize=7.5,
        loc="upper center",
        bbox_to_anchor=(0.45, -0.25),
        ncol=1,
    )
    fig.savefig(out)
    plt.close(fig)


# --- encoders --------------------------------------------------------------------------


def encoders(numbers: Numbers) -> dict[str, Any] | None:
    try:
        data = numbers.load("v5-encoder-comparison.json")
    except FileNotFoundError:
        return None
    numbers.put("encoders", data, "v5-encoder-comparison.json")
    return data


#: The encoder sweep ran on the five-bird rook in place of the rook at full width.
SWEEP_LABEL = {name: label for name, label, _ in ENDPOINTS}
SWEEP_LABEL["rookid-ch2"] = "Rook, five birds, one call type"
SWEEP_ORDER = ["rookid-ch2" if name.startswith("rook-full") else name for name, _, _ in ENDPOINTS]


def sweep_label(endpoint: str) -> str:
    return SWEEP_LABEL.get(endpoint, endpoint)


def figure_encoders(data: dict[str, Any], out: Path) -> None:
    per = data["per_endpoint"]
    names = [n for n in SWEEP_ORDER if n in per] + sorted(set(per) - set(SWEEP_ORDER))
    fig, ax = plt.subplots(figsize=(6.8, 0.52 * len(names) + 1.3))
    for y, endpoint in enumerate(reversed(names)):
        paired = per[endpoint]["paired"]
        if not paired:
            continue
        cells = [
            (m, c) for m, c in sorted(paired["comparisons"].items()) if not m.startswith("clip-")
        ]
        for i, (_, cell) in enumerate(cells):
            offset = -0.36 + 0.72 * i / max(len(cells) - 1, 1)
            if cell["differs_at_0.05_after_holm"]:
                colour = BLUE if cell["difference_from_reference"] > 0 else ORANGE
            else:
                colour = LIGHT
            low, high = cell["difference_95"]
            ax.plot([low, high], [y + offset] * 2, color=colour, lw=0.7, alpha=0.9)
            ax.plot(cell["difference_from_reference"], y + offset, "o", ms=2.6, color=colour)
    ax.axvline(0, color=INK, lw=0.8)
    for y in range(len(names) - 1):
        ax.axhline(y + 0.5, color="#e6e6e6", lw=0.6, zorder=0)
    ax.set_ylim(-0.6, len(names) - 0.4)
    ax.set_yticks(range(len(names)))
    ax.set_yticklabels(
        [f"{sweep_label(n)} ({per[n]['paired']['animals']})" for n in reversed(names)],
        fontsize=7.5,
    )
    ax.set_xlabel("encoder accuracy minus BirdNET v2.4, with paired 95% interval over individuals")
    handles = [
        plt.Line2D(
            [], [], color=BLUE, marker="o", lw=0.7, label="above BirdNET after Holm correction"
        ),
        plt.Line2D(
            [], [], color=ORANGE, marker="o", lw=0.7, label="below BirdNET after Holm correction"
        ),
        plt.Line2D(
            [], [], color=LIGHT, marker="o", lw=0.7, label="not different after Holm correction"
        ),
    ]
    ax.legend(
        handles=handles,
        frameon=False,
        fontsize=7.5,
        loc="upper center",
        bbox_to_anchor=(0.45, -0.12),
        ncol=3,
    )
    fig.savefig(out)
    plt.close(fig)


# --- supporting-information tables, written as Markdown -----------------------------


def ranged(stat: dict[str, float]) -> str:
    return f"{stat['median']:.3f} ({stat['minimum']:.3f}-{stat['maximum']:.3f})"


def pvalue(p: float | None) -> str | None:
    """Four decimals, so that 9,999 shuffles keep their smallest attainable p of 0.0001."""
    return None if p is None else f"{p:.4f}"


def pairing(cell: dict[str, Any]) -> str:
    low, high = cell["interval"]
    return f"{cell['accuracy']:.3f} ({low:.3f} to {high:.3f}), p {pvalue(cell['p'])}"


def md_table(header: list[str], rows: list[list[Any]]) -> str:
    def cell(value: Any) -> str:
        if value is None:
            return "not constructible"
        if isinstance(value, float):
            return f"{value:.3f}"
        if (
            isinstance(value, list)
            and len(value) == 2
            and all(isinstance(v, (int, float)) for v in value)
        ):
            return f"{value[0]:.3f} to {value[1]:.3f}"
        return str(value)

    lines = ["| " + " | ".join(header) + " |", "|" + "|".join("---" for _ in header) + "|"]
    lines += ["| " + " | ".join(cell(v) for v in row) + " |" for row in rows]
    return "\n".join(lines)


AXIS_ORDER = (
    "source_recording",
    "recording",
    "session",
    "date",
    "recording_day",
    "recorded",
    "stem",
    "channel",
    "treatment",
    "nestbox",
    "cohort",
    "year",
    "call_type",
    "span",
)


def table_confound(numbers: Numbers) -> str:
    data = numbers.load("v5-confound-index.json")
    rows = []
    for endpoint, entry in sorted(data["endpoints"].items()):
        for axis, cells in sorted(entry["axes"].items()):
            if axis in ("split", "condition"):
                continue
            whole = cells["all"]
            rows.append(
                [
                    endpoint,
                    axis,
                    whole["levels"],
                    whole["nmi"],
                    whole["cramers_v"],
                    f"{whole['levels_with_two_or_more_identities']} of {whole['levels']}",
                    f"{whole['identities_on_two_or_more_levels']} of {whole['identities']}",
                ]
            )
    numbers.put("confound_index", data["endpoints"], "v5-confound-index.json")
    return md_table(
        [
            "Endpoint",
            "Axis",
            "Levels",
            "NMI with identity",
            "Cramér's V",
            "Levels holding 2+ individuals",
            "Individuals on 2+ levels",
        ],
        rows,
    )


def table_place(numbers: Numbers) -> str:
    data = numbers.load("v5-place-control-great-tit.json")
    rows = [
        [
            f"GT-{index:02d}",
            "yes" if r["shares_a_nestbox"] else "no",
            "yes" if r["shares_a_source_recording"] else "no",
            "yes" if r["years_differ"] else "no",
            r["metres_between_nestboxes"],
        ]
        for index, r in enumerate(data["per_identity"], 1)
    ]
    numbers.put(
        "place_control",
        {k: v for k, v in data.items() if k != "per_identity"},
        "v5-place-control-great-tit.json",
    )
    return md_table(
        [
            "Bird",
            "Same nest box both sides",
            "Same recording both sides",
            "Years differ",
            "Metres between nest boxes",
        ],
        rows,
    )


def table_headline_full(rows: list[dict[str, Any]]) -> str:
    return md_table(
        [
            "Endpoint",
            "Individuals",
            "Enrolment / query clips",
            "Accuracy",
            "95% interval, individuals",
            "95% interval, three-stage",
            "Uniform chance",
            "Majority class",
            "Permutation 95th percentile",
            "Permutation p",
            "Within-recording p",
            "Within-recording 95th percentile",
            "ROC area",
            "Trials: animal minus recording",
            "Standardised",
            "Pairwise EER",
        ],
        [
            [
                r["label"],
                r["individuals"],
                f"{r['enrolment_clips']} / {r['query_clips']}",
                r["accuracy"],
                r["interval"],
                r["three_stage"],
                r["uniform_chance"],
                r["majority"],
                r["permutation_95"],
                pvalue(r["permutation_p"]),
                pvalue(r["within_p"]),
                r["within_95"],
                r["roc_auc"],
                r["trials_difference"],
                r["trials_standardised"],
                r["trials_eer"],
            ]
            for r in rows
        ],
    )


def table_rook(numbers: Numbers) -> str:
    variants = [
        ("across-year", "Enrol 2020, score 2021, channel 0"),
        ("across-year-capped-90", "The same, enrolment capped at 90 clips per bird"),
        ("across-year-random-fold", "The same clips, divided at random within each bird"),
        ("across-day-2020", "Enrol and score on different 2020 days, channel 0"),
        (
            "across-day-2020-ch0-to-ch1",
            "The same calls, enrolled on channel 0 and scored on channel 1",
        ),
    ]
    rows = []
    for stem, label in variants:
        name = f"v5-identity-rook-full-{stem}.json"
        try:
            data = numbers.load(name)
        except FileNotFoundError:
            continue
        e = data["evaluation"]
        w = e.get("within_session_permutation") or {}
        a = (data.get("stratified_trials") or {}).get("animal_against_recording") or {}
        row = {
            "label": label,
            "individuals": data["identities"],
            "clips": f"{e['enrollment_calls']} / {e['query_calls']}",
            "accuracy": r3(e["classification"]["accuracy"]),
            "interval": [r3(v) for v in e["identity_block_bootstrap_accuracy_95"]],
            "three_stage": [r3(v) for v in e["hierarchical_bootstrap"]["accuracy_95"]]
            if e.get("hierarchical_bootstrap")
            else None,
            "uniform_chance": r3(data["uniform_chance"]),
            "permutation_95": r3(e["permutation_upper_tail"]["percentile_95"]),
            "permutation_p": e["permutation_control"]["p_value_plus_one"],
            "within_p": w.get("p_value_plus_one"),
            "trials_difference": a.get("difference"),
            "trials_standardised": a.get("standardised_difference"),
            "trials_eer": (a.get("equal_error_rate") or {}).get("equal_error_rate"),
        }
        numbers.put(f"rook.{stem}", row, name)
        rows.append([pvalue(row[k]) if k in ("permutation_p", "within_p") else row[k] for k in row])
    return md_table(
        [
            "Division",
            "Birds",
            "Enrolment / query clips",
            "Accuracy",
            "95% interval, individuals",
            "95% interval, three-stage",
            "Uniform chance",
            "Permutation 95th percentile",
            "Permutation p",
            "Within-recording p",
            "Trials: animal minus recording",
            "Standardised",
            "Pairwise EER",
        ],
        rows,
    )


def table_frozen(numbers: Numbers) -> str:
    endpoints = [
        "rookid",
        "zebra",
        "great-tit",
        "littleowl",
        "chiffchaff-withinyear",
        "chiffchaff-acrossyear",
        "pipit-withinyear",
        "pipit-acrossyear",
        "birdpark-juv01",
        "birdpark-juv03",
    ]
    rows = []
    for endpoint in endpoints:
        name = f"frozen-{endpoint}.json"
        data = numbers.load(name)
        head = "kernel ridge to one-hot"
        accuracies = {
            key: cand["heads"][head]["foreground_to_foreground"]["classification"]["accuracy"]
            for key, cand in data["candidates"].items()
        }
        selected = data["selected"]
        best = max(accuracies, key=accuracies.get)
        rows.append(
            [
                endpoint,
                selected,
                r3(accuracies[selected]),
                pvalue(
                    data["selected_p_bonferroni_over_candidates"][head]["foreground_to_foreground"]
                ),
                r3(accuracies["embedding.mean"]),
                f"{best} ({accuracies[best]:.3f})",
            ]
        )
    numbers.put("frozen_layers", rows, "frozen-*.json")
    return md_table(
        [
            "Endpoint",
            "Enrolment-selected candidate",
            "Its accuracy",
            "Its adjusted p",
            "Final embedding",
            "Best candidate chosen with query labels",
        ],
        rows,
    )


def table_open_set_full(result: dict[str, Any]) -> str:
    rows = []
    for endpoint, entry in result.items():
        for rep, s in entry["representations"].items():
            rows.append(
                [
                    endpoint,
                    rep,
                    ranged(s["closed_set_accuracy"]),
                    ranged(s["known_vs_unknown_auroc"]),
                    f"{s['equal_error_rate']['median']:.3f}",
                ]
                + [
                    f"{s[f'best_acceptance_at_{b}']['median']:.3f}"
                    for b in ("0.01", "0.05", "0.10", "0.20", "0.50")
                ]
                + [
                    f"{s[f'true_accept_at_false_accept_{b}']['median']:.3f}"
                    for b in ("0.01", "0.10")
                ]
                + [f"{s['acceptance_calibrated_at_0.10']['median']:.3f}"]
            )
    return md_table(
        [
            "Endpoint",
            "Representation",
            "Names right (range)",
            "Separation (range)",
            "EER",
            "Accept and name, 1%",
            "5%",
            "10%",
            "20%",
            "50%",
            "True accept at 1% false accept",
            "at 10%",
            "Calibrated on other birds, 10%",
        ],
        rows,
    )


def table_learned(numbers: Numbers) -> str:
    rows = []
    for endpoint in [
        "rookid",
        "zebra-finch",
        "great-tit",
        "littleowl-acrossyear",
        "chiffchaff-withinyear",
        "chiffchaff-acrossyear",
        "pipit-withinyear",
        "pipit-acrossyear",
    ]:
        name = f"learned-combination-{endpoint}.json"
        data = numbers.load(name)
        for arm, heads in data["arms"].items():
            rows.append(
                [endpoint, arm]
                + [
                    r3(heads[h]["foreground_to_foreground"]["classification"]["accuracy"])
                    for h in heads
                ]
            )
    numbers.put("learned_combination", rows, "learned-combination-*.json")
    return md_table(["Endpoint", "Arm", "Kernel ridge", "Mean cosine", "Nearest clip"], rows)


def table_sequence(numbers: Numbers) -> str:
    rows = []
    for endpoint in [
        "rookid",
        "zebra-finch",
        "great-tit",
        "littleowl-acrossyear",
        "chiffchaff-withinyear",
        "bat-acrosstreatment",
    ]:
        name = f"sequence-metric-{endpoint}.json"
        data = numbers.load(name)
        for layer, arms in data["arms"].items():
            cells = [
                arms[a]["closed_set"]["accuracy"]
                if arms.get(a) and arms[a].get("closed_set")
                else None
                for a in ("layer average", "sequence average", "learned sequence")
            ]
            identities = next(
                (arms[a]["closed_set"]["identities"] for a in arms if arms[a].get("closed_set")),
                None,
            )
            rows.append([endpoint, layer, identities] + [r3(c) for c in cells])
    numbers.put("sequence_metric", rows, "sequence-metric-*.json")
    return md_table(
        [
            "Endpoint",
            "Layer",
            "Evaluation individuals",
            "Layer average",
            "Sequence average",
            "Learned sequence",
        ],
        rows,
    )


def table_whale(numbers: Numbers) -> str:
    rows = []
    for shift in range(0, 11000, 1000):
        name = f"gate-right-whale-shift-{shift}.json"
        data = numbers.load(name)
        cells = data["pairings"]
        rows.append(
            [shift, r3(data["foreground_enrolment_fisher_ratio"])]
            + [f"{cells[k]['accuracy']:.3f} / {cells[k]['permutation_p']}" for k, _ in PAIRINGS]
        )
    numbers.put("right_whale", rows, "gate-right-whale-shift-*.json")
    return md_table(
        ["Frequency shift (Hz)", "Enrolment Fisher ratio"]
        + [label + " (accuracy / p)" for _, label in PAIRINGS],
        rows,
    )


def rate_and_layer(selected: str) -> str:
    """'wav2vec2-base-x2-l01' -> 'x2, layer 1'; 'birdnet-v2.4-x1' -> 'x1'."""
    match = re.search(r"-x(\d)(?:-l(\d+))?$", selected)
    if not match:
        return selected
    rate, layer = match.groups()
    return f"x{rate}" if layer is None else f"x{rate}, layer {int(layer)}"


def table_encoders(data: dict[str, Any]) -> str:
    per = data["per_endpoint"]
    names = [n for n in SWEEP_ORDER if n in per] + sorted(set(per) - set(SWEEP_ORDER))
    rows = []
    for endpoint in names:
        entry = per[endpoint]
        paired = entry["paired"]["comparisons"] if entry.get("paired") else {}
        for model, sel in sorted(
            entry["selection"].items(), key=lambda item: -item[1]["selected_accuracy"]
        ):
            cmp = paired.get(model, {})
            final = sel["final_layer_at_selected_rate"]
            rows.append(
                [
                    sweep_label(endpoint),
                    model,
                    rate_and_layer(sel["selected"]),
                    r3(sel["selected_accuracy"]),
                    "one output" if final is None else r3(final),
                    r3(sel["best_layer_chosen_with_query_labels"]),
                    "reference" if not cmp else r3(cmp.get("difference_from_reference")),
                    cmp.get("difference_95") and [r3(v) for v in cmp["difference_95"]] or "-",
                    "-" if not cmp else r3(cmp.get("p_holm")),
                ]
            )
    return md_table(
        [
            "Endpoint",
            "Representation",
            "Selected rate, layer",
            "Selected accuracy",
            "Final layer",
            "Best layer (query labels)",
            "Difference from BirdNET",
            "95% interval",
            "Holm p",
        ],
        rows,
    )


def table_not_computed(numbers: Numbers) -> str:
    rows = []
    for endpoint in SWEEP_ORDER:
        name = f"v5-sweep-{'rookid' if endpoint == 'rookid-ch2' else endpoint}.json"
        try:
            sweep = numbers.load(name)
        except FileNotFoundError:
            continue
        missing = sweep.get("not_computed") or []
        numbers.put(f"not_computed/{endpoint}", missing, name)
        for item in missing:
            reason = item["reason"]
            if "did not fit" in reason:
                why = "did not fit in graphics memory on a long clip"
            else:
                why = "clip shorter than the encoder's first convolution or padding"
            rows.append(
                [sweep_label(endpoint), item["model"].split("/")[-1], f"x{item['slowdown']}", why]
            )
    return md_table(["Endpoint", "Encoder", "Playback slowdown", "Reason"], rows)


def table_gate(rows: list[dict[str, Any]]) -> str:
    out = []
    for row in rows:
        cells = [row["label"], row["individuals"], row["chance"]]
        for key, _ in PAIRINGS:
            c = row[key]
            cells.append(f"{pairing(c)}, ROC {c['roc_auc']:.3f}")
        out.append(cells)
    return md_table(
        ["Endpoint", "Individuals", "Uniform chance"] + [label for _, label in PAIRINGS], out
    )


def table_added(rows: list[dict[str, Any]]) -> str:
    out = []
    for row in rows:
        for level in row["levels"]:
            out.append(
                [
                    row["label"],
                    row["original"],
                    row["sham"],
                    level["added_background_relative_to_original"],
                    level["own"],
                    level["other"],
                    level["changed"],
                    level["difference"],
                    level["interval"],
                ]
            )
    return md_table(
        [
            "Endpoint",
            "Original",
            "Volume-only sham",
            "Added background relative to original",
            "Own donor",
            "Other donor",
            "Share of answers changed",
            "Own minus other, macro recall",
            "95% interval, individuals",
        ],
        out,
    )


def table_remedy(result: dict[str, Any]) -> str:
    out = []
    for stem, arms in result.items():
        for arm, cells in arms.items():
            out.append([stem, arm] + [pairing(cells[key]) for key, _ in PAIRINGS])
    return md_table(["Endpoint", "Enrolment"] + [label for _, label in PAIRINGS], out)


def table_split(rows: list[dict[str, Any]], rook: dict[str, Any] | None) -> str:
    out = [
        [
            r["species"],
            r["vectors"],
            r["head"],
            "yes" if r["clip_matched"] else "no",
            r["random_fold"],
            r["recordists"],
            r["fall"],
            r["fall_interval"],
        ]
        for r in rows
    ]
    if rook:
        out.append(
            [
                "rook, full width",
                "BirdNET (this work)",
                "kernel ridge to one-hot",
                "yes",
                rook["random_fold"],
                rook["session"],
                rook["fall"],
                rook["fall_interval"],
            ]
        )
    return md_table(
        [
            "Species",
            "Vectors",
            "Head",
            "Clip-matched",
            "Random fold",
            "Session division",
            "Fall",
            "95% interval of fall, individuals",
        ],
        out,
    )


def table_compensation(result: dict[str, Any]) -> str:
    bat = md_table(
        [
            "Treatment",
            "Setting",
            "Identity accuracy",
            "Recording day read back",
            "Day majority floor",
        ],
        [
            [s["treatment"], s["parameter"], s["identity"], s["day"], s["day_floor"]]
            for s in result["bat_settings"]
        ],
    )
    owl = md_table(
        ["Treatment", "Head", "Setting", "Accuracy", "95% interval, individuals"],
        [
            [r["treatment"], r["head"], r["parameter"], r["accuracy"], r["interval"]]
            for r in result["owl"]
        ],
    )
    every, kept = result["spearman_all"], result["spearman_without_centring"]
    rank = (
        f"Spearman rank correlation, identity against day, all {result['settings']} settings: "
        f"{every[0]:.3f} (p {every[1]:.3f}); without per-session centring: "
        f"{kept[0]:.3f} (p {kept[1]:.3f})."
    )
    return bat + "\n\n" + rank + "\n\n" + owl


def supporting(
    numbers: Numbers,
    rows: list[dict[str, Any]],
    opened: dict[str, Any],
    comparison: dict[str, Any] | None,
    out: Path,
    extra: dict[str, str] | None = None,
) -> None:
    parts = {
        **(extra or {}),
        "confound": table_confound(numbers),
        "place": table_place(numbers),
        "headline": table_headline_full(rows),
        "rook": table_rook(numbers),
        "frozen": table_frozen(numbers),
        "open_set": table_open_set_full(opened),
        "learned": table_learned(numbers),
        "sequence": table_sequence(numbers),
        "whale": table_whale(numbers),
    }
    if comparison:
        parts["encoders"] = table_encoders(comparison)
        parts["not_computed"] = table_not_computed(numbers)
    out.write_text(json.dumps(parts, indent=1) + "\n")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", type=Path, required=True)
    parser.add_argument("--gate", type=Path, required=True, help="the four-way gate results")
    parser.add_argument(
        "--compensation", type=Path, required=True, help="the bat compensation result"
    )
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    figures = args.out / "figures"
    figures.mkdir(parents=True, exist_ok=True)
    numbers = Numbers(args.results)
    figure_concept(figures / "fig1-confound-and-design.png")
    rows = headline(numbers)
    if rows:
        figure_headline(rows, figures / "fig2-closed-set.png")
    gate_rows = gate(numbers, args.gate)
    figure_gate(gate_rows, figures / "fig3-four-way-gate.png")
    added_rows = added_background(numbers)
    figure_added_background(added_rows, figures / "fig4-added-background.png")
    remedy_result = remedy(numbers)
    compensation_result = compensation(
        numbers, args.compensation, "compensation-littleowl-acrossyear-birdnet-v2.4.json"
    )
    split_rows, rook = published_split(numbers), rook_split(numbers)
    figure_split(split_rows, rook, figures / "fig5-data-division.png")
    extra = {
        "gate": table_gate(gate_rows),
        "added_background": table_added(added_rows),
        "remedy": table_remedy(remedy_result),
        "split": table_split(split_rows, rook),
        "compensation": table_compensation(compensation_result),
    }
    comparison = encoders(numbers)
    if comparison:
        figure_encoders(comparison, figures / "fig6-encoders.png")
    opened = open_set(numbers)
    if opened:
        figure_open_set(opened, figures / "fig7-open-set.png")
    supporting(numbers, rows, opened, comparison, args.out / "tables.json", extra)
    numbers.write(args.out / "numbers.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
