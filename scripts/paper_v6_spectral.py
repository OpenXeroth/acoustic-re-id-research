"""Render archived spectral-subtraction arms with their conditional interpretation."""

from __future__ import annotations

import argparse
import math
from pathlib import Path

from paper_figures import BLUE, GREY, ORANGE, Numbers, md_table, plt
from paper_figures import ENDPOINTS as LABELS
from paper_v6_additional import interval
from paper_v6_evidence import BACKGROUND_ENDPOINTS, validate

PAIRINGS = {
    "foreground_gallery_foreground_query": "C→C",
    "background_gallery_background_query": "B→B",
    "foreground_gallery_background_query": "C→B",
    "background_gallery_foreground_query": "B→C",
}
ARMS = [
    ("original", "Original", BLUE),
    ("alpha_1", "Alpha 1", GREY),
    ("alpha_2", "Alpha 2", ORANGE),
]


def render(results: Path, endpoints: list[str], out: Path, *, partial: bool) -> None:
    if len(set(endpoints)) != len(endpoints) or not set(endpoints) <= set(BACKGROUND_ENDPOINTS):
        raise ValueError("unexpected or repeated endpoint")
    if not partial and set(endpoints) != set(BACKGROUND_ENDPOINTS):
        raise ValueError("all five endpoints are required unless --partial is explicit")
    if not endpoints:
        raise ValueError("no completed spectral results")
    endpoints = [name for name, _, _ in LABELS if name in endpoints]
    numbers = Numbers(results)
    measured = {}
    for endpoint in endpoints:
        source = f"v6-counts/v6-spectral-subtraction-{endpoint}.json"
        data = numbers.load(source)
        validate(data, "spectral", endpoint, ())
        if data["config"] != {"alphas": [1.0, 2.0], "beta": 0.05, "frame": 1024, "hop": 256}:
            raise ValueError("unexpected spectral-subtraction recipe")
        measured[endpoint] = {}
        for arm, _, _ in ARMS:
            if set(data[arm]) != set(PAIRINGS):
                raise ValueError("a recording-condition pairing is absent")
            measured[endpoint][arm] = {}
            for pairing, row in data[arm].items():
                if row["permutation"]["permutations"] != 9999:
                    raise ValueError("a pairing has the wrong null count")
                measured[endpoint][arm][pairing] = {
                    "accuracy": row["accuracy"],
                    "interval_95": row["identity_block_bootstrap_accuracy_95"],
                    "label_permutation_p": row["permutation"]["p_value_plus_one"],
                    "roc_auc": row["roc_auc"],
                }
        numbers.put(endpoint, measured[endpoint], source)
    rows = []
    names = {name: label for name, label, _ in LABELS}
    nrows = math.ceil(len(endpoints) / 2)
    fig, axes = plt.subplots(nrows, 2, figsize=(10, nrows * 3.1 + 0.6), squeeze=False)
    for axis, endpoint in zip(axes.flat, endpoints, strict=False):
        for y, (pairing, label) in enumerate(PAIRINGS.items()):
            for offset, (arm, title, colour) in zip((-0.2, 0.0, 0.2), ARMS, strict=True):
                values = measured[endpoint][arm][pairing]
                axis.plot(values["interval_95"], [y + offset] * 2, color=colour, lw=0.9)
                axis.plot(values["accuracy"], y + offset, "o", color=colour, ms=4)
                rows.append(
                    [
                        names[endpoint],
                        label,
                        title,
                        interval(values["accuracy"], values["interval_95"]),
                        f"{values['label_permutation_p']:.4f}",
                        f"{values['roc_auc']:.3f}",
                    ]
                )
        axis.set(
            title=names[endpoint],
            yticks=range(len(PAIRINGS)),
            yticklabels=list(PAIRINGS.values()),
            xlim=(0, 1),
            xlabel="Accuracy (95% individual interval)",
            ylabel="Gallery→query",
        )
        axis.invert_yaxis()
    for axis in list(axes.flat)[len(endpoints) :]:
        axis.set_visible(False)
    fig.legend(
        handles=[plt.Line2D([], [], color=c, marker="o", label=t) for _, t, c in ARMS],
        loc="upper center",
        bbox_to_anchor=(0.5, 0.96),
        ncol=3,
        frameon=False,
    )
    fig.suptitle(f"Spectral subtraction: {len(endpoints)} of 5 registered endpoints", y=0.995)
    fig.tight_layout(rect=(0, 0, 1, 0.90))
    out.mkdir(parents=True, exist_ok=True)
    fig.savefig(out / "spectral-subtraction.png")
    plt.close(fig)
    numbers.write(out / "spectral-subtraction-numbers.json")
    text = f"# Revision 6 spectral subtraction: {len(endpoints)} of 5 endpoints\n\n"
    if partial:
        text += (
            "This overview is explicitly partial and does not establish completion of the "
            "five-endpoint analysis or the paper.\n\n"
        )
    text += (
        "BirdNET v2.4 is evaluated through the fixed kernel-ridge head before treatment and "
        "at the two registered subtraction strengths. C denotes calls; B denotes backgrounds. "
        "Both gallery and query waveforms receive the treatment in each subtraction arm. "
        "Intervals resample individuals 10,000 times within each arm. They are not paired "
        "intervals on a treatment difference. The table's label-permutation p values use "
        "9,999 shuffles and are unadjusted; they do not test whether subtraction helps.\n\n"
        "The registered profile groups ambient recordings by individual label and split, "
        "using query-side ambient material for query clips. These groups are not independently "
        "identified recording sessions. The treatment therefore requires the label-associated "
        "recording-group mapping. It is a conditional analysis, not evidence for an "
        "identity-blind online denoising system. Group-specific filtering can preserve or "
        "introduce recording-group cues; the background pairings must accompany interpretation.\n\n"
        "![Spectral-subtraction accuracies by recording condition](spectral-subtraction.png)\n\n"
    )
    for endpoint in endpoints:
        values = measured[endpoint]
        key = "foreground_gallery_foreground_query"
        scores = [f"{values[arm][key]['accuracy']:.3f}" for arm, _, _ in ARMS]
        text += (
            f"- {names[endpoint]}: call→call accuracy {' → '.join(scores)} "
            "(original, alpha 1, alpha 2).\n"
        )
    if "pipit-withinyear" in measured:
        key = "background_gallery_background_query"
        scores = [f"{measured['pipit-withinyear'][a][key]['accuracy']:.3f}" for a, _, _ in ARMS]
        text += (
            "\nOn tree pipit within a year, background→background accuracy is "
            + ", ".join(scores)
            + " in the same arm order. Higher call accuracy therefore does not by itself "
            "demonstrate removal of recording-associated information.\n"
        )
    text += "\n" + md_table(
        ["Endpoint", "Gallery→query", "Arm", "Accuracy (95% interval)", "Label p", "ROC AUC"], rows
    )
    text += (
        "\n\nAll displayed values and archived source digests are in the "
        "[numerical ledger](spectral-subtraction-numbers.json). No source result was altered.\n"
        "\nRebuild from byte-verified result objects with "
        "`uv run --with matplotlib python scripts/paper_v6_spectral.py "
        "--results PRIVATE_RESULTS --out OUT/spectral`, as `scripts/paper_v6_build.sh` does.\n"
    )
    (out / "spectral-subtraction.md").write_text(text)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, required=True)
    parser.add_argument("--endpoint", action="append", choices=BACKGROUND_ENDPOINTS)
    parser.add_argument("--partial", action="store_true")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    render(
        args.results, args.endpoint or list(BACKGROUND_ENDPOINTS), args.out, partial=args.partial
    )


if __name__ == "__main__":
    main()
