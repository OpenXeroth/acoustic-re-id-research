"""Render completed v6 augmentation, sensitivity and expanded great-tit evidence."""

from __future__ import annotations

import argparse
from pathlib import Path

from paper_figures import BLUE, ORANGE, Numbers, md_table, plt
from paper_v6_evidence import aggregate, validate


def interval(value: float, bounds: list[float]) -> str:
    return f"{value:.3f} ({bounds[0]:.3f}–{bounds[1]:.3f})"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    numbers = Numbers(args.results)
    files = {
        "v6-sensitivity-floor.json": ("sensitivity", None),
        "v6-counts/v6-stowell-augmentation-pipit-withinyear.json": (
            "augmentation",
            "pipit-withinyear",
        ),
        "v6-headline-great-tit-full.json": ("headline", "great-tit-full"),
        "v6-place-control-great-tit-full.json": ("place", "great-tit-full"),
    }
    loaded = {}
    for name, (role, endpoint) in files.items():
        data = numbers.load(name)
        validate(data, role, endpoint, ())
        loaded[name] = data
    for row in loaded["v6-sensitivity-floor.json"]["levels"].values():
        if row["evaluation"]["permutation_control"]["permutations"] != 9999:
            raise ValueError("sensitivity endpoint lacks its registered 9,999 shuffles")
    headline_data = loaded["v6-headline-great-tit-full.json"]
    place_data = loaded["v6-place-control-great-tit-full.json"]
    if headline_data["manifest_sha256"] != place_data["manifest_sha256"]:
        raise ValueError("headline and place control describe different samples")
    tables = aggregate(loaded)
    sensitivity = tables["sensitivity"]
    augmentation = tables["augmentation"]["pipit-withinyear"]
    headline = tables["headline"]["great-tit-full"]
    place = tables["place"]["great-tit-full"]
    # Publication-safe, explicit fields only. No raw identity correspondences.
    for name, value, source in [
        ("sensitivity", sensitivity, "v6-sensitivity-floor.json"),
        ("augmentation", augmentation, "v6-counts/v6-stowell-augmentation-pipit-withinyear.json"),
        ("great_tit_headline", headline, "v6-headline-great-tit-full.json"),
        (
            "great_tit_place",
            {k: v for k, v in place.items() if k != "per_individual"},
            "v6-place-control-great-tit-full.json",
        ),
    ]:
        numbers.put(name, value, source)
    levels = ["-10db", "+0db", "+10db"]
    sensitivity_table = md_table(
        ["Call/background ratio", "Accuracy (95% interval)", "Label-permutation p"],
        [
            [
                level,
                interval(sensitivity[level]["accuracy"], sensitivity[level]["interval_95"]),
                f"{sensitivity[level]['permutation_p']:.4f}",
            ]
            for level in levels
        ],
    )
    arms = ["untouched", "stratified-background-augmentation"]
    heads = list(augmentation["untouched"]["pairings_by_head"])
    pairings = ["foreground_to_foreground", "foreground_to_background"]
    augmentation_rows = []
    for head in heads:
        for pairing in pairings:
            cells = [head, pairing.replace("_", " ")]
            for arm in arms:
                row = augmentation[arm]["pairings_by_head"][head][pairing]
                cells.append(interval(row["accuracy"], row["identity_block_bootstrap_accuracy_95"]))
            augmentation_rows.append(cells)
    augmentation_table = md_table(
        ["Head", "Gallery to query", "Untouched (95% interval)", "Augmented (95% interval)"],
        augmentation_rows,
    )
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.2))
    for x, level in enumerate(levels):
        row = sensitivity[level]
        low, high = row["interval_95"]
        value = row["accuracy"]
        axes[0].plot([x, x], [low, high], color=BLUE)
        axes[0].plot(
            x,
            value,
            "o",
            markeredgecolor=BLUE,
            color=BLUE if row["permutation_p"] <= 0.05 else "white",
        )
    axes[0].axhline(sensitivity[levels[0]]["chance"], color="grey", ls=":", lw=1)
    axes[0].set(
        xticks=range(3),
        xticklabels=["−10", "0", "+10"],
        xlabel="call/background ratio (dB)",
        title="A. Planted identity signal",
    )
    ridge = "kernel ridge to one-hot"
    for arm_index, arm in enumerate(arms):
        colour = [BLUE, ORANGE][arm_index]
        for x, pairing in enumerate(pairings):
            row = augmentation[arm]["pairings_by_head"][ridge][pairing]
            position = x + (arm_index - 0.5) * 0.16
            axes[1].plot([position] * 2, row["identity_block_bootstrap_accuracy_95"], color=colour)
            axes[1].plot(
                position,
                row["accuracy"],
                "o",
                color=colour,
                label=["untouched", "augmented"][arm_index] if x == 0 else None,
            )
    axes[1].set(
        xticks=range(2),
        xticklabels=["call → call", "call → background"],
        title="B. Tree-pipit augmentation",
    )
    axes[1].legend(frameon=False, fontsize=8)
    for ax in axes:
        ax.set(ylim=(0, 1), ylabel="identification accuracy")
    fig.tight_layout()
    args.out.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out / "additional-results-v6.png")
    plt.close(fig)
    numbers.write(args.out / "additional-numbers-v6.json")
    text = (
        "# Revision 6 additional completed results\n\n"
        "These components are ready for manuscript assembly. They do not establish completion "
        "of the expanded encoder comparison, spectral-subtraction series or open-set analysis.\n\n"
        "## Planted-signal sensitivity\n\n"
        "Within-year chiffchaff calls are planted in across-year territory backgrounds. "
        "This is a positive-control construction, not an estimate of the unknown identity "
        "signal in the original recordings. Intervals use 10,000 individual resamples; "
        "unadjusted endpoint p values use 9,999 label permutations.\n\n"
        + sensitivity_table
        + "\n\nThe −10 dB construction does not clear the label-permutation null, while "
        "the 0 and +10 dB constructions do. A failed test at the lowest mixture level "
        "therefore remains compatible with deliberately present identity information.\n\n"
        "## Within-year tree-pipit augmentation\n\n"
        "Only enrolment calls are augmented with other individuals' enrolment backgrounds; "
        "query clips remain untouched. Intervals use 10,000 individual resamples in both arms. "
        "The table gives arm-specific intervals, not a paired interval for their difference. "
        "The unchanged background-gallery pairings are retained in the numerical ledger.\n\n"
        + augmentation_table
        + "\n\n![Sensitivity and augmentation](additional-results-v6.png)\n\n"
        "Points show observed accuracy and lines the 95% interval. The open point in panel A "
        "does not clear its permutation null at 0.05; the dotted line is uniform chance. "
        "Panel B uses the fixed kernel-ridge head.\n\n"
        "## Expanded great-tit endpoint\n\n"
        f"BirdNET accuracy over {headline['individuals']} nest-attributed labels is "
        f"{interval(headline['accuracy'], headline['interval_95'])}, with a label-permutation "
        f"p value of {headline['permutation_p']:.4f}. "
        f"{place['shared_nestbox']} labels share a nestbox across the split, and "
        f"{place['shared_recording']} share a source recording. "
        f"The median distance between nestboxes is {place['distance_metres']['median']:.1f} m. "
        "Songs are attributed through nest attempts, not observations of the singer. "
        "This expanded sample is supplementary and excluded from the primary rank test; "
        "its difference from the smaller endpoint is not an isolated effect of sample size.\n\n"
        "All displayed values and source digests are in the "
        "[numerical ledger](additional-numbers-v6.json).\n"
    )
    (args.out / "additional-results-v6.md").write_text(text)
    print("Rendered four verified supplementary source files")


if __name__ == "__main__":
    main()
