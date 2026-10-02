"""Render a completed endpoint's paired comparisons without a partial rank test."""

from __future__ import annotations

import argparse
from pathlib import Path

from paper_figures import BLUE, GREY, ORANGE, Numbers, md_table, plt
from paper_figures import ENDPOINTS as LABELS
from paper_v6_additional import interval
from paper_v6_evidence import ALL_MODELS, ENDPOINTS


def difference(value: float, bounds: list[float]) -> str:
    return f"{value:+.3f} ({bounds[0]:+.3f}–{bounds[1]:+.3f})"


def render(results: Path, endpoint: str, out: Path, *, source: str | None = None) -> None:
    numbers = Numbers(results)
    source = source or f"v6-comparison-{endpoint}.json"
    data = numbers.load(source)
    if endpoint not in data["per_endpoint"]:
        raise ValueError("requested endpoint is absent from the archived comparison")
    row = data["per_endpoint"][endpoint]
    rules = {"Fisher": row, "Session holdout": row["held_out_rule"]}
    expected = {model.split("/")[-1] for model in ALL_MODELS}
    for rule, block in rules.items():
        paired = block["paired"]
        if rule == "Fisher" and set(block["selection"]) != expected:
            raise ValueError("comparison does not cover every model/control entry")
        if paired["replicates"] != 10000 or paired["reference"] != "birdnet-v2.4":
            raise ValueError("comparison has the wrong reference or resample count")
        if set(paired["comparisons"]) != set(block["selection"]) - {"birdnet-v2.4"}:
            raise ValueError("paired results do not cover all available choices")
        numbers.put(rule, {key: block[key] for key in ("selection", "paired")}, source)
    names = sorted(expected - {"birdnet-v2.4"})
    title = {name: label for name, label, _ in LABELS}.get(endpoint, endpoint)
    title = {
        "penguin-acrossnight": "Little penguin, across night (nest-attributed)",
        "great-tit": "Great tit, across year (nest-attributed)",
        "great-tit-full": "Great tit, expanded nest-attributed sample",
    }.get(endpoint, title)
    fig, axes = plt.subplots(1, 2, figsize=(7.1, 9.2), sharey=True, sharex=True)
    rows = []
    references = []
    for axis, (rule, block) in zip(axes, rules.items(), strict=True):
        paired = block["paired"]
        references.append(
            f"{rule}: BirdNET accuracy "
            + interval(paired["reference_accuracy"], paired["reference_accuracy_95"])
            + f"; {len(paired['comparisons'])} paired comparisons."
        )
        axis.axvline(0, color=GREY, lw=0.7, ls="--")
        axis.set(title=rule, xlabel="Accuracy difference\nfrom BirdNET")
        axis.tick_params(axis="x", labelsize=8.5)
        axis.xaxis.label.set_size(9)
        axis.title.set_size(10)
        for y, model in enumerate(reversed(names)):
            entry = paired["comparisons"].get(model)
            if entry is None:
                continue
            delta = entry["difference_from_reference"]
            significant = entry["differs_at_0.05_after_holm"]
            colour = (BLUE if delta > 0 else ORANGE) if significant else GREY
            axis.plot(entry["difference_95"], [y, y], color=colour, lw=0.8)
            axis.plot(
                delta,
                y,
                "o",
                color=colour,
                ms=3,
                markerfacecolor=colour if significant else "white",
            )
            rows.append(
                [
                    model,
                    rule,
                    block["selection"][model]["selected"],
                    interval(entry["accuracy"], entry["accuracy_95"]),
                    difference(delta, entry["difference_95"]),
                    f"{entry['p_holm']:.4f}",
                ]
            )
    axes[0].set(yticks=range(len(names)), yticklabels=list(reversed(names)))
    axes[0].tick_params(axis="y", labelsize=8.5)
    fig.suptitle(title, fontsize=11)
    fig.tight_layout()
    out.mkdir(parents=True, exist_ok=True)
    fig.savefig(out / f"{endpoint}-paired.png")
    plt.close(fig)
    numbers.write(out / f"{endpoint}-paired-numbers.json")
    table = md_table(
        [
            "Model",
            "Rule",
            "Representation",
            "Accuracy (95% interval)",
            "Difference (95% paired interval)",
            "Holm-adjusted tail proportion",
        ],
        sorted(rows, key=lambda item: (item[0], item[1])),
    )
    text = (
        f"# Revision 6 paired comparison: {title}\n\n"
        "Each comparison uses the same 10,000 individual resamples for the model and BirdNET. "
        "Holm adjustment applies within this endpoint and representation rule, not across "
        "endpoints or across both rules together. The reported two-sided bootstrap tail "
        "proportion is an empirical estimate; 0.0000 records no draws on the opposing side, "
        "not a claim of a mathematically zero probability.\n\n"
        + "\n\n".join(references)
        + f"\n\n![Paired accuracy differences]({endpoint}-paired.png)\n\n"
        "Filled blue/orange points identify positive/negative differences that clear the "
        "Holm-adjusted 0.05 gate; hollow grey points do not. Lines show unadjusted paired "
        "95% intervals. Selection uses enrolment only. Unavailable session-holdout choices "
        "are omitted from that rule's panel and denominator. Session holdout selects "
        "representations within enrolment; it does not change the gallery/query split. "
        "The cockatoo retains its published random clip fold under both rules.\n\n"
        + table
        + f"\n\nValues and the archived source digest are in the [numerical ledger]"
        f"({endpoint}-paired-numbers.json). This endpoint analysis does not establish "
        "completion of the thirteen-endpoint rank comparison.\n"
    )
    (out / f"{endpoint}-paired.md").write_text(text)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, required=True)
    parser.add_argument("--endpoint", action="append", choices=ENDPOINTS, required=True)
    parser.add_argument(
        "--comparison",
        help="Archived comparison filename under --results; may contain multiple endpoints",
    )
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    for endpoint in args.endpoint:
        render(args.results, endpoint, args.out, source=args.comparison)


if __name__ == "__main__":
    main()
