"""Render completed endpoint analyses without declaring the whole paper complete."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

from paper_figures import BLUE, ORANGE, Numbers, md_table, plt
from paper_figures import ENDPOINTS as LABELS
from paper_v6_additional import interval
from paper_v6_evidence import ENDPOINTS, validate


def metrics(row: dict[str, Any]) -> dict[str, Any]:
    """Only aggregate fields enter the publication ledger."""
    nulls = row["permutation_nulls"]
    within = nulls.get("within_session_permutation") or {}
    information = row.get("beecher_hs") or {}
    return {
        "representation": row["representation"],
        "accuracy": nulls["classification"]["accuracy"],
        "interval_95": nulls["identity_block_bootstrap_accuracy_95"],
        "label_permutation_p": nulls["permutation_control"]["p_value_plus_one"],
        "within_recording_p": within.get("p_value_plus_one"),
        "hs_bits": information.get("hs_bits"),
        "hs_bits_by_components": information.get("hs_bits_by_components"),
        "components_used": information.get("components_used"),
    }


def render(results: Path, endpoint: str, out: Path) -> None:
    numbers = Numbers(results)
    source = f"v6-analyses-{endpoint}.json"
    data = numbers.load(source)
    validate(data, "analyses", endpoint, ())
    models = {}
    for model, row in sorted(data["models"].items()):
        fisher = metrics(row)
        other = row.get("held_out_rule")
        held = None
        if other:
            held = fisher if other.get("same_as_fisher_rule") else metrics(other)
        models[model] = {"fisher": fisher, "held_out": held}
    numbers.put("models", models, source)
    title = dict((name, label) for name, label, _ in LABELS).get(endpoint, endpoint)
    title = {
        "rookid-full-width": "Rook, full aviary, across year",
        "penguin-acrossnight": "Little penguin, across night (nest-attributed)",
        "great-tit": "Great tit, across year (nest-attributed)",
        "great-tit-full": "Great tit, expanded nest-attributed sample",
    }.get(endpoint, title)
    rows = []
    for model, rules in models.items():
        for rule, values in rules.items():
            if values is None:
                rows.append([model, "Session holdout", "unavailable", "—", "—", "—", "—"])
                continue
            rows.append(
                [
                    model,
                    "Fisher" if rule == "fisher" else "Session holdout",
                    values["representation"],
                    interval(values["accuracy"], values["interval_95"]),
                    f"{values['label_permutation_p']:.4f}",
                    "not constructible"
                    if values["within_recording_p"] is None
                    else f"{values['within_recording_p']:.4f}",
                    "not computed"
                    if values["hs_bits"] is None
                    else f"{values['hs_bits']:.3f} ({values['components_used']})",
                ]
            )
    table = md_table(
        [
            "Model",
            "Rule",
            "Representation",
            "Accuracy (95% interval)",
            "Label p",
            "Within-recording p",
            "Hs bits (PCs)",
        ],
        rows,
    )
    names = list(models)
    fig, axes = plt.subplots(
        1, 2, figsize=(7.1, 9.2), sharey=True, gridspec_kw={"width_ratios": [2, 1]}
    )
    for y, model in enumerate(reversed(names)):
        for offset, rule, colour in [(-0.12, "fisher", BLUE), (0.12, "held_out", ORANGE)]:
            values = models[model][rule]
            if values is None:
                continue
            low, high = values["interval_95"]
            axes[0].plot([low, high], [y + offset] * 2, color=colour, lw=0.8)
            axes[0].plot(values["accuracy"], y + offset, "o", color=colour, ms=3)
            if values["hs_bits"] is not None:
                axes[1].plot(values["hs_bits"], y + offset, "o", color=colour, ms=3)
    axes[0].set(
        yticks=range(len(names)),
        yticklabels=[name.split("/")[-1] for name in reversed(names)],
        xlim=(0, 1.02),
        xlabel="Accuracy (95% interval)",
    )
    axes[0].tick_params(axis="y", labelsize=8.5)
    axes[1].set(xlabel="Descriptive Hs (bits)", xlim=(0, None))
    for axis in axes:
        axis.tick_params(axis="x", labelsize=8.5)
        axis.xaxis.label.set_size(9)
    axes[0].legend(
        handles=[
            plt.Line2D([], [], color=colour, marker="o", label=label)
            for colour, label in [(BLUE, "Fisher"), (ORANGE, "Session holdout")]
        ],
        loc="lower left",
        bbox_to_anchor=(0, 1.005),
        frameon=False,
        ncol=2,
        fontsize=8,
    )
    fig.suptitle(title, fontsize=11, y=0.995)
    fig.tight_layout()
    out.mkdir(parents=True, exist_ok=True)
    fig.savefig(out / f"{endpoint}.png")
    plt.close(fig)
    numbers.write(out / f"{endpoint}-numbers.json")
    text = (
        f"# Revision 6 completed analysis: {title}\n\n"
        "This endpoint's 39 model/control entries have completed their registered stored-vector "
        "analyses. Other endpoints and the final across-endpoint comparison remain separate "
        "requirements for the manuscript.\n\n"
        "Both representation rules use enrolment information only. Session holdout is unavailable "
        "when no enrolment session division is measurable; a one-candidate model retains that "
        "sole candidate. The same representation under both rules repeats the same measurement. "
        "Session holdout selects representations within enrolment and does not change the "
        "gallery/query split; the cockatoo retains its published random clip fold. "
        "Intervals resample individuals 10,000 times. Label and constructible within-recording "
        "nulls use 9,999 shuffles. The p values below are unadjusted; this table does not replace "
        "the paired, Holm-adjusted comparisons against BirdNET.\n\n"
        f"![Accuracy and descriptive identity information]({endpoint}.png)\n\n"
        "Hs is the registered descriptive statistic on pooled foreground vectors, with at most "
        "one fewer principal component than labels. It can describe recording information "
        "associated with labels and is not a held-out identification test. No paired interval "
        "for the difference between representation rules is inferred from these separate "
        "accuracy intervals.\n\n"
        + table
        + f"\n\nAll plotted and tabulated values, representation names and the source digest "
        f"are in the [numerical ledger]({endpoint}-numbers.json).\n"
    )
    (out / f"{endpoint}.md").write_text(text)
    print(f"Rendered {endpoint}: {len(models)} model/control entries")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, required=True)
    parser.add_argument("--endpoint", action="append", choices=ENDPOINTS, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    for endpoint in args.endpoint:
        render(args.results, endpoint, args.out)


if __name__ == "__main__":
    main()
