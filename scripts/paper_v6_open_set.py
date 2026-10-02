"""Render complete v6 open-set results with achieved calibrated stranger rates."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
from paper_v6_controls import complete_gate
from paper_v6_evidence import OPEN_ENDPOINTS, validate

from xinyenyana.open_set import STRANGER_BUDGETS, summarise_allocations


def spread(values: list[float]) -> dict[str, float]:
    return {
        "minimum": float(np.min(values)),
        "median": float(np.median(values)),
        "maximum": float(np.max(values)),
    }


def display(row: dict[str, float]) -> str:
    return f"{row['median']:.3f} ({row['minimum']:.3f}–{row['maximum']:.3f})"


def public_metrics(row: dict[str, Any]) -> dict[str, Any]:
    """Recompute summaries from all allocations and expose no identity observations."""
    allocations = row["allocations"]
    if len(allocations) != 16:
        raise ValueError("all sixteen allocations are required")
    metrics = summarise_allocations(allocations)
    if metrics != row["allocation_sensitivity"]:
        raise ValueError("open-set allocation summaries do not reconstruct")
    for budget in STRANGER_BUDGETS:
        key = f"{budget:.2f}"
        metrics[f"achieved_stranger_acceptance_calibrated_at_{key}"] = spread(
            [
                allocation["calibrated_on_other_birds"][key]["test"]["unknown_false_accept_rate"]
                for allocation in allocations
            ]
        )
    return {"representation": row["representation"], "allocation_sensitivity": metrics}


DISPLAY = {"perch-v2": "Perch 2.0"}

FIGURE_LABELS = {
    "great-tit": "Great tit (16)",
    "great-tit-full": "Great tit, 50 birds (50)",
    "littleowl-acrossyear": "Little owl (16)",
    "rookid-full-width": "Rook (11)",
    "cockatoo-fold1": "Cockatoo (16), random split",
}


def figure_open_set(endpoints: dict[str, Any], leader: str, out: Path) -> None:
    """Figure 6: naming among known birds, and acceptance with correct naming at 10%."""
    from paper_figures import BLUE, ORANGE, plt

    names = [ep for ep in FIGURE_LABELS if ep in endpoints]
    fig, axes = plt.subplots(1, 2, figsize=(7.1, 3.4), sharey=True, sharex=True)
    for ax, (rule, title) in zip(
        axes,
        [("reference", "BirdNET v2.4"), ("best average rank", DISPLAY.get(leader, leader))],
        strict=True,
    ):
        for y, ep in enumerate(reversed(names)):
            m = endpoints[ep]["models"][endpoints[ep]["models_by_rule"][rule]]
            m = m["allocation_sensitivity"]
            for key, colour, marker, dy in [
                ("closed_set_accuracy", ORANGE, "D", -0.15),
                ("acceptance_calibrated_at_0.10", BLUE, "o", 0.15),
            ]:
                row = m[key]
                ax.plot([row["minimum"], row["maximum"]], [y + dy, y + dy], color=colour, lw=1.2)
                ax.plot(row["median"], y + dy, marker, color=colour, ms=5)
            low, high = endpoints[ep]["chance"]
            for value in {low, high}:
                ax.plot(value, y - 0.15, "|", color="#555555", ms=9, mew=1.5)
        ax.set(xlim=(0, 1), title=title)
        ax.set_xlabel("Proportion of known birds' calls\n(median and range, 16 allocations)")
    axes[0].set(yticks=range(len(names)), yticklabels=[FIGURE_LABELS[ep] for ep in reversed(names)])
    handles = [
        plt.Line2D([], [], color=ORANGE, marker="D", lw=1.2, label="named correctly, no strangers"),
        plt.Line2D(
            [], [], color="#555555", marker="|", lw=0, ms=9, mew=1.5, label="chance for naming"
        ),
        plt.Line2D(
            [],
            [],
            color=BLUE,
            marker="o",
            lw=1.2,
            label="accepted and named correctly, calibrated threshold, 10% stranger budget",
        ),
    ]
    fig.legend(handles=handles, loc="lower center", ncol=1, frameon=False, fontsize=8.5)
    for ax in axes:
        ax.title.set_size(10)
        ax.tick_params(labelsize=8.5)
        ax.xaxis.label.set_size(8.5)
    fig.tight_layout(rect=(0, 0.19, 1, 1))
    fig.savefig(out)
    plt.close(fig)


def render(results: Path, evidence: Path, out: Path) -> None:
    from paper_figures import BLUE, ORANGE, Numbers, md_table, plt
    from paper_v6_overview import LABELS

    report = complete_gate(evidence)
    numbers = Numbers(results)
    comparison = numbers.load("v6-encoder-comparison.json")
    if (
        numbers.sources["v6-encoder-comparison.json"]
        != report["sources"]["v6-encoder-comparison.json"]
    ):
        raise ValueError("comparison bytes differ from the complete evidence inventory")
    ranks = comparison["rank_consistency"]["average_rank"]
    # BirdNET is the reference; the other model shown is the one with the best average
    # rank over the thirteen closed-set datasets, the same one on every dataset. All 39
    # entries are in the numerical ledger and the supporting table.
    leader = min((rank, name) for name, rank in ranks.items() if name != "birdnet-v2.4")[1]
    endpoints = {}
    for ep in OPEN_ENDPOINTS:
        name = f"v6-counts/v6-open-set-{ep}.json"
        data = numbers.load(name)
        if numbers.sources[name] != report["sources"][name]:
            raise ValueError(f"{name}: bytes differ from the complete evidence inventory")
        validate(data, "open-set", ep, ())
        models = {model: public_metrics(row) for model, row in data["representations"].items()}
        short = {model.split("/")[-1]: model for model in models}
        choices = {"reference": "birdnet-v2.4", "best average rank": short[leader]}
        # Chance for naming a known individual correctly: one over the test-known birds.
        known = [
            len(allocation["roles"]["test_known"])
            for allocation in next(iter(data["representations"].values()))["allocations"]
        ]
        chance = [1 / max(known), 1 / min(known)]
        endpoints[ep] = {"models_by_rule": choices, "models": models, "chance": chance}
        numbers.put(ep, endpoints[ep], name)
    out.mkdir(parents=True, exist_ok=True)
    introduction = (
        "All entries are medians across sixteen allocations, with minimum–maximum ranges. "
        "Ranges show allocation sensitivity, not confidence intervals. All 39 embeddings and "
        "controls were measured; BirdNET and the network with the best average rank over the "
        "thirteen closed-set datasets are shown here, the same network on every dataset. "
        "The expanded great-tit endpoint is supplementary and "
        "nest-attributed.\n\n"
        "Each allocation's gallery contains only its test-known subset. Its closed-set accuracy "
        "therefore has fewer competing labels than the full endpoint's primary closed-set test. "
        "Both great-tit constructions use nest-attributed labels; neither directly observes "
        "the singer.\n\n"
        "Calibrated thresholds use other individuals. Their achieved test-stranger acceptance may "
        "exceed or fall below the calibration budget and is reported separately. Test-derived "
        "operating "
        "points use the test scores and constrain stranger acceptance to at most the stated "
        "budget; "
        "they are descriptive operating points, not thresholds calibrated independently of test "
        "strangers. "
        "Acceptance with correct naming and acceptance ignoring naming are distinct measures. "
        "Balanced known accuracy averages correct acceptance and naming within each individual, "
        "then across individuals; balanced stranger accuracy similarly averages correct rejection. "
        "Their geometric mean is computed within each allocation before summarising "
        "allocations.\n\n"
    )
    overview = "# Revision 6 open-set identification\n\n" + introduction
    summary_rows, calibrated_rows = [], []
    for ep, endpoint in endpoints.items():
        title = LABELS.get(ep, "Great tit\nexpanded nest-attributed sample").replace("\n", ", ")
        rows, calibration = [], []
        for rule, model in endpoint["models_by_rule"].items():
            m = endpoint["models"][model]["allocation_sensitivity"]
            label = model.split("/")[-1]
            summary_rows.append(
                [
                    title,
                    rule,
                    label,
                    display(m["closed_set_accuracy"]),
                    display(m["known_vs_unknown_auroc"]),
                    display(m["best_acceptance_at_0.10"]),
                ]
            )
            calibrated_rows.append(
                [
                    title,
                    rule,
                    label,
                    display(m["acceptance_calibrated_at_0.10"]),
                    display(m["achieved_stranger_acceptance_calibrated_at_0.10"]),
                    display(m["balanced_accuracy_known_calibrated_at_0.10"]),
                    display(m["balanced_accuracy_unknown_calibrated_at_0.10"]),
                    display(m["geometric_mean_known_unknown_calibrated_at_0.10"]),
                ]
            )
            for budget in STRANGER_BUDGETS:
                key = f"{budget:.2f}"
                rows.append(
                    [
                        rule,
                        label,
                        key,
                        display(m[f"best_acceptance_at_{key}"]),
                        display(m[f"true_accept_at_false_accept_{key}"]),
                        display(m[f"acceptance_calibrated_at_{key}"]),
                        display(m[f"achieved_stranger_acceptance_calibrated_at_{key}"]),
                    ]
                )
                calibration.append(
                    [
                        rule,
                        label,
                        key,
                        display(m[f"balanced_accuracy_known_calibrated_at_{key}"]),
                        display(m[f"balanced_accuracy_unknown_calibrated_at_{key}"]),
                        display(m[f"geometric_mean_known_unknown_calibrated_at_{key}"]),
                    ]
                )
        text = "# Open set: " + title + "\n\n" + introduction
        text += md_table(
            [
                "Choice",
                "Model",
                "Stranger budget",
                "Test-derived accept and name",
                "Test-derived accept, naming ignored",
                "Calibrated accept and name",
                "Achieved test-stranger acceptance",
            ],
            rows,
        )
        text += "\n\n## Balanced performance at calibrated thresholds\n\n" + md_table(
            [
                "Choice",
                "Model",
                "Calibration budget",
                "Balanced known accuracy",
                "Balanced stranger accuracy",
                "Geometric mean",
            ],
            calibration,
        )
        text += (
            "\n\nAll metric summaries, including equal error rate, and source digests are in "
            "[the numerical ledger](open-set-numbers.json).\n"
        )
        (out / f"{ep}-open-set.md").write_text(text)
        fig, axes = plt.subplots(1, 2, figsize=(7.1, 3.8), sharey=True)
        for y, (_rule, model) in enumerate(endpoint["models_by_rule"].items()):
            m = endpoint["models"][model]["allocation_sensitivity"]
            for ax, key, colour in [
                (axes[0], "best_acceptance_at_0.10", BLUE),
                (axes[1], "acceptance_calibrated_at_0.10", ORANGE),
            ]:
                row = m[key]
                ax.plot([row["minimum"], row["maximum"]], [y, y], color=colour)
                ax.plot(row["median"], y, "o", color=colour)
                ax.set(xlim=(0, 1), xlabel="Accepted and named correctly")
        axes[0].set(
            yticks=range(len(endpoint["models_by_rule"])),
            yticklabels=[
                r + "\n" + m.split("/")[-1] for r, m in endpoint["models_by_rule"].items()
            ],
            title="Test-derived: stranger rate ≤ 0.10",
        )
        axes[1].set_title("Calibrated at stranger budget 0.10")
        axes[0].tick_params(axis="y", labelsize=8.5)
        for ax in axes:
            ax.tick_params(axis="x", labelsize=8.5)
            ax.xaxis.label.set_size(8.5)
            ax.title.set_size(9)
        fig.suptitle(title, fontsize=11)
        fig.tight_layout()
        fig.savefig(out / f"{ep}-open-set.png")
        plt.close(fig)
        overview += f"![{title}: open-set operating points]({ep}-open-set.png)\n\n"
    overview += "## Closed set, separation and test-derived operating point\n\n" + md_table(
        [
            "Endpoint",
            "Choice",
            "Model",
            "Names correctly (range)",
            "Known/stranger AUROC (range)",
            "Test-derived accept and name at ≤10% strangers (range)",
        ],
        summary_rows,
    )
    overview += "\n\n## Calibrated at a 10% stranger budget\n\n" + md_table(
        [
            "Endpoint",
            "Choice",
            "Model",
            "Accept and name (range)",
            "Achieved stranger acceptance (range)",
            "Balanced known (range)",
            "Balanced strangers (range)",
            "Geometric mean (range)",
        ],
        calibrated_rows,
    )
    overview += (
        "\n\nSeparate endpoint tables retain all five registered stranger budgets. Sources and "
        "numerical values are in [the numerical ledger](open-set-numbers.json).\n"
    )
    figure_open_set(endpoints, leader, out / "fig6-open-set-v6.png")
    (out / "open-set-overview.md").write_text(overview)
    numbers.write(out / "open-set-numbers.json")
    (out / "assembly-provenance.json").write_text(
        json.dumps(
            {
                "evidence_sha256": hashlib.sha256(evidence.read_bytes()).hexdigest(),
                "endpoints": list(OPEN_ENDPOINTS),
                "allocations_per_model": 16,
            },
            indent=2,
        )
        + "\n"
    )


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--results", type=Path, required=True)
    p.add_argument("--evidence", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    a = p.parse_args()
    render(a.results, a.evidence, a.out)


if __name__ == "__main__":
    main()
