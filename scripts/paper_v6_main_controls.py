"""Main-text recording controls from the controls ledger, checked against the gate."""

import argparse
import hashlib
import json
from pathlib import Path

from paper_figures import BLUE, GREY, ORANGE, md_table, plt
from paper_v6_additional import interval
from paper_v6_controls import PAIRINGS, complete_gate

ENDPOINTS = [
    ("chiffchaff-withinyear", "Chiffchaff\nwithin year"),
    ("littleowl-acrossyear", "Little owl\nacross year"),
    ("pipit-withinyear", "Tree pipit\nwithin year"),
    ("pipit-acrossyear", "Tree pipit\nacross year"),
    ("chiffchaff-acrossyear", "Chiffchaff\nacross year"),
]


def render(component_root, evidence_path, out):
    evidence = complete_gate(evidence_path)
    path = component_root / "recording-controls/recording-controls-numbers.json"
    ledger = json.loads(path.read_text())
    assert all(evidence["sources"][name] == sha for name, sha in ledger["sources"].items())
    pair_rows = []
    challenge_rows = []
    summary = []
    fig1, axes1 = plt.subplots(1, 5, figsize=(7.1, 3.1), sharey=True, sharex=True)
    fig2, axes2 = plt.subplots(1, 5, figsize=(7.1, 3.1), sharey=True, sharex=True)
    for idx, (ep, title) in enumerate(ENDPOINTS):
        models = {
            key.split(".controls.", 1)[1]: v["value"]
            for key, v in ledger["values"].items()
            if key.startswith(ep + ".controls.")
        }
        assert len(models) == 39
        block = models["birdnet-v2.4"]
        a, b = axes1[idx], axes2[idx]
        pair_rows.append(
            [title.replace("\n", ", ")]
            + [
                interval(
                    block["four_pairings"][key]["accuracy"],
                    block["four_pairings"][key]["interval_95"],
                )
                + f", p {block['four_pairings'][key]['permutation_p']:.4f}"
                for key in PAIRINGS
            ]
        )
        for y, key in enumerate(reversed(PAIRINGS)):
            row = block["four_pairings"][key]
            significant = row["permutation_p"] < 0.05
            a.plot(row["interval_95"], [y, y], color=BLUE, lw=1)
            a.plot(
                row["accuracy"],
                y,
                "o",
                color=BLUE,
                markerfacecolor=BLUE if significant else "white",
                ms=4,
            )
        a.set(
            title=title,
            xlim=(-0.02, 1.02),
            xlabel="Re-ID accuracy",
            yticks=range(4),
            yticklabels=list(reversed(PAIRINGS.values())),
        )
        b.axvline(0, color=GREY, lw=0.7, ls="--")
        challenge_counts = {}
        for y, (level, label) in enumerate([("-10db", "−10"), ("0db", "0"), ("10db", "+10")]):
            row = block["challenge"]["own_vs_other"][level]
            ci = row["identity_block_difference_95"]
            diff = row["own_minus_other_macro_recall"]
            sig = ci[0] > 0 or ci[1] < 0
            b.plot(ci, [y, y], color=ORANGE, lw=1)
            b.plot(diff, y, "o", color=ORANGE, markerfacecolor=ORANGE if sig else "white", ms=4)
            own = block["challenge"]["arms"]["own_" + level]
            other = block["challenge"]["arms"]["other_" + level]
            challenge_rows.append(
                [
                    title.replace("\n", ", "),
                    label,
                    interval(own["accuracy"], own["identity_block_bootstrap_accuracy_95"]),
                    interval(other["accuracy"], other["identity_block_bootstrap_accuracy_95"]),
                    f"{row['answer_changed_fraction']:.3f}",
                    interval(diff, ci),
                ]
            )
            entries = [m["challenge"]["own_vs_other"][level] for m in models.values()]
            challenge_counts[level] = {
                "positive_interval": sum(v["identity_block_difference_95"][0] > 0 for v in entries),
                "negative_interval": sum(v["identity_block_difference_95"][1] < 0 for v in entries),
                "models": len(entries),
            }
        b.set(
            title=title,
            xlabel="Own minus other,\nbalanced accuracy",
            yticks=range(3),
            # Rows are the ledger's call-to-background ratios -10, 0 and +10 dB, labelled
            # as the background's level relative to the call, as in Table 5.
            yticklabels=["+10", "0", "−10"],
        )
        norms = ledger["values"][ep + ".as_norm"]["value"]
        changes = {}
        for condition in ["calls_scored", "backgrounds_scored"]:
            deltas = [
                v[condition]["as_norm"]["accuracy"] - v[condition]["raw_class_mean"]["accuracy"]
                for v in norms.values()
            ]
            changes[condition] = {
                "increased": sum(v > 0 for v in deltas),
                "decreased": sum(v < 0 for v in deltas),
                "tied": sum(v == 0 for v in deltas),
                "minimum_change": min(deltas),
                "maximum_change": max(deltas),
            }
        background = [
            v["four_pairings"]["background_gallery_background_query"] for v in models.values()
        ]
        summary.append(
            {
                "endpoint": ep,
                "models": len(models),
                "background_accuracy_range": [
                    min(v["accuracy"] for v in background),
                    max(v["accuracy"] for v in background),
                ],
                "background_p_below_0_05_unadjusted": sum(
                    v["permutation_p"] < 0.05 for v in background
                ),
                "challenge": challenge_counts,
                "as_norm": changes,
            }
        )
    axes2[0].set_ylabel("Background relative to call (dB)")
    out.mkdir(parents=True, exist_ok=True)
    for fig, axes, name in [
        (fig1, axes1, "fig3-four-pairings-v6.png"),
        (fig2, axes2, "fig4-background-challenge-v6.png"),
    ]:
        for ax in axes:
            ax.title.set_size(9)
            ax.xaxis.label.set_size(8)
            ax.tick_params(labelsize=8)
        fig.tight_layout()
        fig.savefig(out / name)
        plt.close(fig)
    (out / "table3-four-pairings-v6.md").write_text(
        (
            "**Table 3.** BirdNET v2.4 recording controls on the recordists' "
            "divisions. C denotes calls and B backgrounds; the first condition "
            "supplies the gallery. Intervals use 10,000 individual resamples, and "
            "unadjusted p values use 9,999 label shuffles. These four-pairing tests "
            "have different resampling settings from the headline tests in Table 2.\n\n"
        )
        + md_table(["Endpoint", *PAIRINGS.values()], pair_rows)
        + "\n"
    )
    (out / "table4-background-challenge-v6.md").write_text(
        (
            "**Table 4.** BirdNET v2.4 added-background challenge, with the gallery "
            "fixed. Accuracy intervals are separate 10,000-draw arm intervals. The "
            "own-minus-other macro-recall difference uses paired individual draws; "
            "these intervals are pointwise and unadjusted across endpoints or "
            "levels. Ratios are call to added background.\n\n"
        )
        + md_table(
            [
                "Endpoint",
                "Ratio (dB)",
                "Own accuracy (95% interval)",
                "Other accuracy (95% interval)",
                "Answers changed",
                "Own − other macro recall (paired 95% interval)",
            ],
            challenge_rows,
        )
        + "\n"
    )
    (out / "expanded-controls-summary.json").write_text(
        json.dumps(
            {
                "component_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                "sources": ledger["sources"],
                "endpoints": summary,
            },
            indent=2,
        )
        + "\n"
    )
    text = (
        "Across the 39 entries, unadjusted background/background label tests were below 0.05 for "
        + "; ".join(
            f"{r['background_p_below_0_05_unadjusted']}/39 on "
            f"{dict(ENDPOINTS)[r['endpoint']].replace(chr(10), ', ')}"
            for r in summary
        )
        + (
            ". These counts describe pointwise tests, not an across-model "
            "multiplicity-adjusted discovery count. Complete pairings and challenge "
            "arms are in Appendix S1.\n"
        )
    )
    (out / "expanded-controls-summary.md").write_text(text)
    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--components", type=Path, required=True)
    parser.add_argument("--evidence", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    render(args.components, args.evidence, args.out)


if __name__ == "__main__":
    main()
