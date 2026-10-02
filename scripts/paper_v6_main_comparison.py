"""Comparison with BirdNET under both layer rules, from the gated comparison result."""

import argparse
import hashlib
import json
import statistics
from pathlib import Path

from paper_figures import md_table
from paper_v6_controls import complete_gate
from paper_v6_evidence import ENDPOINTS
from paper_v6_overview import LABELS

RULES = (("Fisher", "paired", "selection"), ("Session held out", "held_out_rule", None))


def checked(path: Path, digest: str) -> dict:
    raw = path.read_bytes()
    assert hashlib.sha256(raw).hexdigest() == digest, path.name
    return json.loads(raw)


def rule_blocks(endpoint: dict, rule: str) -> tuple[dict, dict]:
    if rule == "paired":
        return endpoint["paired"], endpoint["selection"]
    return endpoint["held_out_rule"]["paired"], endpoint["held_out_rule"]["selection"]


def contrasts(paired: dict) -> tuple[int, int, int]:
    rows = paired["comparisons"].values()
    higher = sum(
        r["differs_at_0.05_after_holm"] and r["difference_from_reference"] > 0 for r in rows
    )
    lower = sum(
        r["differs_at_0.05_after_holm"] and r["difference_from_reference"] < 0 for r in rows
    )
    return higher, lower, len(paired["comparisons"])


def render(results: Path, components: Path, evidence_path: Path, out: Path) -> Path:
    evidence = complete_gate(evidence_path)
    name = "v6-encoder-comparison.json"
    comparison = checked(results / name, evidence["sources"][name])
    rows, totals = [], {rule: [0, 0, 0] for _, rule, _ in RULES}
    changed, differences = 0, []
    for ep in ENDPOINTS[:-1]:
        endpoint = comparison["per_endpoint"][ep]
        cells = [LABELS[ep].replace("\n", ", ")]
        for _, rule, _ in RULES:
            paired, selection = rule_blocks(endpoint, rule)
            higher, lower, count = contrasts(paired)
            for i, v in enumerate((higher, lower, count)):
                totals[rule][i] += v
            cells += [len(selection), f"{higher} / {lower}"]
        fisher, held = endpoint["selection"], endpoint["held_out_rule"]["selection"]
        different = [
            m
            for m in sorted(set(fisher) & set(held))
            if fisher[m]["selected"] != held[m]["selected"]
        ]
        changed += len(different)
        differences += [
            held[m]["selected_accuracy"] - fisher[m]["selected_accuracy"] for m in different
        ]
        cells.append(len(different))
        rows.append(cells)
    rank_rows = []
    for label, key in (
        ("Fisher", "rank_consistency"),
        ("Session held out", "rank_consistency_held_out_rule"),
    ):
        r = comparison[key]
        assert r["computed"] and len(r["endpoints"]) == 13
        best = list(r["average_rank"].items())[:3]
        rank_rows.append(
            [
                label,
                len(r["representations"]),
                f"{r['friedman_chi_square']:.1f}",
                f"{r['friedman_p']:.2g}",
                f"{r['kendall_w']:.2f}",
                f"{r['nemenyi_critical_difference_0.05']:.1f}",
                f"{r['average_rank']['birdnet-v2.4']:.2f}",
                "; ".join(f"{m} ({v:.2f})" for m, v in best),
            ]
        )
    summary = {
        "contrasts": {
            label: dict(zip(("higher", "lower", "comparisons"), totals[rule], strict=True))
            for label, rule, _ in RULES
        },
        "choices_changed": changed,
        "held_out_higher": sum(d > 0 for d in differences),
        "held_out_lower": sum(d < 0 for d in differences),
        "held_out_same": sum(d == 0 for d in differences),
        "median_change": statistics.median(differences) if differences else None,
    }
    text = (
        "**Table S10.** Entries compared with BirdNET v2.4 on each dataset under the two "
        "layer rules, and the number above / below BirdNET after Holm correction within "
        "each dataset. The last column counts entries whose chosen layer differed between "
        "the rules.\n\n"
        + md_table(
            [
                "Dataset",
                "Fisher entries",
                "Above / below",
                "Held-out entries",
                "Above / below",
                "Different layer",
            ],
            rows,
        )
        + "\n\n**Table S11.** Rank consistency across the 13 datasets.\n\n"
        + md_table(
            [
                "Rule",
                "Entries ranked",
                "Friedman χ²",
                "p",
                "Kendall's W",
                "Nemenyi CD",
                "BirdNET rank",
                "Three best average ranks",
            ],
            rank_rows,
        )
        + "\n"
    )
    information = []
    for ep in ENDPOINTS:
        path = components / "information" / f"{ep}-numbers.json"
        d = json.loads(path.read_text())["values"]["models"]["value"]
        measured = {
            m: v["fisher"]["hs_bits"] for m, v in d.items() if v["fisher"]["hs_bits"] is not None
        }
        assert len(measured) == 37
        information.append(
            {
                "endpoint": ep,
                "minimum_hs_bits": min(measured.values()),
                "maximum_hs_bits": max(measured.values()),
                "birdnet_hs_bits": measured["birdnet-v2.4"],
            }
        )
    out.mkdir(parents=True, exist_ok=True)
    (out / "comparison-main-text.md").write_text(text)
    (out / "comparison-summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    (out / "information-summary.json").write_text(json.dumps(information, indent=2) + "\n")
    (out / "information-summary.md").write_text(
        "Beecher's information statistic (bits) on the enrolment calls, for BirdNET and the "
        "range across the 37 neural networks.\n\n"
        + md_table(
            ["Dataset", "BirdNET (bits)", "Range (bits)"],
            [
                [
                    LABELS.get(v["endpoint"], "Great tit, 50 birds").replace("\n", ", "),
                    f"{v['birdnet_hs_bits']:.2f}",
                    f"{v['minimum_hs_bits']:.2f} to {v['maximum_hs_bits']:.2f}",
                ]
                for v in information
            ],
        )
        + "\n"
    )
    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, required=True)
    parser.add_argument("--components", type=Path, required=True)
    parser.add_argument("--evidence", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    render(args.results, args.components, args.evidence, args.out)


if __name__ == "__main__":
    main()
