"""Prepare the complete primary v6 headline component from verified JSON evidence.

This component can be prepared before the expanded encoder experiments finish.
It does not declare the full paper complete or substitute for those experiments.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

from paper_figures import ENDPOINTS, Numbers, figure_headline, headline, md_table


class HeadlineNumbers(Numbers):
    def load(self, name: str) -> dict[str, Any]:
        return super().load(name.replace("v5-identity-", "v6-headline-"))

    def put(self, key: str, value: Any, source: str) -> Any:
        return super().put(key, value, source.replace("v5-identity-", "v6-headline-"))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, required=True)
    parser.add_argument("--v5-results", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    numbers = HeadlineNumbers(args.results)
    old_numbers = Numbers(args.v5_results)
    for stem, _, _ in ENDPOINTS:
        data = numbers.load(f"v5-identity-{stem}.json")
        old = old_numbers.load(f"v5-identity-{stem}.json")
        e = data["evaluation"]
        if e["permutation_control"]["permutations"] != 9999:
            raise ValueError(f"{stem}: expected 9,999 label permutations")
        within = e.get("within_session_permutation") or {}
        if within.get("permutations", 9999) != 9999:
            raise ValueError(f"{stem}: wrong within-recording permutation count")
        hierarchical = e.get("hierarchical_bootstrap")
        if hierarchical and hierarchical["replicates"] != 10000:
            raise ValueError(f"{stem}: expected 10,000 hierarchical resamples")
        if (
            data["manifest_sha256"] != old["manifest_sha256"]
            or data["query_identities"] != old["query_identities"]
            or data["query_correct"] != old["query_correct"]
            or e["classification"]["accuracy"] != old["evaluation"]["classification"]["accuracy"]
        ):
            raise ValueError(f"{stem}: headline does not reproduce v5 on identical query rows")
    rows = headline(numbers)
    if len(rows) != 13:
        raise ValueError("all thirteen primary endpoints are required")
    table = md_table(
        [
            "Endpoint",
            "n",
            "Accuracy (95% interval)",
            "Three-stage interval",
            "Majority class",
            "Null 95th percentile",
            "p",
            "Within-recording null 95th percentile (p)",
        ],
        [
            [
                r["label"],
                r["individuals"],
                f"{r['accuracy']:.3f} ({r['interval'][0]:.3f}–{r['interval'][1]:.3f})",
                r["three_stage"],
                r["majority"],
                r["permutation_95"],
                f"{r['permutation_p']:.4f}",
                f"{r['within_95']:.3f} ({r['within_p']:.4f})"
                if r["within_p"] is not None
                else "not constructible",
            ]
            for r in rows
        ],
    )
    args.out.mkdir(parents=True, exist_ok=True)
    figure_headline(rows, args.out / "fig2-closed-set-v6.png")
    numbers.sources.update(old_numbers.sources)
    numbers.values["headline_reproduction"] = {
        "value": "All thirteen primary endpoints reproduce v5 accuracy and per-query correctness",
        "source": "paired v5/v6 headline source files in this ledger",
    }
    numbers.write(args.out / "headline-numbers-v6.json")
    text = (
        "# Revision 6 headline component\n\n"
        "This completed component covers the thirteen primary BirdNET endpoints. "
        "It is not the completed revision-6 manuscript: the expanded encoder, "
        "recording-control and open-set analyses remain separate prerequisites.\n\n"
        "**Table 2.** BirdNET v2.4 through the fixed kernel-ridge head. The 95% interval "
        "resamples individuals 10,000 times. The three-stage interval also resamples "
        "recordings within individuals and clips within recordings. Label and "
        "within-recording nulls use 9,999 shuffles; the smallest attainable p is 0.0001. "
        "These endpoint-wise p values are unadjusted for multiple testing. "
        "All thirteen accuracies and per-query correctness strings reproduce v5. "
        "The cockatoo release supplies a random clip fold; the penguin labels remain "
        "nest-attributed and do not separate birds from nest or recorder.\n\n"
        + table
        + "\n\n![Closed-set headline results](fig2-closed-set-v6.png)\n\n"
        "**Figure 2.** Points show accuracy and lines the 95% interval over individuals. "
        "Vertical ticks show the 95th percentile of the label-permutation null; open "
        "points do not clear that null at 0.05. Counts in parentheses are individuals "
        "or the release's attribution labels.\n\n"
        "Source file digests and all displayed values are retained in "
        "[the numerical ledger](headline-numbers-v6.json). A recording-associated "
        "label can carry site, nest or recording information as well as vocal identity; "
        "clearing a label-permutation null alone does not distinguish these explanations.\n"
    )
    (args.out / "headline-v6.md").write_text(text)
    print("Verified and rendered all thirteen primary headline endpoints")


if __name__ == "__main__":
    main()
