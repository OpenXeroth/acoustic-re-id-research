"""Publish allocation sizes without disclosing identity-to-role assignments."""

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

from paper_v6_controls import complete_gate
from paper_v6_evidence import OPEN_ENDPOINTS

ROLES = ("calibration_known", "calibration_unknown", "test_known", "test_unknown")


def counts(data: dict[str, Any]) -> dict[str, Any]:
    common = None
    label_count = None
    for row in data["representations"].values():
        allocations = row["allocations"]
        if len(allocations) != 16:
            raise ValueError("all sixteen allocations are required")
        actual = [a["roles"] for a in allocations]
        if common is not None and actual != common:
            raise ValueError("models use different role assignments")
        common = actual
    if not common:
        raise ValueError("no model allocations")
    result = []
    for assignment in common:
        if set(assignment) != set(ROLES) or any(not assignment[role] for role in ROLES):
            raise ValueError("four nonempty roles are required")
        combined = [identity for role in ROLES for identity in assignment[role]]
        if len(combined) != len(set(combined)):
            raise ValueError("allocation roles overlap")
        if label_count is not None and label_count != len(combined):
            raise ValueError("allocation label counts differ")
        label_count = len(combined)
        result.append({role: len(assignment[role]) for role in ROLES})
    return {"attribution_labels": label_count, "allocations": result}


def render(results: Path, evidence_path: Path, out: Path) -> None:
    evidence = complete_gate(evidence_path)
    from paper_figures import md_table
    from paper_v6_overview import LABELS

    values = {}
    sources = {}
    table = []

    def spread(rows, key):
        sizes = [r[key] for r in rows]
        lo, hi = min(sizes), max(sizes)
        return str(lo) if lo == hi else f"{lo}–{hi}"

    for ep in OPEN_ENDPOINTS:
        name = f"v6-counts/v6-open-set-{ep}.json"
        raw = (results / name).read_bytes()
        digest = hashlib.sha256(raw).hexdigest()
        if evidence["sources"][name] != digest:
            raise ValueError(f"{name}: bytes differ from the complete evidence inventory")
        values[ep] = counts(json.loads(raw))
        sources[name] = digest
        rows = values[ep]["allocations"]
        title = LABELS.get(ep, "Great tit, expanded nest-attributed sample").replace("\n", ", ")
        table.append(
            [
                title,
                str(values[ep]["attribution_labels"]),
                spread(rows, "calibration_known") + " / " + spread(rows, "calibration_unknown"),
                spread(rows, "test_known") + " / " + spread(rows, "test_unknown"),
                str(len(rows)),
            ]
        )
    out.mkdir(parents=True, exist_ok=True)
    text = (
        "# Open-set allocation sizes\n\nCounts are attribution labels, not independent "
        "confirmation of singer identity. The test gallery contains only test-known labels,"
        " so its closed-set accuracy has fewer competing labels than the primary full-"
        "gallery result. Calibration and testing use disjoint individuals or attribution "
        "labels. All models use the same roles within each allocation.\n\n"
    )
    text += md_table(
        [
            "Endpoint",
            "All labels",
            "Calibration known / stranger",
            "Test gallery / stranger",
            "Allocations",
        ],
        table,
    )
    text += (
        "\n\nRanges, if present, describe counts across allocations. The [count ledger](open-"
        "set-role-counts.json) records every allocation size and its source digest; private"
        " identity assignments are excluded.\n"
    )
    (out / "open-set-role-counts.md").write_text(text)
    (out / "open-set-role-counts.json").write_text(
        json.dumps({"sources": sources, "endpoints": values}, indent=2, sort_keys=True) + "\n"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, required=True)
    parser.add_argument("--evidence", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    render(args.results, args.evidence, args.out)


if __name__ == "__main__":
    main()
