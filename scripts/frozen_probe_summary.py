"""Does the registered selector beat the final embedding? The table in measured.md.

Reads `run-frozen-probe` result files and prints, for each endpoint, the candidate
the enrolment-Fisher selector chose and how it scores against `embedding.mean`,
the reading this project already uses, on each of the three heads and on the
foreground-to-foreground pairing.

    python scripts/frozen_probe_summary.py results/frozen-*.json
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

HEADS = (
    "kernel ridge to one-hot",
    "mean cosine similarity to each identity",
    "single nearest enrollment clip, cosine",
)
PAIRING = "foreground_to_foreground"
BASELINE = "embedding.mean"
TOLERANCE = 1e-12


def accuracy(result: dict[str, object], candidate: str, head: str) -> float:
    heads = result["candidates"][candidate]["heads"]  # type: ignore[index]
    return float(heads[head][PAIRING]["classification"]["accuracy"])


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("results", nargs="+", type=Path)
    arguments = parser.parse_args()
    results = [json.loads(path.read_text()) for path in arguments.results]
    results.sort(key=lambda result: str(result["endpoint"]))

    verdicts: Counter[str] = Counter()
    print(
        f"{'endpoint':<24} {'selected':<18} "
        f"{'ridge sel/emb':>19} {'mean cos sel/emb':>19} {'near clip sel/emb':>19}"
    )
    for result in results:
        selected = str(result["selected"])
        cells = []
        for head in HEADS:
            chosen = accuracy(result, selected, head)
            baseline = accuracy(result, BASELINE, head)
            if abs(chosen - baseline) <= TOLERANCE:
                verdicts["equal"] += 1
            elif chosen > baseline:
                verdicts["selector higher"] += 1
            else:
                verdicts["selector lower"] += 1
            cells.append(f"{chosen:.4f}/{baseline:.4f}")
        print(f"{result['endpoint']:<24} {selected:<18} " + " ".join(f"{c:>19}" for c in cells))
    total = sum(verdicts.values())
    print(f"\n{total} endpoint-by-head cells on {PAIRING}")
    for verdict in ("selector lower", "equal", "selector higher"):
        print(f"  {verdict}: {verdicts[verdict]}")
    counted = sorted({int(r["distinct_candidates"]) for r in results if "distinct_candidates" in r})
    keys = sorted({int(r["candidate_keys"]) for r in results if "candidate_keys" in r})
    missing = [str(r["endpoint"]) for r in results if "distinct_candidates" not in r]
    print(f"\ndistinct candidates {counted} of {keys} extracted keys")
    if missing:
        print(
            "  not recorded, these ran before the divisor counted candidates: " + ", ".join(missing)
        )


if __name__ == "__main__":
    main()
