"""Public candidate/rate inventory, derived only from complete archived sweeps."""

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

from paper_v6_controls import complete_gate
from paper_v6_evidence import ALL_MODELS, ENDPOINTS

GROUPS = ("bio", "birdmae", "tf", "onnx", "avex", "xeus", "merged")


def endpoint_inventory(
    results: Path, endpoint: str, sources: dict[str, str]
) -> tuple[dict[str, Any], dict[str, str]]:
    rows = {}
    used = {}
    for group in (*GROUPS[:-1], "v5new" if endpoint in ENDPOINTS[-2:] else "merged"):
        name = f"v6-{group}-{endpoint}.json"
        raw = (results / name).read_bytes()
        digest = hashlib.sha256(raw).hexdigest()
        if sources[name] != digest:
            raise ValueError(f"{name}: bytes differ from the complete evidence inventory")
        used[name] = digest
        data = json.loads(raw)
        for model, curve in data["curve"].items():
            if model in rows:
                raise ValueError(f"{endpoint}, {model}: overlapping candidate inventories")
            chosen = data["chosen"][model]["representation"]
            if chosen not in curve:
                raise ValueError(f"{model}: Fisher choice is absent from the measured candidates")
            held = data.get("chosen_held_out", {}).get(model)
            held_name = None if held is None else held["representation"]
            if held_name is not None and held_name not in curve:
                raise ValueError(f"{model}: session choice is absent from the measured candidates")
            candidates = {
                name: {
                    key: entry.get(key)
                    for key in (
                        "layer",
                        "slowdown",
                        "read_in_windows_of_seconds",
                        "numeric_precision",
                    )
                }
                for name, entry in curve.items()
            }
            rows[model] = {
                "source": name,
                "candidate_count": len(candidates),
                "fisher_choice": chosen,
                "session_holdout_choice": held_name,
                "candidates": candidates,
            }
    if set(rows) != ALL_MODELS:
        raise ValueError(f"{endpoint}: model inventory differs: {set(rows) ^ ALL_MODELS}")
    return rows, used


def render(results: Path, evidence_path: Path, out: Path) -> None:
    evidence = complete_gate(evidence_path)
    endpoints = {}
    sources = {}
    for ep in ENDPOINTS:
        endpoints[ep], used = endpoint_inventory(results, ep, evidence["sources"])
        sources.update(used)
    from paper_figures import md_table

    table = []
    for model in sorted(ALL_MODELS):
        entries = [v[model] for v in endpoints.values()]
        counts = [r["candidate_count"] for r in entries]
        candidates = [c for r in entries for c in r["candidates"].values()]
        rates = sorted({c["slowdown"] for c in candidates if c["slowdown"] is not None})
        windows = sorted(
            {
                c["read_in_windows_of_seconds"]
                for c in candidates
                if c["read_in_windows_of_seconds"] is not None
            }
        )
        count = str(min(counts)) if min(counts) == max(counts) else f"{min(counts)}–{max(counts)}"
        table.append(
            [
                model.split("/")[-1],
                count,
                ", ".join(str(x) for x in rates) or "Not varied",
                ", ".join(str(x) for x in windows) or "None",
                str(sum(r["session_holdout_choice"] is not None for r in entries)),
            ]
        )
    text = (
        "# Candidate and playback-rate coverage\n\nCounts span all fourteen endpoints, "
        "including the supplementary expanded great tit. They count measured candidates, "
        "not independent models. Playback factors and long-clip windowing refer to the "
        "speech extraction path; fixed input windows of added audio adapters are specified "
        "in the model-provenance table. An unavailable session-holdout choice remains "
        "unavailable. The numerical ledger lists every measured candidate and both selected"
        " representations by endpoint, without query or identity annotations.\n\n"
    )
    text += md_table(
        [
            "Model/control",
            "Candidates per endpoint",
            "Playback factors",
            "Long-clip windows (seconds)",
            "Endpoints with session choice",
        ],
        table,
    )
    text += (
        "\n\nAll endpoint-specific candidates, precision metadata and source digests are in "
        "[the coverage ledger](candidate-coverage.json). This inventory does not assert "
        "that all theoretical layers, playback rates or architectures were tested.\n"
    )
    out.mkdir(parents=True, exist_ok=True)
    (out / "candidate-coverage.md").write_text(text)
    (out / "candidate-coverage.json").write_text(
        json.dumps({"sources": sources, "endpoints": endpoints}, indent=2, sort_keys=True) + "\n"
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
