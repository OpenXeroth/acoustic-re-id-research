"""Verify and publish all nineteen retrospective model recipes after the v6 gate."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

from paper_v6_controls import complete_gate
from paper_v6_evidence import BIO_GROUPS, ENDPOINTS

from xinyenyana.model_load_audit import audit_loads


def verify_group(
    group: str,
    record: dict[str, Any],
    results: Path,
    weights: dict[str, Any],
    evidence: dict[str, Any],
) -> dict[str, Any]:
    """Check the metadata against every measured endpoint, not just a sample."""
    names = BIO_GROUPS[group]
    if (
        record.get("retrospective") is not True
        or record.get("audio_read") is not False
        or set(record.get("models", {})) != set(names)
        or record.get("weight_inventory", {}).get("sha256") != weights["sha256"]
        or any(
            record["weight_inventory"].get(k) != weights[k]
            for k in ("verified_files", "verified_bytes", "all_match")
        )
    ):
        raise ValueError(f"{group}: incomplete retrospective model or weight coverage")
    expected = {f"v6-{group}-{endpoint}.json" for endpoint in ENDPOINTS}
    sources = record.get("sources", {})
    if len(sources) != len(expected) or {Path(p).name for p in sources} != expected:
        raise ValueError(f"{group}: every endpoint source is required")
    sweeps = {}
    for original, digest in sources.items():
        name = Path(original).name
        raw = (results / name).read_bytes()
        if hashlib.sha256(raw).hexdigest() != digest or evidence["sources"][name] != digest:
            raise ValueError(f"{name}: source differs from the complete evidence inventory")
        sweeps[original] = json.loads(raw)
        if not set(names) <= sweeps[original].get("curve", {}).keys():
            raise ValueError(f"{name}: missing model candidates")
    descriptions = record["models"]
    checked = audit_loads(
        names, sweeps, record["environment"]["packages"], lambda name: descriptions[name]
    )
    if checked["models"] != descriptions:
        raise ValueError(f"{group}: module mapping or measured coverage differs")
    return {
        name: {
            **{
                key: row[key]
                for key in ("source", "sample_rate", "window_samples", "candidate_to_module")
            },
            "window_seconds": row["window_samples"] / row["sample_rate"],
            "measured_endpoints": len(row["measured_candidate_sources"]),
        }
        for name, row in descriptions.items()
    }


def render(results: Path, weights_path: Path, evidence_path: Path, out: Path) -> None:
    from paper_figures import md_table

    evidence = complete_gate(evidence_path)
    raw = weights_path.read_bytes()
    files = json.loads(raw)["files"]
    if not files:
        raise ValueError("empty original weight inventory")
    weights = {
        "sha256": hashlib.sha256(raw).hexdigest(),
        "verified_files": len(files),
        "verified_bytes": sum(row["bytes"] for row in files.values()),
        "all_match": True,
    }
    models, sources, revisions = {}, {}, {}
    for group in BIO_GROUPS:
        path = results / "model-loads" / f"v6-model-loads-{group}.json"
        raw = path.read_bytes()
        record = json.loads(raw)
        models.update(verify_group(group, record, results, weights, evidence))
        sources[path.name] = hashlib.sha256(raw).hexdigest()
        revisions[group] = record["code_revision"]
    rows = [
        [
            name,
            row["source"],
            str(row["sample_rate"]),
            f"{row['window_seconds']:g}",
            str(len(row["candidate_to_module"])),
            str(row["measured_endpoints"]),
        ]
        for name, row in sorted(models.items())
    ]
    text = (
        "# Revision 6 model input recipes and provenance\n\n"
        "These nineteen records are retrospective reloads under the measured package inventories. "
        "They read no experiment audio and are not original execution traces. Every model's "
        "ordinal candidate list matches all fourteen endpoint sweeps, whose bytes also match "
        "the complete numerical evidence inventory. Named module mappings are in the linked "
        "ledger.\n\n"
        "Each reload first streamed the cache files and checked their sizes and SHA-256 digests "
        "against the original weight inventory. The inventory includes auxiliary files and unused "
        "cached releases; its file count is not a count of weights used by these models. "
        "This check does not identify every file accessed by an original extraction.\n\n"
        + md_table(["Model", "Adapter", "Input Hz", "Window seconds", "Blocks", "Endpoints"], rows)
        + "\n\nBlocks are additional token-averaged candidates; every model also supplies its "
        "published embedding. Window lengths describe the model recipe before clip-level "
        "averaging and final-window padding. Source hashes, the cache-inventory digest and "
        "candidate-to-module mappings are retained in "
        "[the provenance ledger](model-provenance-v6.json).\n"
    )
    out.mkdir(parents=True, exist_ok=True)
    (out / "model-provenance-v6.md").write_text(text)
    (out / "model-provenance-v6.json").write_text(
        json.dumps(
            {
                "retrospective": True,
                "sources": sources,
                "code_revisions": revisions,
                "weight_inventory": weights,
                "models": models,
            },
            indent=2,
            sort_keys=True,
        )
        + "\n"
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, required=True)
    parser.add_argument("--weights", type=Path, required=True)
    parser.add_argument("--evidence", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    render(args.results, args.weights, args.evidence, args.out)
