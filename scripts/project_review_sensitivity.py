"""Publish whitelisted review aggregates, retaining verified raw-result hashes."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from statistics import median


def score_summary(value):
    summary = dict(value["summary"])
    for budget in ("0.01", "0.05", "0.10", "0.20", "0.50"):
        rates = [
            a["calibrated_on_other_birds"][budget]["test"]["unknown_false_accept_rate"]
            for a in value["allocations"]
        ]
        summary["unknown_false_accept_rate_calibrated_at_" + budget] = {
            "median": median(rates),
            "minimum": min(rates),
            "maximum": max(rates),
        }
        summary["allocations_exceeding_unknown_budget_at_" + budget] = sum(
            rate > float(budget) + 1e-12 for rate in rates
        )
    return summary


def project(source: Path, output: Path) -> None:
    output.mkdir(parents=True, exist_ok=True)
    manifest = {}
    for receipt_path in sorted(source.glob("*.archive.json")):
        raw = receipt_path.with_name(receipt_path.name.replace(".archive.json", ".json"))
        receipt = json.loads(receipt_path.read_text())
        content = raw.read_bytes()
        digest = hashlib.sha256(content).hexdigest()
        if not receipt.get("archived") or receipt.get("sha256") != digest:
            raise ValueError(f"Unverified archive: {raw.name}")
        d = json.loads(content)
        public = {
            k: d[k]
            for k in (
                "experiment",
                "species",
                "endpoint",
                "model",
                "representation",
                "clips",
                "identities",
                "enrollment",
                "query",
                "code_revision",
                "source_sha256",
                "runner_sha256",
                "protocol_sha256",
                "lock_sha256",
                "environment",
                "manifest_sha256",
                "source_sha256_original",
                "role_digest",
                "vectors_sha256",
                "cached_vectors_sha256",
                "pair_rule",
                "summary",
            )
            if k in d
        }
        public["original_result_sha256"] = digest
        if "scorers" in d:
            public["scorers"] = {k: score_summary(v) for k, v in d["scorers"].items()}
            public["calibration_size"] = {
                scorer: {count: score_summary(v) for count, v in values.items()}
                for scorer, values in d["calibration_size"].items()
            }
        if "baseline" in d:
            public["cosine_medians"] = {
                "session": d["baseline"]["nearest_same_individual_cosine_median"],
                "random": [r["nearest_same_individual_cosine_median"] for r in d["random_splits"]],
            }
            public["split_hashes"] = {
                "session": d["baseline"]["split_sha256"],
                "random": {str(r["seed"]): r["split_sha256"] for r in d["random_splits"]},
            }
            public["per_seed"] = [
                {
                    "seed": r["seed"],
                    "heads": {
                        h: {k: v[k] for k in ("accuracy", "macro_recall")}
                        for h, v in r["heads"].items()
                    },
                }
                for r in d["random_splits"]
            ]
        path = output / raw.name
        path.write_text(json.dumps(public, indent=2, sort_keys=True) + "\n")
        manifest[raw.name] = {
            "raw_sha256": digest,
            "projection_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            "archive_verified": True,
        }
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"Projected {len(manifest)} verified results")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    project(args.source, args.output)
