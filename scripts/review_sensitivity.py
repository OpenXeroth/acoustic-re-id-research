"""Run the fixed review protocol on xen1's retained vectors, without inference.

Outputs contain private per-clip annotations: archive them, do not commit them.
"""

from __future__ import annotations

import argparse
import importlib.metadata
import json
import os
import re
import subprocess
import sys
from dataclasses import replace
from datetime import datetime
from pathlib import Path

import numpy as np

from xinyenyana.a6 import load_vectors
from xinyenyana.archive import archive_result, canonical_sha256, sha256_file, source_digest
from xinyenyana.cli import _load_endpoint
from xinyenyana.cosine_controls import is_control
from xinyenyana.huang import load_published_vectors
from xinyenyana.identity_run import balance_enrolment
from xinyenyana.open_set import evaluate_allocation, summarise_allocations
from xinyenyana.review_sensitivity import SPLIT_SEEDS, matched_rows, score_split, split_summary

REPO = Path(__file__).resolve().parents[1]


def save(path, data):
    data.update(
        {
            "code_revision": subprocess.check_output(
                ["git", "rev-parse", "HEAD"], cwd=REPO, text=True
            ).strip(),
            "source_sha256": source_digest(REPO / "src/xinyenyana"),
            "runner_sha256": sha256_file(Path(__file__)),
            "protocol_sha256": sha256_file(REPO / "docs/experiments/review-sensitivity.md"),
            "lock_sha256": sha256_file(REPO / "uv.lock"),
            "environment": {
                "python": sys.version,
                "packages": {
                    d.metadata["Name"]: d.version for d in importlib.metadata.distributions()
                },
                "lease": os.environ.get("XENWARDEN_LEASE"),
            },
        }
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n")
    receipt = archive_result(path, prefix=os.environ["XINYENYANA_RESULT_ARCHIVE"])
    path.with_suffix(".archive.json").write_text(json.dumps(receipt, indent=2) + "\n")
    if not receipt["archived"]:
        raise RuntimeError(receipt)
    print(path.name, receipt["sha256"], flush=True)


def split_runs(root, out):
    source = root / "benchmarks/bird-aiid-zenodo-17576155/unpacked/dataset"
    for species in ("chiffchaff", "littleowl", "pipit"):
        for model in ("birdnet", "google-perch"):
            target = out / f"matched-{species}-{model}.json"
            if target.with_suffix(".archive.json").exists():
                continue
            ep, vectors = load_published_vectors(
                species=species, model=model, split="published", root=source
            )
            records = ep.records
            train = [i for i, r in enumerate(records) if r.split == "enrollment"]
            query = [i for i, r in enumerate(records) if r.split == "query"]
            baseline = score_split(records, vectors, train, query)
            runs = []
            for seed in SPLIT_SEEDS:
                ti, qi = matched_rows(records, seed)
                runs.append({"seed": seed, **score_split(records, vectors, ti, qi)})
                print("split", species, model, seed, flush=True)
            save(
                target,
                {
                    "experiment": "review-count-matched",
                    "species": species,
                    "model": model,
                    "manifest_sha256": ep.manifest_sha256,
                    "clips": len(records),
                    "identities": len(ep.identities),
                    "enrollment": len(train),
                    "query": len(query),
                    "clip_names": [r.filename for r in records],
                    "baseline": baseline,
                    "random_splits": runs,
                    "summary": split_summary(baseline, runs),
                },
            )


def endpoint(root, name):
    special = {
        "great-tit": root / "a1/e03s/sample",
        "great-tit-full": root / "v6run/samples/great-tit-full",
        "littleowl-acrossyear": root / "benchmarks/stowell-2019-zenodo-1413495/unpacked",
        "cockatoo-fold1": root / "benchmarks/bird-aiid-zenodo-17576155/unpacked/dataset",
    }
    return _load_endpoint(name, special.get(name, root / "samples" / name))


def open_runs(root, out):
    folder = root / "v6-corrected-20260926/results/v6-counts"
    for name in (
        "great-tit",
        "littleowl-acrossyear",
        "rookid-full-width",
        "great-tit-full",
        "cockatoo-fold1",
    ):
        ep = endpoint(root, name)
        if name == "rookid-full-width":
            ep = replace(ep, records=tuple(balance_enrolment(ep.records, 90)))
        source = folder / f"v6-open-set-{name}.json"
        original = json.loads(source.read_text())
        assert original["manifest_sha256"] == ep.manifest_sha256
        assert original["splits"] == ep.split_digests()
        rows = [
            i
            for i, r in enumerate(ep.records)
            if r.context.get("condition", "foreground") == "foreground"
        ]
        records = [ep.records[i] for i in rows]
        for model, old in original["representations"].items():
            if is_control(model):
                continue
            target = out / f"open-{name}-{model.replace(chr(47), chr(95))}.json"
            if target.with_suffix(".archive.json").exists():
                continue
            entry = old["vector_source"]
            p = Path(entry["path"])
            assert sha256_file(p) == entry["sha256"]
            matrix, _, representation = load_vectors(p, ep)
            vectors = matrix[rows]
            roles = [a["roles"] for a in old["allocations"]]
            results = {}
            for scorer in ("mean-profile", "nearest-clip"):
                allocations = [
                    evaluate_allocation(
                        vectors=vectors,
                        records=records,
                        roles=role,
                        standardise=False,
                        balanced=name == "great-tit",
                        scorer=scorer,
                    )
                    for role in roles
                ]
                if scorer == "mean-profile":
                    for a, b in zip(allocations, old["allocations"], strict=True):
                        assert abs(a["closed_set_accuracy"] - b["closed_set_accuracy"]) < 1e-12
                        for budget in ("0.10",):
                            assert (
                                abs(
                                    a["calibrated_on_other_birds"][budget]["test"][
                                        "known_correct_accept_rate"
                                    ]
                                    - b["calibrated_on_other_birds"][budget]["test"][
                                        "known_correct_accept_rate"
                                    ]
                                )
                                < 1e-12
                            )
                results[scorer] = {
                    "summary": summarise_allocations(allocations),
                    "allocations": allocations,
                }
            calibration = {}
            if name == "great-tit-full" and model in (
                "birdnet",
                "birdnet-v2.4",
                "perch",
                "perch-v2",
                "perch-2.0",
            ):
                for scorer in results:
                    calibration[scorer] = {}
                    for count in (4, 8, 12):
                        allocations = [
                            evaluate_allocation(
                                vectors=vectors,
                                records=records,
                                roles={
                                    **role,
                                    "calibration_unknown": sorted(role["calibration_unknown"])[
                                        :count
                                    ],
                                },
                                standardise=False,
                                balanced=False,
                                scorer=scorer,
                            )
                            for role in roles
                        ]
                        calibration[scorer][str(count)] = {
                            "summary": summarise_allocations(allocations),
                            "allocations": allocations,
                        }
            save(
                target,
                {
                    "experiment": "review-open-set-scorers",
                    "endpoint": name,
                    "model": model,
                    "representation": representation,
                    "source_sha256_original": sha256_file(source),
                    "vectors_sha256": entry["sha256"],
                    "manifest_sha256": ep.manifest_sha256,
                    "role_digest": original["role_digest"],
                    "scorers": results,
                    "calibration_size": calibration,
                },
            )


def paired_runs(root, out):
    for name, filename in (
        ("birdpark-juv01", "birdpark-juv01"),
        ("birdpark-juv03", "birdpark-juv03"),
        ("rookid-full-width", "rook-full-across-year"),
    ):
        source = root / "v6run/results" / f"v6-headline-{filename}.json"
        d = json.loads(source.read_text())
        ep = endpoint(root, name)
        records = ep.records
        assert ep.manifest_sha256 == d["manifest_sha256"]
        q = [r for r in records if r.split == "query"]
        assert [r.identity for r in q] == d["query_identities"]
        cache = root / "phase-a-20260913/cache/birdnet" / canonical_sha256(d["extraction"])
        hashes = {r["filename"]: r["sha256"] for r in d["data"]["audio_files"]}
        matrix = []
        cache_hashes = []
        days = []
        for r in q:
            path = cache / (hashes[r.filename] + ".npz")
            with np.load(path, allow_pickle=False) as z:
                matrix.append(z["embedding.mean"])
            cache_hashes.append(sha256_file(path))
            date = str(r.context.get("recorded", ""))
            if not date:
                match = re.search(r"\d{4}-\d{2}-\d{2}", str(r.session))
                if match:
                    date = match[0]
            days.append(
                datetime.strptime(date, "%Y-%m-%d" if "-" in date else "%Y%m%d").toordinal()
            )
        values = np.array(matrix, dtype=float)
        values /= np.maximum(np.linalg.norm(values, axis=1, keepdims=True), 1e-12)
        labels = np.array([r.identity for r in q])
        sessions = np.array([r.session for r in q])
        days = np.array(days)
        aggregates = {
            key: [0, 0.0, 0.0]
            for key in (
                "recording",
                "animal_all",
                "animal_same_day",
                "animal_1_to_7_days",
                "animal_over_7_days",
            )
        }
        # Streaming exact pair means avoids quadratic RAM and random pair subsampling.
        for i in range(len(q) - 1):
            scores = values[i + 1 :] @ values[i]
            same = labels[i + 1 :] == labels[i]
            shared = sessions[i + 1 :] == sessions[i]
            gaps = abs(days[i + 1 :] - days[i])
            animal = same & ~shared
            masks = {
                "recording": ~same & shared,
                "animal_all": animal,
                "animal_same_day": animal & (gaps == 0),
                "animal_1_to_7_days": animal & (gaps >= 1) & (gaps <= 7),
                "animal_over_7_days": animal & (gaps > 7),
            }
            for key, mask in masks.items():
                selected = scores[mask]
                a = aggregates[key]
                a[0] += len(selected)
                a[1] += float(selected.sum())
                a[2] += float((selected**2).sum())
        summary = {
            k: {"pairs": int(n), "mean_cosine": total / n if n else None}
            for k, (n, total, _) in aggregates.items()
        }
        save(
            out / f"paired-time-{name}.json",
            {
                "experiment": "review-paired-time",
                "endpoint": name,
                "source_sha256_original": sha256_file(source),
                "cached_vectors_sha256": canonical_sha256(cache_hashes),
                "manifest_sha256": ep.manifest_sha256,
                "summary": summary,
                "query_metadata": [
                    {"identity": r.identity, "session": r.session, "day": int(day)}
                    for r, day in zip(q, days, strict=True)
                ],
                "pair_rule": "All unordered query pairs; descriptive means only",
            },
        )


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("mode", choices=("split", "open", "paired"))
    p.add_argument("--root", type=Path, default=Path("/mnt/data/xinyenyana"))
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    if not os.environ.get("XINYENYANA_RESULT_ARCHIVE"):
        p.error("XINYENYANA_RESULT_ARCHIVE must be set")
    {"split": split_runs, "open": open_runs, "paired": paired_runs}[args.mode](
        args.root, args.output
    )


if __name__ == "__main__":
    main()
