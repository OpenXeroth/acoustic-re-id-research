"""The frozen single-layer and pooling comparison preceding a learned probe.

Every candidate is reported. The named selection uses enrolment Fisher ratio
only; a query accuracy is never used to choose a layer or pooling operation.
Both predictions and background-control results are retained for the paper.
"""

from __future__ import annotations

import importlib.metadata
import os
import platform
import sys
from pathlib import Path
from typing import Any

from xinyenyana.a2 import Endpoint
from xinyenyana.a5 import fisher_ratio
from xinyenyana.archive import canonical_sha256, sha256_file
from xinyenyana.birdnet_reader import BirdNETReader
from xinyenyana.evaluation import HEADS, PER_CLIP_L2, evaluate_endpoint
from xinyenyana.extraction_cache import checkpointed_interrupts
from xinyenyana.representations import BIRDNET_LAYERS


def runtime_record() -> dict[str, Any]:
    return {
        "hostname": platform.node(),
        "python": sys.version,
        "executable": sys.executable,
        "uid": os.getuid(),
        "lease": os.environ.get("XENWARDEN_LEASE"),
        "packages": {
            distribution.metadata["Name"]: distribution.version
            for distribution in importlib.metadata.distributions()
            if distribution.metadata["Name"]
        },
    }


def endpoint_audio_provenance(endpoint: Endpoint) -> dict[str, Any]:
    """Bind a run to all input audio bytes, including its enrollment recordings."""

    files = [
        {"filename": record.filename, "sha256": sha256_file(record.path)}
        for record in sorted(endpoint.records, key=lambda record: record.filename)
    ]
    return {"audio_files": files, "audio_files_sha256": canonical_sha256(files)}


def extract_endpoint(
    endpoint: Endpoint, *, cache_root: Path, threads: int
) -> tuple[dict[str, Any], dict[str, Any]]:
    import numpy as np

    reader = BirdNETReader(cache_root, threads=threads)
    vectors: dict[str, list[Any]] = {}
    with checkpointed_interrupts():
        for index, record in enumerate(endpoint.records):
            extracted = reader.read(record.path)
            for key, value in extracted.items():
                if not key.endswith(".sequence"):
                    vectors.setdefault(key, []).append(value)
            if index % 50 == 0 or index + 1 == len(endpoint.records):
                print(
                    f"extracted {index + 1}/{len(endpoint.records)} clips",
                    file=sys.stderr,
                    flush=True,
                )
    return {key: np.vstack(values) for key, values in vectors.items()}, reader.specification


def distinct_candidates(vectors: dict[str, Any]) -> dict[str, list[str]]:
    """The candidate keys grouped by the test they actually run.

    Three of the fourteen keys this project extracts carry one representation.
    `post_convolution.mean` is the same vector as `embedding.mean`, because
    averaging the post-convolution grid over time and frequency is what the
    network's own global average pool does. `embedding.mean_std` is
    `embedding.mean` followed by 1,024 zeros, because the network has already
    pooled that tensor and there is no time axis left to take a spread over.
    Checked on 800 clips drawn at random from the extraction cache on
    2026-09-14: maximum absolute difference 0.0, no exceptions.

    Candidates are grouped after dropping the columns that are zero for every
    clip, which is the only rule needed to catch both cases and which none of
    the three heads can tell apart from the shorter vector.

    The grouping decides the significance correction: dividing by fourteen when
    twelve tests were run is a correction for tests that were never made.
    """

    import numpy as np

    groups: dict[str, list[str]] = {}
    signatures: dict[bytes, str] = {}
    for key in sorted(vectors):
        matrix = np.ascontiguousarray(np.asarray(vectors[key], dtype=np.float64))
        carried = matrix[:, np.any(matrix != 0.0, axis=0)]
        signature = np.ascontiguousarray(carried).tobytes()
        first = signatures.setdefault(signature, key)
        groups.setdefault(first, []).append(key)
    return groups


def evaluate_frozen_layers(
    endpoint: Endpoint,
    vectors: dict[str, Any],
    *,
    permutations: int = 999,
    bootstrap_replicates: int = 2000,
) -> dict[str, Any]:
    enrollment = [
        i
        for i, record in enumerate(endpoint.records)
        if record.split == "enrollment"
        and record.context.get("condition", "foreground") == "foreground"
    ]
    labels = [endpoint.records[i].identity for i in enrollment]
    fisher = {key: fisher_ratio(matrix[enrollment], labels) for key, matrix in vectors.items()}
    selection = sorted(fisher, key=lambda key: (-fisher[key], key))[0]
    groups = distinct_candidates(vectors)
    correction = len(groups)
    conditions = sorted({r.context.get("condition", "foreground") for r in endpoint.records})
    pairings = [("foreground", "foreground")]
    if "background" in conditions:
        pairings.extend(
            [
                ("background", "background"),
                ("foreground", "background"),
                ("background", "foreground"),
            ]
        )
    rows: dict[str, Any] = {}
    for key, matrix in sorted(vectors.items()):
        rows[key] = {"enrollment_fisher_ratio": fisher[key], "heads": {}}
        for head in HEADS:
            rows[key]["heads"][head] = {}
            for train_condition, query_condition in pairings:
                rows[key]["heads"][head][f"{train_condition}_to_{query_condition}"] = (
                    evaluate_endpoint(
                        records=[r.as_evaluation_record() for r in endpoint.records],
                        vectors=matrix,
                        representation=f"birdnet-layers:{key}",
                        enrollment_condition=train_condition,
                        query_condition=query_condition,
                        ridge_lambda=1.0,
                        seed=17,
                        permutations=permutations,
                        bootstrap_replicates=bootstrap_replicates,
                        standardisation=PER_CLIP_L2,
                        head=head,
                    )
                )
    return {
        "experiment": "PA-FROZEN-LAYERS",
        "endpoint": endpoint.name,
        "manifest_sha256": endpoint.manifest_sha256,
        "splits": endpoint.split_digests(),
        "config": {
            "layers": list(BIRDNET_LAYERS),
            "pooling": ["mean", "mean_std"],
            "ridge_lambda": 1.0,
            "seed": 17,
            "permutations": permutations,
            "bootstrap_replicates": bootstrap_replicates,
            "selection": "largest enrolment Fisher ratio; alphabetical key breaks ties",
            "significance_correction": (
                "the number of distinct candidates, counted after dropping columns that are "
                "zero for every clip, not the number of extracted keys"
            ),
        },
        "selected": selection,
        "candidate_keys": len(vectors),
        "distinct_candidates": correction,
        "candidate_groups": {first: members for first, members in sorted(groups.items())},
        "selected_p_bonferroni_over_candidates": {
            head: {
                pairing: min(
                    1.0,
                    value["permutation_control"]["p_value_plus_one"] * correction,
                )
                for pairing, value in rows[selection]["heads"][head].items()
            }
            for head in HEADS
        },
        "candidates": rows,
        "claim_boundary": (
            "Frozen individual layers and pooling only; the learned multi-layer combination "
            "and sequence/track models are separate experiments."
        ),
    }


def run_frozen_probe(
    endpoint: Endpoint,
    *,
    cache_root: Path,
    threads: int,
    permutations: int = 999,
    bootstrap_replicates: int = 2000,
) -> dict[str, Any]:
    data = endpoint_audio_provenance(endpoint)
    vectors, extraction = extract_endpoint(endpoint, cache_root=cache_root, threads=threads)
    return {
        **evaluate_frozen_layers(
            endpoint,
            vectors,
            permutations=permutations,
            bootstrap_replicates=bootstrap_replicates,
        ),
        "extraction": extraction,
        "environment": runtime_record(),
        "data": data,
    }
