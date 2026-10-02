"""A paired additive-background challenge with the gallery held fixed.

This first executable arm measures the frozen BirdNET embedding and the two
published nuisance controls. It neither separates sources nor measures the
speech encoders. Original enrollment audio is used in every arm.
"""

from __future__ import annotations

import sys
import tempfile
from dataclasses import replace
from pathlib import Path
from typing import Any

from xinyenyana.a2 import Endpoint
from xinyenyana.a5 import BIRDNET, SHORTCUTS, shortcut_vectors
from xinyenyana.archive import sha256_file
from xinyenyana.background_challenge import (
    CHALLENGE_DB,
    PAIRING_SALT,
    background_pairs,
    mix_background,
)
from xinyenyana.birdnet_reader import BirdNETReader
from xinyenyana.evaluation import (
    KERNEL_RIDGE,
    evaluate_endpoint,
    paired_difference_interval,
    standardisation_for,
)
from xinyenyana.extraction_cache import checkpointed_interrupts
from xinyenyana.frozen_probe import endpoint_audio_provenance, runtime_record


def evaluate_background_challenge(
    endpoint: Endpoint,
    original: dict[str, Any],
    challenged: dict[str, dict[str, Any]],
    *,
    permutations: int = 999,
    bootstrap_replicates: int = 2000,
) -> dict[str, Any]:
    """Evaluate aligned query-only challenges against unchanged enrollment vectors."""

    import numpy as np

    enrollment = [
        i
        for i, r in enumerate(endpoint.records)
        if r.split == "enrollment" and r.context.get("condition", "foreground") == "foreground"
    ]
    query = [
        i
        for i, r in enumerate(endpoint.records)
        if r.split == "query" and r.context.get("condition", "foreground") == "foreground"
    ]
    records = [endpoint.records[i].as_evaluation_record() for i in enrollment + query]
    results: dict[str, Any] = {}
    for representation, values in sorted(original.items()):
        train = values[enrollment]
        arms = {"original": values[query]}
        for arm, vectors in challenged.items():
            if vectors[representation].shape != values[query].shape:
                raise ValueError("a challenge must retain every foreground query in original order")
            arms[arm] = vectors[representation]
        evaluated = {}
        for arm, matrix in arms.items():
            evaluated[arm] = evaluate_endpoint(
                records=records,
                vectors=np.vstack((train, matrix)),
                representation=representation,
                enrollment_condition="foreground",
                query_condition="foreground",
                ridge_lambda=1.0,
                seed=17,
                permutations=permutations,
                bootstrap_replicates=bootstrap_replicates,
                standardisation=standardisation_for(train.shape[1]),
                head=KERNEL_RIDGE,
            )
        paired = {}
        for ratio in CHALLENGE_DB:
            own, other = evaluated[f"own_{ratio:g}db"], evaluated[f"other_{ratio:g}db"]
            own_predictions = own["predictions"]
            other_predictions = other["predictions"]
            paired[f"{ratio:g}db"] = {
                "own_minus_other_macro_recall": (
                    own["classification"]["macro_recall"] - other["classification"]["macro_recall"]
                ),
                "identity_block_difference_95": paired_difference_interval(
                    own, other, seed=17, replicates=bootstrap_replicates
                ),
                "answer_changed_fraction": float(
                    np.mean(
                        [
                            a["predicted_label"] != b["predicted_label"]
                            for a, b in zip(own_predictions, other_predictions, strict=True)
                        ]
                    )
                ),
            }
        results[representation] = {"arms": evaluated, "own_vs_other": paired}
    return {
        "experiment": "PA-BACKGROUND-CHALLENGE",
        "endpoint": endpoint.name,
        "manifest_sha256": endpoint.manifest_sha256,
        "splits": endpoint.split_digests(),
        "config": {
            "head": KERNEL_RIDGE,
            "ridge_lambda": 1.0,
            "seed": 17,
            "permutations": permutations,
            "bootstrap_replicates": bootstrap_replicates,
            "foreground_over_added_background_db": list(CHALLENGE_DB),
            "pairing_salt": PAIRING_SALT,
            "enrollment": "unaltered original foreground in every arm",
            "sham": "foreground alone, peak scaled to 0.95 and encoded as PCM16",
        },
        "representations": results,
        "claim_boundary": (
            "Added-background sensitivity, not source separation or true vocal SNR; "
            "original recordings already contain ambient sound. Paired intervals are "
            "descriptive, without a family-wide rejection gate."
        ),
    }


def _read_mono(path: Path) -> tuple[Any, int]:
    import numpy as np
    import soundfile

    signal, rate = soundfile.read(str(path), dtype="int16")
    signal = np.asarray(signal, dtype=np.float32) / 32768.0
    return (signal.mean(axis=1) if signal.ndim == 2 else signal), int(rate)


def run_background_challenge(
    endpoint: Endpoint, *, cache_root: Path, threads: int
) -> dict[str, Any]:
    import numpy as np
    import soundfile
    from scipy.signal import resample

    data = endpoint_audio_provenance(endpoint)
    pairs = [p for p in background_pairs(endpoint.records) if p.foreground.split == "query"]
    reader = BirdNETReader(cache_root, threads=threads)
    original = {key: shortcut_vectors(endpoint.records, key) for key in SHORTCUTS}
    original[BIRDNET] = []
    challenged: dict[str, dict[str, list[Any]]] = {}
    donor_records: list[dict[str, Any]] = []
    with (
        checkpointed_interrupts(),
        tempfile.TemporaryDirectory(prefix="background-waveforms-", dir=cache_root) as work,
    ):
        temporary_audio = Path(work) / "challenge.wav"
        for record in endpoint.records:
            original[BIRDNET].append(reader.read(record.path)["embedding.mean"])
        original[BIRDNET] = np.vstack(original[BIRDNET])
        for index, pair in enumerate(pairs):
            foreground, rate = _read_mono(pair.foreground.path)
            peak = float(np.max(np.abs(foreground)))
            if peak == 0:
                raise ValueError("a silent foreground has no defined challenge ratio")
            waves = {"sham_peak_scaled": foreground * (0.95 / peak)}
            detail: dict[str, Any] = {
                "foreground": pair.foreground.filename,
                "foreground_sha256": sha256_file(pair.foreground.path),
                "split": pair.foreground.split,
                "sample_rate_hz": rate,
                "sample_count": len(foreground),
                "donors": {},
            }
            for kind, donor in (("own", pair.own), ("other", pair.other)):
                ambient, background_rate = _read_mono(donor.path)
                if background_rate != rate:
                    ambient = resample(ambient, round(len(ambient) * rate / background_rate))
                detail["donors"][kind] = {
                    "filename": donor.filename,
                    "sha256": sha256_file(donor.path),
                    "split": donor.split,
                    "same_identity": donor.identity == pair.foreground.identity,
                    "source_rate_hz": background_rate,
                    "resampled_samples": len(ambient),
                    "repeated": len(ambient) < len(foreground),
                    "repetitions_needed": int(np.ceil(len(foreground) / len(ambient))),
                }
                for ratio in CHALLENGE_DB:
                    waves[f"{kind}_{ratio:g}db"] = mix_background(
                        foreground, ambient, foreground_over_background_db=ratio
                    )
            for arm, samples in waves.items():
                soundfile.write(str(temporary_audio), samples, rate, subtype="PCM_16")
                generated = replace(pair.foreground, path=temporary_audio)
                vectors = {BIRDNET: reader.read(temporary_audio)["embedding.mean"]}
                vectors.update({key: shortcut_vectors([generated], key)[0] for key in SHORTCUTS})
                for representation, vector in vectors.items():
                    challenged.setdefault(arm, {}).setdefault(representation, []).append(vector)
            donor_records.append(detail)
            if index % 25 == 0 or index + 1 == len(pairs):
                print(
                    f"background challenges {index + 1}/{len(pairs)}", file=sys.stderr, flush=True
                )
    matrices = {
        arm: {name: np.vstack(rows) for name, rows in representations.items()}
        for arm, representations in challenged.items()
    }
    result = evaluate_background_challenge(endpoint, original, matrices)
    return {
        **result,
        "extraction": reader.specification,
        "environment": runtime_record(),
        "donor_pairs": donor_records,
        "data": data,
    }
