"""The recording-context controls for any representation, from stored vectors.

v5 ran the background controls with BirdNET (and, on one endpoint, one speech
encoder). Cauzinille et al. (2024, Interspeech, doi
10.21437/Interspeech.2024-1096) found on gibbons that bird classifiers named
individuals from call-free clips of their territory far more readily than
speech encoders did. PA-V6 therefore runs both controls for every
representation on the five endpoints that carry ambient recordings:

* the four gallery-to-query pairings: calls on calls, backgrounds on
  backgrounds, a gallery of calls scored on backgrounds and a gallery of
  backgrounds scored on calls;
* the added-background challenge: every scoring call mixed with a background
  from its own territory and from another's, at -10, 0 and +10 dB, with the
  gallery unchanged (``background_probe.evaluate_background_challenge``).

The mixed waveforms are written once per endpoint and shared by every model,
so every representation is challenged with byte-identical audio. The
volume-only sham of v5 is not repeated.
"""

from __future__ import annotations

import os
from collections.abc import Sequence
from dataclasses import replace
from pathlib import Path
from typing import Any

from xinyenyana.a2 import ClipRecord, Endpoint
from xinyenyana.background_challenge import CHALLENGE_DB, background_pairs, mix_background

CONDITIONS = ("foreground", "background")
#: Every interval PA-V6 reports resamples the animals this many times (raised
#: from 2,000 in review, before any PA-V6 run).
REPLICATES = 10_000
ARMS: tuple[str, ...] = tuple(
    f"{kind}_{ratio:g}db" for kind in ("own", "other") for ratio in CHALLENGE_DB
)


def has_backgrounds(endpoint: Endpoint) -> bool:
    return any(str(r.context.get("condition")) == "background" for r in endpoint.records)


def query_foreground(endpoint: Endpoint) -> list[ClipRecord]:
    return [
        r
        for r in endpoint.records
        if r.split == "query" and str(r.context.get("condition", "foreground")) == "foreground"
    ]


def write_arms(endpoint: Endpoint, scratch: Path) -> dict[str, list[ClipRecord]]:
    """One mixed WAV per scoring call per arm, in scoring-call order, written once."""

    import numpy as np
    import soundfile
    from scipy.signal import resample

    from xinyenyana.background_probe import _read_mono

    pairs = {p.foreground.filename: p for p in background_pairs(endpoint.records)}
    arms: dict[str, list[ClipRecord]] = {arm: [] for arm in ARMS}
    for record in query_foreground(endpoint):
        pair = pairs[record.filename]
        foreground, rate = _read_mono(record.path)
        for kind, donor in (("own", pair.own), ("other", pair.other)):
            ambient = None
            for ratio in CHALLENGE_DB:
                arm = f"{kind}_{ratio:g}db"
                path = scratch / arm / f"{record.filename}.wav"
                if not path.exists():
                    if ambient is None:
                        ambient, donor_rate = _read_mono(donor.path)
                        if donor_rate != rate:
                            ambient = resample(ambient, round(len(ambient) * rate / donor_rate))
                    path.parent.mkdir(parents=True, exist_ok=True)
                    mixed = mix_background(
                        np.asarray(foreground), ambient, foreground_over_background_db=ratio
                    )
                    # Written beside and then renamed, so a run started in
                    # parallel on the same endpoint never reads a partial file.
                    partial = path.with_name(f".{path.name}.{os.getpid()}.partial.wav")
                    soundfile.write(str(partial), mixed, rate, subtype="PCM_16")
                    os.replace(partial, path)
                arms[arm].append(replace(record, path=path))
    return arms


def four_pairings(
    *,
    endpoint: Endpoint,
    vectors: Any,
    seed: int = 17,
    permutations: int = 999,
    retained_predictions: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Every gallery condition against every query condition, through the one head."""

    import numpy as np

    from xinyenyana.evaluation import KERNEL_RIDGE, evaluate_endpoint, standardisation_for

    records = [record.as_evaluation_record() for record in endpoint.records]
    rule = standardisation_for(int(np.asarray(vectors).shape[1]))
    result: dict[str, Any] = {}
    for gallery in CONDITIONS:
        for query in CONDITIONS:
            evaluated = evaluate_endpoint(
                records=records,
                vectors=np.asarray(vectors),
                representation="stored",
                enrollment_condition=gallery,
                query_condition=query,
                ridge_lambda=1.0,
                seed=seed,
                permutations=permutations,
                bootstrap_replicates=REPLICATES,
                standardisation=rule,
                head=KERNEL_RIDGE,
                extended_nulls=False,
            )
            key = f"{gallery}_gallery_{query}_query"
            if retained_predictions is not None:
                retained_predictions[key] = {
                    field: evaluated[field]
                    for field in (
                        "classification",
                        "verification",
                        "enrollment_calls",
                        "query_calls",
                        "predictions",
                    )
                }
            result[key] = {
                "accuracy": evaluated["classification"]["accuracy"],
                "identity_block_bootstrap_accuracy_95": evaluated[
                    "identity_block_bootstrap_accuracy_95"
                ],
                "permutation": evaluated["permutation_control"],
                "roc_auc": evaluated["verification"]["roc_auc"],
            }
    return result


def challenge(
    *, endpoint: Endpoint, vectors: Any, arm_vectors: dict[str, Any], name: str
) -> dict[str, Any]:
    """The added-background challenge for one representation."""

    import numpy as np

    from xinyenyana.background_probe import evaluate_background_challenge

    evaluated = evaluate_background_challenge(
        endpoint,
        {name: np.asarray(vectors)},
        {arm: {name: np.asarray(matrix)} for arm, matrix in arm_vectors.items()},
        bootstrap_replicates=REPLICATES,
    )
    entry = evaluated["representations"][name]
    arms = {
        arm: {
            "accuracy": values["classification"]["accuracy"],
            "macro_recall": values["classification"]["macro_recall"],
            "identity_block_bootstrap_accuracy_95": values["identity_block_bootstrap_accuracy_95"],
        }
        for arm, values in entry["arms"].items()
    }
    return {"arms": arms, "own_vs_other": entry["own_vs_other"]}


def stored_order(records: Sequence[ClipRecord]) -> list[str]:
    """The file-name order a stored matrix follows; checked on every read."""

    return [record.filename for record in records]
