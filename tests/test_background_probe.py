from pathlib import Path
from typing import Any

import numpy as np
import pytest

from xinyenyana.a2 import ClipRecord, Endpoint
from xinyenyana.background_challenge import CHALLENGE_DB
from xinyenyana.background_probe import evaluate_background_challenge


def fixture() -> tuple[Endpoint, dict[str, Any], dict[str, dict[str, Any]]]:
    records = tuple(
        ClipRecord(f"{split}-{label}-{clip}", Path("unused"), str(label), split, {})
        for split in ("enrollment", "query")
        for label in range(3)
        for clip in range(3)
    )
    vectors = np.vstack([np.eye(3)[int(r.identity)] for r in records])
    endpoint = Endpoint("fixture", records, (), "fixture", "generated test")
    arms = {"sham_peak_scaled": {"fixture": vectors[9:].copy()}}
    for ratio in CHALLENGE_DB:
        arms[f"own_{ratio:g}db"] = {"fixture": vectors[9:].copy()}
        arms[f"other_{ratio:g}db"] = {"fixture": np.roll(vectors[9:], 1, axis=1)}
    return endpoint, {"fixture": vectors}, arms


def test_query_challenge_changes_answers_without_refitting_the_reference_calls() -> None:
    endpoint, original, arms = fixture()
    result = evaluate_background_challenge(
        endpoint, original, arms, permutations=9, bootstrap_replicates=20
    )
    measured = result["representations"]["fixture"]
    for ratio in CHALLENGE_DB:
        assert measured["arms"][f"own_{ratio:g}db"]["classification"]["accuracy"] == 1.0
        assert measured["arms"][f"other_{ratio:g}db"]["classification"]["accuracy"] == 0.0
        difference = measured["own_vs_other"][f"{ratio:g}db"]
        assert difference["answer_changed_fraction"] == 1.0
        assert difference["identity_block_difference_95"] == [1.0, 1.0]
    for values in measured["arms"].values():
        assert values["enrollment_calls"] == 9
        assert len(values["predictions"]) == 9
        assert [p["query_id"] for p in values["predictions"]] == [
            r.filename for r in endpoint.records if r.split == "query"
        ]


def test_dropping_a_difficult_query_is_refused() -> None:
    endpoint, original, arms = fixture()
    arms["own_0db"]["fixture"] = arms["own_0db"]["fixture"][:-1]
    with pytest.raises(ValueError, match="retain every"):
        evaluate_background_challenge(endpoint, original, arms)
