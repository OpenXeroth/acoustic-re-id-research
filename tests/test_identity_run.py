"""The v5 headline measurement: each piece can give the opposite answer."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from xinyenyana.a2 import ClipRecord, Endpoint
from xinyenyana.identity_run import measure, one_sided_equivalence, place_control, random_fold


def _records(birds: int = 8, per_split: int = 12, sessions: int = 3) -> list[ClipRecord]:
    records = []
    for b in range(birds):
        for split in ("enrollment", "query"):
            for i in range(per_split):
                records.append(
                    ClipRecord(
                        filename=f"b{b}-{split}-{i}.wav",
                        path=Path(f"/nonexistent/b{b}-{split}-{i}.wav"),
                        identity=f"bird{b}",
                        split=split,
                        context={
                            "condition": "foreground",
                            "recording": f"{split}-rec{i % sessions}",
                        },
                    )
                )
    return records


def _endpoint(records: list[ClipRecord]) -> Endpoint:
    return Endpoint(
        name="synthetic",
        records=tuple(records),
        categorical_targets=(),
        manifest_sha256="0" * 64,
        source_document="tests",
    )


def test_the_random_fold_keeps_the_clips_and_each_animals_scoring_count() -> None:
    records = _records()
    folded, report = random_fold(records, seed=3)
    assert sorted(r.filename for r in folded) == sorted(r.filename for r in records)
    for bird in {r.identity for r in records}:
        before = sum(1 for r in records if r.identity == bird and r.split == "query")
        after = sum(1 for r in folded if r.identity == bird and r.split == "query")
        assert before == after
    assert 0 < report["share_of_scoring_clips_moved"] <= 1


def test_equivalence_passes_at_chance_and_fails_well_above_it() -> None:
    identities = [f"bird{b}" for b in range(16) for _ in range(25)]
    rng = np.random.default_rng(0)
    at_chance = list(rng.random(len(identities)) < 1 / 16)
    well_above = list(rng.random(len(identities)) < 0.5)
    assert one_sided_equivalence(identities=identities, correct=at_chance, bound=0.2, seed=1)[
        "equivalent_at_0.05"
    ]
    assert not one_sided_equivalence(identities=identities, correct=well_above, bound=0.2, seed=1)[
        "equivalent_at_0.05"
    ]


def test_identity_signal_clears_every_null_and_noise_clears_none() -> None:
    records = _records()
    rng = np.random.default_rng(5)
    centres = {f"bird{b}": rng.normal(size=16) * 3 for b in range(8)}
    signal = np.vstack([centres[r.identity] + rng.normal(size=16) for r in records])
    noise = rng.normal(size=(len(records), 16))
    good = measure(endpoint=_endpoint(records), records=records, vectors=signal)
    bad = measure(endpoint=_endpoint(records), records=records, vectors=noise)
    assert good["evaluation"]["classification"]["accuracy"] > 0.9
    assert good["evaluation"]["permutation_control"]["p_value_plus_one"] == 0.001
    assert good["evaluation"]["within_session_permutation"]["p_value_plus_one"] == 0.001
    assert bad["evaluation"]["permutation_control"]["p_value_plus_one"] > 0.05
    assert good["stratified_trials"]["animal_against_recording"]["reads"] == "the animal"
    assert len(good["query_correct"]) == len(good["query_identities"]) == 96


def test_the_place_control_finds_a_shared_nestbox_when_there_is_one() -> None:
    def row(bird: str, split: str, box: str, year: str, x: float) -> ClipRecord:
        return ClipRecord(
            filename=f"{bird}-{split}-{box}",
            path=Path("/nonexistent"),
            identity=bird,
            split=split,
            context={
                "nestbox": box,
                "year": year,
                "source_recording": f"{box}-{year}",
                "nest_x": x,
                "nest_y": 0.0,
            },
        )

    records = [
        row("moved", "enrollment", "A", "2020", 0.0),
        row("moved", "query", "B", "2021", 60.0),
        row("stayed", "enrollment", "C", "2020", 0.0),
        row("stayed", "query", "C", "2021", 0.0),
    ]
    result = place_control(records)
    assert result["identities_sharing_a_nestbox_across_the_split"] == 1
    assert result["identities_whose_years_differ_across_the_split"] == 2
    assert result["metres_between_nestboxes"]["maximum"] == 60.0
