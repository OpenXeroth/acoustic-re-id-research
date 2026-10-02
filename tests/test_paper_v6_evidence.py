"""Publication coverage must fail closed and exclude individual annotations."""

import importlib.util
import json
from pathlib import Path

import pytest

spec = importlib.util.spec_from_file_location(
    "paper_v6_evidence", Path(__file__).parents[1] / "scripts" / "paper_v6_evidence.py"
)
assert spec and spec.loader
paper = importlib.util.module_from_spec(spec)
spec.loader.exec_module(paper)


def test_missing_measurements_are_pending_not_negative(tmp_path):
    report, loaded = paper.inventory(tmp_path, tmp_path)
    assert not report["ready_for_numerical_assembly"]
    assert report["validated_files"] == 0
    assert len(report["missing"]) == report["required_files"]
    assert loaded == {}


def test_pre_correction_sweep_cannot_enter_the_paper():
    with pytest.raises(ValueError, match="corrected-run"):
        paper.validate({"endpoint": "e", "chosen": {"m": {}}}, "corrected-sweep", "e", ("m",))


def test_intersection_only_reproduction_cannot_pass():
    rows = {f"endpoint-{n}": {"identical": 10, "candidates_compared": 10} for n in range(12)}
    with pytest.raises(ValueError, match="not every v5 candidate"):
        paper.validate({"per_endpoint": rows}, "reproduction", None, ())


def test_open_set_aggregation_keeps_operating_points_and_drops_role_identities():
    raw = {
        "endpoint": "e",
        "representations": {
            "birdnet": {
                "representation": "embedding",
                "allocation_sensitivity": {"closed_set_accuracy": {"median": 0.5}},
                "allocations": [
                    {
                        "roles": {"known": ["PRIVATE-ANIMAL"]},
                        "observations": {"test": [{"identity": "PRIVATE-PREDICTION"}]},
                    }
                ],
            }
        },
    }
    result = paper.aggregate({"v6-open-set-e.json": raw})
    assert "PRIVATE-ANIMAL" not in json.dumps(result)
    assert "PRIVATE-PREDICTION" not in json.dumps(result)
    assert result["open_set"]["e"]["models"]["birdnet"]["allocation_sensitivity"] == {
        "closed_set_accuracy": {"median": 0.5}
    }


def test_open_set_without_per_query_decisions_cannot_pass_the_evidence_gate():
    from xinyenyana.open_set import ALLOCATION_SEED, ALLOCATIONS

    result = {
        "endpoint": "e",
        "representations": {m: {"allocations": [{}] * ALLOCATIONS} for m in paper.ALL_MODELS},
        "allocation_seed": ALLOCATION_SEED,
        "allocations": ALLOCATIONS,
    }
    with pytest.raises(ValueError, match="per-query observations"):
        paper.validate(result, "open-set", "e", ())


def test_headline_requires_the_registered_null_count():
    result = {"endpoint": "e", "evaluation": {"permutation_control": {"permutations": 999}}}
    with pytest.raises(ValueError, match="9,999"):
        paper.validate(result, "headline", "e", ())


def test_as_norm_private_decisions_are_checked_then_excluded_from_publication():
    import copy

    block = {
        "predictions": {
            "labels": ["PRIVATE-ANIMAL", "OTHER-ANIMAL"],
            "query_ids": ["PRIVATE-FILE", "OTHER-FILE"],
            "actual_label_indices": [0, 1],
            "raw_class_mean": [0, 0],
            "as_norm": [1, 1],
        },
        "raw_class_mean": {"accuracy": 0.5, "query_correct": "10"},
        "as_norm": {"accuracy": 0.5, "query_correct": "01"},
        "chance": 0.5,
    }
    raw = {
        "endpoint": "pipit-withinyear",
        "models": {
            model: {
                "permutation_nulls": {"permutations": 9999},
                "as_norm": {
                    "cohort_clips": 12,
                    "cohort_top": 50,
                    "calls_scored": copy.deepcopy(block),
                    "backgrounds_scored": copy.deepcopy(block),
                },
            }
            for model in paper.ALL_MODELS
        },
    }
    paper.validate(raw, "analyses", "pipit-withinyear", ())
    result = paper.aggregate({"v6-analyses-pipit-withinyear.json": raw})
    text = json.dumps(result)
    for private in ("PRIVATE-ANIMAL", "OTHER-ANIMAL", "PRIVATE-FILE", "OTHER-FILE", "predictions"):
        assert private not in text
    saved = result["information"]["pipit-withinyear"]["birdnet-v2.4"]["as_norm"]
    assert saved["cohort_clips"] == 12
    assert saved["calls_scored"]["as_norm"] == block["as_norm"]
    raw["models"]["birdnet-v2.4"]["as_norm"]["calls_scored"]["predictions"]["as_norm"] = [0, 0]
    with pytest.raises(ValueError, match="do not reconstruct"):
        paper.validate(raw, "analyses", "pipit-withinyear", ())
    del raw["models"]["birdnet-v2.4"]["as_norm"]["calls_scored"]["predictions"]
    with pytest.raises(ValueError, match="observations are missing"):
        paper.validate(raw, "analyses", "pipit-withinyear", ())


def test_partial_analysis_cannot_claim_complete_model_coverage():
    result = {
        "endpoint": "e",
        "models": {"birdnet-v2.4": {"permutation_nulls": {"permutations": 9999}}},
    }
    with pytest.raises(ValueError, match="all 39"):
        paper.validate(result, "analyses", "e", ())


def test_whole_clip_replay_can_defer_an_explained_model_to_the_gap_run():
    missing = "microsoft/wavlm-large"
    result = {
        "endpoint": "e",
        "curve": {m: {} if m == missing else {"candidate": {}} for m in paper.SPEECH_MODELS},
        "chosen": {m: {"representation": "candidate"} for m in paper.SPEECH_MODELS if m != missing},
        "selection_correction": {},
        "not_computed": [{"model": missing, "reason": "whole clip exceeds device capacity"}],
    }
    paper.validate(result, "speech-replay", "e", ())
    with pytest.raises(ValueError, match="19 entries"):
        paper.validate(result, "speech", "e", ())
