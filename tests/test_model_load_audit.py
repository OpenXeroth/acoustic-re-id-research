"""A reconstructed model recipe must agree with the measured environment and layers."""

import hashlib
from pathlib import Path
from typing import Any

import pytest

from xinyenyana.model_load_audit import audit_loads, verify_weight_inventory


def test_changed_weight_bytes_fail_even_when_file_size_is_unchanged(tmp_path: Path) -> None:
    weights = tmp_path / "weights.bin"
    weights.write_bytes(b"original")
    inventory = {
        "files": {str(weights): {"bytes": 8, "sha256": hashlib.sha256(b"original").hexdigest()}}
    }
    assert verify_weight_inventory(inventory) == {
        "verified_files": 1,
        "verified_bytes": 8,
        "all_match": True,
    }
    weights.write_bytes(b"modified")
    with pytest.raises(ValueError, match="cache differs"):
        verify_weight_inventory(inventory)


def test_empty_cache_marker_is_valid_but_missing_file_is_not(tmp_path: Path) -> None:
    marker = tmp_path / "marker"
    marker.touch()
    inventory = {"files": {str(marker): {"bytes": 0, "sha256": hashlib.sha256(b"").hexdigest()}}}
    assert verify_weight_inventory(inventory)["verified_bytes"] == 0
    marker.unlink()
    with pytest.raises(ValueError, match="cache differs"):
        verify_weight_inventory(inventory)


def source() -> dict[str, Any]:
    return {
        "run.json": {
            "environment": {"packages": {"bacpipe": "1.3.5"}},
            "curve": {"birdmae": {"embedding": {}, "block 00": {}, "block 01": {}}},
        }
    }


def description(model: str) -> dict[str, Any]:
    return {
        "model": model,
        "source": "bacpipe:birdmae",
        "sample_rate": 32000,
        "window_samples": 160000,
        "blocks": ["encoder.blocks.0", "encoder.blocks.1"],
    }


def test_reconstructed_load_maps_ordinal_candidates_without_claiming_original_trace() -> None:
    result = audit_loads(["birdmae"], source(), {"bacpipe": "1.3.5"}, description)
    assert result["retrospective"] is True
    assert result["audio_read"] is False
    assert result["models"]["birdmae"]["candidate_to_module"] == {
        "block 00": "encoder.blocks.0",
        "block 01": "encoder.blocks.1",
    }


def test_changed_environment_is_rejected_before_loading_a_model() -> None:
    def must_not_load(model: str) -> dict[str, Any]:
        raise AssertionError("environment should be checked first")

    with pytest.raises(ValueError, match="package inventory differs"):
        audit_loads(["birdmae"], source(), {"bacpipe": "changed"}, must_not_load)


def test_changed_architecture_cannot_describe_measured_candidates() -> None:
    def changed(model: str) -> dict[str, Any]:
        return {**description(model), "blocks": ["encoder.blocks.0"]}

    with pytest.raises(ValueError, match="block count differs"):
        audit_loads(["birdmae"], source(), {"bacpipe": "1.3.5"}, changed)


def test_no_measured_source_is_not_a_successful_audit() -> None:
    with pytest.raises(ValueError, match="no measured candidate source"):
        audit_loads(["birdmae", "vggish"], source(), {"bacpipe": "1.3.5"}, description)
