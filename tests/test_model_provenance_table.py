"""A publication recipe must describe all measured sources with identical bytes."""

import hashlib
import importlib
import json
from pathlib import Path
from typing import Any

import pytest

from xinyenyana.model_load_audit import audit_loads


@pytest.fixture
def audit(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> tuple[Any, ...]:
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1] / "scripts"))
    module = importlib.import_module("paper_v6_model_provenance")
    packages = {"bacpipe": "example-version"}
    sources, sweeps = {}, {}
    for endpoint in module.ENDPOINTS:
        path = tmp_path / f"v6-birdmae-{endpoint}.json"
        data = {
            "environment": {"packages": packages},
            "curve": {"birdmae": {"embedding": {}, "block 00": {}}},
            "private_query_annotations": "PRIVATE_SENTINEL",
        }
        path.write_text(json.dumps(data))
        original = "/original/results/" + path.name
        sources[original] = hashlib.sha256(path.read_bytes()).hexdigest()
        sweeps[original] = data
    record = audit_loads(
        ["birdmae"],
        sweeps,
        packages,
        lambda model: {
            "model": model,
            "source": "bacpipe:birdmae",
            "sample_rate": 32000,
            "window_samples": 160000,
            "blocks": ["encoder.blocks.0"],
        },
    )
    weights = {"sha256": "a" * 64, "verified_files": 1, "verified_bytes": 8, "all_match": True}
    record.update(sources=sources, environment={"packages": packages}, weight_inventory=weights)
    evidence = {"sources": {Path(name).name: digest for name, digest in sources.items()}}
    return module, record, tmp_path, weights.copy(), evidence


def test_complete_model_recipe_omits_private_annotations(audit: tuple[Any, ...]) -> None:
    module, *args = audit
    public = module.verify_group("birdmae", *args)
    assert public["birdmae"]["measured_endpoints"] == 14
    assert public["birdmae"]["candidate_to_module"] == {"block 00": "encoder.blocks.0"}
    assert "PRIVATE_SENTINEL" not in json.dumps(public)


@pytest.mark.parametrize(
    "fault", ["changed_source", "missing_endpoint", "mapping", "unchecked_weights"]
)
def test_inconsistent_recipe_cannot_enter_publication(audit: tuple[Any, ...], fault: str) -> None:
    module, record, results, weights, evidence = audit
    if fault == "changed_source":
        path = next(results.glob("*.json"))
        path.write_text(path.read_text() + " ")
    elif fault == "missing_endpoint":
        record["sources"].pop(next(iter(record["sources"])))
    elif fault == "mapping":
        record["models"]["birdmae"]["candidate_to_module"]["block 00"] = "different.module"
    else:
        record["weight_inventory"]["all_match"] = False
    with pytest.raises(ValueError):
        module.verify_group("birdmae", record, results, weights, evidence)
