"""Retrospectively identify model input recipes and named transformer blocks."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any

from xinyenyana.archive import sha256_file
from xinyenyana.bioacoustic import MODELS


def verify_weight_inventory(inventory: dict[str, Any]) -> dict[str, Any]:
    """Stream-check the current cache against the original file sizes and digests."""
    files = inventory.get("files")
    if not isinstance(files, dict) or not files:
        raise ValueError("weight inventory has no files")
    total = 0
    for name, row in files.items():
        path = Path(name)
        if (
            not isinstance(row.get("bytes"), int)
            or row["bytes"] < 0
            or not isinstance(row.get("sha256"), str)
            or len(row["sha256"]) != 64
            or not path.is_file()
            or path.stat().st_size != row["bytes"]
            or sha256_file(path) != row["sha256"]
        ):
            raise ValueError(f"weight cache differs from archived inventory: {name}")
        total += row["bytes"]
    return {"verified_files": len(files), "verified_bytes": total, "all_match": True}


def audit_loads(
    models: Sequence[str],
    sweeps: dict[str, dict[str, Any]],
    packages: dict[str, str],
    describe_model: Callable[[str], dict[str, Any]],
) -> dict[str, Any]:
    """Load metadata only after checking archived environments and candidate coverage."""
    if not models or len(set(models)) != len(models) or not set(models) <= MODELS.keys():
        raise ValueError("declare distinct registered bioacoustic models")
    coverage: dict[str, dict[str, list[str]]] = {model: {} for model in models}
    for source, sweep in sweeps.items():
        present = set(models) & sweep.get("curve", {}).keys()
        if not present:
            continue
        if sweep.get("environment", {}).get("packages") != packages:
            raise ValueError(f"{source}: package inventory differs from the measured run")
        for model in present:
            coverage[model][source] = sorted(sweep["curve"][model])
    if any(not rows for rows in coverage.values()):
        raise ValueError("a declared model has no measured candidate source")
    descriptions = {}
    for model in models:
        description = describe_model(model)
        blocks = description["blocks"]
        if (
            description["model"] != model
            or description["source"] != MODELS[model].source
            or len(blocks) != len(set(blocks))
            or description["sample_rate"] <= 0
            or description["window_samples"] <= 0
            or bool(blocks) != MODELS[model].transformer
        ):
            raise ValueError(f"{model}: invalid reconstructed load description")
        names = sorted(["embedding", *[f"block {index:02d}" for index in range(len(blocks))]])
        if any(candidates != names for candidates in coverage[model].values()):
            raise ValueError(f"{model}: reconstructed block count differs from measured candidates")
        descriptions[model] = {
            **description,
            "candidate_to_module": {f"block {i:02d}": name for i, name in enumerate(blocks)},
            "measured_candidate_sources": sorted(coverage[model]),
            "candidate_names_match": True,
        }
    return {
        "experiment": "PA-V6 retrospective model-load audit",
        "retrospective": True,
        "audio_read": False,
        "limitation": (
            "Reloaded from retained weights and matching package inventories; "
            "not an original execution trace"
        ),
        "models": descriptions,
    }
