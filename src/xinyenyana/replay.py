"""Reconstruct a replay only from explicitly measured, exactly reproducing models."""

from __future__ import annotations

from copy import deepcopy
from typing import Any

from xinyenyana.a6 import compare_sweep_candidates


def apply_corrections(
    reference: dict[str, Any], base: dict[str, Any], corrections: list[dict[str, Any]]
) -> dict[str, Any]:
    """Replace whole models; never select among alternatives by query accuracy."""
    compare_sweep_candidates(reference, base)  # Also validates manifest and query alignment.
    result = deepcopy(base)
    checked: dict[str, Any] = {}
    mappings = (
        "curve",
        "chosen",
        "chosen_held_out",
        "candidate_vectors",
        "stored_vectors",
        "controls",
    )
    for correction in corrections:
        if correction.get("splits") != base.get("splits"):
            raise ValueError("correction uses different split or enrolment-cap digests")
        for model, curve in correction["curve"].items():
            if model in checked:
                raise ValueError(f"multiple alternative corrections for {model}")
            if not reference["curve"].get(model):
                raise ValueError(f"no historical candidates to reproduce for {model}")
            precision = correction.get("speech_precision", {}).get(model)
            if (
                precision not in {"fp32", "tf32"}
                or not curve
                or any(
                    row.get("numeric_precision", {}).get("speech_matmul") != precision
                    for row in curve.values()
                )
            ):
                raise ValueError(f"no consistent measured arithmetic for {model}")
            expected = {**reference, "curve": {model: reference["curve"][model]}}
            check = compare_sweep_candidates(expected, correction)
            if not check["passed"]:
                raise ValueError(
                    f"correction does not reproduce every historical candidate for {model}"
                )
            for key in ("chosen", "candidate_vectors", "stored_vectors"):
                if not correction.get(key, {}).get(model):
                    raise ValueError(f"correction lacks {key} for {model}")
            choice = correction["chosen"][model]
            name = choice["representation"]
            if (
                name not in curve
                or choice != {"representation": name, **curve[name]}
                or correction["stored_vectors"][model]["representation"] != name
            ):
                raise ValueError(f"correction choice and retained vectors disagree for {model}")
            if base.get("controls", {}).get(model) and not correction.get("controls", {}).get(
                model
            ):
                raise ValueError(f"correction lacks recording controls for {model}")
            checked[model] = check
            for key in mappings:
                result.setdefault(key, {}).pop(model, None)
                if model in correction.get(key, {}):
                    result[key][model] = deepcopy(correction[key][model])
            result.setdefault("speech_precision", {})[model] = precision
            result["not_computed"] = [
                row for row in result.get("not_computed", []) if row.get("model") != model
            ] + [row for row in correction.get("not_computed", []) if row.get("model") == model]
    reproduction = compare_sweep_candidates(reference, result)
    if not checked or not reproduction["passed"]:
        raise ValueError("reconstructed replay still fails the complete historical-candidate check")
    result["replay_correction"] = {
        "per_model": checked,
        "reproduction": reproduction,
        "base_code_revision": base.get("code_revision"),
        "base_source_sha256": base.get("source_sha256"),
        "rule": "one declared arithmetic context per replaced model; exact historical reproduction",
    }
    result["ranking"] = sorted(
        [
            {
                "model": model,
                **{
                    key: choice[key]
                    for key in (
                        "representation",
                        "accuracy",
                        "identity_block_bootstrap_accuracy_95",
                        "permutation_p",
                    )
                },
            }
            for model, choice in result["chosen"].items()
        ],
        key=lambda row: (-float(row["accuracy"]), str(row["model"])),
    )
    return result
