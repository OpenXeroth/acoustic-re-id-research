"""A3: what each cleaning step costs or buys, per endpoint and per layer.

The protocol is [`docs/a3-call-cleaning.md`](../../docs/a3-call-cleaning.md).

Five configurations per endpoint: no cleaning, each step alone, and all three
together. The no-cleaning configuration is run here rather than read across from
A2, so every comparison in A3 is between runs of one code revision on one
machine on the same day.

Trimming is not applied to the motor layer. The motor layer measures note and
gap timing, and trimming removes the gaps: its features after trimming would
describe the trimming.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from xinyenyana.a2 import (
    LAYERS,
    SEED,
    Endpoint,
    estimability,
    evaluate_layer,
    extract_endpoint_layers,
)
from xinyenyana.cleaning import (
    BAND_EDGE_ENERGY,
    MINIMUM_NOISE_FRAMES,
    OVER_SUBTRACTION,
    SPECTRAL_FLOOR,
    Band,
    band_from_clips,
)

CONFIGURATIONS: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("none", ()),
    ("trim", ("trim",)),
    ("band", ("band",)),
    ("denoise", ("denoise",)),
    ("all", ("trim", "band", "denoise")),
)

# The motor layer measures the gaps that trimming removes.
LAYERS_EXCLUDED_BY_TRIM = ("motor",)


@dataclass(frozen=True)
class CleaningOutcome:
    """What one configuration did to one endpoint's clips."""

    name: str
    steps: tuple[str, ...]
    clips_with_no_note: int
    clips_with_no_noise_estimate: int


def endpoint_band(endpoint: Endpoint) -> Band:
    """Compute the endpoint's band from its enrollment clips only."""

    return band_from_clips(
        record.path for record in endpoint.records if record.split == "enrollment"
    )


def _cleaning_outcome(
    name: str, steps: tuple[str, ...], diagnostics: list[dict[str, Any]]
) -> CleaningOutcome:
    return CleaningOutcome(
        name=name,
        steps=steps,
        clips_with_no_note=sum(
            1 for row in diagnostics if row["cleaning"].get("trim_found_no_note")
        ),
        clips_with_no_noise_estimate=sum(
            1 for row in diagnostics if row["cleaning"].get("noise_not_estimated")
        ),
    )


def run_a3_endpoint(*, endpoint: Endpoint) -> dict[str, Any]:
    """Run every cleaning configuration on one endpoint and return the measurement."""

    band = endpoint_band(endpoint)
    summary: dict[str, Any] = {
        "endpoint": endpoint.name,
        "band": band.as_dict(),
        "chance_accuracy": 1.0 / len(endpoint.identities),
        "manifest_sha256": endpoint.manifest_sha256,
        "splits": endpoint.split_digests(),
        "seed": SEED,
        "band_edge_energy": BAND_EDGE_ENERGY,
        "over_subtraction": OVER_SUBTRACTION,
        "spectral_floor": SPECTRAL_FLOOR,
        "minimum_noise_frames": MINIMUM_NOISE_FRAMES,
        "configurations": {},
    }
    for index, (name, steps) in enumerate(CONFIGURATIONS):
        matrices, diagnostics, layer_names = extract_endpoint_layers(
            endpoint.records, steps=steps, band=band if "band" in steps else None
        )
        outcome = _cleaning_outcome(name, steps, diagnostics)
        entry: dict[str, Any] = {
            "steps": list(steps),
            "clips_with_no_note": outcome.clips_with_no_note,
            "clips_with_no_noise_estimate": outcome.clips_with_no_noise_estimate,
            "estimability": estimability(diagnostics),
            "dimensions": layer_names,
            "layers": {},
        }
        for offset, layer in enumerate(
            layer for layer in LAYERS if not ("trim" in steps and layer in LAYERS_EXCLUDED_BY_TRIM)
        ):
            entry["layers"][layer] = evaluate_layer(
                endpoint=endpoint,
                layer=layer,
                vectors=matrices[layer],
                diagnostics=diagnostics,
                seed=SEED + index * 10_000 + offset * 1_000,
            )
        summary["configurations"][name] = entry
    return summary


def adopt(summary: dict[str, Any], *, layer: str) -> dict[str, Any]:
    """Apply the registered adoption rule to one endpoint and layer.

    A step is adopted when it raises identity accuracy and does not raise any
    context probe. "Does not raise" rather than "lowers", because several probes
    already sit at their floor of zero on these endpoints and a rule requiring a
    fall could never be satisfied there.

    A probe that has no baseline reading is compared against its floor rather
    than skipped. Its target was constant on the untreated clips, so the step
    under test is what made it vary, and skipping it would let a step be adopted
    for creating the very signal the probe exists to catch. The floor is what no
    information scores: zero for a continuous target, the majority-class rate
    for a categorical one.

    Reads the same result dictionaries ``evaluate_layer`` returns, so a change
    to their shape breaks this loudly rather than silently comparing nothing.
    """

    baseline = summary["configurations"]["none"]["layers"].get(layer)
    verdicts: dict[str, Any] = {}
    if baseline is None:
        return verdicts
    before = float(baseline["classification"]["accuracy"])
    for name, entry in summary["configurations"].items():
        if name == "none" or layer not in entry["layers"]:
            continue
        candidate = entry["layers"][layer]
        after = float(candidate["classification"]["accuracy"])
        context_rose = False
        for probe, values in candidate["context_probes"].items():
            key = "r_squared" if "r_squared" in values else "accuracy"
            earlier = baseline["context_probes"].get(probe)
            if earlier is None:
                floor = 0.0 if key == "r_squared" else float(values["majority_class_rate"])
            else:
                floor = float(earlier[key])
            if float(values[key]) > floor:
                context_rose = True
        verdicts[name] = {
            "accuracy_before": before,
            "accuracy_after": after,
            "raised_accuracy": after > before,
            "raised_a_context_probe": context_rose,
            "adopted": bool(after > before and not context_rose),
        }
    return verdicts
