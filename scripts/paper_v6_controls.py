"""Render all v6 recording controls and AS-norm after the complete evidence gate."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

from paper_v6_evidence import ALL_MODELS, BACKGROUND_ENDPOINTS, required_count, validate

from xinyenyana.cosine_controls import EXCLUDED_CONTROLS, is_control

PAIRINGS = {
    "foreground_gallery_foreground_query": "C → C",
    "background_gallery_background_query": "B → B",
    "foreground_gallery_background_query": "C → B",
    "background_gallery_foreground_query": "B → C",
}


def clean_controls(block: dict[str, Any]) -> dict[str, Any]:
    """Public numeric fields only; identity and query annotations never pass through."""
    return {
        "four_pairings": {
            pairing: {
                "accuracy": row["accuracy"],
                "interval_95": row["identity_block_bootstrap_accuracy_95"],
                "permutation_p": row["permutation"]["p_value_plus_one"],
                "permutations": row["permutation"]["permutations"],
                "roc_auc": row["roc_auc"],
            }
            for pairing, row in block["four_pairings"].items()
        },
        "challenge": {
            "arms": {
                arm: {
                    k: row[k]
                    for k in ("accuracy", "identity_block_bootstrap_accuracy_95", "macro_recall")
                }
                for arm, row in block["challenge"]["arms"].items()
            },
            "own_vs_other": {
                level: {
                    k: row[k]
                    for k in (
                        "answer_changed_fraction",
                        "identity_block_difference_95",
                        "own_minus_other_macro_recall",
                    )
                }
                for level, row in block["challenge"]["own_vs_other"].items()
            },
        },
    }


def clean_as_norm(block: dict[str, Any]) -> dict[str, Any]:
    return {
        "cohort_clips": block["cohort_clips"],
        "cohort_top": block["cohort_top"],
        **{
            condition: {
                "chance": block[condition]["chance"],
                **{
                    arm: {
                        k: block[condition][arm][k]
                        for k in ("accuracy", "identity_block_bootstrap_accuracy_95")
                    }
                    for arm in ("raw_class_mean", "as_norm")
                },
            }
            for condition in ("calls_scored", "backgrounds_scored")
        },
    }


def complete_gate(path: Path) -> dict[str, Any]:
    report = json.loads(path.read_text())
    total = required_count()
    if (
        report.get("ready_for_numerical_assembly") is not True
        or report.get("required_files") != total
        or report.get("validated_files") != total
        or report.get("missing")
        or report.get("invalid")
    ):
        raise ValueError(f"all {total} evidence contracts must pass before final control assembly")
    return report


def render(results: Path, evidence: Path, out: Path) -> None:
    from paper_figures import Numbers, md_table
    from paper_v6_additional import interval
    from paper_v6_overview import LABELS

    report = complete_gate(evidence)
    numbers = Numbers(results)

    def load(name: str) -> dict[str, Any]:
        data = numbers.load(name)
        if numbers.sources[name] != report["sources"][name]:
            raise ValueError(f"{name}: bytes do not match the complete evidence inventory")
        return data

    controls, sources, norms = {}, {}, {}
    for ep in BACKGROUND_ENDPOINTS:
        selected, provenance = {}, {}
        for group in ("bio", "birdmae", "tf", "onnx", "avex", "xeus", "v5sweep", "merged", "fixed"):
            name = f"v6-{group}-{ep}.json"
            data = load(name)
            for model, block in data.get("controls", {}).items():
                selected[model] = clean_controls(block)
                provenance[model] = name
        if set(selected) != ALL_MODELS:
            raise ValueError(f"{ep}: incomplete model/control coverage")
        rescored = load(f"v6-pairings-{ep}.json")
        validate(rescored, "pairings", ep, ())
        for model, block in selected.items():
            if set(block["four_pairings"]) != set(PAIRINGS):
                raise ValueError(f"{ep}/{model}: incomplete condition pairings")
            # The sweeps scored the pairings with 999 shuffles; the p values come
            # from the same stored vectors rescored with 9,999.
            for pairing, row in block["four_pairings"].items():
                other = rescored["models"][model]["pairings"][pairing]
                if (
                    other["accuracy"] != row["accuracy"]
                    or other["identity_block_bootstrap_accuracy_95"] != row["interval_95"]
                ):
                    raise ValueError(f"{ep}/{model}: rescored pairing differs from the sweep")
                row["permutation_p"] = other["permutation"]["p_value_plus_one"]
                row["permutations"] = other["permutation"]["permutations"]
            provenance[model] = [provenance[model], f"v6-pairings-{ep}.json"]
            numbers.put(f"{ep}.controls.{model}", block, provenance[model])
        name = f"v6-analyses-{ep}.json"
        analysis = load(name)
        validate(analysis, "analyses", ep, ())
        norms[ep] = {
            model: clean_as_norm(row["as_norm"])
            for model, row in analysis["models"].items()
            if not is_control(model)
        }
        if set(norms[ep]) != ALL_MODELS - EXCLUDED_CONTROLS:
            raise ValueError(f"{ep}: incomplete AS-norm coverage")
        numbers.put(f"{ep}.as_norm", norms[ep], name)
        controls[ep], sources[ep] = selected, provenance
    out.mkdir(parents=True, exist_ok=True)
    for ep in BACKGROUND_ENDPOINTS:
        title = LABELS[ep].replace("\n", ", ")
        text = f"# Recording controls: {title}\n\n"
        text += (
            "All 39 model/control entries use their Fisher-selected representation and the same "
            "fixed ridge head. "
            "C denotes call and B ambient background; the first condition supplies the gallery. "
            "Intervals use 10,000 individual resamples. Pairing p values use 9,999 label shuffles "
            "and are unadjusted. "
            "These tests predict attribution labels and do not identify the acoustic cause.\n\n"
        )
        pairing_rows, challenge_rows, norm_rows = [], [], []
        for model, block in sorted(controls[ep].items()):
            label = model.split("/")[-1]
            for pairing, display in PAIRINGS.items():
                row = block["four_pairings"][pairing]
                pairing_rows.append(
                    [
                        label,
                        display,
                        interval(row["accuracy"], row["interval_95"]),
                        f"{row['permutation_p']:.4f}",
                        f"{row['roc_auc']:.3f}",
                    ]
                )
            for level in ("-10db", "0db", "10db"):
                own = block["challenge"]["arms"]["own_" + level]
                other = block["challenge"]["arms"]["other_" + level]
                difference = block["challenge"]["own_vs_other"][level]
                challenge_rows.append(
                    [
                        label,
                        level,
                        interval(own["accuracy"], own["identity_block_bootstrap_accuracy_95"]),
                        interval(other["accuracy"], other["identity_block_bootstrap_accuracy_95"]),
                        f"{difference['answer_changed_fraction']:.3f}",
                        interval(
                            difference["own_minus_other_macro_recall"],
                            difference["identity_block_difference_95"],
                        ),
                    ]
                )
            for condition, display in [
                ("calls_scored", "Calls"),
                ("backgrounds_scored", "Backgrounds"),
            ]:
                row = norms[ep][model][condition]
                raw, treated = row["raw_class_mean"], row["as_norm"]
                norm_rows.append(
                    [
                        label,
                        display,
                        interval(raw["accuracy"], raw["identity_block_bootstrap_accuracy_95"]),
                        interval(
                            treated["accuracy"], treated["identity_block_bootstrap_accuracy_95"]
                        ),
                        f"{treated['accuracy'] - raw['accuracy']:+.3f}",
                    ]
                )
        text += "## Four condition pairings\n\n" + md_table(
            ["Model", "Gallery → query", "Accuracy (95% interval)", "Label p", "Verification AUC"],
            pairing_rows,
        )
        text += (
            "\n\n## Added-background challenge\n\nRatios are call to added background: −10 dB "
            "means the added background is louder. "
            "Accuracy intervals are arm-specific; the macro-recall difference has its own paired "
            "individual interval. "
            "The gallery remains fixed.\n\n"
            + md_table(
                [
                    "Model",
                    "Call/background",
                    "Own donor accuracy (95% interval)",
                    "Other donor accuracy (95% interval)",
                    "Answers changed",
                    "Own − other macro recall (paired 95% interval)",
                ],
                challenge_rows,
            )
        )
        text += (
            "\n\n## Adaptive score normalisation\n\nThe comparator is raw class-mean cosine "
            "scoring, not the ridge head. "
            "Enrolment backgrounds supply the cohort; its fifty highest scores are used on each "
            "side. "
            "Raw and normalised arm intervals are separate 10,000-draw intervals, not paired "
            "intervals on their difference. "
            "The final column is descriptive.\n\n"
            + md_table(
                [
                    "Model",
                    "Queries",
                    "Raw class mean (95% interval)",
                    "AS-norm (95% interval)",
                    "Accuracy change",
                ],
                norm_rows,
            )
        )
        text += (
            "\n\nAll displayed values and their original source digests are in [the numerical "
            "ledger](recording-controls-numbers.json).\n"
        )
        (out / f"{ep}-controls.md").write_text(text)
    numbers.write(out / "recording-controls-numbers.json")
    (out / "assembly-provenance.json").write_text(
        json.dumps(
            {
                "evidence_sha256": hashlib.sha256(evidence.read_bytes()).hexdigest(),
                "endpoints": list(BACKGROUND_ENDPOINTS),
                "model_entries_per_endpoint": len(ALL_MODELS),
                "source_for_controls": sources,
            },
            indent=2,
        )
        + "\n"
    )


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--results", type=Path, required=True)
    p.add_argument("--evidence", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    a = p.parse_args()
    render(a.results, a.evidence, a.out)


if __name__ == "__main__":
    main()
