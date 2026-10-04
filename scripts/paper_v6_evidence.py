"""Check paper-v6 evidence coverage before any manuscript is called complete.

Read JSON results only. This does not run experiments or turn an absent result
into a negative finding. ``--status`` writes a coverage report even when results
are pending; without it, every required result and contract must pass first.
The output contains aggregate values and provenance, never identity annotations.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

from xinyenyana.a5 import MODELS as SPEECH_MODELS
from xinyenyana.cosine_controls import EXCLUDED_CONTROLS, EXCLUSION_REASON

ENDPOINTS = (
    "birdpark-juv01",
    "zebra-finch",
    "birdpark-juv03",
    "great-tit",
    "pipit-acrossyear",
    "chiffchaff-acrossyear",
    "pipit-withinyear",
    "littleowl-acrossyear",
    "penguin-acrossnight",
    "cockatoo-fold1",
    "bat-acrosstreatment",
    "chiffchaff-withinyear",
    "rookid-full-width",
    "great-tit-full",
)
BACKGROUND_ENDPOINTS = (
    "pipit-acrossyear",
    "chiffchaff-acrossyear",
    "pipit-withinyear",
    "littleowl-acrossyear",
    "chiffchaff-withinyear",
)
OPEN_ENDPOINTS = (
    "littleowl-acrossyear",
    "cockatoo-fold1",
    "rookid-full-width",
    "great-tit",
    "great-tit-full",
)
#: Carried v5 analyses rerun at the v6 counts (protocol, 2026-09-28), by file.
CARRIED = (
    *(
        f"split-difference-{s}-{m}.json"
        for s in ("chiffchaff", "littleowl", "pipit")
        for m in ("birdnet", "google-perch")
    ),
    *(
        f"gate-right-whale-shift-{shift}{suffix}.json"
        for shift in range(0, 10001, 1000)
        for suffix in ("", "-clip-duration-only", "-clip-level-only")
    ),
    "stowell-remedy-littleowl-acrossyear.json",
    "stowell-remedy-chiffchaff-withinyear.json",
    "song-removed-great-tit.json",
    *(
        f"frozen-{name}.json"
        for name in (
            "zebra",
            "great-tit",
            "rookid",
            "birdpark-juv01",
            "birdpark-juv03",
            "littleowl",
            "chiffchaff-acrossyear",
            "pipit-withinyear",
            "pipit-acrossyear",
            "chiffchaff-withinyear",
        )
    ),
)
BIO_GROUPS = {
    "bio": (
        "avesecho-passt",
        "audioprotopnet",
        "convnext-birdset",
        "protoclr",
        "rcl-fs-bsed",
        "birdaves",
        "aves",
        "naturebeats",
        "biolingual",
        "beats",
        "audiomae",
    ),
    "birdmae": ("birdmae",),
    "tf": ("perch-bird", "surfperch", "vggish"),
    "onnx": ("perch-v2", "birdnet-v3-preview"),
    "avex": ("esp-aves2-sl-beats-all", "esp-aves2-effnetb0-all"),
}

ALL_MODELS = {
    *SPEECH_MODELS,
    "espnet/xeus",
    *(model for models in BIO_GROUPS.values() for model in models),
}


def sha256(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def required_results() -> dict[str, tuple[str, str | None, tuple[str, ...]]]:
    """File -> scientific role, endpoint, required model names."""

    required = {}
    for endpoint in ENDPOINTS:
        for group, models in BIO_GROUPS.items():
            required[f"v6-{group}-{endpoint}.json"] = ("corrected-sweep", endpoint, models)
        required[f"v6-xeus-{endpoint}.json"] = ("selection-refresh", endpoint, ("espnet/xeus",))
        required[f"v6-analyses-{endpoint}.json"] = ("analyses", endpoint, ())
        speech = "v5new" if endpoint in ENDPOINTS[-2:] else "merged"
        if speech == "merged":
            required[f"v6-v5sweep-{endpoint}.json"] = ("speech-replay", endpoint, ())
        required[f"v6-{speech}-{endpoint}.json"] = ("speech", endpoint, ())
        if speech == "merged":
            required[f"v6-fixed-{endpoint}.json"] = ("fixed", endpoint, ())
    for endpoint in OPEN_ENDPOINTS:
        required[f"v6-counts/v6-open-set-{endpoint}.json"] = ("open-set", endpoint, ())
    for endpoint in BACKGROUND_ENDPOINTS:
        required[f"v6-counts/v6-spectral-subtraction-{endpoint}.json"] = ("spectral", endpoint, ())
        required[f"v6-pairings-{endpoint}.json"] = ("pairings", endpoint, ())
    for name in CARRIED:
        required[f"carried/{name}"] = ("carried", None, ())
    required.update(
        {
            "v6-headline-great-tit-full.json": ("headline", "great-tit-full", ()),
            "v6-place-control-great-tit-full.json": ("place", "great-tit-full", ()),
            "v6-counts/v6-stowell-augmentation-pipit-withinyear.json": (
                "augmentation",
                "pipit-withinyear",
                (),
            ),
            "v6-sensitivity-floor.json": ("sensitivity", None, ()),
            "v6-perch-window-audit.json": ("input-audit", None, ()),
            "v6-v5-reproduction.json": ("reproduction", None, ()),
            "v6-encoder-comparison.json": ("comparison", None, ()),
            "v6-encoder-comparison-great-tit-full.json": ("comparison-supplement", None, ()),
        }
    )
    return required


def _walk(value: Any) -> list[tuple[str, Any]]:
    """Every (key, value) pair in a nested result, depth first."""
    found: list[tuple[str, Any]] = []
    if isinstance(value, dict):
        for key, item in value.items():
            found.append((key, item))
            found.extend(_walk(item))
    elif isinstance(value, list):
        for item in value:
            found.extend(_walk(item))
    return found


def validate(
    data: dict[str, Any], role: str, endpoint: str | None, models: tuple[str, ...]
) -> None:
    if endpoint is not None:
        expected = "rookid-full-across-year" if endpoint == "rookid-full-width" else endpoint
        if data.get("endpoint") != expected:
            raise ValueError("wrong endpoint")
    if role == "input-audit":
        rows = data.get("per_endpoint", {})
        if set(rows) != set(ENDPOINTS) or any(
            row.get("zero_denominator_windows") != 0 or row.get("nonfinite_windows") != 0
            for row in rows.values()
        ):
            raise ValueError("Perch input warning has not been resolved across every endpoint")
    elif role == "corrected-sweep":
        if not data.get("code_revision") or not data.get("inputs", {}).get("audio_files_sha256"):
            raise ValueError("missing corrected-run code/input provenance")
        if not data.get("environment", {}).get("lease"):
            raise ValueError("no recorded measurement lease")
        for model in models:
            if not data.get("curve", {}).get(model) or model not in data.get("chosen", {}):
                raise ValueError(f"no measured {model}")
            for kind in ("stored_vectors", "candidate_vectors"):
                entry = data.get(kind, {}).get(model, {})
                if Path(entry.get("path", "")).name != entry.get("sha256", "missing") + ".npz":
                    raise ValueError(f"{model}: store is not an immutable digest object")
            if endpoint in BACKGROUND_ENDPOINTS and not data.get("controls", {}).get(model):
                raise ValueError(f"{model}: recording controls missing")
    elif role == "selection-refresh":
        if not data.get("selection_correction", {}).get("source_result", {}).get("sha256"):
            raise ValueError("legacy selection has not been corrected")
        if not set(models) <= data.get("chosen", {}).keys():
            raise ValueError("XEUS choice is absent")
    elif role == "reproduction":
        rows = data.get("per_endpoint", {})
        if set(rows) != set(ENDPOINTS[:-2]) or not all(row.get("passed") for row in rows.values()):
            raise ValueError("not every v5 candidate reproduced on all twelve replay endpoints")
    elif role == "fixed":
        if data.get("nothing_changed"):
            return
        requested = set(data.get("models", []))
        rows = data.get("reproduces_fixed_sweep", {})
        if (
            not requested
            or set(rows) != requested
            or not all(r["identical"] for r in rows.values())
        ):
            raise ValueError("fixed choices do not all reproduce")
    elif role == "augmentation":
        arms = data.get("arms", {})
        if len(arms) != 2 or any(
            a.get("bootstrap_replicates") != 10000 or a.get("permutations") != 9999
            for a in arms.values()
        ):
            raise ValueError("augmentation does not use 10,000 draws and 9,999 shuffles")
    elif role.startswith("comparison"):
        expected = set(ENDPOINTS[:-1]) if role == "comparison" else {"great-tit-full"}
        rows = data.get("per_endpoint", {})
        if set(rows) != expected:
            raise ValueError("comparison endpoint coverage differs from registration")
        for row in rows.values():
            if set(row.get("selection", {})) != {m.split("/")[-1] for m in ALL_MODELS}:
                raise ValueError("comparison does not cover all 39 model/control entries")
            if (row.get("paired") or {}).get("replicates") != 10000:
                raise ValueError("comparison interval count differs from registration")
    elif role == "analyses":
        if set(data.get("models", {})) != ALL_MODELS:
            raise ValueError("stored-vector analyses do not cover all 39 model/control entries")
        for model, row in data["models"].items():
            if row.get("permutation_nulls", {}).get("permutations") != 9999:
                raise ValueError("reported representation lacks its 9,999-shuffle null")
            if endpoint in BACKGROUND_ENDPOINTS and not (
                model in EXCLUDED_CONTROLS and row.get("as_norm_excluded") == EXCLUSION_REASON
            ):
                for condition in ("calls_scored", "backgrounds_scored"):
                    block = row.get("as_norm", {}).get(condition, {})
                    predictions = block.get("predictions", {})
                    actual = predictions.get("actual_label_indices", [])
                    labels = predictions.get("labels", [])
                    query_ids = predictions.get("query_ids", [])
                    if not actual or not labels or len(query_ids) != len(actual):
                        raise ValueError("AS-norm per-query observations are missing")
                    for head in ("raw_class_mean", "as_norm"):
                        predicted = predictions.get(head, [])
                        if len(predicted) != len(actual) or any(
                            not isinstance(i, int) or not 0 <= i < len(labels)
                            for i in [*actual, *predicted]
                        ):
                            raise ValueError("AS-norm prediction indices are invalid")
                        flags = "".join(
                            "1" if a == p else "0" for a, p in zip(actual, predicted, strict=True)
                        )
                        if flags != block[head]["query_correct"] or (
                            flags.count("1") / len(flags) != block[head]["accuracy"]
                        ):
                            raise ValueError("AS-norm predictions do not reconstruct its summary")
            other = row.get("held_out_rule", {})
            if other and not other.get("same_as_fisher_rule"):
                if other.get("permutation_nulls", {}).get("permutations") != 9999:
                    raise ValueError("held-out choice lacks its 9,999-shuffle null")
    elif role == "headline":
        if data.get("evaluation", {}).get("permutation_control", {}).get("permutations") != 9999:
            raise ValueError("headline does not use 9,999 shuffles")
    elif role == "open-set":
        from xinyenyana.archive import canonical_sha256
        from xinyenyana.open_set import ALLOCATION_SEED, ALLOCATIONS, ROLES

        excluded = data.get("excluded_representations", {})
        if excluded and (
            set(excluded) != EXCLUDED_CONTROLS
            or any(reason != EXCLUSION_REASON for reason in excluded.values())
        ):
            raise ValueError("open-set exclusions must identify only the two invalid raw controls")
        if set(data.get("representations", {})) != ALL_MODELS - set(excluded):
            raise ValueError(
                "open set does not cover the required neural networks and declared scope"
            )
        if data.get("allocation_seed") != ALLOCATION_SEED or data.get("allocations") != ALLOCATIONS:
            raise ValueError("open-set allocation provenance is missing")
        for block in data["representations"].values():
            allocations = block.get("allocations", [])
            if len(allocations) != ALLOCATIONS or any(
                not allocation.get("observations", {}).get(partition)
                for allocation in allocations
                for partition in ("calibration", "test")
            ):
                raise ValueError("open-set per-query observations are missing")
            digest = canonical_sha256(
                [{role: a["roles"][role] for role in ROLES} for a in allocations]
            )
            if digest != data.get("role_digest"):
                raise ValueError("open-set allocations do not match the recorded role digest")
    elif role == "spectral":
        if not {"original", "alpha_1", "alpha_2"} <= data.keys():
            raise ValueError("one or more spectral subtraction arms are absent")
        if data.get("permutations") != 9999:
            raise ValueError("spectral subtraction does not use 9,999 shuffles")
    elif role == "pairings":
        if data.get("permutations") != 9999 or set(data.get("models", {})) != ALL_MODELS:
            raise ValueError("pairings do not cover all 39 entries at 9,999 shuffles")
    elif role == "carried":
        counts = [
            value
            for key, value in _walk(data)
            if key in ("replicates", "bootstrap_replicates", "permutations")
        ]
        if not counts or any(value not in (0, 9999, 10000) for value in counts):
            raise ValueError("carried analysis is not at the v6 counts")
    elif role == "sensitivity":
        if set(data.get("levels", {})) != {"-10db", "+0db", "+10db"}:
            raise ValueError("one or more planted sensitivity levels are absent")
    elif role == "speech":
        if set(data.get("chosen", {})) != set(SPEECH_MODELS):
            raise ValueError("speech comparison lacks one or more of its 19 entries")
    elif role == "speech-replay":
        # The replay repeats the original whole-clip recipe. Registered
        # windowed gap runs can supply a model absent from that replay; the
        # merged speech result, rather than each input, must cover all models.
        curves = data.get("curve", {})
        if set(curves) != set(SPEECH_MODELS) or "birdnet-v2.4" not in data.get("chosen", {}):
            raise ValueError("speech replay lacks its complete attempt inventory")
        explained = {row.get("model") for row in data.get("not_computed", []) if row.get("reason")}
        if any(not curve and model not in explained for model, curve in curves.items()):
            raise ValueError("speech replay has an unexplained absent model")
        if "selection_correction" not in data:
            raise ValueError("speech replay has not received its selection correction")


def required_count() -> int:
    """Files the complete inventory checks: the results plus the thirteen headline runs."""
    return len(required_results()) + len(ENDPOINTS) - 2 + 1


def inventory(results: Path, headline: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    missing, invalid, sources, loaded = [], {}, {}, {}
    paths = {name: (results / name, contract) for name, contract in required_results().items()}
    for endpoint in ENDPOINTS[:-2]:
        name = f"v6-headline-{endpoint}.json"
        paths[name] = (headline / name, ("headline", endpoint, ()))
    name = "v6-headline-rook-full-across-year-capped-90.json"
    paths[name] = (headline / name, ("headline", "rookid-full-width", ()))
    for name, (path, contract) in paths.items():
        if not path.is_file():
            missing.append(name)
            continue
        try:
            data = json.loads(path.read_text())
            if not isinstance(data, dict):
                raise ValueError("result is not a JSON object")
            validate(data, *contract)
            loaded[name] = data
            sources[name] = sha256(path)
        except (OSError, ValueError, KeyError, TypeError) as exc:
            invalid[name] = str(exc)
    return {
        "ready_for_numerical_assembly": not missing and not invalid,
        "required_files": len(paths),
        "validated_files": len(sources),
        "missing": sorted(missing),
        "invalid": invalid,
        "sources": sources,
        "boundary": "File contracts do not replace final numerical and manuscript review.",
    }, loaded


def as_norm_aggregate(block: dict[str, Any]) -> dict[str, Any]:
    """Keep score-normalisation summaries, excluding private query annotations."""

    return {
        **{key: block[key] for key in ("cohort_clips", "cohort_top")},
        **{
            condition: {
                key: block[condition][key] for key in ("raw_class_mean", "as_norm", "chance")
            }
            for condition in ("calls_scored", "backgrounds_scored")
        },
    }


def aggregate(loaded: dict[str, Any]) -> dict[str, Any]:
    """Whitelist publication-safe aggregates; never publish whole raw result objects."""

    tables: dict[str, Any] = {
        "headline": {},
        "comparisons": {},
        "information": {},
        "background": {},
        "open_set": {},
        "spectral": {},
        "augmentation": {},
        "sensitivity": {},
        "place": {},
        "pairings": {},
    }
    for name, data in loaded.items():
        base = name.rsplit("/", 1)[-1]
        if name.startswith("carried/"):
            continue
        if base.startswith("v6-pairings-"):
            tables["pairings"][data["endpoint"]] = {
                model: {"representation": row["representation"], "pairings": row["pairings"]}
                for model, row in data["models"].items()
            }
        elif base.startswith("v6-headline-"):
            e = data["evaluation"]
            tables["headline"][data["endpoint"]] = {
                "individuals": data["identities"],
                "accuracy": e["classification"]["accuracy"],
                "interval_95": e["identity_block_bootstrap_accuracy_95"],
                "permutation_p": e["permutation_control"]["p_value_plus_one"],
                "chance": data["uniform_chance"],
                "source": name,
            }
        elif base.startswith("v6-encoder-comparison"):
            tables["comparisons"][name] = {
                "per_endpoint": {
                    endpoint: {k: row[k] for k in ("selection", "paired", "held_out_rule")}
                    for endpoint, row in data["per_endpoint"].items()
                },
                "rank_consistency": data["rank_consistency"],
                "rank_consistency_held_out_rule": data["rank_consistency_held_out_rule"],
            }
        elif base.startswith("v6-analyses-"):
            tables["information"][data["endpoint"]] = {
                model: {
                    k: as_norm_aggregate(row[k]) if k == "as_norm" else row[k]
                    for k in (
                        "representation",
                        "permutation_nulls",
                        "held_out_rule",
                        "beecher_hs",
                        "as_norm",
                    )
                    if k in row
                }
                for model, row in data["models"].items()
            }
        elif base.startswith("v6-open-set-"):
            tables["open_set"][data["endpoint"]] = {
                "models": {
                    model: {
                        "representation": row["representation"],
                        "allocation_sensitivity": row["allocation_sensitivity"],
                    }
                    for model, row in data["representations"].items()
                },
            }
        elif base.startswith("v6-spectral-subtraction-"):
            tables["spectral"][data["endpoint"]] = {
                key: data[key] for key in ("config", "original", "alpha_1", "alpha_2")
            }
        elif base.startswith("v6-stowell-augmentation-"):
            tables["augmentation"][data["endpoint"]] = {
                arm: {
                    key: row[key]
                    for key in ("pairings_by_head", "bootstrap_replicates", "permutations")
                }
                for arm, row in data["arms"].items()
            }
        elif base == "v6-sensitivity-floor.json":
            tables["sensitivity"] = {
                level: {
                    "accuracy": row["evaluation"]["classification"]["accuracy"],
                    "interval_95": row["evaluation"]["identity_block_bootstrap_accuracy_95"],
                    "permutation_p": row["evaluation"]["permutation_control"]["p_value_plus_one"],
                    "chance": row["uniform_chance"],
                }
                for level, row in data["levels"].items()
            }
        elif base.startswith("v6-place-control-"):
            tables["place"][data["endpoint"]] = {
                "individuals": data["identities"],
                "shared_nestbox": data["identities_sharing_a_nestbox_across_the_split"],
                "shared_recording": data["identities_sharing_a_source_recording_across_the_split"],
                "distance_metres": data["metres_between_nestboxes"],
                "per_individual": [
                    {
                        "label": f"GT-full-{index:02d}",
                        **{
                            key: row[key]
                            for key in (
                                "shares_a_nestbox",
                                "shares_a_source_recording",
                                "years_differ",
                                "metres_between_nestboxes",
                            )
                        },
                    }
                    for index, row in enumerate(data["per_identity"], 1)
                ],
            }
        elif data.get("controls"):
            tables["background"].setdefault(data["endpoint"], {}).update(data["controls"])
    return tables


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, required=True)
    parser.add_argument("--headline", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--status", action="store_true")
    args = parser.parse_args()
    report, loaded = inventory(args.results, args.headline)
    if not args.status and not report["ready_for_numerical_assembly"]:
        raise SystemExit(
            f"Paper v6 is incomplete: {len(report['missing'])} missing, "
            f"{len(report['invalid'])} invalid"
        )
    if not args.status:
        report["aggregates"] = aggregate(loaded)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(f"{report['validated_files']}/{report['required_files']} evidence files validated")


if __name__ == "__main__":
    main()
