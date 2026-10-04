"""PA-V6 analyses that read stored vectors, and the two new BirdNET-only runs.

Nothing here runs a neural network except the two BirdNET runs at the end
(spectral subtraction and the sensitivity floor), which read audio that no
stored vector covers. Everything else reads the vectors ``a6`` stored, checked
against the endpoint's manifest and record order before use.
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import Any

from xinyenyana.a2 import ClipRecord, Endpoint
from xinyenyana.v6_controls import REPLICATES


def _stored(results: Sequence[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    """Model -> stored-vector entry, over the result files in the order given.

    A model stored by two files takes the later one: a re-extraction at the
    representation chosen after merging (``run-a6 --fixed-from``) is passed after
    the sweep it replaces. The later entry must name the merged choice, which
    the caller checks.
    """

    found: dict[str, dict[str, Any]] = {}
    for result in results:
        for model, entry in result.get("stored_vectors", {}).items():
            if entry is not None:
                found[model] = entry
    return found


def _held_out(results: Sequence[dict[str, Any]]) -> dict[str, tuple[str, dict[str, Any]]]:
    """Model -> (held-out choice, candidate store entry), where both were recorded."""

    found: dict[str, tuple[str, Any]] = {}
    for result in results:
        # A merged curve can make a previously reported rule unavailable.
        # Do not retain a choice from an earlier fixed-only extraction.
        for model in result.get("curve", {}):
            found.pop(model, None)
        for model, entry in result.get("chosen_held_out", {}).items():
            candidates = result.get("candidate_vectors", {}).get(model)
            if entry is not None and candidates is not None:
                found[model] = (str(entry["representation"]), candidates)
    return found


def _candidate(endpoint: Endpoint, stores: Any, name: str) -> Any:
    """A candidate's matrix from whichever of a model's candidate stores holds it."""

    from xinyenyana.a6 import load_candidate

    for entry in reversed(stores) if isinstance(stores, list) else [stores]:
        path = _check_digest(entry)
        import numpy as np

        with np.load(path, allow_pickle=False) as data:
            names = [str(v) for v in data["names"]]
        if name in names:
            return load_candidate(path, endpoint, name)
    raise ValueError(f"no candidate store holds {name}")


#: Label-permutation and within-session nulls for every reported representation
#: use this many shuffles (raised from 999 in review, before these analyses ran).
PERMUTATIONS = 9_999


def permutation_nulls(endpoint: Endpoint, vectors: Any) -> dict[str, Any]:
    """The label-permutation and within-session nulls with 9,999 shuffles each."""

    import numpy as np

    from xinyenyana.evaluation import KERNEL_RIDGE, evaluate_endpoint, standardisation_for

    evaluated = evaluate_endpoint(
        records=[record.as_evaluation_record() for record in endpoint.records],
        vectors=np.asarray(vectors),
        representation="stored",
        enrollment_condition="foreground",
        query_condition="foreground",
        ridge_lambda=1.0,
        seed=17,
        permutations=PERMUTATIONS,
        bootstrap_replicates=REPLICATES,
        standardisation=standardisation_for(int(np.asarray(vectors).shape[1])),
        head=KERNEL_RIDGE,
        extended_nulls=True,
    )
    evaluated.pop("predictions", None)
    keep = {
        key: evaluated[key]
        for key in evaluated
        if "permutation" in key or "bootstrap" in key or key == "classification"
    }
    return {"permutations": PERMUTATIONS, **keep}


def _check_digest(entry: dict[str, Any]) -> Path:
    from xinyenyana.archive import sha256_file

    path = Path(entry["path"])
    digest = sha256_file(path)
    if digest != entry["sha256"]:
        raise ValueError(f"{path} has changed since its result was written")
    return path


def _foreground_rows(endpoint: Endpoint) -> list[int]:
    return [
        i
        for i, r in enumerate(endpoint.records)
        if str(r.context.get("condition", "foreground")) == "foreground"
    ]


def as_norm_block(endpoint: Endpoint, vectors: Any) -> dict[str, Any]:
    """Class-mean scores before and after AS-norm, calls and backgrounds as queries."""

    import numpy as np

    from xinyenyana.score_normalisation import (
        COHORT_TOP,
        accuracy_with_interval,
        adaptive_s_norm,
    )

    records = endpoint.records
    values = np.asarray(vectors)

    def rows(split: str, condition: str) -> list[int]:
        return [
            i
            for i, r in enumerate(records)
            if r.split == split and str(r.context.get("condition", "foreground")) == condition
        ]

    gallery = rows("enrollment", "foreground")
    cohort = rows("enrollment", "background")
    block: dict[str, Any] = {"cohort_clips": len(cohort), "cohort_top": COHORT_TOP}
    for label, query in (
        ("calls_scored", rows("query", "foreground")),
        ("backgrounds_scored", rows("query", "background")),
    ):
        raw, normalised, order = adaptive_s_norm(
            enrolment=values[gallery],
            enrolment_identities=[records[i].identity for i in gallery],
            query=values[query],
            cohort=values[cohort],
        )
        identities = [records[i].identity for i in query]
        block[label] = {
            "raw_class_mean": accuracy_with_interval(
                scores=raw, order=order, identities=identities, replicates=REPLICATES
            ),
            "as_norm": accuracy_with_interval(
                scores=normalised, order=order, identities=identities, replicates=REPLICATES
            ),
            "chance": 1.0 / len(order),
            "predictions": {
                "labels": order,
                "query_ids": [records[i].filename for i in query],
                "actual_label_indices": [order.index(identity) for identity in identities],
                "raw_class_mean": np.argmax(raw, axis=1).tolist(),
                "as_norm": np.argmax(normalised, axis=1).tolist(),
            },
        }
    return block


def analyse_endpoint(
    endpoint: Endpoint, results: Sequence[dict[str, Any]], *, sources: Sequence[Path]
) -> dict[str, Any]:
    """Beecher's statistic for every stored model, and AS-norm where there are backgrounds."""

    from xinyenyana.a6 import load_vectors
    from xinyenyana.archive import sha256_file
    from xinyenyana.identity_information import beecher_hs
    from xinyenyana.v6_controls import has_backgrounds

    stored = _stored(results)
    held_out = _held_out(results)
    merged = [result for result in results if "merge" in result]
    choices = {
        model: entry for result in results for model, entry in result.get("chosen", {}).items()
    }
    for model, choice in choices.items():
        if model not in stored:
            raise ValueError(f"{model}: no stored vectors for the reported choice")
        if stored[model]["representation"] != choice["representation"]:
            raise ValueError(f"{model}: stored vectors do not match the reported choice")
    foreground = _foreground_rows(endpoint)
    identities = [endpoint.records[i].identity for i in foreground]
    with_backgrounds = has_backgrounds(endpoint)
    models: dict[str, Any] = {}
    for model, entry in sorted(stored.items()):
        vectors, _, representation = load_vectors(_check_digest(entry), endpoint)
        row: dict[str, Any] = {
            "representation": representation,
            "vectors_sha256": entry["sha256"],
            "permutation_nulls": permutation_nulls(endpoint, vectors),
        }
        if merged and model in merged[-1]["chosen"]:
            row["is_the_merged_choice"] = (
                representation == merged[-1]["chosen"][model]["representation"]
            )
        if int(vectors.shape[1]) > 3:
            row["beecher_hs"] = beecher_hs(vectors[foreground], identities)
        if with_backgrounds:
            from xinyenyana.cosine_controls import EXCLUSION_REASON, is_control

            if is_control(model):
                row["as_norm_excluded"] = EXCLUSION_REASON
            else:
                row["as_norm"] = as_norm_block(endpoint, vectors)
        other = held_out.get(model)
        if other is not None:
            chosen, candidates = other
            if chosen == representation:
                row["held_out_rule"] = {"representation": chosen, "same_as_fisher_rule": True}
            else:
                matrix = _candidate(endpoint, candidates, chosen)
                block: dict[str, Any] = {
                    "representation": chosen,
                    "same_as_fisher_rule": False,
                    "permutation_nulls": permutation_nulls(endpoint, matrix),
                }
                if int(matrix.shape[1]) > 3:
                    block["beecher_hs"] = beecher_hs(matrix[foreground], identities)
                row["held_out_rule"] = block
        models[model] = row
    return {
        "experiment": "PA-V6 stored-vector analyses",
        "endpoint": endpoint.name,
        "manifest_sha256": endpoint.manifest_sha256,
        "splits": endpoint.split_digests(),
        "sources": {str(path): sha256_file(path) for path in sources},
        "models": models,
    }


def pairings_endpoint(
    endpoint: Endpoint,
    results: Sequence[dict[str, Any]],
    *,
    sources: Sequence[Path],
    permutations: int = PERMUTATIONS,
) -> dict[str, Any]:
    """The four gallery/query pairings for every reported representation, at the v6 null count.

    The sweeps scored the pairings with the 999 shuffles registered for them; this
    rescores the same stored vectors with the 9,999 used for every other reported
    null, so every permutation p in the paper has the same resolution.
    """

    from xinyenyana.a6 import load_vectors
    from xinyenyana.archive import sha256_file
    from xinyenyana.v6_controls import four_pairings, has_backgrounds

    if not has_backgrounds(endpoint):
        raise ValueError(f"{endpoint.name} has no background recordings to pair")
    stored = _stored(results)
    choices = {
        model: entry for result in results for model, entry in result.get("chosen", {}).items()
    }
    for model, choice in choices.items():
        if model not in stored:
            raise ValueError(f"{model}: no stored vectors for the reported choice")
        if stored[model]["representation"] != choice["representation"]:
            raise ValueError(f"{model}: stored vectors do not match the reported choice")
    models: dict[str, Any] = {}
    for model, entry in sorted(stored.items()):
        vectors, _, representation = load_vectors(_check_digest(entry), endpoint)
        models[model] = {
            "representation": representation,
            "vectors_sha256": entry["sha256"],
            "pairings": four_pairings(
                endpoint=endpoint, vectors=vectors, permutations=permutations
            ),
        }
    return {
        "experiment": "PA-V6 four pairings at the v6 null count",
        "endpoint": endpoint.name,
        "manifest_sha256": endpoint.manifest_sha256,
        "splits": endpoint.split_digests(),
        "permutations": permutations,
        "sources": {str(path): sha256_file(path) for path in sources},
        "models": models,
    }


def open_set_models(results: Sequence[dict[str, Any]]) -> dict[str, str]:
    """The three representations the open set is run for on one endpoint (PA-V6).

    BirdNET v2.4, the bioacoustic model with the highest closed-set accuracy,
    and the speech encoder with the highest closed-set accuracy, each read from
    the ``chosen`` entries of the given results; a tie goes to the name that
    sorts first, so the choice does not depend on file order.
    """

    from xinyenyana.a5 import HIDDEN_STATE_MODELS, SPEAKER_MODELS
    from xinyenyana.bioacoustic import MODELS as BIOACOUSTIC
    from xinyenyana.xeus import MODEL as XEUS

    speech = set(HIDDEN_STATE_MODELS) | set(SPEAKER_MODELS) | {XEUS}
    accuracy: dict[str, float] = {}
    for result in results:
        for model, chosen in result.get("chosen", {}).items():
            accuracy[model] = float(chosen["accuracy"])

    def best(group: set[str]) -> str:
        present = [model for model in accuracy if model in group]
        if not present:
            raise ValueError(f"none of {sorted(group)[:3]}... was scored")
        return min(present, key=lambda model: (-accuracy[model], model))

    if "birdnet-v2.4" not in accuracy:
        raise ValueError("BirdNET v2.4 was not scored")
    return {
        "reference": "birdnet-v2.4",
        "bioacoustic": best(set(BIOACOUSTIC)),
        "speech": best(speech),
    }


def open_set_from_store(
    endpoint: Endpoint,
    results: Sequence[dict[str, Any]],
    *,
    models: Sequence[str],
    sources: Sequence[Path],
) -> dict[str, Any]:
    """``open_set.run_open_set_endpoint`` with stored vectors in place of extraction.

    The sixteen-bird great tit keeps its year-cohort allocation, as registered on
    2026-09-01; every other endpoint, the full-width great tit included, is
    allocated evenly, as the v5 open-set runs were.
    """

    import numpy as np

    from xinyenyana.a6 import load_vectors
    from xinyenyana.archive import canonical_sha256, sha256_file
    from xinyenyana.open_set import (
        ALLOCATION_SEED,
        ALLOCATIONS,
        ROLES,
        allocate_roles,
        allocate_roles_evenly,
        evaluate_allocation,
        summarise_allocations,
    )

    stored = _stored(results)
    rows = _foreground_rows(endpoint)
    records: list[ClipRecord] = [endpoint.records[i] for i in rows]
    by_cohort: dict[str, list[str]] = {}
    for record in records:
        if "cohort" not in record.context:
            by_cohort = {}
            break
        cohort = str(record.context["cohort"])
        if record.identity not in by_cohort.setdefault(cohort, []):
            by_cohort[cohort].append(record.identity)
    balanced = bool(by_cohort) and endpoint.name == "great-tit"
    allocations = (
        [allocate_roles(by_cohort, seed=ALLOCATION_SEED, allocation=i) for i in range(ALLOCATIONS)]
        if balanced
        else [
            allocate_roles_evenly([r.identity for r in records], seed=ALLOCATION_SEED, allocation=i)
            for i in range(ALLOCATIONS)
        ]
    )
    summary: dict[str, Any] = {
        "endpoint": endpoint.name,
        "manifest_sha256": endpoint.manifest_sha256,
        "splits": endpoint.split_digests(),
        "allocation_seed": ALLOCATION_SEED,
        "allocations": ALLOCATIONS,
        "role_digest": canonical_sha256(
            [{role: roles[role] for role in ROLES} for roles in allocations]
        ),
        "allocation_scheme": "year cohorts" if balanced else "even, no cohorts",
        "sources": {str(path): sha256_file(path) for path in sources},
        "representations": {},
    }
    for model in models:
        from xinyenyana.cosine_controls import EXCLUSION_REASON, is_control

        if is_control(model):
            summary.setdefault("excluded_representations", {})[model] = EXCLUSION_REASON
            continue
        choices = {
            name: entry for result in results for name, entry in result.get("chosen", {}).items()
        }
        if model not in stored or model not in choices:
            raise ValueError(f"{model}: no scored and stored representation for open set")
        if stored[model]["representation"] != choices[model]["representation"]:
            raise ValueError(f"{model}: open-set vectors do not match the reported choice")
        vectors, _, representation = load_vectors(_check_digest(stored[model]), endpoint)
        if np.asarray(vectors).ndim != 2 or np.asarray(vectors).shape[1] <= 3:
            raise ValueError(
                f"{model}: raw low-dimensional controls cannot use cosine open-set scoring"
            )
        chosen = np.asarray(vectors)[rows]
        evaluated = [
            evaluate_allocation(
                vectors=chosen,
                records=records,
                roles=roles,
                standardise=False,
                balanced=balanced,
            )
            for roles in allocations
        ]
        summary["representations"][model] = {
            "representation": representation,
            "vector_source": stored[model],
            "allocation_sensitivity": summarise_allocations(evaluated),
            "allocations": evaluated,
        }
    return summary


def merge_sweeps(sweeps: Sequence[dict[str, Any]]) -> dict[str, Any]:
    """Several a5/a6 results for one endpoint as one, candidates united per model.

    Every file must describe the same scoring clips in the same order. A model
    measured in more than one file has its candidates united and its reported
    candidate chosen again by the enrolment Fisher ratio over the union, which is
    what one run holding every candidate would have chosen. BirdNET's per-clip
    correctness must be identical wherever it appears.
    """

    if not sweeps:
        raise ValueError("nothing to merge")
    first = sweeps[0]
    identities = first["query_identities"]
    merged: dict[str, Any] = {
        "endpoint": first["endpoint"],
        "manifest_sha256": first.get("manifest_sha256"),
        "query_identities": identities,
        "curve": {},
        "chosen": {},
        "not_computed": [],
        "merge": {"files": len(sweeps), "models_united": []},
    }
    birdnet = None
    for sweep in sweeps:
        if sweep["query_identities"] != identities:
            raise ValueError(f"{sweep['endpoint']}: the files score different clips")
        if sweep["endpoint"] != first["endpoint"]:
            raise ValueError("merging results from two endpoints")
        if sweep.get("manifest_sha256") != first.get("manifest_sha256"):
            raise ValueError(f"{sweep['endpoint']}: the files were measured on other manifests")
        merged["not_computed"].extend(sweep.get("not_computed", []))
        for model, stores in sweep.get("candidate_vectors", {}).items():
            if stores is not None:
                merged.setdefault("candidate_vectors", {}).setdefault(model, []).append(stores)
        for model, curve in sweep["curve"].items():
            if model in merged["curve"]:
                merged["merge"]["models_united"].append(model)
            merged["curve"].setdefault(model, {}).update(curve)
        reference = sweep["chosen"].get("birdnet-v2.4")
        if reference is not None:
            if birdnet is not None and reference["query_correct"] != birdnet:
                raise ValueError("BirdNET's per-clip correctness differs between files")
            birdnet = reference["query_correct"]
    from xinyenyana.layer_selection import choose as choose_held_out

    merged["chosen_held_out"] = {}
    for model, curve in merged["curve"].items():
        if not curve:
            continue
        best = max(curve, key=lambda name: float(curve[name]["enrollment_fisher_ratio"]))
        merged["chosen"][model] = {"representation": best, **curve[best]}
        held_out = choose_held_out(curve)
        if held_out is not None:
            merged["chosen_held_out"][model] = {"representation": held_out, **curve[held_out]}
    # A gap one file records and another file fills is no longer a gap.
    filled = {
        (model, int(entry["slowdown"]))
        for model, curve in merged["curve"].items()
        for entry in curve.values()
        if entry.get("slowdown") is not None
    }
    merged["not_computed"] = [
        gap
        for gap in merged["not_computed"]
        if (gap.get("model"), gap.get("slowdown")) not in filled
    ]
    return merged


def spectral_subtraction_run(
    endpoint: Endpoint,
    *,
    scratch: Path,
    store: Path | None = None,
    previous_result: Path | None = None,
    permutations: int = 999,
) -> dict[str, Any]:
    """BirdNET's four pairings on the original clips and after subtraction at each alpha."""

    import json
    from dataclasses import replace

    import numpy as np

    from xinyenyana.a5 import birdnet_vectors
    from xinyenyana.a6 import store_vectors
    from xinyenyana.archive import sha256_file
    from xinyenyana.background_probe import _read_mono
    from xinyenyana.frozen_probe import endpoint_audio_provenance
    from xinyenyana.representations import birdnet_v2_4_vectors
    from xinyenyana.spectral_subtraction import ALPHAS, BETA, FRAME, HOP, noise_profile, subtract
    from xinyenyana.v6_controls import four_pairings
    from xinyenyana.validity import PAIRINGS

    config = {"alphas": list(ALPHAS), "beta": BETA, "frame": FRAME, "hop": HOP}
    previous = None
    previous_digest = None
    if previous_result is not None:
        if store is None:
            raise ValueError("replacement spectral measurements require prospective retention")
        previous_digest = sha256_file(previous_result)
        previous = json.loads(previous_result.read_text())
        if (
            previous.get("experiment") != "PA-V6 spectral subtraction"
            or previous.get("endpoint") != endpoint.name
            or previous.get("manifest_sha256") != endpoint.manifest_sha256
            or previous.get("config") != config
        ):
            raise ValueError("previous spectral result has a different endpoint or recipe")
        keys = {f"{gallery}_gallery_{query}_query" for gallery, query in PAIRINGS}
        for arm in ("original", "alpha_1", "alpha_2"):
            rows = previous.get(arm, {})
            if set(rows) != keys or any(
                not {"accuracy", "roc_auc"} <= row.keys() for row in rows.values()
            ):
                raise ValueError("previous spectral result has incomplete point metrics")
        if any(
            not (scratch / f"alpha{alpha:g}" / f"{record.filename}.wav").is_file()
            for alpha in ALPHAS
            for record in endpoint.records
        ):
            raise ValueError("replacement requires all retained treatment waveforms")

    import soundfile

    retained: dict[str, Any] = {}

    def measure_arm(current: Endpoint, arm: str) -> dict[str, Any]:
        if store is None:
            return four_pairings(
                endpoint=current,
                vectors=birdnet_vectors(current.records),
                permutations=permutations,
            )
        inputs = endpoint_audio_provenance(current)
        vectors, extraction = birdnet_v2_4_vectors([r.path for r in current.records])
        matrix = np.asarray(vectors)
        if (
            matrix.shape != (len(current.records), 1024)
            or not np.isfinite(matrix).all()
            or not np.array_equal(matrix, matrix.astype(np.float32))
        ):
            raise ValueError("spectral retention would alter embedding shape or precision")
        entry = store_vectors(
            store=store,
            endpoint=current,
            model="birdnet-v2.4",
            representation="embedding.mean",
            vectors=matrix,
            arms=None,
        )
        predictions: dict[str, Any] = {}
        measured = four_pairings(
            endpoint=current,
            vectors=matrix,
            permutations=permutations,
            retained_predictions=predictions,
        )
        retained[arm] = {
            "inputs": inputs,
            "extraction": extraction,
            "vectors": entry,
            "pairings": predictions,
        }
        return measured

    sessions: dict[tuple[str, str], list[ClipRecord]] = {}
    for record in endpoint.records:
        if str(record.context.get("condition")) == "background":
            sessions.setdefault((record.identity, record.split), []).append(record)

    def signals(members: Sequence[ClipRecord], rate: int) -> Any:
        for member in members:
            signal, member_rate = _read_mono(member.path)
            if member_rate != rate:
                raise ValueError(f"{member.filename}: session mixes sample rates")
            yield np.asarray(signal)

    profiles: dict[tuple[str, str], tuple[Any, int]] = {}
    for key, members in sorted(sessions.items()):
        rate = int(soundfile.info(members[0].path).samplerate)
        profiles[key] = (noise_profile(signals(members, rate), rate), rate)
    result: dict[str, Any] = {
        "experiment": "PA-V6 spectral subtraction",
        "endpoint": endpoint.name,
        "manifest_sha256": endpoint.manifest_sha256,
        "config": config,
        "permutations": permutations,
        "original": measure_arm(endpoint, "original"),
    }
    for alpha in ALPHAS:
        cleaned: list[ClipRecord] = []
        for record in endpoint.records:
            key = (record.identity, record.split)
            if key not in profiles:
                raise ValueError(f"{record.filename} has no session backgrounds")
            profile, rate = profiles[key]
            signal, record_rate = _read_mono(record.path)
            if record_rate != rate:
                raise ValueError(f"{record.filename} is {record_rate} Hz, its session {rate}")
            path = scratch / f"alpha{alpha:g}" / f"{record.filename}.wav"
            if not path.exists():
                path.parent.mkdir(parents=True, exist_ok=True)
                out = subtract(np.asarray(signal), profile, rate, alpha=alpha)
                peak = float(np.max(np.abs(out))) if len(out) else 0.0
                if peak > 0.99:
                    out = out * (0.95 / peak)
                soundfile.write(str(path), out, rate, subtype="PCM_16")
            cleaned.append(replace(record, path=path))
        arm = f"alpha_{alpha:g}"
        result[arm] = measure_arm(replace(endpoint, records=tuple(cleaned)), arm)
    if store is not None:
        result["retention_mode"] = "prospective"
        result["retained_arms"] = retained
    if previous is not None:
        assert previous_result is not None
        if sha256_file(previous_result) != previous_digest:
            raise ValueError("previous spectral result changed during measurement")
        result["previous_result"] = {"path": str(previous_result), "sha256": previous_digest}
        result["previous_point_metric_comparison"] = {
            arm: {
                pairing: {
                    metric: {
                        "previous": old[metric],
                        "current": result[arm][pairing][metric],
                        "difference": result[arm][pairing][metric] - old[metric],
                    }
                    for metric in ("accuracy", "roc_auc")
                }
                for pairing, old in previous[arm].items()
            }
            for arm in ("original", "alpha_1", "alpha_2")
        }
    return result


def sensitivity_floor_run(donor: Endpoint, target: Endpoint, *, scratch: Path) -> dict[str, Any]:
    """The planted endpoint at each level, scored as the headline is scored."""

    from xinyenyana.a5 import birdnet_vectors
    from xinyenyana.background_challenge import CHALLENGE_DB
    from xinyenyana.identity_run import measure
    from xinyenyana.sensitivity_floor import pair_donors, planted_endpoint

    result: dict[str, Any] = {
        "experiment": "PA-V6 sensitivity floor",
        "donor": donor.name,
        "target": target.name,
        "donor_manifest_sha256": donor.manifest_sha256,
        "target_manifest_sha256": target.manifest_sha256,
        "pairs": [list(pair) for pair in pair_donors(donor, target)],
        "levels": {},
    }
    for level in CHALLENGE_DB:
        planted = planted_endpoint(donor, target, level_db=level, scratch=scratch / f"{level:+g}")
        vectors = birdnet_vectors(planted.records)
        measured = measure(
            endpoint=planted,
            records=list(planted.records),
            vectors=vectors,
            replicates=REPLICATES,
            permutations=PERMUTATIONS,
        )
        result["levels"][f"{level:+g}db"] = measured
    return result
