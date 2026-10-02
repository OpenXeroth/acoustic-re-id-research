"""PA-V6: every model against BirdNET, with the vectors kept for the controls.

A5 compared BirdNET v2.4 with sixteen speech and speaker encoders. PA-V6 adds
the bioacoustic models a reader of this literature expects (``bioacoustic``),
one more speech encoder (XEUS, ``xeus``), and the rook at full width. It uses
A5's machinery unchanged wherever A5 already decides something: the head, the
nulls, the context probes, the enrolment Fisher ratio that chooses a model's
reported layer and rate, and the paired comparison against BirdNET.

What is new here:

* **Candidates of a bioacoustic model.** The model's own embedding, as its
  publishers pool it, and for transformer models the output of every block,
  averaged over tokens and then over windows. All candidates of one model come
  from one pass over the audio.
* **The chosen vectors are kept.** The chosen candidate's vectors over every
  clip of the endpoint, backgrounds included, are written to a store on xen1.
  Beecher's statistic, the open set, score normalisation and the recording
  controls then read them, so no model is run twice for them.
* **Controls in the same pass.** On an endpoint with ambient recordings the
  chosen candidate is also computed for every added-background arm, and the
  four pairings and the challenge are evaluated before the model is released.
* **A fixed representation.** A model can be asked for one named
  representation instead of a sweep. That is how the sixteen v5 speech
  encoders are re-extracted at the layer and rate v5 chose for them, and the
  accuracy recomputed from those vectors must equal the one v5 archived.
"""

from __future__ import annotations

import json
import os
import tempfile
from collections.abc import Iterator, Sequence
from pathlib import Path
from typing import Any

from xinyenyana.a2 import ClipRecord, Endpoint
from xinyenyana.a5 import (
    HIDDEN_STATE_MODELS,
    SEED,
    evaluate_representation,
    fisher_ratio,
    representation_from_name,
    representations_of,
    vectors_for,
)
from xinyenyana.a5 import MODELS as A5_MODELS
from xinyenyana.layer_selection import choose as choose_held_out
from xinyenyana.layer_selection import held_out_accuracy

STORE_SCHEMA = "xyy-v6-vectors-1"


def _publish_vectors(directory: Path, payload: dict[str, Any]) -> tuple[Path, str]:
    """Publish complete, immutable bytes so concurrent sweeps cannot replace a store."""

    import numpy as np

    from xinyenyana.archive import sha256_file

    directory.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=directory, suffix=".partial", delete=False) as handle:
        temporary = Path(handle.name)
        try:
            np.savez(handle, **payload)
            handle.flush()
            os.fsync(handle.fileno())
            digest = sha256_file(temporary)
            destination = directory / f"{digest}.npz"
            try:
                os.link(temporary, destination)
            except FileExistsError:
                if sha256_file(destination) != digest:
                    raise ValueError(f"corrupt vector object: {destination}") from None
            return destination, digest
        finally:
            temporary.unlink(missing_ok=True)


def _unconfigured_producers(
    model: str,
    records: Sequence[ClipRecord],
    *,
    device: str,
    not_computed: list[dict[str, Any]],
    fixed: str | None,
    slowdowns: Sequence[int] | None = None,
    window_seconds: float | None = None,
    fixed_window: float | None = None,
) -> Iterator[tuple[str, dict[str, Any], Any]]:
    """(candidate name, description, vectors) for every candidate of a model."""

    from xinyenyana import bioacoustic, xeus

    if model in bioacoustic.MODELS:
        candidates = bioacoustic.candidate_vectors(model, records, device=device)
        for name, vectors in candidates.items():
            if fixed is not None and name != fixed:
                continue
            yield name, {"slowdown": 1, "layer": bioacoustic.layer_of(name)}, vectors
        return
    if model == xeus.MODEL:
        yield from xeus.candidate_vectors(
            records, device=device, not_computed=not_computed, fixed=fixed
        )
        return
    if model not in A5_MODELS:
        raise ValueError(f"no producer for {model!r}")
    if fixed is not None:
        representation = representation_from_name(fixed)
        yield (
            representation.name,
            {
                "slowdown": representation.slowdown,
                "layer": representation.layer,
                "read_in_windows_of_seconds": fixed_window,
            },
            vectors_for(representation, records, device=device, window_seconds=fixed_window),
        )
        return
    extra: dict[str, Any] = {}
    if slowdowns is not None:
        extra["slowdowns"] = tuple(slowdowns)
    if window_seconds is not None:
        extra["window_seconds"] = window_seconds
    for representation, vectors in representations_of(
        model, records, device=device, not_computed=not_computed, **extra
    ):
        yield (
            representation.name,
            {
                "slowdown": representation.slowdown,
                "layer": representation.layer,
                "read_in_windows_of_seconds": window_seconds,
            },
            vectors,
        )


def _producers(
    model: str,
    records: Sequence[ClipRecord],
    *,
    matmul_precision: str = "tf32",
    **options: Any,
) -> Iterator[tuple[str, dict[str, Any], Any]]:
    """Bind speech arithmetic independently of skipped earlier checkpoints."""
    if model not in HIDDEN_STATE_MODELS:
        yield from _unconfigured_producers(model, records, **options)
        return
    from xinyenyana.precision import speech_matmul

    with speech_matmul(matmul_precision) as context:
        for name, description, vectors in _unconfigured_producers(model, records, **options):
            yield name, {**description, "numeric_precision": context}, vectors


def arm_vectors(
    model: str,
    chosen: str,
    arms: dict[str, list[ClipRecord]],
    *,
    device: str,
    window: float | None = None,
    matmul_precision: str = "tf32",
) -> dict[str, Any]:
    """The chosen candidate over every added-background arm."""

    result: dict[str, Any] = {}
    for arm, records in arms.items():
        produced = list(
            _producers(
                model,
                records,
                device=device,
                not_computed=[],
                fixed=chosen,
                fixed_window=window,
                matmul_precision=matmul_precision,
            )
        )
        if len(produced) != 1:
            raise RuntimeError(f"{model} produced {len(produced)} vectors for {chosen} on {arm}")
        result[arm] = produced[0][2]
    return result


def store_vectors(
    *,
    store: Path,
    endpoint: Endpoint,
    model: str,
    representation: str,
    vectors: Any,
    arms: dict[str, Any] | None,
) -> dict[str, Any]:
    """Write one model's chosen vectors; returns the path and digest to record."""

    import numpy as np

    directory = store / endpoint.name
    payload: dict[str, Any] = {
        "vectors": np.asarray(vectors, dtype=np.float32),
        "filenames": np.asarray([r.filename for r in endpoint.records]),
        "representation": np.asarray(representation),
        "schema": np.asarray(STORE_SCHEMA),
        "manifest_sha256": np.asarray(endpoint.manifest_sha256),
        "model": np.asarray(model),
    }
    for arm, matrix in (arms or {}).items():
        payload[f"arm__{arm}"] = np.asarray(matrix, dtype=np.float32)
    path, digest = _publish_vectors(directory, payload)
    return {"path": str(path), "sha256": digest, "representation": representation}


def store_candidates(
    *, store: Path, endpoint: Endpoint, model: str, candidates: dict[str, Any]
) -> dict[str, Any]:
    """Every candidate of one model over every clip, so a later rule needs no extraction."""

    import numpy as np

    directory = store / endpoint.name / "candidates"
    names = sorted(candidates)
    payload: dict[str, Any] = {
        f"candidate__{index:03d}": np.asarray(candidates[name], dtype=np.float32)
        for index, name in enumerate(names)
    }
    payload["names"] = np.asarray(names)
    payload["filenames"] = np.asarray([r.filename for r in endpoint.records])
    payload["schema"] = np.asarray(STORE_SCHEMA)
    payload["manifest_sha256"] = np.asarray(endpoint.manifest_sha256)
    payload["model"] = np.asarray(model)
    path, digest = _publish_vectors(directory, payload)
    return {"path": str(path), "sha256": digest, "candidates": len(names)}


def load_candidate(path: Path, endpoint: Endpoint, name: str) -> Any:
    """One stored candidate's matrix, checked against the endpoint."""

    import numpy as np

    with np.load(path, allow_pickle=False) as data:
        if str(data["schema"]) != STORE_SCHEMA:
            raise ValueError(f"{path} is not a v6 vector store")
        if str(data["manifest_sha256"]) != endpoint.manifest_sha256:
            raise ValueError(f"{path} was written for another manifest")
        if [str(v) for v in data["filenames"]] != [r.filename for r in endpoint.records]:
            raise ValueError(f"{path} does not follow this endpoint's record order")
        names = [str(v) for v in data["names"]]
        return data[f"candidate__{names.index(name):03d}"]


def load_vectors(path: Path, endpoint: Endpoint) -> tuple[Any, dict[str, Any], str]:
    """A stored matrix, its arms and its representation, checked against the endpoint."""

    import numpy as np

    with np.load(path, allow_pickle=False) as data:
        if str(data["schema"]) != STORE_SCHEMA:
            raise ValueError(f"{path} is not a v6 vector store")
        if str(data["manifest_sha256"]) != endpoint.manifest_sha256:
            raise ValueError(f"{path} was written for another manifest")
        order = [str(v) for v in data["filenames"]]
        if order != [r.filename for r in endpoint.records]:
            raise ValueError(f"{path} does not follow this endpoint's record order")
        arms = {key[5:]: data[key] for key in data.files if key.startswith("arm__")}
        return data["vectors"], arms, str(data["representation"])


def run_a6_endpoint(
    *,
    endpoint: Endpoint,
    models: Sequence[str],
    store: Path,
    device: str = "cuda",
    checkpoint_dir: Path | None = None,
    fixed: dict[str, str] | None = None,
    arms_scratch: Path | None = None,
    slowdowns: Sequence[int] | None = None,
    window_seconds: float | None = None,
    with_controls: bool = True,
    fixed_windows: dict[str, float | None] | None = None,
    speech_precision: dict[str, str] | None = None,
) -> dict[str, Any]:
    """Every candidate of every listed model, the chosen vectors stored, controls run.

    ``slowdowns`` and ``window_seconds`` restrict and alter the speech path; they
    exist for filling the combinations v5 could not compute, and every candidate
    they produce says so in its description. ``with_controls`` off skips the
    challenge and pairings (used by gap-filling runs, whose chosen candidate is
    only known after merging with the v5 sweep).
    """

    import numpy as np

    from xinyenyana.a2 import clip_diagnostics
    from xinyenyana.archive import canonical_sha256, source_digest
    from xinyenyana.v6_controls import challenge, four_pairings, has_backgrounds, write_arms

    fixed = dict(fixed or {})
    fixed_windows = dict(fixed_windows or {})
    precision = {model: "tf32" for model in models if model in HIDDEN_STATE_MODELS}
    for model, choice in (speech_precision or {}).items():
        if model not in precision or choice not in {"fp32", "tf32"}:
            raise ValueError(f"invalid speech precision override: {model}={choice}")
        precision[model] = choice
    fingerprint = canonical_sha256(
        {
            "speech_precision": precision,
            "fixed_windows": fixed_windows,
            "manifest": endpoint.manifest_sha256,
            "splits": endpoint.split_digests(),
            "source": source_digest(Path(__file__).resolve().parent),
            "fixed": fixed,
            "slowdowns": list(slowdowns) if slowdowns is not None else None,
            "window_seconds": window_seconds,
            "with_controls": with_controls,
        }
    )
    diagnostics = clip_diagnostics(endpoint.records)
    enrolment = [
        i
        for i, r in enumerate(endpoint.records)
        if r.split == "enrollment" and str(r.context.get("condition", "foreground")) == "foreground"
    ]
    enrolment_identities = [endpoint.records[i].identity for i in enrolment]
    # Missing recording metadata cannot justify a random division over clips.
    enrolment_sessions = [endpoint.records[i].session for i in enrolment]
    with_controls = with_controls and has_backgrounds(endpoint)
    arms_records = None
    if with_controls:
        if arms_scratch is None:
            raise ValueError(f"{endpoint.name} has backgrounds, so the arms need a scratch folder")
        arms_records = write_arms(endpoint, arms_scratch / endpoint.name)
    summary: dict[str, Any] = {
        "experiment": "PA-V6",
        "endpoint": endpoint.name,
        "chance_accuracy": 1.0 / len(endpoint.identities),
        "clips": len(endpoint.records),
        "identities": len(endpoint.identities),
        "manifest_sha256": endpoint.manifest_sha256,
        "splits": endpoint.split_digests(),
        "seed": SEED,
        "device": device,
        "models": list(models),
        "fixed_representations": fixed,
        "speech_precision": precision,
        "slowdowns": list(slowdowns) if slowdowns is not None else None,
        "window_seconds": window_seconds,
        "curve": {},
        "chosen": {},
        "chosen_held_out": {},
        "enrolment_clips_with_a_named_session": sum(
            1 for i in enrolment if endpoint.records[i].session is not None
        ),
        "enrolment_clips": len(enrolment),
        "candidate_vectors": {},
        "stored_vectors": {},
        "controls": {},
        "query_identities": [
            r.identity
            for r in endpoint.records
            if r.split == "query" and str(r.context.get("condition", "foreground")) == "foreground"
        ],
        "not_computed": [],
    }
    not_computed: list[dict[str, Any]] = summary["not_computed"]
    seed = SEED
    for model in models:
        stored = (
            checkpoint_dir / f"{fingerprint[:16]}-{model.replace('/', '_')}.json"
            if checkpoint_dir is not None
            else None
        )
        if stored is not None and stored.exists():
            saved = json.loads(stored.read_text())
            if saved.get("fingerprint") == fingerprint:
                for key in (
                    "curve",
                    "chosen",
                    "chosen_held_out",
                    "candidate_vectors",
                    "stored_vectors",
                    "controls",
                ):
                    if saved.get(key) is not None:
                        summary[key][model] = saved[key]
                not_computed.extend(saved["not_computed"])
                seed = int(saved["seed_after"])
                continue
        dropped_before = len(not_computed)
        candidates: list[tuple[float, str, dict[str, Any], Any]] = []
        for name, description, vectors in _producers(
            model,
            endpoint.records,
            device=device,
            not_computed=not_computed,
            fixed=fixed.get(model),
            slowdowns=slowdowns,
            window_seconds=window_seconds,
            fixed_window=fixed_windows.get(model),
            matmul_precision=precision.get(model, "tf32"),
        ):
            matrix = np.asarray(vectors)
            ratio = fisher_ratio(matrix[enrolment], enrolment_identities)
            description = {
                **description,
                "enrolment_held_out": held_out_accuracy(
                    matrix[enrolment], enrolment_identities, enrolment_sessions
                ),
            }
            candidates.append((ratio, name, description, matrix))
        results: dict[str, Any] = {}
        for ratio, name, description, matrix in candidates:
            seed += 1
            evaluated = evaluate_representation(
                endpoint=endpoint, vectors=matrix, name=name, seed=seed, diagnostics=diagnostics
            )
            results[name] = {
                **description,
                "enrollment_fisher_ratio": ratio,
                "accuracy": evaluated["classification"]["accuracy"],
                "identity_block_bootstrap_accuracy_95": evaluated[
                    "identity_block_bootstrap_accuracy_95"
                ],
                "permutation_p": evaluated["permutation_control"]["p_value_plus_one"],
                "roc_auc": evaluated["verification"]["roc_auc"],
                "context_probes": evaluated["context_probes"],
                "standardisation": evaluated["standardisation"],
                "query_correct": evaluated["query_correct"],
            }
        summary["curve"][model] = results
        held_out_name = choose_held_out(results)
        held_out_entry = (
            None
            if held_out_name is None
            else {"representation": held_out_name, **results[held_out_name]}
        )
        if held_out_entry is not None:
            summary["chosen_held_out"][model] = held_out_entry
        candidates_entry = (
            store_candidates(
                store=store,
                endpoint=endpoint,
                model=model,
                candidates={name: matrix for _, name, _, matrix in candidates},
            )
            if candidates
            else None
        )
        if candidates_entry is not None:
            summary["candidate_vectors"][model] = candidates_entry
        chosen_entry = None
        vectors_entry = None
        controls_entry = None
        if candidates:
            best = max(candidates, key=lambda entry: entry[0])
            chosen_entry = {"representation": best[1], **results[best[1]]}
            summary["chosen"][model] = chosen_entry
            arm_matrices = None
            if with_controls and arms_records is not None:
                # The arms are read as the chosen candidate was: in windows
                # only if it was measured in windows.
                window = best[2].get("read_in_windows_of_seconds")
                arm_matrices = arm_vectors(
                    model,
                    best[1],
                    arms_records,
                    device=device,
                    window=window,
                    matmul_precision=precision.get(model, "tf32"),
                )
                controls_entry = {
                    "four_pairings": four_pairings(endpoint=endpoint, vectors=best[3]),
                    "challenge": challenge(
                        endpoint=endpoint, vectors=best[3], arm_vectors=arm_matrices, name=best[1]
                    ),
                }
                summary["controls"][model] = controls_entry
            vectors_entry = store_vectors(
                store=store,
                endpoint=endpoint,
                model=model,
                representation=best[1],
                vectors=best[3],
                arms=arm_matrices,
            )
            summary["stored_vectors"][model] = vectors_entry
        del candidates
        if stored is not None:
            stored.parent.mkdir(parents=True, exist_ok=True)
            checkpoint_text = json.dumps(
                {
                    "fingerprint": fingerprint,
                    "curve": results,
                    "chosen": chosen_entry,
                    "chosen_held_out": held_out_entry,
                    "candidate_vectors": candidates_entry,
                    "stored_vectors": vectors_entry,
                    "controls": controls_entry,
                    "not_computed": not_computed[dropped_before:],
                    "seed_after": seed,
                }
            )
            with tempfile.NamedTemporaryFile(
                mode="w", dir=stored.parent, suffix=".partial", delete=False
            ) as handle:
                temporary = Path(handle.name)
                try:
                    handle.write(checkpoint_text)
                    handle.flush()
                    os.fsync(handle.fileno())
                    temporary.replace(stored)
                finally:
                    temporary.unlink(missing_ok=True)
    summary["ranking"] = sorted(
        (
            {
                "model": model,
                "representation": chosen["representation"],
                "accuracy": chosen["accuracy"],
                "identity_block_bootstrap_accuracy_95": chosen[
                    "identity_block_bootstrap_accuracy_95"
                ],
                "permutation_p": chosen["permutation_p"],
            }
            for model, chosen in summary["chosen"].items()
        ),
        key=lambda entry: (-float(entry["accuracy"]), str(entry["model"])),
    )
    return summary


def fixed_from_sweep(sweep: dict[str, Any], models: Sequence[str]) -> dict[str, str]:
    """The representation an earlier sweep chose for each of these models."""

    return {model: str(sweep["chosen"][model]["representation"]) for model in models}


def windows_from_sweep(sweep: dict[str, Any], models: Sequence[str]) -> dict[str, float | None]:
    """For each model, the window its reported candidate was read in, if any."""

    return {model: sweep["chosen"][model].get("read_in_windows_of_seconds") for model in models}


def check_against_sweep(new: dict[str, Any], old: dict[str, Any]) -> dict[str, Any]:
    """Whether every fixed representation reproduced the accuracy the old sweep archived."""

    rows = {}
    for model, chosen in new["chosen"].items():
        before = old["chosen"].get(model)
        if before is None:
            continue
        rows[model] = {
            "representation": chosen["representation"],
            "v5_accuracy": before["accuracy"],
            "v6_accuracy": chosen["accuracy"],
            "identical": chosen["representation"] == before["representation"]
            and abs(float(chosen["accuracy"]) - float(before["accuracy"])) < 1e-12
            and chosen["query_correct"] == before["query_correct"],
        }
    return rows


def compare_sweep_candidates(old: dict[str, Any], new: dict[str, Any]) -> dict[str, Any]:
    """Check every old candidate against the new sweep.

    It passes when no old candidate is missing and every representation the old
    sweep reported gives the same accuracy and the same answer on every query.
    Other candidates that differ are listed with both accuracies; they are
    reported, not waived.
    """

    for key in ("endpoint", "manifest_sha256", "query_identities"):
        if old.get(key) != new.get(key):
            raise ValueError(f"cannot compare sweeps with different {key}")
    compared = identical = total = 0
    missing = []
    differing = []
    for model, curve in old["curve"].items():
        for name, entry in curve.items():
            total += 1
            other = new.get("curve", {}).get(model, {}).get(name)
            if other is None:
                missing.append({"model": model, "candidate": name})
                continue
            compared += 1
            if (
                abs(float(entry["accuracy"]) - float(other["accuracy"])) < 1e-12
                and entry.get("query_correct") is not None
                and entry["query_correct"] == other.get("query_correct")
            ):
                identical += 1
            else:
                differing.append(
                    {
                        "model": model,
                        "candidate": name,
                        "v5_accuracy": entry["accuracy"],
                        "v6_accuracy": other["accuracy"],
                    }
                )
    different = {(row["model"], row["candidate"]) for row in differing}
    reported = {
        model: choice["representation"]
        for model, choice in old.get("chosen", {}).items()
        if model in old["curve"]
    }
    reported_differing = [
        {"model": model, "candidate": name}
        for model, name in sorted(reported.items())
        if (model, name) in different
    ]
    return {
        "candidates_expected": total,
        "candidates_compared": compared,
        "identical": identical,
        "missing": missing,
        "differing": differing,
        "reported_representations_checked": len(reported),
        "reported_differing": reported_differing,
        "passed": total > 0 and not missing and not reported_differing,
    }


def refresh_selection(endpoint: Endpoint, original: dict[str, Any]) -> dict[str, Any]:
    """Replay only affected enrolment folds; retain every query score and Fisher choice."""

    import copy

    import numpy as np

    from xinyenyana.v6_analyses import _check_digest

    if (
        original["endpoint"] != endpoint.name
        or original["manifest_sha256"] != endpoint.manifest_sha256
    ):
        raise ValueError("selection correction requires the original endpoint and manifest")
    if original.get("splits") != endpoint.split_digests():
        raise ValueError("selection correction requires the original split and enrolment cap")
    result = copy.deepcopy(original)
    rows = [
        i
        for i, r in enumerate(endpoint.records)
        if r.split == "enrollment" and r.context.get("condition", "foreground") == "foreground"
    ]
    labels = [endpoint.records[i].identity for i in rows]
    sessions = [endpoint.records[i].session for i in rows]
    counts = {
        animal: len({s for a, s in zip(labels, sessions, strict=True) if a == animal})
        for animal in set(labels)
    }
    missing_sessions = any(s is None for s in sessions)
    affected = missing_sessions or (
        any(n == 1 for n in counts.values()) and any(n > 1 for n in counts.values())
    )
    incomplete = {a for a, s in zip(labels, sessions, strict=True) if s is None}
    no_eligible = not any(n >= 2 and a not in incomplete for a, n in counts.items())
    changed = []
    recomputed = 0
    for model, curve in result["curve"].items():
        candidate_paths: dict[str, Path] = {}
        if not no_eligible and (
            affected or any("enrolment_held_out" not in entry for entry in curve.values())
        ):
            stores = original.get("candidate_vectors", {}).get(model)
            if stores is None:
                raise ValueError(f"{model}: no retained candidates to correct selection")
            for store in stores if isinstance(stores, list) else [stores]:
                path = _check_digest(store)
                with np.load(path, allow_pickle=False) as data:
                    candidate_paths.update({str(name): path for name in data["names"]})
        for name, entry in curve.items():
            if affected or "enrolment_held_out" not in entry:
                if no_eligible:
                    # The rule cannot score any animal. No vector is needed
                    # to report that absence of session-held-out evidence.
                    entry["enrolment_held_out"] = held_out_accuracy(
                        np.zeros((len(rows), 1)), labels, sessions
                    )
                    recomputed += 1
                    continue
                if name not in candidate_paths:
                    raise ValueError(f"{model}: no retained vectors for {name}")
                matrix = load_candidate(candidate_paths[name], endpoint, name)
                entry["enrolment_held_out"] = held_out_accuracy(matrix[rows], labels, sessions)
                recomputed += 1
        name = choose_held_out(curve)
        old = original.get("chosen_held_out", {}).get(model, {}).get("representation")
        result.setdefault("chosen_held_out", {}).pop(model, None)
        if name is not None:
            result["chosen_held_out"][model] = {"representation": name, **curve[name]}
        if name != old:
            changed.append({"model": model, "before": old, "after": name})
        fisher = result.get("chosen", {}).get(model)
        if fisher is not None:
            fisher["enrolment_held_out"] = curve[fisher["representation"]]["enrolment_held_out"]
    result["selection_correction"] = {
        "reason": "retain single-session competitors; do not substitute clips for missing sessions",
        "selection_metadata_affected": affected,
        "candidates_recomputed": recomputed,
        "changed_choices": changed,
        "original_code_revision": original.get("code_revision"),
        "original_source_sha256": original.get("source_sha256"),
        "original_environment": original.get("environment"),
        "query_scores_recomputed": False,
    }
    return result
