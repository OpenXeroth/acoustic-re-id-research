"""The PA-V6 commands, registered on the one command-line application.

They are kept beside ``cli.py`` rather than inside it because they share one
thing that the older commands do not: they read and write the stored vectors
of ``a6``. Each is registered in ``docs/measurement-protocol.md`` (PA-V6).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Annotated, Any

import typer

from xinyenyana.cli import (
    ROOK_FULL_WIDTH,
    _load_endpoint,
    _require_lease_and_archive,
    app,
)
from xinyenyana.cli import (
    _emit as _emit_base,
)


def _emit(payload: Any, output: Path, *, require_archive: bool = True) -> None:
    from xinyenyana.frozen_probe import runtime_record

    _emit_base(
        {**payload, "environment": runtime_record()}, output, require_archive=require_archive
    )


def _endpoint(name: str, sample_root: Path, balance_enrolment: int | None) -> Any:
    from dataclasses import replace

    from xinyenyana.great_tit_full import ENDPOINT as GREAT_TIT_FULL
    from xinyenyana.great_tit_full import load_great_tit_full

    if name == GREAT_TIT_FULL:
        loaded = load_great_tit_full(sample_root)
    else:
        loaded = _load_endpoint(name, sample_root)
    if balance_enrolment is not None:
        from xinyenyana.identity_run import balance_enrolment as balance

        records = balance(list(loaded.records), balance_enrolment)
        loaded = replace(loaded, records=tuple(records))
    return loaded


@app.command("audit-v6-model-loads")
def audit_v6_model_loads_command(
    model: Annotated[list[str], typer.Option(help="Registered model, once per model")],
    sweep: Annotated[list[Path], typer.Option(help="Measured candidate source")],
    weights: Annotated[Path, typer.Option(help="Archived cache-weight inventory")],
    output: Annotated[Path, typer.Option(help="New retrospective load record")],
) -> None:
    """Record input rates, windows and named blocks without reading experiment audio."""
    import os

    from xinyenyana.archive import sha256_file
    from xinyenyana.bioacoustic import describe
    from xinyenyana.frozen_probe import runtime_record
    from xinyenyana.model_load_audit import audit_loads, verify_weight_inventory

    _require_lease_and_archive()
    if output.exists() or len(set(sweep)) != len(sweep):
        raise typer.BadParameter("use a new output and distinct measured sources")
    inventory = json.loads(weights.read_text())
    verified_weights = verify_weight_inventory(inventory)
    sources = {str(path): json.loads(path.read_text()) for path in sweep}
    # TensorFlow may otherwise discover the GPU despite an adapter's CPU setting.
    os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
    result = audit_loads(
        model, sources, runtime_record()["packages"], lambda name: describe(name, device="cpu")
    )
    result["sources"] = {str(path): sha256_file(path) for path in sweep}
    result["weight_inventory"] = {
        "path": str(weights),
        "sha256": sha256_file(weights),
        **verified_weights,
    }
    _emit(result, output)


@app.command("run-a6")
def run_a6_command(
    endpoint: Annotated[str, typer.Option(help="Endpoint name")],
    sample_root: Annotated[Path, typer.Option(help="That endpoint's built sample")],
    output: Annotated[Path, typer.Option(help="Result file")],
    store: Annotated[Path, typer.Option(help="Where the chosen vectors are kept on xen1")],
    model: Annotated[list[str], typer.Option(help="A model to run, once per model")],
    arms_scratch: Annotated[
        Path | None, typer.Option(help="Where the added-background waveforms are written")
    ] = None,
    checkpoint_dir: Annotated[
        Path | None, typer.Option(help="Keep each finished model here, so a displaced run resumes")
    ] = None,
    fixed_from: Annotated[
        Path | None,
        typer.Option(help="A sweep whose chosen representation each listed model is fixed to"),
    ] = None,
    balance_enrolment: Annotated[
        int | None, typer.Option(help="Cap every animal's enrolment at this many clips")
    ] = None,
    slowdown: Annotated[
        list[int] | None, typer.Option(help="Speech models: only these playback rates")
    ] = None,
    window_seconds: Annotated[
        float | None, typer.Option(help="Speech models: read long clips in windows of this")
    ] = None,
    controls: Annotated[bool, typer.Option(help="Run the recording controls")] = True,
    speech_precision: Annotated[
        list[str] | None,
        typer.Option(help="Speech transformer override MODEL=fp32 or MODEL=tf32; default tf32"),
    ] = None,
    device: Annotated[str, typer.Option(help="Torch device")] = "cuda",
) -> None:
    """Every candidate of every listed model, chosen vectors stored, controls run."""

    from xinyenyana.a6 import (
        check_against_sweep,
        fixed_from_sweep,
        run_a6_endpoint,
        windows_from_sweep,
    )
    from xinyenyana.frozen_probe import endpoint_audio_provenance

    _require_lease_and_archive()
    if endpoint == ROOK_FULL_WIDTH and balance_enrolment is None:
        raise typer.BadParameter("the registered rook sweep caps enrolment at 90 clips")
    loaded = _endpoint(endpoint, sample_root, balance_enrolment)
    inputs = endpoint_audio_provenance(loaded)
    fixed = None
    fixed_windows = None
    old = None
    precision = {}
    for item in speech_precision or []:
        model_name, separator, choice = item.partition("=")
        if not separator or choice not in {"fp32", "tf32"} or model_name in precision:
            raise typer.BadParameter("use one MODEL=fp32 or MODEL=tf32 entry per model")
        precision[model_name] = choice
    if fixed_from is not None:
        old = json.loads(fixed_from.read_text())
        if old["manifest_sha256"] != loaded.manifest_sha256:
            raise typer.BadParameter(f"{fixed_from} was measured on another manifest")
        fixed = fixed_from_sweep(old, model)
        fixed_windows = windows_from_sweep(old, model)
        for model_name in model:
            recorded = old["chosen"][model_name].get("numeric_precision", {}).get("speech_matmul")
            if recorded is not None:
                if model_name in precision and precision[model_name] != recorded:
                    raise typer.BadParameter(
                        "fixed extraction must retain recorded speech precision"
                    )
                precision[model_name] = recorded
    summary = run_a6_endpoint(
        endpoint=loaded,
        models=model,
        store=store,
        device=device,
        checkpoint_dir=checkpoint_dir,
        fixed=fixed,
        arms_scratch=arms_scratch,
        slowdowns=slowdown,
        window_seconds=window_seconds,
        with_controls=controls,
        fixed_windows=fixed_windows,
        speech_precision=precision,
    )
    if balance_enrolment is not None:
        summary["balanced_enrolment_cap"] = balance_enrolment
    summary["inputs"] = inputs
    if old is not None and fixed_from is not None:
        from xinyenyana.archive import sha256_file

        summary["fixed_from"] = {"path": str(fixed_from), "sha256": sha256_file(fixed_from)}
        summary["reproduces_fixed_sweep"] = check_against_sweep(summary, old)
    _emit(summary, output, require_archive=True)


@app.command("refresh-v6-selection")
def refresh_v6_selection_command(
    endpoint: Annotated[str, typer.Option(help="Endpoint name")],
    sample_root: Annotated[Path, typer.Option(help="That endpoint's built sample")],
    sweep: Annotated[Path, typer.Option(help="Existing sweep with stored candidates")],
    output: Annotated[Path, typer.Option(help="New result; the original is retained")],
    balance_enrolment: Annotated[int | None, typer.Option(help="Original enrolment cap")] = None,
) -> None:
    """Correct the held-out rule from retained vectors without rerunning an encoder."""

    from xinyenyana.a6 import refresh_selection
    from xinyenyana.archive import sha256_file

    _require_lease_and_archive()
    if sweep.resolve() == output.resolve() or output.exists():
        raise typer.BadParameter("selection correction requires a new output path")
    loaded = _endpoint(endpoint, sample_root, balance_enrolment)
    original = json.loads(sweep.read_text())
    summary = refresh_selection(loaded, original)
    summary["selection_correction"]["source_result"] = {
        "path": str(sweep),
        "sha256": sha256_file(sweep),
    }
    _emit(summary, output)


@app.command("apply-v6-replay-corrections")
def apply_v6_replay_corrections_command(
    reference: Annotated[Path, typer.Option(help="Archived v5 sweep")],
    base: Annotated[Path, typer.Option(help="Original v6 replay; retained unchanged")],
    correction: Annotated[list[Path], typer.Option(help="Measured replacement models")],
    output: Annotated[Path, typer.Option(help="New reconstructed replay")],
) -> None:
    """Adopt measured model corrections only after complete exact reproduction."""
    from xinyenyana.archive import sha256_file
    from xinyenyana.replay import apply_corrections

    _require_lease_and_archive()
    if output.exists() or output.resolve() in {p.resolve() for p in [reference, base, *correction]}:
        raise typer.BadParameter("reconstruction requires a new output path")
    patches = [json.loads(path.read_text()) for path in correction]
    for patch in patches:
        for key in ("candidate_vectors", "stored_vectors"):
            for entry in patch.get(key, {}).values():
                path = Path(entry["path"])
                if path.name != entry["sha256"] + ".npz" or sha256_file(path) != entry["sha256"]:
                    raise typer.BadParameter(f"invalid immutable vector object: {path}")
    result = apply_corrections(
        json.loads(reference.read_text()), json.loads(base.read_text()), patches
    )
    result["replay_correction"]["sources"] = {
        str(path): sha256_file(path) for path in [reference, base, *correction]
    }
    _emit(result, output, require_archive=True)


@app.command("build-great-tit-full-sample")
def build_great_tit_full_command(
    source: Annotated[Path, typer.Option(help="The unpacked OSF N8AC9 release")],
    destination: Annotated[Path, typer.Option(help="Where the clips and manifest go")],
    output: Annotated[Path, typer.Option(help="Build record")],
) -> None:
    """The Wytham great tits with songs in two or more years, across years."""

    from xinyenyana.great_tit_full import build_sample

    _require_lease_and_archive()
    _emit(build_sample(source=source, destination=destination), output, require_archive=True)


@app.command("run-v6-analyses")
def run_v6_analyses_command(
    endpoint: Annotated[str, typer.Option(help="Endpoint name")],
    sample_root: Annotated[Path, typer.Option(help="That endpoint's built sample")],
    sweep: Annotated[list[Path], typer.Option(help="Every a6 result for this endpoint")],
    output: Annotated[Path, typer.Option(help="Result file")],
    balance_enrolment: Annotated[
        int | None, typer.Option(help="The cap the sweep used, if any")
    ] = None,
) -> None:
    """Beecher's statistic and score normalisation from the stored vectors."""

    from xinyenyana.v6_analyses import analyse_endpoint

    _require_lease_and_archive()
    loaded = _endpoint(endpoint, sample_root, balance_enrolment)
    results = [json.loads(path.read_text()) for path in sweep]
    _emit(analyse_endpoint(loaded, results, sources=sweep), output, require_archive=True)


@app.command("run-v6-pairings")
def run_v6_pairings_command(
    endpoint: Annotated[str, typer.Option(help="Endpoint with published backgrounds")],
    sample_root: Annotated[Path, typer.Option(help="That endpoint's built sample")],
    sweep: Annotated[list[Path], typer.Option(help="Every a6 result for this endpoint")],
    output: Annotated[Path, typer.Option(help="Result file")],
    permutations: Annotated[
        int, typer.Option(min=1, help="Label shuffles for each pairing (PA-V6: 9,999)")
    ] = 9999,
) -> None:
    """The four pairings for every reported representation, from the stored vectors."""

    from xinyenyana.v6_analyses import pairings_endpoint

    _require_lease_and_archive()
    loaded = _endpoint(endpoint, sample_root, None)
    results = [json.loads(path.read_text()) for path in sweep]
    _emit(
        pairings_endpoint(loaded, results, sources=sweep, permutations=permutations),
        output,
        require_archive=True,
    )


@app.command("run-open-set-stored")
def run_open_set_stored_command(
    endpoint: Annotated[str, typer.Option(help="Endpoint name")],
    sample_root: Annotated[Path, typer.Option(help="That endpoint's built sample")],
    sweep: Annotated[list[Path], typer.Option(help="Every a6 result for this endpoint")],
    output: Annotated[Path, typer.Option(help="Result file")],
    model: Annotated[
        list[str] | None,
        typer.Option(help="A model to score; by default the three the protocol names"),
    ] = None,
    balance_enrolment: Annotated[
        int | None, typer.Option(help="The cap the sweep used, if any")
    ] = None,
) -> None:
    """The registered open-set protocol on stored vectors."""

    from xinyenyana.v6_analyses import open_set_from_store, open_set_models

    _require_lease_and_archive()
    loaded = _endpoint(endpoint, sample_root, balance_enrolment)
    results = [json.loads(path.read_text()) for path in sweep]
    chosen = None if model else open_set_models(results)
    models = list(model) if model else list(chosen.values()) if chosen else []
    summary = open_set_from_store(loaded, results, models=models, sources=sweep)
    if chosen is not None:
        summary["models_by_rule"] = chosen
    _emit(summary, output, require_archive=True)


@app.command("run-spectral-subtraction")
def run_spectral_subtraction_command(
    endpoint: Annotated[str, typer.Option(help="An endpoint with background recordings")],
    sample_root: Annotated[Path, typer.Option(help="That endpoint's built sample")],
    scratch: Annotated[Path, typer.Option(help="Where the subtracted waveforms go")],
    output: Annotated[Path, typer.Option(help="Result file")],
    store: Annotated[
        Path | None, typer.Option(help="Retain measured vectors and predictions")
    ] = None,
    previous_result: Annotated[
        Path | None, typer.Option(help="Preserved prior result for a transparent replacement")
    ] = None,
    permutations: Annotated[
        int, typer.Option(min=1, help="Label shuffles for each pairing (PA-V6: 9,999)")
    ] = 999,
) -> None:
    """BirdNET's four pairings on calls and backgrounds after spectral subtraction."""

    from xinyenyana.v6_analyses import spectral_subtraction_run

    _require_lease_and_archive()
    if output.exists() or (
        previous_result is not None and output.resolve() == previous_result.resolve()
    ):
        raise typer.BadParameter("spectral measurement requires a new output path")
    loaded = _load_endpoint(endpoint, sample_root)
    _emit(
        spectral_subtraction_run(
            loaded,
            scratch=scratch,
            store=store,
            previous_result=previous_result,
            permutations=permutations,
        ),
        output,
        require_archive=True,
    )


@app.command("run-sensitivity-floor")
def run_sensitivity_floor_command(
    donor_root: Annotated[Path, typer.Option(help="The Stowell release (donor endpoint)")],
    scratch: Annotated[Path, typer.Option(help="Where the planted waveforms go")],
    output: Annotated[Path, typer.Option(help="Result file")],
    donor: Annotated[str, typer.Option(help="Donor endpoint")] = "chiffchaff-withinyear",
    target: Annotated[str, typer.Option(help="Target endpoint")] = "chiffchaff-acrossyear",
) -> None:
    """Within-year chiffchaff calls planted into across-year backgrounds, then scored."""

    from xinyenyana.v6_analyses import sensitivity_floor_run

    _require_lease_and_archive()
    summary = sensitivity_floor_run(
        _load_endpoint(donor, donor_root), _load_endpoint(target, donor_root), scratch=scratch
    )
    _emit(summary, output, require_archive=True)


@app.command("run-encoder-comparison-v6")
def run_encoder_comparison_v6_command(
    sweep: Annotated[
        list[str], typer.Option(help="endpoint=path of an a5 or a6 result; repeat freely")
    ],
    output: Annotated[Path, typer.Option(help="Result file")],
) -> None:
    """Paired tests against BirdNET over every model, merging several sweeps per endpoint."""

    from xinyenyana.archive import sha256_file
    from xinyenyana.encoder_comparison import rank_consistency, summarise_sweep
    from xinyenyana.v6_analyses import merge_sweeps
    from xinyenyana.v6_controls import REPLICATES

    _require_lease_and_archive()
    grouped: dict[str, list[tuple[Path, dict[str, Any]]]] = {}
    for pair in sweep:
        name, _, raw = pair.partition("=")
        path = Path(raw)
        grouped.setdefault(name, []).append((path, json.loads(path.read_text())))
    per_endpoint: dict[str, Any] = {}
    accuracy: dict[str, dict[str, float]] = {}
    accuracy_held_out: dict[str, dict[str, float]] = {}
    for name, loaded in sorted(grouped.items()):
        merged = merge_sweeps([result for _, result in loaded])
        held_out = {**merged, "chosen": merged.get("chosen_held_out", {})}
        per_endpoint[name] = {
            **summarise_sweep(merged, replicates=REPLICATES),
            "held_out_rule": summarise_sweep(held_out, replicates=REPLICATES),
            "sources": {str(path): sha256_file(path) for path, _ in loaded},
            "merge": merged["merge"],
        }
        accuracy[name] = {
            model.split("/")[-1]: float(entry["accuracy"])
            for model, entry in merged["chosen"].items()
        }
        accuracy_held_out[name] = {
            model.split("/")[-1]: float(entry["accuracy"])
            for model, entry in held_out["chosen"].items()
        }
    _emit(
        {
            "per_endpoint": per_endpoint,
            "rank_consistency": rank_consistency(accuracy),
            "rank_consistency_held_out_rule": rank_consistency(accuracy_held_out),
        },
        output,
        require_archive=True,
    )


@app.command("merge-sweeps")
def merge_sweeps_command(
    sweep: Annotated[list[Path], typer.Option(help="An a5 or a6 result for the endpoint")],
    output: Annotated[Path, typer.Option(help="The merged result")],
) -> None:
    """Several results for one endpoint as one, the reported candidate chosen again."""

    from xinyenyana.archive import sha256_file
    from xinyenyana.v6_analyses import merge_sweeps

    _require_lease_and_archive()
    merged = merge_sweeps([json.loads(path.read_text()) for path in sweep])
    merged["sources"] = {str(path): sha256_file(path) for path in sweep}
    _emit(merged, output, require_archive=True)


@app.command("run-place-control-v6")
def run_place_control_v6_command(
    endpoint: Annotated[str, typer.Option(help="great-tit or great-tit-full")],
    sample_root: Annotated[Path, typer.Option(help="That endpoint's built sample")],
    output: Annotated[Path, typer.Option(help="Result file")],
) -> None:
    """Whether any great tit nestbox or recording appears on both sides of the split."""

    from xinyenyana.identity_run import place_control

    _require_lease_and_archive()
    loaded = _endpoint(endpoint, sample_root, None)
    _emit(
        {
            "endpoint": loaded.name,
            "manifest_sha256": loaded.manifest_sha256,
            **place_control(loaded.records),
        },
        output,
        require_archive=True,
    )


@app.command("check-v5-reproduction")
def check_v5_reproduction_command(
    pair: Annotated[list[str], typer.Option(help="v5-result=v6-result, once per endpoint")],
    output: Annotated[Path, typer.Option(help="Result file")],
) -> None:
    """Whether every candidate v5 computed gives the same accuracy and answers in v6."""

    from xinyenyana.a6 import compare_sweep_candidates
    from xinyenyana.archive import sha256_file

    _require_lease_and_archive()
    rows: dict[str, Any] = {}
    for item in pair:
        old_path, _, new_path = item.partition("=")
        old = json.loads(Path(old_path).read_text())
        new = json.loads(Path(new_path).read_text())
        rows[old["endpoint"]] = {
            "v5": {"path": old_path, "sha256": sha256_file(Path(old_path))},
            "v6": {"path": new_path, "sha256": sha256_file(Path(new_path))},
            **compare_sweep_candidates(old, new),
        }
    _emit({"per_endpoint": rows}, output, require_archive=True)
    if not all(row["passed"] for row in rows.values()):
        raise typer.Exit(1)
