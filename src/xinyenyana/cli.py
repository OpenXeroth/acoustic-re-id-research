"""The commands that run Phase A's measurements.

Every measurement this project reports is produced by one of these. They exist
so a run is a command in the repository rather than a script improvised on the
machine, and so the same command produces the same number tomorrow.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Annotated, Any

import typer

from xinyenyana.a2 import (
    STOWELL_ENDPOINTS,
    Endpoint,
    load_great_tit,
    load_rookid,
    load_stowell,
    load_zebra_finch,
    run_a2_endpoint,
)
from xinyenyana.a3 import adopt, run_a3_endpoint
from xinyenyana.a5 import representation_from_name, run_a5_endpoint, vectors_for
from xinyenyana.archive import ARCHIVE_VARIABLE, archive_result, sha256_file, source_digest
from xinyenyana.bats import BAT_ENDPOINTS, BAT_MANIFEST, build_bat_sample, load_bats
from xinyenyana.birdpark import BIRDPARK_ENDPOINTS, BIRDPARK_MANIFEST, load_birdpark
from xinyenyana.compensation import run_compensation
from xinyenyana.constraints import (
    attach_song_times,
    audit_constraints,
    constraint_sensitivity,
    endpoint_events,
)
from xinyenyana.endpoints import (
    benchmark_root,
    check_recorded_files,
    check_rookid_members,
    extract_rookid_annotations,
)
from xinyenyana.evaluation import HEADS, KERNEL_RIDGE
from xinyenyana.huang import (
    HUANG_ENDPOINTS,
    PUBLISHED_EMBEDDINGS,
    PUBLISHED_MODELS,
    PUBLISHED_SPLITS,
    load_huang,
    run_published_embeddings,
)
from xinyenyana.local_clips import build_e01d_sample
from xinyenyana.open_set import BIRDNET as OPEN_SET_BIRDNET
from xinyenyana.open_set import LOG_MEL as OPEN_SET_LOG_MEL
from xinyenyana.open_set import REPRESENTATIONS, run_open_set_endpoint
from xinyenyana.right_whale import ENDPOINT as RIGHT_WHALE_ENDPOINT
from xinyenyana.right_whale import RIGHT_WHALE_MANIFEST, load_right_whale
from xinyenyana.rookid_channels import audit_recording_channels, choose_channel
from xinyenyana.validity import run_validity_gate

GENERAL_OPEN_SET_REPRESENTATIONS = (OPEN_SET_BIRDNET, OPEN_SET_LOG_MEL)

app = typer.Typer(no_args_is_help=True, pretty_exceptions_show_locals=False)

if __name__ == "__main__":  # pragma: no cover
    # `python -m xinyenyana.cli` runs this file as `__main__`. The PA-V6 commands
    # in `cli_v6` register on `xinyenyana.cli.app`, so that name has to be this
    # module and not a second copy of it.
    sys.modules.setdefault("xinyenyana.cli", sys.modules[__name__])

ROOK_FULL_WIDTH = "rookid-full-width"
E01D_PROTOCOL = Path("configs/experiments/e01d-rookid-context-matched.json")
ROOKID_SAMPLE = "rookid-zenodo/6091940/context-matched-mono"


def _code_revision() -> str | None:
    """The commit the measurement ran at, or None outside a repository."""

    result = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=Path(__file__).resolve().parents[2],
        check=False,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip() if result.returncode == 0 else None


def _source_root() -> Path:
    return Path(__file__).resolve().parent


#: Provenance, taken when the command starts rather than when it writes.
#: A sweep can run for hours; taking these at write time recorded whatever the
#: working tree and HEAD happened to be when it finished, which on a machine
#: where the next change is already being prepared is a different revision from
#: the one that produced the numbers.
_PROVENANCE: dict[str, Any] = {}


@app.callback()
def _snapshot_provenance() -> None:
    """Record the revision and source digest before any command does work."""

    _PROVENANCE["code_revision"] = _code_revision()
    _PROVENANCE["source_sha256"] = source_digest(_source_root())


def _emit(payload: Any, output: Path | None, *, require_archive: bool = False) -> None:
    """Write the result, and put a copy where one disk failure cannot lose it.

    A build summary is archived on the same terms as a measurement, because it
    records what a sample is made of and every later figure is read against it.
    The three sample builders did not require it, so on 2026-09-15 the four
    BirdPark summaries existed on one disk only and the right whale's eleven
    had been written to a path that no longer holds them. The warning those
    runs printed went to standard error and nobody read it.

    ``code_revision`` names a commit that stops existing when the branch is
    squash-merged, so ``source_sha256`` is recorded beside it: a digest of the
    measurement code, which survives any merge strategy. Both are taken when the
    command starts, not here.

    Archiving happens only when ``XINYENYANA_RESULT_ARCHIVE`` names a
    destination, so tests and local runs never reach the network, and it is set
    on the machine where results are actually produced so that it cannot be
    forgotten there.
    """

    if require_archive and (output is None or not os.environ.get(ARCHIVE_VARIABLE)):
        raise typer.BadParameter("this measurement requires an output and a result archive")
    if isinstance(payload, dict):
        payload = {
            **payload,
            "code_revision": _PROVENANCE.get("code_revision", _code_revision()),
            "source_sha256": _PROVENANCE.get("source_sha256", source_digest(_source_root())),
        }
    text = json.dumps(payload, indent=2, sort_keys=True) + "\n"
    if output is not None:
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(text)
        prefix = os.environ.get(ARCHIVE_VARIABLE)
        if prefix:
            receipt = archive_result(output, prefix=prefix)
            typer.echo(json.dumps(receipt, sort_keys=True))
            if require_archive and not receipt["archived"]:
                raise typer.Exit(1)
        else:
            typer.echo(
                json.dumps({"archived": False, "reason": f"{ARCHIVE_VARIABLE} is unset"}),
                err=True,
            )
    if output is None:
        typer.echo(text)


def _load_endpoint(name: str, sample_root: Path) -> Endpoint:
    if name == "rookid":
        return load_rookid(
            manifest_path=sample_root / "xinyenyana-manifest.json", sample_root=sample_root
        )
    if name == "zebra-finch":
        return load_zebra_finch(
            manifest_path=sample_root / "identity-manifest.json", sample_root=sample_root
        )
    if name == "great-tit":
        return load_great_tit(
            manifest_path=sample_root / "private-enriched-manifest.json", sample_root=sample_root
        )
    if name == "great-tit-full":
        from xinyenyana.great_tit_full import load_great_tit_full

        return load_great_tit_full(sample_root)
    if name in STOWELL_ENDPOINTS:
        # The release unpacks to `csv/` beside `wav/`, so one root reaches both.
        return load_stowell(
            endpoint=name, csv_root=sample_root / "csv", audio_root=sample_root / "wav"
        )
    if name in HUANG_ENDPOINTS:
        # The release unpacks to one `dataset` folder holding every species.
        return load_huang(endpoint=name, root=sample_root)
    if name in BIRDPARK_ENDPOINTS:
        return load_birdpark(manifest_path=sample_root / BIRDPARK_MANIFEST, sample_root=sample_root)
    if name == RIGHT_WHALE_ENDPOINT:
        # One arm is one shift, so the sample root names the arm.
        return load_right_whale(
            manifest_path=sample_root / RIGHT_WHALE_MANIFEST, sample_root=sample_root.parent
        )
    if name == ROOK_FULL_WIDTH:
        # The across-year split on channel 0, which is how the open-set run
        # reads it. `run-identity` loads the other splits and channels itself.
        from xinyenyana.e02 import load_e02_rookid

        loaded, _ = load_e02_rookid(
            manifest_path=sample_root / "e02-manifest.json",
            sample_root=sample_root,
            split="across-year",
            channel=0,
        )
        rook: Endpoint = loaded
        return rook
    if name in BAT_ENDPOINTS:
        # A bat clip is a span inside a recording, so the endpoint is read from
        # the sample `build-bat-sample` cut rather than from whole files.
        return load_bats(manifest_path=sample_root / BAT_MANIFEST, sample_root=sample_root)
    known = [
        "rookid",
        "zebra-finch",
        "great-tit",
        "great-tit-full",
        *sorted(STOWELL_ENDPOINTS),
        *sorted(HUANG_ENDPOINTS),
        *sorted(BAT_ENDPOINTS),
        RIGHT_WHALE_ENDPOINT,
        *sorted(BIRDPARK_ENDPOINTS),
        ROOK_FULL_WIDTH,
    ]
    raise typer.BadParameter(f"unknown endpoint {name!r}: {', '.join(known)}")


@app.command("verify-data")
def verify_data(
    root: Annotated[Path | None, typer.Option(help="Local archive root")] = None,
    protocol: Annotated[Path, typer.Option(help="E01d protocol")] = E01D_PROTOCOL,
) -> None:
    """Check the local archives against the digests the recorded runs used."""

    files = check_recorded_files(root=root)
    members = check_rookid_members(protocol=json.loads(protocol.read_bytes()), root=root)
    payload: dict[str, Any] = {
        "root": str(root if root is not None else benchmark_root()),
        "files": [
            {"path": c.relative_path, "ok": c.ok, "recorded_in": c.recorded_in} for c in files
        ],
        "rookid_members": [{"member": c.member, "ok": c.ok} for c in members],
    }
    payload["all_ok"] = all(c.ok for c in files) and all(c.ok for c in members)
    _emit(payload, None)
    if not payload["all_ok"]:
        raise typer.Exit(code=1)


@app.command("audit-rookid-channels")
def audit_rookid_channels(
    sample_root: Annotated[Path, typer.Option(help="A built RookID sample")],
    root: Annotated[Path | None, typer.Option(help="Local archive root")] = None,
) -> None:
    """Measure every channel of each RookID recording and apply the channel rule.

    The rule is in ``docs/measurement-protocol.md``: the channel with the
    smallest share of energy below 50 Hz over the enrollment recording's
    annotated windows, used for both recordings.
    """

    files = json.loads((sample_root / "xinyenyana-manifest.json").read_text())["sample"]["files"]
    enrollment = sorted({str(f["recording"]) for f in files if f["split"] == "enrollment"})
    payload: dict[str, Any] = {"recordings": {}}
    for stem in sorted({str(f["recording"]) for f in files}):
        windows = [
            (float(f["annotation_start_seconds"]), float(f["annotation_end_seconds"]))
            for f in files
            if str(f["recording"]) == stem
        ]
        statistics = audit_recording_channels(stem=stem, windows=windows, root=root)
        payload["recordings"][stem] = [s.as_dict() for s in statistics]
        if stem in enrollment:
            payload["chosen_channel"] = choose_channel(statistics)
            payload["chosen_from"] = stem
    _emit(payload, None)


@app.command("build-rookid-sample")
def build_rookid_sample(
    output_root: Annotated[Path, typer.Option(help="Where the sample is written")],
    channel_index: Annotated[
        int | None, typer.Option(help="Channel to cut from; the protocol's if unset")
    ] = None,
    root: Annotated[Path | None, typer.Option(help="Local archive root")] = None,
    protocol: Annotated[Path, typer.Option(help="E01d protocol")] = E01D_PROTOCOL,
) -> None:
    """Cut the RookID clips from the local archive on the given channel."""

    sample_root = output_root / ROOKID_SAMPLE
    extract_rookid_annotations(
        protocol=json.loads(protocol.read_bytes()), destination=sample_root, root=root
    )
    manifest = build_e01d_sample(
        protocol_path=protocol, output_root=output_root, root=root, channel_index=channel_index
    )
    _emit(
        {
            "sample_root": str(sample_root),
            "channel": manifest["channel"],
            "selection_sha256": manifest["selection"]["selection_sha256"],
            "files": manifest["sample"]["file_count"],
            "bytes": manifest["sample"]["total_bytes"],
        },
        None,
    )


@app.command("build-bat-sample")
def build_bat_sample_command(
    endpoint: Annotated[str, typer.Option(help="A bat endpoint")],
    release_root: Annotated[Path, typer.Option(help="The unpacked Prat release")],
    sample_root: Annotated[Path, typer.Option(help="Where the cut clips are written")],
    output: Annotated[Path, typer.Option(help="Where the build summary is written")],
) -> None:
    """Cut each annotated bat vocalisation out of the recording that holds it."""

    if endpoint not in BAT_ENDPOINTS:
        raise typer.BadParameter(f"unknown bat endpoint {endpoint!r}: {sorted(BAT_ENDPOINTS)}")
    _emit(
        build_bat_sample(
            endpoint=endpoint,
            annotations_path=release_root / "Annotations.csv",
            file_info_path=release_root / "FileInfo.csv",
            audio_root=release_root / "unpacked",
            sample_root=sample_root,
        ),
        output,
        require_archive=True,
    )


@app.command("build-right-whale-sample")
def build_right_whale_sample_command(
    release_root: Annotated[Path, typer.Option(help="The narw-acoustic-identification checkout")],
    sample_root: Annotated[Path, typer.Option(help="Where the transformed clips are written")],
    output: Annotated[Path, typer.Option(help="Where the build summary is written")],
    shift_hz: Annotated[int, typer.Option(help="One of the registered shifts; 0 is unshifted")],
) -> None:
    """Write one arm of the right whale endpoint at one registered shift.

    The release declares no licence. Neither its audio nor anything derived from
    it leaves xen1; results do.
    """

    from xinyenyana.right_whale import SHIFTS, build_right_whale_sample

    if shift_hz not in SHIFTS:
        raise typer.BadParameter(f"unregistered shift {shift_hz}: {list(SHIFTS)}")
    _emit(
        build_right_whale_sample(
            release_root=release_root, output_root=sample_root, shift_hz=shift_hz
        ),
        output,
        require_archive=True,
    )


@app.command("build-birdpark-sample")
def build_birdpark_sample_command(
    endpoint: Annotated[str, typer.Option(help="A BirdPark endpoint")],
    release_root: Annotated[Path, typer.Option(help="The fetched Zenodo 13144875 directory")],
    sample_root: Annotated[Path, typer.Option(help="Where the cut clips are written")],
    scratch_root: Annotated[Path, typer.Option(help="Working space for one archive member")],
    output: Annotated[Path, typer.Option(help="Where the build summary is written")],
) -> None:
    """Cut every kept segment out of the microphone every bird in the box shares."""

    from xinyenyana.birdpark import BIRDPARK_ENDPOINTS, build_birdpark_sample

    if endpoint not in BIRDPARK_ENDPOINTS:
        raise typer.BadParameter(
            f"unknown BirdPark endpoint {endpoint!r}: {sorted(BIRDPARK_ENDPOINTS)}"
        )
    _emit(
        build_birdpark_sample(
            endpoint=endpoint,
            segments_path=release_root / "segments.h5",
            data_archive=release_root / "Data.zip",
            sample_root=sample_root,
            scratch_root=scratch_root,
        ),
        output,
        require_archive=True,
    )


@app.command("run-a2")
def run_a2(
    endpoint: Annotated[str, typer.Option(help="rookid, zebra-finch or great-tit")],
    sample_root: Annotated[Path, typer.Option(help="That endpoint's built sample")],
    output: Annotated[Path, typer.Option(help="Where the measurement is written")],
) -> None:
    """Measure which of the three call layers separates individuals."""

    summary = run_a2_endpoint(endpoint=_load_endpoint(endpoint, sample_root))
    _emit(summary, output)


@app.command("run-a3")
def run_a3(
    endpoint: Annotated[str, typer.Option(help="rookid, zebra-finch or great-tit")],
    sample_root: Annotated[Path, typer.Option(help="That endpoint's built sample")],
    output: Annotated[Path, typer.Option(help="Where the measurement is written")],
) -> None:
    """Measure what note trimming, band-limiting and noise subtraction cost."""

    summary = run_a3_endpoint(endpoint=_load_endpoint(endpoint, sample_root))
    summary["adoption"] = {
        layer: adopt(summary, layer=layer)
        for layer in ("source", "filter", "filter_normalised", "motor")
    }
    _emit(summary, output)


@app.command("run-open-set")
def run_open_set(
    endpoint: Annotated[str, typer.Option(help="great-tit, or any endpoint with 8+ animals")],
    sample_root: Annotated[Path, typer.Option(help="That endpoint's built sample")],
    output: Annotated[Path, typer.Option(help="Where the measurement is written")],
    scratch: Annotated[Path, typer.Option(help="Working directory for the masked clips")],
    representation: Annotated[
        list[str] | None, typer.Option(help="Limit the run to these representations")
    ] = None,
) -> None:
    """Measure whether a call can be told to belong to any enrolled bird at all."""

    # Only the great tit carries the year, nestbox, level measurements and song
    # annotations four of the six representations are built from.
    default = REPRESENTATIONS if endpoint == "great-tit" else GENERAL_OPEN_SET_REPRESENTATIONS
    names = tuple(representation) if representation else default
    unknown = [name for name in names if name not in REPRESENTATIONS]
    if unknown:
        raise typer.BadParameter(f"unknown representations {unknown}: {list(REPRESENTATIONS)}")
    summary = run_open_set_endpoint(
        endpoint=_load_endpoint(endpoint, sample_root),
        scratch=scratch,
        representations=names,
    )
    _emit(summary, output, require_archive=endpoint != "great-tit")


@app.command("audit-physical-constraints")
def audit_physical_constraints(
    endpoint: Annotated[str, typer.Option(help="great-tit")],
    sample_root: Annotated[Path, typer.Option(help="That endpoint built sample")],
    output: Annotated[Path, typer.Option(help="Where the measurement is written")],
    song_metadata: Annotated[
        Path, typer.Option(help="Published great-tit-hits.csv with song start times")
    ],
) -> None:
    """Count which pairings of clips the physics rules out, without using audio."""

    from xinyenyana.frozen_probe import runtime_record

    if not os.environ.get("XENWARDEN_LEASE") or not os.environ.get(ARCHIVE_VARIABLE):
        raise typer.BadParameter(
            "this audit requires an admitted lease and configured result archive"
        )
    loaded = attach_song_times(_load_endpoint(endpoint, sample_root), song_metadata)
    events = endpoint_events(loaded)
    _emit(
        {
            "experiment": "PA-CONSTRAINT-TIME-CORRECTION",
            "endpoint": endpoint,
            "manifest_sha256": loaded.manifest_sha256,
            "song_metadata_sha256": sha256_file(song_metadata),
            "splits": loaded.split_digests(),
            "environment": runtime_record(),
            "timing": "published song datetime plus song-relative first note onset",
            "assumption": (
                "publisher clock convention; nestbox positions proxy sources; "
                "independent clock synchronisation not established"
            ),
            "claim_boundary": (
                "conditional geometric audit, not proof of simultaneous birds "
                "or a track-identification result"
            ),
            "timed_events": [
                {
                    "filename": record.filename,
                    "song_datetime": record.context["song_datetime"],
                    "first_note_seconds": event.seconds,
                }
                for record, event in zip(loaded.records, events, strict=True)
            ],
            **audit_constraints(events),
            "uncertainty_sensitivity": constraint_sensitivity(events),
        },
        output,
        require_archive=True,
    )


@app.command("run-validity-gate")
def run_validity_gate_command(
    endpoint: Annotated[str, typer.Option(help="An endpoint that carries background clips")],
    sample_root: Annotated[Path, typer.Option(help="That endpoint's unpacked release")],
    output: Annotated[Path, typer.Option(help="Where the measurement is written")],
    representation: Annotated[
        str,
        typer.Option(help="A representation by the name a sweep recorded it under"),
    ] = "birdnet-v2.4",
    device: Annotated[str, typer.Option(help="Where an encoder runs")] = "cuda",
    permutations: Annotated[
        int, typer.Option(min=0, help="Label shuffles for each null (v5: 999; PA-V6: 9,999)")
    ] = 999,
    replicates: Annotated[
        int, typer.Option(min=1, help="Animal bootstrap draws (v5: 2,000; PA-V6: 10,000)")
    ] = 2000,
) -> None:
    """Score one representation on the backgrounds alone, to see how much is the place."""

    loaded = _load_endpoint(endpoint, sample_root)
    try:
        chosen = representation_from_name(representation)
    except ValueError as error:
        raise typer.BadParameter(str(error)) from error
    vectors = vectors_for(chosen, loaded.records, device=device)
    _emit(
        run_validity_gate(
            endpoint=loaded,
            vectors=vectors,
            representation=representation,
            bootstrap_replicates=replicates,
            permutations=permutations,
        ),
        output,
    )


@app.command("run-published-embeddings")
def run_published_embeddings_command(
    species: Annotated[str, typer.Option(help="A species in the Huang release")],
    model: Annotated[str, typer.Option(help="birdnet or google-perch")],
    split: Annotated[str, typer.Option(help="fold1 to fold5, or published")],
    release_root: Annotated[Path, typer.Option(help="The unpacked Huang release")],
    output: Annotated[Path, typer.Option(help="Where the measurement is written")],
    head: Annotated[str, typer.Option(help="Which classifier")] = KERNEL_RIDGE,
) -> None:
    """Run this project's head on the embeddings the release's authors published."""

    if species not in PUBLISHED_EMBEDDINGS:
        raise typer.BadParameter(f"unknown species {species!r}: {sorted(PUBLISHED_EMBEDDINGS)}")
    if model not in PUBLISHED_MODELS:
        raise typer.BadParameter(f"unknown model {model!r}: {sorted(PUBLISHED_MODELS)}")
    if split not in PUBLISHED_SPLITS:
        raise typer.BadParameter(f"unknown split {split!r}: {list(PUBLISHED_SPLITS)}")
    if head not in HEADS:
        raise typer.BadParameter(f"unknown head {head!r}: {list(HEADS)}")
    _emit(
        run_published_embeddings(
            species=species, model=model, split=split, root=release_root, head=head
        ),
        output,
    )


@app.command("run-compensation")
def run_compensation_command(
    endpoint: Annotated[str, typer.Option(help="Any endpoint this project can load")],
    sample_root: Annotated[Path, typer.Option(help="That endpoint's built sample")],
    output: Annotated[Path, typer.Option(help="Where the measurement is written")],
    representation: Annotated[
        str,
        typer.Option(help="A representation by the name a sweep recorded it under"),
    ] = "birdnet-v2.4",
    enrollment_context_key: Annotated[
        str | None, typer.Option(help="Restrict enrolment by this context field")
    ] = None,
    enrollment_context_values: Annotated[
        str | None, typer.Option(help="Comma-separated values to keep")
    ] = None,
    enrollment_cap: Annotated[
        int | None, typer.Option(help="Subsample enrolment to this many clips")
    ] = None,
    device: Annotated[str, typer.Option(help="Where an encoder runs")] = "cuda",
    permutations: Annotated[
        int, typer.Option(min=0, help="Label shuffles for each null (v5: 999; PA-V6: 9,999)")
    ] = 999,
    replicates: Annotated[
        int, typer.Option(min=1, help="Animal bootstrap draws (v5: 2,000; PA-V6: 10,000)")
    ] = 2000,
) -> None:
    """E-REID-01: every session compensation, on one endpoint and representation."""

    import numpy as np

    from xinyenyana.evaluation import standardisation_for

    loaded = _load_endpoint(endpoint, sample_root)
    try:
        chosen = representation_from_name(representation)
    except ValueError as error:
        raise typer.BadParameter(str(error)) from error
    vectors = vectors_for(chosen, loaded.records, device=device)
    restriction = None
    if enrollment_context_key is not None:
        if enrollment_context_values is None:
            raise typer.BadParameter("--enrollment-context-values is required with a key")
        restriction = {
            "key": enrollment_context_key,
            "values": [value.strip() for value in enrollment_context_values.split(",")],
            "cap": enrollment_cap,
        }
    _emit(
        run_compensation(
            endpoint=loaded,
            vectors=vectors,
            representation=representation,
            standardisation=standardisation_for(int(np.asarray(vectors).shape[1])),
            enrollment_filter=restriction,
            permutations=permutations,
            bootstrap_replicates=replicates,
        ),
        output,
    )


@app.command("run-learned-combination")
def run_learned_combination_command(
    endpoint: Annotated[str, typer.Option(help="A published benchmark endpoint")],
    sample_root: Annotated[Path, typer.Option(help="That endpoint's existing sample or release")],
    cache_root: Annotated[Path, typer.Option(help="Resumable feature cache on xen1")],
    output: Annotated[Path, typer.Option(help="Result file, including every prediction")],
    threads: Annotated[int, typer.Option(min=1, max=8)] = 1,
    permutations: Annotated[
        int, typer.Option(min=0, help="Label shuffles for each null (v5: 999; PA-V6: 9,999)")
    ] = 999,
    replicates: Annotated[
        int, typer.Option(min=1, help="Animal bootstrap draws (v5: 2,000; PA-V6: 10,000)")
    ] = 2000,
) -> None:
    """Architecture one: every BirdNET layer combined, against the layers alone."""

    from xinyenyana.frozen_probe import (
        endpoint_audio_provenance,
        extract_endpoint,
        runtime_record,
    )
    from xinyenyana.learned_combination import run_learned_combination

    if not os.environ.get("XENWARDEN_LEASE"):
        raise typer.BadParameter("run this measurement through an admitted XenWarden lease")
    if not os.environ.get(ARCHIVE_VARIABLE):
        raise typer.BadParameter(f"{ARCHIVE_VARIABLE} must be configured before extraction")
    loaded = _load_endpoint(endpoint, sample_root)
    data = endpoint_audio_provenance(loaded)
    vectors, extraction = extract_endpoint(loaded, cache_root=cache_root, threads=threads)
    _emit(
        {
            **run_learned_combination(
                loaded, vectors, permutations=permutations, bootstrap_replicates=replicates
            ),
            "extraction": extraction,
            "environment": runtime_record(),
            "data": data,
        },
        output,
        require_archive=True,
    )


@app.command("run-sequence-metric")
def run_sequence_metric_command(
    endpoint: Annotated[str, typer.Option(help="A published benchmark endpoint")],
    sample_root: Annotated[Path, typer.Option(help="That endpoint's existing sample or release")],
    cache_root: Annotated[Path, typer.Option(help="Resumable feature cache on xen1")],
    output: Annotated[Path, typer.Option(help="Result file")],
    session_key: Annotated[
        str | None, typer.Option(help="Context field naming the recording session")
    ] = None,
    threads: Annotated[int, typer.Option(min=1, max=8)] = 1,
    permutations: Annotated[
        int, typer.Option(min=0, help="Label shuffles for each null (v5: 999; PA-V6: 9,999)")
    ] = 999,
    replicates: Annotated[
        int, typer.Option(min=1, help="Animal bootstrap draws (v5: 2,000; PA-V6: 10,000)")
    ] = 2000,
) -> None:
    """Architecture two: a pooling learned over the time inside a call."""

    import sys

    import numpy as np

    from xinyenyana.birdnet_reader import BirdNETReader
    from xinyenyana.extraction_cache import checkpointed_interrupts
    from xinyenyana.frozen_probe import endpoint_audio_provenance, runtime_record
    from xinyenyana.sequence_metric import SEQUENCE_LAYERS, run_sequence_metric

    if not os.environ.get("XENWARDEN_LEASE"):
        raise typer.BadParameter("run this measurement through an admitted XenWarden lease")
    if not os.environ.get(ARCHIVE_VARIABLE):
        raise typer.BadParameter(f"{ARCHIVE_VARIABLE} must be configured before extraction")
    loaded = _load_endpoint(endpoint, sample_root)
    sessions: list[str] | None = None
    if session_key is not None:
        missing = [r.filename for r in loaded.records if session_key not in r.context]
        if missing:
            raise typer.BadParameter(
                f"{len(missing)} clips carry no {session_key!r}, first {missing[0]}"
            )
        sessions = [str(record.context[session_key]) for record in loaded.records]
    data = endpoint_audio_provenance(loaded)
    reader = BirdNETReader(cache_root, threads=threads)
    sequences: dict[str, list[Any]] = {key: [] for key in SEQUENCE_LAYERS}
    means: dict[str, list[Any]] = {}
    with checkpointed_interrupts():
        for index, record in enumerate(loaded.records):
            extracted = reader.read(record.path)
            for key in SEQUENCE_LAYERS:
                sequences[key].append(extracted[key])
            for key, value in extracted.items():
                if key.endswith(".mean"):
                    means.setdefault(key, []).append(value)
            if index % 50 == 0 or index + 1 == len(loaded.records):
                print(f"read {index + 1}/{len(loaded.records)} clips", file=sys.stderr, flush=True)
    _emit(
        {
            **run_sequence_metric(
                loaded,
                sequences,
                {key: np.vstack(rows) for key, rows in means.items()},
                sessions=sessions,
                permutations=permutations,
                bootstrap_replicates=replicates,
            ),
            "extraction": reader.specification,
            "environment": runtime_record(),
            "data": data,
        },
        output,
        require_archive=True,
    )


@app.command("run-split-difference")
def run_split_difference_command(
    species: Annotated[str, typer.Option(help="A species in the Huang release")],
    model: Annotated[str, typer.Option(help="birdnet or google-perch")],
    release_root: Annotated[Path, typer.Option(help="The unpacked Huang release")],
    output: Annotated[Path, typer.Option(help="Where the measurement is written")],
    random_split: Annotated[str, typer.Option(help="The release's own fold")] = "fold1",
    replicates: Annotated[
        int, typer.Option(min=1, help="Animal bootstrap draws (v5: 2,000; PA-V6: 10,000)")
    ] = 2000,
) -> None:
    """Put an interval on the fall from a random division to the recordists' own.

    The authors' own vectors under their own two splits, so the division is the
    only thing that differs. The same draw of animals is applied to both splits
    and the difference taken inside the draw, because the unit of evidence is an
    animal and not a clip.
    """

    from xinyenyana.evaluation import PER_CLIP_L2, evaluate_endpoint
    from xinyenyana.huang import RIDGE_LAMBDA, SEED, load_published_vectors
    from xinyenyana.split_difference import paired_split_interval

    if species not in PUBLISHED_EMBEDDINGS:
        raise typer.BadParameter(f"unknown species {species!r}: {sorted(PUBLISHED_EMBEDDINGS)}")
    if model not in PUBLISHED_MODELS:
        raise typer.BadParameter(f"unknown model {model!r}: {sorted(PUBLISHED_MODELS)}")
    for name in (random_split, "published"):
        if name not in PUBLISHED_SPLITS:
            raise typer.BadParameter(f"unknown split {name!r}: {list(PUBLISHED_SPLITS)}")

    scored: dict[str, dict[str, Any]] = {}
    for label, split in (("random_fold", random_split), ("recordists", "published")):
        loaded, vectors = load_published_vectors(
            species=species, model=model, split=split, root=release_root
        )
        scored[label] = {
            "split": split,
            "identities": len(loaded.identities),
            "clips": len(loaded.records),
            "manifest_sha256": loaded.manifest_sha256,
            "splits": loaded.split_digests(),
            "heads": {
                head: evaluate_endpoint(
                    records=[record.as_evaluation_record() for record in loaded.records],
                    vectors=vectors,
                    representation=f"published-{model}",
                    enrollment_condition="foreground",
                    query_condition="foreground",
                    ridge_lambda=RIDGE_LAMBDA,
                    seed=SEED,
                    permutations=0,
                    bootstrap_replicates=0,
                    standardisation=PER_CLIP_L2,
                    head=head,
                )
                for head in HEADS
            },
        }
    _emit(
        {
            "experiment": "PA-SPLIT-DIFFERENCE",
            "species": species,
            "model": model,
            "random_split": random_split,
            "clips": {label: entry["clips"] for label, entry in scored.items()},
            "manifest_sha256": {label: entry["manifest_sha256"] for label, entry in scored.items()},
            "splits": {label: entry["splits"] for label, entry in scored.items()},
            "heads": {
                head: paired_split_interval(
                    random_fold=scored["random_fold"]["heads"][head]["predictions"],
                    recordists=scored["recordists"]["heads"][head]["predictions"],
                    replicates=replicates,
                )
                for head in HEADS
            },
        },
        output,
        require_archive=True,
    )


@app.command("run-published-compensation")
def run_published_compensation_command(
    species: Annotated[str, typer.Option(help="A species in the Huang release")],
    model: Annotated[str, typer.Option(help="birdnet or google-perch")],
    split: Annotated[str, typer.Option(help="fold1 to fold5, or published")],
    release_root: Annotated[Path, typer.Option(help="The unpacked Huang release")],
    output: Annotated[Path, typer.Option(help="Where the measurement is written")],
    permutations: Annotated[
        int, typer.Option(min=0, help="Label shuffles for each null (v5: 999; PA-V6: 9,999)")
    ] = 999,
    replicates: Annotated[
        int, typer.Option(min=1, help="Animal bootstrap draws (v5: 2,000; PA-V6: 10,000)")
    ] = 2000,
) -> None:
    """E-REID-01 on the authors' own vectors, which recompute nothing."""

    import numpy as np

    from xinyenyana.evaluation import standardisation_for
    from xinyenyana.huang import load_published_vectors

    if species not in PUBLISHED_EMBEDDINGS:
        raise typer.BadParameter(f"unknown species {species!r}: {sorted(PUBLISHED_EMBEDDINGS)}")
    if model not in PUBLISHED_MODELS:
        raise typer.BadParameter(f"unknown model {model!r}: {sorted(PUBLISHED_MODELS)}")
    if split not in PUBLISHED_SPLITS:
        raise typer.BadParameter(f"unknown split {split!r}: {list(PUBLISHED_SPLITS)}")
    loaded, vectors = load_published_vectors(
        species=species, model=model, split=split, root=release_root
    )
    summary = run_compensation(
        endpoint=loaded,
        vectors=vectors,
        representation=f"published-{model}",
        standardisation=standardisation_for(int(np.asarray(vectors).shape[1])),
        permutations=permutations,
        bootstrap_replicates=replicates,
    )
    _emit({**summary, "species": species, "model": model, "split": split}, output)


@app.command("run-background-challenge")
def run_background_challenge_command(
    endpoint: Annotated[str, typer.Option(help="A published endpoint carrying backgrounds")],
    sample_root: Annotated[Path, typer.Option(help="That endpoint's existing release")],
    cache_root: Annotated[Path, typer.Option(help="Resumable feature cache on xen1")],
    output: Annotated[Path, typer.Option(help="Result file, including every prediction")],
    threads: Annotated[int, typer.Option(min=1, max=8)] = 1,
) -> None:
    """Challenge BirdNET with own/other backgrounds, keeping enrollment audio fixed."""

    from xinyenyana.background_probe import run_background_challenge

    if not os.environ.get("XENWARDEN_LEASE"):
        raise typer.BadParameter("run this measurement through an admitted XenWarden lease")
    if not os.environ.get(ARCHIVE_VARIABLE):
        raise typer.BadParameter(f"{ARCHIVE_VARIABLE} must be configured before extraction")
    result = run_background_challenge(
        _load_endpoint(endpoint, sample_root), cache_root=cache_root, threads=threads
    )
    _emit(result, output, require_archive=True)


@app.command("audit-multiview-gate")
def audit_multiview_gate_command(
    archive: Annotated[Path, typer.Option(help="Existing RookID.zip on xen1")],
    cache_root: Annotated[
        Path, typer.Option(help="Per-recording checkpoints and temporary local audio")
    ],
    output: Annotated[Path, typer.Option(help="Private result with complete filter provenance")],
) -> None:
    """Measure E-REID-02 acoustic applicability without fitting an identity model."""

    from xinyenyana.multiview_gate import run_multiview_gate

    if not os.environ.get("XENWARDEN_LEASE") or not os.environ.get(ARCHIVE_VARIABLE):
        raise typer.BadParameter(
            "this gate requires an admitted lease and configured result archive"
        )
    _emit(run_multiview_gate(archive, cache_root), output, require_archive=True)


@app.command("run-frozen-probe")
def run_frozen_probe_command(
    endpoint: Annotated[str, typer.Option(help="A published benchmark endpoint")],
    sample_root: Annotated[Path, typer.Option(help="That endpoint's existing sample or release")],
    cache_root: Annotated[Path, typer.Option(help="Resumable feature cache on xen1")],
    output: Annotated[Path, typer.Option(help="Result file, including every prediction")],
    threads: Annotated[int, typer.Option(min=1, max=8)] = 1,
    permutations: Annotated[
        int, typer.Option(min=0, help="Label shuffles for each null (v5: 999; PA-V6: 9,999)")
    ] = 999,
    replicates: Annotated[
        int, typer.Option(min=1, help="Animal bootstrap draws (v5: 2,000; PA-V6: 10,000)")
    ] = 2000,
) -> None:
    """Compare frozen BirdNET layers and pooling, with every available background control."""

    from xinyenyana.frozen_probe import run_frozen_probe

    if not os.environ.get("XENWARDEN_LEASE"):
        raise typer.BadParameter("run this measurement through an admitted XenWarden lease")
    if not os.environ.get(ARCHIVE_VARIABLE):
        raise typer.BadParameter(f"{ARCHIVE_VARIABLE} must be configured before extraction")
    result = run_frozen_probe(
        _load_endpoint(endpoint, sample_root),
        cache_root=cache_root,
        threads=threads,
        permutations=permutations,
        bootstrap_replicates=replicates,
    )
    _emit(result, output, require_archive=True)


@app.command("run-song-removed")
def run_song_removed_command(
    sample_root: Annotated[Path, typer.Option(help="The Great Tit sample")],
    cache_root: Annotated[Path, typer.Option(help="Resumable feature cache on xen1")],
    scratch_root: Annotated[Path, typer.Option(help="Where the masked clips are written")],
    output: Annotated[Path, typer.Option(help="Result file, including every prediction")],
    endpoint: Annotated[str, typer.Option(help="The endpoint that publishes notes")] = "great-tit",
    threads: Annotated[int, typer.Option(min=1, max=8)] = 1,
    permutations: Annotated[
        int, typer.Option(min=0, help="Label shuffles for each null (v5: 999; PA-V6: 9,999)")
    ] = 999,
    replicates: Annotated[
        int, typer.Option(min=1, help="Animal bootstrap draws (v5: 2,000; PA-V6: 10,000)")
    ] = 2000,
) -> None:
    """Score the untouched and song-removed arms on the same clips and split.

    Only the Great Tit publishes per-note annotations, so no other endpoint gets
    this arm. What survives note removal is either the recording or sound the
    bird made outside the annotated notes, and this does not separate them.
    """

    from xinyenyana.frozen_probe import endpoint_audio_provenance, extract_endpoint, runtime_record
    from xinyenyana.song_removed import compare_arms, masked_endpoint

    if not os.environ.get("XENWARDEN_LEASE"):
        raise typer.BadParameter("run this measurement through an admitted XenWarden lease")
    if not os.environ.get(ARCHIVE_VARIABLE):
        raise typer.BadParameter(f"{ARCHIVE_VARIABLE} must be configured before extraction")
    loaded = _load_endpoint(endpoint, sample_root)
    masked = masked_endpoint(loaded, scratch=scratch_root)
    untouched_vectors, untouched_extraction = extract_endpoint(
        loaded, cache_root=cache_root, threads=threads
    )
    masked_vectors, masked_extraction = extract_endpoint(
        masked, cache_root=cache_root, threads=threads
    )
    _emit(
        {
            **compare_arms(
                loaded,
                untouched_vectors=untouched_vectors["embedding.mean"],
                masked_vectors=masked_vectors["embedding.mean"],
                representation="embedding.mean",
                permutations=permutations,
                bootstrap_replicates=replicates,
            ),
            "extraction": {
                "untouched": untouched_extraction,
                "song_removed": masked_extraction,
            },
            "environment": runtime_record(),
            "data": {
                "untouched": endpoint_audio_provenance(loaded),
                "song_removed": endpoint_audio_provenance(masked),
            },
        },
        output,
        require_archive=True,
    )


@app.command("run-nuisance")
def run_nuisance_command(
    endpoint: Annotated[str, typer.Option(help="A published benchmark endpoint")],
    sample_root: Annotated[Path, typer.Option(help="That endpoint's existing sample or release")],
    cache_root: Annotated[Path, typer.Option(help="Resumable feature cache on xen1")],
    output: Annotated[Path, typer.Option(help="Result file")],
    session_key: Annotated[
        str | None, typer.Option(help="The context field holding the session, where one exists")
    ] = None,
    threads: Annotated[int, typer.Option(min=1, max=8)] = 1,
) -> None:
    """Read the session out of a representation, and score the context baseline.

    Two readings, both registered in `docs/measurement-protocol.md`. Neither
    separates a recording effect from a social or spatial one and the result
    says so.
    """

    from xinyenyana.frozen_probe import endpoint_audio_provenance, extract_endpoint, runtime_record
    from xinyenyana.nuisance import run_nuisance

    if not os.environ.get("XENWARDEN_LEASE"):
        raise typer.BadParameter("run this measurement through an admitted XenWarden lease")
    if not os.environ.get(ARCHIVE_VARIABLE):
        raise typer.BadParameter(f"{ARCHIVE_VARIABLE} must be configured before extraction")
    loaded = _load_endpoint(endpoint, sample_root)
    data = endpoint_audio_provenance(loaded)
    vectors, extraction = extract_endpoint(loaded, cache_root=cache_root, threads=threads)
    _emit(
        {
            **run_nuisance(
                loaded,
                vectors["embedding.mean"],
                representation="embedding.mean",
                session_key=session_key,
            ),
            "extraction": extraction,
            "environment": runtime_record(),
            "data": data,
        },
        output,
        require_archive=True,
    )


@app.command("run-stowell-augmentation")
def run_stowell_augmentation_command(
    endpoint: Annotated[str, typer.Option(help="An endpoint publishing background recordings")],
    sample_root: Annotated[Path, typer.Option(help="That endpoint's unpacked release")],
    cache_root: Annotated[Path, typer.Option(help="Resumable feature cache on xen1")],
    scratch_root: Annotated[Path, typer.Option(help="Where the mixed training clips are written")],
    output: Annotated[Path, typer.Option(help="Result file")],
    threads: Annotated[int, typer.Option(min=1, max=8)] = 1,
    replicates: Annotated[
        int, typer.Option(min=1, help="Animal bootstrap draws (PA-V6: 10,000)")
    ] = 2000,
    permutations: Annotated[
        int, typer.Option(min=0, help="Label shuffles for each null (v5: 999; PA-V6: 9,999)")
    ] = 999,
) -> None:
    """Run the 2019 paper's own remedy and score it with the 2019 paper's own test.

    Their fix enlarges the training material so that every individual's ambient
    sound appears under every individual's calls. Their diagnostic trains on
    calls and scores background-only clips. Both arms are scored by the same
    gate on the same query clips, so the arms differ in the training material
    and nothing else.
    """

    from xinyenyana.frozen_probe import endpoint_audio_provenance, extract_endpoint, runtime_record
    from xinyenyana.stowell_augmentation import AUGMENTED, UNTOUCHED, augmented_endpoint
    from xinyenyana.validity import run_validity_gate

    if not os.environ.get("XENWARDEN_LEASE"):
        raise typer.BadParameter("run this measurement through an admitted XenWarden lease")
    if not os.environ.get(ARCHIVE_VARIABLE):
        raise typer.BadParameter(f"{ARCHIVE_VARIABLE} must be configured before extraction")
    loaded = _load_endpoint(endpoint, sample_root)
    data = endpoint_audio_provenance(loaded)
    plain, plain_extraction = extract_endpoint(loaded, cache_root=cache_root, threads=threads)
    enlarged, plan = augmented_endpoint(loaded, scratch=scratch_root)
    grown, grown_extraction = extract_endpoint(enlarged, cache_root=cache_root, threads=threads)
    _emit(
        {
            "experiment": "PA-STOWELL-REMEDY",
            "endpoint": loaded.name,
            "augmentation": plan,
            "arms": {
                UNTOUCHED: run_validity_gate(
                    endpoint=loaded,
                    vectors=plain["embedding.mean"],
                    representation="embedding.mean",
                    bootstrap_replicates=replicates,
                    permutations=permutations,
                ),
                AUGMENTED: run_validity_gate(
                    endpoint=enlarged,
                    vectors=grown["embedding.mean"],
                    representation="embedding.mean",
                    bootstrap_replicates=replicates,
                    permutations=permutations,
                ),
            },
            "claim_boundary": (
                "the augmented arm's query clips are the untouched endpoint's query "
                "clips; only the training material differs between the two arms"
            ),
            "extraction": {UNTOUCHED: plain_extraction, AUGMENTED: grown_extraction},
            "environment": runtime_record(),
            "data": data,
        },
        output,
        require_archive=True,
    )


@app.command("run-geometry")
def run_geometry_command(
    endpoint: Annotated[str, typer.Option(help="A published benchmark endpoint")],
    sample_root: Annotated[Path, typer.Option(help="That endpoint's existing sample or release")],
    cache_root: Annotated[Path, typer.Option(help="Resumable feature cache on xen1")],
    output: Annotated[Path, typer.Option(help="Result file")],
    session_key: Annotated[str, typer.Option(help="The context field holding the session")],
    representation: Annotated[
        str, typer.Option(help="A representation by the name a sweep recorded it under")
    ] = "embedding.mean",
    device: Annotated[str, typer.Option(help="Where an encoder runs")] = "cuda",
    threads: Annotated[int, typer.Option(min=1, max=8)] = 1,
) -> None:
    """Ask whether identity and the recording occupy the same directions.

    Registered in `docs/measurement-protocol.md`. Runs only where sessions hold
    more than one individual and individuals appear in more than one session;
    anywhere else it records why it could not run rather than returning a number.
    """

    from xinyenyana.frozen_probe import endpoint_audio_provenance, extract_endpoint, runtime_record
    from xinyenyana.geometry import run_geometry
    from xinyenyana.nuisance import _session_granularity, _session_value

    if not os.environ.get("XENWARDEN_LEASE"):
        raise typer.BadParameter("run this measurement through an admitted XenWarden lease")
    if not os.environ.get(ARCHIVE_VARIABLE):
        raise typer.BadParameter(f"{ARCHIVE_VARIABLE} must be configured before extraction")
    loaded = _load_endpoint(endpoint, sample_root)
    missing = [r.filename for r in loaded.records if session_key not in r.context]
    if missing:
        raise typer.BadParameter(
            f"{len(missing)} clips carry no {session_key!r}, first {missing[0]}"
        )
    granularity = _session_granularity(loaded.name, session_key)
    sessions = [_session_value(r.context[session_key], granularity) for r in loaded.records]
    data = endpoint_audio_provenance(loaded)
    # A BirdNET layer comes from the resumable cache every other measurement
    # uses. Any other representation goes through the same loader the sweep used,
    # so a geometry figure and a ranking figure cannot describe different vectors.
    if representation.startswith(("embedding.", "post_", "stage")):
        cached, extraction = extract_endpoint(loaded, cache_root=cache_root, threads=threads)
        if representation not in cached:
            raise typer.BadParameter(f"{representation!r} is not a cached layer: {sorted(cached)}")
        matrix = cached[representation]
    else:
        try:
            chosen = representation_from_name(representation)
        except ValueError as error:
            raise typer.BadParameter(str(error)) from error
        matrix = vectors_for(chosen, loaded.records, device=device)
        extraction = {"representation": representation, "loader": "xinyenyana.a5.vectors_for"}
    splits: dict[str, Any] = {}
    for split in ("enrollment", "query"):
        rows = [index for index, record in enumerate(loaded.records) if record.split == split]
        if not rows:
            continue
        import numpy as np

        splits[split] = run_geometry(
            features=np.asarray(matrix)[rows],
            identities=[loaded.records[index].identity for index in rows],
            sessions=[sessions[index] for index in rows],
            representation=representation,
        )
    _emit(
        {
            "experiment": "PA-REPRESENTATION-GEOMETRY",
            "endpoint": loaded.name,
            "representation": representation,
            "session_key": session_key,
            "session_granularity": granularity,
            "manifest_sha256": loaded.manifest_sha256,
            "splits_measured": splits,
            "splits": loaded.split_digests(),
            "extraction": extraction,
            "environment": runtime_record(),
            "data": data,
        },
        output,
        require_archive=True,
    )


@app.command("run-a5")
def run_a5(
    endpoint: Annotated[str, typer.Option(help="rookid, zebra-finch or great-tit")],
    sample_root: Annotated[Path, typer.Option(help="That endpoint's built sample")],
    output: Annotated[Path, typer.Option(help="Where the measurement is written")],
    device: Annotated[str, typer.Option(help="Torch device")] = "cuda",
    checkpoint_dir: Annotated[
        Path | None, typer.Option(help="Keep each finished model here, so a displaced run resumes")
    ] = None,
) -> None:
    """Rank every pretrained encoder in A5 against BirdNET on one endpoint."""

    summary = run_a5_endpoint(
        endpoint=_load_endpoint(endpoint, sample_root), device=device, checkpoint_dir=checkpoint_dir
    )
    _emit(summary, output)


def _require_lease_and_archive() -> None:
    if not os.environ.get("XENWARDEN_LEASE"):
        raise typer.BadParameter("run this measurement through an admitted XenWarden lease")
    if not os.environ.get(ARCHIVE_VARIABLE):
        raise typer.BadParameter(f"{ARCHIVE_VARIABLE} must be configured before the run")


@app.command("run-identity")
def run_identity_command(
    endpoint: Annotated[str, typer.Option(help="Any endpoint, or rookid-full-width")],
    sample_root: Annotated[Path, typer.Option(help="That endpoint's built sample")],
    cache_root: Annotated[Path, typer.Option(help="Resumable BirdNET feature cache on xen1")],
    output: Annotated[Path, typer.Option(help="Result file")],
    condition: Annotated[str, typer.Option(help="foreground or background")] = "foreground",
    rook_split: Annotated[
        str, typer.Option(help="rookid-full-width only: across-year or across-day-2020")
    ] = "across-year",
    channel: Annotated[int, typer.Option(help="rookid-full-width only: enrolment channel")] = 0,
    score_channel: Annotated[
        int | None,
        typer.Option(help="rookid-full-width only: score the same calls on this channel"),
    ] = None,
    balance_enrolment: Annotated[
        int | None, typer.Option(help="Cap every animal's enrolment at this many clips")
    ] = None,
    random_fold_seed: Annotated[
        int | None, typer.Option(help="Divide the same clips at random per animal instead")
    ] = None,
    equivalence_margin: Annotated[
        float | None, typer.Option(help="Test accuracy below chance plus this margin")
    ] = None,
    threads: Annotated[int, typer.Option(min=1, max=8)] = 4,
    permutations: Annotated[
        int, typer.Option(help="Label shuffles for each null (v5: 999; PA-V6: 9,999)")
    ] = 999,
    replicates: Annotated[
        int, typer.Option(help="Bootstrap draws for each interval (v5: 2,000; PA-V6: 10,000)")
    ] = 2000,
) -> None:
    """BirdNET's final embedding on one endpoint, with every null the paper reports."""

    from xinyenyana.frozen_probe import endpoint_audio_provenance, extract_endpoint, runtime_record
    from xinyenyana.identity_run import (
        FINAL_EMBEDDING,
        call_key,
        measure,
        random_fold,
    )
    from xinyenyana.identity_run import (
        balance_enrolment as balance,
    )

    _require_lease_and_archive()
    selection: dict[str, Any] = {}
    if endpoint == ROOK_FULL_WIDTH:
        from xinyenyana.e02 import load_e02_rookid

        loaded, selection = load_e02_rookid(
            manifest_path=sample_root / "e02-manifest.json",
            sample_root=sample_root,
            split=rook_split,
            channel=channel,
        )
        records = list(loaded.records)
        if score_channel is not None:
            other, _ = load_e02_rookid(
                manifest_path=sample_root / "e02-manifest.json",
                sample_root=sample_root,
                split=rook_split,
                channel=score_channel,
            )
            wanted = {call_key(r) for r in records if r.split == "query"}
            scored = [r for r in other.records if r.split == "query" and call_key(r) in wanted]
            records = [r for r in records if r.split == "enrollment"] + scored
            selection["enrolled_on_channel"] = channel
            selection["scored_on_channel"] = score_channel
            selection["paired_channel_query_clips"] = len(scored)
    else:
        if score_channel is not None or channel != 0:
            raise typer.BadParameter("channels are chosen only on rookid-full-width")
        loaded = _load_endpoint(endpoint, sample_root)
        records = list(loaded.records)
    if balance_enrolment is not None:
        records = balance(records, balance_enrolment)
        selection["balanced_enrolment_cap"] = balance_enrolment
    if random_fold_seed is not None:
        records, fold = random_fold(records, random_fold_seed)
        selection["random_fold"] = fold
    subset = type(loaded)(
        name=loaded.name,
        records=tuple(records),
        categorical_targets=loaded.categorical_targets,
        manifest_sha256=loaded.manifest_sha256,
        source_document=loaded.source_document,
    )
    vectors, extraction = extract_endpoint(subset, cache_root=cache_root, threads=threads)
    _emit(
        {
            **measure(
                endpoint=subset,
                records=records,
                vectors=vectors[FINAL_EMBEDDING],
                condition=condition,
                equivalence_margin=equivalence_margin,
                replicates=replicates,
                permutations=permutations,
            ),
            "selection": selection,
            "manifest_sha256": loaded.manifest_sha256,
            "splits": subset.split_digests(),
            "extraction": extraction,
            "environment": runtime_record(),
            "data": endpoint_audio_provenance(subset),
        },
        output,
        require_archive=True,
    )


@app.command("run-confound-index")
def run_confound_index_command(
    output: Annotated[Path, typer.Option(help="Result file")],
    endpoint: Annotated[list[str], typer.Option(help="name=sample-root, once per endpoint")],
) -> None:
    """Identity against every categorical axis each endpoint's own loader carries."""

    from xinyenyana.confound_index import confound_index, verdicts

    _require_lease_and_archive()
    report: dict[str, Any] = {"endpoints": {}}
    for pair in endpoint:
        name, _, root = pair.partition("=")
        loaded = _load_endpoint(name, Path(root))
        index = confound_index(name=loaded.name, records=loaded.records)
        index["verdicts"] = verdicts(index)
        index["sample_root"] = root
        index["manifest_sha256"] = loaded.manifest_sha256
        report["endpoints"][name] = index
    _emit(report, output, require_archive=True)


@app.command("run-place-control")
def run_place_control_command(
    sample_root: Annotated[Path, typer.Option(help="The great tit sample")],
    output: Annotated[Path, typer.Option(help="Result file")],
) -> None:
    """Whether any great tit nestbox or recording appears on both sides of the split."""

    from xinyenyana.identity_run import place_control

    _require_lease_and_archive()
    loaded = _load_endpoint("great-tit", sample_root)
    _emit(
        {
            "endpoint": loaded.name,
            "manifest_sha256": loaded.manifest_sha256,
            **place_control(loaded.records),
        },
        output,
        require_archive=True,
    )


@app.command("run-encoder-comparison")
def run_encoder_comparison_command(
    sweep: Annotated[list[Path], typer.Option(help="A run-a5 result file, once per endpoint")],
    output: Annotated[Path, typer.Option(help="Result file")],
) -> None:
    """Paired tests against BirdNET per endpoint, and rank consistency across endpoints."""

    from xinyenyana.archive import sha256_file as digest
    from xinyenyana.encoder_comparison import rank_consistency, summarise_sweep

    _require_lease_and_archive()
    per_endpoint: dict[str, Any] = {}
    accuracy: dict[str, dict[str, float]] = {}
    for path in sweep:
        loaded = json.loads(path.read_text())
        name = str(loaded["endpoint"])
        per_endpoint[name] = {**summarise_sweep(loaded), "source_sha256": digest(path)}
        accuracy[name] = {
            model.split("/")[-1]: float(entry["accuracy"])
            for model, entry in loaded["chosen"].items()
        }
    _emit(
        {"per_endpoint": per_endpoint, "rank_consistency": rank_consistency(accuracy)},
        output,
        require_archive=True,
    )


from xinyenyana import cli_v6  # noqa: E402,F401

if __name__ == "__main__":  # pragma: no cover
    app()
