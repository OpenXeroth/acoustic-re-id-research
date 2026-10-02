"""The Egyptian fruit bat corpus, and the split its own metadata already carries.

Prat, Taub and Yovel published 91,080 annotated vocalisations from a captive
colony across 87,986 recordings, with the emitter of each identified from
synchronised video. A recording can hold more than one annotated vocalisation,
and each annotation names the span of samples it covers, so a vocalisation is
cut from its recording rather than taken as the whole file.

Nearly half the annotations carry a minus-signed emitter, which the authors
define as the case where the interacting pair was recognised but their roles are
in doubt, so the call could have come from either bat. Those are not
attributions and are not used here.

The colony was housed in numbered treatments with recorded date ranges. Six bats
were recorded in treatments 16 to 19, from December 2012 to June 2013, and again
in treatment 20 in February 2014. Enrolling on the first period and scoring on
the second is an across-session test the release already contains.
"""

from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from xinyenyana.a2 import ClipRecord, Endpoint
from xinyenyana.archive import canonical_sha256, sha256_file

#: The six bats recorded in both periods, and the treatments of each period.
ACROSS_TREATMENT_IDENTITIES: tuple[str, ...] = ("207", "208", "211", "215", "216", "221")
ACROSS_TREATMENT_ENROLLMENT: tuple[str, ...] = ("16", "17", "18", "19")
ACROSS_TREATMENT_QUERY: tuple[str, ...] = ("20",)

#: Endpoint name to the identities, the enrolling treatments and the scoring
#: treatments. Every split here is read from the release's own metadata.
BAT_ENDPOINTS: dict[str, tuple[tuple[str, ...], tuple[str, ...], tuple[str, ...]]] = {
    "bat-acrosstreatment": (
        ACROSS_TREATMENT_IDENTITIES,
        ACROSS_TREATMENT_ENROLLMENT,
        ACROSS_TREATMENT_QUERY,
    ),
}

#: The name of the manifest a built sample is read back through.
BAT_MANIFEST = "bat-manifest.json"


@dataclass(frozen=True)
class Span:
    """One annotated vocalisation: where it is, whose it is, and which split."""

    identity: str
    split: str
    folder: str
    source_name: str
    start: int
    end: int
    treatment: str
    recorded: str
    channel: str

    @property
    def name(self) -> str:
        """A name unique across the release that says where it came from."""

        stem = self.source_name.removesuffix(".WAV").removesuffix(".wav")
        return f"{self.folder}-{stem}-{self.start}-{self.end}.wav"


def attributed_emitter(value: str) -> str | None:
    """The bat that called, or nothing.

    ``0`` is an unknown emitter. A minus sign means the pair was recognised and
    which of the two called is in doubt, so the row names a bat but does not say
    it called. Neither is an attribution.
    """

    identity = value.strip()
    if not identity or identity == "0" or identity.startswith("-"):
        return None
    return identity


def _read_csv(path: Path, *, required: tuple[str, ...]) -> list[dict[str, str]]:
    if not path.is_file():
        raise ValueError(f"the release holds no table at {path}")
    csv.field_size_limit(10**7)
    with path.open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise ValueError(f"{path} lists nothing")
    for column in required:
        if column not in rows[0]:
            raise ValueError(f"{path} has no {column} column")
    return rows


def annotated_spans(*, endpoint: str, annotations_path: Path, file_info_path: Path) -> list[Span]:
    """Every annotated vocalisation this endpoint covers, in the table's order.

    One recording can hold several, so a span is keyed by its sample range
    rather than by the file it sits in.
    """

    if endpoint not in BAT_ENDPOINTS:
        raise ValueError(f"unknown bat endpoint {endpoint!r}: {sorted(BAT_ENDPOINTS)}")
    identities, enrolling, scoring = BAT_ENDPOINTS[endpoint]

    annotations = _read_csv(
        annotations_path, required=("FileID", "Emitter", "Start sample", "End sample")
    )
    files = {
        row["FileID"]: row
        for row in _read_csv(
            file_info_path,
            required=("FileID", "Treatment ID", "File name", "File folder", "Recording time"),
        )
    }

    spans = []
    for row in annotations:
        identity = attributed_emitter(row["Emitter"])
        if identity is None or identity not in identities:
            continue
        described = files.get(row["FileID"])
        if described is None:
            raise ValueError(f"annotation {row['FileID']} names a file the file table does not")
        treatment = described["Treatment ID"]
        if treatment in enrolling:
            split = "enrollment"
        elif treatment in scoring:
            split = "query"
        else:
            continue
        start = int(row["Start sample"])
        end = int(row["End sample"])
        if start < 1 or end < start:
            raise ValueError(
                f"annotation {row['FileID']} covers samples {start} to {end}, which is not a span"
            )
        spans.append(
            Span(
                identity=identity,
                split=split,
                folder=described["File folder"],
                source_name=described["File name"],
                start=start,
                end=end,
                treatment=treatment,
                recorded=described["Recording time"],
                channel=described.get("Recording channel", ""),
            )
        )
    return spans


def _read_pcm16(path: Path) -> tuple[Any, int]:
    """Every sample of a mono 16-bit recording, and its rate.

    The release is mono 16-bit at 250 kHz. Anything else is refused rather than
    read with an assumption about how its bytes are laid out.
    """

    import wave
    from array import array

    with wave.open(str(path), "rb") as stream:
        if stream.getnchannels() != 1 or stream.getsampwidth() != 2:
            raise ValueError(
                f"{path} is {stream.getnchannels()} channel and "
                f"{stream.getsampwidth() * 8} bit, not mono 16 bit"
            )
        rate = stream.getframerate()
        samples = array("h")
        samples.frombytes(stream.readframes(stream.getnframes()))
    return samples, rate


def _write_pcm16(path: Path, samples: Any, sample_rate: int) -> None:
    import wave

    with wave.open(str(path), "wb") as stream:
        stream.setnchannels(1)
        stream.setsampwidth(2)
        stream.setframerate(sample_rate)
        stream.writeframes(samples.tobytes())


def build_bat_sample(
    *,
    endpoint: str,
    annotations_path: Path,
    file_info_path: Path,
    audio_root: Path,
    sample_root: Path,
) -> dict[str, Any]:
    """Cut every annotated vocalisation into its own file and write the manifest.

    Only some of the release's archives are unpacked, so a span whose recording
    is not on disk is counted in the manifest rather than dropped silently.
    """

    spans = annotated_spans(
        endpoint=endpoint, annotations_path=annotations_path, file_info_path=file_info_path
    )
    clips = sample_root / "clips"
    entries = []
    absent = 0
    short = 0
    for span in spans:
        source = audio_root / span.folder / span.source_name
        if not source.is_file():
            absent += 1
            continue
        samples, sample_rate = _read_pcm16(source)
        cut = samples[span.start - 1 : span.end]
        if not len(cut):
            # The recording is shorter than the annotation says. Counted here,
            # never padded out to the length the annotation claims.
            short += 1
            continue
        destination = clips / span.folder / span.name
        destination.parent.mkdir(parents=True, exist_ok=True)
        _write_pcm16(destination, cut, sample_rate)
        entries.append(
            {
                "filename": span.name,
                "local_path": str(destination.relative_to(sample_root)),
                "identity": span.identity,
                "split": span.split,
                "treatment": span.treatment,
                "recorded": span.recorded,
                "channel": span.channel,
                "folder": span.folder,
                "source_name": span.source_name,
                "start_sample": span.start,
                "end_sample": span.end,
                "sample_rate": int(sample_rate),
                "samples": int(len(cut)),
            }
        )
    manifest = {
        "endpoint": endpoint,
        "annotated_spans": len(spans),
        "clips": len(entries),
        "recordings_not_unpacked": absent,
        "annotations_past_the_end_of_their_recording": short,
        "annotations_sha256": sha256_file(annotations_path),
        "file_info_sha256": sha256_file(file_info_path),
        "entries": entries,
    }
    sample_root.mkdir(parents=True, exist_ok=True)
    (sample_root / BAT_MANIFEST).write_text(json.dumps(manifest, indent=2, sort_keys=True))
    return {key: value for key, value in manifest.items() if key != "entries"}


def load_bats(*, manifest_path: Path, sample_root: Path) -> Endpoint:
    """One bat endpoint, read back from the sample its build step wrote."""

    if not manifest_path.is_file():
        raise ValueError(
            f"no bat sample at {manifest_path}; build it with `xinyenyana build-bat-sample`"
        )
    manifest = json.loads(manifest_path.read_text())
    endpoint = str(manifest["endpoint"])
    if endpoint not in BAT_ENDPOINTS:
        raise ValueError(f"unknown bat endpoint {endpoint!r}: {sorted(BAT_ENDPOINTS)}")
    identities = BAT_ENDPOINTS[endpoint][0]
    records = tuple(
        ClipRecord(
            filename=str(entry["filename"]),
            path=sample_root / str(entry["local_path"]),
            identity=str(entry["identity"]),
            split=str(entry["split"]),
            context={
                "condition": "foreground",
                "species": "egyptian-fruit-bat",
                "treatment": str(entry["treatment"]),
                "recorded": str(entry["recorded"]),
                "channel": str(entry["channel"]),
                "folder": str(entry["folder"]),
                "source_recording": str(entry["source_name"]),
            },
        )
        for entry in manifest["entries"]
    )
    if not records:
        raise ValueError(f"{endpoint} carries no clips; the sample was built from nothing")
    for identity in identities:
        for split in ("enrollment", "query"):
            if not any(r.identity == identity and r.split == split for r in records):
                raise ValueError(
                    f"{endpoint} has no {split} clip for bat {identity}; "
                    f"{manifest['recordings_not_unpacked']} of its annotated recordings "
                    "are not unpacked"
                )
    return Endpoint(
        name=endpoint,
        records=records,
        categorical_targets=(),
        manifest_sha256=canonical_sha256(
            sorted([str(manifest["annotations_sha256"]), str(manifest["file_info_sha256"])])
        ),
        source_document="docs/benchmark-data.md",
    )


def coverage(endpoint: Endpoint) -> dict[str, Any]:
    """Per-bat counts, so a partial unpack is visible rather than silent."""

    per_identity: dict[str, dict[str, Any]] = {}
    for record in endpoint.records:
        entry = per_identity.setdefault(
            record.identity, {"enrollment": 0, "query": 0, "days": set()}
        )
        entry[record.split] += 1
        entry["days"].add(str(record.context["recorded"])[:10])
    return {
        identity: {
            "enrollment": entry["enrollment"],
            "query": entry["query"],
            "recording_days": len(entry["days"]),
        }
        for identity, entry in sorted(per_identity.items())
    }
