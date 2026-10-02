"""BirdPark: zebra finches in one box, on one microphone, named by their own sensors.

Every other endpoint in this project gives each animal its own recordings, so
the recording stays a candidate explanation for any identity result and the
project can only probe for it. RookID is the exception and it carries five birds
and 45 clips.

Here eight birds share one aviary and one wall-mounted microphone across seven
sessions, and each bird's identity comes from an accelerometer transmitter it is
wearing. Two things follow. The label cannot have been derived from the audio,
which is the weakness the Great Tit, the little owl and the tree pipit carry.
And within a session the microphone, the room, the equipment and the ambient
sound are the same for every bird, so none of them can be what separates them.

What is not held constant is where a bird was sitting, so per-bird distance to
the microphone, level and reverberation remain available. This endpoint removes
the recording and it does not remove the position.

The publisher states that a consolidated segment's ``bird`` equals its
``transmitter`` when a transmitter recognised the vocalisation, and that a
``transmitter`` of ``None`` means the bird id was guessed. Only the confirmed
ones are used.
"""

from __future__ import annotations

import hashlib
import json
import wave
import zipfile
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from xinyenyana.a2 import ClipRecord, Endpoint
from xinyenyana.archive import sha256_file

#: The experiments the release carries, and the endpoint name each becomes.
#: The two `cop` experiments hold two birds in one recording file; they are
#: built and they carry no identity claim, because two identities and one
#: session cannot support one.
BIRDPARK_ENDPOINTS: dict[str, str] = {
    "birdpark-juv01": "juvExpBP01",
    "birdpark-juv03": "juvExpBP03",
    "birdpark-cop08": "copExpBP08",
    "birdpark-cop09": "copExpBP09",
}

#: Endpoints whose shape supports a session-disjoint split at all.
SPLITTABLE = ("birdpark-juv01", "birdpark-juv03")

BIRDPARK_MANIFEST = "birdpark-manifest.json"

#: The publisher's own rate for both the transmitter and microphone channels.
SAMPLE_RATE_HZ = 24414.0625

#: A WAV header carries an integer rate. 24,414 against 24414.0625 is 2.6 parts
#: per million, which is far below the 3e-6 agreement the audio path was
#: established at, and the true rate is kept in the manifest.
WAV_RATE_HZ = 24_414

#: The one microphone every bird in a group shares. Taking each segment's own
#: `firstMic` would let the channel vary between birds and reintroduce the
#: variable this endpoint exists to hold constant.
#:
#: Read from all sixteen recordings on 2026-09-15: `daqChNames` is `Mic1` to
#: `Mic7`, and `daqChDescs` puts `Mic1` centred above the rear stage wall, a
#: Rode NT1 at about 17.0 dB re 1V/Pa. The metadata description document gives
#: `Microphone1` to `Microphone4` as an EXAMPLE value for that field, and the
#: release uses neither those names nor that count. `Mic6` and `Mic7` sit inside
#: the two nests on another microphone type at another gain, so they are
#: per-location and are never read here.
MICROPHONE = "Mic1"
FIRST_MIC = "Mic1"

SPLIT_SALT = "xyy-birdpark-20260915"

#: Three of the seven sessions are scored. Fixed here; no score moves it.
QUERY_SESSIONS = 3

CONSOLIDATED = "consolidated"


def _text(value: Any) -> str:
    return value.decode() if isinstance(value, bytes) else str(value)


def select_segments(rows: Sequence[Any], *, experiment: str) -> list[dict[str, Any]]:
    """The registered filters, applied in one place so they can be counted.

    A row is kept when it is consolidated, its transmitter recognised the
    vocalisation so the bird id was not guessed, it overlaps no other
    consolidated segment, and it is visible on the shared microphone.
    """

    kept: list[dict[str, Any]] = []
    for row in rows:
        if _text(row["expInstanceId"]) != experiment:
            continue
        if _text(row["segType"]) != CONSOLIDATED:
            continue
        if _text(row["transmitter"]) == "None":
            continue
        bird = _text(row["bird"])
        if bird in ("None", ""):
            continue
        if float(row["nOverlapSyl"]) != 0.0:
            continue
        if _text(row["firstMic"]) != FIRST_MIC:
            continue
        onset, offset = int(row["onset"]), int(row["offset"])
        if onset < 1 or offset < onset:
            raise ValueError(f"{experiment}: a segment runs from {onset} to {offset}")
        kept.append(
            {
                "filename": _text(row["filename"]),
                "bird": bird,
                "transmitter": _text(row["transmitter"]),
                "onset": onset,
                "offset": offset,
            }
        )
    return kept


def session_split(sessions: Sequence[str], *, experiment: str) -> dict[str, str]:
    """Which recording files are scored. No timestamp orders this and no score moves it."""

    ranked = sorted(
        set(sessions),
        key=lambda name: hashlib.sha256(f"{SPLIT_SALT}:{experiment}:{name}".encode()).digest(),
    )
    query = set(ranked[:QUERY_SESSIONS])
    return {name: ("query" if name in query else "enrollment") for name in sorted(set(sessions))}


def read_segments_table(path: Path, *, experiment: str) -> list[dict[str, Any]]:
    """The release's annotation table, filtered to one experiment."""

    import h5py

    with h5py.File(str(path), "r") as handle:
        rows = handle["segments"][:]
    return select_segments(rows, experiment=experiment)


def _check_recording(handle: Any, member: str) -> int:
    """Refuse a file whose own header disagrees with the publisher's description.

    A disagreement here is an execution error, not a scientific finding.
    """

    info = handle["recInfo"].attrs
    rate = float(info["sampleRate_Hz"])
    if abs(rate - SAMPLE_RATE_HZ) > 1e-6:
        raise ValueError(f"{member}: recInfo gives {rate} Hz, not the published {SAMPLE_RATE_HZ}")
    names = [_text(value) for value in info["daqChNames"]]
    if MICROPHONE not in names:
        raise ValueError(f"{member}: daqChNames is {names}, which does not carry {MICROPHONE}")
    index = names.index(MICROPHONE)
    rows = int(handle["daqSignals"].shape[0])
    if rows != len(names):
        raise ValueError(f"{member}: {rows} daq rows for {len(names)} channel names")
    return index


def _write_pcm16(path: Path, samples: Any) -> None:
    import numpy as np

    peak = float(np.max(np.abs(samples))) if samples.size else 0.0
    if not peak > 0:
        raise ValueError(f"{path.name} is silent")
    values = np.clip(np.round(np.asarray(samples) * 32767.0), -32768, 32767).astype("<i2")
    with wave.open(str(path), "wb") as audio:
        audio.setnchannels(1)
        audio.setsampwidth(2)
        audio.setframerate(WAV_RATE_HZ)
        audio.writeframes(values.tobytes())


def build_birdpark_sample(
    *,
    endpoint: str,
    segments_path: Path,
    data_archive: Path,
    sample_root: Path,
    scratch_root: Path,
) -> dict[str, Any]:
    """Cut every kept segment out of the shared microphone channel.

    One archive member is streamed to scratch at a time and removed afterwards,
    because the release is 12.3 GB and none of it belongs in the sample.

    One common gain is applied across the endpoint rather than a peak scaling
    per clip, so level differences between clips survive into the level control
    instead of being flattened out of it.
    """

    import h5py
    import numpy as np

    if endpoint not in BIRDPARK_ENDPOINTS:
        raise ValueError(f"unknown BirdPark endpoint {endpoint!r}: {sorted(BIRDPARK_ENDPOINTS)}")
    experiment = BIRDPARK_ENDPOINTS[endpoint]
    kept = read_segments_table(segments_path, experiment=experiment)
    if not kept:
        raise ValueError(f"{endpoint}: the annotation table holds no usable segment")
    split_of = session_split([row["filename"] for row in kept], experiment=experiment)

    sample_root.mkdir(parents=True, exist_ok=True)
    scratch_root.mkdir(parents=True, exist_ok=True)
    cut: dict[str, Any] = {}
    members: dict[str, dict[str, Any]] = {}
    with zipfile.ZipFile(data_archive) as bundle:
        names = {Path(name).as_posix(): name for name in bundle.namelist()}
        for session in sorted({row["filename"] for row in kept}):
            member = next(
                (name for key, name in names.items() if key.endswith(session)),
                None,
            )
            if member is None:
                raise ValueError(f"{data_archive} holds no member for {session}")
            working = scratch_root / Path(session).name
            with bundle.open(member) as source, working.open("wb") as sink:
                while chunk := source.read(1 << 24):
                    sink.write(chunk)
            try:
                with h5py.File(str(working), "r") as handle:
                    channel = _check_recording(handle, member)
                    signal = np.asarray(handle["daqSignals"][channel, :], dtype=np.float64)
                members[session] = {
                    "member": member,
                    "samples": int(signal.size),
                    "channel_index": channel,
                }
                for row in kept:
                    if row["filename"] != session:
                        continue
                    start, stop = row["onset"] - 1, row["offset"]
                    if stop > signal.size:
                        raise ValueError(
                            f"{session}: a segment ends at {stop} in a channel of {signal.size}"
                        )
                    key = f"{Path(session).stem}-{row['onset']}"
                    if key in cut:
                        # Two kept segments at one onset in one recording would
                        # overlap, and an overlapping segment is already
                        # excluded, so this cannot happen in the release. It
                        # raises rather than overwriting, because a silent
                        # overwrite loses a clip without saying so.
                        raise ValueError(f"{session}: two kept segments start at {row['onset']}")
                    cut[key] = (row, signal[start:stop])
            finally:
                working.unlink(missing_ok=True)

    peak = max(float(np.max(np.abs(values))) for _, values in cut.values())
    if not peak > 0:
        raise ValueError(f"{endpoint}: every kept segment is silent")
    gain = 0.95 / peak

    entries: list[dict[str, Any]] = []
    for name, (row, values) in sorted(cut.items()):
        relative = f"clips/{name}.wav"
        (sample_root / "clips").mkdir(parents=True, exist_ok=True)
        _write_pcm16(sample_root / relative, values * gain)
        entries.append(
            {
                "filename": name,
                "identity": row["bird"],
                "transmitter": row["transmitter"],
                "session": row["filename"],
                "split": split_of[row["filename"]],
                "onset": row["onset"],
                "offset": row["offset"],
                "samples": int(values.size),
                "local_path": relative,
                "sha256": sha256_file(sample_root / relative),
            }
        )

    present = {(entry["identity"], entry["split"]) for entry in entries}
    dropped = sorted(
        {
            entry["identity"]
            for entry in entries
            if (entry["identity"], "enrollment") not in present
            or (entry["identity"], "query") not in present
        }
    )
    manifest = {
        "endpoint": endpoint,
        "experiment": experiment,
        "source": "https://doi.org/10.5281/zenodo.13144875",
        "licence": "GPL-2.0-or-later",
        "segments_sha256": sha256_file(segments_path),
        "microphone": MICROPHONE,
        "sample_rate_hz": SAMPLE_RATE_HZ,
        "wav_rate_hz": WAV_RATE_HZ,
        "split_salt": SPLIT_SALT,
        "query_sessions": QUERY_SESSIONS,
        "session_split": split_of,
        "common_gain": gain,
        "recordings": members,
        "identities_missing_a_side": dropped,
        "entries": entries,
    }
    (sample_root / BIRDPARK_MANIFEST).write_text(json.dumps(manifest, indent=2, sort_keys=True))
    return {key: value for key, value in manifest.items() if key not in ("entries", "recordings")}


def load_birdpark(*, manifest_path: Path, sample_root: Path) -> Endpoint:
    """The endpoint a build wrote, with any one-sided bird dropped."""

    if not manifest_path.is_file():
        raise ValueError(
            f"no BirdPark sample at {manifest_path}; "
            "build it with `xinyenyana build-birdpark-sample`"
        )
    manifest = json.loads(manifest_path.read_text())
    endpoint = str(manifest["endpoint"])
    if endpoint not in BIRDPARK_ENDPOINTS:
        raise ValueError(f"unknown BirdPark endpoint {endpoint!r}")
    dropped = set(manifest["identities_missing_a_side"])
    records = tuple(
        ClipRecord(
            filename=str(entry["filename"]),
            path=sample_root / str(entry["local_path"]),
            identity=str(entry["identity"]),
            split=str(entry["split"]),
            context={
                "condition": "foreground",
                "species": "zebra-finch",
                "session": str(entry["session"]),
                "microphone": MICROPHONE,
                "transmitter": str(entry["transmitter"]),
                "onset": int(entry["onset"]),
            },
        )
        for entry in manifest["entries"]
        if str(entry["identity"]) not in dropped
    )
    if not records:
        raise ValueError(f"{endpoint} carries no clips once one-sided birds are dropped")
    return Endpoint(
        name=endpoint,
        records=records,
        categorical_targets=(),
        manifest_sha256=str(manifest["segments_sha256"]),
        source_document="docs/benchmark-data.md",
    )


def coverage(endpoint: Endpoint) -> dict[str, Any]:
    """Per-bird and per-session counts, so an unbalanced split is visible."""

    per_identity: dict[str, dict[str, Any]] = {}
    per_session: dict[str, set[str]] = {}
    for record in endpoint.records:
        entry = per_identity.setdefault(
            record.identity, {"enrollment": 0, "query": 0, "sessions": set()}
        )
        entry[record.split] += 1
        entry["sessions"].add(str(record.context["session"]))
        per_session.setdefault(str(record.context["session"]), set()).add(record.identity)
    return {
        "per_identity": {
            identity: {
                "enrollment": entry["enrollment"],
                "query": entry["query"],
                "sessions": len(entry["sessions"]),
            }
            for identity, entry in sorted(per_identity.items())
        },
        "birds_per_session": {
            session: len(identities) for session, identities in sorted(per_session.items())
        },
        "identities": len(per_identity),
        "uniform_chance": 1.0 / len(per_identity),
    }
