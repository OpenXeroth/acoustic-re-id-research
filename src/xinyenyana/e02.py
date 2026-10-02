"""RookID at full width: every annotated call, every identity, all four channels.

E01d cut three calls each from five birds, one event type, one recording per
split, on one channel, and says in its own claim limit that it is a sensitivity
benchmark. This module builds the endpoint the archive actually supports: the
fifteen ringed birds, every admitted event type, all 81 recordings across 23
days and two calendar years, with all four microphone channels retained so that
enrolling on one microphone and scoring on another holds the animal and the
moment constant.

The streaming and integrity discipline is E01d's: each WAV is read once out of
the zip, its CRC-32 and SHA-256 are computed over every byte, and only the
registered signal windows are kept.
"""

from __future__ import annotations

import hashlib
import json
import os
import struct
import sys
import wave
import zipfile
import zlib
from array import array
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from xinyenyana.rookid import RookIDEvent

DEFAULT_PROTOCOL = Path("configs/experiments/e02-rookid-full-width.json")
PCM_SUBFORMAT = bytes.fromhex("0100000000001000800000aa00389b71")


def load_e02_protocol(path: Path) -> tuple[dict[str, Any], str]:
    raw = path.read_bytes()
    protocol = json.loads(raw)
    if (
        protocol.get("schema_version") != 1
        or protocol.get("experiment") != "E02"
        or protocol.get("registration_status") != "prospective_full_width"
    ):
        raise ValueError("unsupported E02 protocol")
    sample = protocol["sample"]
    identities = sample["identities"]
    if len(identities) != len(set(identities)) or len(identities) < 2:
        raise ValueError("E02 identities must be a unique list")
    if set(identities) & set(sample["non_identity_sources"]):
        raise ValueError("E02 identity list overlaps the non-identity sources")
    counts = {int(r["source_channels"]) for r in protocol["source"]["recordings"]}
    if not counts or min(counts) < 2:
        raise ValueError("E02 needs at least two channels on every recording")
    return protocol, hashlib.sha256(raw).hexdigest()


@dataclass(frozen=True)
class Candidate:
    recording: str
    day: str
    year: str
    row: int
    identity: str
    event: str
    comment: str
    start: float
    end: float
    signal_start: float
    signal_end: float


def select_e02_candidates(
    *, protocol: dict[str, Any], events: list[RookIDEvent], recording: str
) -> list[Candidate]:
    """Admit every annotation that is a bird, unambiguous, short enough and isolated."""

    sample = protocol["sample"]
    margin = float(sample["annotation_margin_seconds"])
    maximum = float(sample["maximum_signal_seconds"])
    identities = set(sample["identities"])
    ambiguity = [str(value).casefold() for value in sample["explicit_ambiguity_substrings"]]

    admitted: list[Candidate] = []
    for event in events:
        if event.identity not in identities:
            continue
        notes = f"{event.event} {event.comment}".casefold()
        if any(marker in notes for marker in ambiguity):
            continue
        signal_start = max(0.0, event.start - margin)
        signal_end = event.end + margin
        if signal_end - signal_start > maximum:
            continue
        # Overlap by a differently-labelled annotation, Inc and Pls included.
        if any(
            other.identity != event.identity
            and other.end > signal_start
            and other.start < signal_end
            for other in events
        ):
            continue
        admitted.append(
            Candidate(
                recording=recording,
                day=recording[:8],
                year=recording[:4],
                row=event.row_number,
                identity=event.identity,
                event=event.event,
                comment=event.comment,
                start=event.start,
                end=event.end,
                signal_start=signal_start,
                signal_end=signal_end,
            )
        )
    return sorted(admitted, key=lambda c: (c.identity, c.row))


def _read_exact(stream: Any, size: int, update: Any) -> bytes:
    value = bytearray()
    while len(value) < size:
        block = stream.read(size - len(value))
        if not block:
            raise ValueError(f"unexpected end of RookID WAV after {len(value)} bytes")
        update(block)
        value.extend(block)
    return bytes(value)


def _is_pcm16(payload: bytes, audio_format: int, bits: int) -> bool:
    if bits != 16:
        return False
    if audio_format == 1:
        return True
    return (
        audio_format == 0xFFFE
        and len(payload) >= 40
        and struct.unpack("<H", payload[16:18])[0] >= 22
        and struct.unpack("<H", payload[18:20])[0] == 16
        and payload[24:40] == PCM_SUBFORMAT
    )


def _write_clip(
    *, destination: Path, samples: array[int], sample_rate: int, target_frames: int
) -> dict[str, int]:
    if len(samples) > target_frames:
        raise ValueError("signal exceeds the registered clip duration")
    pad_left = (target_frames - len(samples)) // 2
    padded = array("h", [0]) * pad_left
    padded.extend(samples)
    padded.extend(array("h", [0]) * (target_frames - len(padded)))
    if sys.byteorder != "little":  # pragma: no cover
        padded.byteswap()
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".part")
    with wave.open(str(temporary), "wb") as output:
        output.setnchannels(1)
        output.setsampwidth(2)
        output.setframerate(sample_rate)
        output.writeframes(padded.tobytes())
    os.replace(temporary, destination)
    return {
        "signal_frames": len(samples),
        "clip_frames": target_frames,
        "left_padding_frames": pad_left,
        "right_padding_frames": target_frames - pad_left - len(samples),
    }


def cut_recording(
    *,
    protocol: dict[str, Any],
    archive: Path,
    member: str,
    candidates: list[Candidate],
    clip_root: Path,
) -> Iterator[dict[str, Any]]:
    """Stream one WAV out of the zip and write every admitted call on every channel.

    Windows are flushed as soon as the read position passes their end, so peak
    memory is set by the calls open at one moment rather than by the recording.
    """

    sample = protocol["sample"]
    clip_seconds = float(sample["clip_seconds"])
    # The channel list is not fixed across the archive: 55 recordings carry four
    # channels, 25 carry six and one carries two. Every channel present is cut and
    # the count is recorded on each clip, so an analysis can see what it has.
    wanted_channels: list[int] = []

    digest = hashlib.sha256()
    crc = 0
    bytes_read = 0

    def update(block: bytes) -> None:
        nonlocal crc, bytes_read
        digest.update(block)
        crc = zlib.crc32(block, crc)
        bytes_read += len(block)

    with zipfile.ZipFile(archive) as bundle:
        info = bundle.getinfo(member)
        sample_rate = 0
        channels = 0
        with bundle.open(info) as stream:
            header = _read_exact(stream, 12, update)
            if header[:4] != b"RIFF" or header[8:] != b"WAVE":
                raise ValueError(f"unsupported container: {member}")
            while bytes_read < int(info.file_size):
                chunk_header = _read_exact(stream, 8, update)
                chunk_id, chunk_size = struct.unpack("<4sI", chunk_header)
                if chunk_id == b"fmt ":
                    payload = _read_exact(stream, chunk_size, update)
                    audio_format, channels, sample_rate, _, block_align, bits = struct.unpack(
                        "<HHIIHH", payload[:16]
                    )
                    if (
                        not _is_pcm16(payload, audio_format, bits)
                        or sample_rate != 48_000
                        or channels < 2
                        or block_align != channels * 2
                    ):
                        raise ValueError(
                            f"unsupported format in {member}: "
                            f"{channels}ch {sample_rate}Hz {bits}bit fmt={audio_format}"
                        )
                elif chunk_id == b"data":
                    if not channels or not sample_rate:
                        raise ValueError(f"data precedes fmt: {member}")
                    wanted_channels = list(range(channels))
                    frame_bytes = channels * 2
                    source_frames = chunk_size // frame_bytes
                    target_frames = round(clip_seconds * sample_rate)
                    windows = {
                        candidate.row: (
                            round(candidate.signal_start * sample_rate),
                            round(candidate.signal_end * sample_rate),
                            candidate,
                        )
                        for candidate in candidates
                    }
                    for start, end, candidate in windows.values():
                        if start < 0 or end <= start or end > source_frames:
                            raise ValueError(f"window outside source: {member} row {candidate.row}")
                    buffers: dict[int, dict[int, array[int]]] = {
                        row: {index: array("h") for index in wanted_channels} for row in windows
                    }
                    open_rows = set(windows)
                    frame_position = 0
                    remaining = chunk_size
                    read_size = (4 * 1024 * 1024 // frame_bytes) * frame_bytes
                    while remaining:
                        block = _read_exact(stream, min(remaining, read_size), update)
                        remaining -= len(block)
                        interleaved = array("h")
                        interleaved.frombytes(block)
                        if sys.byteorder != "little":  # pragma: no cover
                            interleaved.byteswap()
                        block_frames = len(interleaved) // channels
                        block_end = frame_position + block_frames
                        finished = []
                        for row in list(open_rows):
                            start, end, candidate = windows[row]
                            if start >= block_end:
                                continue
                            overlap_start = max(start, frame_position)
                            overlap_end = min(end, block_end)
                            if overlap_end > overlap_start:
                                for index in wanted_channels:
                                    first = (overlap_start - frame_position) * channels + index
                                    last = (overlap_end - frame_position) * channels + index
                                    buffers[row][index].extend(interleaved[first:last:channels])
                            if end <= block_end:
                                finished.append(row)
                        for row in finished:
                            open_rows.discard(row)
                            start, end, candidate = windows[row]
                            for index in wanted_channels:
                                destination = (
                                    clip_root
                                    / f"ch{index}"
                                    / candidate.recording
                                    / f"{candidate.recording}_r{row}_ch{index}.wav"
                                )
                                padding = _write_clip(
                                    destination=destination,
                                    samples=buffers[row][index],
                                    sample_rate=sample_rate,
                                    target_frames=target_frames,
                                )
                                yield {
                                    "recording": candidate.recording,
                                    "day": candidate.day,
                                    "year": candidate.year,
                                    "annotation_row": row,
                                    "identity": candidate.identity,
                                    "event": candidate.event,
                                    "channel": index,
                                    "source_channels": channels,
                                    "local_path": str(destination.relative_to(clip_root)),
                                    "annotation_start_seconds": candidate.start,
                                    "annotation_end_seconds": candidate.end,
                                    "signal_start_seconds": candidate.signal_start,
                                    "signal_end_seconds": candidate.signal_end,
                                    "sample_rate_hz": sample_rate,
                                    **padding,
                                }
                            buffers.pop(row)
                        frame_position = block_end
                    if open_rows:
                        raise ValueError(
                            f"windows never completed in {member}: {sorted(open_rows)}"
                        )
                else:
                    remaining = chunk_size
                    while remaining:
                        block = _read_exact(stream, min(remaining, 1024 * 1024), update)
                        remaining -= len(block)
                if chunk_size % 2:
                    _read_exact(stream, 1, update)
    if bytes_read != int(info.file_size) or (crc & 0xFFFFFFFF) != int(info.CRC):
        raise ValueError(f"integrity mismatch streaming {member}")


# --- Endpoints cut from the built manifest -------------------------------------

#: The splits E02 registers. Each one names what it holds constant and what it
#: varies, because that is what decides what a number on it can mean.
E02_SPLITS = {
    "across-year": "enrol on 2020, score on 2021: the recording, the day and the year all change",
    "across-day-2020": "enrol on odd-indexed 2020 days, score on even-indexed 2020 days: "
    "the recording and the day change, the year does not",
    "across-day-2021": "the same within 2021",
}

MINIMUM_CALLS_PER_SIDE = 20


def _split_of(entry: dict[str, Any], split: str, enrolment_days: set[str]) -> str | None:
    if split == "across-year":
        return {"2020": "enrollment", "2021": "query"}.get(str(entry["year"]))
    year = split.rsplit("-", 1)[-1]
    if str(entry["year"]) != year:
        return None
    return "enrollment" if str(entry["day"]) in enrolment_days else "query"


def e02_records(
    *,
    manifest: dict[str, Any],
    split: str,
    channel: int = 0,
    minimum_calls_per_side: int = MINIMUM_CALLS_PER_SIDE,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Select the clips for one E02 split, and say what was dropped and why."""

    if split not in E02_SPLITS:
        raise ValueError(f"unknown E02 split {split!r}: {sorted(E02_SPLITS)}")
    entries = [e for e in manifest["entries"] if int(e["channel"]) == channel]
    if not entries:
        raise ValueError(f"no clips on channel {channel}")

    enrolment_days: set[str] = set()
    if split.startswith("across-day"):
        year = split.rsplit("-", 1)[-1]
        days = sorted({str(e["day"]) for e in entries if str(e["year"]) == year})
        enrolment_days = {day for index, day in enumerate(days) if index % 2 == 0}

    sided = []
    for entry in entries:
        side = _split_of(entry, split, enrolment_days)
        if side is not None:
            sided.append((side, entry))

    counts: dict[str, dict[str, int]] = {}
    for side, entry in sided:
        counts.setdefault(str(entry["identity"]), {"enrollment": 0, "query": 0})[side] += 1
    kept = {
        identity
        for identity, tally in counts.items()
        if min(tally.values()) >= minimum_calls_per_side
    }
    records = [dict(entry, split=side) for side, entry in sided if str(entry["identity"]) in kept]
    report = {
        "split": split,
        "split_means": E02_SPLITS[split],
        "channel": channel,
        "minimum_calls_per_side": minimum_calls_per_side,
        "identities_kept": sorted(kept),
        "identities_dropped": {
            identity: tally for identity, tally in sorted(counts.items()) if identity not in kept
        },
        "enrollment_days": sorted(enrolment_days) or None,
        "clips": len(records),
        "enrollment_clips": sum(1 for r in records if r["split"] == "enrollment"),
        "query_clips": sum(1 for r in records if r["split"] == "query"),
    }
    return records, report


def load_e02_rookid(
    *, manifest_path: Path, sample_root: Path, split: str = "across-year", channel: int = 0
) -> tuple[Any, dict[str, Any]]:
    """Build one E02 endpoint from the manifest the extractor wrote."""

    from xinyenyana.a2 import ClipRecord, Endpoint
    from xinyenyana.archive import canonical_sha256

    manifest = json.loads(manifest_path.read_bytes())
    records, report = e02_records(manifest=manifest, split=split, channel=channel)
    clips = sample_root / "clips"
    endpoint = Endpoint(
        name=f"rookid-full-{split}",
        records=tuple(
            ClipRecord(
                filename=str(entry["local_path"]),
                path=clips / str(entry["local_path"]),
                identity=str(entry["identity"]),
                split=str(entry["split"]),
                context={
                    "condition": "foreground",
                    "species": "rook",
                    "recording": str(entry["recording"]),
                    "recorded": str(entry["day"]),
                    "year": str(entry["year"]),
                    "channel": str(entry["channel"]),
                    "source_channels": str(entry["source_channels"]),
                    "event": str(entry["event"]),
                },
            )
            for entry in records
        ),
        categorical_targets=("year", "event", "channel"),
        manifest_sha256=canonical_sha256(
            sorted((str(r["local_path"]), str(r["identity"]), str(r["split"])) for r in records)
        ),
        source_document="configs/experiments/e02-rookid-full-width.json",
    )
    return endpoint, report
