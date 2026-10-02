"""Exact-context RookID acquisition and preregistered E01d gate."""

from __future__ import annotations

import hashlib
import json
import math
import os
import struct
import sys
import wave
import zlib
from array import array
from pathlib import Path
from typing import Any

DEFAULT_PROTOCOL = Path("configs/experiments/e01d-rookid-context-matched.json")
DEFAULT_SAMPLE_ROOT = Path("data/raw/benchmarks")
SAMPLE_DIRECTORY = "context-matched-mono"


def _canonical_sha256(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(value, separators=(",", ":"), sort_keys=True).encode()
    ).hexdigest()


def load_e01d_protocol(path: Path) -> tuple[dict[str, Any], str]:
    """Load and validate the feature-blind exact-context registration."""

    raw = path.read_bytes()
    protocol = json.loads(raw)
    if (
        protocol.get("schema_version") != 1
        or protocol.get("experiment") != "E01d"
        or protocol.get("registration_status") != "prospective_context_control"
    ):
        raise ValueError("unsupported E01d protocol")
    if protocol.get("head", {}).get("name") != "dual_ridge":
        raise ValueError("E01d requires the registered dual-ridge head")
    sample = protocol.get("sample", {})
    identities = sample.get("identities")
    if (
        not isinstance(identities, list)
        or len(identities) < 2
        or len(identities) != len(set(identities))
    ):
        raise ValueError("E01d identities must be a unique list")
    if not str(sample.get("required_event", "")):
        raise ValueError("E01d requires one exact source Event value")
    recordings = protocol.get("source", {}).get("recordings")
    if not isinstance(recordings, list) or len(recordings) != 2:
        raise ValueError("E01d requires exactly one recording per split")
    if {str(recording.get("split")) for recording in recordings} != {
        "enrollment",
        "query",
    }:
        raise ValueError("E01d requires enrollment and query recordings")
    stems = [str(recording.get("stem", "")) for recording in recordings]
    if len(set(stems)) != 2 or any(not stem for stem in stems):
        raise ValueError("E01d source recording stems must be unique")
    if stems[0][:8] == stems[1][:8]:
        raise ValueError("E01d enrollment and query dates must differ")
    if any(int(recording.get("calls_per_identity", 0)) < 1 for recording in recordings):
        raise ValueError("E01d calls per identity must be positive")
    representations = protocol.get("representations", {})
    nuisance = [str(value.get("name")) for value in representations.get("nuisance", [])]
    if nuisance != ["duration_only", "level_only"]:
        raise ValueError("E01d nuisance representations changed")
    inference = protocol.get("inference", {})
    chance = float(inference.get("chance_accuracy", -1))
    if not math.isclose(chance, 1 / len(identities)):
        raise ValueError("E01d registered chance does not match identity count")
    if (
        int(inference.get("permutations", 0)) < 1
        or int(inference.get("bootstrap_replicates", 0)) < 1
    ):
        raise ValueError("E01d inference counts must be positive")
    return protocol, hashlib.sha256(raw).hexdigest()


def select_e01d_events(
    *, protocol: dict[str, Any], annotations: dict[str, list[Any]]
) -> list[dict[str, Any]]:
    """Select exact-event, isolated calls at the registered per-recording quotas."""

    sample = protocol["sample"]
    margin = float(sample["annotation_margin_seconds"])
    maximum = float(sample["maximum_signal_seconds"])
    separation = float(sample["minimum_same_recording_separation_seconds"])
    seed = int(sample["selection_seed"])
    identities = [str(identity) for identity in sample["identities"]]
    required_event = str(sample["required_event"])
    ambiguity = [str(value).casefold() for value in sample["explicit_ambiguity_substrings"]]
    selected: list[dict[str, Any]] = []

    for recording in protocol["source"]["recordings"]:
        stem = str(recording["stem"])
        split = str(recording["split"])
        requested = int(recording["calls_per_identity"])
        events = annotations.get(stem)
        if events is None:
            raise ValueError(f"E01d annotations missing registered source: {stem}")
        for identity in identities:
            candidates: list[dict[str, Any]] = []
            for event in events:
                if event.identity != identity or event.event != required_event:
                    continue
                source_notes = f"{event.event} {event.comment}".casefold()
                if any(marker in source_notes for marker in ambiguity):
                    continue
                signal_start = max(0.0, event.start - margin)
                signal_end = event.end + margin
                if signal_end - signal_start > maximum:
                    continue
                if any(
                    other.identity != identity
                    and other.end > signal_start
                    and other.start < signal_end
                    for other in events
                ):
                    continue
                rank = hashlib.sha256(f"{seed}:{stem}:{event.row_number}".encode()).hexdigest()
                candidates.append(
                    {
                        "rank": rank,
                        "recording": stem,
                        "split": split,
                        "identity": identity,
                        "annotation_row": event.row_number,
                        "event": event.event,
                        "comment": event.comment,
                        "annotation_start_seconds": event.start,
                        "annotation_end_seconds": event.end,
                        "signal_start_seconds": signal_start,
                        "signal_end_seconds": signal_end,
                    }
                )
            chosen: list[dict[str, Any]] = []
            for candidate in sorted(candidates, key=lambda value: str(value["rank"])):
                midpoint = (
                    float(candidate["signal_start_seconds"])
                    + float(candidate["signal_end_seconds"])
                ) / 2
                if any(
                    abs(
                        (float(other["signal_start_seconds"]) + float(other["signal_end_seconds"]))
                        / 2
                        - midpoint
                    )
                    < separation
                    for other in chosen
                ):
                    continue
                chosen.append(candidate)
                if len(chosen) == requested:
                    break
            if len(chosen) != requested:
                raise ValueError(
                    f"not enough E01d events for {(stem, identity)}: "
                    f"need {requested}, found {len(chosen)}"
                )
            selected.extend(chosen)
    return sorted(
        selected,
        key=lambda value: (
            str(value["split"]),
            str(value["identity"]),
            str(value["recording"]),
            int(value["annotation_row"]),
        ),
    )


def _read_exact(stream: Any, size: int, update: Any) -> bytes:
    value = bytearray()
    while len(value) < size:
        block = stream.read(size - len(value))
        if not block:
            raise ValueError(f"unexpected end of streamed RookID WAV after {len(value)} bytes")
        update(block)
        value.extend(block)
    return bytes(value)


def _write_streamed_clip(
    *,
    destination: Path,
    samples: array[int],
    sample_rate: int,
    target_frames: int,
) -> dict[str, int]:
    if len(samples) > target_frames:
        raise ValueError("streamed RookID signal exceeds the registered clip duration")
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
    if destination.exists():
        if destination.read_bytes() != temporary.read_bytes():
            temporary.unlink()
            raise ValueError(f"refusing to overwrite a different RookID clip: {destination}")
        temporary.unlink()
    else:
        os.replace(temporary, destination)
    return {
        "signal_frames": len(samples),
        "clip_frames": target_frames,
        "left_padding_frames": pad_left,
        "right_padding_frames": target_frames - pad_left - len(samples),
    }


def _is_pcm16_format(payload: bytes, audio_format: int, bits: int) -> bool:
    if bits != 16:
        return False
    if audio_format == 1:
        return True
    pcm_subformat = bytes.fromhex("0100000000001000800000aa00389b71")
    return (
        audio_format == 0xFFFE
        and len(payload) >= 40
        and struct.unpack("<H", payload[16:18])[0] >= 22
        and struct.unpack("<H", payload[18:20])[0] == 16
        and payload[24:40] == pcm_subformat
    )


def _stream_registered_wav(
    *,
    remote: Any,
    info: Any,
    recording: dict[str, Any],
    plans: list[dict[str, Any]],
    protocol: dict[str, Any],
) -> tuple[dict[str, Any], dict[str, dict[str, int]]]:
    """Verify a complete remote WAV while retaining only registered signal windows."""

    digest = hashlib.sha256()
    crc = 0
    bytes_read = 0

    def update(block: bytes) -> None:
        nonlocal crc, bytes_read
        digest.update(block)
        crc = zlib.crc32(block, crc)
        bytes_read += len(block)

    buffers = {str(plan["rank"]): array("h") for plan in plans}
    sample_rate = 0
    channels = 0
    source_frames = 0
    with remote.open(info) as stream:
        header = _read_exact(stream, 12, update)
        if header[:4] != b"RIFF" or header[8:] != b"WAVE":
            raise ValueError(f"unsupported streamed RookID WAV container: {info.filename}")
        while bytes_read < int(info.file_size):
            chunk_header = _read_exact(stream, 8, update)
            chunk_id, chunk_size = struct.unpack("<4sI", chunk_header)
            if chunk_id == b"fmt ":
                payload = _read_exact(stream, chunk_size, update)
                if len(payload) < 16:
                    raise ValueError(f"truncated RookID WAV fmt chunk: {info.filename}")
                audio_format, channels, sample_rate, _, block_align, bits = struct.unpack(
                    "<HHIIHH", payload[:16]
                )
                if (
                    not _is_pcm16_format(payload, audio_format, bits)
                    or sample_rate != 48_000
                    or channels <= int(protocol["sample"]["channel_index"])
                    or block_align != channels * 2
                ):
                    raise ValueError(f"unsupported registered RookID WAV format: {info.filename}")
            elif chunk_id == b"data":
                if not channels or not sample_rate:
                    raise ValueError(f"RookID WAV data precedes fmt chunk: {info.filename}")
                frame_bytes = channels * 2
                if chunk_size % frame_bytes:
                    raise ValueError(f"unaligned RookID WAV data chunk: {info.filename}")
                source_frames = chunk_size // frame_bytes
                intervals = {
                    str(plan["rank"]): (
                        round(float(plan["signal_start_seconds"]) * sample_rate),
                        round(float(plan["signal_end_seconds"]) * sample_rate),
                    )
                    for plan in plans
                }
                if any(
                    start < 0 or end <= start or end > source_frames
                    for start, end in intervals.values()
                ):
                    raise ValueError(f"registered E01d event lies outside source: {info.filename}")
                frame_position = 0
                remaining = chunk_size
                read_size = (1024 * 1024 // frame_bytes) * frame_bytes
                while remaining:
                    block = _read_exact(stream, min(remaining, read_size), update)
                    remaining -= len(block)
                    interleaved = array("h")
                    interleaved.frombytes(block)
                    if sys.byteorder != "little":  # pragma: no cover
                        interleaved.byteswap()
                    block_frames = len(interleaved) // channels
                    block_end = frame_position + block_frames
                    for token, (start, end) in intervals.items():
                        overlap_start = max(start, frame_position)
                        overlap_end = min(end, block_end)
                        if overlap_end <= overlap_start:
                            continue
                        first = (overlap_start - frame_position) * channels + int(
                            protocol["sample"]["channel_index"]
                        )
                        last = (overlap_end - frame_position) * channels + int(
                            protocol["sample"]["channel_index"]
                        )
                        buffers[token].extend(interleaved[first:last:channels])
                    frame_position = block_end
            else:
                remaining = chunk_size
                while remaining:
                    block = _read_exact(stream, min(remaining, 1024 * 1024), update)
                    remaining -= len(block)
            if chunk_size % 2:
                _read_exact(stream, 1, update)
    if bytes_read != int(info.file_size) or (crc & 0xFFFFFFFF) != int(info.CRC):
        raise ValueError(f"streamed RookID WAV integrity mismatch: {info.filename}")

    audio: dict[str, dict[str, int]] = {}
    target_frames = round(float(protocol["sample"]["clip_seconds"]) * sample_rate)
    for plan in plans:
        token = str(plan["rank"])
        expected_frames = round(float(plan["signal_end_seconds"]) * sample_rate) - round(
            float(plan["signal_start_seconds"]) * sample_rate
        )
        if len(buffers[token]) != expected_frames:
            raise ValueError(f"streamed RookID window is incomplete: {plan['destination']}")
        padding = _write_streamed_clip(
            destination=plan["destination"],
            samples=buffers[token],
            sample_rate=sample_rate,
            target_frames=target_frames,
        )
        audio[token] = {
            "source_channels": channels,
            "source_sample_rate_hz": sample_rate,
            "source_frames": source_frames,
            **padding,
        }
    return (
        {
            "source_member": info.filename,
            "bytes": info.file_size,
            "compressed_bytes": info.compress_size,
            "crc32": f"{info.CRC:08x}",
            "sha256": digest.hexdigest(),
            "recording": str(recording["stem"]),
            "kind": "wav",
            "retained_locally": False,
        },
        audio,
    )
