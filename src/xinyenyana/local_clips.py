"""Cut endpoint clips from the local benchmark archives.

The archives are read from xen1's own disk. The windowing, channel selection,
padding and integrity checks live in ``e01d._stream_registered_wav`` and are
called from here unchanged, so one implementation cuts every clip, and a clip
cut today is the clip the recorded run cut or it fails and says so.
"""

from __future__ import annotations

import hashlib
import json
import os
import zipfile
import zlib
from datetime import UTC, datetime
from pathlib import Path, PurePosixPath
from typing import Any

from xinyenyana.archive import canonical_sha256
from xinyenyana.e01d import (
    SAMPLE_DIRECTORY,
    _stream_registered_wav,
    load_e01d_protocol,
    select_e01d_events,
)
from xinyenyana.endpoints import ROOKID_ARCHIVE, benchmark_root
from xinyenyana.rookid import ROOKID_RECORD, read_rookid_annotations


def clip_destination(root: Path, relative: str) -> Path:
    """Join a relative clip path to ``root``, refusing anything that escapes it.

    Same rule as ``rookid._safe_path``: the identity and split come from data,
    so a path built from them is checked rather than trusted.
    """

    candidate = PurePosixPath(relative)
    if candidate.is_absolute() or ".." in candidate.parts:
        raise ValueError(f"unsafe clip path: {relative}")
    destination = root.joinpath(*candidate.parts)
    if root.resolve() not in destination.resolve().parents:
        raise ValueError(f"clip path escapes output root: {relative}")
    return destination


def _clip_plans(
    *, selected: list[dict[str, Any]], stem: str, sample_root: Path
) -> list[dict[str, Any]]:
    """Build the per-event clip plan for one source recording.

    The destination name is the one the recorded run used, so a clip written
    here lands at the path that run wrote it to.
    """

    plans: list[dict[str, Any]] = []
    for event in selected:
        if event["recording"] != stem:
            continue
        token = str(event["rank"])[:16]
        filename = f"{event['identity']}-{token}.wav"
        relative = f"clips/{event['split']}/{event['identity']}/{filename}"
        plans.append({**event, "destination": clip_destination(sample_root, relative)})
    return plans


def build_e01d_clips(
    *,
    protocol: dict[str, Any],
    selected: list[dict[str, Any]],
    sample_root: Path,
    root: Path | None = None,
) -> tuple[list[dict[str, Any]], dict[str, dict[str, int]]]:
    """Write the E01d clips from the local RookID archive.

    Every registered member is checked against the size, compressed size and
    CRC-32 the protocol registered before a byte of it is decoded, and
    ``_stream_registered_wav`` verifies the whole member's CRC-32 as it reads,
    so a corrupt archive fails here rather than producing quiet nonsense.
    """

    base = root if root is not None else benchmark_root()
    archive = base / ROOKID_ARCHIVE
    sources: list[dict[str, Any]] = []
    audio: dict[str, dict[str, int]] = {}
    with zipfile.ZipFile(archive) as bundle:
        members = {info.filename: info for info in bundle.infolist() if not info.is_dir()}
        for recording in protocol["source"]["recordings"]:
            stem = str(recording["stem"])
            member = f"RookID/{stem}.wav"
            info = members.get(member)
            if info is None:
                raise ValueError(f"registered RookID member is absent: {member}")
            if (
                info.file_size != int(recording["wav_bytes"])
                or info.compress_size != int(recording["wav_compressed_bytes"])
                or f"{info.CRC:08x}" != str(recording["wav_crc32"])
            ):
                raise ValueError(f"registered RookID member changed: {member}")
            source, source_audio = _stream_registered_wav(
                remote=bundle,
                info=info,
                recording=recording,
                plans=_clip_plans(selected=selected, stem=stem, sample_root=sample_root),
                protocol=protocol,
            )
            sources.append(source)
            audio.update(source_audio)
    return sources, audio


def local_registered_members(
    *,
    protocol: dict[str, Any],
    source_root: Path,
    root: Path | None = None,
    extensions: tuple[str, ...] = ("tsv",),
) -> list[dict[str, Any]]:
    """Write registered members to ``source_root/sources`` and record them.

    The same record shape ``rookid._fetch_registered_members`` produced over
    HTTP, so the manifest built from it validates against the same protocol.
    """

    base = root if root is not None else benchmark_root()
    archive = base / ROOKID_ARCHIVE
    records: list[dict[str, Any]] = []
    with zipfile.ZipFile(archive) as bundle:
        members = {info.filename: info for info in bundle.infolist() if not info.is_dir()}
        for recording in protocol["source"]["recordings"]:
            stem = str(recording["stem"])
            for extension in extensions:
                member = f"RookID/{stem}.{extension}"
                info = members.get(member)
                if info is None:
                    raise ValueError(f"registered RookID member is absent: {member}")
                if info.file_size != int(recording[f"{extension}_bytes"]) or (
                    f"{info.CRC:08x}" != str(recording[f"{extension}_crc32"])
                ):
                    raise ValueError(f"registered RookID member changed: {member}")
                relative = f"sources/{stem}.{extension}"
                destination = clip_destination(source_root, relative)
                destination.parent.mkdir(parents=True, exist_ok=True)
                payload = bundle.read(member)
                if zlib.crc32(payload) & 0xFFFFFFFF != info.CRC:
                    raise ValueError(f"RookID member CRC mismatch: {member}")
                destination.write_bytes(payload)
                records.append(
                    {
                        "source_member": info.filename,
                        "local_path": relative,
                        "bytes": info.file_size,
                        "crc32": f"{info.CRC:08x}",
                        "sha256": hashlib.sha256(payload).hexdigest(),
                        "recording": stem,
                        "kind": extension,
                    }
                )
    return records


def build_e01d_sample(
    *,
    protocol_path: Path,
    output_root: Path,
    root: Path | None = None,
    channel_index: int | None = None,
) -> dict[str, Any]:
    """Render the fixed E01d sample and its manifest from the local archive.

    The same manifest ``e01d.fetch_e01d_rookid_sample`` wrote when it fetched
    over HTTP, so ``run_e01d_context_gate`` runs on it unchanged.

    ``channel_index`` overrides the channel the protocol registered. The E01d
    protocol registered channel 0 without measuring it; the audit in
    ``rookid_channels`` found channel 0 to be, in both recordings, the quietest
    of the four and the one whose energy is most nearly all below 50 Hz. The
    override exists so a corrected endpoint can be built without editing a
    protocol whose digest other recorded results are pinned to. The manifest
    records that the override was used and what it replaced.
    """

    protocol, protocol_sha = load_e01d_protocol(protocol_path)
    registered_channel = int(protocol["sample"]["channel_index"])
    if channel_index is not None and channel_index != registered_channel:
        protocol = {**protocol, "sample": {**protocol["sample"], "channel_index": channel_index}}
    sample_root = output_root / "rookid-zenodo" / ROOKID_RECORD / SAMPLE_DIRECTORY
    sources = local_registered_members(
        protocol=protocol, source_root=sample_root, root=root, extensions=("tsv",)
    )
    annotations = read_rookid_annotations(protocol=protocol, source_root=sample_root)
    selected = select_e01d_events(protocol=protocol, annotations=annotations)
    clip_sources, clip_audio = build_e01d_clips(
        protocol=protocol, selected=selected, sample_root=sample_root, root=root
    )
    sources.extend(clip_sources)

    clips: list[dict[str, Any]] = []
    for event in selected:
        token = str(event["rank"])[:16]
        filename = f"{event['identity']}-{token}.wav"
        relative = f"clips/{event['split']}/{event['identity']}/{filename}"
        destination = clip_destination(sample_root, relative)
        clips.append(
            {
                **{key: value for key, value in event.items() if key != "rank"},
                "filename": filename,
                "local_path": relative,
                "condition": "foreground",
                "bytes": destination.stat().st_size,
                "sha256": hashlib.sha256(destination.read_bytes()).hexdigest(),
                "channel_index": int(protocol["sample"]["channel_index"]),
                **clip_audio[str(event["rank"])],
            }
        )

    manifest: dict[str, Any] = {
        "schema_version": 1,
        "dataset": "rookid-context-matched-mono",
        "retrieved_at": datetime.now(UTC).isoformat(),
        "protocol_sha256": protocol_sha,
        "source": {
            "record": f"https://zenodo.org/records/{ROOKID_RECORD}",
            "doi": protocol["source"]["doi"],
            "license_spdx": protocol["source"]["license_spdx"],
            "archive": protocol["source"]["archive"],
            "range_fetched_members": sources,
        },
        "selection": {
            "selection_sha256": canonical_sha256(selected),
            "identity_evidence": protocol["source"]["identity_evidence"],
            "recording_context": protocol["source"]["recording_context"],
            "rules": protocol["sample"],
        },
        "sample": {
            "file_count": len(clips),
            "total_bytes": sum(int(clip["bytes"]) for clip in clips),
            "files": clips,
        },
        "integrity_limit": (
            "The full archive MD5 is registered but cannot be recomputed from selected ZIP "
            "members; member CRC and size plus local source/clip SHA-256 are retained"
        ),
        "read_from": "local archive on disk, not range-fetched over HTTP",
        "channel": {
            "registered_by_protocol": registered_channel,
            "used": int(protocol["sample"]["channel_index"]),
            "overridden": int(protocol["sample"]["channel_index"]) != registered_channel,
        },
    }
    manifest_path = sample_root / "xinyenyana-manifest.json"
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = manifest_path.with_suffix(".json.part")
    temporary.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, manifest_path)
    return manifest


def clip_inventory(sample_root: Path) -> dict[str, Any]:
    """Count and measure the clips under ``sample_root``.

    The E01d result records ``45 files; 12,961,980 bytes``, so a count and a
    byte total are directly comparable with the record.
    """

    paths = sorted(p for p in (sample_root / "clips").rglob("*.wav") if p.is_file())
    sizes = [p.stat().st_size for p in paths]
    return {
        "files": len(paths),
        "bytes": sum(sizes),
        "distinct_sizes": sorted(set(sizes)),
    }
