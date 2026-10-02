"""Tests for cutting endpoint clips from a local archive.

These build a synthetic multi-channel WAV inside a synthetic zip, so they run in
CI where no archive exists and they can assert the exact samples that come out.
"""

from __future__ import annotations

import struct
import wave
import zipfile
import zlib
from array import array
from pathlib import Path

import pytest

from xinyenyana.endpoints import ROOKID_ARCHIVE
from xinyenyana.local_clips import build_e01d_clips, clip_destination, clip_inventory

SAMPLE_RATE = 48_000
STEM = "20200214_090050"


def _wav_bytes(*, channels: int, frames: int) -> bytes:
    """A PCM16 WAV whose sample value encodes its frame and channel."""

    interleaved = array("h")
    for frame in range(frames):
        for channel in range(channels):
            interleaved.append((frame * 10 + channel) % 30_000)
    payload = interleaved.tobytes()
    header = b"RIFF" + struct.pack("<I", 36 + len(payload)) + b"WAVE"
    header += b"fmt " + struct.pack(
        "<IHHIIHH", 16, 1, channels, SAMPLE_RATE, SAMPLE_RATE * channels * 2, channels * 2, 16
    )
    header += b"data" + struct.pack("<I", len(payload))
    return header + payload


def _archive(root: Path, wav: bytes) -> zipfile.ZipInfo:
    archive = root / ROOKID_ARCHIVE
    archive.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(archive, "w", compression=zipfile.ZIP_DEFLATED) as bundle:
        bundle.writestr(f"RookID/{STEM}.wav", wav)
    with zipfile.ZipFile(archive) as bundle:
        return bundle.getinfo(f"RookID/{STEM}.wav")


def _protocol(info: zipfile.ZipInfo, wav: bytes, *, channel_index: int = 0) -> dict:
    return {
        "source": {
            "recordings": [
                {
                    "stem": STEM,
                    "split": "enrollment",
                    "wav_bytes": info.file_size,
                    "wav_compressed_bytes": info.compress_size,
                    "wav_crc32": f"{zlib.crc32(wav) & 0xFFFFFFFF:08x}",
                }
            ]
        },
        "sample": {"channel_index": channel_index, "clip_seconds": 3.0},
    }


def _event(rank: str, identity: str, start: float, end: float) -> dict:
    return {
        "rank": rank,
        "recording": STEM,
        "split": "enrollment",
        "identity": identity,
        "annotation_row": 2,
        "signal_start_seconds": start,
        "signal_end_seconds": end,
    }


def _read_clip(path: Path) -> tuple[int, int, array]:
    with wave.open(str(path), "rb") as handle:
        samples = array("h")
        samples.frombytes(handle.readframes(handle.getnframes()))
        return handle.getframerate(), handle.getnchannels(), samples


def test_clips_are_three_second_mono_and_carry_the_selected_channel(tmp_path: Path) -> None:
    frames = SAMPLE_RATE * 10
    wav = _wav_bytes(channels=2, frames=frames)
    info = _archive(tmp_path, wav)
    protocol = _protocol(info, wav, channel_index=1)
    selected = [_event("aaaa1111", "Balbo", 1.0, 1.5)]
    sample_root = tmp_path / "sample"

    sources, audio = build_e01d_clips(
        protocol=protocol, selected=selected, sample_root=sample_root, root=tmp_path
    )

    assert len(sources) == 1
    assert sources[0]["crc32"] == f"{zlib.crc32(wav) & 0xFFFFFFFF:08x}"
    clip = sample_root / "clips" / "enrollment" / "Balbo" / "Balbo-aaaa1111.wav"
    rate, channels, samples = _read_clip(clip)
    assert (rate, channels) == (SAMPLE_RATE, 1)
    assert len(samples) == SAMPLE_RATE * 3

    signal_frames = round(1.5 * SAMPLE_RATE) - round(1.0 * SAMPLE_RATE)
    pad_left = (SAMPLE_RATE * 3 - signal_frames) // 2
    assert audio["aaaa1111"]["signal_frames"] == signal_frames
    assert audio["aaaa1111"]["left_padding_frames"] == pad_left
    assert set(samples[:pad_left]) == {0}
    assert set(samples[pad_left + signal_frames :]) == {0}
    # channel 1 of frame f holds (f * 10 + 1) % 30000
    first_source_frame = round(1.0 * SAMPLE_RATE)
    assert samples[pad_left] == (first_source_frame * 10 + 1) % 30_000
    assert samples[pad_left + 1] == ((first_source_frame + 1) * 10 + 1) % 30_000


def test_channel_index_zero_selects_a_different_signal(tmp_path: Path) -> None:
    frames = SAMPLE_RATE * 10
    wav = _wav_bytes(channels=2, frames=frames)
    info = _archive(tmp_path, wav)
    selected = [_event("bbbb2222", "Bashir", 2.0, 2.25)]
    sample_root = tmp_path / "sample"
    build_e01d_clips(
        protocol=_protocol(info, wav, channel_index=0),
        selected=selected,
        sample_root=sample_root,
        root=tmp_path,
    )
    _, _, samples = _read_clip(
        sample_root / "clips" / "enrollment" / "Bashir" / "Bashir-bbbb2222.wav"
    )
    signal_frames = round(2.25 * SAMPLE_RATE) - round(2.0 * SAMPLE_RATE)
    pad_left = (SAMPLE_RATE * 3 - signal_frames) // 2
    first_source_frame = round(2.0 * SAMPLE_RATE)
    assert samples[pad_left] == (first_source_frame * 10 + 0) % 30_000


def test_clip_inventory_counts_files_and_bytes(tmp_path: Path) -> None:
    frames = SAMPLE_RATE * 10
    wav = _wav_bytes(channels=1, frames=frames)
    info = _archive(tmp_path, wav)
    selected = [
        _event("cccc3333", "Balbo", 1.0, 1.4),
        _event("dddd4444", "Bashir", 3.0, 3.4),
    ]
    sample_root = tmp_path / "sample"
    build_e01d_clips(
        protocol=_protocol(info, wav),
        selected=selected,
        sample_root=sample_root,
        root=tmp_path,
    )
    inventory = clip_inventory(sample_root)
    assert inventory["files"] == 2
    # 3 s of mono PCM16 at 48 kHz is 288,000 bytes plus a 44-byte header
    assert inventory["distinct_sizes"] == [288_044]
    assert inventory["bytes"] == 2 * 288_044


def test_a_member_that_differs_from_the_protocol_is_refused(tmp_path: Path) -> None:
    wav = _wav_bytes(channels=1, frames=SAMPLE_RATE * 5)
    info = _archive(tmp_path, wav)
    protocol = _protocol(info, wav)
    protocol["source"]["recordings"][0]["wav_crc32"] = "deadbeef"
    with pytest.raises(ValueError, match="registered RookID member changed"):
        build_e01d_clips(
            protocol=protocol,
            selected=[_event("eeee5555", "Balbo", 1.0, 1.4)],
            sample_root=tmp_path / "sample",
            root=tmp_path,
        )


def test_an_absent_member_is_refused(tmp_path: Path) -> None:
    wav = _wav_bytes(channels=1, frames=SAMPLE_RATE * 5)
    info = _archive(tmp_path, wav)
    protocol = _protocol(info, wav)
    protocol["source"]["recordings"][0]["stem"] = "19000101_000000"
    with pytest.raises(ValueError, match="registered RookID member is absent"):
        build_e01d_clips(
            protocol=protocol,
            selected=[],
            sample_root=tmp_path / "sample",
            root=tmp_path,
        )


def test_a_window_beyond_the_source_is_refused(tmp_path: Path) -> None:
    wav = _wav_bytes(channels=1, frames=SAMPLE_RATE * 5)
    info = _archive(tmp_path, wav)
    with pytest.raises(ValueError, match="lies outside source"):
        build_e01d_clips(
            protocol=_protocol(info, wav),
            selected=[_event("ffff6666", "Balbo", 4.9, 6.0)],
            sample_root=tmp_path / "sample",
            root=tmp_path,
        )


def test_a_signal_longer_than_the_clip_is_refused(tmp_path: Path) -> None:
    wav = _wav_bytes(channels=1, frames=SAMPLE_RATE * 10)
    info = _archive(tmp_path, wav)
    with pytest.raises(ValueError, match="exceeds the registered clip duration"):
        build_e01d_clips(
            protocol=_protocol(info, wav),
            selected=[_event("gggg7777", "Balbo", 1.0, 5.0)],
            sample_root=tmp_path / "sample",
            root=tmp_path,
        )


def test_clip_destination_refuses_paths_that_escape_the_root(tmp_path: Path) -> None:
    assert clip_destination(tmp_path, "clips/enrollment/Balbo/x.wav") == (
        tmp_path / "clips" / "enrollment" / "Balbo" / "x.wav"
    )
    with pytest.raises(ValueError, match="unsafe clip path"):
        clip_destination(tmp_path, "../escaped.wav")
    with pytest.raises(ValueError, match="unsafe clip path"):
        clip_destination(tmp_path, "/etc/passwd")
