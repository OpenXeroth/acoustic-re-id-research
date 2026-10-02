from __future__ import annotations

import copy
import io
import wave
import zlib
from array import array
from pathlib import Path

from xinyenyana.e01d import (
    _is_pcm16_format,
    _stream_registered_wav,
    load_e01d_protocol,
    select_e01d_events,
)
from xinyenyana.rookid import RookIDEvent

PROTOCOL_PATH = Path("tests/fixtures/e01d-synthetic.json")


def test_checked_in_e01d_protocol_freezes_exact_shared_context() -> None:
    protocol, digest = load_e01d_protocol(PROTOCOL_PATH)

    assert protocol["registration_status"] == "prospective_context_control"
    assert protocol["sample"]["required_event"] == "apb"
    assert protocol["feature_blinding"].startswith("Only archive metadata")
    assert {recording["split"] for recording in protocol["source"]["recordings"]} == {
        "enrollment",
        "query",
    }
    assert len(protocol["source"]["recordings"]) == 2
    assert len(digest) == 64


def test_e01d_selection_requires_exact_event_and_recording_quota() -> None:
    protocol, _ = load_e01d_protocol(PROTOCOL_PATH)
    protocol = copy.deepcopy(protocol)
    protocol["sample"].update(
        {
            "identities": ["a", "b"],
            "minimum_same_recording_separation_seconds": 1.0,
        }
    )
    protocol["source"]["recordings"] = [
        {"stem": "20200101_a", "split": "enrollment", "calls_per_identity": 1},
        {"stem": "20200102_b", "split": "query", "calls_per_identity": 1},
    ]
    annotations: dict[str, list[RookIDEvent]] = {}
    for split, stem in (("enrollment", "20200101_a"), ("query", "20200102_b")):
        annotations[stem] = [
            RookIDEvent(stem, split, 2, "a", "sing", "", 1.0, 1.2),
            RookIDEvent(stem, split, 3, "a", "apb", "", 3.0, 3.2),
            RookIDEvent(stem, split, 4, "b", "apb", "unclear", 5.0, 5.2),
            RookIDEvent(stem, split, 5, "b", "apb", "", 7.0, 7.2),
        ]

    selected = select_e01d_events(protocol=protocol, annotations=annotations)

    assert len(selected) == 4
    assert {value["event"] for value in selected} == {"apb"}
    assert {int(value["annotation_row"]) for value in selected} == {3, 5}


def test_streamed_wav_verifies_source_and_retains_only_registered_window(
    tmp_path: Path,
) -> None:
    source = io.BytesIO()
    interleaved = array("h")
    for frame in range(48_000):
        interleaved.extend((frame % 1000, -2000))
    with wave.open(source, "wb") as output:
        output.setnchannels(2)
        output.setsampwidth(2)
        output.setframerate(48_000)
        output.writeframes(interleaved.tobytes())
    payload = source.getvalue()

    class Info:
        filename = "RookID/test.wav"
        file_size = len(payload)
        compress_size = len(payload)
        CRC = zlib.crc32(payload) & 0xFFFFFFFF

    class Remote:
        def open(self, _info: object) -> io.BytesIO:
            return io.BytesIO(payload)

    protocol, _ = load_e01d_protocol(PROTOCOL_PATH)
    destination = tmp_path / "clip.wav"
    plan = {
        "rank": "abc",
        "signal_start_seconds": 0.25,
        "signal_end_seconds": 0.75,
        "destination": destination,
    }

    source_record, audio = _stream_registered_wav(
        remote=Remote(),
        info=Info(),
        recording={"stem": "test"},
        plans=[plan],
        protocol=protocol,
    )

    assert source_record["retained_locally"] is False
    assert source_record["sha256"]
    assert audio["abc"]["source_channels"] == 2
    assert audio["abc"]["signal_frames"] == 24_000
    with wave.open(str(destination), "rb") as clip:
        rendered = array("h")
        rendered.frombytes(clip.readframes(clip.getnframes()))
    assert set(rendered[:60_000]) == {0}
    assert rendered[60_000:60_010] == interleaved[24_000:24_020:2]
    assert set(rendered[84_000:]) == {0}


def test_wave_format_extensible_pcm16_header_is_supported() -> None:
    payload = bytes.fromhex(
        "feff040080bb000000dc05000800100016001000330000000100000000001000800000aa00389b71"
    )

    assert _is_pcm16_format(payload, 0xFFFE, 16)
    assert not _is_pcm16_format(payload, 0xFFFE, 24)
