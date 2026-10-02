"""Tests for the RookID channel audit and the channel-selection rule."""

from __future__ import annotations

import io
import struct
import zipfile
from pathlib import Path

import numpy as np
import pytest

from xinyenyana.endpoints import ROOKID_ARCHIVE
from xinyenyana.rookid_channels import (
    ChannelStatistics,
    audit_recording_channels,
    choose_channel,
    wav_header,
)

RATE = 48_000


def _stats(channel: int, low: float, *, recording: str = "r1") -> ChannelStatistics:
    return ChannelStatistics(
        recording=recording,
        channel=channel,
        windows=15,
        median_rms=1000.0,
        median_dc_offset=0.0,
        median_low_frequency_share=low,
        median_ninety_ninth_percentile_hz=5000.0,
    )


def test_the_rule_picks_the_channel_with_least_low_frequency_energy() -> None:
    """The measured RookID shares: channel 0 worst at 0.934, channel 2 best."""

    measured = [_stats(0, 0.934), _stats(1, 0.757), _stats(2, 0.047), _stats(3, 0.163)]
    assert choose_channel(measured) == 2


def test_the_rule_does_not_pick_on_loudness() -> None:
    """Channel 3 was the loudest of the four and is not the choice."""

    loud = ChannelStatistics("r1", 3, 15, 9015.7, 792.0, 0.163, 6328.0)
    quiet_but_clean = ChannelStatistics("r1", 2, 15, 3964.7, 432.0, 0.047, 7547.0)
    assert choose_channel([loud, quiet_but_clean]) == 2


def test_a_tie_breaks_on_the_lower_index() -> None:
    assert choose_channel([_stats(2, 0.10), _stats(1, 0.10), _stats(3, 0.50)]) == 1


def test_no_channels_is_refused() -> None:
    with pytest.raises(ValueError, match="no channel statistics"):
        choose_channel([])


def test_channels_from_two_recordings_are_refused() -> None:
    """Only the enrollment recording votes; mixing the two would let the query
    recording into the choice."""

    with pytest.raises(ValueError, match="one recording's channels"):
        choose_channel([_stats(0, 0.9), _stats(2, 0.1, recording="r2")])


# --- the header reader ------------------------------------------------------


def _extensible_header(channels: int, rate: int, bits: int, data_bytes: int) -> bytes:
    fmt_body = struct.pack(
        "<HHIIHH", 0xFFFE, channels, rate, rate * channels * bits // 8, channels * bits // 8, bits
    )
    fmt_body += struct.pack("<H", 22) + b"\x00" * 22
    fact = b"fact" + struct.pack("<I", 4) + struct.pack("<I", 0)
    chunks = b"fmt " + struct.pack("<I", len(fmt_body)) + fmt_body + fact
    body = b"WAVE" + chunks + b"data" + struct.pack("<I", data_bytes)
    return b"RIFF" + struct.pack("<I", 4 + len(body)) + body


def test_the_header_reader_walks_past_chunks_it_does_not_need() -> None:
    header = _extensible_header(4, RATE, 16, 1024)
    stream = io.BytesIO(header + b"\x00" * 1024)
    channels, rate, bits, offset = wav_header(stream)
    assert (channels, rate, bits) == (4, RATE, 16)
    assert offset == len(header)


def test_a_stream_that_is_not_a_wave_is_refused() -> None:
    with pytest.raises(ValueError, match="not a RIFF/WAVE"):
        wav_header(io.BytesIO(b"OggS" + b"\x00" * 100))


def test_a_stream_with_no_data_chunk_in_range_is_refused() -> None:
    body = b"WAVE" + b"junk" + struct.pack("<I", 8192) + b"\x00" * 8192
    with pytest.raises(ValueError, match="no data chunk"):
        wav_header(io.BytesIO(b"RIFF" + struct.pack("<I", len(body)) + body))


# --- the audit itself, reading a real archive -------------------------------


def _four_channel_archive(root: Path, *, stem: str, seconds: float = 2.0) -> None:
    """One recording whose four channels differ in the way the rule cares about.

    Channel 0 is a 12 Hz rumble, which is what the RookID recording chain put on
    the channel the earlier work used. Channels 1 to 3 carry a call-band tone at
    rising levels. Nothing here says which is loudest, because the rule is not
    allowed to choose on loudness.
    """

    import wave
    from array import array

    rate = 48_000
    frames = int(rate * seconds)
    time = np.arange(frames) / rate
    columns = [
        8_000 * np.sin(2 * np.pi * 12.0 * time),
        3_000 * np.sin(2 * np.pi * 2_000.0 * time),
        6_000 * np.sin(2 * np.pi * 2_500.0 * time),
        9_000 * np.sin(2 * np.pi * 3_000.0 * time),
    ]
    interleaved = np.stack(columns, axis=1).astype(np.int16).reshape(-1)
    payload = io.BytesIO()
    with wave.open(payload, "wb") as handle:
        handle.setnchannels(4)
        handle.setsampwidth(2)
        handle.setframerate(rate)
        handle.writeframes(array("h", interleaved.tolist()).tobytes())

    archive = root / ROOKID_ARCHIVE
    archive.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(archive, "w") as bundle:
        bundle.writestr(f"RookID/{stem}.wav", payload.getvalue())


def test_the_audit_measures_every_channel_of_a_real_archive(tmp_path: Path) -> None:
    _four_channel_archive(tmp_path, stem="20200214_090050")
    statistics = audit_recording_channels(
        stem="20200214_090050", windows=[(0.0, 1.0), (1.0, 2.0)], root=tmp_path
    )
    assert [entry.channel for entry in statistics] == [0, 1, 2, 3]
    assert all(entry.windows == 2 for entry in statistics)
    assert all(entry.recording == "20200214_090050" for entry in statistics)


def test_the_rumbling_channel_is_the_one_the_rule_rejects(tmp_path: Path) -> None:
    _four_channel_archive(tmp_path, stem="20200214_090050")
    statistics = audit_recording_channels(
        stem="20200214_090050", windows=[(0.0, 1.0)], root=tmp_path
    )
    shares = {entry.channel: entry.median_low_frequency_share for entry in statistics}
    assert shares[0] > 0.9
    assert all(shares[channel] < 0.05 for channel in (1, 2, 3))
    assert choose_channel(statistics) != 0


def test_the_rule_does_not_choose_the_loudest_channel(tmp_path: Path) -> None:
    """Channel 3 is the loudest here and channel 1 the quietest of the three."""

    _four_channel_archive(tmp_path, stem="20200214_090050")
    statistics = audit_recording_channels(
        stem="20200214_090050", windows=[(0.0, 1.0)], root=tmp_path
    )
    loudest = max(statistics, key=lambda entry: entry.median_rms).channel
    assert loudest == 3
    assert choose_channel(statistics) in (1, 2, 3)


def test_an_audit_of_a_recording_that_is_not_in_the_archive_is_refused(tmp_path: Path) -> None:
    _four_channel_archive(tmp_path, stem="20200214_090050")
    with pytest.raises(KeyError):
        audit_recording_channels(stem="absent", windows=[(0.0, 1.0)], root=tmp_path)
