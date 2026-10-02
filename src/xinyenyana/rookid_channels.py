"""Which of the RookID recording's channels to take, and the evidence for it.

The RookID recordings are multichannel. The publishers deployed one to three
Song Meter 4 recorders, each carrying two microphones set to different gains,
and merged the result into a single file of two to six channels. The
annotations give a start and an end and no channel: they apply to the whole
file, so which channel a clip is cut from is a choice the experiment makes.

E01d made that choice by writing ``channel_index: 0`` in its protocol. Nothing
measured whether channel 0 carried the birds. This module measures it.

The rule, registered in
[`docs/measurement-protocol.md`](../../docs/measurement-protocol.md):
**take the channel with the smallest share of its energy below 50 Hz, measured
over the annotated call windows of the enrollment recording only, and use that
same index for the query recording.** Low-frequency share is a property of the
recording chain, not of any bird, so the choice cannot favour one identity over
another; and the query recording is excluded so that the choice is not fitted
to the clips it will be scored on.
"""

from __future__ import annotations

import struct
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from xinyenyana.endpoints import ROOKID_ARCHIVE, benchmark_root
from xinyenyana.features import AUDIT_WINDOW_SECONDS, LOW_FREQUENCY_HZ, energy_share_below_hz

__all__ = ["LOW_FREQUENCY_HZ"]


@dataclass(frozen=True)
class ChannelStatistics:
    """What one channel of one recording holds, over the annotated windows."""

    recording: str
    channel: int
    windows: int
    median_rms: float
    median_dc_offset: float
    median_low_frequency_share: float
    median_ninety_ninth_percentile_hz: float

    def as_dict(self) -> dict[str, Any]:
        return {
            "recording": self.recording,
            "channel": self.channel,
            "windows": self.windows,
            "median_rms": self.median_rms,
            "median_dc_offset": self.median_dc_offset,
            "median_low_frequency_share": self.median_low_frequency_share,
            "median_ninety_ninth_percentile_hz": self.median_ninety_ninth_percentile_hz,
        }


def wav_header(stream: Any) -> tuple[int, int, int, int]:
    """Read channels, sample rate, bits and the data offset from a WAV stream.

    ``wave`` cannot read these files: they are WAVE_FORMAT_EXTENSIBLE, and the
    stream inside a zip is not seekable backwards. The chunks are walked here
    instead.
    """

    head = stream.read(4096)
    if head[:4] != b"RIFF" or head[8:12] != b"WAVE":
        raise ValueError("not a RIFF/WAVE stream")
    position = 12
    channels = rate = bits = 0
    while position + 8 <= len(head):
        chunk = head[position : position + 4]
        size = struct.unpack("<I", head[position + 4 : position + 8])[0]
        if chunk == b"fmt ":
            _, channels, rate, _, _, bits = struct.unpack(
                "<HHIIHH", head[position + 8 : position + 24]
            )
        if chunk == b"data":
            return channels, rate, bits, position + 8
        position += 8 + size + (size % 2)
    raise ValueError("no data chunk in the first 4096 bytes")


def audit_recording_channels(
    *, stem: str, windows: list[tuple[float, float]], root: Path | None = None
) -> list[ChannelStatistics]:
    """Measure every channel of one recording over the given time windows.

    The member is streamed once, forwards, and the windows are read in order.
    Seeking backwards inside a zip means decompressing from the start again, so
    a per-window seek on a 600 MB member costs hours rather than seconds.
    """

    import numpy as np

    base = root if root is not None else benchmark_root()
    member = f"RookID/{stem}.wav"
    ordered = sorted(windows)
    collected: dict[int, list[dict[str, float]]] = {}
    with zipfile.ZipFile(base / ROOKID_ARCHIVE) as bundle:
        info = bundle.getinfo(member)
        with bundle.open(info) as stream:
            channels, rate, bits, offset = wav_header(stream)
            if bits != 16:
                raise ValueError(f"{member} is {bits}-bit; this audit reads 16-bit PCM")
            with bundle.open(info) as data:
                data.read(offset)
                frame_bytes = channels * 2
                position = 0
                for start_seconds, end_seconds in ordered:
                    start = int(start_seconds * rate)
                    minimum = int(round(rate * AUDIT_WINDOW_SECONDS))
                    length = max(minimum, int(end_seconds * rate) - start)
                    skip = start - position
                    while skip > 0:
                        block = data.read(min(skip * frame_bytes, 1 << 24))
                        if not block:
                            break
                        skip -= len(block) // frame_bytes
                    position = start
                    raw = data.read(length * frame_bytes)
                    position += len(raw) // frame_bytes
                    frame = np.frombuffer(raw, dtype="<i2").reshape(-1, channels).astype(np.float64)
                    for channel in range(channels):
                        column = frame[:, channel]
                        low, ninety_ninth = energy_share_below_hz(column, rate)
                        collected.setdefault(channel, []).append(
                            {
                                "rms": float(np.sqrt(np.mean(np.square(column)))),
                                "dc": float(column.mean()),
                                "low": low,
                                "f99": ninety_ninth,
                            }
                        )
    return [
        ChannelStatistics(
            recording=stem,
            channel=channel,
            windows=len(rows),
            median_rms=float(np.median([row["rms"] for row in rows])),
            median_dc_offset=float(np.median([row["dc"] for row in rows])),
            median_low_frequency_share=float(np.median([row["low"] for row in rows])),
            median_ninety_ninth_percentile_hz=float(np.median([row["f99"] for row in rows])),
        )
        for channel, rows in sorted(collected.items())
    ]


def choose_channel(statistics: list[ChannelStatistics]) -> int:
    """Return the channel with the smallest share of energy below 50 Hz.

    Ties break on the lower index, so the choice is a function of the numbers
    and not of the order they arrive in.
    """

    if not statistics:
        raise ValueError("no channel statistics, so no channel can be chosen")
    recordings = {entry.recording for entry in statistics}
    if len(recordings) != 1:
        raise ValueError(
            f"the rule reads one recording's channels; got {sorted(recordings)}. "
            "The enrollment recording decides, and its index is used for the query one."
        )
    return min(
        statistics, key=lambda entry: (entry.median_low_frequency_share, entry.channel)
    ).channel
