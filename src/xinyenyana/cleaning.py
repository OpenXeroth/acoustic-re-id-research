"""A3: the three cleaning steps, and the rule that sets the band.

The protocol is [`docs/a3-call-cleaning.md`](../../docs/a3-call-cleaning.md),
registered before any cleaned score was read.

Three operations, each one a function of a clip and nothing else, except the
band, which is a property of an endpoint and is computed from its enrollment
clips only. Query clips never enter the band. That matters: a band fitted to the
clips it is then scored on is a threshold measured on its own sample, and would
make any improvement it produced unreadable.

None of these has a parameter chosen by looking at an identity score. The three
free numbers, the 1% energy edges of the band, the over-subtraction factor and
the spectral floor, are fixed in this module and recorded in every run.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from xinyenyana.features import (
    DEFAULT_ENVELOPE_FRAME,
    DEFAULT_ENVELOPE_HOP,
    DEFAULT_FRAME,
    DEFAULT_HOP,
    note_boundaries,
    read_clip,
)

BAND_EDGE_ENERGY = 0.01
BAND_ROUNDING_HZ = 50.0
BAND_FLOOR_HZ = 50.0

OVER_SUBTRACTION = 1.0
SPECTRAL_FLOOR = 0.05
MINIMUM_NOISE_FRAMES = 4

STEPS = ("trim", "band", "denoise")


@dataclass(frozen=True)
class Band:
    """One endpoint's call band, and how much of the pooled energy it holds."""

    low_hz: float
    high_hz: float
    clips: int

    def as_dict(self) -> dict[str, float]:
        return {"low_hz": self.low_hz, "high_hz": self.high_hz, "clips": float(self.clips)}


def _normalised_power_spectrum(signal: Any, sample_rate: int, *, frame: int, hop: int) -> Any:
    """One clip's power spectrum, summed over frames and scaled to sum to one.

    Scaling to one is what stops a loud clip deciding the band for the endpoint.
    """

    import numpy as np

    from xinyenyana.features import _frames

    samples = np.asarray(signal, dtype=np.float64)
    windows = _frames(samples, frame, hop)
    if len(windows) == 0:
        return None
    power = np.abs(np.fft.rfft(windows * np.hamming(frame), axis=1)) ** 2
    total = power.sum(axis=0)
    scale = float(total.sum())
    return total / scale if scale > 0 else None


def band_from_clips(
    paths: Iterable[Path], *, frame: int = DEFAULT_FRAME, hop: int = DEFAULT_HOP
) -> Band:
    """Compute an endpoint's call band from the clips handed in.

    The band runs from the frequency below which 1% of the pooled energy lies to
    the one above which 1% lies, rounded outward to 50 Hz. Only enrollment clips
    are ever passed here.
    """

    import numpy as np

    spectra: list[Any] = []
    sample_rate = 0
    for path in paths:
        signal, rate = read_clip(path)
        if sample_rate and rate != sample_rate:
            raise ValueError(f"mixed sample rates in one endpoint: {sample_rate} and {rate}")
        sample_rate = rate
        spectrum = _normalised_power_spectrum(signal, rate, frame=frame, hop=hop)
        if spectrum is not None:
            spectra.append(spectrum)
    if not spectra:
        raise ValueError("no clip produced a spectrum, so no band can be computed")
    pooled = np.mean(np.vstack(spectra), axis=0)
    frequencies = np.fft.rfftfreq(frame, 1.0 / sample_rate)
    cumulative = np.cumsum(pooled) / pooled.sum()
    low = float(frequencies[int(np.searchsorted(cumulative, BAND_EDGE_ENERGY))])
    high = float(frequencies[int(np.searchsorted(cumulative, 1.0 - BAND_EDGE_ENERGY))])
    low = max(BAND_FLOOR_HZ, np.floor(low / BAND_ROUNDING_HZ) * BAND_ROUNDING_HZ)
    high = min(sample_rate / 2.0, np.ceil(high / BAND_ROUNDING_HZ) * BAND_ROUNDING_HZ)
    if high <= low:
        raise ValueError(f"band collapsed: {low} Hz to {high} Hz")
    return Band(low_hz=float(low), high_hz=float(high), clips=len(spectra))


def band_limit(signal: Any, sample_rate: int, band: Band) -> Any:
    """Zero every frequency outside the band, over the whole clip.

    A brick wall in the frequency domain rather than a designed filter, because
    a designed filter carries an order and a ripple that would be two more free
    numbers to justify. The cost is ringing at the edges, which is why the
    result of band-limiting is measured rather than assumed to help.
    """

    import numpy as np

    samples = np.asarray(signal, dtype=np.float64)
    spectrum = np.fft.rfft(samples)
    frequencies = np.fft.rfftfreq(len(samples), 1.0 / sample_rate)
    spectrum[(frequencies < band.low_hz) | (frequencies > band.high_hz)] = 0.0
    return np.fft.irfft(spectrum, n=len(samples))


def trim_to_notes(
    signal: Any,
    sample_rate: int,
    *,
    frame: int = DEFAULT_ENVELOPE_FRAME,
    hop: int = DEFAULT_ENVELOPE_HOP,
) -> tuple[Any, int]:
    """Keep the samples inside detected notes and join them, dropping the gaps.

    Returns the trimmed signal and the number of notes. A clip with no detected
    note is returned unchanged with a count of zero, because an empty clip
    cannot be evaluated and silently substituting one would hide the failure.
    """

    import numpy as np

    samples = np.asarray(signal, dtype=np.float64)
    notes = note_boundaries(samples, sample_rate, frame=frame, hop=hop)
    if not notes:
        return samples, 0
    pieces = []
    for start, end in notes:
        first = max(0, int(round(start * sample_rate)))
        last = min(len(samples), int(round(end * sample_rate)))
        if last > first:
            pieces.append(samples[first:last])
    if not pieces:
        return samples, 0
    return np.concatenate(pieces), len(pieces)


def spectral_subtract(
    signal: Any,
    sample_rate: int,
    *,
    frame: int = DEFAULT_FRAME,
    hop: int = DEFAULT_HOP,
    over_subtraction: float = OVER_SUBTRACTION,
    spectral_floor: float = SPECTRAL_FLOOR,
) -> tuple[Any, int]:
    """Subtract a noise spectrum estimated from this clip's own non-note frames.

    The noise estimate is the median magnitude of the frames whose centre falls
    outside every detected note, so it comes from the same recording, the same
    microphone and the same minute as the call.

    Two cases return the clip unchanged and say so. A clip in which no note was
    detected has no frames that are known not to be call, and taking the
    estimate from every frame would subtract the call from itself. A clip with
    fewer than four non-note frames, about 46 ms at the default framing, has too
    few for a median to mean anything. Both are counted in a run rather than
    quietly given a bad estimate.

    Returns the signal and how many frames the estimate used, zero when none was
    made.
    """

    import numpy as np

    from xinyenyana.features import _frames

    samples = np.asarray(signal, dtype=np.float64)
    windows = _frames(samples, frame, hop)
    if len(windows) == 0:
        return samples, 0
    notes = note_boundaries(samples, sample_rate)
    if not notes:
        return samples, 0
    centres = (np.arange(len(windows)) * hop + frame / 2.0) / sample_rate
    inside = np.zeros(len(windows), dtype=bool)
    for start, end in notes:
        inside |= (centres >= start) & (centres <= end)
    if int((~inside).sum()) < MINIMUM_NOISE_FRAMES:
        return samples, 0
    window_function = np.hamming(frame)
    spectra = np.fft.rfft(windows * window_function, axis=1)
    magnitude = np.abs(spectra)
    noise = np.median(magnitude[~inside], axis=0)
    cleaned = np.maximum(magnitude - over_subtraction * noise, spectral_floor * magnitude)
    phase = np.exp(1j * np.angle(spectra))
    rebuilt = np.fft.irfft(cleaned * phase, n=frame) * window_function
    output = np.zeros(len(samples))
    weight = np.zeros(len(samples))
    for index in range(len(windows)):
        start = index * hop
        end = min(start + frame, len(samples))
        output[start:end] += rebuilt[index, : end - start]
        weight[start:end] += window_function[: end - start] ** 2
    covered = weight > 1e-9
    output[covered] /= weight[covered]
    output[~covered] = samples[~covered]
    return output, int((~inside).sum())


def clean(
    signal: Any,
    sample_rate: int,
    *,
    steps: tuple[str, ...],
    band: Band | None,
) -> tuple[Any, dict[str, Any]]:
    """Apply the named steps in the fixed order trim, band, denoise.

    The order is fixed here rather than chosen per run so that "trim and band"
    means one thing. The diagnostics say what each step did to this clip.
    """

    unknown = set(steps) - set(STEPS)
    if unknown:
        raise ValueError(f"unknown cleaning steps: {sorted(unknown)}")
    if "band" in steps and band is None:
        raise ValueError("band-limiting needs a band computed from enrollment clips")
    diagnostics: dict[str, Any] = {"steps": list(steps)}
    output = signal
    if "trim" in steps:
        output, notes = trim_to_notes(output, sample_rate)
        diagnostics["notes_kept"] = notes
        diagnostics["trim_found_no_note"] = notes == 0
    if "band" in steps and band is not None:
        output = band_limit(output, sample_rate, band)
        diagnostics["band"] = band.as_dict()
    if "denoise" in steps:
        output, noise_frames = spectral_subtract(output, sample_rate)
        diagnostics["noise_frames"] = noise_frames
        diagnostics["noise_not_estimated"] = noise_frames == 0
    return output, diagnostics
