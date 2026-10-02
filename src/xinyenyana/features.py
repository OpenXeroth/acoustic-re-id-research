"""The three layers of a call, as vectors.

Source-filter theory splits a vocalisation into what the syrinx does, what the
tract above it does to that sound, and what the bird does with timing. Those
three carry individual information for different reasons and are not equally
stable, so A2 measures them separately rather than as one vector:

- **source**: fundamental frequency and its contour, cycle-to-cycle irregularity
  (jitter and shimmer), and how much of the energy is harmonic. Partly anatomy,
  partly motor control.
- **filter**: the spectral envelope, as cepstral coefficients, and formants
  where the call is harmonic enough to estimate them. Set by the size and shape
  of the tract, so the most stable between sessions and years.
- **motor**: note and gap durations, note count, rhythm, and the shape of the
  amplitude envelope. Behavioural, and the layer most likely to be learned.

Everything here is numpy. Nothing calls a library that would have to be
installed to run the tests, and nothing is a wrapper around an implementation
that cannot be inspected, because a difference between layers is the finding
and it has to be attributable.

Each extractor returns a fixed-length vector and a diagnostics dict. The
diagnostics carry what failed rather than a filled-in default, so A2 can report
the fraction of calls where formant estimation failed instead of silently
averaging a fallback into the result.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from xinyenyana.representations import mel_filterbank as _mel_filterbank
from xinyenyana.representations import read_pcm_wav as _read_pcm_wav

DEFAULT_FRAME = 1024
DEFAULT_HOP = 256
LOW_FREQUENCY_HZ = 50.0
AUDIT_WINDOW_SECONDS = 0.05
AUDIT_WINDOWS_PER_HOP = 4
DEFAULT_F0_MIN = 80.0
DEFAULT_F0_MAX = 4000.0
DEFAULT_VOICING_THRESHOLD = 0.3
OCTAVE_TOLERANCE = 0.85
DEFAULT_ENERGY_THRESHOLD_DB = 12.0
DEFAULT_MINIMUM_CONTRAST_DB = 15.0
DEFAULT_ENVELOPE_FRAME = 256
DEFAULT_ENVELOPE_HOP = 64
ENVELOPE_POINTS = 8
N_CEPSTRA = 13
N_FORMANTS = 3


@dataclass(frozen=True)
class LayerVector:
    """One layer's vector, its field names, and what could not be computed."""

    names: tuple[str, ...]
    values: Any
    diagnostics: dict[str, Any]

    def __post_init__(self) -> None:
        if len(self.names) != len(self.values):
            raise ValueError(f"{len(self.names)} names for {len(self.values)} values")


def energy_share_below_hz(
    samples: Any, sample_rate: int, *, cutoff_hz: float = LOW_FREQUENCY_HZ
) -> tuple[float, float]:
    """Share of a signal's energy below ``cutoff_hz``, and where 99% of it lies.

    The analysis window is a fixed duration rather than a fixed number of
    samples, so the frequency grid is the same at every sample rate and figures
    from recordings made at different rates mean the same thing.

    Both readings interpolate the cumulative spectrum. A 50 Hz cutoff falls
    between bin centres at every usable window length, and taking the nearest
    bin edge instead reports the share below that edge while calling it the
    share below 50 Hz.

    The cutoff is soft. A 0.05 s Hamming window spreads a pure tone over about
    40 Hz, so a reading at 50 Hz cannot separate 40 Hz from 60 Hz. It separates
    recording-chain rumble, which sits in the tens of hertz, from a bird call,
    which does not, and that is what it is for.

    Returns ``nan`` for both when the signal is shorter than one window or
    carries no energy, so a caller reports the clip as unmeasured rather than
    averaging a filled-in value into a median.
    """

    import numpy as np

    frame = int(round(sample_rate * AUDIT_WINDOW_SECONDS))
    if frame < 8 or len(samples) < frame:
        return float("nan"), float("nan")
    hop = max(1, frame // AUDIT_WINDOWS_PER_HOP)
    windows = np.lib.stride_tricks.sliding_window_view(np.asarray(samples, dtype=np.float64), frame)
    windows = windows[::hop]
    power = (np.abs(np.fft.rfft(windows * np.hamming(frame), axis=1)) ** 2).sum(axis=0)
    total = float(power.sum())
    if not total > 0.0:
        return float("nan"), float("nan")
    frequencies = np.fft.rfftfreq(frame, 1.0 / sample_rate)
    cumulative = np.cumsum(power) / total
    return (
        float(np.interp(cutoff_hz, frequencies, cumulative)),
        float(np.interp(0.99, cumulative, frequencies)),
    )


def read_clip(path: Path) -> tuple[Any, int]:
    """Read a PCM WAV as float32 samples and its sample rate."""

    return _read_pcm_wav(path)


def _frames(signal: Any, frame: int, hop: int) -> Any:
    import numpy as np

    if len(signal) < frame:
        signal = np.pad(signal, (0, frame - len(signal)))
    return np.lib.stride_tricks.sliding_window_view(signal, frame)[::hop]


def _normalised_autocorrelation(window: Any) -> Any:
    """Autocorrelation of one frame, divided by its zero-lag value."""

    import numpy as np

    centred = window - window.mean()
    size = 1 << (2 * len(centred) - 1).bit_length()
    spectrum = np.fft.rfft(centred, size)
    correlation = np.fft.irfft(spectrum * np.conjugate(spectrum), size)[: len(centred)]
    zero = correlation[0]
    if zero <= 0:
        return np.zeros_like(correlation)
    return correlation / zero


def frame_pitch(
    window: Any,
    sample_rate: int,
    *,
    f0_min: float = DEFAULT_F0_MIN,
    f0_max: float = DEFAULT_F0_MAX,
    octave_tolerance: float = OCTAVE_TOLERANCE,
) -> tuple[float, float]:
    """Return one frame's fundamental frequency and its autocorrelation peak.

    The peak doubles as the voicing decision and as the harmonic-to-noise
    estimate: a periodic frame correlates with itself at the period lag, a noisy
    one does not.
    """

    import numpy as np

    correlation = _normalised_autocorrelation(window)
    low = max(1, int(sample_rate / f0_max))
    high = min(len(correlation) - 1, int(sample_rate / f0_min))
    if high <= low:
        return 0.0, 0.0
    band = correlation[low : high + 1]
    if not len(band):
        return 0.0, 0.0
    best = int(np.argmax(band))
    strongest = float(band[best])
    if strongest <= 0:
        return 0.0, 0.0

    # Prefer the shortest lag that is nearly as strong as the strongest one.
    # A period of 24.5 samples correlates better at lag 49, two whole periods,
    # than at either 24 or 25, so taking the maximum reports half the true
    # frequency. Any lag within `octave_tolerance` of the peak is a candidate,
    # and the shortest candidate is the fundamental rather than a sub-harmonic.
    # A candidate must be a local maximum: without that, a point on the flank
    # of the zero-lag lobe qualifies and a low-pitched harmonic-rich call is
    # reported as a very high frequency.
    interior = np.arange(1, len(band) - 1)
    local_maximum = (band[interior] > band[interior - 1]) & (band[interior] >= band[interior + 1])
    strong = band[interior] >= octave_tolerance * strongest
    candidates = interior[local_maximum & strong]
    offset = int(candidates.min()) if len(candidates) else best

    # Parabolic interpolation on the three points around the chosen lag, so a
    # period that is not a whole number of samples is not rounded to one.
    refined = float(offset)
    if 0 < offset < len(band) - 1:
        left, centre, right = band[offset - 1], band[offset], band[offset + 1]
        denominator = left - 2.0 * centre + right
        if abs(denominator) > 1e-12:
            shift = 0.5 * (left - right) / denominator
            if abs(shift) <= 1.0:
                refined = offset + shift

    lag = low + refined
    peak = float(band[offset])
    if peak <= 0 or lag <= 0:
        return 0.0, 0.0
    return float(sample_rate) / lag, peak


def high_pass(signal: Any, sample_rate: int, cutoff_hz: float) -> Any:
    """Remove content below ``cutoff_hz`` by subtracting a moving average.

    The RookID clips carry most of their energy below 23 Hz. Nothing a bird does
    is down there; it is rumble, and it dominates both the spectrum and the
    autocorrelation, which is why the pitch tracker returned whatever ceiling it
    was given rather than a fundamental.

    Subtracting a boxcar moving average is a first-order high-pass. It has a
    gentle roll-off and some passband ripple, which does not matter here: the
    aim is to stop energy an octave or more below the lowest plausible
    fundamental from dominating, not to build a precise filter. It is written
    with a cumulative sum so it stays linear in the length of the clip, and it
    is short enough to read, which a designed filter from a library would not
    be.
    """

    import numpy as np

    samples = np.asarray(signal, dtype=np.float64)
    width = int(round(sample_rate / max(cutoff_hz, 1e-6)))
    if width < 2 or width >= len(samples):
        return samples - samples.mean() if len(samples) else samples
    padded = np.pad(samples, (width // 2, width - width // 2 - 1), mode="edge")
    cumulative = np.cumsum(np.insert(padded, 0, 0.0))
    average = (cumulative[width:] - cumulative[:-width]) / width
    return samples - average[: len(samples)]


def energy_gate(
    windows: Any,
    *,
    threshold_db: float = DEFAULT_ENERGY_THRESHOLD_DB,
    minimum_contrast_db: float = DEFAULT_MINIMUM_CONTRAST_DB,
) -> Any:
    """Mark the frames loud enough for their periodicity to mean anything.

    The autocorrelation is normalised by each frame's own energy, so it is
    scale-free: a frame of background noise correlates with itself and looks as
    periodic as a call. Without a gate, every E01d clip came back 100% voiced
    with a fundamental pinned at the search ceiling.

    Two situations have to be handled by one rule.

    A call inside background has a floor to measure: the E01d clips sit at about
    -10.5 dB with peaks near 13.6 dB, and the call is the part more than
    ``threshold_db`` above the floor. The floor is the 10th percentile of frame
    energy rather than the minimum, so one unusually quiet frame cannot set it.

    A call that runs the whole clip has no floor: its 10th percentile is as loud
    as its peak. Gating on floor plus a threshold would then reject the entire
    call. So when the contrast between floor and peak is smaller than
    ``minimum_contrast_db``, there is nothing to gate on and every frame passes,
    leaving the decision to the correlation threshold.
    """

    import numpy as np

    energy = 10.0 * np.log10(np.maximum((windows**2).mean(axis=1), 1e-20))
    if not len(energy):
        return np.zeros(0, dtype=bool)
    floor = float(np.percentile(energy, 10))
    contrast = float(energy.max()) - floor
    if contrast < minimum_contrast_db:
        return np.ones(len(energy), dtype=bool)
    return energy >= floor + threshold_db


@dataclass(frozen=True)
class EndpointSettings:
    """Per-endpoint feature settings, because one default fits no species.

    The fundamental-frequency search range is the setting that matters. A rook
    caws around 500 Hz and a zebra finch calls far higher; a range wide enough
    for both lets the tracker report a harmonic for one and rumble for the
    other. On the RookID clips a 80 to 4000 Hz range returned the ceiling for
    every clip, and a 250 to 900 Hz range returned a median of 490 Hz.

    These are search bounds, not claims about the species. They are recorded
    with every run so a result can be read against the range that produced it.
    """

    name: str
    f0_min: float
    f0_max: float
    note: str = ""

    def as_dict(self) -> dict[str, Any]:
        return {
            "endpoint": self.name,
            "f0_min_hz": self.f0_min,
            "f0_max_hz": self.f0_max,
            "note": self.note,
        }


def source_layer(
    signal: Any,
    sample_rate: int,
    *,
    frame: int = DEFAULT_FRAME,
    hop: int = DEFAULT_HOP,
    f0_min: float = DEFAULT_F0_MIN,
    f0_max: float = DEFAULT_F0_MAX,
    voicing_threshold: float = DEFAULT_VOICING_THRESHOLD,
    energy_threshold_db: float = DEFAULT_ENERGY_THRESHOLD_DB,
    high_pass_signal: bool = True,
) -> LayerVector:
    """Fundamental frequency statistics, jitter, shimmer and harmonic-to-noise.

    The clip is high-passed at ``f0_min`` first, because content below the
    lowest plausible fundamental cannot be one and does dominate the
    autocorrelation when it is present.
    """

    import numpy as np

    filtered = high_pass(signal, sample_rate, f0_min) if high_pass_signal else signal
    windows = _frames(np.asarray(filtered, dtype=np.float64), frame, hop)
    window_function = np.hamming(frame)
    pitches, peaks = [], []
    for row in windows:
        f0, peak = frame_pitch(row * window_function, sample_rate, f0_min=f0_min, f0_max=f0_max)
        pitches.append(f0)
        peaks.append(peak)
    pitch = np.asarray(pitches)
    peaks_array = np.asarray(peaks)

    loud_enough = energy_gate(windows, threshold_db=energy_threshold_db)
    voiced = loud_enough & (peaks_array >= voicing_threshold) & (pitch > 0)
    names = (
        "f0_mean",
        "f0_std",
        "f0_min",
        "f0_max",
        "f0_slope_hz_per_second",
        "jitter_local",
        "shimmer_local",
        "harmonic_to_noise_db",
        "voiced_fraction",
    )
    diagnostics: dict[str, Any] = {
        "frames": int(len(pitch)),
        "voiced_frames": int(voiced.sum()),
        "voiced_fraction": float(voiced.mean()) if len(voiced) else 0.0,
    }
    if voiced.sum() < 3:
        diagnostics["failed"] = "fewer than three voiced frames"
        return LayerVector(names, np.zeros(len(names), dtype=np.float32), diagnostics)

    voiced_pitch = pitch[voiced]
    times = np.nonzero(voiced)[0] * (hop / sample_rate)
    slope = float(np.polyfit(times, voiced_pitch, 1)[0]) if len(set(times)) > 1 else 0.0

    periods = 1.0 / voiced_pitch
    jitter = float(np.mean(np.abs(np.diff(periods))) / np.mean(periods))

    amplitudes = np.sqrt((windows[voiced] ** 2).mean(axis=1))
    positive = amplitudes[amplitudes > 0]
    shimmer = (
        float(np.mean(np.abs(np.diff(positive))) / np.mean(positive)) if len(positive) > 1 else 0.0
    )

    clipped = np.clip(peaks_array[voiced], 1e-6, 1 - 1e-6)
    hnr = float(np.mean(10.0 * np.log10(clipped / (1.0 - clipped))))

    values = np.asarray(
        [
            float(voiced_pitch.mean()),
            float(voiced_pitch.std()),
            float(voiced_pitch.min()),
            float(voiced_pitch.max()),
            slope,
            jitter,
            shimmer,
            hnr,
            float(voiced.mean()),
        ],
        dtype=np.float32,
    )
    return LayerVector(names, values, diagnostics)


def _levinson_durbin(autocorrelation: Any, order: int) -> Any:
    """Solve the Yule-Walker equations for linear-prediction coefficients."""

    import numpy as np

    coefficients = np.zeros(order + 1)
    coefficients[0] = 1.0
    error = autocorrelation[0]
    if error <= 0:
        return coefficients
    for index in range(1, order + 1):
        accumulator = autocorrelation[index]
        for j in range(1, index):
            accumulator += coefficients[j] * autocorrelation[index - j]
        reflection = -accumulator / error
        updated = coefficients.copy()
        for j in range(1, index):
            updated[j] = coefficients[j] + reflection * coefficients[index - j]
        updated[index] = reflection
        coefficients = updated
        error *= 1.0 - reflection * reflection
        if error <= 0:
            break
    return coefficients


def formants(
    signal: Any,
    sample_rate: int,
    *,
    count: int = N_FORMANTS,
    order: int | None = None,
    minimum_hz: float = 150.0,
    maximum_bandwidth_hz: float = 1500.0,
) -> tuple[list[float], list[float]]:
    """Estimate formant centre frequencies and bandwidths by linear prediction.

    Returns as many as it can find, which may be fewer than ``count``. The
    caller decides what an incomplete estimate means; nothing is padded here.
    """

    import numpy as np

    samples = np.asarray(signal, dtype=np.float64)
    if len(samples) < 32:
        return [], []
    emphasised = np.append(samples[0], samples[1:] - 0.97 * samples[:-1])
    windowed = emphasised * np.hamming(len(emphasised))
    lpc_order = order if order is not None else int(2 + sample_rate / 1000)
    lpc_order = max(4, min(lpc_order, len(windowed) - 1))
    # By FFT, not np.correlate: the direct form is quadratic, and a three-second
    # clip at 48 kHz is 144,000 samples.
    size = 1 << (2 * len(windowed) - 1).bit_length()
    spectrum = np.fft.rfft(windowed, size)
    correlation = np.fft.irfft(spectrum * np.conjugate(spectrum), size)[: lpc_order + 1]
    if correlation[0] <= 0:
        return [], []
    coefficients = _levinson_durbin(correlation, lpc_order)
    roots = np.roots(coefficients)
    roots = roots[np.imag(roots) > 0]
    if not len(roots):
        return [], []
    angles = np.arctan2(np.imag(roots), np.real(roots))
    frequencies = angles * (sample_rate / (2 * np.pi))
    magnitudes = np.abs(roots)
    bandwidths = -(sample_rate / np.pi) * np.log(np.clip(magnitudes, 1e-12, 1 - 1e-12))
    keep = (frequencies > minimum_hz) & (bandwidths < maximum_bandwidth_hz)
    frequencies, bandwidths = frequencies[keep], bandwidths[keep]
    order_index = np.argsort(frequencies)
    return (
        [float(v) for v in frequencies[order_index][:count]],
        [float(v) for v in bandwidths[order_index][:count]],
    )


def frame_formants(
    signal: Any,
    sample_rate: int,
    *,
    count: int = N_FORMANTS,
    frame: int = DEFAULT_FRAME,
    hop: int = DEFAULT_HOP,
    voicing_threshold: float = DEFAULT_VOICING_THRESHOLD,
) -> tuple[list[float], list[float], int]:
    """Estimate formants per voiced frame and take the median across frames.

    Estimating over a whole clip would fit one all-pole model to the call, its
    padding and its silence together, and the padding has no vocal tract. Only
    frames the pitch tracker calls voiced contribute, and the median is used so
    a single badly fitted frame cannot move the result.

    Returns the centres, the bandwidths, and how many frames contributed.
    """

    import numpy as np

    samples = high_pass(signal, sample_rate, DEFAULT_F0_MIN)
    windows = _frames(samples, frame, hop)
    window_function = np.hamming(frame)
    loud_enough = energy_gate(windows)
    centres: list[list[float]] = []
    bandwidths: list[list[float]] = []
    for index, row in enumerate(windows):
        if not loud_enough[index]:
            continue
        _, peak = frame_pitch(row * window_function, sample_rate)
        if peak < voicing_threshold:
            continue
        found, widths = formants(row, sample_rate, count=count)
        if len(found) == count:
            centres.append(found)
            bandwidths.append(widths)
    if not centres:
        return [], [], 0
    centre_array = np.asarray(centres)
    bandwidth_array = np.asarray(bandwidths)
    return (
        [float(v) for v in np.median(centre_array, axis=0)],
        [float(v) for v in np.median(bandwidth_array, axis=0)],
        len(centres),
    )


def cepstra(
    signal: Any,
    sample_rate: int,
    *,
    frame: int = DEFAULT_FRAME,
    hop: int = DEFAULT_HOP,
    n_mels: int = 40,
    n_cepstra: int = N_CEPSTRA,
    fmin: float = 200.0,
    fmax: float | None = None,
) -> Any:
    """Per-frame cepstral coefficients: log mel filterbank then a DCT-II."""

    import numpy as np

    samples = np.asarray(signal, dtype=np.float64)
    windows = _frames(samples, frame, hop) * np.hamming(frame)
    power = np.abs(np.fft.rfft(windows, axis=1)) ** 2
    ceiling = fmax if fmax is not None else sample_rate / 2
    filters = _mel_filterbank(sample_rate, frame, n_mels, fmin, min(ceiling, sample_rate / 2))
    log_mel = np.log(np.maximum(power @ filters.T, 1e-10))
    indices = np.arange(n_mels)
    basis = np.cos(np.pi / n_mels * (indices[None, :] + 0.5) * np.arange(n_cepstra)[:, None])
    return log_mel @ basis.T


def cepstral_mean_variance_normalise(coefficients: Any) -> Any:
    """Remove the mean and scale of the frames handed in, from those frames.

    A fixed microphone, distance and propagation path add the same pattern to
    the log spectrum of every frame. Subtracting the mean removes that pattern;
    dividing by the standard deviation removes a constant gain difference.

    The scope is whatever is passed in. ``filter_layer`` passes one clip, so
    what this removes there is that clip's own mean, not the recording's, and
    what survives is what varied inside the clip. Compensating a session means
    pooling the frames of every clip from that recording, which is a caller's
    decision and not this function's.
    """

    import numpy as np

    values = np.asarray(coefficients, dtype=np.float64)
    mean = values.mean(axis=0, keepdims=True)
    scale = values.std(axis=0, keepdims=True)
    scale[scale < 1e-9] = 1.0
    return (values - mean) / scale


def filter_layer(
    signal: Any,
    sample_rate: int,
    *,
    normalise: bool,
    frame: int = DEFAULT_FRAME,
    hop: int = DEFAULT_HOP,
    n_cepstra: int = N_CEPSTRA,
    formant_count: int = N_FORMANTS,
) -> LayerVector:
    """Cepstral summary, optionally normalised per recording, plus formants."""

    import numpy as np

    coefficients = cepstra(signal, sample_rate, frame=frame, hop=hop, n_cepstra=n_cepstra)
    if normalise:
        coefficients = cepstral_mean_variance_normalise(coefficients)
    summary = np.concatenate((coefficients.mean(axis=0), coefficients.std(axis=0)))

    centres, bandwidths, formant_frames = frame_formants(
        signal, sample_rate, count=formant_count, frame=frame, hop=hop
    )
    diagnostics: dict[str, Any] = {
        "normalised": bool(normalise),
        "formants_found": len(centres),
        "formants_requested": formant_count,
        "formant_estimation_failed": len(centres) < formant_count,
        "formant_frames": formant_frames,
    }
    padded_centres = list(centres) + [0.0] * (formant_count - len(centres))
    padded_bandwidths = list(bandwidths) + [0.0] * (formant_count - len(bandwidths))
    if diagnostics["formant_estimation_failed"]:
        diagnostics["failed"] = f"only {len(centres)} of {formant_count} formants estimable"

    names = tuple(
        [f"cepstrum_{i}_mean" for i in range(n_cepstra)]
        + [f"cepstrum_{i}_std" for i in range(n_cepstra)]
        + [f"formant_{i + 1}_hz" for i in range(formant_count)]
        + [f"formant_{i + 1}_bandwidth_hz" for i in range(formant_count)]
    )
    values = np.concatenate((summary, padded_centres, padded_bandwidths)).astype(np.float32)
    return LayerVector(names, values, diagnostics)


def note_boundaries(
    signal: Any,
    sample_rate: int,
    *,
    frame: int = DEFAULT_ENVELOPE_FRAME,
    hop: int = DEFAULT_ENVELOPE_HOP,
    threshold_db_above_floor: float = 12.0,
    minimum_note_seconds: float = 0.01,
) -> list[tuple[float, float]]:
    """Find note start and end times from the clip's own energy envelope.

    The threshold is relative to the clip's own noise floor, taken as the 10th
    percentile of frame energy, so a quiet recording and a loud one are treated
    alike. An absolute threshold would find notes in one and none in the other.

    The frame here is much shorter than the one the spectral layers use. A frame
    lights up as soon as a note enters its window, so a 1024-sample frame
    reports every note about 46 ms early and 43 ms too long at 22 kHz. Timing is
    what this layer measures, so it is framed for time rather than frequency,
    and each frame is placed at its centre rather than its start.
    """

    import numpy as np

    samples = np.asarray(signal, dtype=np.float64)
    windows = _frames(samples, frame, hop)
    energy = 10.0 * np.log10(np.maximum((windows**2).mean(axis=1), 1e-20))
    if not len(energy):
        return []
    floor = float(np.percentile(energy, 10))
    active = energy >= floor + threshold_db_above_floor
    boundaries: list[tuple[float, float]] = []
    start: int | None = None
    for index, value in enumerate(active):
        if value and start is None:
            start = index
        elif not value and start is not None:
            boundaries.append((start, index))
            start = None
    if start is not None:
        boundaries.append((start, len(active)))
    seconds = hop / sample_rate
    centre = frame / (2.0 * sample_rate)
    spans = [(a * seconds + centre, b * seconds + centre) for a, b in boundaries]
    return [(a, b) for a, b in spans if b - a >= minimum_note_seconds]


def motor_layer(
    signal: Any,
    sample_rate: int,
    *,
    frame: int = DEFAULT_ENVELOPE_FRAME,
    hop: int = DEFAULT_ENVELOPE_HOP,
    envelope_points: int = ENVELOPE_POINTS,
) -> LayerVector:
    """Note and gap timing, rhythm, and the shape of the amplitude envelope."""

    import numpy as np

    samples = np.asarray(signal, dtype=np.float64)
    notes = note_boundaries(samples, sample_rate, frame=frame, hop=hop)
    names = tuple(
        [
            "note_count",
            "note_duration_mean",
            "note_duration_std",
            "gap_duration_mean",
            "gap_duration_std",
            "note_rate_per_second",
            "voiced_time_fraction",
            "inter_onset_variation",
        ]
        + [f"envelope_{i}" for i in range(envelope_points)]
    )
    diagnostics: dict[str, Any] = {"notes": len(notes)}

    duration = len(samples) / sample_rate
    if not notes:
        diagnostics["failed"] = "no note above the threshold"
        return LayerVector(names, np.zeros(len(names), dtype=np.float32), diagnostics)

    lengths = np.asarray([end - start for start, end in notes])
    gaps = np.asarray([notes[i + 1][0] - notes[i][1] for i in range(len(notes) - 1)])
    onsets = np.asarray([start for start, _ in notes])
    intervals = np.diff(onsets)
    variation = (
        float(intervals.std() / intervals.mean())
        if len(intervals) and intervals.mean() > 0
        else 0.0
    )

    windows = _frames(samples, frame, hop)
    envelope = np.sqrt((windows**2).mean(axis=1))
    peak = envelope.max()
    if peak > 0:
        envelope = envelope / peak
    resampled = np.interp(
        np.linspace(0.0, 1.0, envelope_points),
        np.linspace(0.0, 1.0, len(envelope)),
        envelope,
    )

    values = np.concatenate(
        (
            np.asarray(
                [
                    float(len(notes)),
                    float(lengths.mean()),
                    float(lengths.std()),
                    float(gaps.mean()) if len(gaps) else 0.0,
                    float(gaps.std()) if len(gaps) else 0.0,
                    float(len(notes)) / duration if duration > 0 else 0.0,
                    float(lengths.sum()) / duration if duration > 0 else 0.0,
                    variation,
                ]
            ),
            resampled,
        )
    ).astype(np.float32)
    return LayerVector(names, values, diagnostics)


LAYERS = ("source", "filter", "filter_normalised", "motor")


def layers_from_signal(signal: Any, sample_rate: int, **kwargs: Any) -> dict[str, LayerVector]:
    """Extract every layer from a signal already in memory.

    Separate from ``extract_layers`` so that a cleaned signal goes through the
    identical extraction as an uncleaned one, rather than through a second copy
    of it.
    """

    return {
        "source": source_layer(signal, sample_rate, **kwargs),
        "filter": filter_layer(signal, sample_rate, normalise=False),
        "filter_normalised": filter_layer(signal, sample_rate, normalise=True),
        "motor": motor_layer(signal, sample_rate),
    }


def extract_layers(path: Path, **kwargs: Any) -> dict[str, LayerVector]:
    """Extract every layer from one clip."""

    signal, sample_rate = read_clip(path)
    return layers_from_signal(signal, sample_rate, **kwargs)
