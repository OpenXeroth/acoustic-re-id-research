"""Tests for the three call layers.

Each test builds a signal with a property put into it deliberately and checks
the extractor recovers that property. A feature that cannot recover a value it
was handed cannot be trusted to find one that is genuinely there.
"""

from __future__ import annotations

import wave
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from xinyenyana.features import (
    AUDIT_WINDOW_SECONDS,
    ENVELOPE_POINTS,
    N_CEPSTRA,
    N_FORMANTS,
    cepstra,
    cepstral_mean_variance_normalise,
    energy_share_below_hz,
    extract_layers,
    filter_layer,
    formants,
    frame_pitch,
    motor_layer,
    note_boundaries,
    source_layer,
)

RATE = 22_050


def _sawtooth(f0: float, seconds: float, *, rate: int = RATE) -> Any:
    t = np.arange(int(seconds * rate)) / rate
    return (2.0 * ((t * f0) % 1.0) - 1.0) * 8000.0


def _tone(frequency: float, seconds: float, *, rate: int = RATE) -> Any:
    t = np.arange(int(seconds * rate)) / rate
    return np.sin(2 * np.pi * frequency * t) * 8000.0


def _write_wav(path: Path, signal: Any, *, rate: int = RATE) -> Path:
    with wave.open(str(path), "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(rate)
        handle.writeframes(np.clip(signal, -32768, 32767).astype("<i2").tobytes())
    return path


# --- source layer ---------------------------------------------------------


@pytest.mark.parametrize("f0", [150.0, 400.0, 900.0])
def test_frame_pitch_recovers_the_fundamental_it_was_given(f0: float) -> None:
    window = _sawtooth(f0, 0.05)[:1024] * np.hamming(1024)
    estimated, peak = frame_pitch(window, RATE)
    assert estimated == pytest.approx(f0, rel=0.05)
    assert peak > 0.5


def test_frame_pitch_reports_no_periodicity_for_noise() -> None:
    rng = np.random.default_rng(0)
    window = rng.normal(size=1024) * np.hamming(1024)
    _, peak = frame_pitch(window, RATE)
    assert peak < 0.3


def test_source_layer_recovers_a_constant_fundamental() -> None:
    layer = source_layer(_sawtooth(300.0, 0.5), RATE)
    values = dict(zip(layer.names, layer.values, strict=True))
    assert values["f0_mean"] == pytest.approx(300.0, rel=0.05)
    assert values["f0_std"] < 20.0
    assert values["voiced_fraction"] > 0.9
    assert "failed" not in layer.diagnostics


def test_source_layer_recovers_a_rising_contour_as_a_positive_slope() -> None:
    seconds, start, end = 0.6, 200.0, 600.0
    t = np.arange(int(seconds * RATE)) / RATE
    sweep = start + (end - start) * t / seconds
    phase = 2 * np.pi * np.cumsum(sweep) / RATE
    layer = source_layer((2.0 * ((phase / (2 * np.pi)) % 1.0) - 1.0) * 8000.0, RATE)
    values = dict(zip(layer.names, layer.values, strict=True))
    expected = (end - start) / seconds
    assert values["f0_slope_hz_per_second"] == pytest.approx(expected, rel=0.25)
    assert values["f0_min"] < values["f0_max"]


def test_harmonic_to_noise_separates_a_tone_from_noise() -> None:
    rng = np.random.default_rng(1)
    tone = source_layer(_sawtooth(300.0, 0.5), RATE)
    noisy = source_layer(rng.normal(scale=8000.0, size=int(0.5 * RATE)), RATE)
    tone_hnr = dict(zip(tone.names, tone.values, strict=True))["harmonic_to_noise_db"]
    if "failed" in noisy.diagnostics:
        return  # noise had too few voiced frames to summarise, which is the point
    noise_hnr = dict(zip(noisy.names, noisy.values, strict=True))["harmonic_to_noise_db"]
    assert tone_hnr > noise_hnr


def test_jitter_is_larger_for_a_fundamental_that_wanders() -> None:
    rng = np.random.default_rng(2)
    steady = source_layer(_sawtooth(300.0, 0.6), RATE)

    seconds = 0.6
    frames = int(seconds * RATE)
    wobble = 300.0 + rng.normal(scale=25.0, size=frames)
    phase = 2 * np.pi * np.cumsum(wobble) / RATE
    unsteady = source_layer((2.0 * ((phase / (2 * np.pi)) % 1.0) - 1.0) * 8000.0, RATE)

    steady_jitter = dict(zip(steady.names, steady.values, strict=True))["jitter_local"]
    unsteady_jitter = dict(zip(unsteady.names, unsteady.values, strict=True))["jitter_local"]
    assert unsteady_jitter > steady_jitter


def test_shimmer_is_larger_when_amplitude_varies() -> None:
    steady = _sawtooth(300.0, 0.6)
    varying = steady * (1.0 + 0.6 * np.sin(2 * np.pi * 7.0 * np.arange(len(steady)) / RATE))
    a = dict(zip(*(lambda v: (v.names, v.values))(source_layer(steady, RATE)), strict=True))
    b = dict(zip(*(lambda v: (v.names, v.values))(source_layer(varying, RATE)), strict=True))
    assert b["shimmer_local"] > a["shimmer_local"]


def test_source_layer_reports_failure_rather_than_zeros_for_silence() -> None:
    layer = source_layer(np.zeros(int(0.3 * RATE)), RATE)
    assert layer.diagnostics["failed"] == "fewer than three voiced frames"
    assert layer.diagnostics["voiced_frames"] == 0


# --- filter layer ---------------------------------------------------------


def test_formants_find_a_resonance_that_was_put_there() -> None:
    """A two-pole resonator at 1200 Hz should show up as a formant near 1200 Hz."""

    rng = np.random.default_rng(3)
    excitation = rng.normal(size=int(0.2 * RATE))
    centre, bandwidth = 1200.0, 120.0
    radius = np.exp(-np.pi * bandwidth / RATE)
    theta = 2 * np.pi * centre / RATE
    a1, a2 = -2 * radius * np.cos(theta), radius**2
    resonated = np.zeros_like(excitation)
    for n in range(2, len(excitation)):
        resonated[n] = excitation[n] - a1 * resonated[n - 1] - a2 * resonated[n - 2]

    centres, bandwidths = formants(resonated * 1000.0, RATE, count=3)
    assert centres, "no formant found in a signal built from one resonance"
    assert min(abs(f - centre) for f in centres) < 150.0
    assert len(bandwidths) == len(centres)


def test_formants_return_nothing_for_a_signal_too_short_to_estimate() -> None:
    assert formants(np.zeros(8), RATE) == ([], [])


def test_filter_layer_records_formant_failure_instead_of_hiding_it() -> None:
    layer = filter_layer(np.zeros(int(0.2 * RATE)), RATE, normalise=False)
    assert layer.diagnostics["formant_estimation_failed"] is True
    assert layer.diagnostics["formants_found"] < N_FORMANTS
    assert "failed" in layer.diagnostics


def test_filter_layer_has_the_length_its_names_promise() -> None:
    layer = filter_layer(_sawtooth(300.0, 0.4), RATE, normalise=False)
    assert len(layer.values) == 2 * N_CEPSTRA + 2 * N_FORMANTS
    assert len(layer.names) == len(layer.values)


def test_cepstral_normalisation_removes_a_fixed_channel_colouring() -> None:
    """The same call through two different channels should look more alike after CMVN.

    This is the property the normalisation exists for. A first-order filter is
    a stand-in for a microphone and a propagation path: it adds a fixed tilt to
    the log spectrum of every frame.
    """

    call = _sawtooth(320.0, 0.5) + _sawtooth(517.0, 0.5) * 0.4
    bright = np.append(call[0], call[1:] - 0.8 * call[:-1])
    dull = np.convolve(call, np.ones(5) / 5.0, mode="same")

    raw_bright, raw_dull = cepstra(bright, RATE), cepstra(dull, RATE)
    raw_distance = float(
        np.linalg.norm(raw_bright.mean(axis=0) - raw_dull.mean(axis=0))
        / np.linalg.norm(raw_bright.mean(axis=0))
    )
    normalised_distance = float(
        np.linalg.norm(
            cepstral_mean_variance_normalise(raw_bright).mean(axis=0)
            - cepstral_mean_variance_normalise(raw_dull).mean(axis=0)
        )
    )
    assert normalised_distance < raw_distance


def test_cepstral_normalisation_makes_each_recording_zero_mean_and_unit_scale() -> None:
    values = cepstral_mean_variance_normalise(cepstra(_sawtooth(300.0, 0.4), RATE))
    assert np.allclose(values.mean(axis=0), 0.0, atol=1e-9)
    assert np.allclose(values.std(axis=0), 1.0, atol=1e-6)


def test_cepstral_normalisation_survives_a_constant_coefficient() -> None:
    constant = np.ones((10, 3))
    assert np.all(np.isfinite(cepstral_mean_variance_normalise(constant)))


# --- motor layer ----------------------------------------------------------


def _three_notes(note: float, gap: float) -> Any:
    silence = np.zeros(int(gap * RATE))
    burst = _sawtooth(400.0, note)
    return np.concatenate([silence, burst, silence, burst, silence, burst, silence])


def test_note_boundaries_count_the_notes_that_were_put_in() -> None:
    spans = note_boundaries(_three_notes(0.12, 0.10), RATE)
    assert len(spans) == 3
    for start, end in spans:
        assert end - start == pytest.approx(0.12, abs=0.04)


def test_note_boundaries_find_nothing_in_silence() -> None:
    assert note_boundaries(np.zeros(int(0.5 * RATE)), RATE) == []


def test_note_boundaries_are_relative_to_the_clip_and_not_an_absolute_level() -> None:
    loud = _three_notes(0.12, 0.10)
    quiet = loud / 50.0
    assert len(note_boundaries(loud, RATE)) == len(note_boundaries(quiet, RATE)) == 3


def test_motor_layer_recovers_note_count_and_timing() -> None:
    layer = motor_layer(_three_notes(0.12, 0.10), RATE)
    values = dict(zip(layer.names, layer.values, strict=True))
    assert values["note_count"] == 3
    assert values["note_duration_mean"] == pytest.approx(0.12, abs=0.04)
    assert values["gap_duration_mean"] == pytest.approx(0.10, abs=0.04)
    assert len(layer.values) == 8 + ENVELOPE_POINTS


def test_motor_layer_separates_an_even_rhythm_from_an_uneven_one() -> None:
    even = motor_layer(_three_notes(0.10, 0.10), RATE)
    uneven = np.concatenate(
        [
            np.zeros(int(0.05 * RATE)),
            _sawtooth(400.0, 0.10),
            np.zeros(int(0.05 * RATE)),
            _sawtooth(400.0, 0.10),
            np.zeros(int(0.35 * RATE)),
            _sawtooth(400.0, 0.10),
            np.zeros(int(0.05 * RATE)),
        ]
    )
    a = dict(zip(even.names, even.values, strict=True))["inter_onset_variation"]
    b = dict(zip(motor_layer(uneven, RATE).names, motor_layer(uneven, RATE).values, strict=True))[
        "inter_onset_variation"
    ]
    assert b > a


def test_motor_layer_reports_failure_rather_than_zeros_for_silence() -> None:
    layer = motor_layer(np.zeros(int(0.4 * RATE)), RATE)
    assert layer.diagnostics["failed"] == "no note above the threshold"


# --- all layers -----------------------------------------------------------


def test_extract_layers_reads_a_wav_and_returns_every_layer(tmp_path: Path) -> None:
    path = _write_wav(tmp_path / "call.wav", _three_notes(0.12, 0.10))
    layers = extract_layers(path)
    assert set(layers) == {"source", "filter", "filter_normalised", "motor"}
    for name, layer in layers.items():
        assert len(layer.names) == len(layer.values), name
        assert np.all(np.isfinite(layer.values)), name


def test_the_normalised_and_unnormalised_filter_layers_differ(tmp_path: Path) -> None:
    path = _write_wav(tmp_path / "call.wav", _sawtooth(320.0, 0.5))
    layers = extract_layers(path)
    assert not np.allclose(layers["filter"].values, layers["filter_normalised"].values)


def test_a_layer_vector_refuses_a_name_count_that_does_not_match() -> None:
    from xinyenyana.features import LayerVector

    with pytest.raises(ValueError, match="names for"):
        LayerVector(("a", "b"), np.zeros(3), {})


def test_pitch_does_not_report_an_octave_error_on_a_non_integer_period() -> None:
    """A period of 24.5 samples correlates best at twice that lag.

    Taking the strongest autocorrelation peak reports 450 Hz for a 900 Hz call.
    This is the case that produced that error before the octave check existed.
    """

    window = _sawtooth(900.0, 0.05)[:1024] * np.hamming(1024)
    estimated, _ = frame_pitch(window, RATE)
    assert estimated == pytest.approx(900.0, rel=0.05)
    assert estimated > 700.0, "reported a sub-harmonic instead of the fundamental"


def test_the_octave_check_does_not_push_every_call_upwards() -> None:
    """A genuine 450 Hz call must still be reported as 450 Hz.

    Preferring the shortest strong lag would otherwise turn every low call into
    a high one, which is the same error in the other direction.
    """

    for true_f0 in (120.0, 150.0, 300.0, 450.0):
        window = _sawtooth(true_f0, 0.05)[:1024] * np.hamming(1024)
        estimated, _ = frame_pitch(window, RATE)
        assert estimated == pytest.approx(true_f0, rel=0.05), f"{true_f0} Hz became {estimated}"


def test_note_onsets_land_close_to_where_the_notes_actually_start() -> None:
    """The envelope frame sets the timing resolution, so it is checked directly.

    With a 1024-sample spectral frame these onsets came out 42 ms early and the
    durations 43 ms long, because a frame lights up as soon as a note enters its
    window. The envelope is framed for time instead.
    """

    spans = note_boundaries(_three_notes(0.12, 0.10), RATE)
    starts = [start for start, _ in spans]
    assert starts == pytest.approx([0.10, 0.32, 0.54], abs=0.015)
    for start, end in spans:
        assert end - start == pytest.approx(0.12, abs=0.015)


def test_high_pass_removes_rumble_that_would_otherwise_capture_the_pitch() -> None:
    """Infrasonic energy dominated the RookID clips and the tracker followed it.

    Half to three quarters of the energy in those clips sits below 23 Hz. This
    reproduces that: a 500 Hz call under a much louder 15 Hz rumble.
    """

    seconds = 0.5
    t = np.arange(int(seconds * RATE)) / RATE
    call = _sawtooth(500.0, seconds)
    rumble = np.sin(2 * np.pi * 15.0 * t) * 40000.0

    unfiltered = source_layer(
        call + rumble, RATE, f0_min=250.0, f0_max=900.0, high_pass_signal=False
    )
    filtered = source_layer(call + rumble, RATE, f0_min=250.0, f0_max=900.0)

    unfiltered_f0 = dict(zip(unfiltered.names, unfiltered.values, strict=True))["f0_mean"]
    filtered_f0 = dict(zip(filtered.names, filtered.values, strict=True))["f0_mean"]
    assert filtered_f0 == pytest.approx(500.0, rel=0.10)
    assert abs(filtered_f0 - 500.0) < abs(unfiltered_f0 - 500.0)


def test_high_pass_leaves_a_clean_call_alone() -> None:
    clean = source_layer(_sawtooth(500.0, 0.5), RATE, f0_min=250.0, f0_max=900.0)
    value = dict(zip(clean.names, clean.values, strict=True))["f0_mean"]
    assert value == pytest.approx(500.0, rel=0.05)


def test_the_energy_gate_finds_the_call_inside_background() -> None:
    """A short call in background must not report the whole clip as voiced."""

    background = np.random.default_rng(7).normal(scale=100.0, size=int(2.0 * RATE))
    start = int(0.9 * RATE)
    call = _sawtooth(500.0, 0.2)
    background[start : start + len(call)] += call
    layer = source_layer(background, RATE, f0_min=250.0, f0_max=900.0)
    fraction = dict(zip(layer.names, layer.values, strict=True))["voiced_fraction"]
    assert 0.02 < fraction < 0.5, f"voiced fraction {fraction} for a 10% call"


def test_the_energy_gate_passes_everything_when_there_is_no_floor() -> None:
    """A call filling the whole clip has no background to measure a floor from."""

    layer = source_layer(_sawtooth(500.0, 0.5), RATE, f0_min=250.0, f0_max=900.0)
    fraction = dict(zip(layer.names, layer.values, strict=True))["voiced_fraction"]
    assert fraction > 0.9


def test_endpoint_settings_round_trip_their_search_range() -> None:
    from xinyenyana.features import EndpointSettings

    settings = EndpointSettings("rookid", 250.0, 900.0, note="caws around 500 Hz")
    recorded = settings.as_dict()
    assert recorded["f0_min_hz"] == 250.0
    assert recorded["f0_max_hz"] == 900.0
    assert recorded["endpoint"] == "rookid"


# --- energy below a stated cutoff -------------------------------------------


def test_a_low_rumble_reads_as_nearly_all_energy_below_fifty_hertz() -> None:
    rate = 48_000
    t = np.arange(rate) / rate
    share, _ = energy_share_below_hz(np.sin(2 * np.pi * 12.0 * t), rate)
    assert share > 0.9


def test_a_call_band_tone_reads_as_almost_none_below_fifty_hertz() -> None:
    rate = 48_000
    t = np.arange(rate) / rate
    share, ninety_ninth = energy_share_below_hz(np.sin(2 * np.pi * 2_000.0 * t), rate)
    assert share < 0.01
    assert 1_800.0 < ninety_ninth < 2_200.0


def test_a_signal_shorter_than_one_window_reports_nothing_rather_than_guessing() -> None:
    rate = 48_000
    frame = int(round(rate * AUDIT_WINDOW_SECONDS))
    share, ninety_ninth = energy_share_below_hz(np.zeros(frame - 1), rate)
    assert np.isnan(share) and np.isnan(ninety_ninth)


def test_a_silent_signal_reports_nothing_rather_than_dividing_by_zero() -> None:
    rate = 48_000
    share, ninety_ninth = energy_share_below_hz(np.zeros(rate), rate)
    assert np.isnan(share) and np.isnan(ninety_ninth)


def test_the_same_tone_reads_the_same_at_three_sample_rates() -> None:
    """The reason the window is a duration and not a sample count.

    A fixed sample count gives a different frequency grid at each rate, so the
    first bin edge at or above 50 Hz lands at 93.75 Hz, 86.13 Hz and 64.60 Hz on
    the three rates this project's endpoints were recorded at. Three figures
    labelled "below 50 Hz" then measured three different things.
    """

    shares = []
    for rate in (48_000, 44_100, 22_050):
        t = np.arange(rate) / rate
        signal = np.sin(2 * np.pi * 30.0 * t) + np.sin(2 * np.pi * 500.0 * t)
        shares.append(energy_share_below_hz(signal, rate)[0])
    assert max(shares) - min(shares) < 0.02


def test_the_cutoff_moves_when_it_is_told_to() -> None:
    """A tone at 120 Hz sits above a 50 Hz cutoff and below a 200 Hz one.

    The two frequencies are far apart because a 0.05 s Hamming window spreads a
    tone over roughly 40 Hz, so the cutoff is soft to about that width. That is
    wide enough to separate recording-chain rumble from a bird call and too
    wide to call 50 Hz an edge.
    """

    rate = 48_000
    t = np.arange(rate) / rate
    signal = np.sin(2 * np.pi * 120.0 * t)
    assert energy_share_below_hz(signal, rate, cutoff_hz=50.0)[0] < 0.05
    assert energy_share_below_hz(signal, rate, cutoff_hz=200.0)[0] > 0.95
