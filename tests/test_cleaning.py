"""Tests for the A3 cleaning steps.

Each names an outcome the step must produce or refuse. The band tests build
signals whose content is known, so the computed band can be checked against the
frequency that was actually put there rather than against another run of the
same code.
"""

from __future__ import annotations

import wave
from array import array
from pathlib import Path

import numpy as np
import pytest

from xinyenyana.cleaning import (
    Band,
    band_from_clips,
    band_limit,
    clean,
    spectral_subtract,
    trim_to_notes,
)

SAMPLE_RATE = 22_050


def _write(path: Path, samples: np.ndarray) -> Path:
    scaled = np.clip(samples / max(float(np.abs(samples).max()), 1e-9) * 0.8, -1, 1)
    path.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(path), "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(SAMPLE_RATE)
        handle.writeframes(array("h", (int(v * 32_000) for v in scaled)).tobytes())
    return path


def _tone(frequency: float, seconds: float = 1.0) -> np.ndarray:
    t = np.arange(int(SAMPLE_RATE * seconds)) / SAMPLE_RATE
    return np.sin(2 * np.pi * frequency * t)


def _notes(frequency: float, *, count: int, note: float, gap: float) -> np.ndarray:
    pieces = []
    for _ in range(count):
        pieces.append(_tone(frequency, note))
        pieces.append(np.zeros(int(SAMPLE_RATE * gap)))
    return np.concatenate(pieces)


# --- the band ---------------------------------------------------------------


def test_the_band_brackets_the_frequency_that_is_in_the_clips(tmp_path: Path) -> None:
    paths = [_write(tmp_path / f"c{i}.wav", _tone(2_000.0)) for i in range(4)]
    band = band_from_clips(paths)
    assert band.low_hz <= 2_000.0 <= band.high_hz
    assert band.high_hz - band.low_hz < 400.0
    assert band.clips == 4


def test_two_tones_widen_the_band_to_hold_both(tmp_path: Path) -> None:
    narrow = band_from_clips([_write(tmp_path / "n.wav", _tone(2_000.0))])
    wide = band_from_clips([_write(tmp_path / "w.wav", _tone(800.0) + _tone(5_000.0))])
    assert wide.high_hz > narrow.high_hz
    assert wide.low_hz < narrow.low_hz
    assert wide.low_hz <= 800.0 and wide.high_hz >= 5_000.0


def test_a_loud_clip_does_not_decide_the_band_on_its_own(tmp_path: Path) -> None:
    quiet = [_write(tmp_path / f"q{i}.wav", _tone(1_000.0)) for i in range(5)]
    loud = _write(tmp_path / "loud.wav", _tone(6_000.0) * 40.0)
    band = band_from_clips([*quiet, loud])
    assert band.low_hz <= 1_000.0 <= band.high_hz


def test_mixed_sample_rates_are_refused(tmp_path: Path) -> None:
    first = _write(tmp_path / "a.wav", _tone(1_000.0))
    second = tmp_path / "b.wav"
    with wave.open(str(second), "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(48_000)
        handle.writeframes(array("h", [0] * 48_000).tobytes())
    with pytest.raises(ValueError, match="mixed sample rates"):
        band_from_clips([first, second])


def test_band_limiting_removes_what_is_outside_and_keeps_what_is_inside() -> None:
    signal = _tone(1_000.0) + _tone(7_000.0)
    band = Band(low_hz=500.0, high_hz=2_000.0, clips=1)
    limited = band_limit(signal, SAMPLE_RATE, band)
    spectrum = np.abs(np.fft.rfft(limited))
    frequencies = np.fft.rfftfreq(len(limited), 1.0 / SAMPLE_RATE)
    inside = spectrum[np.argmin(np.abs(frequencies - 1_000.0))]
    outside = spectrum[np.argmin(np.abs(frequencies - 7_000.0))]
    assert outside < inside * 1e-6


# --- trimming ---------------------------------------------------------------


def test_trimming_removes_the_gaps_and_counts_the_notes() -> None:
    signal = _notes(1_500.0, count=3, note=0.2, gap=0.2)
    trimmed, notes = trim_to_notes(signal, SAMPLE_RATE)
    assert notes == 3
    assert len(trimmed) < len(signal)
    assert len(trimmed) == pytest.approx(3 * 0.2 * SAMPLE_RATE, rel=0.2)


def test_a_clip_with_no_note_is_returned_unchanged_and_says_so() -> None:
    silence = np.zeros(SAMPLE_RATE)
    trimmed, notes = trim_to_notes(silence, SAMPLE_RATE)
    assert notes == 0
    assert len(trimmed) == len(silence)


# --- noise subtraction ------------------------------------------------------


def test_subtraction_lowers_the_noise_between_notes_more_than_the_notes() -> None:
    rng = np.random.default_rng(19)
    clean_signal = _notes(1_500.0, count=3, note=0.25, gap=0.25)
    noisy = clean_signal + rng.normal(scale=0.05, size=len(clean_signal))
    output, noise_frames = spectral_subtract(noisy, SAMPLE_RATE)
    assert noise_frames >= 4
    gap = slice(int(0.28 * SAMPLE_RATE), int(0.45 * SAMPLE_RATE))
    note = slice(int(0.05 * SAMPLE_RATE), int(0.20 * SAMPLE_RATE))
    gap_before = float(np.sqrt(np.mean(noisy[gap] ** 2)))
    gap_after = float(np.sqrt(np.mean(output[gap] ** 2)))
    note_before = float(np.sqrt(np.mean(noisy[note] ** 2)))
    note_after = float(np.sqrt(np.mean(output[note] ** 2)))
    assert gap_after < gap_before * 0.5
    assert note_after > note_before * 0.5


def test_a_clip_with_no_detected_note_is_returned_unchanged() -> None:
    """A continuous tone has no note: nothing in it is known not to be call.

    Estimating noise from every frame would subtract the tone from itself.
    """

    continuous = _tone(1_500.0, seconds=1.0)
    output, noise_frames = spectral_subtract(continuous, SAMPLE_RATE)
    assert noise_frames == 0
    assert np.array_equal(output, continuous)


def test_a_clip_with_too_few_non_note_frames_is_returned_unchanged() -> None:
    """One long note with a sliver of silence gives a median of two frames."""

    signal = np.concatenate([np.zeros(int(SAMPLE_RATE * 0.02)), _tone(1_500.0, seconds=0.9) * 20.0])
    output, noise_frames = spectral_subtract(signal, SAMPLE_RATE)
    assert noise_frames == 0
    assert np.array_equal(output, signal)


# --- the pipeline -----------------------------------------------------------


def test_the_steps_run_in_the_fixed_order_and_report_what_they_did() -> None:
    signal = _notes(1_500.0, count=3, note=0.2, gap=0.2)
    band = Band(low_hz=500.0, high_hz=4_000.0, clips=10)
    output, diagnostics = clean(signal, SAMPLE_RATE, steps=("trim", "band", "denoise"), band=band)
    assert diagnostics["steps"] == ["trim", "band", "denoise"]
    assert diagnostics["notes_kept"] == 3
    assert diagnostics["band"] == band.as_dict()
    assert "noise_frames" in diagnostics
    assert len(output) < len(signal)


def test_no_steps_leaves_the_signal_alone() -> None:
    signal = _tone(1_000.0)
    output, diagnostics = clean(signal, SAMPLE_RATE, steps=(), band=None)
    assert diagnostics == {"steps": []}
    assert output is signal


def test_an_unknown_step_is_refused() -> None:
    with pytest.raises(ValueError, match="unknown cleaning steps"):
        clean(_tone(1_000.0), SAMPLE_RATE, steps=("sharpen",), band=None)


def test_band_limiting_without_a_band_is_refused() -> None:
    with pytest.raises(ValueError, match="needs a band"):
        clean(_tone(1_000.0), SAMPLE_RATE, steps=("band",), band=None)
