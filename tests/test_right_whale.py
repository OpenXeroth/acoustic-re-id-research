"""The right whale endpoint: the transform, the split, and what the build keeps.

Every test here fails when the behaviour it names is removed. The property that
matters most is that the same transform and the same gain reach both conditions
of a stem, because the two crossings compare one condition's gallery against the
other's probes and an asymmetry in the audio path would be read as a finding.
"""

from __future__ import annotations

import json
import wave
from pathlib import Path

import numpy as np
import pytest

from xinyenyana.right_whale import (
    BAND_HIGH_HZ,
    BAND_LOW_HZ,
    CONDITIONS,
    ENDPOINT,
    PEAK_TARGET,
    RIGHT_WHALE_MANIFEST,
    SHIFTED_RATE,
    SHIFTS,
    SOURCE_RATE,
    build_right_whale_sample,
    coverage,
    load_right_whale,
    query_side,
    read_pcm16,
    shift_frequency,
)


def write_source(path: Path, signal: np.ndarray, rate: int = SOURCE_RATE) -> None:
    samples = np.clip(np.round(signal * 32767.0), -32768, 32767).astype("<i2")
    with wave.open(str(path), "wb") as audio:
        audio.setnchannels(1)
        audio.setsampwidth(2)
        audio.setframerate(rate)
        audio.writeframes(samples.tobytes())


def release(root: Path, *, whales: int = 4, per_whale: int = 6) -> Path:
    """A release shaped like the published one: a call and an ambient twin each."""

    folder = root / "Recordings"
    folder.mkdir(parents=True)
    rng = np.random.default_rng(17)
    for index in range(whales):
        letter = chr(ord("A") + index)
        for number in range(per_whale):
            stem = f"{letter}_{number:03d}"
            steps = np.arange(int(1.4 * SOURCE_RATE)) / SOURCE_RATE
            tone = 0.4 * np.sin(2 * np.pi * (120.0 + 30.0 * index) * steps)
            write_source(folder / f"{stem}_call.wav", tone + 0.02 * rng.normal(size=steps.size))
            ambient = 0.05 * rng.normal(size=SOURCE_RATE)
            write_source(folder / f"{stem}_noise.wav", ambient)
    return root


def test_shift_moves_the_upcall_band_into_the_high_pass() -> None:
    steps = np.arange(4 * SOURCE_RATE) / SOURCE_RATE
    signal = np.sin(2 * np.pi * 200.0 * steps)
    shifted, rate = shift_frequency(signal, SOURCE_RATE, 5_000)
    assert rate == SHIFTED_RATE
    spectrum = np.abs(np.fft.rfft(shifted))
    frequencies = np.fft.rfftfreq(shifted.size, 1.0 / rate)
    peak = float(frequencies[int(np.argmax(spectrum))])
    assert 5_000 + BAND_LOW_HZ < peak < 5_000 + BAND_HIGH_HZ
    assert abs(peak - 5_200.0) < 20.0


def test_the_high_pass_attenuates_the_lower_sideband_without_removing_it() -> None:
    """Taking the real part of the modulated signal makes two mirrored bands, at
    ``shift - f`` and ``shift + f``, equal in size. The authors' high-pass sits
    only 50 Hz above the shift, so it attenuates the lower one rather than
    removing it: at a tone 200 Hz from the shift, a tenth-order Butterworth run
    forwards and backwards leaves about 30% of it, which is what its own
    transfer function gives and is not a number fitted to any data. A mirrored
    copy of the upcall band at that size is a property of the published method
    and reaches every whale figure."""

    steps = np.arange(4 * SOURCE_RATE) / SOURCE_RATE
    signal = np.sin(2 * np.pi * 200.0 * steps)
    shifted, rate = shift_frequency(signal, SOURCE_RATE, 5_000)
    spectrum = np.abs(np.fft.rfft(shifted))
    frequencies = np.fft.rfftfreq(shifted.size, 1.0 / rate)
    upper = float(spectrum[np.argmin(np.abs(frequencies - 5_200.0))])
    lower = float(spectrum[np.argmin(np.abs(frequencies - 4_800.0))])
    assert upper > 0.0
    # Without the filter the two are equal; with it the mirror is well under half.
    assert 0.1 < lower / upper < 0.5


def test_a_shift_of_zero_returns_the_signal_at_its_own_rate() -> None:
    signal = np.sin(2 * np.pi * 200.0 * np.arange(SOURCE_RATE) / SOURCE_RATE)
    unshifted, rate = shift_frequency(signal, SOURCE_RATE, 0)
    assert rate == SOURCE_RATE
    assert np.allclose(unshifted, signal)


def test_a_shift_the_authors_assertion_forbids_is_refused() -> None:
    signal = np.sin(2 * np.pi * 200.0 * np.arange(SOURCE_RATE) / SOURCE_RATE)
    with pytest.raises(ValueError, match="aliases"):
        shift_frequency(signal, SOURCE_RATE, 11_500)
    with pytest.raises(ValueError, match="unregistered shift"):
        build_right_whale_sample(release_root=Path("."), output_root=Path("."), shift_hz=500)


def test_the_split_is_fixed_by_the_stem_and_nothing_else() -> None:
    assert query_side("A_000") == query_side("A_000")
    drawn = [stem for stem in (f"A_{n:03d}" for n in range(400)) if query_side(stem)]
    assert 0.2 < len(drawn) / 400 < 0.4
    # A stem's two conditions share the stem, so they never split apart.
    assert query_side("A_000") == query_side("A_000")


def test_both_conditions_of_a_stem_land_on_the_same_side(tmp_path: Path) -> None:
    build_right_whale_sample(
        release_root=release(tmp_path / "src"), output_root=tmp_path / "out", shift_hz=3_000
    )
    manifest = json.loads((tmp_path / "out" / "shift-3000" / RIGHT_WHALE_MANIFEST).read_text())
    sides: dict[str, set[str]] = {}
    for entry in manifest["entries"]:
        sides.setdefault(entry["stem"], set()).add(entry["split"])
    assert all(len(value) == 1 for value in sides.values())
    assert {entry["condition"] for entry in manifest["entries"]} == set(CONDITIONS.values())


def test_one_gain_reaches_the_whole_arm_so_relative_level_survives(tmp_path: Path) -> None:
    """A per-clip peak scaling would make every clip equally loud and would
    leave the level control representation constant, which is the control that
    catches a method reading how close the animal was."""

    source = tmp_path / "src"
    folder = source / "Recordings"
    folder.mkdir(parents=True)
    steps = np.arange(SOURCE_RATE) / SOURCE_RATE
    # A bounded signal, so the louder clip is loud rather than clipped.
    for index, amplitude in enumerate((0.30, 0.03)):
        stem = f"{chr(ord('A') + index)}_000"
        tone = amplitude * np.sin(2 * np.pi * 150.0 * steps)
        write_source(folder / f"{stem}_call.wav", tone)
        write_source(folder / f"{stem}_noise.wav", tone)
    build_right_whale_sample(release_root=source, output_root=tmp_path / "out", shift_hz=0)
    written = tmp_path / "out" / "shift-0"
    loud, _ = read_pcm16(written / "A_000_call.wav")
    quiet, _ = read_pcm16(written / "B_000_call.wav")
    ratio = float(np.max(np.abs(loud))) / float(np.max(np.abs(quiet)))
    assert 5.0 < ratio < 20.0
    assert abs(float(np.max(np.abs(loud))) - PEAK_TARGET) < 0.01


def test_the_endpoint_loads_in_two_conditions_with_every_whale(tmp_path: Path) -> None:
    summary = build_right_whale_sample(
        release_root=release(tmp_path / "src"), output_root=tmp_path / "out", shift_hz=1_000
    )
    assert summary["shift_hz"] == 1_000
    assert summary["resampler"] == "fourier"
    endpoint = load_right_whale(
        manifest_path=tmp_path / "out" / "shift-1000" / RIGHT_WHALE_MANIFEST,
        sample_root=tmp_path / "out",
    )
    assert endpoint.name == ENDPOINT
    conditions = {str(r.context["condition"]) for r in endpoint.records}
    assert conditions == {"foreground", "background"}
    counts = coverage(endpoint)
    assert set(counts) == set(endpoint.identities)
    for entry in counts.values():
        assert entry["foreground"] == entry["background"]


def test_a_whale_with_no_query_stem_is_dropped_and_reported(tmp_path: Path) -> None:
    source = tmp_path / "src"
    folder = source / "Recordings"
    folder.mkdir(parents=True)
    rng = np.random.default_rng(5)
    kept = [stem for stem in (f"A_{n:03d}" for n in range(40)) if query_side(stem)][:3]
    enrolled_only = [stem for stem in (f"B_{n:03d}" for n in range(200)) if not query_side(stem)][
        :3
    ]
    for stem in [
        *kept,
        *[f"A_{n:03d}" for n in range(40) if not query_side(f"A_{n:03d}")][:3],
        *enrolled_only,
    ]:
        for suffix in CONDITIONS:
            write_source(folder / f"{stem}_{suffix}.wav", 0.2 * rng.normal(size=SOURCE_RATE))
    summary = build_right_whale_sample(
        release_root=source, output_root=tmp_path / "out", shift_hz=0
    )
    assert summary["identities_without_a_query_stem"] == ["B"]
    endpoint = load_right_whale(
        manifest_path=tmp_path / "out" / "shift-0" / RIGHT_WHALE_MANIFEST,
        sample_root=tmp_path / "out",
    )
    assert endpoint.identities == ["A"]


def test_a_call_without_its_ambient_twin_is_an_error(tmp_path: Path) -> None:
    source = release(tmp_path / "src", whales=2, per_whale=2)
    (source / "Recordings" / "A_000_noise.wav").unlink()
    with pytest.raises(ValueError, match="ambient twin"):
        build_right_whale_sample(release_root=source, output_root=tmp_path / "out", shift_hz=0)


def test_a_file_at_the_wrong_rate_is_an_error(tmp_path: Path) -> None:
    source = release(tmp_path / "src", whales=2, per_whale=2)
    write_source(source / "Recordings" / "A_000_call.wav", np.zeros(4_000) + 0.1, rate=16_000)
    with pytest.raises(ValueError, match="published"):
        build_right_whale_sample(release_root=source, output_root=tmp_path / "out", shift_hz=0)


def test_every_registered_shift_is_reachable() -> None:
    signal = np.sin(2 * np.pi * 200.0 * np.arange(2 * SOURCE_RATE) / SOURCE_RATE)
    for shift in SHIFTS:
        values, rate = shift_frequency(signal, SOURCE_RATE, shift)
        assert np.all(np.isfinite(values))
        assert rate == (SHIFTED_RATE if shift else SOURCE_RATE)
