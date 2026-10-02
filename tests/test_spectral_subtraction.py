import numpy as np
import pytest

from xinyenyana.spectral_subtraction import _stft, noise_profile, subtract


def test_a_steady_tone_is_removed_and_a_different_tone_survives() -> None:
    rate = 16_000
    t = np.arange(rate) / rate
    hum = 0.5 * np.sin(2 * np.pi * 1000 * t)
    call = 0.5 * np.sin(2 * np.pi * 3000 * t)
    profile = noise_profile([hum, hum], rate)
    cleaned = subtract(hum + call, profile, rate, alpha=1.0)
    spectrum = np.abs(np.fft.rfft(cleaned))
    freqs = np.fft.rfftfreq(len(cleaned), 1 / rate)
    at = lambda f: spectrum[np.argmin(np.abs(freqs - f))]  # noqa: E731
    assert at(3000) > 10 * at(1000)  # the floor keeps 5% of the hum
    assert len(cleaned) == len(t)


def test_short_clips_keep_their_length() -> None:
    rate = 22_050
    clip = np.random.default_rng(0).normal(size=300)
    profile = noise_profile([np.random.default_rng(1).normal(size=5000)], rate)
    assert len(subtract(clip, profile, rate, alpha=2.0)) == 300


def test_streamed_profile_preserves_frame_weighting_and_cleaned_audio() -> None:
    rng = np.random.default_rng(17)
    # Unequal durations and levels distinguish a frame mean from a clip mean.
    signals = [rng.normal(size=n) * scale for n, scale in ((300, 0.2), (9000, 1), (32123, 3))]
    old = np.concatenate([np.abs(_stft(x, 16000)) for x in signals], axis=1).mean(axis=1)
    profile = noise_profile((x for x in signals), 16000)
    np.testing.assert_allclose(profile, old, rtol=1e-14, atol=1e-15)
    for alpha in (1.0, 2.0):
        np.testing.assert_array_equal(
            subtract(signals[-1], profile, 16000, alpha=alpha),
            subtract(signals[-1], old, 16000, alpha=alpha),
        )


def test_empty_profile_is_rejected() -> None:
    with pytest.raises(ValueError, match="at least one"):
        noise_profile(iter(()), 16000)
