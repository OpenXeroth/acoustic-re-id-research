"""Spectral subtraction against a per-session noise profile.

Tolkova et al. (2026, PLoS Computational Biology 22: e1013321) denoised right
whale upcalls and the tag noise beside them and found that once the noise-only
clips identified whales at chance, the calls still identified them. This is
the same remedy on the endpoints that carry ambient recordings: every clip,
call or background, has the mean magnitude spectrum of its session's
background recordings subtracted from it, bin by bin, with a spectral floor.

Registered in ``docs/measurement-protocol.md`` (PA-V6):

* the session is the individual within a split, which in the Stowell release
  is one territory in one season;
* the profile is the mean short-time magnitude spectrum over every background
  clip of that session (Hann window, 1024 samples, hop 256);
* the subtracted magnitude is ``max(|X| - alpha * profile, beta * |X|)`` with
  the original phase kept, for alpha 1 and 2 and beta 0.05;
* background clips are processed with the same session profile, so a
  background is reduced, not zeroed, because its profile is an average over
  the session rather than its own spectrum.
"""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any

FRAME = 1024
HOP = 256
ALPHAS: tuple[float, ...] = (1.0, 2.0)
BETA = 0.05


def _stft(signal: Any, sample_rate: int) -> Any:
    import numpy as np
    from scipy.signal import stft

    samples = np.asarray(signal, dtype=np.float64)
    if len(samples) < FRAME:
        samples = np.pad(samples, (0, FRAME - len(samples)))
    _, _, spectrum = stft(
        samples, fs=sample_rate, window="hann", nperseg=FRAME, noverlap=FRAME - HOP
    )
    return spectrum


def noise_profile(signals: Iterable[Any], sample_rate: int) -> Any:
    """Mean magnitude per frequency bin over every frame of every signal."""

    import numpy as np

    total = np.zeros(FRAME // 2 + 1, dtype=np.float64)
    count = 0
    for signal in signals:
        magnitude = np.abs(_stft(signal, sample_rate))
        total += magnitude.sum(axis=1)
        count += magnitude.shape[1]
    if not count:
        raise ValueError("a noise profile needs at least one background clip")
    return total / count


def subtract(
    signal: Any, profile: Any, sample_rate: int, *, alpha: float, beta: float = BETA
) -> Any:
    """The signal with ``alpha`` times the profile removed from every frame."""

    import numpy as np
    from scipy.signal import istft

    length = len(signal)
    spectrum = _stft(signal, sample_rate)
    magnitude = np.abs(spectrum)
    cleaned = np.maximum(magnitude - alpha * profile[:, None], beta * magnitude)
    phase = np.exp(1j * np.angle(spectrum))
    _, restored = istft(
        cleaned * phase, fs=sample_rate, window="hann", nperseg=FRAME, noverlap=FRAME - HOP
    )
    restored = np.asarray(restored, dtype=np.float64)[:length]
    if len(restored) < length:
        restored = np.pad(restored, (0, length - len(restored)))
    return restored.astype(np.float32)
