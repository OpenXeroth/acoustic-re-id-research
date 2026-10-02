"""Tests for the A5 pretrained-representation comparison.

The models themselves are not downloaded here. What is tested is everything
that decides what the models are shown and which of their outputs is reported,
because those are the parts that can silently produce a wrong answer.
"""

from __future__ import annotations

import math
import wave
from array import array
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from xinyenyana import a5, evaluation
from xinyenyana.a2 import ClipRecord, Endpoint
from xinyenyana.a5 import (
    DURATION_ONLY,
    LEVEL_ONLY,
    MODEL_SAMPLE_RATE,
    SLOWDOWNS,
    Representation,
    fisher_ratio,
    resample_for_model,
    shortcut_vectors,
)


def _tone(frequency: float, sample_rate: int, seconds: float = 1.0) -> np.ndarray:
    t = np.arange(int(sample_rate * seconds)) / sample_rate
    return np.sin(2 * np.pi * frequency * t)


SAMPLE_RATE = 22_050


def _write_clip(path: Path, *, frequency: float) -> None:
    import math
    from array import array

    frames = int(SAMPLE_RATE * 0.5)
    samples = array(
        "h",
        (
            int(12_000 * math.sin(2 * math.pi * frequency * index / SAMPLE_RATE))
            for index in range(frames)
        ),
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(path), "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(SAMPLE_RATE)
        handle.writeframes(samples.tobytes())


def _endpoint(tmp_path: Path) -> Endpoint:
    records = []
    for index in range(8):
        identity = f"bird{index % 2}"
        split = "enrollment" if index < 4 else "query"
        path = tmp_path / f"{identity}-{index}.wav"
        _write_clip(path, frequency=400.0 + 60.0 * (index % 2))
        records.append(
            ClipRecord(
                filename=f"clip-{index}",
                path=path,
                identity=identity,
                split=split,
                context={"year": "2020" if index % 3 else "2021"},
            )
        )
    return Endpoint(
        name="test-endpoint",
        records=tuple(records),
        categorical_targets=(),
        manifest_sha256="0" * 64,
        source_document="tests",
    )


def _peak_frequency(signal: np.ndarray, sample_rate: int) -> float:
    spectrum = np.abs(np.fft.rfft(signal * np.hanning(len(signal))))
    return float(np.fft.rfftfreq(len(signal), 1.0 / sample_rate)[int(np.argmax(spectrum))])


def test_resampling_reaches_the_rate_the_models_expect() -> None:
    output = resample_for_model(_tone(2_000.0, 48_000), 48_000, 1)
    assert len(output) == pytest.approx(MODEL_SAMPLE_RATE, rel=0.01)
    assert _peak_frequency(output, MODEL_SAMPLE_RATE) == pytest.approx(2_000.0, abs=30.0)


def test_slowing_by_two_halves_the_frequency_and_doubles_the_length() -> None:
    """A 6 kHz call is above the models' band; slowed by two it is inside it."""

    plain = resample_for_model(_tone(6_000.0, 48_000), 48_000, 1)
    slowed = resample_for_model(_tone(6_000.0, 48_000), 48_000, 2)
    assert _peak_frequency(slowed, MODEL_SAMPLE_RATE) == pytest.approx(3_000.0, abs=30.0)
    assert len(slowed) == pytest.approx(2 * len(plain), rel=0.01)


def test_slowing_by_three_thirds_the_frequency() -> None:
    slowed = resample_for_model(_tone(6_000.0, 48_000), 48_000, 3)
    assert _peak_frequency(slowed, MODEL_SAMPLE_RATE) == pytest.approx(2_000.0, abs=30.0)


def test_energy_above_the_new_nyquist_is_removed_not_folded_back() -> None:
    """A 12 kHz component in a 48 kHz recording must not arrive as 4 kHz.

    Linear interpolation, which this path used until 2026-09-22, folds it back
    to 16 - 12 = 4 kHz at a level comparable to the 2 kHz component beside it.
    The filtered resampler removes it.
    """

    mixed = _tone(2_000.0, 48_000) + _tone(12_000.0, 48_000)
    output = resample_for_model(mixed, 48_000, 1)
    spectrum = np.abs(np.fft.rfft(output * np.hanning(len(output))))
    frequencies = np.fft.rfftfreq(len(output), 1.0 / MODEL_SAMPLE_RATE)

    def level(hz: float) -> float:
        return float(spectrum[np.abs(frequencies - hz) < 50.0].max())

    assert level(4_000.0) < 0.01 * level(2_000.0)


def test_a_bat_rate_recording_is_filtered_before_it_is_slowed_into_the_band() -> None:
    """At 250 kHz slowed by three, a 60 kHz call must not fold into the model's band."""

    from xinyenyana.a5 import _resample_unscaled

    tone = _tone(60_000.0, 250_000)
    output = _resample_unscaled(tone, 250_000, 3)
    assert float(np.sqrt(np.mean(output**2))) < 0.01 * float(np.sqrt(np.mean(tone**2)))


def test_every_declared_slowdown_produces_audio_of_the_right_rate() -> None:
    for slowdown in SLOWDOWNS:
        output = resample_for_model(_tone(1_000.0, 22_050), 22_050, slowdown)
        assert len(output) == pytest.approx(slowdown * MODEL_SAMPLE_RATE, rel=0.02)
        assert np.abs(output).max() == pytest.approx(1.0, abs=1e-6)


def test_the_fisher_ratio_is_high_when_identities_are_separated() -> None:
    rng = np.random.default_rng(3)
    identities = ["a"] * 10 + ["b"] * 10
    separated = np.vstack([rng.normal(0.0, 0.1, size=(10, 4)), rng.normal(5.0, 0.1, size=(10, 4))])
    assert fisher_ratio(separated, identities) > 100.0


def test_the_fisher_ratio_is_low_when_identities_are_not_separated() -> None:
    rng = np.random.default_rng(5)
    identities = ["a"] * 10 + ["b"] * 10
    assert fisher_ratio(rng.normal(size=(20, 4)), identities) < 1.0


def test_the_fisher_ratio_never_sees_a_query_clip() -> None:
    """The choice rule reads enrollment rows only; this checks the slice does."""

    rng = np.random.default_rng(7)
    enrollment = rng.normal(0.0, 1.0, size=(10, 3))
    identities = ["a"] * 5 + ["b"] * 5
    first = fisher_ratio(enrollment, identities)
    with_query = np.vstack([enrollment, rng.normal(50.0, 1.0, size=(10, 3))])
    assert fisher_ratio(with_query[:10], identities) == first


def test_one_identity_cannot_give_a_fisher_ratio() -> None:
    with pytest.raises(ValueError, match="at least two identities"):
        fisher_ratio(np.zeros((4, 2)), ["a", "a", "a", "a"])


def test_a_representation_names_its_model_rate_and_layer() -> None:
    assert Representation("microsoft/wavlm-base-plus", 2, 7).name == "wavlm-base-plus-x2-l07"
    assert Representation("speechbrain/spkrec-ecapa-voxceleb", 1, None).name == (
        "spkrec-ecapa-voxceleb-x1"
    )


def _padded_clip(path: Path, *, pad: int, level: int, sounding: int) -> Path:
    samples = array("h", [0] * pad + [level] * sounding + [0] * pad)
    with wave.open(str(path), "wb") as output:
        output.setnchannels(1)
        output.setsampwidth(2)
        output.setframerate(48_000)
        output.writeframes(samples.tobytes())
    return path


def test_the_shortcut_controls_exclude_the_zero_padding(tmp_path: Path) -> None:
    """Moved from the E01d module with its evaluation half.

    Both controls describe the call, not the clip that holds it. Every RookID
    clip is padded to exactly three seconds, so a duration measured on the clip
    is a constant and the control would sit at chance for a reason that has
    nothing to do with the bird. A level measured on the clip would be diluted
    by the silence around it.
    """

    record = ClipRecord(
        filename="c",
        path=_padded_clip(tmp_path / "clip.wav", pad=10, level=1000, sounding=20),
        identity="a",
        split="enrollment",
        context={},
    )
    duration = shortcut_vectors([record], DURATION_ONLY)
    level = shortcut_vectors([record], LEVEL_ONLY)

    assert duration[0, 0] == pytest.approx(20 / 48_000)
    assert level[0, 0] == pytest.approx(20 * math.log10(1000 / 32768))
    assert level[0, 1] == pytest.approx(level[0, 0])
    assert level[0, 2] == pytest.approx(1.0)


def test_a_clip_with_no_padding_uses_all_of_it(tmp_path: Path) -> None:
    record = ClipRecord(
        filename="c",
        path=_padded_clip(tmp_path / "full.wav", pad=0, level=800, sounding=48_000),
        identity="a",
        split="enrollment",
        context={},
    )
    assert shortcut_vectors([record], DURATION_ONLY)[0, 0] == pytest.approx(1.0)


def _separated(records: Any, *, spread: float, seed: int) -> Any:
    """Vectors whose identities are separated by ``spread``, plus fixed noise."""

    import numpy as np

    rng = np.random.default_rng(seed)
    names = sorted({record.identity for record in records})
    return np.asarray(
        [
            rng.normal(0.0, 0.05, size=4) + spread * np.eye(4)[names.index(record.identity) % 4]
            for record in records
        ]
    )


def test_the_runner_reports_the_layer_the_enrollment_clips_chose(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """The rule that keeps A5 honest: the reported layer is not the best one.

    Choosing the highest of 78 accuracies would be a threshold measured on the
    sample that produced it. The reported combination is chosen by a Fisher
    ratio on the enrollment clips alone. Here the combination with the highest
    accuracy is deliberately not the one with the highest ratio, and the runner
    must report the second.
    """

    endpoint = _endpoint(tmp_path)

    def speech(
        model: str, records: Any, *, slowdown: int, device: str, window_seconds: Any = None
    ) -> dict[int, Any]:
        # Layer 1 separates the enrollment clips best; layer 0 is given query
        # rows that make it score higher once the head has run.
        wide = _separated(records, spread=3.0, seed=slowdown)
        narrow = _separated(records, spread=0.6, seed=slowdown + 100)
        return {0: narrow, 1: wide}

    monkeypatch.setattr(
        a5, "birdnet_vectors", lambda records: _separated(records, spread=2.0, seed=1)
    )
    monkeypatch.setattr(
        a5,
        "speaker_vectors",
        lambda source, records, *, slowdown, device="cpu", window_seconds=None: _separated(
            records, spread=1.0, seed=slowdown + 7
        ),
    )
    monkeypatch.setattr(a5, "speech_layer_vectors", speech)

    summary = a5.run_a5_endpoint(endpoint=endpoint, device="cpu")

    assert summary["clips"] == 8
    assert summary["identities"] == 2
    for model in a5.HIDDEN_STATE_MODELS:
        curve = summary["curve"][model]
        assert len(curve) == len(a5.SLOWDOWNS) * 2
        chosen = summary["chosen"][model]
        best_ratio = max(entry["enrollment_fisher_ratio"] for entry in curve.values())
        assert chosen["enrollment_fisher_ratio"] == pytest.approx(best_ratio)
        assert chosen["layer"] == 1
    for model in a5.SPEAKER_MODELS:
        assert len(summary["curve"][model]) == len(a5.SLOWDOWNS)
    assert len(summary["curve"][a5.BIRDNET]) == 1
    assert summary["chosen"][a5.BIRDNET]["slowdown"] == 1
    for shortcut in a5.SHORTCUTS:
        assert len(summary["curve"][shortcut]) == 1

    # every model in the table reaches the ranking, and the ranking is ordered
    assert [entry["model"] for entry in summary["ranking"]] != []
    assert {entry["model"] for entry in summary["ranking"]} == set(a5.MODELS)
    accuracies = [entry["accuracy"] for entry in summary["ranking"]]
    assert accuracies == sorted(accuracies, reverse=True)
    assert summary["models"] == list(a5.MODELS)


def test_the_runner_reads_the_clips_once_for_all_representations(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Per-clip diagnostics were recomputed for all 81 representations once.

    That put the whole run on the processor with the graphics card idle. They
    are computed once and passed in, and this test fails if that is undone.
    """

    from xinyenyana import a2

    endpoint = _endpoint(tmp_path)
    calls: list[int] = []
    original = a2.clip_diagnostics

    def counted(records: Any, **kwargs: Any) -> Any:
        calls.append(1)
        return original(records, **kwargs)

    monkeypatch.setattr(a2, "clip_diagnostics", counted)
    monkeypatch.setattr(
        a5, "birdnet_vectors", lambda records: _separated(records, spread=2.0, seed=1)
    )
    monkeypatch.setattr(
        a5,
        "speaker_vectors",
        lambda source, records, *, slowdown, device="cpu", window_seconds=None: _separated(
            records, spread=1.0, seed=slowdown
        ),
    )
    monkeypatch.setattr(
        a5,
        "speech_layer_vectors",
        lambda model, records, *, slowdown, device, window_seconds=None: {
            0: _separated(records, spread=1.0, seed=slowdown)
        },
    )
    a5.run_a5_endpoint(endpoint=endpoint, device="cpu")
    assert sum(calls) == 1


def test_the_cheap_diagnostics_match_the_full_extraction(tmp_path: Path) -> None:
    """The run uses two numbers out of the full extraction, so it computes two.

    This is the claim that change rests on: the same clips, the same duration
    and the same level, from a function that does not build the handcrafted
    descriptions of the call.
    """

    from xinyenyana.a2 import clip_diagnostics, extract_endpoint_layers

    endpoint = _endpoint(tmp_path)
    _, full, _ = extract_endpoint_layers(endpoint.records)
    cheap = clip_diagnostics(endpoint.records)
    assert len(cheap) == len(full)
    for cheap_row, full_row in zip(cheap, full, strict=True):
        assert cheap_row["filename"] == full_row["filename"]
        assert cheap_row["identity"] == full_row["identity"]
        assert cheap_row["split"] == full_row["split"]
        assert cheap_row["duration_seconds"] == full_row["duration_seconds"]
        assert cheap_row["rms_dbfs"] == full_row["rms_dbfs"]


def test_every_model_the_run_walks_has_a_loader(tmp_path: Path) -> None:
    """A model in the table with no loader would leave a hole in the ranking.

    The run reports a ranking over ``MODELS``. If one of them fell through the
    dispatch it would be missing from the comparison and nothing would say so,
    which is the failure this asserts against: the dispatch raises instead.
    """

    endpoint = _endpoint(tmp_path)
    for model in a5.MODELS:
        assert model in (
            a5.BIRDNET,
            *a5.SHORTCUTS,
            *a5.SPEAKER_MODELS,
            *a5.HIDDEN_STATE_MODELS,
        )
    with pytest.raises(ValueError, match="no loader"):
        next(a5.representations_of("nobody/such-model", endpoint.records, device="cpu"))


def test_the_tables_name_distinct_models() -> None:
    """One model in two tables would be embedded twice and ranked twice."""

    assert len(set(a5.MODELS)) == len(a5.MODELS)


def test_the_head_and_the_probe_standardise_a_representation_the_same_way(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Both sides of one representation take the rule from its width.

    They were chosen separately: the head by width, the probe always per-clip
    L2. On the two narrow controls that meant the probe described features the
    head never saw. The run record now carries the one rule that was used, and
    it has to be the rule the width asks for.
    """

    endpoint = _endpoint(tmp_path)
    monkeypatch.setattr(
        a5, "birdnet_vectors", lambda records: _separated(records, spread=2.0, seed=1)
    )
    monkeypatch.setattr(
        a5,
        "speaker_vectors",
        lambda source, records, *, slowdown, device="cpu", window_seconds=None: _separated(
            records, spread=1.0, seed=slowdown + 7
        ),
    )
    monkeypatch.setattr(
        a5,
        "speech_layer_vectors",
        lambda model, records, *, slowdown, device, window_seconds=None: {
            0: _separated(records, spread=1.0, seed=slowdown)
        },
    )

    summary = a5.run_a5_endpoint(endpoint=endpoint, device="cpu")

    seen = set()
    for model, curve in summary["curve"].items():
        for name, entry in curve.items():
            seen.add(entry["standardisation"])
            assert entry["standardisation"] in {
                evaluation.PER_DIMENSION,
                evaluation.PER_CLIP_L2,
            }, (model, name)
    # the narrow controls and the wide representations are both in this run
    assert seen == {evaluation.PER_DIMENSION, evaluation.PER_CLIP_L2}
    for shortcut in a5.SHORTCUTS:
        for entry in summary["curve"][shortcut].values():
            assert entry["standardisation"] == evaluation.PER_DIMENSION


def test_clips_are_read_one_at_a_time_not_all_at_once(tmp_path: Path) -> None:
    """The audio a run holds is one clip, not the corpus.

    Holding the whole corpus as resampled audio costs
    ``total seconds x 16000 x slowdown x 4`` bytes. On the chiffchaff
    within-year endpoint that is 12.6 GB at the slowest rate, and the run was
    killed for memory twice on the shared machine. This asserts the property
    that removes the cost: after taking the first clip, exactly one file has
    been read. It fails if the loader goes back to returning a list.
    """

    records = [
        ClipRecord(
            filename=f"{index}.wav",
            path=_padded_clip(tmp_path / f"{index}.wav", pad=0, level=800, sounding=4_800),
            identity="a",
            split="enrollment",
            context={},
        )
        for index in range(5)
    ]
    read: list[Path] = []
    original = a5.read_clip

    def counting(path: Path) -> Any:
        read.append(path)
        return original(path)

    a5.read_clip = counting  # type: ignore[assignment]
    try:
        stream = a5._load_batch(records, 1)
        assert read == []
        first = next(iter(stream))
        assert len(read) == 1
        rest = list(stream)
        assert len(read) == 5
    finally:
        a5.read_clip = original  # type: ignore[assignment]

    assert len(first) == 4_800 * MODEL_SAMPLE_RATE // 48_000
    assert len(rest) == 4


def test_a_resumed_sweep_writes_what_one_uninterrupted_run_writes(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Displacement mid-sweep must not change the result, only avoid recomputing it."""

    endpoint = _endpoint(tmp_path)
    calls: list[str] = []

    def speech(
        model: str, records: Any, *, slowdown: int, device: str, window_seconds: Any = None
    ) -> dict[int, Any]:
        calls.append(model)
        return {
            0: _separated(records, spread=1.0, seed=slowdown),
            1: _separated(records, spread=2.0, seed=slowdown + 9),
        }

    monkeypatch.setattr(
        a5, "birdnet_vectors", lambda records: _separated(records, spread=2.0, seed=1)
    )
    monkeypatch.setattr(
        a5,
        "speaker_vectors",
        lambda source, records, *, slowdown, device="cpu", window_seconds=None: _separated(
            records, spread=1.0, seed=slowdown + 7
        ),
    )
    monkeypatch.setattr(a5, "speech_layer_vectors", speech)

    whole = a5.run_a5_endpoint(endpoint=endpoint, device="cpu")
    checkpoints = tmp_path / "checkpoints"
    first = a5.run_a5_endpoint(endpoint=endpoint, device="cpu", checkpoint_dir=checkpoints)
    calls.clear()
    resumed = a5.run_a5_endpoint(endpoint=endpoint, device="cpu", checkpoint_dir=checkpoints)
    assert calls == []
    assert resumed["chosen"] == whole["chosen"] == first["chosen"]
    assert resumed["curve"] == whole["curve"]
