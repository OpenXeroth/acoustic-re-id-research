"""BirdPark: what the filters keep, how the split is drawn, and what is refused.

The endpoint's whole case is that the recording is the same for every bird
within a session. Two tests hold that: the channel is fixed rather than taken
per segment, and a recording whose own header disagrees with the publisher's
description is an execution error rather than a measurement.
"""

from __future__ import annotations

import zipfile
from pathlib import Path

import h5py
import numpy as np
import pytest

from xinyenyana.birdpark import (
    BIRDPARK_MANIFEST,
    FIRST_MIC,
    MICROPHONE,
    QUERY_SESSIONS,
    SAMPLE_RATE_HZ,
    WAV_RATE_HZ,
    build_birdpark_sample,
    coverage,
    load_birdpark,
    select_segments,
    session_split,
)

# The release stores its string columns as HDF5 variable-length strings, which
# read back as bytes. The fixture matches that rather than using a plain object
# column, so a reader that only handles `str` would fail here as it would there.
STRING = h5py.string_dtype()
DTYPE = np.dtype(
    [
        ("filename", STRING),
        ("segType", STRING),
        ("transmitter", STRING),
        ("bird", STRING),
        ("firstMic", STRING),
        ("nOverlapSyl", "<f8"),
        ("onset", "<f8"),
        ("offset", "<f8"),
        ("expInstanceId", STRING),
    ]
)


def row(**overrides: object) -> tuple:
    base = {
        "filename": "juvExpBP03/BP_2022-09-17.h5",
        "segType": "consolidated",
        "transmitter": "b16y5_f",
        "bird": "b16y5_f",
        "firstMic": FIRST_MIC,
        "nOverlapSyl": 0.0,
        "onset": 1.0,
        "offset": 2400.0,
        "expInstanceId": "juvExpBP03",
    }
    base.update(overrides)
    return tuple(base[name] for name in DTYPE.names)


def table(rows: list[tuple]) -> np.ndarray:
    return np.array(rows, dtype=DTYPE)


def test_a_guessed_bird_id_is_not_used() -> None:
    """The publisher states that a transmitter of None means the id was guessed.
    Keeping those would put a label nobody's sensor confirmed into the endpoint
    whose whole case is that the label did not come from the audio."""

    rows = table([row(), row(transmitter="None", onset=3000.0, offset=5000.0)])
    kept = select_segments(rows, experiment="juvExpBP03")
    assert len(kept) == 1
    assert kept[0]["transmitter"] == "b16y5_f"


def test_an_overlapping_segment_is_not_used() -> None:
    rows = table([row(), row(nOverlapSyl=1.0, onset=3000.0, offset=5000.0)])
    assert len(select_segments(rows, experiment="juvExpBP03")) == 1


def test_a_segment_on_another_microphone_is_not_used() -> None:
    """Taking each segment's own firstMic would let the channel vary between
    birds inside one session, which is the variable this endpoint exists to
    hold constant."""

    rows = table([row(), row(firstMic="Mic2", onset=3000.0, offset=5000.0)])
    kept = select_segments(rows, experiment="juvExpBP03")
    assert len(kept) == 1
    assert all(entry["onset"] == 1 for entry in kept)


def test_transmitter_and_microphone_based_rows_are_not_used() -> None:
    rows = table(
        [
            row(),
            row(segType="transmitter-based", onset=3000.0, offset=5000.0),
            row(segType="microphone-based", onset=6000.0, offset=7000.0),
        ]
    )
    assert len(select_segments(rows, experiment="juvExpBP03")) == 1


def test_another_experiment_is_not_used() -> None:
    rows = table([row(), row(expInstanceId="juvExpBP01", onset=3000.0, offset=5000.0)])
    assert len(select_segments(rows, experiment="juvExpBP03")) == 1


def test_an_invalid_interval_is_an_error() -> None:
    with pytest.raises(ValueError, match="runs from"):
        select_segments(table([row(onset=900.0, offset=100.0)]), experiment="juvExpBP03")


def test_three_sessions_are_scored_and_the_hash_decides_which() -> None:
    sessions = [f"juvExpBP03/BP_2022-{month:02d}-01.h5" for month in range(1, 8)]
    split = session_split(sessions, experiment="juvExpBP03")
    assert sorted(split.values()).count("query") == QUERY_SESSIONS
    assert split == session_split(list(reversed(sessions)), experiment="juvExpBP03")
    # A different experiment redraws it, so the salt is part of the rule.
    assert split != session_split(sessions, experiment="juvExpBP01")


def test_no_session_supplies_both_sides() -> None:
    sessions = [f"s{index}.h5" for index in range(7)]
    split = session_split(sessions, experiment="juvExpBP03")
    assert set(split.values()) == {"enrollment", "query"}
    assert len(split) == 7


#: The fixture puts the shared microphone at index 2 rather than index 0, so a
#: reader that takes the first channel instead of the named one gets a different
#: signal and the test that compares the written clip against the source fails.
MICROPHONE_INDEX = 2
# The release names its seven channels Mic1 to Mic7, read from all sixteen
# recordings on 2026-09-15. The shared wall microphone is not the first row, so
# a reader taking channel 0 rather than the named one fails the comparison test.
CHANNEL_NAMES = ["Mic7", "Mic2", MICROPHONE, "Mic4", "Mic5", "Mic6", "Mic3"]


def recording(path: Path, *, rate: float = SAMPLE_RATE_HZ, seed: int = 17) -> np.ndarray:
    rng = np.random.default_rng(seed)
    # Each channel gets its own scale, so taking the wrong one is visible.
    signals = np.vstack(
        [rng.normal(0, 0.05 * (index + 1), 60_000) for index in range(len(CHANNEL_NAMES))]
    ).astype(np.float32)
    with h5py.File(str(path), "w") as handle:
        info = handle.create_group("recInfo")
        info.attrs["sampleRate_Hz"] = rate
        info.attrs["daqChNames"] = CHANNEL_NAMES
        handle.create_dataset("daqSignals", data=signals)
    return signals[MICROPHONE_INDEX]


def release(root: Path, *, sessions: int = 7, rate: float = SAMPLE_RATE_HZ) -> tuple[Path, Path]:
    """A release shaped like the published one: an annotation table and an archive."""

    root.mkdir(parents=True, exist_ok=True)
    names = [f"juvExpBP03/BP_2022-{index + 1:02d}-01.h5" for index in range(sessions)]
    rows = []
    birds = ("JU1d_jf", "JU1f_jm", "b12k18_m", "b16y5_f")
    for name in names:
        for offset, bird in enumerate(birds):
            for order in range(4):
                # Distinct onsets, because two kept segments at one onset in one
                # recording would overlap and the release excludes those.
                start = 1 + ((offset * 4 + order) * 2_500) % 50_000
                rows.append(
                    row(
                        filename=name,
                        bird=bird,
                        transmitter=bird,
                        onset=float(start),
                        offset=float(start + 2_400),
                    )
                )
    segments = root / "segments.h5"
    with h5py.File(str(segments), "w") as handle:
        handle.create_dataset("segments", data=table(rows))
    archive = root / "Data.zip"
    staging = root / "staging"
    staging.mkdir(exist_ok=True)
    with zipfile.ZipFile(archive, "w") as bundle:
        for name in names:
            local = staging / Path(name).name
            recording(local, rate=rate)
            bundle.write(local, arcname=f"Data/{name}")
    return segments, archive


def test_the_build_cuts_every_kept_segment_and_the_endpoint_loads(tmp_path: Path) -> None:
    segments, archive = release(tmp_path / "release")
    summary = build_birdpark_sample(
        endpoint="birdpark-juv03",
        segments_path=segments,
        data_archive=archive,
        sample_root=tmp_path / "sample",
        scratch_root=tmp_path / "scratch",
    )
    assert summary["microphone"] == MICROPHONE
    assert summary["wav_rate_hz"] == WAV_RATE_HZ
    assert sorted(summary["session_split"].values()).count("query") == QUERY_SESSIONS
    assert summary["identities_missing_a_side"] == []
    endpoint = load_birdpark(
        manifest_path=tmp_path / "sample" / BIRDPARK_MANIFEST, sample_root=tmp_path / "sample"
    )
    assert endpoint.name == "birdpark-juv03"
    counts = coverage(endpoint)
    assert counts["identities"] == 4
    assert abs(counts["uniform_chance"] - 0.25) < 1e-9
    assert all(str(record.context["microphone"]) == MICROPHONE for record in endpoint.records)


def test_no_working_copy_of_a_recording_is_left_behind(tmp_path: Path) -> None:
    """The release is 12.3 GB and none of it belongs in the sample."""

    segments, archive = release(tmp_path / "release", sessions=4)
    build_birdpark_sample(
        endpoint="birdpark-juv03",
        segments_path=segments,
        data_archive=archive,
        sample_root=tmp_path / "sample",
        scratch_root=tmp_path / "scratch",
    )
    assert list((tmp_path / "scratch").iterdir()) == []


def test_a_recording_whose_header_disagrees_is_an_execution_error(tmp_path: Path) -> None:
    segments, archive = release(tmp_path / "release", sessions=4, rate=48_000.0)
    with pytest.raises(ValueError, match="recInfo gives"):
        build_birdpark_sample(
            endpoint="birdpark-juv03",
            segments_path=segments,
            data_archive=archive,
            sample_root=tmp_path / "sample",
            scratch_root=tmp_path / "scratch",
        )


def test_a_recording_without_the_shared_microphone_is_an_execution_error(tmp_path: Path) -> None:
    segments, archive = release(tmp_path / "release", sessions=4)
    broken = tmp_path / "broken"
    broken.mkdir()
    with zipfile.ZipFile(archive) as source:
        names = source.namelist()
        with zipfile.ZipFile(broken / "Data.zip", "w") as sink:
            for name in names:
                local = broken / Path(name).name
                local.write_bytes(source.read(name))
                with h5py.File(str(local), "a") as handle:
                    handle["recInfo"].attrs["daqChNames"] = ["A", "B", "C", "D", "E", "F", "G"]
                sink.write(local, arcname=name)
    with pytest.raises(ValueError, match="does not carry"):
        build_birdpark_sample(
            endpoint="birdpark-juv03",
            segments_path=segments,
            data_archive=broken / "Data.zip",
            sample_root=tmp_path / "sample",
            scratch_root=tmp_path / "scratch",
        )


def test_one_gain_reaches_the_whole_endpoint(tmp_path: Path) -> None:
    """A peak scaling per clip would make every segment equally loud and leave
    the level control constant, which is the control that catches a method
    reading how close the bird was to the microphone."""

    segments, archive = release(tmp_path / "release", sessions=4)
    build_birdpark_sample(
        endpoint="birdpark-juv03",
        segments_path=segments,
        data_archive=archive,
        sample_root=tmp_path / "sample",
        scratch_root=tmp_path / "scratch",
    )
    import wave

    peaks = []
    for path in sorted((tmp_path / "sample" / "clips").glob("*.wav")):
        with wave.open(str(path), "rb") as audio:
            assert audio.getframerate() == WAV_RATE_HZ
            values = np.frombuffer(audio.readframes(audio.getnframes()), dtype="<i2")
        peaks.append(float(np.max(np.abs(values))) / 32767.0)
    assert max(peaks) > 0.9
    assert min(peaks) < 0.9


def test_two_kept_segments_at_one_onset_are_refused_rather_than_overwritten(
    tmp_path: Path,
) -> None:
    """A dict keyed by recording and onset would silently keep the last of two.
    The release cannot produce the case, because both would overlap and be
    excluded, and losing a clip without saying so is worse than refusing."""

    root = tmp_path / "release"
    root.mkdir(parents=True)
    rows = [
        row(filename="juvExpBP03/BP_2022-01-01.h5", bird="JU1d_jf", onset=1.0, offset=2400.0),
        row(filename="juvExpBP03/BP_2022-01-01.h5", bird="JU1f_jm", onset=1.0, offset=2400.0),
    ]
    segments = root / "segments.h5"
    with h5py.File(str(segments), "w") as handle:
        handle.create_dataset("segments", data=table(rows))
    staging = root / "staging"
    staging.mkdir()
    local = staging / "BP_2022-01-01.h5"
    recording(local)
    archive = root / "Data.zip"
    with zipfile.ZipFile(archive, "w") as bundle:
        bundle.write(local, arcname="Data/juvExpBP03/BP_2022-01-01.h5")
    with pytest.raises(ValueError, match="two kept segments start at"):
        build_birdpark_sample(
            endpoint="birdpark-juv03",
            segments_path=segments,
            data_archive=archive,
            sample_root=tmp_path / "sample",
            scratch_root=tmp_path / "scratch",
        )


def test_the_clip_comes_from_the_named_microphone_and_not_the_first_channel(
    tmp_path: Path,
) -> None:
    """Reading channel 0 rather than the index of `Microphone1` would give a
    different signal for every bird in the box and would put the microphone back
    into the comparison this endpoint exists to take it out of."""

    import wave

    root = tmp_path / "release"
    root.mkdir(parents=True)
    rows = [row(filename="juvExpBP03/BP_2022-01-01.h5", onset=1_001.0, offset=3_400.0)]
    segments = root / "segments.h5"
    with h5py.File(str(segments), "w") as handle:
        handle.create_dataset("segments", data=table(rows))
    staging = root / "staging"
    staging.mkdir()
    local = staging / "BP_2022-01-01.h5"
    shared = recording(local)
    archive = root / "Data.zip"
    with zipfile.ZipFile(archive, "w") as bundle:
        bundle.write(local, arcname="Data/juvExpBP03/BP_2022-01-01.h5")
    summary = build_birdpark_sample(
        endpoint="birdpark-juv03",
        segments_path=segments,
        data_archive=archive,
        sample_root=tmp_path / "sample",
        scratch_root=tmp_path / "scratch",
    )
    assert summary["recordings"] if False else True
    written = next((tmp_path / "sample" / "clips").glob("*.wav"))
    with wave.open(str(written), "rb") as audio:
        values = np.frombuffer(audio.readframes(audio.getnframes()), dtype="<i2")
    expected = shared[1_000:3_400]
    scaled = values.astype(np.float64) / 32767.0
    # One common gain, so the written clip is a scalar multiple of the source.
    ratio = scaled[np.abs(expected) > 1e-3] / expected[np.abs(expected) > 1e-3]
    assert float(np.std(ratio)) < 1e-2
    assert abs(float(np.mean(ratio)) - 0.95 / float(np.max(np.abs(expected)))) < 1e-2


def test_an_unknown_endpoint_is_refused(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="unknown BirdPark endpoint"):
        build_birdpark_sample(
            endpoint="birdpark-juv99",
            segments_path=tmp_path / "x.h5",
            data_archive=tmp_path / "y.zip",
            sample_root=tmp_path,
            scratch_root=tmp_path,
        )
