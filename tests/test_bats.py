"""Tests for reading the bat release's attributions and cutting its spans."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from xinyenyana.bats import (
    ACROSS_TREATMENT_IDENTITIES,
    BAT_ENDPOINTS,
    BAT_MANIFEST,
    annotated_spans,
    attributed_emitter,
    build_bat_sample,
    coverage,
    load_bats,
)


def _write(path: Path, values: object, rate: int) -> None:
    import wave
    from array import array

    samples = array("h", values)
    with wave.open(str(path), "wb") as stream:
        stream.setnchannels(1)
        stream.setsampwidth(2)
        stream.setframerate(rate)
        stream.writeframes(samples.tobytes())


def _read(path: Path) -> tuple[list[int], int]:
    import wave
    from array import array

    with wave.open(str(path), "rb") as stream:
        rate = stream.getframerate()
        samples = array("h")
        samples.frombytes(stream.readframes(stream.getnframes()))
    return list(samples), rate


ANNOTATION_HEADER = "FileID,Emitter,Addressee,Context,Start sample,End sample"
FILE_HEADER = "FileID,Treatment ID,File name,File folder,Recording channel,Recording time"
RATE = 250_000
SOURCE_SAMPLES = 4_000


def _release(tmp_path: Path, rows: list[tuple[str, str, str, str, str, int, int]]) -> Path:
    """Rows are (file id, emitter, treatment, folder, date, start, end)."""

    root = tmp_path / "prat"
    root.mkdir()
    annotations = [ANNOTATION_HEADER]
    files = [FILE_HEADER]
    written = set()
    for file_id, emitter, treatment, folder, date, start, end in rows:
        name = f"{file_id}.WAV"
        annotations.append(f"{file_id},{emitter},0,11,{start},{end}")
        if file_id not in written:
            files.append(f"{file_id},{treatment},{name},{folder},4,{date} 01:02:03")
            clip = root / "unpacked" / folder / name
            clip.parent.mkdir(parents=True, exist_ok=True)
            _write(clip, range(SOURCE_SAMPLES), RATE)
            written.add(file_id)
    (root / "Annotations.csv").write_text("\n".join(annotations) + "\n")
    (root / "FileInfo.csv").write_text("\n".join(files) + "\n")
    return root


def _both_periods() -> list[tuple[str, str, str, str, str, int, int]]:
    rows = []
    counter = 0
    for identity in ACROSS_TREATMENT_IDENTITIES:
        for treatment, date in (("17", "2013-01-04"), ("18", "2013-05-20"), ("20", "2014-02-10")):
            counter += 1
            rows.append((str(counter), identity, treatment, "files220", date, 1, 1000))
    return rows


def _build(
    tmp_path: Path, rows: list[tuple[str, str, str, str, str, int, int]]
) -> tuple[Path, dict]:
    root = _release(tmp_path, rows)
    sample = tmp_path / "sample"
    summary = build_bat_sample(
        endpoint="bat-acrosstreatment",
        annotations_path=root / "Annotations.csv",
        file_info_path=root / "FileInfo.csv",
        audio_root=root / "unpacked",
        sample_root=sample,
    )
    return sample, summary


def _load(sample: Path):
    return load_bats(manifest_path=sample / BAT_MANIFEST, sample_root=sample)


# --- what counts as an attribution ---------------------------------------------


def test_a_minus_sign_is_not_an_identity() -> None:
    """The authors define it as the pair being known and their roles in doubt."""

    assert attributed_emitter("215") == "215"
    assert attributed_emitter("-215") is None
    assert attributed_emitter("0") is None
    assert attributed_emitter("") is None


def test_ambiguous_and_unknown_emitters_are_left_out(tmp_path: Path) -> None:
    rows = _both_periods()
    rows.append(("900", "-215", "17", "files220", "2013-01-04", 1, 1000))
    rows.append(("901", "0", "17", "files220", "2013-01-04", 1, 1000))
    root = _release(tmp_path, rows)
    spans = annotated_spans(
        endpoint="bat-acrosstreatment",
        annotations_path=root / "Annotations.csv",
        file_info_path=root / "FileInfo.csv",
    )
    assert len(spans) == len(_both_periods())


# --- one recording can hold several vocalisations ------------------------------


def test_two_annotations_on_one_recording_become_two_clips(tmp_path: Path) -> None:
    """A vocalisation is a span, not a file, so the same file yields both."""

    rows = _both_periods()
    rows.append(("1", "207", "17", "files220", "2013-01-04", 1001, 2000))
    sample, summary = _build(tmp_path, rows)
    endpoint = _load(sample)
    assert summary["annotated_spans"] == len(rows)
    assert summary["clips"] == len(rows)
    names = [record.filename for record in endpoint.records if record.filename.endswith(".wav")]
    assert len(names) == len(set(names))
    assert any(name.endswith("-1-1000.wav") for name in names)
    assert any(name.endswith("-1001-2000.wav") for name in names)


def test_a_clip_carries_exactly_the_samples_the_annotation_names(tmp_path: Path) -> None:
    sample, _ = _build(tmp_path, _both_periods())
    endpoint = _load(sample)
    first = next(r for r in endpoint.records if r.filename.endswith("-1-1000.wav"))
    samples, rate = _read(first.path)
    assert rate == RATE
    assert len(samples) == 1000
    assert samples[0] == 0
    assert samples[-1] == 999


# --- the split the release carries ---------------------------------------------


def test_the_earlier_treatments_enroll_and_the_later_one_is_scored(tmp_path: Path) -> None:
    sample, _ = _build(tmp_path, _both_periods())
    endpoint = _load(sample)
    assert endpoint.identities == sorted(ACROSS_TREATMENT_IDENTITIES)
    enrolled = {r.context["treatment"] for r in endpoint.records if r.split == "enrollment"}
    queried = {r.context["treatment"] for r in endpoint.records if r.split == "query"}
    assert enrolled == {"17", "18"}
    assert queried == {"20"}


def test_a_bat_or_treatment_outside_the_endpoint_is_left_out(tmp_path: Path) -> None:
    rows = _both_periods()
    rows.append(("902", "230", "17", "files220", "2013-01-04", 1, 1000))
    rows.append(("903", "215", "9", "files220", "2012-10-04", 1, 1000))
    sample, _ = _build(tmp_path, rows)
    endpoint = _load(sample)
    assert "230" not in endpoint.identities
    assert "9" not in {r.context["treatment"] for r in endpoint.records}


# --- what a partial unpack does ------------------------------------------------


def test_a_recording_that_is_not_unpacked_is_counted(tmp_path: Path) -> None:
    root = _release(tmp_path, _both_periods())
    (root / "unpacked" / "files220" / "3.WAV").unlink()
    sample = tmp_path / "sample"
    summary = build_bat_sample(
        endpoint="bat-acrosstreatment",
        annotations_path=root / "Annotations.csv",
        file_info_path=root / "FileInfo.csv",
        audio_root=root / "unpacked",
        sample_root=sample,
    )
    assert summary["recordings_not_unpacked"] == 1
    assert summary["clips"] == len(_both_periods()) - 1
    with pytest.raises(ValueError, match="has no query clip for bat"):
        _load(sample)


def test_a_span_past_the_end_of_its_recording_is_counted_not_padded(tmp_path: Path) -> None:
    rows = _both_periods()
    rows.append(
        ("904", "207", "17", "files220", "2013-01-04", SOURCE_SAMPLES + 10, SOURCE_SAMPLES + 20)
    )
    sample, summary = _build(tmp_path, rows)
    assert summary["annotations_past_the_end_of_their_recording"] == 1
    assert summary["clips"] == len(_both_periods())


def test_a_manifest_that_is_not_there_says_how_to_build_it(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="build-bat-sample"):
        load_bats(manifest_path=tmp_path / BAT_MANIFEST, sample_root=tmp_path)


def test_an_unknown_endpoint_is_refused(tmp_path: Path) -> None:
    assert set(BAT_ENDPOINTS) == {"bat-acrosstreatment"}
    with pytest.raises(ValueError, match="unknown bat endpoint"):
        annotated_spans(
            endpoint="bat",
            annotations_path=tmp_path / "Annotations.csv",
            file_info_path=tmp_path / "FileInfo.csv",
        )


def test_a_span_that_runs_backwards_is_refused(tmp_path: Path) -> None:
    rows = _both_periods()
    rows.append(("905", "207", "17", "files220", "2013-01-04", 900, 100))
    root = _release(tmp_path, rows)
    with pytest.raises(ValueError, match="which is not a span"):
        annotated_spans(
            endpoint="bat-acrosstreatment",
            annotations_path=root / "Annotations.csv",
            file_info_path=root / "FileInfo.csv",
        )


# --- what the manifest and the coverage report say -----------------------------


def test_the_manifest_records_the_two_tables_it_was_built_from(tmp_path: Path) -> None:
    sample, _ = _build(tmp_path, _both_periods())
    manifest = json.loads((sample / BAT_MANIFEST).read_text())
    assert len(manifest["annotations_sha256"]) == 64
    assert len(manifest["file_info_sha256"]) == 64
    assert manifest["endpoint"] == "bat-acrosstreatment"


def test_coverage_reports_each_bat_on_both_sides(tmp_path: Path) -> None:
    sample, _ = _build(tmp_path, _both_periods())
    counted = coverage(_load(sample))
    assert set(counted) == set(ACROSS_TREATMENT_IDENTITIES)
    for entry in counted.values():
        assert entry == {"enrollment": 2, "query": 1, "recording_days": 3}
