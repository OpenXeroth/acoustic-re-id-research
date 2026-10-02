"""Tests for the physical-constraint audit."""

from __future__ import annotations

from pathlib import Path

import pytest

from xinyenyana.a2 import ClipRecord, Endpoint
from xinyenyana.constraints import (
    ClipEvent,
    attach_song_times,
    audit_constraints,
    endpoint_events,
    impossible_pairs,
)

HOUR = 3600.0


def _event(identity: str, x: float, y: float, seconds: float) -> ClipEvent:
    return ClipEvent(identity=identity, easting=x, northing=y, seconds=seconds)


# --- what the physics excludes ----------------------------------------------


def test_two_places_at_one_instant_is_impossible_at_any_speed() -> None:
    """The constraint that does the work is being in two places at once."""

    events = [_event("a", 0.0, 0.0, 100.0), _event("b", 1000.0, 0.0, 100.0)]
    for speed in (5.0, 30.0, 300.0):
        assert impossible_pairs(events, speed_ms=speed) == [(0, 1)]


def test_a_gap_long_enough_to_fly_the_distance_is_not_excluded() -> None:
    """1,000 m at 10 m/s needs 100 s; an hour is ample."""

    events = [_event("a", 0.0, 0.0, 0.0), _event("b", 1000.0, 0.0, HOUR)]
    assert impossible_pairs(events, speed_ms=10.0) == []


def test_the_boundary_is_the_distance_the_animal_can_cover() -> None:
    near = [_event("a", 0.0, 0.0, 0.0), _event("b", 999.0, 0.0, 100.0)]
    far = [_event("a", 0.0, 0.0, 0.0), _event("b", 1001.0, 0.0, 100.0)]
    assert impossible_pairs(near, speed_ms=10.0) == []
    assert impossible_pairs(far, speed_ms=10.0) == [(0, 1)]


def test_two_detections_at_one_position_carry_no_spatial_evidence() -> None:
    """Whatever the interval, one position says nothing about how many animals."""

    events = [_event("a", 5.0, 5.0, 0.0), _event("b", 5.0, 5.0, 0.0)]
    assert impossible_pairs(events, speed_ms=10.0) == []


def test_a_speed_of_zero_is_refused() -> None:
    with pytest.raises(ValueError, match="positive"):
        impossible_pairs([_event("a", 0.0, 0.0, 0.0)], speed_ms=0.0)


# --- the audit ---------------------------------------------------------------


def test_the_audit_reports_labels_that_contradict_the_physics() -> None:
    """One label on two simultaneous detections a kilometre apart is a fault."""

    events = [_event("same", 0.0, 0.0, 10.0), _event("same", 1000.0, 0.0, 10.0)]
    summary = audit_constraints(events, speeds=(10.0,))
    assert summary["by_speed"][0]["impossible_pairs"] == 1
    assert summary["by_speed"][0]["impossible_pairs_labelled_one_animal"] == 1


def test_the_audit_reports_no_contradiction_when_the_labels_agree() -> None:
    events = [_event("a", 0.0, 0.0, 10.0), _event("b", 1000.0, 0.0, 10.0)]
    summary = audit_constraints(events, speeds=(10.0,))
    assert summary["by_speed"][0]["impossible_pairs"] == 1
    assert summary["by_speed"][0]["impossible_pairs_labelled_one_animal"] == 0


def test_a_count_that_does_not_move_with_speed_means_simultaneity_is_what_excludes() -> None:
    """This is the reading the speed sweep exists to support."""

    events = [_event("a", 0.0, 0.0, 0.0), _event("b", 2000.0, 0.0, 0.5)]
    summary = audit_constraints(events, speeds=(5.0, 30.0))
    counts = {entry["impossible_pairs"] for entry in summary["by_speed"]}
    assert counts == {1}


def test_the_audit_counts_near_simultaneous_pairs() -> None:
    events = [
        _event("a", 0.0, 0.0, 0.0),
        _event("b", 10.0, 0.0, 0.4),
        _event("c", 20.0, 0.0, 500.0),
    ]
    summary = audit_constraints(events, speeds=(10.0,), simultaneity_seconds=1.0)
    assert summary["pairs_within_simultaneity_window"] == 1
    assert summary["positions"] == 3
    assert summary["identities"] == 3


def test_an_audit_of_one_event_is_refused() -> None:
    with pytest.raises(ValueError, match="at least two"):
        audit_constraints([_event("a", 0.0, 0.0, 0.0)])


# --- reading an endpoint -----------------------------------------------------


def _record(
    identity: str,
    source_recording: str,
    x: float,
    y: float,
    onset: float,
    split: str = "query",
) -> ClipRecord:
    return ClipRecord(
        filename=identity + source_recording,
        path=Path("unused.wav"),
        identity=identity,
        split=split,
        context={
            "nest_x": x,
            "nest_y": y,
            "source_recording": source_recording,
            "song_key": source_recording + "_48000",
            "song_datetime": "2020-04-16 06:01:00",
            "annotation": {"onsets": [onset]},
        },
    )


def _endpoint(records: tuple[ClipRecord, ...]) -> Endpoint:
    return Endpoint(
        name="synthetic",
        records=records,
        categorical_targets=(),
        manifest_sha256="0" * 64,
        source_document="tests",
    )


def test_an_endpoint_is_read_into_events_with_the_onset_added() -> None:
    endpoint = _endpoint(
        (
            _record("a", "2020A01_20200416_060000", 0.0, 0.0, 2.5, split="enrollment"),
            _record("b", "2020A02_20200416_060000", 100.0, 0.0, 0.0),
        )
    )
    events = endpoint_events(endpoint)
    assert len(events) == 2
    assert events[0].seconds - events[1].seconds == pytest.approx(2.5)


def test_an_endpoint_without_a_position_is_refused_and_says_what_is_missing() -> None:
    """Returning nothing here would read as a finding about the birds."""

    record = _record("a", "2020A01_20200416_060000", 0.0, 0.0, 0.0)
    stripped = ClipRecord(
        filename=record.filename,
        path=record.path,
        identity=record.identity,
        split=record.split,
        context={"source_recording": "2020A01_20200416_060000"},
    )
    sound = _record("b", "2020A02_20200416_060000", 5.0, 0.0, 0.0, split="enrollment")
    with pytest.raises(ValueError, match="nest_x"):
        endpoint_events(_endpoint((sound, stripped)))


def test_missing_song_time_cannot_fall_back_to_recording_start() -> None:
    from dataclasses import replace

    record = _record("a", "2020A01_20200416_060000", 0.0, 0.0, 0.0)
    context = {k: v for k, v in record.context.items() if k != "song_datetime"}
    enrollment = replace(record, filename="enrollment", split="enrollment")
    with pytest.raises(ValueError, match="song_datetime"):
        endpoint_events(_endpoint((enrollment, replace(record, context=context))))


def test_song_start_offset_prevents_false_simultaneity(tmp_path: Path) -> None:
    records = (
        _record("a", "2020A01_20200416_060000", 0.0, 0.0, 0.25, split="enrollment"),
        _record("b", "2020A02_20200416_060000", 100.0, 0.0, 0.25),
    )
    metadata = tmp_path / "songs.csv"
    metadata.write_text(
        ",datetime\n2020A01_20200416_060000_48000,2020-04-16 06:01:00\n"
        "2020A02_20200416_060000_48000,2020-04-16 06:11:00\n"
    )
    events = endpoint_events(attach_song_times(_endpoint(records), metadata))
    assert events[1].seconds - events[0].seconds == 600
    assert impossible_pairs(events, speed_ms=5.0) == []


@pytest.mark.parametrize("rows", ["", "same,2020-04-16 06:00:00\nsame,2020-04-16 06:00:00\n"])
def test_missing_or_duplicate_metadata_refuses_an_audit(tmp_path: Path, rows: str) -> None:
    from dataclasses import replace

    record = _record("a", "2020A01_20200416_060000", 0.0, 0.0, 0.0)
    record = replace(record, context={**record.context, "song_key": "same"})
    metadata = tmp_path / "songs.csv"
    metadata.write_text(",datetime\n" + rows)
    enrollment = replace(record, filename="enrollment", split="enrollment")
    with pytest.raises(ValueError, match="missing|duplicate"):
        attach_song_times(_endpoint((enrollment, record)), metadata)


def test_uncertain_clocks_or_source_positions_can_prevent_exclusion() -> None:
    events = [_event("a", 0, 0, 0), _event("b", 100, 0, 0)]
    assert impossible_pairs(events, speed_ms=5.0) == [(0, 1)]
    assert impossible_pairs(events, speed_ms=5.0, clock_error_seconds=10) == []
    assert impossible_pairs(events, speed_ms=5.0, position_error_metres=50) == []
