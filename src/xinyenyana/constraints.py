"""Physical constraints on which detections can be the same animal.

An animal cannot be in two places at once and cannot travel faster than it can
fly. Applying those facts to a dataset requires valid event clocks and source
positions. Nestbox coordinates and unsynchronised recorder clocks are proxies,
so an exclusion computed from them is conditional on the stated error bounds.

This module measures how much they exclude. It does not assign identities; it
counts conditional exclusions and reports disagreements with published labels.
A disagreement can arise from timing, localisation, identity attribution or
the assumed speed. It is reported rather than used to change the labels.
"""

from __future__ import annotations

import csv
import math
from collections.abc import Sequence
from dataclasses import dataclass, replace
from datetime import UTC, datetime
from itertools import combinations
from pathlib import Path
from typing import Any

from xinyenyana.a2 import Endpoint

#: Flight speeds the audit reports at, metres per second. The span is wide on
#: purpose: if the count does not move across it, what excludes a pairing is
#: simultaneity rather than any claim about how fast the animal flies.
SPEEDS_MS: tuple[float, ...] = (5.0, 10.0, 15.0, 20.0, 30.0)

#: Two detections closer than this in time are treated as simultaneous.
SIMULTANEITY_SECONDS = 1.0


@dataclass(frozen=True)
class ClipEvent:
    """One detection, placed in time and space."""

    identity: str
    easting: float
    northing: float
    seconds: float

    def __post_init__(self) -> None:
        if not all(math.isfinite(value) for value in (self.easting, self.northing, self.seconds)):
            raise ValueError("event positions and times must be finite")

    def separation(self, other: ClipEvent) -> float:
        return math.dist((self.easting, self.northing), (other.easting, other.northing))


def attach_song_times(endpoint: Endpoint, metadata_path: Path) -> Endpoint:
    """Join published song starts by exact song key, without inventing a clock.

    Note onsets are relative to the segmented song. The source recording's
    filename clock alone omits the often substantial offset to that song.
    """

    required = {str(record.context["song_key"]) for record in endpoint.records}
    matched: dict[str, str] = {}
    with metadata_path.open(newline="") as stream:
        reader = csv.DictReader(stream)
        if (
            not reader.fieldnames
            or "" not in reader.fieldnames
            or "datetime" not in reader.fieldnames
        ):
            raise ValueError("song metadata requires its published index and datetime columns")
        for row in reader:
            key = row[""]
            if key not in required:
                continue
            if key in matched:
                raise ValueError("song metadata contains a duplicate requested key")
            datetime.fromisoformat(row["datetime"])
            matched[key] = row["datetime"]
    if required != matched.keys():
        raise ValueError(f"song metadata is missing {len(required - matched.keys())} sample songs")
    return replace(
        endpoint,
        records=tuple(
            replace(
                record,
                context={
                    **record.context,
                    "song_datetime": matched[str(record.context["song_key"])],
                },
            )
            for record in endpoint.records
        ),
    )


def _song_seconds(value: str) -> float:
    # The publisher's naive wall-clock timestamps are represented on a common
    # numerical axis. Attaching UTC is arithmetic, not evidence of UTC accuracy
    # or synchronisation of independently deployed recorders.
    parsed = datetime.fromisoformat(value)
    return (parsed if parsed.tzinfo else parsed.replace(tzinfo=UTC)).timestamp()


def endpoint_events(endpoint: Endpoint) -> list[ClipEvent]:
    """Place every clip of an endpoint in time and space.

    Refuses rather than guesses. An endpoint whose manifest carries no position
    or no recording time cannot be audited this way, and saying so names what is
    missing instead of returning an empty result that reads like a finding.
    """

    required = ("nest_x", "nest_y", "song_datetime")
    events: list[ClipEvent] = []
    for record in endpoint.records:
        missing = [field for field in required if field not in record.context]
        if missing:
            raise ValueError(
                f"endpoint {endpoint.name!r} carries no {', '.join(missing)}, "
                "so no physical constraint can be evaluated on it"
            )
        onsets = record.context.get("annotation", {}).get("onsets")
        if not onsets or not math.isfinite(float(onsets[0])) or float(onsets[0]) < 0:
            raise ValueError("a constraint event requires a finite nonnegative song-relative onset")
        events.append(
            ClipEvent(
                identity=record.identity,
                easting=float(record.context["nest_x"]),
                northing=float(record.context["nest_y"]),
                seconds=_song_seconds(str(record.context["song_datetime"])) + float(onsets[0]),
            )
        )
    return events


def impossible_pairs(
    events: Sequence[ClipEvent],
    *,
    speed_ms: float,
    clock_error_seconds: float = 0.0,
    position_error_metres: float = 0.0,
) -> list[tuple[int, int]]:
    """Index pairs that cannot be one animal at that speed.

    Only pairs at different positions can be excluded. Two detections at one
    position carry no spatial information about whether they are one animal.
    """

    if not math.isfinite(speed_ms) or speed_ms <= 0:
        raise ValueError("a flight speed must be positive")
    if not all(
        math.isfinite(value) and value >= 0
        for value in (clock_error_seconds, position_error_metres)
    ):
        raise ValueError("uncertainty bounds must be nonnegative")
    found: list[tuple[int, int]] = []
    for (first, one), (second, two) in combinations(enumerate(events), 2):
        separation = one.separation(two)
        if separation == 0:
            continue
        if (
            max(0.0, separation - 2 * position_error_metres)
            > (abs(one.seconds - two.seconds) + 2 * clock_error_seconds) * speed_ms
        ):
            found.append((first, second))
    return found


def constraint_sensitivity(events: Sequence[ClipEvent]) -> list[dict[str, Any]]:
    """Registered assumed per-detection error bounds; none is a clock validation."""

    return [
        {
            "speed_metres_per_second": speed,
            "clock_error_seconds_per_detection": clock,
            "position_error_metres_per_detection": position,
            "impossible_pairs": len(
                impossible_pairs(
                    events,
                    speed_ms=speed,
                    clock_error_seconds=clock,
                    position_error_metres=position,
                )
            ),
        }
        for speed in SPEEDS_MS
        for clock in (0.0, 1.0, 10.0, 60.0)
        for position in (0.0, 10.0, 50.0)
    ]


def audit_constraints(
    events: Sequence[ClipEvent],
    *,
    speeds: Sequence[float] = SPEEDS_MS,
    simultaneity_seconds: float = SIMULTANEITY_SECONDS,
) -> dict[str, Any]:
    """How much the physics excludes, and whether the labels agree with it."""

    if len(events) < 2:
        raise ValueError("an audit needs at least two events")
    cross_position = sum(1 for one, two in combinations(events, 2) if one.separation(two) > 0)
    by_speed = []
    for speed in speeds:
        pairs = impossible_pairs(events, speed_ms=speed)
        disagreeing = sum(
            1 for first, second in pairs if events[first].identity == events[second].identity
        )
        by_speed.append(
            {
                "speed_metres_per_second": float(speed),
                "impossible_pairs": len(pairs),
                "share_of_cross_position_pairs": (
                    len(pairs) / cross_position if cross_position else 0.0
                ),
                "impossible_pairs_labelled_one_animal": disagreeing,
            }
        )
    separations = sorted(
        one.separation(two) for one, two in combinations(events, 2) if one.separation(two) > 0
    )
    intervals = sorted(abs(one.seconds - two.seconds) for one, two in combinations(events, 2))
    return {
        "events": len(events),
        "identities": len({event.identity for event in events}),
        "positions": len({(event.easting, event.northing) for event in events}),
        "cross_position_pairs": cross_position,
        "pairs_within_simultaneity_window": sum(
            1
            for one, two in combinations(events, 2)
            if abs(one.seconds - two.seconds) < simultaneity_seconds
        ),
        "simultaneity_seconds": float(simultaneity_seconds),
        "separation_metres": _spread(separations),
        "interval_seconds": _spread(intervals),
        "by_speed": by_speed,
    }


def _spread(values: Sequence[float]) -> dict[str, float]:
    if not values:
        return {"minimum": 0.0, "median": 0.0, "maximum": 0.0}
    ordered = sorted(values)
    return {
        "minimum": float(ordered[0]),
        "median": float(ordered[len(ordered) // 2]),
        "maximum": float(ordered[-1]),
    }
