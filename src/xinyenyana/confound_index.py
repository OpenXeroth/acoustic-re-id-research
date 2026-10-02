"""What each endpoint can test, measured before any model runs.

An accuracy does not say whether the method read the animal or the
circumstance. Whether an endpoint can tell those apart at all is a property of
how its recordings were made, and it is countable from the manifest.

For each endpoint this tabulates identity against every categorical axis the
endpoint's own loader carries in ``ClipRecord.context``, plus the split, and
reports four numbers per axis:

``nmi``
    Mutual information divided by the smaller entropy. For nonconstant axes,
    0 denotes empirical independence and 1 means one variable determines the
    other, including a nested relationship. Neither value by itself establishes
    what a split or conditional comparison can distinguish. Constant axes are
    assigned 0 by convention, not as evidence that recording effects are absent.
``cramers_v``
    The same association on a different scale. The two disagree when one
    variable has many sparse levels, so both are reported rather than one.
``identities_on_two_or_more_levels`` and ``levels_with_two_or_more_identities``
    The two counts that decide whether a within-axis permutation or a
    cross-axis split is constructible. These counts and the actual division must
    accompany the index; a shared metadata value does not eliminate unrecorded
    spatial or behavioural differences.

Nothing here reads audio and nothing fits anything.
"""

from __future__ import annotations

import math
from collections import Counter, defaultdict
from collections.abc import Sequence
from typing import Any

#: Context keys that are not an axis: they are constant, continuous, or a
#: restatement of the identity itself. Listed so that a new endpoint's axes are
#: found automatically rather than enumerated by hand.
NOT_AN_AXIS = frozenset(
    {
        "species",
        "transmitter",
        "tag_deployment",
        "onset",
        "calendar_date_ordinal",
        "nest_x",
        "nest_y",
        "quality",
        "annotation",
        "song_key",
        "shift_hz",
        "fold",
    }
)


def normalised_mutual_information(pairs: Sequence[tuple[str, str]]) -> float:
    """Mutual information divided by the smaller of the two entropies."""

    total = len(pairs)
    if total == 0:
        return 0.0
    left = Counter(a for a, _ in pairs)
    right = Counter(b for _, b in pairs)
    joint = Counter(pairs)
    entropy_left = -sum(v / total * math.log(v / total) for v in left.values())
    entropy_right = -sum(v / total * math.log(v / total) for v in right.values())
    smaller = min(entropy_left, entropy_right)
    if smaller <= 0:
        return 0.0
    information = sum(
        count / total * math.log((count / total) / ((left[a] / total) * (right[b] / total)))
        for (a, b), count in joint.items()
    )
    return information / smaller


def cramers_v(pairs: Sequence[tuple[str, str]]) -> float:
    """Chi-squared association, scaled to [0, 1]."""

    total = len(pairs)
    if total == 0:
        return 0.0
    left = Counter(a for a, _ in pairs)
    right = Counter(b for _, b in pairs)
    joint = Counter(pairs)
    smaller = min(len(left), len(right))
    if smaller < 2:
        return 0.0
    chi_squared = 0.0
    for a, count_a in left.items():
        for b, count_b in right.items():
            expected = count_a * count_b / total
            observed = joint.get((a, b), 0)
            chi_squared += (observed - expected) ** 2 / expected
    return math.sqrt(chi_squared / (total * (smaller - 1)))


def axis_reading(pairs: Sequence[tuple[str, str]]) -> dict[str, Any]:
    """The four numbers, plus the counts a reader needs to interpret them."""

    levels_by_identity: dict[str, set[str]] = defaultdict(set)
    identities_by_level: dict[str, set[str]] = defaultdict(set)
    for identity, level in pairs:
        levels_by_identity[identity].add(level)
        identities_by_level[level].add(identity)
    return {
        "clips": len(pairs),
        "identities": len(levels_by_identity),
        "levels": len(identities_by_level),
        "nmi": round(normalised_mutual_information(pairs), 4),
        "cramers_v": round(cramers_v(pairs), 4),
        "identities_on_two_or_more_levels": sum(
            1 for levels in levels_by_identity.values() if len(levels) > 1
        ),
        "levels_with_two_or_more_identities": sum(
            1 for identities in identities_by_level.values() if len(identities) > 1
        ),
        "clips_in_levels_with_two_or_more_identities": sum(
            1 for _, level in pairs if len(identities_by_level[level]) > 1
        ),
    }


def _derived_axes(context: dict[str, Any]) -> dict[str, str]:
    """Axes built from a context value rather than read straight out of it."""

    derived: dict[str, str] = {}
    stamp = context.get("recorded")
    if isinstance(stamp, str) and len(stamp) >= 10:
        derived["recording_day"] = stamp[:10]
    date = context.get("date")
    if isinstance(date, str) and len(date) >= 6:
        derived["recording_day"] = date
    return derived


def endpoint_axes(records: Sequence[Any]) -> dict[str, list[tuple[str, str]]]:
    """Every usable axis for this endpoint, as identity-against-level pairs."""

    axes: dict[str, list[tuple[str, str]]] = defaultdict(list)
    for record in records:
        identity = str(record.identity)
        axes["split"].append((identity, str(record.split)))
        for key, value in dict(record.context).items():
            if key in NOT_AN_AXIS or value is None:
                continue
            if isinstance(value, (dict, list)):
                continue
            axes[key].append((identity, str(value)))
        for key, value in _derived_axes(dict(record.context)).items():
            axes[key].append((identity, value))
    # An axis with one level says nothing about separability, but the fact that
    # it is constant is itself worth recording, so it is kept and marked.
    return dict(axes)


def confound_index(*, name: str, records: Sequence[Any]) -> dict[str, Any]:
    """The index for one endpoint, over all clips and within each split."""

    identities = sorted({str(record.identity) for record in records})
    result: dict[str, Any] = {
        "endpoint": name,
        "clips": len(records),
        "identities": len(identities),
        "axes": {},
    }
    for axis, pairs in sorted(endpoint_axes(records).items()):
        if axis == "split":
            reading = {"all": axis_reading(pairs)}
        else:
            reading = {"all": axis_reading(pairs)}
            for split in ("enrollment", "query"):
                in_split = [
                    (str(record.identity), str(dict(record.context).get(axis, "")))
                    for record in records
                    if record.split == split and axis in dict(record.context)
                ]
                if not in_split:
                    in_split = [
                        (str(record.identity), _derived_axes(dict(record.context))[axis])
                        for record in records
                        if record.split == split and axis in _derived_axes(dict(record.context))
                    ]
                if in_split:
                    reading[split] = axis_reading(in_split)
        result["axes"][axis] = reading
    return result


def verdicts(index: dict[str, Any], *, high: float = 0.9, low: float = 0.3) -> dict[str, str]:
    """One sentence per axis, in the form the paper prints beside the index."""

    lines: dict[str, str] = {}
    for axis, reading in index["axes"].items():
        overall = reading["all"]
        value = float(overall["nmi"])
        if overall["levels"] < 2:
            lines[axis] = (
                "constant in the recorded metadata; association cannot be estimated "
                "and unrecorded differences remain possible"
            )
        else:
            degree = "high" if value >= high else "low" if value <= low else "intermediate"
            lines[axis] = (
                f"{degree} marginal association with identity (NMI {value:.3f}); "
                f"{overall['levels_with_two_or_more_identities']} of {overall['levels']} "
                "levels hold two or more individuals; "
                f"{overall['identities_on_two_or_more_levels']} of {overall['identities']} "
                "individuals span levels. Read these counts with the actual split; "
                "the index alone does not establish separability"
            )
    return lines
