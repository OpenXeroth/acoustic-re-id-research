"""Open set: is this call one of the enrolled birds, or a stranger?

Closed-set identity asks which of the enrolled birds is calling and is answered
in [`a2`](a2.py) and [`a5`](a5.py). It presumes the caller is enrolled. At a
waterhole most callers are not, so the question that decides whether any of this
reaches Djuma is the one measured here: does this call belong to any enrolled
bird at all?

The protocol is in [`docs/measurement-protocol.md`](../../docs/measurement-protocol.md),
registered 2026-09-01 before any open-set score was read, and is reproduced here
in the constants and the rules below rather than read from a configuration file,
so a change to it is a change to this module and shows up in a diff.

Sixteen Great Tits are split into four roles of four: two roles calibrate the
threshold, two are scored against it, and no bird is in both. Sixteen such
splits are drawn so that no claim rests on one lucky division of the birds.
Every threshold is chosen on calibration-unknown birds only, and the test
gallery is built after that threshold is fixed, so no number here was tuned on
the clips it is scored on.
"""

from __future__ import annotations

import hashlib
import math
import wave
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from xinyenyana.a2 import ClipRecord, Endpoint
from xinyenyana.archive import canonical_sha256

ROLES = ("calibration_known", "calibration_unknown", "test_known", "test_unknown")
ALLOCATIONS = 16
ALLOCATION_SEED = 317
#: The string the allocation hash is salted with. It reads as an experiment name
#: because it is the one the protocol was registered under. Changing it silently
#: redraws all sixteen allocations and every number below them, so it is frozen
#: with the seed rather than renamed.
ALLOCATION_SALT = "E03s"
IDENTITIES_PER_ROLE = 4
COHORT_SIZES = {"2020-2021": 5, "2020-2022": 3, "2021-2022": 8}

MASK_MARGIN_SECONDS = 0.02
CLIPPING_SLICE_THRESHOLD = 0.01
WILSON_Z = 1.959963984540054

#: The stranger budgets the trade-off curve is reported at. These are reporting
#: points, not gates. A previous version of this module carried five pass/fail
#: thresholds, and reporting "passed 0 of 16" turned clearing them into the
#: objective. What the project wants to know is what can and cannot be done, so
#: the curve is reported and nothing is declared passed or failed.
STRANGER_BUDGETS: tuple[float, ...] = (0.01, 0.05, 0.10, 0.20, 0.50)

BIRDNET = "birdnet-v2.4"
MASKED_BIRDNET = "masked-birdnet-v2.4"
LOG_MEL = "classical-log-mel-summary"
CONTEXT = "acquisition-context"
LEVEL_DURATION = "level-duration-clipping"
SONG_STRUCTURE = "annotation-song-structure"

REPRESENTATIONS: tuple[str, ...] = (
    BIRDNET,
    MASKED_BIRDNET,
    LOG_MEL,
    CONTEXT,
    LEVEL_DURATION,
    SONG_STRUCTURE,
)

#: Representations whose dimensions carry different units and must be scaled per
#: dimension before they are compared. An embedding is left alone: its
#: dimensions already share a scale, and rescaling them destroys that.
STANDARDISED: frozenset[str] = frozenset({LOG_MEL, CONTEXT, LEVEL_DURATION, SONG_STRUCTURE})


def allocate_roles(
    identities_by_cohort: dict[str, list[str]], *, seed: int, allocation: int
) -> dict[str, list[str]]:
    """One of the sixteen cohort-balanced splits of the birds into four roles.

    The sixteen are four groups of four. Within a group the birds are ordered by
    a hash of their name, so the order does not depend on the manifest, and are
    then dealt so that every role holds two birds from the eight-bird cohort and
    two from the other two, with one role holding two from the five-bird cohort.
    Rotating the four dealt groups gives the four allocations of that group, so
    every bird holds every role once per group.

    The cohort sizes are checked rather than assumed: a sample that is not the
    frozen 5/3/8 split cannot be dealt this way, and silently dealing it anyway
    would make sixteen allocations that are not balanced.
    """

    sizes = {key: len(set(value)) for key, value in identities_by_cohort.items()}
    if sizes != COHORT_SIZES:
        raise ValueError(f"open set requires the frozen cohorts {COHORT_SIZES}; got {sizes}")
    everyone = [value for values in identities_by_cohort.values() for value in values]
    if len(set(everyone)) != sum(COHORT_SIZES.values()):
        raise ValueError("the cohorts overlap")
    if not 0 <= allocation < ALLOCATIONS:
        raise ValueError(f"allocation must be in [0, {ALLOCATIONS - 1}]")

    group, shift = divmod(allocation, IDENTITIES_PER_ROLE)

    def ordered(cohort: str) -> list[str]:
        return sorted(
            identities_by_cohort[cohort],
            key=lambda identity: hashlib.sha256(
                f"{ALLOCATION_SALT}:{seed}:{group}:{cohort}:{identity}".encode()
            ).hexdigest(),
        )

    five, three, eight = ordered("2020-2021"), ordered("2020-2022"), ordered("2021-2022")
    dealt: list[list[str]] = [[] for _ in ROLES]
    for role in range(IDENTITIES_PER_ROLE):
        dealt[role].extend(eight[role * 2 : role * 2 + 2])
    for role, identity in zip(
        [role for role in range(IDENTITIES_PER_ROLE) if role != group], three, strict=True
    ):
        dealt[role].append(identity)
    cursor = 0
    for role in range(IDENTITIES_PER_ROLE):
        take = 2 if role == group else 1
        dealt[role].extend(five[cursor : cursor + take])
        cursor += take

    roles = {
        ROLES[(role + shift) % len(ROLES)]: sorted(values) for role, values in enumerate(dealt)
    }
    flattened = [identity for role in ROLES for identity in roles[role]]
    if len(flattened) != len(set(flattened)) or len(flattened) != len(everyone):
        raise AssertionError("the role allocation is not a partition of the birds")
    return roles


def allocate_roles_evenly(
    identities: Sequence[str], *, seed: int, allocation: int
) -> dict[str, list[str]]:
    """The same sixteen-allocation scheme for an endpoint with no year cohorts.

    Registered 2026-09-22 for the little owl, the cockatoo and the rook, which
    have no cohort structure to balance on. The birds are ordered by a salted
    hash within each of four groups, dealt round the four roles, and the dealt
    groups are rotated so that every bird holds every role once per group. Role
    sizes differ by at most one bird.
    """

    everyone = sorted(set(identities))
    if len(everyone) < 2 * len(ROLES):
        raise ValueError(f"open set needs at least {2 * len(ROLES)} birds, got {len(everyone)}")
    if not 0 <= allocation < ALLOCATIONS:
        raise ValueError(f"allocation must be in [0, {ALLOCATIONS - 1}]")
    group, shift = divmod(allocation, IDENTITIES_PER_ROLE)
    ordered = sorted(
        everyone,
        key=lambda identity: hashlib.sha256(
            f"{ALLOCATION_SALT}:{seed}:{group}:{identity}".encode()
        ).hexdigest(),
    )
    dealt = [ordered[role :: len(ROLES)] for role in range(len(ROLES))]
    roles = {
        ROLES[(role + shift) % len(ROLES)]: sorted(values) for role, values in enumerate(dealt)
    }
    flattened = [identity for role in ROLES for identity in roles[role]]
    if sorted(flattened) != everyone:
        raise AssertionError("the role allocation is not a partition of the birds")
    return roles


def wilson_interval(successes: int, total: int) -> list[float]:
    """The 95% interval for a proportion, which stays inside 0 and 1 at the ends.

    A rate of 0 out of 40 has a normal interval of zero width, which would
    report a certainty the forty clips do not support.
    """

    if total <= 0:
        raise ValueError("a Wilson interval needs at least one observation")
    proportion = successes / total
    denominator = 1 + WILSON_Z * WILSON_Z / total
    centre = (proportion + WILSON_Z * WILSON_Z / (2 * total)) / denominator
    half = (
        WILSON_Z
        * math.sqrt(
            proportion * (1 - proportion) / total + WILSON_Z * WILSON_Z / (4 * total * total)
        )
        / denominator
    )
    return [centre - half, centre + half]


def known_vs_unknown_auroc(known: Sequence[float], unknown: Sequence[float]) -> float:
    """How often an enrolled bird's call outscores a stranger's, ties counted half.

    This is the threshold-free half of the question. A method can separate the
    two groups well and still accept almost nothing at a tight stranger budget,
    because that also needs the separation to sit where the threshold falls.
    """

    import numpy as np

    if not len(known) or not len(unknown):
        raise ValueError("this needs both enrolled and stranger scores")
    ordered = np.sort(np.asarray(unknown, dtype=np.float64))
    values = np.asarray(known, dtype=np.float64)
    below = np.searchsorted(ordered, values, side="left")
    at_or_below = np.searchsorted(ordered, values, side="right")
    wins = float(below.sum())
    ties = float((at_or_below - below).sum())
    return (wins + 0.5 * ties) / (len(values) * len(ordered))


def equal_error_rate(known: Sequence[float], unknown: Sequence[float]) -> float:
    """Where rejecting an enrolled bird and accepting a stranger are equally likely.

    Computed on the known-against-stranger decision alone, naming ignored, so it
    is the detection half of the open-set task.
    """

    import numpy as np

    positive = np.asarray(known, dtype=np.float64)
    negative = np.asarray(unknown, dtype=np.float64)
    thresholds = np.unique(np.concatenate([positive, negative]))
    thresholds = np.append(thresholds, np.nextafter(thresholds[-1], np.inf))
    false_reject = np.searchsorted(np.sort(positive), thresholds, side="left") / len(positive)
    false_accept = 1.0 - np.searchsorted(np.sort(negative), thresholds, side="left") / len(negative)
    crossing = int(np.argmin(np.abs(false_accept - false_reject)))
    return float((false_accept[crossing] + false_reject[crossing]) / 2)


def true_accept_at_false_accept(
    known: Sequence[float], unknown: Sequence[float], *, budget: float
) -> float:
    """The share of enrolled calls accepted, naming ignored, with strangers held to the budget."""

    import numpy as np

    positive = np.sort(np.asarray(known, dtype=np.float64))
    negative = np.sort(np.asarray(unknown, dtype=np.float64))
    # A threshold just above a stranger's score excludes that stranger, so the
    # candidates are every score and the next value above every stranger score.
    thresholds = np.unique(np.concatenate([positive, negative, np.nextafter(negative, np.inf)]))
    far = 1.0 - np.searchsorted(negative, thresholds, side="left") / len(negative)
    feasible = thresholds[far <= budget]
    lowest = float(feasible.min())
    return float(1.0 - np.searchsorted(positive, lowest, side="left") / len(positive))


def calibrate_threshold(unknown_scores: Sequence[float], *, target_far: float) -> float:
    """The lowest score a call must reach to be accepted, set on strangers alone.

    Chosen as the lowest boundary at which no more than ``target_far`` of the
    calibration strangers are accepted. Enrolled birds do not vote, so the
    threshold cannot be moved by how well the method happens to recognise them.
    """

    if not unknown_scores:
        raise ValueError("calibration needs stranger scores")
    candidates = sorted({float(value) for value in unknown_scores})
    candidates.append(math.nextafter(candidates[-1], math.inf))
    feasible = [
        threshold
        for threshold in candidates
        if sum(value >= threshold for value in unknown_scores) / len(unknown_scores) <= target_far
    ]
    if not feasible:  # pragma: no cover - the appended boundary accepts nobody
        raise ValueError("no threshold reaches the calibration target")
    return min(feasible)


def normalise(vectors: Any, train_rows: Sequence[int], *, standardise: bool) -> Any:
    """Scale per dimension on the calibration enrollment rows only, then to unit length.

    The mean and spread come from calibration-known enrollment clips, never from
    any clip that will be scored, so no query clip influences its own scaling.
    """

    import numpy as np

    values = np.asarray(vectors, dtype=np.float64)
    if standardise:
        rows = np.asarray(list(train_rows), dtype=int)
        if rows.size == 0:
            raise ValueError("standardisation needs calibration enrollment rows")
        mean = values[rows].mean(axis=0, keepdims=True)
        scale = values[rows].std(axis=0, keepdims=True)
        scale[scale == 0] = 1.0
        values = (values - mean) / scale
    return values / np.maximum(np.linalg.norm(values, axis=1, keepdims=True), 1e-12)


def score_against_gallery(
    *,
    values: Any,
    records: Sequence[ClipRecord],
    known: Sequence[str],
    unknown: Sequence[str],
    partition: str,
    balanced: bool = True,
    scorer: str = "mean-profile",
) -> list[dict[str, Any]]:
    """Score every query clip of these birds against a gallery of the known ones.

    A bird's gallery entry is the mean of its enrollment clips. A query clip's
    score is the highest similarity to any entry, and the identity that achieved
    it is the guess. A stranger has no entry, so its highest score is what
    decides whether the system wrongly claims to know it.
    """

    import numpy as np

    if scorer not in {"mean-profile", "nearest-clip"}:
        raise ValueError(f"unknown gallery scorer: {scorer}")
    prototypes = []
    gallery_identities = []
    for identity in known:
        rows = [
            index
            for index, record in enumerate(records)
            if record.identity == identity and record.split == "enrollment"
        ]
        if not rows:
            raise ValueError(f"{identity} has no enrollment clips")
        if scorer == "nearest-clip":
            prototypes.extend(values[rows])
            gallery_identities.extend([identity] * len(rows))
        else:
            prototype = values[rows].mean(axis=0)
            prototypes.append(prototype / max(float(np.linalg.norm(prototype)), 1e-12))
            gallery_identities.append(identity)
    gallery = np.asarray(prototypes)

    selected = set(known) | set(unknown)
    observations = []
    counts: dict[str, int] = {}
    for index, record in enumerate(records):
        if record.split != "query" or record.identity not in selected:
            continue
        scores = values[index] @ gallery.T
        top = int(np.argmax(scores))
        counts[record.identity] = counts.get(record.identity, 0) + 1
        observations.append(
            {
                "partition": partition,
                "clip": record.filename,
                "identity": record.identity,
                "cohort": str(record.context.get("cohort", "")),
                "is_known": record.identity in set(known),
                "top_identity": gallery_identities[top],
                "maximum_score": float(scores[top]),
                "clipped_over_slice_threshold": (
                    float(record.context["quality"]["clipped_sample_fraction"])
                    > CLIPPING_SLICE_THRESHOLD
                    if "quality" in record.context
                    else None
                ),
            }
        )
    if set(counts) != selected:
        raise ValueError(
            f"{partition} has birds with no query clips: {sorted(selected - set(counts))}"
        )
    if balanced and len(set(counts.values())) != 1:
        raise ValueError(
            f"{partition} does not give every bird the same number of query clips: {counts}"
        )
    return observations


def balanced_known_unknown(
    observations: Sequence[dict[str, Any]], *, threshold: float
) -> dict[str, float]:
    """Balanced accuracy on enrolled birds, on strangers, and their geometric mean.

    The AnimalCLEF 2025 metric for visual re-identification (PA-V6). An enrolled
    bird's call is right when it clears the threshold and names that bird; a
    stranger's call is right when it does not clear it. Each is averaged within
    each bird first, then over birds. The geometric mean is used because a
    system that calls every animal new scores 0.5 on the arithmetic mean and 0
    on the geometric.
    """

    import math

    def per_bird(entries: Sequence[dict[str, Any]], right: Any) -> float:
        birds: dict[str, list[bool]] = {}
        for entry in entries:
            birds.setdefault(str(entry["identity"]), []).append(bool(right(entry)))
        return sum(sum(v) / len(v) for v in birds.values()) / len(birds)

    known = [value for value in observations if value["is_known"]]
    unknown = [value for value in observations if not value["is_known"]]
    baks = per_bird(
        known,
        lambda v: float(v["maximum_score"]) >= threshold and v["top_identity"] == v["identity"],
    )
    baus = per_bird(unknown, lambda v: float(v["maximum_score"]) < threshold)
    return {
        "balanced_accuracy_known": baks,
        "balanced_accuracy_unknown": baus,
        "geometric_mean_known_unknown": math.sqrt(baks * baus),
    }


def evaluate_observations(
    observations: Sequence[dict[str, Any]], *, threshold: float
) -> dict[str, Any]:
    """Turn scored clips into rates, at one threshold.

    An enrolled bird counts as correctly accepted only when the call clears the
    threshold **and** the guess names the right bird. Accepting a call and
    naming the wrong bird is counted separately, because at a waterhole it is a
    different failure from rejecting a bird you know.
    """

    known = [value for value in observations if value["is_known"]]
    unknown = [value for value in observations if not value["is_known"]]
    if not known or not unknown:
        raise ValueError("evaluation needs both enrolled and stranger clips")
    accepted_unknown = sum(float(value["maximum_score"]) >= threshold for value in unknown)
    correct = sum(
        float(value["maximum_score"]) >= threshold and value["top_identity"] == value["identity"]
        for value in known
    )
    rejected_known = sum(float(value["maximum_score"]) < threshold for value in known)
    misidentified = sum(
        float(value["maximum_score"]) >= threshold and value["top_identity"] != value["identity"]
        for value in known
    )
    unknown_far = accepted_unknown / len(unknown)
    correct_rate = correct / len(known)
    return {
        "known_queries": len(known),
        "unknown_queries": len(unknown),
        "threshold": threshold,
        "unknown_false_accepts": accepted_unknown,
        "unknown_false_accept_rate": unknown_far,
        "unknown_false_accept_wilson_95": wilson_interval(accepted_unknown, len(unknown)),
        "known_correct_accepts": correct,
        "known_correct_accept_rate": correct_rate,
        "known_correct_accept_wilson_95": wilson_interval(correct, len(known)),
        "known_false_rejects": rejected_known,
        "known_false_reject_rate": rejected_known / len(known),
        "known_misidentifications": misidentified,
        "known_misidentification_rate": misidentified / len(known),
        "open_set_balanced_accuracy": (correct_rate + 1 - unknown_far) / 2,
        **balanced_known_unknown(observations, threshold=threshold),
        "known_vs_unknown_auroc": known_vs_unknown_auroc(
            [float(value["maximum_score"]) for value in known],
            [float(value["maximum_score"]) for value in unknown],
        ),
    }


def acceptance_curve(observations: Sequence[dict[str, Any]]) -> dict[str, float]:
    """The best acceptance available at each stranger budget, over all thresholds.

    A single calibrated threshold is one point. This walks every threshold the
    scores allow and reports, for each budget, the highest share of enrolled
    calls that are accepted and named right while strangers stay inside it.
    That is the whole trade-off the method offers, rather than a verdict on one
    point of it.
    """

    import numpy as np

    known = [entry for entry in observations if entry["is_known"]]
    unknown = [entry for entry in observations if not entry["is_known"]]
    if not known or not unknown:
        raise ValueError("a curve needs both enrolled and stranger clips")
    thresholds = np.asarray(sorted({float(entry["maximum_score"]) for entry in observations}))
    thresholds = np.append(thresholds, math.nextafter(float(thresholds[-1]), math.inf))
    known_scores = np.asarray([float(entry["maximum_score"]) for entry in known])
    named = np.asarray([entry["top_identity"] == entry["identity"] for entry in known])
    named_scores = np.sort(known_scores[named])
    unknown_scores = np.sort(np.asarray([float(entry["maximum_score"]) for entry in unknown]))
    accepted_and_named = (
        len(named_scores) - np.searchsorted(named_scores, thresholds, side="left")
    ) / len(known)
    stranger_rate = (
        len(unknown_scores) - np.searchsorted(unknown_scores, thresholds, side="left")
    ) / len(unknown)
    curve: dict[str, float] = {}
    for budget in STRANGER_BUDGETS:
        reachable = accepted_and_named[stranger_rate <= budget]
        curve[f"{budget:.2f}"] = float(reachable.max()) if len(reachable) else 0.0
    return curve


def evaluate_allocation(
    *,
    vectors: Any,
    records: Sequence[ClipRecord],
    roles: dict[str, list[str]],
    standardise: bool,
    balanced: bool = True,
    scorer: str = "mean-profile",
) -> dict[str, Any]:
    """Calibrate on one pair of roles, then score the other pair against it."""

    train_rows = [
        index
        for index, record in enumerate(records)
        if record.split == "enrollment" and record.identity in set(roles["calibration_known"])
    ]
    values = normalise(vectors, train_rows, standardise=standardise)
    calibration = score_against_gallery(
        values=values,
        records=records,
        known=roles["calibration_known"],
        unknown=roles["calibration_unknown"],
        partition="calibration",
        balanced=balanced,
        scorer=scorer,
    )
    test = score_against_gallery(
        values=values,
        records=records,
        known=roles["test_known"],
        unknown=roles["test_unknown"],
        partition="test",
        balanced=balanced,
        scorer=scorer,
    )
    transferred: dict[str, dict[str, Any]] = {}
    for budget in STRANGER_BUDGETS:
        threshold = calibrate_threshold(
            [value["maximum_score"] for value in calibration if not value["is_known"]],
            target_far=budget,
        )
        transferred[f"{budget:.2f}"] = {
            "threshold": threshold,
            "test": evaluate_observations(test, threshold=threshold),
        }
    return {
        "roles": {role: list(members) for role, members in roles.items()},
        "observations": {"calibration": calibration, "test": test},
        "closed_set_accuracy": sum(
            entry["top_identity"] == entry["identity"] for entry in test if entry["is_known"]
        )
        / sum(entry["is_known"] for entry in test),
        "known_vs_unknown_auroc": known_vs_unknown_auroc(
            [float(entry["maximum_score"]) for entry in test if entry["is_known"]],
            [float(entry["maximum_score"]) for entry in test if not entry["is_known"]],
        ),
        "best_acceptance_at_budget": acceptance_curve(test),
        "equal_error_rate": equal_error_rate(
            [float(entry["maximum_score"]) for entry in test if entry["is_known"]],
            [float(entry["maximum_score"]) for entry in test if not entry["is_known"]],
        ),
        "true_accept_at_false_accept": {
            f"{budget:.2f}": true_accept_at_false_accept(
                [float(entry["maximum_score"]) for entry in test if entry["is_known"]],
                [float(entry["maximum_score"]) for entry in test if not entry["is_known"]],
                budget=budget,
            )
            for budget in STRANGER_BUDGETS
        },
        "calibrated_on_other_birds": transferred,
        "slices": _slices(test, threshold=transferred["0.05"]["threshold"]),
    }


def _slices(test: Sequence[dict[str, Any]], *, threshold: float) -> dict[str, Any]:
    """The same two rates broken down by year pair and by whether a clip clips."""

    result: dict[str, Any] = {}
    for field in ("cohort", "clipped_over_slice_threshold"):
        if any(value[field] in ("", None) for value in test):
            continue
        cells: dict[str, Any] = {}
        for cell in sorted({str(value[field]) for value in test}):
            selected = [value for value in test if str(value[field]) == cell]
            known = [value for value in selected if value["is_known"]]
            unknown = [value for value in selected if not value["is_known"]]
            cells[cell] = {
                "queries": len(selected),
                "known_correct_accept_rate": (
                    sum(
                        value["maximum_score"] >= threshold
                        and value["top_identity"] == value["identity"]
                        for value in known
                    )
                    / len(known)
                    if known
                    else None
                ),
                "unknown_queries": len(unknown),
            }
        result[field] = cells
    return result


def summarise_allocations(allocations: Sequence[dict[str, Any]]) -> dict[str, Any]:
    """The smallest, middle and largest value each statistic takes over the sixteen.

    A single allocation is one draw of four birds out of sixteen. Quoting it
    alone is how this project previously published a finding that its own
    sixteen-allocation spread contradicted.
    """

    import numpy as np

    def spread(values: Sequence[float]) -> dict[str, float]:
        return {
            "minimum": float(np.min(values)),
            "median": float(np.median(values)),
            "maximum": float(np.max(values)),
        }

    summary = {
        "closed_set_accuracy": spread([entry["closed_set_accuracy"] for entry in allocations]),
        "known_vs_unknown_auroc": spread(
            [entry["known_vs_unknown_auroc"] for entry in allocations]
        ),
        "equal_error_rate": spread([entry["equal_error_rate"] for entry in allocations]),
    }
    for budget in STRANGER_BUDGETS:
        key = f"{budget:.2f}"
        summary[f"best_acceptance_at_{key}"] = spread(
            [entry["best_acceptance_at_budget"][key] for entry in allocations]
        )
        summary[f"true_accept_at_false_accept_{key}"] = spread(
            [entry["true_accept_at_false_accept"][key] for entry in allocations]
        )
        summary[f"acceptance_calibrated_at_{key}"] = spread(
            [
                entry["calibrated_on_other_birds"][key]["test"]["known_correct_accept_rate"]
                for entry in allocations
            ]
        )
        for field in (
            "balanced_accuracy_known",
            "balanced_accuracy_unknown",
            "geometric_mean_known_unknown",
        ):
            summary[f"{field}_calibrated_at_{key}"] = spread(
                [entry["calibrated_on_other_birds"][key]["test"][field] for entry in allocations]
            )
    return summary


# --- the representations ----------------------------------------------------


def _annotation_vector(record: ClipRecord) -> list[float]:
    annotation = record.context["annotation"]
    units = [float(value) for value in annotation["unit_durations"]]
    gaps = [float(value) for value in annotation["silence_durations"]]
    duration = float(record.context["quality"]["duration_seconds"])
    lower = float(annotation["lower_frequency_hz"])
    upper = float(annotation["upper_frequency_hz"])
    return [
        math.log(max(len(units), 1)),
        sum(units) / duration if duration > 0 else 0.0,
        _mean(units),
        _deviation(units),
        _mean(gaps),
        _deviation(gaps),
        lower,
        upper,
        upper - lower,
    ]


def _mean(values: Sequence[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def _deviation(values: Sequence[float]) -> float:
    if len(values) < 2:
        return 0.0
    average = _mean(values)
    return math.sqrt(sum((value - average) ** 2 for value in values) / len(values))


def _level_vector(record: ClipRecord) -> list[float]:
    metrics = record.context["quality"]
    return [
        math.log(float(metrics["duration_seconds"])),
        float(metrics["rms_dbfs"]),
        float(metrics["peak_dbfs"]),
        float(metrics["peak_dbfs"]) - float(metrics["rms_dbfs"]),
        float(metrics["clipped_sample_fraction"]),
        float(metrics["absolute_dc_fraction"]),
    ]


def _context_vector(record: ClipRecord) -> list[float]:
    return [
        float(record.context["year"]),
        float(record.context["nest_x"]),
        float(record.context["nest_y"]),
    ]


def write_masked_clip(record: ClipRecord, destination: Path) -> Path:
    """Zero every frame inside an annotated note, keeping everything else exact.

    What is left is the recording with the song removed: the background, the
    distance, the microphone and whatever the annotations missed. A method that
    still separates the birds from this is separating recordings, not voices.
    """

    import numpy as np

    with wave.open(str(record.path), "rb") as source:
        channels = source.getnchannels()
        width = source.getsampwidth()
        rate = source.getframerate()
        frames = source.getnframes()
        payload = source.readframes(frames)
    if width != 2:
        raise ValueError(f"{record.path} is {width * 8}-bit; the mask is defined on 16-bit PCM")
    samples = np.frombuffer(payload, dtype="<i2").copy()
    annotation = record.context["annotation"]
    for onset, offset in zip(annotation["onsets"], annotation["offsets"], strict=True):
        start = max(0, int(math.floor((float(onset) - MASK_MARGIN_SECONDS) * rate)) * channels)
        end = min(
            len(samples), int(math.ceil((float(offset) + MASK_MARGIN_SECONDS) * rate)) * channels
        )
        if end > start:
            samples[start:end] = 0
    destination.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(destination), "wb") as target:
        target.setnchannels(channels)
        target.setsampwidth(width)
        target.setframerate(rate)
        target.writeframes(samples.tobytes())
    return destination


def representation_vectors(name: str, records: Sequence[ClipRecord], *, scratch: Path) -> Any:
    """The clip vectors for one representation, in the order the records are in."""

    import numpy as np

    from xinyenyana.a5 import birdnet_vectors
    from xinyenyana.representations import log_mel_summary

    # Every representation is assembled in single precision, which is what the
    # embedding models return and what the published numbers were computed in.
    # Widening the handcrafted ones alone would make them incomparable with the
    # measurements already on record.

    if name == BIRDNET:
        return birdnet_vectors(records)
    if name == MASKED_BIRDNET:
        masked = [
            ClipRecord(
                filename=record.filename,
                path=write_masked_clip(record, scratch / f"{record.filename}.wav"),
                identity=record.identity,
                split=record.split,
                context=record.context,
            )
            for record in records
        ]
        return birdnet_vectors(masked)
    if name == LOG_MEL:
        return np.vstack([log_mel_summary(record.path) for record in records])
    if name == CONTEXT:
        return np.asarray([_context_vector(record) for record in records], dtype=np.float32)
    if name == LEVEL_DURATION:
        return np.asarray([_level_vector(record) for record in records], dtype=np.float32)
    if name == SONG_STRUCTURE:
        return np.asarray([_annotation_vector(record) for record in records], dtype=np.float32)
    raise ValueError(f"unknown representation: {name}")


def run_open_set_endpoint(
    *,
    endpoint: Endpoint,
    scratch: Path,
    representations: Sequence[str] = REPRESENTATIONS,
) -> dict[str, Any]:
    """Run every representation through all sixteen allocations of the birds."""

    # Background recordings are not calls and never enter the open-set task.
    records = [
        record
        for record in endpoint.records
        if str(record.context.get("condition", "foreground")) == "foreground"
    ]
    identities_by_cohort: dict[str, list[str]] = {}
    for record in records:
        if "cohort" not in record.context:
            identities_by_cohort = {}
            break
        cohort = str(record.context["cohort"])
        if record.identity not in identities_by_cohort.setdefault(cohort, []):
            identities_by_cohort[cohort].append(record.identity)

    cohort_balanced = bool(identities_by_cohort)
    if cohort_balanced:
        allocations = [
            allocate_roles(identities_by_cohort, seed=ALLOCATION_SEED, allocation=index)
            for index in range(ALLOCATIONS)
        ]
    else:
        allocations = [
            allocate_roles_evenly(
                [record.identity for record in records], seed=ALLOCATION_SEED, allocation=index
            )
            for index in range(ALLOCATIONS)
        ]
    summary: dict[str, Any] = {
        "endpoint": endpoint.name,
        "identities": len(endpoint.identities),
        "clips": len(records),
        "manifest_sha256": endpoint.manifest_sha256,
        "allocation_seed": ALLOCATION_SEED,
        "allocations": ALLOCATIONS,
        "allocation_scheme": "year cohorts" if cohort_balanced else "even, no cohorts",
        "query_clips_per_bird": {
            identity: sum(1 for r in records if r.identity == identity and r.split == "query")
            for identity in sorted({r.identity for r in records})
        },
        "stranger_budgets": list(STRANGER_BUDGETS),
        "role_digest": canonical_sha256(
            [{role: roles[role] for role in ROLES} for roles in allocations]
        ),
        "representations": {},
    }
    for name in representations:
        vectors = representation_vectors(name, records, scratch=scratch / name)
        evaluated = [
            evaluate_allocation(
                vectors=vectors,
                records=records,
                roles=roles,
                standardise=name in STANDARDISED,
                balanced=cohort_balanced,
            )
            for roles in allocations
        ]
        summary["representations"][name] = {
            "dimension": int(vectors.shape[1]),
            "standardised": name in STANDARDISED,
            "allocation_sensitivity": summarise_allocations(evaluated),
            "allocations": evaluated,
        }
    return summary
