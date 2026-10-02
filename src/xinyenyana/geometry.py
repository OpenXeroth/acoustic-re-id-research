"""Whether individual identity and the recording session occupy the same directions.

The project has already measured that removing the recording from a BirdNET
embedding removes the animal with it: on the bat, the treatment that takes the
recording day from readable to invisible costs a tenth of the identity accuracy,
and every treatment that leaves the day where it was leaves identity where it
was too. That is a fact about what happens. This module asks why.

The question is geometric. A representation holds identity in some set of
directions and the recording session in some other set. If those two sets are
far apart, a linear method can remove one and keep the other, and the failure
above would be a defect of the particular methods tried. If they coincide, no
linear method can separate them, and the failure is a property of the
representation.

**The rank trap this module is built to avoid.** A between-class scatter matrix
over C classes has rank at most C - 1. Counting its eigen-directions therefore
counts the classes, not the representation: six bats give at most five
directions whatever the encoder does. Every subspace here is capped at that
rank, the cap is reported beside the figure, and no statement is made about how
much of a representation identity occupies from a number that the class count
fixed in advance. The compression curve, which uses principal components of the
clips rather than of the class means, carries no such cap and is the figure that
speaks to how compressible each signal is.

**Where this can run at all.** Both directions of crossing are needed: sessions
holding more than one individual, and individuals appearing in more than one
session. A dataset that gives each animal its own recordings cannot separate the
two by any method, and the check reports that rather than returning a number.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from xinyenyana.evaluation import PER_CLIP_L2, leave_one_identity_out, standardisation_for

#: Principal components kept in the compression curve. Unsupervised, so no class
#: count caps them.
COMPRESSION_DIMENSIONS: tuple[int, ...] = (1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024)

#: Random subspaces drawn to show what no alignment looks like at this width.
OVERLAP_SHUFFLES = 999

SEED = 17
RIDGE_LAMBDA = 1.0


def crossing(identities: Sequence[str], sessions: Sequence[str]) -> dict[str, Any]:
    """How far the session is crossed with identity rather than nested in it."""

    import collections

    by_session: dict[str, set[str]] = collections.defaultdict(set)
    by_identity: dict[str, set[str]] = collections.defaultdict(set)
    for identity, session in zip(identities, sessions, strict=True):
        by_session[session].add(identity)
        by_identity[identity].add(session)
    shared = sum(1 for names in by_session.values() if len(names) > 1)
    revisited = sum(1 for seen in by_identity.values() if len(seen) > 1)
    admissible = shared > 0 and revisited > 1
    return {
        "clips": len(identities),
        "identities": len(by_identity),
        "sessions": len(by_session),
        "sessions_holding_more_than_one_identity": shared,
        "identities_appearing_in_more_than_one_session": revisited,
        "admissible": admissible,
        "reason": (
            ""
            if admissible
            else (
                f"{shared} session(s) hold more than one identity and {revisited} "
                "identity/identities appear in more than one session; identity and "
                "session cannot be separated on this split by any method"
            )
        ),
    }


def _standardised(features: Any) -> Any:
    import numpy as np

    values = np.asarray(features, dtype=np.float64)
    lengths = np.maximum(np.linalg.norm(values, axis=1, keepdims=True), 1e-12)
    return values / lengths


def between_class_directions(features: Any, labels: Sequence[str], dimensions: int) -> Any:
    """The leading directions of the scatter between class means.

    Built identically for identity and for session so that the two subspaces are
    comparable. No whitening by a within-class covariance, because that would use
    a different matrix for each of the two and the angle between the results
    would then depend on which nuisance each was whitened against rather than on
    where the signals sit.
    """

    import numpy as np

    values = np.asarray(features, dtype=np.float64)
    names = np.asarray(labels, dtype=object)
    distinct = sorted(set(names.tolist()))
    centre = values.mean(axis=0, keepdims=True)
    means = np.vstack([values[names == name].mean(axis=0) for name in distinct])
    weights = np.asarray([float((names == name).sum()) for name in distinct])
    centred = (means - centre) * np.sqrt(weights)[:, None]
    scatter = (centred.T @ centred) / max(len(distinct) - 1, 1)
    _, directions = np.linalg.eigh(scatter)
    return directions[:, ::-1][:, :dimensions]


def subspace_overlap(first: Any, second: Any) -> dict[str, Any]:
    """How close two subspaces of equal dimension are, as cosines of their angles.

    One means the two subspaces coincide, zero means every direction of one is
    orthogonal to every direction of the other.
    """

    import numpy as np
    from scipy.linalg import subspace_angles

    angles = subspace_angles(np.asarray(first), np.asarray(second))
    cosines = np.cos(np.asarray(angles, dtype=np.float64))
    return {
        "dimensions": int(np.asarray(first).shape[1]),
        "cosines": [float(value) for value in sorted(cosines.tolist(), reverse=True)],
        "mean_cosine": float(cosines.mean()),
        "largest_cosine": float(cosines.max()),
    }


def run_geometry(
    *,
    features: Any,
    identities: Sequence[str],
    sessions: Sequence[str],
    representation: str,
    shuffles: int = OVERLAP_SHUFFLES,
    seed: int = SEED,
) -> dict[str, Any]:
    """The whole measurement on one split of one endpoint and one representation."""

    import collections

    import numpy as np

    def majority_share(labels: Sequence[str]) -> float:
        """What always answering the commonest label would score."""

        counts = collections.Counter(labels)
        total = sum(counts.values())
        return float(max(counts.values()) / total) if total else 0.0

    structure = crossing(identities, sessions)
    report: dict[str, Any] = {
        "representation": representation,
        "standardisation": PER_CLIP_L2,
        "seed": seed,
        "crossing": structure,
        "construction": (
            "leading directions of the scatter between class means, on per-clip "
            "L2 standardised vectors, identity and session built identically"
        ),
    }
    if not structure["admissible"]:
        report["measured"] = False
        return report

    unit = _standardised(features)
    identity_cap = structure["identities"] - 1
    session_cap = structure["sessions"] - 1
    dimensions = int(min(identity_cap, session_cap))
    report["identity_rank_cap"] = identity_cap
    report["session_rank_cap"] = session_cap
    report["subspace_dimensions"] = dimensions
    report["rank_note"] = (
        f"a between-class scatter over C classes has rank at most C - 1, so the "
        f"identity subspace here cannot exceed {identity_cap} directions whatever "
        f"the representation holds; both subspaces are taken at {dimensions} so "
        "that the angle between them is defined"
    )

    identity_space = between_class_directions(unit, identities, dimensions)
    session_space = between_class_directions(unit, sessions, dimensions)
    observed = subspace_overlap(identity_space, session_space)

    # A cosine needs something to be read against. The reference used here is a
    # subspace of the same width drawn uniformly at random in the same number of
    # dimensions, which is what no alignment looks like.
    #
    # **A reference that was tried and removed, because it cannot work.** The
    # obvious null is to permute the session labels across clips and rebuild the
    # session subspace from the shuffled labels. On a fixture whose session was
    # planted orthogonal to identity, the true overlap was 0.011 and the
    # shuffled reference was 0.666: far higher than the truth it was meant to
    # bound. The reason is structural rather than a defect of the fixture. The
    # means of randomly chosen groups of clips vary mostly along whichever
    # directions carry the most variance in the data, and in an
    # identity-labelled corpus those are the identity directions. So any
    # labelling of these clips, related to the recording or not, produces a
    # subspace already aligned with identity, and the null is high whatever the
    # answer is. It is not reported.
    generator = np.random.default_rng(seed)
    random_means = np.empty(shuffles, dtype=np.float64)
    width = unit.shape[1]
    for index in range(shuffles):
        drawn = np.linalg.qr(generator.normal(size=(width, dimensions)))[0]
        random_means[index] = subspace_overlap(identity_space, drawn)["mean_cosine"]
    report["overlap"] = {
        **observed,
        "draws": shuffles,
        "unaligned_reference_median": float(np.median(random_means)),
        "unaligned_reference_95": [
            float(np.percentile(random_means, 2.5)),
            float(np.percentile(random_means, 97.5)),
        ],
        "reference": (
            "a subspace of the same width drawn uniformly at random in the same "
            "number of dimensions, which is what no alignment looks like"
        ),
        "claim_boundary": (
            "this cosine is descriptive. It says how close the two sets of "
            "directions are and it does not on its own establish that a linear "
            "method cannot separate them; the probe below is what establishes "
            "that"
        ),
    }

    # The sharpest statement available: can the recording be read out of the
    # directions built only to separate the animals? If it can, no linear removal
    # of the recording can leave identity untouched, because there is nothing to
    # remove that identity does not also occupy.
    identity_coordinates = unit @ identity_space
    probe = leave_one_identity_out(
        features=identity_coordinates,
        identities=np.asarray(identities, dtype=object),
        targets={"session": np.asarray(sessions, dtype=object)},
        categorical=frozenset({"session"}),
        standardisation=standardisation_for(dimensions),
        ridge_lambda=RIDGE_LAMBDA,
    )["session"]
    report["session_read_from_identity_directions"] = {
        **probe,
        "coordinates": dimensions,
        "means": (
            "the session predicted from the coordinates of each clip in the "
            "identity subspace alone, by the registered leave-one-identity-out "
            "probe, so a session predictable only because one animal was present "
            "cannot produce it"
        ),
    }

    # And the reverse, for symmetry.
    session_coordinates = unit @ session_space
    reverse = leave_one_identity_out(
        features=session_coordinates,
        identities=np.asarray(sessions, dtype=object),
        targets={"identity": np.asarray(identities, dtype=object)},
        categorical=frozenset({"identity"}),
        standardisation=standardisation_for(dimensions),
        ridge_lambda=RIDGE_LAMBDA,
    )["identity"]
    report["identity_read_from_session_directions"] = {**reverse, "coordinates": dimensions}

    # How compressible each signal is, with no rank cap: principal components of
    # the clips rather than of the class means.
    centred = unit - unit.mean(axis=0, keepdims=True)
    _, _, components = np.linalg.svd(centred, full_matrices=False)
    curve = []
    for keep in COMPRESSION_DIMENSIONS:
        if keep > components.shape[0]:
            continue
        projected = centred @ components[:keep].T
        identity_probe = leave_one_identity_out(
            features=projected,
            identities=np.asarray(sessions, dtype=object),
            targets={"identity": np.asarray(identities, dtype=object)},
            categorical=frozenset({"identity"}),
            standardisation=standardisation_for(keep),
            ridge_lambda=RIDGE_LAMBDA,
        )["identity"]
        session_probe = leave_one_identity_out(
            features=projected,
            identities=np.asarray(identities, dtype=object),
            targets={"session": np.asarray(sessions, dtype=object)},
            categorical=frozenset({"session"}),
            standardisation=standardisation_for(keep),
            ridge_lambda=RIDGE_LAMBDA,
        )["session"]
        curve.append(
            {
                "components": keep,
                "identity_accuracy": identity_probe["accuracy"],
                "identity_majority_class_rate": identity_probe["majority_class_rate"],
                "session_accuracy": session_probe["accuracy"],
                "session_majority_class_rate": session_probe["majority_class_rate"],
            }
        )
    report["compression_curve"] = curve
    report["compression_note"] = (
        "principal components of the clips, not of the class means, so no class "
        "count caps this; each row asks what survives when the representation is "
        "cut to that many numbers"
    )
    report["standardisation_note"] = (
        "every probe here calls standardisation_for on the width it is handed, "
        "which is the rule the classifier uses, so nothing in this module is "
        "produced by a preparation the rest of the project does not apply"
    )
    report["identity_majority_class_rate"] = majority_share(identities)
    report["session_majority_class_rate"] = majority_share(sessions)
    report["measured"] = True
    return report
