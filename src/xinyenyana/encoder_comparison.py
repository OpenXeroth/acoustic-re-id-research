"""Whether one encoder beats another, tested rather than read off a ranking.

A sweep reports one accuracy per encoder per endpoint. Ordering those numbers
says nothing about whether the order would survive a different draw of
animals. Two tests answer the two questions a reader asks of the ordering:

**Is an encoder better than BirdNET on this endpoint?** The same animals are
resampled for both encoders in every bootstrap draw, so the difference is
paired: an animal that happens to be easy is easy for both, and what remains is
the encoder. Holm's correction is applied across the comparisons made on one
endpoint, because choosing the best of many encoders is where chance finds a
winner.

**Does the ranking carry from one species to the next?** Friedman's test on the
ranks across endpoints, with Kendall's coefficient of concordance W, which is 0
when the endpoints rank the encoders independently and 1 when every endpoint
ranks them identically. Nemenyi's critical difference says how far apart two
average ranks must be before the difference between them is more than the
spread of ranks would give by chance. This is a statement about consistency of
ranks, and testing more encoders does not make it more likely to come out, so
it takes no multiplicity correction.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

REFERENCE = "birdnet-v2.4"
BOOTSTRAP_REPLICATES = 2000


def paired_bootstrap(
    *,
    identities: Sequence[str],
    correct: Mapping[str, str],
    reference: str = REFERENCE,
    replicates: int = BOOTSTRAP_REPLICATES,
    seed: int = 17,
) -> dict[str, Any]:
    """Every representation against the reference, over the same animal draws.

    ``correct`` maps a representation to a string of ``0`` and ``1``, one per
    query clip in the order ``identities`` lists them.
    """

    import numpy as np

    if reference not in correct:
        raise ValueError(f"the reference {reference!r} was not scored")
    labels = np.asarray(identities)
    animals = sorted(set(identities))
    rows = [np.flatnonzero(labels == animal) for animal in animals]
    matrix = {}
    for name, flags in correct.items():
        if len(flags) != len(identities):
            raise ValueError(f"{name} scored {len(flags)} clips, not {len(identities)}")
        matrix[name] = np.frombuffer(flags.encode(), dtype=np.uint8) == ord("1")
    rng = np.random.default_rng(seed)
    names = sorted(matrix)
    # One row per draw, one column per representation. The draws are made one
    # at a time and not kept, so 10,000 of them over thousands of clips fit in
    # memory; the sequence of draws is the one the list form produced.
    resampled = np.empty((replicates, len(names)))
    for index in range(replicates):
        draw = np.concatenate([rows[i] for i in rng.integers(0, len(animals), size=len(animals))])
        resampled[index] = [matrix[name][draw].mean() for name in names]
    column = {name: resampled[:, position] for position, name in enumerate(names)}
    reference_accuracy = column[reference]
    comparisons: dict[str, Any] = {}
    for name, flags in sorted(matrix.items()):
        if name == reference:
            continue
        differences = column[name] - reference_accuracy
        observed = float(flags.mean() - matrix[reference].mean())
        below = float(np.mean(differences <= 0))
        above = float(np.mean(differences >= 0))
        comparisons[name] = {
            "accuracy": float(flags.mean()),
            "accuracy_95": [float(v) for v in np.quantile(column[name], [0.025, 0.975])],
            "difference_from_reference": observed,
            "difference_95": [float(v) for v in np.quantile(differences, [0.025, 0.975])],
            "p_two_sided": min(1.0, 2 * min(below, above)),
        }
    holm(comparisons)
    return {
        "reference": reference,
        "reference_accuracy": float(matrix[reference].mean()),
        "reference_accuracy_95": [
            float(v) for v in np.quantile(reference_accuracy, [0.025, 0.975])
        ],
        "animals": len(animals),
        "query_clips": len(identities),
        "replicates": replicates,
        "method": "animals resampled with replacement; the same draw scores every representation",
        "comparisons": comparisons,
    }


def holm(comparisons: dict[str, dict[str, Any]], *, alpha: float = 0.05) -> None:
    """Holm's step-down adjustment, written into each comparison in place."""

    ordered = sorted(comparisons, key=lambda name: comparisons[name]["p_two_sided"])
    count = len(ordered)
    running = 0.0
    for rank, name in enumerate(ordered):
        adjusted = min(1.0, (count - rank) * comparisons[name]["p_two_sided"])
        running = max(running, adjusted)
        comparisons[name]["p_holm"] = running
        comparisons[name]["differs_at_0.05_after_holm"] = running < alpha


def rank_consistency(accuracy: Mapping[str, Mapping[str, float]]) -> dict[str, Any]:
    """Friedman's test, Kendall's W and Nemenyi's critical difference.

    ``accuracy`` maps an endpoint to a mapping of representation to accuracy.
    Only representations scored on every endpoint enter, and the ones left out
    are named.
    """

    import math

    import numpy as np
    from scipy.stats import friedmanchisquare, rankdata, studentized_range

    endpoints = sorted(accuracy)
    shared = sorted(set.intersection(*(set(accuracy[e]) for e in endpoints)))
    left_out = sorted(set().union(*(set(accuracy[e]) for e in endpoints)) - set(shared))
    n, k = len(endpoints), len(shared)
    if n < 2 or k < 3:
        return {"computed": False, "reason": f"{n} endpoints and {k} shared representations"}
    table = np.asarray([[accuracy[e][r] for r in shared] for e in endpoints])
    ranks = np.vstack([rankdata(-row) for row in table])
    statistic, p_value = friedmanchisquare(*table.T)
    w = float(statistic) / (n * (k - 1))
    q = float(studentized_range.ppf(0.95, k, 1e6)) / math.sqrt(2)
    critical = q * math.sqrt(k * (k + 1) / (6.0 * n))
    average = {name: float(value) for name, value in zip(shared, ranks.mean(axis=0), strict=True)}
    return {
        "computed": True,
        "endpoints": endpoints,
        "representations": shared,
        "left_out_not_scored_everywhere": left_out,
        "friedman_chi_square": float(statistic),
        "friedman_p": float(p_value),
        "kendall_w": w,
        "average_rank": dict(sorted(average.items(), key=lambda item: item[1])),
        "nemenyi_critical_difference_0.05": critical,
    }


def summarise_sweep(
    sweep: Mapping[str, Any], *, replicates: int = BOOTSTRAP_REPLICATES
) -> dict[str, Any]:
    """Selected, final-layer and best-layer figures, and the paired test, for one sweep.

    The best layer is chosen with the query labels and is reported so a reader
    can see what selection costs; it is not attainable in practice and is never
    the reported figure.
    """

    per_model: dict[str, Any] = {}
    for model, curve in sweep["curve"].items():
        if not curve or model not in sweep["chosen"]:
            continue
        chosen = sweep["chosen"][model]
        layered = [entry for entry in curve.values() if entry.get("layer") is not None]
        final = None
        if layered:
            at_rate = [e for e in layered if e["slowdown"] == chosen["slowdown"]]
            final = max(at_rate, key=lambda entry: entry["layer"])
        oracle = max(curve.values(), key=lambda entry: entry["accuracy"])
        per_model[model.split("/")[-1]] = {
            "selected": chosen["representation"],
            "selected_accuracy": chosen["accuracy"],
            "final_layer_at_selected_rate": None if final is None else final["accuracy"],
            "best_layer_chosen_with_query_labels": oracle["accuracy"],
        }
    correct = {
        model.split("/")[-1]: entry["query_correct"]
        for model, entry in sweep["chosen"].items()
        if "query_correct" in entry
    }
    paired = (
        paired_bootstrap(
            identities=sweep["query_identities"], correct=correct, replicates=replicates
        )
        if correct and "query_identities" in sweep
        else None
    )
    return {"endpoint": sweep["endpoint"], "selection": per_model, "paired": paired}
