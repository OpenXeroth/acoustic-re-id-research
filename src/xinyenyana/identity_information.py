"""Beecher's information statistic on an embedding, in bits.

Beecher's statistic (Beecher 1989) is the field's standard measure of how much
identity information a set of signal features carries. Linhart et al. (2019,
Methods in Ecology and Evolution 10: 1558-1570) compared seven identity
metrics and recommended it. It is defined on independent variables: for each
one, a one-way analysis of variance across individuals gives an F ratio, and
the variable contributes ``log2(sqrt((F + n0 - 1) / n0))`` bits, where ``n0``
is the number of calls per individual. The contributions are summed.

An embedding has hundreds of correlated dimensions, so the variables here are
its principal components, which are uncorrelated over the clips they are
fitted on. Linhart et al. report that the statistic is biased upwards when the
number of variables approaches the number of individuals, so the number of
components used is never more than the number of individuals less one, and the
whole curve over the number of components is reported, not one point of it.

Rules registered in ``docs/measurement-protocol.md`` (PA-V6):

* the clips are every foreground clip of the endpoint, enrolment and scoring;
* each clip vector is scaled to unit length, as the identification head does,
  before the components are fitted;
* with unequal numbers of calls per individual, ``n0`` is the effective group
  size of an unbalanced one-way analysis of variance,
  ``(N - sum(n_i^2) / N) / (K - 1)``;
* a component whose F ratio is below one contributes nothing, because a
  negative number of bits of identity is not defined. This rule is ours and is
  stated as ours.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any


def effective_group_size(counts: Sequence[int]) -> float:
    """``n0`` of an unbalanced one-way ANOVA; equals the group size when balanced."""

    total = float(sum(counts))
    groups = len(counts)
    if groups < 2:
        raise ValueError("an information statistic needs at least two individuals")
    return (total - sum(float(c) ** 2 for c in counts) / total) / (groups - 1)


def f_ratios(scores: Any, labels: Sequence[str]) -> Any:
    """One-way ANOVA F ratio of every column of ``scores`` across ``labels``."""

    import numpy as np

    values = np.asarray(scores, dtype=np.float64)
    names = np.asarray(labels)
    groups = sorted(set(names.tolist()))
    total = values.shape[0]
    grand = values.mean(axis=0)
    between = np.zeros(values.shape[1])
    within = np.zeros(values.shape[1])
    for group in groups:
        rows = values[names == group]
        mean = rows.mean(axis=0)
        between += rows.shape[0] * (mean - grand) ** 2
        within += ((rows - mean) ** 2).sum(axis=0)
    df_between = len(groups) - 1
    df_within = total - len(groups)
    if df_within <= 0:
        raise ValueError("every individual needs more than one call")
    mean_within = within / df_within
    mean_between = between / df_between
    with np.errstate(divide="ignore", invalid="ignore"):
        ratio = np.where(mean_within > 0, mean_between / mean_within, 0.0)
    return ratio


def beecher_hs(vectors: Any, identities: Sequence[str]) -> dict[str, Any]:
    """Beecher's statistic over the leading principal components of an embedding."""

    import numpy as np

    values = np.asarray(vectors, dtype=np.float64)
    labels = list(identities)
    if values.shape[0] != len(labels):
        raise ValueError("one identity per vector")
    groups = sorted(set(labels))
    counts = [labels.count(group) for group in groups]
    n0 = effective_group_size(counts)
    unit = values / np.maximum(np.linalg.norm(values, axis=1, keepdims=True), 1e-12)
    centred = unit - unit.mean(axis=0, keepdims=True)
    _, singular, rows = np.linalg.svd(centred, full_matrices=False)
    rank = int(np.sum(singular > singular.max() * 1e-10)) if len(singular) else 0
    limit = min(len(groups) - 1, rank)
    if limit < 1:
        raise ValueError("no principal component carries variance")
    scores = centred @ rows[:limit].T
    ratio = f_ratios(scores, labels)
    bits = np.where(ratio > 1.0, np.log2(np.sqrt((ratio + n0 - 1.0) / n0)), 0.0)
    curve = np.cumsum(bits)
    return {
        "individuals": len(groups),
        "clips": int(values.shape[0]),
        "n0": n0,
        "components_used": limit,
        "hs_bits": float(curve[-1]),
        "hs_bits_by_components": [float(v) for v in curve],
        "f_ratio_by_component": [float(v) for v in ratio],
        "variance_share_by_component": [
            float(v) for v in (singular[:limit] ** 2 / np.sum(singular**2))
        ],
    }
