"""Guard the comparison's independence, counts and alternative gallery rule."""

from pathlib import Path

import numpy as np
import pytest

from xinyenyana.a2 import ClipRecord
from xinyenyana.open_set import normalise, score_against_gallery
from xinyenyana.review_sensitivity import matched_rows, score_split, split_summary


def records():
    return [
        ClipRecord(
            f"{bird}-{i}",
            Path("unused"),
            bird,
            "enrollment" if i < n else "query",
            {"condition": "foreground"},
        )
        for bird, n in (("a", 2), ("b", 3))
        for i in range(5)
    ]


def test_matched_split_preserves_counts_and_no_clip_crosses_both_sides():
    rs = records()
    for seed in range(101, 111):
        train, query = matched_rows(rs, seed)
        assert set(train).isdisjoint(query)
        assert sorted(train + query) == list(range(len(rs)))
        assert [sum(rs[i].identity == b for i in train) for b in ("a", "b")] == [2, 3]
        assert [sum(rs[i].identity == b for i in query) for b in ("a", "b")] == [3, 2]


def test_summary_detects_unequal_counts_and_identical_split_is_zero():
    rs = records()
    v = np.random.default_rng(3).normal(size=(10, 4))
    base = score_split(rs, v, [0, 1, 5, 6, 7], [2, 3, 4, 8, 9])
    for s in split_summary(base, [base] * 10).values():
        assert s["difference_mean"] == 0
        assert s["difference_95_individual_bootstrap"] == [0, 0]
    with pytest.raises(ValueError, match="counts"):
        split_summary(base, [{**base, "query_counts_per_individual": [2, 3]}])


def test_nearest_gallery_uses_only_enrolment_and_preserves_identity_mapping():
    rs = [
        ClipRecord(str(i), Path("unused"), bird, split, {})
        for i, (bird, split) in enumerate(
            [
                ("a", "enrollment"),
                ("a", "enrollment"),
                ("b", "enrollment"),
                ("a", "query"),
                ("stranger", "query"),
            ]
        )
    ]
    values = normalise(
        np.array([[1.0, 0], [0.0, 1], [0.8, 0.6], [1.0, 0], [-1.0, 0]]),
        [0, 1, 2],
        standardise=False,
    )
    kw = dict(
        values=values,
        records=rs,
        known=["a", "b"],
        unknown=["stranger"],
        partition="test",
        balanced=False,
    )
    with pytest.raises(ValueError, match="no query"):
        score_against_gallery(**kw, scorer="nearest-clip")
    rs.append(ClipRecord("5", Path("unused"), "b", "query", {}))
    kw["values"] = np.vstack([values, values[2]])
    mean = score_against_gallery(**kw)
    near = score_against_gallery(**kw, scorer="nearest-clip")
    assert mean[0]["top_identity"] == "b"
    assert near[0]["top_identity"] == "a"
    assert near[0]["maximum_score"] == 1
    assert near[1]["maximum_score"] == 0
