"""Render historical replay counts and explicit whole-model arithmetic contexts."""

import argparse
import hashlib
import json
from pathlib import Path

from paper_figures import md_table
from paper_v6_controls import complete_gate
from paper_v6_evidence import ENDPOINTS


def checked(path, digest):
    raw = path.read_bytes()
    assert hashlib.sha256(raw).hexdigest() == digest, path.name
    return json.loads(raw)


def counts(row):
    # Replay records written before 2026-09-28 carry no reported_differing field.
    assert row["passed"] and not row["missing"] and not row.get("reported_differing")
    assert row["candidates_expected"] > 0
    assert row["candidates_expected"] == row["candidates_compared"]
    assert row["identical"] + len(row["differing"]) == row["candidates_expected"]
    return {
        **{k: row[k] for k in ("candidates_expected", "candidates_compared", "identical")},
        "differing": row["differing"],
    }


def corrected_models(results, endpoint, sources, archived):
    name = f"v6-v5sweep-{endpoint}.json"
    data = checked(results / name, sources[name])
    correction = data.get("replay_correction")
    if correction is None:
        return []
    counts(correction["reproduction"])
    raw_models = {}
    for original, digest in correction["sources"].items():
        base = Path(original).name
        if not base.startswith(("v6-fp32-", "v6-tf32-")):
            continue
        source = f"precision/{base}"
        raw = checked(results / source, digest)
        assert archived[source]["archive_verified"] and archived[source]["sha256"] == digest
        assert len(raw["models"]) == 1
        model = raw["models"][0]
        assert model not in raw_models
        raw_models[model] = (source, digest, raw)
    assert set(raw_models) == set(correction["per_model"])
    rows = []
    for model, result in correction["per_model"].items():
        count = counts(result)
        source, digest, raw = raw_models[model]
        curve = data["curve"][model]
        assert set(curve) == set(raw["curve"][model])
        contexts = {json.dumps(c["numeric_precision"], sort_keys=True) for c in curve.values()}
        assert len(contexts) == 1
        context = json.loads(contexts.pop())
        assert context["speech_matmul"] in ("fp32", "tf32")
        assert context["cuda_matmul_allow_tf32"] == (context["speech_matmul"] == "tf32")
        assert context["cudnn_allow_tf32"] is True
        for candidate, current in curve.items():
            original = raw["curve"][model][candidate]
            assert original["numeric_precision"] == context
            assert original["query_correct"] == current["query_correct"]
            assert abs(original["accuracy"] - current["accuracy"]) < 1e-12
        rows.append(
            {
                "endpoint": endpoint,
                "model": model,
                **count,
                "measured_candidates": len(curve),
                "numeric_precision": context,
                "corrective_source": source,
                "corrective_sha256": digest,
                "reconstructed_source": name,
                "reconstructed_sha256": sources[name],
            }
        )
    return rows


def render(results, evidence_path, out, supplemental_path):
    evidence = complete_gate(evidence_path)
    sources = evidence["sources"]
    reproduction = checked(results / "v6-v5-reproduction.json", sources["v6-v5-reproduction.json"])
    assert set(reproduction["per_endpoint"]) == set(ENDPOINTS[:-2])
    archive = json.loads(supplemental_path.read_text())
    assert archive["complete_inventory_checked"]
    archived = {r["name"]: r for r in archive["objects"]}
    endpoint_rows = {}
    repairs = []
    for endpoint in ENDPOINTS[:-2]:
        row = reproduction["per_endpoint"][endpoint]
        name = f"v6-v5sweep-{endpoint}.json"
        # Reproduction is run on the refreshed result used for final selection.
        assert row["v6"]["sha256"] == sources[name], endpoint
        endpoint_rows[endpoint] = {
            **counts(row),
            "historical_sha256": row["v5"]["sha256"],
            "replayed_sha256": row["v6"]["sha256"],
        }
        repairs.extend(corrected_models(results, endpoint, sources, archived))
    text = "# Historical reproduction and arithmetic contexts\n\n"
    text += (
        "Every historical candidate was checked for an identical per-query correctness "
        "string and accuracy agreement within 1e-12. This audits historical candidates; "
        "it does not claim that the expanded candidate inventory was measured in v5.\n\n"
    )
    text += md_table(
        ["Endpoint", "Expected", "Compared", "Identical"],
        [
            [ep, r["candidates_expected"], r["candidates_compared"], r["identical"]]
            for ep, r in endpoint_rows.items()
        ],
    )
    text += "\n\nWhole-model corrective replays used the following declared contexts.\n\n"
    text += md_table(
        [
            "Endpoint",
            "Model",
            "Historical candidates matched",
            "Matrix arithmetic",
            "CUDA matrix TF32 allowed",
            "CuDNN TF32 allowed",
        ],
        [
            [
                r["endpoint"],
                r["model"].split("/")[-1],
                r["identical"],
                r["numeric_precision"]["speech_matmul"].upper(),
                str(r["numeric_precision"]["cuda_matmul_allow_tf32"]),
                str(r["numeric_precision"]["cudnn_allow_tf32"]),
            ]
            for r in repairs
        ],
    )
    text += (
        "\n\nOne context was applied to each entire model sweep, rather than chosen by "
        "layer or playback rate. FP32 denotes speech matrix arithmetic: CuDNN TF32 "
        "remained permitted. These are documented contexts sufficient to reproduce "
        "archived decisions, not evidence of the unrecorded flags in the original "
        "process. Foreground-only Fisher reselection is a separate analysis.\n"
    )
    out.mkdir(parents=True, exist_ok=True)
    (out / "historical-reproduction.md").write_text(text)
    (out / "historical-reproduction.json").write_text(
        json.dumps(
            {
                "reproduction_source_sha256": sources["v6-v5-reproduction.json"],
                "endpoints": endpoint_rows,
                "corrective_replays": repairs,
            },
            indent=2,
        )
        + "\n"
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, required=True)
    parser.add_argument("--evidence", type=Path, required=True)
    parser.add_argument("--supplemental-archive", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    render(args.results, args.evidence, args.out, args.supplemental_archive)


if __name__ == "__main__":
    main()
