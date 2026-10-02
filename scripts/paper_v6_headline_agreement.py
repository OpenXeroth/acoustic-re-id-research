"""Link comparison references to the exact foreground headline observations."""

import argparse
import hashlib
import json
from pathlib import Path

from paper_v6_controls import complete_gate
from paper_v6_evidence import ENDPOINTS


def verify(results, headline, evidence_path):
    evidence = complete_gate(evidence_path)
    rows = []

    def load(path):
        raw = path.read_bytes()
        sha = hashlib.sha256(raw).hexdigest()
        assert evidence["sources"][path.name] == sha, path.name
        return json.loads(raw), sha

    for ep in ENDPOINTS:
        stem = "rook-full-across-year-capped-90" if ep == "rookid-full-width" else ep
        hpath = (results if ep == "great-tit-full" else headline) / f"v6-headline-{stem}.json"
        group = "v5new" if ep in ENDPOINTS[-2:] else "merged"
        spath = results / f"v6-{group}-{ep}.json"
        h, hs = load(hpath)
        s, ss = load(spath)
        r = s["curve"]["birdnet-v2.4"][s["chosen"]["birdnet-v2.4"]["representation"]]
        assert h["manifest_sha256"] == s["manifest_sha256"], ep
        assert h["query_identities"] == s["query_identities"], ep
        assert h["query_correct"] == r["query_correct"], ep
        assert abs(h["evaluation"]["classification"]["accuracy"] - r["accuracy"]) < 1e-12, ep
        rows.append(
            {
                "endpoint": ep,
                "headline_sha256": hs,
                "sweep_sha256": ss,
                "query_clips": len(s["query_identities"]),
                "identical_query_correctness": True,
                "accuracy": r["accuracy"],
            }
        )
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, required=True)
    parser.add_argument("--headline", type=Path, required=True)
    parser.add_argument("--evidence", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    rows = verify(args.results, args.headline, args.evidence)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(rows, indent=2) + "\n")


if __name__ == "__main__":
    main()
