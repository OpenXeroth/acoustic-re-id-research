"""Build the E02 full-width RookID endpoint: cut every admitted call on every channel."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from xinyenyana.e02 import DEFAULT_PROTOCOL, cut_recording, load_e02_protocol, select_e02_candidates
from xinyenyana.endpoints import benchmark_root, extract_rookid_annotations
from xinyenyana.rookid import read_rookid_annotations

ARCHIVE = "rookid-zenodo-6091940/RookID.zip"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--protocol", type=Path, default=DEFAULT_PROTOCOL)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument(
        "--only", type=str, default=None, help="comma-separated stems, for a smoke test"
    )
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args()

    protocol, protocol_sha = load_e02_protocol(args.protocol)
    root = benchmark_root()
    archive = root / ARCHIVE
    args.out.mkdir(parents=True, exist_ok=True)

    recordings = protocol["source"]["recordings"]
    if args.only:
        wanted = set(args.only.split(","))
        recordings = [r for r in recordings if r["stem"] in wanted]
    if args.limit:
        recordings = recordings[: args.limit]

    # read_rookid_annotations carries E01d's per-recording split through onto each
    # event. E02 assigns splits from the day and the year instead, so a placeholder
    # goes in here and the real split is set when an endpoint is cut from the manifest.
    reading = {"source": {"recordings": [dict(r, split="unassigned") for r in recordings]}}
    extract_rookid_annotations(protocol=reading, destination=args.out, root=root)
    annotations = read_rookid_annotations(protocol=reading, source_root=args.out)

    clip_root = args.out / "clips"
    entries: list[dict] = []
    started = time.time()
    for n, recording in enumerate(recordings, 1):
        stem = str(recording["stem"])
        candidates = select_e02_candidates(
            protocol=protocol, events=annotations[stem], recording=stem
        )
        if not candidates:
            print(f"[{n}/{len(recordings)}] {stem}: no admitted calls", flush=True)
            continue
        t0 = time.time()
        produced = list(
            cut_recording(
                protocol=protocol,
                archive=archive,
                member=f"RookID/{stem}.wav",
                candidates=candidates,
                clip_root=clip_root,
            )
        )
        entries.extend(produced)
        gb = int(recording["wav_bytes"]) / 1e9
        print(
            f"[{n}/{len(recordings)}] {stem}: {len(candidates)} calls -> {len(produced)} clips "
            f"({gb:.2f} GB in {time.time() - t0:.0f}s, total {len(entries)} clips, "
            f"{time.time() - started:.0f}s elapsed)",
            flush=True,
        )
        manifest = {
            "endpoint": "rookid-full-width",
            "experiment": "E02",
            "protocol_sha256": protocol_sha,
            "archive": ARCHIVE,
            "recordings_done": n,
            "recordings_total": len(recordings),
            "entries": entries,
        }
        (args.out / "e02-manifest.json").write_text(json.dumps(manifest))
    print(
        f"DONE {len(entries)} clips from {len(recordings)} recordings in "
        f"{time.time() - started:.0f}s",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
