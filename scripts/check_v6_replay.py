"""Bounded forensic inference check, to run only under a xen1 lease.

No layer or decision gate is selected. Compare the legacy forward path with
v6's minimum-input probe, then repeat the latter in a newly loaded model.
"""

import argparse
import hashlib
import json
import math
from pathlib import Path


def loading_metadata_json(value):
    """Transformers can return sets of missing/unexpected weight keys."""
    if isinstance(value, set):
        return sorted(value)
    raise TypeError(f"Unsupported loading metadata: {type(value).__name__}")


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--sample-root", type=Path, required=True)
    p.add_argument("--endpoint", default="great-tit")
    p.add_argument("--sweep", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--model", default="facebook/mms-300m")
    p.add_argument("--slowdown", type=int, default=3)
    p.add_argument("--clips", type=int, default=32)
    p.add_argument("--max-input-seconds", type=float)
    args = p.parse_args()
    import numpy as np
    import torch
    from transformers import AutoModel

    from xinyenyana.a5 import _load_batch, minimum_samples, pad_to
    from xinyenyana.a6 import load_candidate
    from xinyenyana.archive import sha256_file
    from xinyenyana.cli import _require_lease_and_archive, _snapshot_provenance
    from xinyenyana.cli_v6 import _emit, _endpoint

    _require_lease_and_archive()
    _snapshot_provenance()
    if not 1 <= args.clips <= 64 or args.slowdown not in (1, 2, 3):
        p.error("use 1 to 64 clips and a registered playback rate")
    if args.max_input_seconds is not None and (
        not math.isfinite(args.max_input_seconds) or args.max_input_seconds <= 0
    ):
        p.error("maximum input duration must be positive and finite")
    endpoint = _endpoint(args.endpoint, args.sample_root, None)
    eligible = list(range(len(endpoint.records)))
    if args.max_input_seconds is not None:
        import soundfile

        eligible = [
            i
            for i in eligible
            if soundfile.info(endpoint.records[i].path).duration * args.slowdown
            <= args.max_input_seconds
        ]
    if not eligible:
        p.error("no clips satisfy the pre-inference duration bound")
    indices = np.asarray(eligible)[
        np.linspace(0, len(eligible) - 1, min(args.clips, len(eligible)), dtype=int)
    ]
    records = [endpoint.records[i] for i in indices]
    sweep = json.loads(args.sweep.read_text())
    if sweep["endpoint"] != endpoint.name or sweep["manifest_sha256"] != endpoint.manifest_sha256:
        raise ValueError("source sweep describes another endpoint or manifest")
    source = sweep["candidate_vectors"][args.model]
    if sha256_file(Path(source["path"])) != source["sha256"]:
        raise ValueError("candidate store digest mismatch")
    result = {
        "purpose": (
            "Compare legacy inference, v6 short-input probing, and a repeated v6 load; "
            "no decision gates changed"
        ),
        "script_sha256": sha256_file(Path(__file__)),
        "script_source": Path(__file__).read_text(),
        "endpoint": endpoint.name,
        "manifest_sha256": endpoint.manifest_sha256,
        "source_sweep_sha256": sha256_file(args.sweep),
        "model": args.model,
        "slowdown": args.slowdown,
        "clip_selection": "equally spaced eligible record indices, fixed before inference",
        "eligible_records": len(eligible),
        "max_input_seconds": args.max_input_seconds,
        "record_indices": indices.tolist(),
        "audio_sha256": [sha256_file(r.path) for r in records],
        "torch": str(torch.__version__),
        "cudnn": torch.backends.cudnn.version(),
        "cudnn_benchmark": torch.backends.cudnn.benchmark,
        "cudnn_deterministic": torch.backends.cudnn.deterministic,
        "cudnn_allow_tf32": torch.backends.cudnn.allow_tf32,
        "matmul_allow_tf32": torch.backends.cuda.matmul.allow_tf32,
        "seed_before_each_model_load": 17,
        "passes": {},
    }
    matrices = {}
    for label, probe in [("legacy", False), ("probed", True), ("probed_repeat", True)]:
        torch.manual_seed(17)
        model, loading = AutoModel.from_pretrained(
            args.model, cache_dir="/mnt/data/xinyenyana/models/hf", output_loading_info=True
        )
        model = model.to("cuda").eval()

        def forward(samples, encoder=model):
            tensor = torch.from_numpy(np.ascontiguousarray(samples))[None, :].to("cuda")
            with torch.no_grad():
                return encoder(tensor, output_hidden_states=True).hidden_states

        shortest = minimum_samples(forward) if probe else 0
        collected = {}
        for signal in _load_batch(records, args.slowdown):
            hidden = forward(pad_to(signal, shortest))
            for layer, value in enumerate(hidden):
                collected.setdefault(layer, []).append(
                    value.squeeze(0).mean(dim=0).detach().cpu().numpy()
                )
            del value, hidden
        matrices[label] = {layer: np.vstack(rows) for layer, rows in collected.items()}
        result["passes"][label] = {
            "minimum_samples": shortest,
            "loading_info": json.loads(json.dumps(loading, default=loading_metadata_json)),
            "model_config_sha256": hashlib.sha256(
                model.config.to_json_string().encode()
            ).hexdigest(),
            "vectors_sha256": {
                str(k): hashlib.sha256(v.tobytes()).hexdigest() for k, v in matrices[label].items()
            },
        }
        del forward, model
        torch.cuda.empty_cache()
        print(label, "complete", flush=True)
    matrices["recorded_v6"] = {}
    for layer in matrices["legacy"]:
        name = f"{args.model.split('/')[-1]}-x{args.slowdown}-l{layer:02d}"
        matrices["recorded_v6"][layer] = load_candidate(Path(source["path"]), endpoint, name)[
            indices
        ]
    result["comparisons"] = {}
    for a, b in [
        ("legacy", "probed"),
        ("probed", "probed_repeat"),
        ("legacy", "recorded_v6"),
        ("probed", "recorded_v6"),
    ]:
        rows = {}
        for layer, x in matrices[a].items():
            y = matrices[b][layer]
            delta = x.astype(np.float64) - y.astype(np.float64)
            rows[str(layer)] = {
                "exact": bool(np.array_equal(x, y)),
                "max_abs": float(np.abs(delta).max()),
                "rms": float(np.sqrt(np.mean(delta**2))),
                "relative_rms": float(
                    np.sqrt(np.mean(delta**2) / max(np.mean(x.astype(np.float64) ** 2), 1e-30))
                ),
            }
        result["comparisons"][a + "__" + b] = rows
    _emit(result, args.output, require_archive=True)


if __name__ == "__main__":
    main()
