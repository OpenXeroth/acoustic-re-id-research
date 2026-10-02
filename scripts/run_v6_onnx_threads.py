"""Run A6 with bounded ONNX CPU sessions and explicit execution provenance.

Use a fresh checkpoint directory for this wrapper. Bounding Perch v2 additionally
requires its reviewed diagnostic, which measured small embedding differences.
The eight-clip diagnostics are evidence about those samples only.
The source package selected by PYTHONPATH remains unchanged.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any

DIAGNOSTIC_SHA256 = "0da6985992236fd4fff9c308805de0d25166b0c99d05cd4f734da69a270d4019"
PERCH_DIAGNOSTIC_SHA256 = "087b41a63070197fbe7259159a338e62ae361e54bac044312a1663f908e567c6"
THREADS = 3


def digest(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def option(arguments: list[str], name: str) -> str:
    if arguments.count(name) != 1:
        raise ValueError(f"exactly one {name} argument is required")
    index = arguments.index(name)
    if index + 1 == len(arguments):
        raise ValueError(f"{name} requires a value")
    return arguments[index + 1]


def checkpoint_context(directory: Path, context: dict[str, Any]) -> Path:
    """Never reuse unlabelled checkpoints or mix execution contexts."""
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / ".onnx-thread-context.json"
    if path.exists():
        if json.loads(path.read_text()) != context:
            raise ValueError("checkpoint directory belongs to another execution context")
    else:
        if any(directory.iterdir()):
            raise ValueError(
                "use a fresh checkpoint directory; existing checkpoints are unlabelled"
            )
        path.write_text(json.dumps(context, indent=2) + "\n")
    return path


def install(
    ort: Any, loaded: Any, sessions: list[dict[str, Any]], *, bound_perch: bool = False
) -> None:
    """Limit sessions only while an explicitly selected adapter is constructed."""
    original_session = ort.InferenceSession
    original_load = loaded._load_bacpipe
    targets = {"birdnet_v3": "birdnet-v3-preview"}
    if bound_perch:
        targets["perch_v2"] = "perch-v2"

    class LimitedSession(original_session):
        model_label: str

        def __init__(self, path_or_bytes: Any, sess_options: Any = None, **kwargs: Any) -> None:
            options = ort.SessionOptions() if sess_options is None else sess_options
            options.intra_op_num_threads = THREADS
            options.inter_op_num_threads = 1
            options.add_session_config_entry("session.intra_op.allow_spinning", "0")
            options.add_session_config_entry("session.inter_op.allow_spinning", "0")
            super().__init__(path_or_bytes, sess_options=options, **kwargs)
            actual = self.get_session_options()
            if self.get_providers() != ["CPUExecutionProvider"]:
                raise ValueError("the bounded session unexpectedly selected a non-CPU provider")
            if (actual.intra_op_num_threads, actual.inter_op_num_threads) != (THREADS, 1):
                raise ValueError("the runtime did not retain the requested thread settings")
            for key in ("session.intra_op.allow_spinning", "session.inter_op.allow_spinning"):
                if actual.get_session_config_entry(key) != "0":
                    raise ValueError("the runtime did not disable session spinning")
            sessions.append(
                {
                    "model": self.model_label,
                    "intra_op_threads": actual.intra_op_num_threads,
                    "inter_op_threads": actual.inter_op_num_threads,
                    "intra_op_spinning": actual.get_session_config_entry(
                        "session.intra_op.allow_spinning"
                    ),
                    "inter_op_spinning": actual.get_session_config_entry(
                        "session.inter_op.allow_spinning"
                    ),
                    "providers": self.get_providers(),
                }
            )

    def load(self: Any, target: str) -> None:
        if target not in targets:
            original_load(self, target)
            return
        previous = ort.InferenceSession

        class ModelSession(LimitedSession):
            model_label = targets[target]

        ort.InferenceSession = ModelSession
        try:
            before = len(sessions)
            original_load(self, target)
            if len(sessions) == before:
                raise ValueError(f"{target} did not create an observed bounded session")
        finally:
            ort.InferenceSession = previous

    loaded._load_bacpipe = load


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--diagnostic", type=Path, required=True)
    parser.add_argument("--perch-diagnostic", type=Path)
    parser.add_argument("command", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    command = args.command[1:] if args.command[:1] == ["--"] else args.command
    if not command or command[0] != "run-a6" or option(command, "--device") != "cpu":
        raise ValueError("this wrapper requires run-a6 with explicit --device cpu")
    models = [command[i + 1] for i, value in enumerate(command[:-1]) if value == "--model"]
    if not models or not set(models) <= {"perch-v2", "birdnet-v3-preview"}:
        raise ValueError("only the two ONNX models are supported")
    if digest(args.diagnostic) != DIAGNOSTIC_SHA256:
        raise ValueError("the reviewed CPU diagnostic has different bytes")
    if args.perch_diagnostic and digest(args.perch_diagnostic) != PERCH_DIAGNOSTIC_SHA256:
        raise ValueError("the reviewed Perch v2 diagnostic has different bytes")
    bound_perch = args.perch_diagnostic is not None

    import onnxruntime as ort

    from xinyenyana import bioacoustic, cli, cli_v6
    from xinyenyana.archive import source_digest

    cli_v6._require_lease_and_archive()
    context = {
        "wrapper_sha256": digest(Path(__file__)),
        "source_sha256": source_digest(Path(bioacoustic.__file__).parent),
        "onnxruntime_version": ort.__version__,
        "diagnostic_sha256": DIAGNOSTIC_SHA256,
        "perch_diagnostic_sha256": PERCH_DIAGNOSTIC_SHA256 if bound_perch else None,
        "birdnet_v3": {"intra_op_threads": THREADS, "inter_op_threads": 1, "spinning": False},
        "perch_v2": {"intra_op_threads": THREADS, "inter_op_threads": 1, "spinning": False}
        if bound_perch
        else "unchanged framework session settings",
        "sample_embedding_max_absolute_difference": {
            "birdnet-v3-preview": 0.0,
            **({"perch-v2": 2.086162567138672e-7} if bound_perch else {}),
        },
    }
    context_path = checkpoint_context(Path(option(command, "--checkpoint-dir")), context)
    sessions: list[dict[str, Any]] = []
    install(ort, bioacoustic.Loaded, sessions, bound_perch=bound_perch)
    original_emit = cli_v6._emit

    def emit(payload: Any, output: Path, *, require_archive: bool = True) -> None:
        original_emit(
            {
                **payload,
                "onnx_cpu_execution": {
                    **context,
                    "checkpoint_context_sha256": digest(context_path),
                    "sessions_created_in_this_process": sessions,
                    "checkpoint_reuse": "only checkpoints in the identical wrapper context",
                    "limitation": (
                        "Eight sampled clips do not establish full-corpus equivalence. "
                        + (
                            "The Perch v2 diagnostic measured small floating-point differences."
                            if bound_perch
                            else ""
                        )
                    ),
                },
            },
            output,
            require_archive=require_archive,
        )

    cli_v6._emit = emit
    sys.argv = [sys.argv[0], *command]
    cli.app()


if __name__ == "__main__":
    main()
