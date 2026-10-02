"""Regression for SpeechBrain precision side effects across checkpoint resumes."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from xinyenyana.a2 import ClipRecord, Endpoint
from xinyenyana.precision import speech_matmul


def backend(monkeypatch):
    matmul = SimpleNamespace(allow_tf32=False)
    cudnn = SimpleNamespace(allow_tf32=False)
    monkeypatch.setitem(
        sys.modules,
        "torch",
        SimpleNamespace(backends=SimpleNamespace(cuda=SimpleNamespace(matmul=matmul), cudnn=cudnn)),
    )
    return matmul, cudnn


def test_precision_restores_prior_context_after_failed_inference(monkeypatch):
    matmul, cudnn = backend(monkeypatch)
    with pytest.raises(RuntimeError, match="inference"):
        with speech_matmul("tf32") as context:
            assert matmul.allow_tf32 and cudnn.allow_tf32
            assert context["speech_matmul"] == "tf32"
            raise RuntimeError("inference failed")
    assert not matmul.allow_tf32 and not cudnn.allow_tf32
    with pytest.raises(RuntimeError, match="changed its declared CUDA arithmetic"):
        with speech_matmul("fp32"):
            matmul.allow_tf32 = True
    assert not matmul.allow_tf32 and not cudnn.allow_tf32
    with pytest.raises(ValueError, match="fp32 or tf32"):
        with speech_matmul("automatic"):
            pass


def test_checkpoint_resume_and_controls_use_declared_precision(monkeypatch, tmp_path):
    from xinyenyana import a2, a6

    matmul, cudnn = backend(monkeypatch)
    speaker = "speechbrain/spkrec-ecapa-voxceleb"
    transformer = "facebook/mms-300m"
    seen = []

    def produce(model, records, **options):
        if model == speaker:
            matmul.allow_tf32 = True  # The installed SpeechBrain import side effect.
        seen.append((model, matmul.allow_tf32))
        name = options.get("fixed") or "candidate"
        yield name, {}, np.full((len(records), 2), float(matmul.allow_tf32), dtype=np.float32)

    monkeypatch.setattr(a6, "_unconfigured_producers", produce)
    monkeypatch.setattr(a2, "clip_diagnostics", lambda records: {})
    monkeypatch.setattr(
        a6,
        "evaluate_representation",
        lambda **kw: {
            "classification": {"accuracy": 0.5},
            "identity_block_bootstrap_accuracy_95": [0, 1],
            "permutation_control": {"p_value_plus_one": 0.5},
            "verification": {"roc_auc": 0.5},
            "context_probes": {},
            "standardisation": "test",
            "query_correct": "10",
        },
    )
    endpoint = Endpoint(
        name="precision",
        records=tuple(
            ClipRecord(
                filename=f"{split}-{identity}",
                path=Path("/unused"),
                identity=identity,
                split=split,
                context={"condition": "foreground"},
            )
            for split in ["enrollment", "query"]
            for identity in ["a", "b"]
        ),
        categorical_targets=(),
        manifest_sha256="precision-test",
        source_document="test",
    )
    # Keep the model list constant while simulating interruption before its
    # transformer checkpoint: otherwise a changed request could explain a miss.
    original_produce = produce

    def interrupt_after_speaker(model, records, **options):
        if model == transformer:
            raise RuntimeError("worker interrupted")
        yield from original_produce(model, records, **options)

    monkeypatch.setattr(a6, "_unconfigured_producers", interrupt_after_speaker)
    kwargs = dict(
        endpoint=endpoint,
        models=[speaker, transformer],
        store=tmp_path / "store",
        checkpoint_dir=tmp_path / "checkpoints",
        with_controls=False,
    )
    with pytest.raises(RuntimeError, match="worker interrupted"):
        a6.run_a6_endpoint(**kwargs)
    matmul.allow_tf32 = False  # A fresh process did not import the skipped speaker model.
    seen.clear()
    monkeypatch.setattr(a6, "_unconfigured_producers", original_produce)
    resumed = a6.run_a6_endpoint(**kwargs)
    assert seen == [(transformer, True)]
    assert resumed["chosen"][transformer]["numeric_precision"]["speech_matmul"] == "tf32"
    assert not matmul.allow_tf32 and not cudnn.allow_tf32

    # Changing precision must invalidate the transformer checkpoint. The
    # speaker's global side effect must not override the requested FP32 mode.
    seen.clear()
    changed = a6.run_a6_endpoint(**kwargs, speech_precision={transformer: "fp32"})
    assert (transformer, False) in seen
    assert (
        changed["candidate_vectors"][transformer]["sha256"]
        != resumed["candidate_vectors"][transformer]["sha256"]
    )
    arms = a6.arm_vectors(
        transformer,
        "candidate",
        {"own": list(endpoint.records)},
        device="cuda",
        matmul_precision="fp32",
    )
    assert not np.any(arms["own"])
    assert matmul.allow_tf32  # The enclosing speaker context is restored.


def test_fixed_cli_inherits_precision_and_rejects_a_conflict(monkeypatch, tmp_path):
    from typer.testing import CliRunner

    from xinyenyana import a6, cli_v6, frozen_probe
    from xinyenyana.cli import app

    model = "facebook/mms-300m"
    choice = {
        "representation": "mms-300m-x3-l24",
        "accuracy": 0.5,
        "query_correct": "10",
        "numeric_precision": {"speech_matmul": "fp32"},
    }
    old = {"manifest_sha256": "same", "chosen": {model: choice}}
    source = tmp_path / "source.json"
    source.write_text(json.dumps(old))
    monkeypatch.setattr(cli_v6, "_require_lease_and_archive", lambda: None)
    monkeypatch.setattr(cli_v6, "_endpoint", lambda *a: SimpleNamespace(manifest_sha256="same"))
    monkeypatch.setattr(frozen_probe, "endpoint_audio_provenance", lambda *a: {})
    calls = []

    def run(**kwargs):
        calls.append(kwargs)
        return {"chosen": {model: choice}}

    monkeypatch.setattr(a6, "run_a6_endpoint", run)
    monkeypatch.setattr(cli_v6, "_emit", lambda *a, **kw: None)
    args = [
        "run-a6",
        "--endpoint",
        "great-tit",
        "--sample-root",
        str(tmp_path),
        "--model",
        model,
        "--store",
        str(tmp_path),
        "--output",
        str(tmp_path / "out.json"),
        "--fixed-from",
        str(source),
    ]
    runner = CliRunner()
    result = runner.invoke(app, args)
    assert result.exit_code == 0, result.output
    assert calls[0]["speech_precision"] == {model: "fp32"}
    conflict = runner.invoke(app, [*args, "--speech-precision", model + "=tf32"])
    assert conflict.exit_code != 0 and "must retain recorded speech precision" in conflict.output
    assert len(calls) == 1
