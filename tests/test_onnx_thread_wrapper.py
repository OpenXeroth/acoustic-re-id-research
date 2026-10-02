"""Execution changes must remain scoped and cannot reuse unlabelled checkpoints."""

import importlib
import runpy
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest


@pytest.fixture
def wrapper(monkeypatch: pytest.MonkeyPatch) -> Any:
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1] / "scripts"))
    return importlib.import_module("run_v6_onnx_threads")


def test_checkpoint_namespace_refuses_old_or_different_execution(
    wrapper: Any, tmp_path: Path
) -> None:
    old = tmp_path / "old"
    old.mkdir()
    (old / "previous-model.json").write_text("{}")
    with pytest.raises(ValueError, match="unlabelled"):
        wrapper.checkpoint_context(old, {"threads": 3})
    fresh = tmp_path / "fresh"
    path = wrapper.checkpoint_context(fresh, {"threads": 3})
    original = path.read_bytes()
    assert wrapper.checkpoint_context(fresh, {"threads": 3}) == path
    assert path.read_bytes() == original
    with pytest.raises(ValueError, match="another execution"):
        wrapper.checkpoint_context(fresh, {"threads": 2})
    assert path.read_bytes() == original


def fake_runtime() -> Any:
    class Options:
        def __init__(self) -> None:
            self.intra_op_num_threads = 0
            self.inter_op_num_threads = 0
            self.graph_optimization_level = "original graph policy"
            self.config: dict[str, str] = {"unrelated": "preserved"}

        def add_session_config_entry(self, key: str, value: str) -> None:
            self.config[key] = value

        def get_session_config_entry(self, key: str) -> str:
            return self.config[key]

    class Session:
        def __init__(self, path: Any, sess_options: Any = None, **kwargs: Any) -> None:
            self.path = path
            self.options = sess_options or Options()
            self.kwargs = kwargs

        def get_session_options(self) -> Any:
            return self.options

        def get_providers(self) -> Any:
            return self.kwargs.get("providers", ["CPUExecutionProvider"])

    return SimpleNamespace(InferenceSession=Session, SessionOptions=Options)


def test_only_birdnet_sessions_change_and_other_options_survive(wrapper: Any) -> None:
    ort = fake_runtime()
    original = ort.InferenceSession

    class Loaded:
        def _load_bacpipe(self, target: str) -> None:
            self.supplied = ort.SessionOptions()
            self.session = ort.InferenceSession(
                target, sess_options=self.supplied, providers=["CPUExecutionProvider"], extra="kept"
            )

    records: list[dict[str, Any]] = []
    wrapper.install(ort, Loaded, records)
    bird = Loaded()
    bird._load_bacpipe("birdnet_v3")
    assert bird.session.options is bird.supplied
    assert bird.session.options.graph_optimization_level == "original graph policy"
    assert bird.session.options.config["unrelated"] == "preserved"
    assert bird.session.kwargs["extra"] == "kept"
    assert records[0]["intra_op_threads"] == 3
    assert records[0]["inter_op_threads"] == 1
    assert records[0]["intra_op_spinning"] == records[0]["inter_op_spinning"] == "0"
    assert ort.InferenceSession is original
    perch = Loaded()
    perch._load_bacpipe("perch_v2")
    assert type(perch.session) is original
    assert perch.session.options.intra_op_num_threads == 0
    assert len(records) == 1


@pytest.mark.parametrize("failure", ["load", "provider", "unobserved"])
def test_original_constructor_restored_when_loading_fails(wrapper: Any, failure: str) -> None:
    ort = fake_runtime()
    original = ort.InferenceSession

    class Loaded:
        def _load_bacpipe(self, target: str) -> None:
            if failure == "load":
                raise RuntimeError("load failed")
            if failure == "unobserved":
                return
            ort.InferenceSession(
                target, providers=["CUDAExecutionProvider", "CPUExecutionProvider"]
            )

    records: list[dict[str, Any]] = []
    wrapper.install(ort, Loaded, records)
    with pytest.raises((RuntimeError, ValueError)):
        Loaded()._load_bacpipe("birdnet_v3")
    assert ort.InferenceSession is original
    assert records == []


def test_spawn_import_is_inert() -> None:
    path = Path(__file__).resolve().parents[1] / "scripts/run_v6_onnx_threads.py"
    # No ONNX dependency, lease, command arguments or output directory is needed.
    runpy.run_path(str(path), run_name="__mp_main__")


def test_explicit_perch_opt_in_keeps_correct_model_labels_and_graph_options(wrapper: Any) -> None:
    ort = fake_runtime()
    original = ort.InferenceSession

    class Loaded:
        def _load_bacpipe(self, target: str) -> None:
            self.options = ort.SessionOptions()
            self.session = ort.InferenceSession(target, sess_options=self.options)

    records: list[dict[str, Any]] = []
    wrapper.install(ort, Loaded, records, bound_perch=True)
    perch = Loaded()
    perch._load_bacpipe("perch_v2")
    bird = Loaded()
    bird._load_bacpipe("birdnet_v3")
    assert [row["model"] for row in records] == ["perch-v2", "birdnet-v3-preview"]
    for model in (perch, bird):
        assert model.session.options is model.options
        assert model.options.graph_optimization_level == "original graph policy"
        assert model.options.intra_op_num_threads == 3
    # A retained constructor still carries its own model label after another load.
    type(perch.session)("another perch session")
    assert records[-1]["model"] == "perch-v2"
    assert ort.InferenceSession is original


def test_perch_opt_in_rejects_unreviewed_diagnostic_before_loading_runtime(
    wrapper: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(
        wrapper,
        "digest",
        lambda path: wrapper.DIAGNOSTIC_SHA256 if path.name == "birdnet.json" else "wrong bytes",
    )
    monkeypatch.setattr(
        wrapper.sys,
        "argv",
        [
            "wrapper",
            "--diagnostic",
            "birdnet.json",
            "--perch-diagnostic",
            "perch.json",
            "--",
            "run-a6",
            "--device",
            "cpu",
            "--model",
            "perch-v2",
        ],
    )
    with pytest.raises(ValueError, match="Perch v2 diagnostic has different bytes"):
        wrapper.main()
