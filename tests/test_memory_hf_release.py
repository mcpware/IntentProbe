from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
PACKAGE = ROOT / "hf" / "intentprobe-qwen2.5-0.5b-memory-write"
SOURCE = (
    ROOT
    / "research"
    / "memory-write-gate"
    / "experiments"
    / "EXP002-qwen-activation-screen"
)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_score_module():
    spec = importlib.util.spec_from_file_location("memory_write_score", PACKAGE / "score.py")
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_release_artifact_matches_frozen_selection() -> None:
    metadata = json.loads((PACKAGE / "metadata.json").read_text(encoding="utf-8"))
    assert sha256(PACKAGE / "probe.npz") == sha256(SOURCE / "selected_probe.npz")
    assert sha256(PACKAGE / "probe.npz") == metadata["artifact_sha256"]
    assert (PACKAGE / "probe.npz").stat().st_size == metadata["artifact_bytes"]


def test_release_metadata_matches_selection_receipt() -> None:
    result = json.loads((SOURCE / "selection_result.json").read_text(encoding="utf-8"))
    summary = json.loads((PACKAGE / "benchmark_summary.json").read_text(encoding="utf-8"))
    winner = result["candidate_search"]["global_winner"]
    observed = result["development_evaluation"]["selected_activation"]["threshold_0_5"]
    assert winner["start_layer"] == winner["end_layer"] == 11
    assert winner["pooling"] == "masked_mean"
    assert summary["activation_probe"]["auprc"] == observed["auprc"]
    assert summary["activation_probe"]["auroc"] == observed["auroc"]


def test_numpy_scorer_and_renderer_contract() -> None:
    module = load_score_module()
    probe = module.load_probe(PACKAGE / "probe.npz")
    assert module.render_text("A") == "USER_UTTERANCE\nA\n\nMEMORY_WRITE_DECISION"
    pooled = np.broadcast_to(probe["train_means"], (2, 896)).copy()
    scores = module.score_pooled_activations(pooled, probe)
    expected = 1.0 / (1.0 + np.exp(-float(probe["intercept"][0])))
    np.testing.assert_allclose(scores, [expected, expected])
