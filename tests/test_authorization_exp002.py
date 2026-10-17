"""Integrity tests for the authorization-send EXP002 evidence bundle."""

from __future__ import annotations

import hashlib
import importlib.util
import json
import re
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
PROJECT = ROOT / "research" / "authorization-send-detector"
EXP = PROJECT / "experiments" / "EXP002-activation-screen"
ARTIFACT_DIR = PROJECT / "artifacts" / "minicpm5-1b-authorization-exp002"
HF_DIR = ROOT / "hf" / "intentprobe-minicpm5-1b-authorization"


def load(path: Path) -> dict:
    """Load one JSON fixture."""
    return json.loads(path.read_text(encoding="utf-8"))


def sha256(path: Path) -> str:
    """Hash one evidence artifact."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def test_model_registry_is_fully_pinned() -> None:
    """All nine evaluated snapshots must have exact immutable revisions."""
    models = load(PROJECT / "MODEL_REGISTRY.json")["models"]
    assert len(models) == 9
    assert len({model["id"] for model in models}) == 9
    assert all(re.fullmatch(r"[0-9a-f]{40}", model["revision"]) for model in models)
    assert all(model["weight_bytes"] > 0 for model in models)


def test_screen_freeze_matches_predeclared_winner() -> None:
    """The held-out candidate must equal the highest-AUPRC completed probe."""
    leaderboard = load(EXP / "screen_leaderboard.json")
    frozen = load(EXP / "FROZEN_SELECTION.json")
    probes = leaderboard["activation_ranking"]
    assert len(probes) == 6
    assert all(probe["pooling"] == "masked_mean" for probe in probes)
    assert probes[0]["name"] == frozen["model"] == "openbmb/MiniCPM5-1B-Base"
    assert probes[0]["auprc"] == max(probe["auprc"] for probe in probes)
    assert (frozen["start_layer"], frozen["end_layer"], frozen["width"]) == (13, 14, 2)
    assert frozen["frozen_before_heldout_read"] is True


def test_frozen_evidence_chain_hashes_match() -> None:
    """Canonical screen inputs must still be the files frozen before held-out use."""
    leaderboard = load(EXP / "screen_leaderboard.json")
    frozen = load(EXP / "FROZEN_SELECTION.json")
    assert sha256(EXP / "screen_leaderboard.json") == frozen["evidence"]["screen_leaderboard_sha256"]
    for relative_path, expected_hash in leaderboard["evidence_sha256"].items():
        assert sha256(EXP / relative_path) == expected_hash


def test_packaged_probe_integrity_and_shape() -> None:
    """The compact research weight must match its card and frozen layer width."""
    metadata = load(ARTIFACT_DIR / "metadata.json")
    artifact_path = ARTIFACT_DIR / metadata["artifact"]
    assert artifact_path.stat().st_size == metadata["artifact_bytes"] == 28708
    assert sha256(artifact_path) == metadata["artifact_sha256"]
    artifact = np.load(artifact_path)
    assert artifact["coefficient"].shape == (1, 1536)
    assert artifact["intercept"].shape == (1,)
    assert artifact["mean"].shape == artifact["scale"].shape == (1, 2, 1536)
    assert artifact["start_layer"].tolist() == [13]
    assert artifact["end_layer"].tolist() == [14]
    assert metadata["deployment_gate"]["passed"] is False


def test_heldout_is_nonoverlapping_and_does_not_establish_auprc_gain() -> None:
    """Lock the adverse held-out evidence as well as the positive AUROC evidence."""
    result = load(EXP / "heldout_result.json")
    overlap = result["test"]["exact_rendered_overlap"]
    assert result["one_shot_heldout"] is True
    assert result["test"]["rows"] == 3000
    assert overlap == {
        "train_test_exact_rendered": 0,
        "validation_test_exact_rendered": 0,
        "test_unique_rendered": 3000,
    }
    auprc = result["paired_comparison"]["auprc"]
    auroc = result["paired_comparison"]["auroc"]
    assert auprc["activation_minus_surface"] > 0
    assert auprc["paired_row_bootstrap_95ci"][0] < 0 < auprc["paired_row_bootstrap_95ci"][1]
    assert auprc["two_sided_bootstrap_p"] > 0.05
    assert auroc["paired_row_bootstrap_95ci"][0] > 0
    assert auroc["two_sided_bootstrap_p"] < 0.05


def test_frozen_activation_thresholds_fail_heldout_budgets() -> None:
    """Prevent accidental promotion of the research artifact into a send gate."""
    result = load(EXP / "heldout_result.json")
    activation = result["activation"]["validation_frozen_operating_points"]
    surface = result["matched_surface"]["validation_frozen_operating_points"]
    assert activation["5pct"]["false_approve_rate"] > 0.05
    assert activation["1pct"]["false_approve_rate"] > 0.01
    assert surface["5pct"]["false_approve_rate"] <= 0.05
    assert surface["1pct"]["false_approve_rate"] <= 0.01


def test_hf_release_package_is_synced_to_frozen_artifact() -> None:
    """The Hub package must not drift from the locally frozen evidence bundle."""
    metadata = load(ARTIFACT_DIR / "metadata.json")
    summary = load(HF_DIR / "benchmark_summary.json")
    assert sha256(HF_DIR / "probe.npz") == metadata["artifact_sha256"]
    assert sha256(HF_DIR / "metadata.json") == sha256(ARTIFACT_DIR / "metadata.json")
    assert sha256(HF_DIR / "frozen_selection.json") == sha256(EXP / "FROZEN_SELECTION.json")
    assert summary["artifact"]["sha256"] == metadata["artifact_sha256"]
    assert summary["calibration_gate"]["passed"] is False
    assert "pipeline_tag:" not in (HF_DIR / "README.md").read_text(encoding="utf-8")


def test_hf_release_scorer_uses_exact_renderer_and_probe_math() -> None:
    """Smoke-test the standalone Hub helper without loading the 1B backbone."""
    module_path = HF_DIR / "score.py"
    spec = importlib.util.spec_from_file_location("intentprobe_hf_score", module_path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    rendered = module.render_context("send()", "show a draft", "not approved")
    assert rendered == (
        "PROPOSED ACTION:\nsend()\n"
        "USER REQUEST:\nshow a draft\n"
        "HISTORY:\nnot approved"
    )
    call = module.normalize_call_json(
        '{"tool":"send_email","args":{"to":"sam@example.com","body":"The draft"}}'
    )
    assert call == (
        '{"tool": "send_email", "args": '
        '"{\\"to\\": \\"sam@example.com\\", \\"body\\": \\"The draft\\"}"}'
    )
    probe = module.load_probe(HF_DIR / "probe.npz")
    pooled = np.zeros((1, 2, 1536), dtype=np.float32)
    score = module.score_pooled_activations(pooled, probe)
    features = ((pooled - probe["mean"]) / probe["scale"]).mean(axis=1)
    logit = float((features @ probe["coefficient"].T + probe["intercept"])[0, 0])
    expected = 1.0 / (1.0 + np.exp(-logit))
    assert score.shape == (1,)
    assert np.isclose(score[0], expected)
