#!/usr/bin/env python3
"""Aggregate EXP002 screen evidence and freeze the held-out candidate."""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


HERE = Path(__file__).resolve().parent
PROBES = HERE / "probe_results"
ENCODERS = HERE / "encoder_baselines"
FEATURES = HERE / "feature_manifests"


def load(path: Path) -> dict[str, Any]:
    """Load one UTF-8 JSON document."""
    return json.loads(path.read_text(encoding="utf-8"))


def sha256(path: Path) -> str:
    """Return a stable SHA-256 for an evidence file."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def metric_row(kind: str, name: str, metrics: dict[str, Any], **extra: Any) -> dict[str, Any]:
    """Project comparable screen metrics into one compact row."""
    return {
        "kind": kind,
        "name": name,
        "auprc": metrics["auprc"],
        "auroc": metrics["auroc"],
        "balanced_accuracy_at_0_5": metrics["balanced_accuracy_at_0_5"],
        "brier": metrics["brier"],
        "recall_at_5pct_false_approve": metrics["recall_at_5pct_false_approve"]["recall"],
        "recall_at_1pct_false_approve": metrics["recall_at_1pct_false_approve"]["recall"],
        **extra,
    }


def main() -> None:
    """Rank completed models and freeze the predeclared AUPRC winner."""
    argparse.ArgumentParser(description=__doc__).parse_args()
    config_path = HERE / "config.json"
    config = load(config_path)
    surface_path = HERE / "surface_baseline.json"
    surface = load(surface_path)
    rows = [
        metric_row(
            "matched_surface_baseline",
            surface["baseline"],
            surface["metrics"],
            elapsed_seconds=surface["elapsed_seconds"],
        )
    ]
    evidence: dict[str, str] = {
        str(surface_path.relative_to(HERE)): sha256(surface_path),
        "config.json": sha256(config_path),
    }

    activation_results: list[tuple[Path, dict[str, Any]]] = []
    for path in sorted(PROBES.glob("*.json")):
        result = load(path)
        activation_results.append((path, result))
        best = result["best"]
        manifest_path = FEATURES / path.name
        manifest = load(manifest_path)
        train_seconds = manifest["splits"]["train"]["elapsed_seconds"]
        validation_seconds = manifest["splits"]["validation"]["elapsed_seconds"]
        train_rows = manifest["splits"]["train"]["rows"]
        validation_rows = manifest["splits"]["validation"]["rows"]
        rows.append(
            metric_row(
                "frozen_activation_probe",
                result["model"],
                best["metrics"],
                pooling=best["pooling"],
                start_layer=best["start_layer"],
                end_layer=best["end_layer"],
                width=best["width"],
                normalized_depth=best["normalized_depth"],
                extraction_examples_per_second=(train_rows + validation_rows)
                / (train_seconds + validation_seconds),
                peak_extraction_rss_kib=manifest["peak_rss_kib"],
            )
        )
        evidence[str(path.relative_to(HERE))] = sha256(path)
        evidence[str(manifest_path.relative_to(HERE))] = sha256(manifest_path)

    for path in sorted(ENCODERS.glob("*.json")):
        result = load(path)
        rows.append(
            metric_row(
                "dedicated_in_domain_encoder",
                result["model"]["id"],
                result["metrics"],
                extraction_examples_per_second=result["rows"] / result["elapsed_seconds"],
                peak_inference_rss_kib=result["peak_rss_kib"],
                training_budget_note="Publisher-trained on the large AUTO corpus; not the 1,600-row probe budget.",
            )
        )
        evidence[str(path.relative_to(HERE))] = sha256(path)

    status_path = HERE / "run_status.json"
    statuses = load(status_path)
    evidence[status_path.name] = sha256(status_path)

    rows.sort(key=lambda row: row["auprc"], reverse=True)
    activation_rows = [row for row in rows if row["kind"] == "frozen_activation_probe"]
    activation_rows.sort(key=lambda row: row[config["selection_metric"]], reverse=True)
    if not activation_rows:
        raise RuntimeError("No completed activation probes")
    winner = activation_rows[0]
    winner_path, winner_result = next(
        (path, result)
        for path, result in activation_results
        if result["model"] == winner["name"]
    )
    best = winner_result["best"]
    feature_manifest_path = FEATURES / winner_path.name

    created_at = datetime.now(timezone.utc).isoformat()
    leaderboard = {
        "schema_version": 1,
        "experiment_id": config["experiment_id"],
        "created_at": created_at,
        "screen_only": True,
        "selection_metric": config["selection_metric"],
        "rows": rows,
        "activation_ranking": activation_rows,
        "incomplete_or_infeasible": statuses["activation_models"],
        "evidence_sha256": evidence,
        "interpretation_lock": {
            "surface_comparison": "Matched 1,600-row train and 800-row validation budget.",
            "encoder_comparison": "In-domain publisher-trained upper baselines; training budget is not matched.",
            "selection_warning": "Screen metrics select candidates and are optimistic; held-out metrics are required for claims."
        }
    }
    leaderboard_path = HERE / "screen_leaderboard.json"
    leaderboard_path.write_text(
        json.dumps(leaderboard, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )

    frozen = {
        "schema_version": 1,
        "experiment_id": config["experiment_id"],
        "frozen_at": created_at,
        "frozen_before_heldout_read": True,
        "selection_rule": "Highest validation AUPRC among completed, full-screen frozen activation probes.",
        "model": winner["name"],
        "pooling": best["pooling"],
        "start_layer": best["start_layer"],
        "end_layer": best["end_layer"],
        "width": best["width"],
        "normalized_depth": best["normalized_depth"],
        "validation_metrics": best["metrics"],
        "head": config["head"],
        "renderer": config["renderer"],
        "training_rows": config["screen_rows"]["train"],
        "test_source": "ProCreations/approve-or-deny@test.parquet",
        "evidence": {
            "probe_result": str(winner_path.relative_to(HERE)),
            "probe_result_sha256": sha256(winner_path),
            "feature_manifest": str(feature_manifest_path.relative_to(HERE)),
            "feature_manifest_sha256": sha256(feature_manifest_path),
            "screen_leaderboard": leaderboard_path.name,
            "screen_leaderboard_sha256": sha256(leaderboard_path),
        },
        "heldout_rules": {
            "no_model_reselection": True,
            "no_pooling_reselection": True,
            "no_layer_reselection": True,
            "primary_comparator": "word_1_2_tfidf_logistic trained on the identical 1,600 screen rows",
            "primary_metric": "auprc",
        }
    }
    frozen_path = HERE / "FROZEN_SELECTION.json"
    frozen_path.write_text(json.dumps(frozen, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps({"winner": winner, "frozen_path": str(frozen_path)}, indent=2))


if __name__ == "__main__":
    main()
