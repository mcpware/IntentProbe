#!/usr/bin/env python3
"""Runtime scanner that scores text with a cached probe artifact."""

from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path
from typing import Any

import numpy as np

try:
    from .model_registry import extract_features, extract_sae_features
    from .static_checks import static_scan, static_score
except ImportError:
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    from intentprobe.scanner.model_registry import extract_features, extract_sae_features  # type: ignore
    from intentprobe.scanner.static_checks import static_scan, static_score  # type: ignore


CORE_VERSION = "activation-scanner-core-2026-06-03-static-v3"
DECISION_POLICY_NAME = "corroborated-block-v3"
STATIC_BLOCK_BUNDLE_MAX_SPAN = 900
STATIC_BLOCK_BUNDLES = (
    frozenset(("static-secret-file", "static-exfiltration", "static-hidden-action")),
)
STATIC_ACTIVATION_CORROBORATORS = {
    "static-exfiltration",
    "static-hidden-action",
    "static-host-modification",
}


def read_input_text(args: argparse.Namespace) -> str:
    if args.text and args.file:
        raise SystemExit("Use either --text or --file, not both.")
    if args.text:
        return args.text
    if args.file:
        return Path(args.file).read_text()
    raise SystemExit("Provide --text or --file.")


def read_batch_inputs(path: Path) -> list[tuple[str | None, str]]:
    rows = json.loads(path.read_text())
    if not isinstance(rows, list):
        raise SystemExit("--batch-file must contain a JSON array.")
    out: list[tuple[str | None, str]] = []
    for idx, row in enumerate(rows):
        if isinstance(row, str):
            out.append((None, row))
            continue
        if isinstance(row, dict) and isinstance(row.get("text"), str):
            raw_id = row.get("id")
            out.append((str(raw_id) if raw_id is not None else None, row["text"]))
            continue
        raise SystemExit(f"Batch row {idx} must be a string or object with a text field.")
    if not out:
        raise SystemExit("--batch-file must contain at least one item.")
    return out


def artifact_dir_from_path(path: Path) -> Path:
    if path.is_file():
        return path.parent
    return path


def load_artifact(path: Path) -> tuple[Path, dict[str, Any], dict[str, np.ndarray]]:
    artifact_dir = artifact_dir_from_path(path)
    metadata_path = artifact_dir / "metadata.json"
    weights_path = artifact_dir / "probe_weights.npz"
    if not metadata_path.exists():
        raise SystemExit(f"Missing artifact metadata: {metadata_path}")
    if not weights_path.exists():
        raise SystemExit(f"Missing artifact weights: {weights_path}")

    metadata = json.loads(metadata_path.read_text())
    with np.load(weights_path) as data:
        weights = {
            "coef": np.asarray(data["coef"], dtype=np.float64),
            "intercept": np.asarray(data["intercept"], dtype=np.float64),
            "classes": np.asarray(data["classes"], dtype=np.int64),
        }
    return artifact_dir, metadata, weights


def runtime_value(cli_value: Any, metadata_value: Any, default: Any) -> Any:
    if cli_value is not None:
        return cli_value
    if metadata_value is not None:
        return metadata_value
    return default


def extract_runtime_matrix_for_texts(
    *,
    texts: list[str],
    metadata: dict[str, Any],
    args: argparse.Namespace,
) -> tuple[np.ndarray, dict[str, Any]]:
    if not texts:
        raise SystemExit("Need at least one text to scan.")

    sensor_model = metadata["sensor_model"]
    feature_details = metadata.get("feature_details", {})
    batch_size = runtime_value(args.batch_size, feature_details.get("batch_size"), 16)
    max_length = runtime_value(args.max_length, feature_details.get("max_length"), 256)
    device = runtime_value(args.device, feature_details.get("device"), "cpu")
    dtype = runtime_value(args.dtype, feature_details.get("dtype"), "float32")
    extractor = metadata.get("extractor") or ("sae" if metadata.get("feature_kind") == "sae" else "raw")

    if extractor == "sae":
        sae = metadata.get("sae") or {}
        sae_name = sae.get("name")
        if not sae_name:
            raise SystemExit("SAE artifact metadata is missing sae.name")
        bundle = extract_sae_features(
            sensor_model,
            sae_name,
            texts,
            batch_size=int(batch_size),
            max_length=int(max_length),
            device=str(device),
            dtype=str(dtype),
            local_files_only=args.local_files_only,
        )
        layer = bundle.layers[0]
        matrix = bundle.features_by_layer[layer]
    else:
        selected_layers = tuple(int(layer) for layer in metadata["layer_policy"]["selected_layers"])
        bundle = extract_features(
            sensor_model,
            texts,
            layers=selected_layers,
            layer_sweep=False,
            batch_size=int(batch_size),
            max_length=int(max_length),
            device=str(device),
            dtype=str(dtype),
            local_files_only=args.local_files_only,
        )
        matrix = np.concatenate([bundle.features_by_layer[layer] for layer in selected_layers], axis=1)

    expected_dim = int(metadata["feature_dim"])
    if matrix.shape[1] != expected_dim:
        raise SystemExit(f"Feature dimension mismatch: artifact expects {expected_dim}, runtime got {matrix.shape[1]}")

    details = {
        "feature_kind": bundle.feature_kind,
        "extractor": extractor,
        "feature_elapsed_seconds": bundle.elapsed_seconds,
        "feature_details": bundle.details,
        "layer_policy": metadata.get("layer_policy"),
    }
    return matrix, details


def extract_runtime_matrix(
    *,
    text: str,
    metadata: dict[str, Any],
    args: argparse.Namespace,
) -> tuple[np.ndarray, dict[str, Any]]:
    return extract_runtime_matrix_for_texts(texts=[text], metadata=metadata, args=args)


def sigmoid(value: float) -> float:
    if value >= 0:
        z = math.exp(-value)
        return 1.0 / (1.0 + z)
    z = math.exp(value)
    return z / (1.0 + z)


def softmax(row: np.ndarray) -> np.ndarray:
    shifted = row - np.max(row)
    exp = np.exp(shifted)
    return exp / exp.sum()


def positive_probabilities(matrix: np.ndarray, weights: dict[str, np.ndarray]) -> np.ndarray:
    coef = weights["coef"]
    intercept = weights["intercept"]
    classes = [int(value) for value in weights["classes"]]
    logits = matrix @ coef.T + intercept

    if logits.shape[1] == 1:
        raw = logits[:, 0]
        positive_class = classes[-1] if classes else 1
        probs = np.asarray([sigmoid(float(value)) for value in raw], dtype=np.float64)
        return probs if positive_class == 1 else 1.0 - probs

    if 1 not in classes:
        return np.zeros(logits.shape[0], dtype=np.float64)
    class_idx = classes.index(1)
    return np.asarray([softmax(row)[class_idx] for row in logits], dtype=np.float64)


def positive_probability(matrix: np.ndarray, weights: dict[str, np.ndarray]) -> float:
    return float(positive_probabilities(matrix, weights)[0])


def top_sae_feature_contributions(matrix: np.ndarray, weights: dict[str, np.ndarray], limit: int) -> list[dict[str, Any]]:
    if limit <= 0 or weights["coef"].shape[0] != 1:
        return []
    values = matrix[0]
    coef = weights["coef"][0]
    contributions = values * coef
    positive = np.flatnonzero(contributions > 0)
    if len(positive) == 0:
        return []
    ranked = positive[np.argsort(contributions[positive])[::-1]][:limit]
    return [
        {
            "feature_index": int(idx),
            "activation_value": float(values[idx]),
            "probe_weight": float(coef[idx]),
            "contribution": float(contributions[idx]),
        }
        for idx in ranked
    ]


def build_reasons(
    *,
    activation_score: float,
    static_findings: list[dict[str, Any]],
    feature_kind: str,
    artifact_id: str,
) -> list[str]:
    reasons = [f"cached {feature_kind} probe {artifact_id} score={activation_score:.3f}"]
    for finding in static_findings:
        reasons.append(f"{finding['severity']} static finding {finding['id']}: {finding['reason']}")
    if not static_findings and activation_score < 0.5:
        reasons.append("No strong cached-probe or static keyword signal.")
    return reasons


def static_finding_ids(static_findings: list[dict[str, Any]]) -> set[str]:
    return {str(finding.get("id")) for finding in static_findings if finding.get("id")}


def static_block_bundle(static_findings: list[dict[str, Any]]) -> frozenset[str] | None:
    ids = static_finding_ids(static_findings)
    for bundle in STATIC_BLOCK_BUNDLES:
        if not bundle.issubset(ids):
            continue
        bundle_findings = [row for row in static_findings if str(row.get("id")) in bundle]
        starts = [int(row.get("start", 0)) for row in bundle_findings]
        ends = [int(row.get("end", 0)) for row in bundle_findings]
        if starts and ends and max(ends) - min(starts) <= STATIC_BLOCK_BUNDLE_MAX_SPAN:
            return bundle
    return None


def activation_static_corroborated(static_findings: list[dict[str, Any]]) -> bool:
    ids = static_finding_ids(static_findings)
    if ids & STATIC_ACTIVATION_CORROBORATORS:
        return True
    return False


def choose_runtime_decision(
    *,
    activation_score: float,
    static_risk_score: float,
    risk_score: float,
    static_findings: list[dict[str, Any]],
    warn_threshold: float,
    block_threshold: float,
) -> tuple[str, list[str]]:
    """Choose the product decision without treating one high score as a hard block.

    Calibration showed that activation-only and single-keyword high scores can
    be useful review signals while still being too noisy for automatic block.
    The block tier therefore needs corroboration.
    """

    reasons: list[str] = []
    bundle = static_block_bundle(static_findings)
    if bundle:
        reasons.append(f"static block bundle: {','.join(sorted(bundle))}")
        return "block", reasons

    if (
        activation_score >= block_threshold
        and static_risk_score >= warn_threshold
        and activation_static_corroborated(static_findings)
    ):
        reasons.append(
            "activation block score corroborated by static finding "
            f"(activation={activation_score:.3f}, static={static_risk_score:.3f})"
        )
        return "block", reasons

    if risk_score >= warn_threshold:
        if risk_score >= block_threshold:
            reasons.append("high risk score without block corroboration; downgraded to warn")
        else:
            reasons.append("risk score reached warn threshold")
        return "warn", reasons

    reasons.append("risk score below warn threshold")
    return "allow", reasons


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact", type=Path, required=True, help="Artifact directory or metadata.json path.")
    parser.add_argument("--text", help="Tool, MCP, skill, plugin, or hook description to scan.")
    parser.add_argument("--file", type=Path, help="Read text to scan from a local file.")
    parser.add_argument("--batch-file", type=Path, help="Read a JSON array of strings or {id,text} objects.")
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--max-length", type=int, default=None)
    parser.add_argument("--device", default=None)
    parser.add_argument("--dtype", choices=("auto", "float32", "bfloat16"), default=None)
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument("--warn-threshold", type=float, default=None)
    parser.add_argument("--block-threshold", type=float, default=None)
    parser.add_argument("--top-sae-features", type=int, default=8)
    parser.add_argument("--pretty", action="store_true")
    return parser


def runtime_args(
    *,
    batch_size: int | None = None,
    max_length: int | None = None,
    device: str | None = None,
    dtype: str | None = None,
    local_files_only: bool = False,
    warn_threshold: float | None = None,
    block_threshold: float | None = None,
    top_sae_features: int = 8,
) -> argparse.Namespace:
    return argparse.Namespace(
        batch_size=batch_size,
        max_length=max_length,
        device=device,
        dtype=dtype,
        local_files_only=local_files_only,
        warn_threshold=warn_threshold,
        block_threshold=block_threshold,
        top_sae_features=top_sae_features,
    )


def build_risk_output(
    *,
    text: str,
    artifact_dir: Path,
    metadata: dict[str, Any],
    weights: dict[str, np.ndarray],
    matrix_row: np.ndarray,
    activation_score: float,
    activation_details: dict[str, Any],
    warn_threshold: float,
    block_threshold: float,
    top_sae_features: int,
    elapsed_seconds: float,
) -> dict[str, Any]:
    findings = static_scan(text)
    static_risk_score = static_score(findings)
    risk_score = max(activation_score, static_risk_score)
    decision, decision_policy_reasons = choose_runtime_decision(
        activation_score=activation_score,
        static_risk_score=static_risk_score,
        risk_score=risk_score,
        static_findings=findings,
        warn_threshold=warn_threshold,
        block_threshold=block_threshold,
    )

    sae_features = []
    if metadata.get("feature_kind") == "sae":
        sae_features = top_sae_feature_contributions(matrix_row, weights, top_sae_features)

    artifact_id = metadata.get("artifact_id", artifact_dir.name)
    return {
        "decision": decision,
        "risk_score": risk_score,
        "risk_reasons": build_reasons(
            activation_score=activation_score,
            static_findings=findings,
            feature_kind=str(metadata.get("feature_kind")),
            artifact_id=str(artifact_id),
        ),
        "activation_score": activation_score,
        "text_baseline_score": None,
        "static_score": static_risk_score,
        "sae_features": sae_features,
        "evidence_spans": findings,
        "scanner_version": CORE_VERSION,
        "artifact_id": artifact_id,
        "artifact_path": str(artifact_dir),
        "model_id": metadata.get("model_id"),
        "sensor_model": metadata.get("sensor_model"),
        "sae": metadata.get("sae", {}).get("name") if metadata.get("sae") else None,
        "mode": "cached_probe_runtime",
        "training_data": metadata.get("dataset"),
        "activation_details": activation_details,
        "thresholds": {
            "warn": warn_threshold,
            "block": block_threshold,
        },
        "decision_policy": {
            "name": DECISION_POLICY_NAME,
            "block_requires": (
                "nearby high-confidence static bundle or activation block score "
                "corroborated by action-oriented exfiltration, hidden-action, "
                "or host-modification finding"
            ),
            "static_block_bundles": [sorted(bundle) for bundle in STATIC_BLOCK_BUNDLES],
            "static_block_bundle_max_span": STATIC_BLOCK_BUNDLE_MAX_SPAN,
            "static_activation_corroborators": sorted(STATIC_ACTIVATION_CORROBORATORS),
            "policy_reasons": decision_policy_reasons,
        },
        "elapsed_seconds": elapsed_seconds,
    }


def scan_texts(
    target_texts: list[str],
    artifact: Path,
    *,
    batch_size: int | None = None,
    max_length: int | None = None,
    device: str | None = None,
    dtype: str | None = None,
    local_files_only: bool = False,
    warn_threshold: float | None = None,
    block_threshold: float | None = None,
    top_sae_features: int = 8,
) -> list[dict[str, Any]]:
    started = time.perf_counter()
    args = runtime_args(
        batch_size=batch_size,
        max_length=max_length,
        device=device,
        dtype=dtype,
        local_files_only=local_files_only,
        warn_threshold=warn_threshold,
        block_threshold=block_threshold,
        top_sae_features=top_sae_features,
    )

    artifact_dir, metadata, weights = load_artifact(artifact)
    matrix, activation_details = extract_runtime_matrix_for_texts(texts=target_texts, metadata=metadata, args=args)
    activation_scores = positive_probabilities(matrix, weights)

    thresholds = metadata.get("thresholds", {})
    warn = float(warn_threshold if warn_threshold is not None else thresholds.get("warn", 0.30))
    block = float(block_threshold if block_threshold is not None else thresholds.get("block", 0.85))
    elapsed_seconds = time.perf_counter() - started

    return [
        build_risk_output(
            text=text,
            artifact_dir=artifact_dir,
            metadata=metadata,
            weights=weights,
            matrix_row=matrix[idx : idx + 1],
            activation_score=float(activation_scores[idx]),
            activation_details=activation_details,
            warn_threshold=warn,
            block_threshold=block,
            top_sae_features=top_sae_features,
            elapsed_seconds=elapsed_seconds,
        )
        for idx, text in enumerate(target_texts)
    ]


def scan_text(
    target_text: str,
    artifact: Path,
    *,
    batch_size: int | None = None,
    max_length: int | None = None,
    device: str | None = None,
    dtype: str | None = None,
    local_files_only: bool = False,
    warn_threshold: float | None = None,
    block_threshold: float | None = None,
    top_sae_features: int = 8,
) -> dict[str, Any]:
    return scan_texts(
        [target_text],
        artifact,
        batch_size=batch_size,
        max_length=max_length,
        device=device,
        dtype=dtype,
        local_files_only=local_files_only,
        warn_threshold=warn_threshold,
        block_threshold=block_threshold,
        top_sae_features=top_sae_features,
    )[0]


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.batch_file:
        if args.text or args.file:
            raise SystemExit("Use --batch-file by itself, not with --text or --file.")
        rows = read_batch_inputs(args.batch_file)
        outputs = scan_texts(
            [text for _, text in rows],
            args.artifact,
            batch_size=args.batch_size,
            max_length=args.max_length,
            device=args.device,
            dtype=args.dtype,
            local_files_only=args.local_files_only,
            warn_threshold=args.warn_threshold,
            block_threshold=args.block_threshold,
            top_sae_features=args.top_sae_features,
        )
        for idx, (input_id, _) in enumerate(rows):
            if input_id is not None:
                outputs[idx]["input_id"] = input_id
        output = {
            "mode": "cached_probe_batch_runtime",
            "scanner_version": CORE_VERSION,
            "artifact": str(args.artifact),
            "count": len(outputs),
            "results": outputs,
        }
    else:
        target_text = read_input_text(args)
        output = scan_text(
            target_text,
            args.artifact,
            batch_size=args.batch_size,
            max_length=args.max_length,
            device=args.device,
            dtype=args.dtype,
            local_files_only=args.local_files_only,
            warn_threshold=args.warn_threshold,
            block_threshold=args.block_threshold,
            top_sae_features=args.top_sae_features,
        )
    print(json.dumps(output, indent=2 if args.pretty else None, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
