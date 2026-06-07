#!/usr/bin/env python3
"""Product-preview CLI for the cached activation scanner."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

try:
    from .core import (
        CORE_VERSION,
        DECISION_POLICY_NAME,
        load_artifact,
        read_batch_inputs,
        scan_texts,
    )
except ImportError:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    from intentprobe.scanner.core import (  # type: ignore
        CORE_VERSION,
        DECISION_POLICY_NAME,
        load_artifact,
        read_batch_inputs,
        scan_texts,
    )


PACKAGE_DIR = Path(__file__).resolve().parent
DEFAULT_ARTIFACT = (
    PACKAGE_DIR
    / "artifacts"
    / "qwen-pooled-curated-core-l13-15-v2"
)
DECISION_RANK = {
    "allow": 0,
    "warn": 1,
    "block": 2,
    "quarantine": 3,
}


def artifact_complete(path: Path) -> bool:
    artifact = path.parent if path.is_file() else path
    return (artifact / "metadata.json").exists() and (artifact / "probe_weights.npz").exists()


def add_runtime_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--artifact", type=Path, default=DEFAULT_ARTIFACT, help="Cached scanner artifact directory.")
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--max-length", type=int, default=None)
    parser.add_argument("--device", default=None)
    parser.add_argument("--dtype", choices=("auto", "float32", "bfloat16"), default=None)
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument("--warn-threshold", type=float, default=None)
    parser.add_argument("--block-threshold", type=float, default=None)
    parser.add_argument("--top-sae-features", type=int, default=8)
    parser.add_argument(
        "--fail-on",
        choices=("never", "warn", "block", "quarantine"),
        default="never",
        help="Exit with status 2 when any result reaches this decision or higher.",
    )
    parser.add_argument(
        "--format",
        choices=("json", "summary"),
        default="json",
        help="Output JSON for hooks or a short human-readable summary.",
    )
    parser.add_argument("--pretty", action="store_true", help="Pretty-print JSON output.")


def read_scan_text(args: argparse.Namespace) -> str:
    selected = [bool(args.text), bool(args.file), bool(args.stdin)]
    if sum(selected) > 1:
        raise SystemExit("Use only one of --text, --file, or --stdin.")
    if args.text:
        return args.text
    if args.file:
        return args.file.read_text()
    if args.stdin or not sys.stdin.isatty():
        return sys.stdin.read()
    raise SystemExit("Provide --text, --file, or pipe text on stdin.")


def max_decision(results: list[dict[str, Any]]) -> str:
    if not results:
        return "allow"
    return max((str(row.get("decision", "allow")) for row in results), key=lambda decision: DECISION_RANK.get(decision, -1))


def exit_code_for(results: list[dict[str, Any]], fail_on: str) -> int:
    if fail_on == "never":
        return 0
    threshold = DECISION_RANK[fail_on]
    return 2 if any(DECISION_RANK.get(str(row.get("decision")), -1) >= threshold for row in results) else 0


def scan_with_args(texts: list[str], args: argparse.Namespace) -> list[dict[str, Any]]:
    if not artifact_complete(args.artifact):
        raise SystemExit(
            "Missing default scanner artifact. Build it with:\n"
            "research/.venv-audit/bin/python -m research.train_probe_artifact "
            "--model qwen2.5-0.5b --feature-kind raw --train-source pooled-curated-core "
            "--layers 13,14,15 --layer-mode concat "
            "--artifact-id qwen-pooled-curated-core-l13-15-v2 "
            "--output-dir intentprobe/scanner/artifacts "
            "--overwrite --warn-threshold 0.30 --block-threshold 0.85 --pretty"
        )
    return scan_texts(
        texts,
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


def print_json(payload: Any, pretty: bool) -> None:
    print(json.dumps(payload, indent=2 if pretty else None, ensure_ascii=False))


def print_summary(results: list[dict[str, Any]]) -> None:
    for idx, row in enumerate(results, start=1):
        label = row.get("input_id") or f"input-{idx}"
        print(
            f"{label}: decision={row.get('decision')} "
            f"risk={float(row.get('risk_score', 0)):.3f} "
            f"activation={float(row.get('activation_score', 0)):.3f} "
            f"static={float(row.get('static_score', 0)):.3f}"
        )
        for reason in row.get("risk_reasons", [])[:3]:
            print(f"  - {reason}")


def command_scan(args: argparse.Namespace) -> int:
    result = scan_with_args([read_scan_text(args)], args)[0]
    if args.format == "summary":
        print_summary([result])
    else:
        print_json(result, args.pretty)
    return exit_code_for([result], args.fail_on)


def command_batch(args: argparse.Namespace) -> int:
    rows = read_batch_inputs(args.batch_file)
    results = scan_with_args([text for _, text in rows], args)
    for idx, (input_id, _) in enumerate(rows):
        if input_id is not None:
            results[idx]["input_id"] = input_id
    payload = {
        "mode": "activation_scanner_cli_batch",
        "scanner_version": CORE_VERSION,
        "decision_policy": DECISION_POLICY_NAME,
        "artifact": str(args.artifact),
        "count": len(results),
        "max_decision": max_decision(results),
        "results": results,
    }
    if args.format == "summary":
        print_summary(results)
    else:
        print_json(payload, args.pretty)
    return exit_code_for(results, args.fail_on)


def command_doctor(args: argparse.Namespace) -> int:
    complete = artifact_complete(args.artifact)
    payload: dict[str, Any] = {
        "scanner_version": CORE_VERSION,
        "decision_policy": DECISION_POLICY_NAME,
        "artifact": str(args.artifact),
        "artifact_complete": complete,
    }
    if complete:
        artifact_dir, metadata, _weights = load_artifact(args.artifact)
        payload.update(
            {
                "artifact_dir": str(artifact_dir),
                "artifact_id": metadata.get("artifact_id", artifact_dir.name),
                "model_id": metadata.get("model_id"),
                "sensor_model": metadata.get("sensor_model"),
                "feature_kind": metadata.get("feature_kind"),
                "feature_dim": metadata.get("feature_dim"),
                "thresholds": metadata.get("thresholds"),
                "layer_policy": metadata.get("layer_policy"),
            }
        )
    print_json(payload, args.pretty)
    return 0 if complete else 1


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--version", action="store_true", help="Print scanner version and exit.")
    subparsers = parser.add_subparsers(dest="command")

    scan = subparsers.add_parser("scan", help="Scan one tool, MCP, skill, hook, or prompt text.")
    scan.add_argument("--text", help="Text to scan.")
    scan.add_argument("--file", type=Path, help="File containing text to scan.")
    scan.add_argument("--stdin", action="store_true", help="Read text from stdin.")
    add_runtime_args(scan)
    scan.set_defaults(func=command_scan)

    batch = subparsers.add_parser("batch", help="Scan a JSON array of strings or {id,text} objects.")
    batch.add_argument("--batch-file", type=Path, required=True)
    add_runtime_args(batch)
    batch.set_defaults(func=command_batch)

    doctor = subparsers.add_parser("doctor", help="Check the cached scanner artifact.")
    doctor.add_argument("--artifact", type=Path, default=DEFAULT_ARTIFACT)
    doctor.add_argument("--pretty", action="store_true")
    doctor.set_defaults(func=command_doctor)

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.version:
        print(CORE_VERSION)
        return 0
    if not hasattr(args, "func"):
        parser.print_help()
        return 2
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
