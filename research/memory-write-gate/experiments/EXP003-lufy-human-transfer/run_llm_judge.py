#!/usr/bin/env python3
"""Run frozen prompted Codex or Claude baselines on quarantined LUFY text."""

from __future__ import annotations

import argparse
import json
import subprocess
import tempfile
from pathlib import Path


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
SAMPLE = (
    ROOT / "data" / "processed" / "exp003" / "llm_judge_unanimous_sample.jsonl"
)
OUTPUT_DIR = ROOT / "data" / "processed" / "exp003" / "llm_judge_outputs"
CLAUDE_WRAPPER = "/home/nicole/.local/bin/claude-codex-worker"

PROMPT_PREFIX = """You are a frozen benchmark classifier. Do not use tools and do not explain.

For each user utterance, predict whether a human annotator would mark the information as important
because it would be used in future conversations.

- SAVE: the utterance contains information that would be useful to carry into future conversations.
- SKIP: it does not contain such information.

Return exactly one prediction for every sample_id. save_probability must be a number from 0 to 1.
Use only the supplied utterance; do not invent missing context.

Items:
"""

SCHEMA = {
    "type": "object",
    "properties": {
        "predictions": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "sample_id": {"type": "string"},
                    "label": {"type": "string", "enum": ["SAVE", "SKIP"]},
                    "save_probability": {"type": "number", "minimum": 0, "maximum": 1},
                },
                "required": ["sample_id", "label", "save_probability"],
                "additionalProperties": False,
            },
        }
    },
    "required": ["predictions"],
    "additionalProperties": False,
}


def load_items(limit: int | None) -> list[dict[str, str]]:
    items: list[dict[str, str]] = []
    with SAMPLE.open(encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            items.append({"sample_id": row["sample_id"], "utterance": row["text"]})
    return items if limit is None else items[:limit]


def validate(result: dict[str, object], expected_ids: list[str]) -> None:
    predictions = result.get("predictions")
    if not isinstance(predictions, list):
        raise ValueError("Missing predictions array")
    observed_ids = [item.get("sample_id") for item in predictions if isinstance(item, dict)]
    if len(observed_ids) != len(expected_ids) or set(observed_ids) != set(expected_ids):
        raise ValueError("Prediction IDs are missing, duplicated, or unexpected")
    for item in predictions:
        if item["label"] not in {"SAVE", "SKIP"}:
            raise ValueError(f"Unexpected label: {item['label']}")
        probability = float(item["save_probability"])
        if not 0 <= probability <= 1:
            raise ValueError(f"Probability out of range: {probability}")


def run_codex(prompt: str, schema_path: Path, raw_path: Path) -> tuple[dict[str, object], dict[str, object]]:
    command = [
        "codex",
        "exec",
        "--ephemeral",
        "--ignore-user-config",
        "--ignore-rules",
        "--skip-git-repo-check",
        "--sandbox",
        "read-only",
        "--model",
        "gpt-6.1-sol",
        "--output-schema",
        str(schema_path),
        "--output-last-message",
        str(raw_path),
        "-",
    ]
    completed = subprocess.run(
        command,
        input=prompt,
        text=True,
        capture_output=True,
        timeout=900,
        check=True,
        cwd="/tmp",
    )
    result = json.loads(raw_path.read_text(encoding="utf-8"))
    metadata = {
        "provider": "openai",
        "requested_model": "gpt-6.1-sol",
        "command_exit_code": completed.returncode,
    }
    return result, metadata


def run_claude(prompt: str, schema: dict[str, object], raw_path: Path) -> tuple[dict[str, object], dict[str, object]]:
    command = [
        CLAUDE_WRAPPER,
        "--print",
        "--safe-mode",
        "--restricted",
        "--no-session-persistence",
        "--permission-prompts",
        "none",
        "--model",
        "sonnet",
        "--effort",
        "medium",
        "--output-format",
        "json",
        "--json-schema",
        json.dumps(schema, separators=(",", ":")),
    ]
    completed = subprocess.run(
        command,
        input=prompt,
        text=True,
        capture_output=True,
        timeout=900,
        check=True,
        cwd="/tmp",
    )
    raw_path.write_text(completed.stdout, encoding="utf-8")
    envelope = json.loads(completed.stdout)
    result = envelope.get("structured_output")
    if not isinstance(result, dict):
        raise ValueError("Claude response did not contain structured_output")
    metadata = {
        "provider": "anthropic",
        "requested_model": "sonnet",
        "command_exit_code": completed.returncode,
        "model_usage": envelope.get("modelUsage"),
        "total_cost_usd": envelope.get("total_cost_usd"),
        "duration_ms": envelope.get("duration_ms"),
    }
    return result, metadata


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("provider", choices=("codex", "claude"))
    parser.add_argument("--limit", type=int)
    args = parser.parse_args()
    items = load_items(args.limit)
    prompt = PROMPT_PREFIX + json.dumps(items, ensure_ascii=False, separators=(",", ":"))
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    suffix = "full" if args.limit is None else f"pilot_{args.limit}"
    raw_path = OUTPUT_DIR / f"{args.provider}_{suffix}_raw.json"
    with tempfile.TemporaryDirectory(prefix="intentprobe-judge-") as temporary:
        schema_path = Path(temporary) / "schema.json"
        schema_path.write_text(json.dumps(SCHEMA), encoding="utf-8")
        if args.provider == "codex":
            result, metadata = run_codex(prompt, schema_path, raw_path)
        else:
            result, metadata = run_claude(prompt, SCHEMA, raw_path)
    expected_ids = [item["sample_id"] for item in items]
    validate(result, expected_ids)
    output = {
        "experiment_id": "MEM-EXP003",
        "provider": args.provider,
        "sample": "unanimous_balanced_lufy",
        "rows": len(items),
        "prompt_version": "future-use-v1",
        "metadata": metadata,
        "predictions": result["predictions"],
    }
    output_path = OUTPUT_DIR / f"{args.provider}_{suffix}.json"
    output_path.write_text(json.dumps(output, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps({"status": "COMPLETE", "provider": args.provider, "rows": len(items), "output": str(output_path)}, indent=2))


if __name__ == "__main__":
    main()
