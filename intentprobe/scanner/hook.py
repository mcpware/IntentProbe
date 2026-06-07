#!/usr/bin/env python3
"""Hook-facing wrapper for the cached activation scanner.

This module accepts real-ish install/runtime payloads such as MCP configs,
tool definitions, skills, hook commands, or plain text. It normalizes them into
scanner subjects, redacts secret values, then emits a gate-friendly JSON result.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

try:
    from .cli import DEFAULT_ARTIFACT, artifact_complete
    from .core import CORE_VERSION, DECISION_POLICY_NAME, scan_texts
except ImportError:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    from intentprobe.scanner.cli import DEFAULT_ARTIFACT, artifact_complete  # type: ignore
    from intentprobe.scanner.core import CORE_VERSION, DECISION_POLICY_NAME, scan_texts  # type: ignore


DECISION_RANK = {
    "allow": 0,
    "warn": 1,
    "block": 2,
    "quarantine": 3,
}
SENSITIVE_KEY_PARTS = (
    "api_key",
    "apikey",
    "auth",
    "bearer",
    "client_secret",
    "credential",
    "key",
    "password",
    "private",
    "secret",
    "token",
)
TEXT_KEYS = (
    "text",
    "description",
    "prompt",
    "instructions",
    "content",
    "body",
    "schema",
    "command",
    "script",
)
RUNTIME_EVENT_KEYS = ("runtime_event", "event_type", "event", "phase")
RUNTIME_EVENT_HINTS = ("tool", "mcp", "call", "input", "argument", "response", "result", "output")
TOOL_INPUT_KEYS = ("tool_input", "arguments", "args", "input", "parameters", "request")
TOOL_RESPONSE_KEYS = ("tool_response", "tool_result", "response", "result", "output")
TOOL_DEFINITION_KEYS = ("tool_definition", "definition")


@dataclass(frozen=True)
class ScanSubject:
    subject_id: str
    kind: str
    text: str
    name: str | None = None
    source: str | None = None
    path: str | None = None


def is_sensitive_key(key: str) -> bool:
    lower = key.lower()
    return any(part in lower for part in SENSITIVE_KEY_PARTS)


def redact_for_scan(value: Any, parent_key: str = "") -> Any:
    if isinstance(value, dict):
        redacted: dict[str, Any] = {}
        for key, child in sorted(value.items(), key=lambda item: str(item[0])):
            key_text = str(key)
            if parent_key.lower() == "env" or is_sensitive_key(key_text):
                redacted[key_text] = redact_secret_value(child)
            else:
                redacted[key_text] = redact_for_scan(child, key_text)
        return redacted
    if isinstance(value, list):
        return [redact_for_scan(item, parent_key) for item in value]
    return value


def redact_secret_value(value: Any) -> str:
    if value in (None, ""):
        return "[REDACTED_EMPTY_VALUE]"
    if isinstance(value, str):
        return f"[REDACTED_VALUE len={len(value)}]"
    return f"[REDACTED_{type(value).__name__.upper()}]"


def stable_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(",", ":"))


def pretty_subject_text(value: Any) -> str:
    redacted = redact_for_scan(value)
    if isinstance(redacted, str):
        return redacted
    return json.dumps(redacted, sort_keys=True, ensure_ascii=False, indent=2)


def content_sha256(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def first_string(mapping: dict[str, Any], keys: tuple[str, ...]) -> str | None:
    for key in keys:
        value = mapping.get(key)
        if isinstance(value, str) and value:
            return value
    return None


def object_subject(mapping: dict[str, Any], subject_id: str, inherited_kind: str | None = None) -> ScanSubject:
    kind = first_string(mapping, ("kind", "type", "category", "subType")) or inherited_kind or "object"
    name = first_string(mapping, ("name", "id", "tool_name", "server", "title"))
    source = first_string(mapping, ("source", "origin", "scope"))
    path = first_string(mapping, ("path", "file", "file_path"))

    text: str | None = None
    for key in TEXT_KEYS:
        raw = mapping.get(key)
        if isinstance(raw, str) and raw.strip():
            text = raw
            break
    if text is None:
        text = pretty_subject_text(mapping)
    elif len(mapping) > 1:
        text = pretty_subject_text(mapping)

    return ScanSubject(
        subject_id=subject_id,
        kind=kind,
        name=name,
        source=source,
        path=path,
        text=text,
    )


def first_present(mapping: dict[str, Any], keys: tuple[str, ...]) -> tuple[str, Any] | None:
    for key in keys:
        if key in mapping:
            return key, mapping[key]
    return None


def runtime_event_name(payload: dict[str, Any]) -> str | None:
    for key in RUNTIME_EVENT_KEYS:
        value = payload.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    return None


def tool_name_from_payload(payload: dict[str, Any]) -> str | None:
    direct = first_string(payload, ("tool_name", "name", "server", "id"))
    if direct:
        return direct
    tool = payload.get("tool")
    if isinstance(tool, str) and tool.strip():
        return tool.strip()
    if isinstance(tool, dict):
        nested = first_string(tool, ("name", "id", "tool_name"))
        if nested:
            return nested
    tool_call = payload.get("tool_call")
    if isinstance(tool_call, dict):
        nested = first_string(tool_call, ("name", "id", "tool_name"))
        if nested:
            return nested
    return None


def looks_like_runtime_event(payload: dict[str, Any]) -> bool:
    event = runtime_event_name(payload)
    if event and any(hint in event.lower() for hint in RUNTIME_EVENT_HINTS):
        return True
    if any(key in payload for key in ("tool_call", "tool_result", "tool_input", "tool_response")):
        return True
    has_tool_identity = isinstance(payload.get("tool_name"), str) or isinstance(payload.get("tool"), str)
    has_runtime_io = any(key in payload for key in (*TOOL_INPUT_KEYS, *TOOL_RESPONSE_KEYS))
    return bool(has_tool_identity and has_runtime_io)


def runtime_part_subject(
    *,
    payload: dict[str, Any],
    subject_id: str,
    kind: str,
    key: str,
    value: Any,
) -> ScanSubject:
    event = runtime_event_name(payload)
    tool_name = tool_name_from_payload(payload)
    if kind in {"runtime_tool_input", "runtime_tool_response"}:
        return ScanSubject(
            subject_id=subject_id,
            kind=kind,
            name=tool_name,
            source=first_string(payload, ("source", "origin", "scope")),
            path=first_string(payload, ("path", "file", "file_path")),
            text=pretty_subject_text(value),
        )
    part_payload = {
        "kind": kind,
        "event_type": event,
        "tool_name": tool_name,
        "source": payload.get("source"),
        "path": payload.get("path"),
        key: value,
    }
    return object_subject(part_payload, subject_id, kind)


def runtime_event_subjects(payload: dict[str, Any], subject_id: str) -> list[ScanSubject]:
    subjects: list[ScanSubject] = []

    tools = payload.get("tools")
    if isinstance(tools, list):
        for idx, tool in enumerate(tools):
            if isinstance(tool, dict):
                tool_id = first_string(tool, ("name", "id", "tool_name")) or str(idx + 1)
                subjects.append(
                    object_subject(
                        {**tool, "kind": "runtime_tool_definition", "source": payload.get("source"), "path": payload.get("path")},
                        f"{subject_id}-definition-{tool_id}",
                        "runtime_tool_definition",
                    )
                )
            else:
                subjects.extend(normalize_payload(tool, f"{subject_id}-definition-{idx + 1}"))

    tool = payload.get("tool")
    if isinstance(tool, dict):
        tool_id = first_string(tool, ("name", "id", "tool_name")) or "tool"
        subjects.append(
            object_subject(
                {**tool, "kind": "runtime_tool_definition", "source": payload.get("source"), "path": payload.get("path")},
                f"{subject_id}-definition-{tool_id}",
                "runtime_tool_definition",
            )
        )

    definition = first_present(payload, TOOL_DEFINITION_KEYS)
    if definition is not None:
        key, value = definition
        subjects.append(
            runtime_part_subject(
                payload=payload,
                subject_id=f"{subject_id}-definition",
                kind="runtime_tool_definition",
                key=key,
                value=value,
            )
        )

    tool_call = payload.get("tool_call")
    if isinstance(tool_call, dict):
        call_id = first_string(tool_call, ("name", "id", "tool_name")) or tool_name_from_payload(payload) or "tool_call"
        subjects.append(
            object_subject(
                {
                    "kind": "runtime_tool_call",
                    "event_type": runtime_event_name(payload),
                    "tool_name": tool_name_from_payload(payload) or call_id,
                    "source": payload.get("source"),
                    "path": payload.get("path"),
                    "tool_call": tool_call,
                },
                f"{subject_id}-call-{call_id}",
                "runtime_tool_call",
            )
        )

    if isinstance(tool_call, dict):
        nested_input = first_present(tool_call, TOOL_INPUT_KEYS)
        if nested_input is not None and first_present(payload, TOOL_INPUT_KEYS) is None:
            key, value = nested_input
            subjects.append(
                runtime_part_subject(
                    payload={**payload, "tool_name": tool_name_from_payload(payload) or first_string(tool_call, ("name", "id", "tool_name"))},
                    subject_id=f"{subject_id}-input",
                    kind="runtime_tool_input",
                    key=key,
                    value=value,
                )
            )

    tool_input = first_present(payload, TOOL_INPUT_KEYS)
    if tool_input is not None:
        key, value = tool_input
        subjects.append(
            runtime_part_subject(
                payload=payload,
                subject_id=f"{subject_id}-input",
                kind="runtime_tool_input",
                key=key,
                value=value,
            )
        )

    tool_response = first_present(payload, TOOL_RESPONSE_KEYS)
    if tool_response is not None:
        key, value = tool_response
        subjects.append(
            runtime_part_subject(
                payload=payload,
                subject_id=f"{subject_id}-response",
                kind="runtime_tool_response",
                key=key,
                value=value,
            )
        )

    if subjects:
        return subjects
    return [object_subject({"kind": "runtime_event", **payload}, subject_id, "runtime_event")]


def normalize_payload(payload: Any, subject_id: str = "input") -> list[ScanSubject]:
    if isinstance(payload, str):
        return [ScanSubject(subject_id=subject_id, kind="text", text=payload)]

    if isinstance(payload, list):
        subjects: list[ScanSubject] = []
        for idx, item in enumerate(payload):
            subjects.extend(normalize_payload(item, f"{subject_id}-{idx + 1}"))
        return subjects

    if not isinstance(payload, dict):
        return [ScanSubject(subject_id=subject_id, kind=type(payload).__name__, text=pretty_subject_text(payload))]

    item_groups = payload.get("items") or payload.get("subjects")
    if isinstance(item_groups, list):
        subjects = []
        for idx, item in enumerate(item_groups):
            subjects.extend(normalize_payload(item, f"{subject_id}-{idx + 1}"))
        return subjects

    if looks_like_runtime_event(payload):
        return runtime_event_subjects(payload, subject_id)

    for servers_key in ("mcpServers", "mcp_servers"):
        servers = payload.get(servers_key)
        if isinstance(servers, dict):
            subjects = []
            for name, config in sorted(servers.items(), key=lambda item: str(item[0])):
                server_payload = {
                    "kind": "mcp_server",
                    "name": str(name),
                    "mcp_config": config,
                    "source": payload.get("source"),
                    "path": payload.get("path"),
                }
                subjects.append(object_subject(server_payload, f"{subject_id}-mcp-{name}", "mcp_server"))
            return subjects

    tools = payload.get("tools")
    if isinstance(tools, list):
        subjects = []
        for idx, tool in enumerate(tools):
            if isinstance(tool, dict):
                tool_id = first_string(tool, ("name", "id", "tool_name")) or str(idx + 1)
                subjects.append(object_subject({"kind": "tool", **tool}, f"{subject_id}-tool-{tool_id}", "tool"))
            else:
                subjects.extend(normalize_payload(tool, f"{subject_id}-tool-{idx + 1}"))
        return subjects

    tool = payload.get("tool")
    if isinstance(tool, dict):
        tool_id = first_string(tool, ("name", "id", "tool_name")) or "tool"
        return [object_subject({"kind": "tool", **tool}, f"{subject_id}-{tool_id}", "tool")]

    return [object_subject(payload, subject_id)]


def parse_input(raw: str, input_format: str) -> Any:
    if input_format == "text":
        return raw
    if input_format == "json":
        return json.loads(raw)
    stripped = raw.lstrip()
    if stripped.startswith("{") or stripped.startswith("[") or stripped.startswith('"'):
        try:
            return json.loads(raw)
        except json.JSONDecodeError:
            return raw
    return raw


def read_payload(args: argparse.Namespace) -> Any:
    selected = [args.text is not None, args.input_file is not None, args.stdin]
    if sum(selected) > 1:
        raise SystemExit("Use only one of --text, --input-file, or --stdin.")
    if args.text is not None:
        raw = args.text
    elif args.input_file is not None:
        raw = args.input_file.read_text()
    elif args.stdin or not sys.stdin.isatty():
        raw = sys.stdin.read()
    else:
        raise SystemExit("Provide --text, --input-file, or pipe payload on stdin.")
    return parse_input(raw, args.input_format)


def max_decision(results: list[dict[str, Any]]) -> str:
    if not results:
        return "allow"
    return max((str(row.get("decision", "allow")) for row in results), key=lambda decision: DECISION_RANK.get(decision, -1))


def exit_code_for_decision(decision: str, fail_on: str) -> int:
    if fail_on == "never":
        return 0
    return 2 if DECISION_RANK.get(decision, -1) >= DECISION_RANK[fail_on] else 0


def subject_public(subject: ScanSubject) -> dict[str, Any]:
    return {
        "id": subject.subject_id,
        "kind": subject.kind,
        "name": subject.name,
        "source": subject.source,
        "path": subject.path,
        "content_sha256": content_sha256(subject.text),
        "text_length": len(subject.text),
    }


def build_normalized_payload(subjects: list[ScanSubject]) -> dict[str, Any]:
    return {
        "mode": "activation_scanner_hook_normalize",
        "scanner_version": CORE_VERSION,
        "decision_policy": DECISION_POLICY_NAME,
        "count": len(subjects),
        "subjects": [
            {
                "subject": subject_public(subject),
                "text": subject.text,
            }
            for subject in subjects
        ],
    }


def scan_subjects(subjects: list[ScanSubject], args: argparse.Namespace) -> dict[str, Any]:
    if not subjects:
        raise SystemExit("No scanner subjects found in payload.")
    if not artifact_complete(args.artifact):
        raise SystemExit(f"Missing scanner artifact: {args.artifact}")

    started = time.perf_counter()
    risk_results = scan_texts(
        [subject.text for subject in subjects],
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
    results = []
    for subject, risk in zip(subjects, risk_results, strict=True):
        results.append(
            {
                "subject": subject_public(subject),
                "decision": risk.get("decision"),
                "risk_score": risk.get("risk_score"),
                "risk": risk,
            }
        )

    decision = max_decision(results)
    return {
        "mode": "activation_scanner_hook",
        "scanner_version": CORE_VERSION,
        "decision_policy": DECISION_POLICY_NAME,
        "artifact": str(args.artifact),
        "count": len(results),
        "max_decision": decision,
        "gate": {
            "decision": decision,
            "fail_on": args.fail_on,
            "exit_code": exit_code_for_decision(decision, args.fail_on),
        },
        "results": results,
        "elapsed_seconds": time.perf_counter() - started,
    }


def print_json(payload: Any, pretty: bool) -> None:
    print(json.dumps(payload, indent=2 if pretty else None, ensure_ascii=False))


def add_payload_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--text", help="Plain text or JSON payload to scan.")
    parser.add_argument("--input-file", type=Path, help="Read text or JSON payload from a file.")
    parser.add_argument("--stdin", action="store_true", help="Read text or JSON payload from stdin.")
    parser.add_argument("--input-format", choices=("auto", "json", "text"), default="auto")


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
        default="block",
        help="Set gate.exit_code=2 when max_decision reaches this decision or higher.",
    )
    parser.add_argument("--pretty", action="store_true")


def command_normalize(args: argparse.Namespace) -> int:
    subjects = normalize_payload(read_payload(args))
    print_json(build_normalized_payload(subjects), args.pretty)
    return 0


def command_scan(args: argparse.Namespace) -> int:
    subjects = normalize_payload(read_payload(args))
    payload = scan_subjects(subjects, args)
    print_json(payload, args.pretty)
    return int(payload["gate"]["exit_code"])


def error_payload(message: str, fail_on: str) -> dict[str, Any]:
    decision = "quarantine"
    return {
        "mode": "activation_scanner_hook_error",
        "scanner_version": CORE_VERSION,
        "decision_policy": DECISION_POLICY_NAME,
        "max_decision": decision,
        "gate": {
            "decision": decision,
            "fail_on": fail_on,
            "exit_code": exit_code_for_decision(decision, fail_on),
        },
        "error": message,
    }


def command_serve_jsonl(args: argparse.Namespace) -> int:
    if not artifact_complete(args.artifact):
        raise SystemExit(f"Missing scanner artifact: {args.artifact}")
    if args.warmup:
        warm_subject = ScanSubject("warmup", "text", "Returns today's weather forecast for a requested city as JSON.")
        scan_subjects([warm_subject], args)
        print(
            stable_json(
                {
                    "mode": "activation_scanner_hook_ready",
                    "scanner_version": CORE_VERSION,
                    "decision_policy": DECISION_POLICY_NAME,
                    "artifact": str(args.artifact),
                }
            ),
            file=sys.stderr,
            flush=True,
        )

    for raw_line in sys.stdin:
        raw_line = raw_line.strip()
        if not raw_line:
            continue
        try:
            payload = parse_input(raw_line, args.input_format)
            output = scan_subjects(normalize_payload(payload), args)
        except Exception as exc:  # keep the protocol alive and fail closed per request
            output = error_payload(str(exc), args.fail_on)
        print(stable_json(output), flush=True)
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command")

    normalize = subparsers.add_parser("normalize", help="Normalize/redact a hook payload without loading the model.")
    add_payload_args(normalize)
    normalize.add_argument("--pretty", action="store_true")
    normalize.set_defaults(func=command_normalize)

    scan = subparsers.add_parser("scan", help="Scan one hook payload and exit with a gate code.")
    add_payload_args(scan)
    add_runtime_args(scan)
    scan.set_defaults(func=command_scan)

    serve = subparsers.add_parser("serve-jsonl", help="Keep a warm scanner process and scan one JSON/text payload per line.")
    add_runtime_args(serve)
    serve.add_argument("--input-format", choices=("auto", "json", "text"), default="auto")
    serve.add_argument("--warmup", dest="warmup", action="store_true", default=True)
    serve.add_argument("--no-warmup", dest="warmup", action="store_false")
    serve.set_defaults(func=command_serve_jsonl)

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if not hasattr(args, "func"):
        parser.print_help()
        return 2
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
