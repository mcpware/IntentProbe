"""MCP client config discovery and product-facing config scan helpers."""

from __future__ import annotations

import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .core import CORE_VERSION, DECISION_POLICY_NAME
from .hook import ScanSubject, max_decision, object_subject


DECISION_RANK = {
    "allow": 0,
    "warn": 1,
    "block": 2,
    "quarantine": 3,
}

REVIEW_FLAG_IDS = {
    "env-secrets",
    "remote-http",
    "browser-access",
    "filesystem-access",
    "code-execution",
    "email-or-identity",
    "repo-or-ticketing-access",
}


@dataclass(frozen=True)
class ConfigCandidate:
    source: str
    path: Path


@dataclass(frozen=True)
class ConfigServer:
    source: str
    path: Path
    name: str
    config: dict[str, Any]
    subject: ScanSubject


def default_config_candidates(cwd: Path | None = None) -> list[ConfigCandidate]:
    """Return common local MCP config paths.

    These are intentionally client-config paths, not broad repo searches. A
    broad filesystem scan belongs to scan-path; scan-config is the "what MCPs
    are configured on this machine?" product entrypoint.
    """

    home = Path.home()
    working_dir = cwd or Path.cwd()
    candidates = [
        ConfigCandidate("Claude Desktop", home / "Library/Application Support/Claude/claude_desktop_config.json"),
        ConfigCandidate("Claude Code", home / ".claude/mcp.json"),
        ConfigCandidate("Cursor", home / ".cursor/mcp.json"),
        ConfigCandidate("Cursor User", home / "Library/Application Support/Cursor/User/mcp.json"),
        ConfigCandidate("Windsurf", home / ".codeium/windsurf/mcp_config.json"),
        ConfigCandidate("Repo .mcp.json", working_dir / ".mcp.json"),
    ]
    return dedupe_candidates(candidates)


def dedupe_candidates(candidates: list[ConfigCandidate]) -> list[ConfigCandidate]:
    seen: set[Path] = set()
    deduped: list[ConfigCandidate] = []
    for candidate in candidates:
        path = candidate.path.expanduser()
        key = path.resolve() if path.exists() else path
        if key in seen:
            continue
        seen.add(key)
        deduped.append(ConfigCandidate(candidate.source, path))
    return deduped


def config_candidates_from_target(target: str | Path | None, cwd: Path | None = None) -> list[ConfigCandidate]:
    if target is None or str(target) == "auto":
        return default_config_candidates(cwd)
    return [ConfigCandidate("custom", Path(target).expanduser())]


def load_json_config(path: Path, max_file_bytes: int) -> tuple[dict[str, Any] | None, str | None]:
    try:
        raw = path.read_bytes()
    except OSError as exc:
        return None, f"read_error: {exc}"
    if len(raw) > max_file_bytes:
        return None, f"file_too_large: {len(raw)} bytes > {max_file_bytes}"
    try:
        payload = json.loads(raw.decode("utf-8"))
    except UnicodeDecodeError as exc:
        return None, f"decode_error: {exc}"
    except json.JSONDecodeError as exc:
        return None, f"json_error: {exc}"
    if not isinstance(payload, dict):
        return None, f"unsupported_json_type: {type(payload).__name__}"
    return payload, None


def extract_mcp_servers(payload: dict[str, Any]) -> tuple[dict[str, Any], str | None]:
    for key in ("mcpServers", "mcp_servers"):
        value = payload.get(key)
        if isinstance(value, dict):
            return value, key
    return {}, None


def subject_for_server(source: str, path: Path, name: str, config: dict[str, Any]) -> ScanSubject:
    server_payload = {
        "kind": "mcp_server",
        "name": name,
        "mcp_config": config,
        "source": source,
        "path": str(path),
    }
    return object_subject(server_payload, f"config-{safe_id(source)}-{safe_id(name)}", "mcp_server")


def safe_id(value: str) -> str:
    return "".join(ch if ch.isalnum() or ch in "._-" else "-" for ch in value).strip("-") or "item"


def collect_config_servers(
    candidates: list[ConfigCandidate],
    *,
    max_file_bytes: int,
) -> tuple[list[dict[str, Any]], list[ConfigServer]]:
    configs: list[dict[str, Any]] = []
    servers: list[ConfigServer] = []

    for candidate in candidates:
        path = candidate.path.expanduser()
        public_config: dict[str, Any] = {
            "source": candidate.source,
            "path": str(path),
            "exists": path.exists(),
            "status": "missing",
            "server_count": 0,
        }
        if not path.exists():
            configs.append(public_config)
            continue

        payload, error = load_json_config(path, max_file_bytes)
        if error is not None or payload is None:
            public_config.update({"status": "invalid", "error": error})
            configs.append(public_config)
            continue

        server_map, servers_key = extract_mcp_servers(payload)
        public_config["servers_key"] = servers_key
        public_config["server_count"] = len(server_map)
        if not server_map:
            public_config["status"] = "no_mcp_servers"
            public_config["skipped_reason"] = "No top-level mcpServers/mcp_servers object found."
            configs.append(public_config)
            continue

        public_config["status"] = "scanned"
        public_config["servers"] = sorted(str(name) for name in server_map)
        configs.append(public_config)

        for name, raw_config in sorted(server_map.items(), key=lambda item: str(item[0])):
            config = raw_config if isinstance(raw_config, dict) else {"value": raw_config}
            server_name = str(name)
            servers.append(
                ConfigServer(
                    source=candidate.source,
                    path=path,
                    name=server_name,
                    config=config,
                    subject=subject_for_server(candidate.source, path, server_name, config),
                )
            )

    return configs, servers


def inventory_flags(server: ConfigServer) -> list[dict[str, str]]:
    flags: list[dict[str, str]] = []
    config = server.config
    command = str(config.get("command") or "")
    args = " ".join(str(arg) for arg in config.get("args") or [])
    url = str(config.get("url") or "")
    server_text = " ".join([server.name, command, args, url]).lower()

    if isinstance(config.get("env"), dict) and config["env"]:
        flags.append(
            {
                "id": "env-secrets",
                "level": "review",
                "reason": "server config includes environment variables; values are redacted",
            }
        )

    if config.get("type") == "http" or url.startswith(("http://", "https://")):
        flags.append(
            {
                "id": "remote-http",
                "level": "review",
                "reason": "server connects to a remote MCP endpoint",
            }
        )

    if command in {"npx", "uvx", "pipx"}:
        flags.append(
            {
                "id": "package-runner",
                "level": "info",
                "reason": f"server launches through {command}; package supply-chain should be pinned/reviewed",
            }
        )

    if any(term in server_text for term in ("chrome", "browser", "playwright", "screenshot")):
        flags.append(
            {
                "id": "browser-access",
                "level": "review",
                "reason": "server appears browser-capable",
            }
        )

    if any(term in server_text for term in ("filesystem", "file-system", "file_system", "fs-", "local-file")):
        flags.append(
            {
                "id": "filesystem-access",
                "level": "review",
                "reason": "server appears file-capable",
            }
        )

    if any(term in server_text for term in ("shell", "terminal", "exec", "code-run", "python", "node")):
        flags.append(
            {
                "id": "code-execution",
                "level": "review",
                "reason": "server may execute local code or commands",
            }
        )

    if any(term in server_text for term in ("gmail", "email", "calendar", "oauth", "identity")):
        flags.append(
            {
                "id": "email-or-identity",
                "level": "review",
                "reason": "server appears connected to email, calendar, OAuth, or identity data",
            }
        )

    if any(term in server_text for term in ("github", "gitlab", "atlassian", "jira", "linear")):
        flags.append(
            {
                "id": "repo-or-ticketing-access",
                "level": "review",
                "reason": "server appears connected to code repositories or ticketing systems",
            }
        )

    return flags


def product_decision_for_config_scan(risk: dict[str, Any], flags: list[dict[str, str]]) -> tuple[str, list[str]]:
    """Turn raw probe output into an operator decision for short MCP manifests.

    Short MCP config manifests are mostly command/package inventory, not full
    tool descriptions. The raw activation score is still returned, but an
    activation-only high score should not become a scary alert without static
    poison evidence or review-worthy inventory capability.
    """

    scanner_decision = str(risk.get("decision", "allow"))
    static_score = float(risk.get("static_score") or 0.0)
    review_flags = [flag for flag in flags if flag.get("id") in REVIEW_FLAG_IDS]
    reasons: list[str] = []

    if scanner_decision == "block":
        reasons.append("poison evidence reached scanner block policy")
        return "block", reasons

    if static_score > 0:
        reasons.append("static poison evidence found in config text")
        return "warn", reasons

    if review_flags:
        reasons.append("review-worthy inventory capability found")
        return "warn", reasons

    if scanner_decision == "warn":
        reasons.append("activation-only config-manifest signal downgraded to inventory-only")
    else:
        reasons.append("no poison evidence or review-worthy inventory capability")
    return "allow", reasons


def summarize_counts(results: list[dict[str, Any]], configs: list[dict[str, Any]]) -> dict[str, Any]:
    decision_counts = {decision: 0 for decision in ("allow", "warn", "block", "quarantine")}
    for row in results:
        decision = str(row.get("decision", "allow"))
        decision_counts[decision] = decision_counts.get(decision, 0) + 1
    return {
        "configs_checked": len(configs),
        "configs_found": sum(1 for config in configs if config.get("exists")),
        "configs_with_mcp_servers": sum(1 for config in configs if config.get("server_count", 0) > 0),
        "mcp_servers_found": sum(int(config.get("server_count", 0)) for config in configs),
        "servers_scanned": len(results),
        "decision_counts": decision_counts,
        "skipped_no_mcp_servers": sum(1 for config in configs if config.get("status") == "no_mcp_servers"),
        "missing_configs": sum(1 for config in configs if config.get("status") == "missing"),
        "invalid_configs": sum(1 for config in configs if config.get("status") == "invalid"),
    }


def gate_for_results(results: list[dict[str, Any]], fail_on: str) -> dict[str, Any]:
    decision = max_decision(results)
    if fail_on == "never":
        exit_code = 0
    else:
        exit_code = 2 if DECISION_RANK.get(decision, -1) >= DECISION_RANK[fail_on] else 0
    return {"decision": decision, "fail_on": fail_on, "exit_code": exit_code}


def build_config_scan_payload(
    *,
    target: str,
    configs: list[dict[str, Any]],
    servers: list[ConfigServer],
    scan_payload: dict[str, Any] | None,
    fail_on: str,
) -> dict[str, Any]:
    started = time.perf_counter()
    raw_results = scan_payload.get("results", []) if scan_payload else []
    results: list[dict[str, Any]] = []

    for server, raw in zip(servers, raw_results, strict=True):
        risk = raw.get("risk") or {}
        flags = inventory_flags(server)
        decision, reasons = product_decision_for_config_scan(risk, flags)
        results.append(
            {
                "server": {
                    "name": server.name,
                    "source": server.source,
                    "path": str(server.path),
                    "command": server.config.get("command"),
                    "args_count": len(server.config.get("args") or []),
                    "type": server.config.get("type"),
                    "url_host": host_from_url(str(server.config.get("url") or "")),
                    "has_env": isinstance(server.config.get("env"), dict) and bool(server.config.get("env")),
                    "config_sha256": raw.get("subject", {}).get("content_sha256"),
                },
                "decision": decision,
                "decision_reasons": reasons,
                "inventory_flags": flags,
                "scanner_decision": raw.get("decision"),
                "risk_score": risk.get("risk_score"),
                "activation_score": risk.get("activation_score"),
                "static_score": risk.get("static_score"),
                "risk_reasons": risk.get("risk_reasons", []),
                "evidence_spans": risk.get("evidence_spans", []),
            }
        )

    counts = summarize_counts(results, configs)
    return {
        "mode": "activation_scanner_cli_config",
        "scanner_version": CORE_VERSION,
        "decision_policy": DECISION_POLICY_NAME,
        "target": target,
        "configs": configs,
        "inventory": counts,
        "gate": gate_for_results(results, fail_on),
        "results": results,
        "elapsed_seconds": (scan_payload or {}).get("elapsed_seconds", time.perf_counter() - started),
        "notes": [
            "Configs without mcpServers/mcp_servers are inventoried but not activation-scanned.",
            "For short MCP manifests, activation-only warn signals are returned as scanner_decision but downgraded unless static poison evidence or review-worthy inventory capability exists.",
        ],
    }


def host_from_url(url: str) -> str | None:
    if not url.startswith(("http://", "https://")):
        return None
    without_scheme = url.split("://", 1)[1]
    return without_scheme.split("/", 1)[0] or None
