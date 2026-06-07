"""Shared static risk checks for the activation scanner preview."""

from __future__ import annotations

import re
from typing import Any


STATIC_PATTERNS: tuple[dict[str, Any], ...] = (
    {
        "id": "static-secret-file",
        "severity": "critical",
        "reason": "Mentions private keys, credential files, or .env-style secrets.",
        "regex": re.compile(
            r"\b(private\s+keys?|ssh\s+keys?|api\s+keys?|recovery[-\s]?codes?|backup[-\s]?codes?|secret\s+files?|credential\s+files?)\b"
            r"|(?<![\w.-])(?:\.env|\.npmrc|\.netrc)(?![\w.-])",
            re.I,
        ),
    },
    {
        "id": "static-secret-handling",
        "severity": "high",
        "reason": "Mentions risky handling of credentials, secrets, or auth tokens.",
        "regex": re.compile(
            r"\b(read|collect|copy|dump|print|log|expose|upload|send|forward|transmit|post|store|write)\b"
            r"[\s\S]{0,90}\b(credentials?|api\s+keys?|secrets?|private\s+keys?|"
            r"(?:api|access|auth|bearer|session|oauth|refresh)\s+tokens?)\b"
            r"|\b(credentials?|api\s+keys?|secrets?|private\s+keys?|"
            r"(?:api|access|auth|bearer|session|oauth|refresh)\s+tokens?)\b"
            r"[\s\S]{0,90}\b(read|collect|copy|dump|print|log|expose|upload|send|forward|transmit|post|store|write)\b",
            re.I,
        ),
    },
    {
        "id": "static-exfiltration",
        "severity": "critical",
        "reason": "Mentions uploading, forwarding, or transmitting data outside the local tool scope.",
        "regex": re.compile(
            r"\bexfiltrat(?:e|es|ed|ing|ion)\b"
            r"|\b(upload(?:s|ed|ing)?|send(?:s|ing)?|forward(?:s|ed|ing)?|transmit(?:s|ted|ting)?|post(?:s|ed|ing)?)\b"
            r"[\s\S]{0,90}\b(credentials?|api\s+keys?|secrets?|private\s+keys?|\.env|"
            r"(?:api|access|auth|bearer|session|oauth|refresh)\s+tokens?|screenshots?|clipboard|cookies?)\b"
            r"|\b(upload(?:s|ed|ing)?|send(?:s|ing)?|forward(?:s|ed|ing)?|transmit(?:s|ted|ting)?|post(?:s|ed|ing)?)\b"
            r"[\s\S]{0,90}\b(files?|payload|data|logs?|environment|host(?:name)?|inventory)\b"
            r"[\s\S]{0,120}\b(webhook|remote\s+server|third[-\s]?party|external|collector|unauthorized)\b",
            re.I,
        ),
    },
    {
        "id": "static-hidden-action",
        "severity": "high",
        "reason": "Mentions hidden, silent, or background behavior.",
        "regex": re.compile(
            r"\b(silently|quietly|covertly)\b[\s\S]{0,90}\b(upload(?:s|ed|ing)?|send(?:s|ing)?|forward(?:s|ed|ing)?|transmit(?:s|ted|ting)?|collect(?:s|ed|ing)?|log(?:s|ged|ging)?|store(?:s|d|ing)?|execute(?:s|d|ing)?|run(?:s|ning)?|monitor(?:s|ed|ing)?|cop(?:y|ies|ied|ying)|read(?:s|ing)?)\b"
            r"|\b(upload(?:s|ed|ing)?|send(?:s|ing)?|forward(?:s|ed|ing)?|transmit(?:s|ted|ting)?|collect(?:s|ed|ing)?|log(?:s|ged|ging)?|store(?:s|d|ing)?|execute(?:s|d|ing)?|run(?:s|ning)?|monitor(?:s|ed|ing)?|cop(?:y|ies|ied|ying)|read(?:s|ing)?)\b[\s\S]{0,90}\b(silently|quietly|covertly)\b"
            r"|\bwithout\s+(telling|notifying|showing)\b[\s\S]{0,90}\b(upload(?:s|ed|ing)?|send(?:s|ing)?|forward(?:s|ed|ing)?|transmit(?:s|ted|ting)?|collect(?:s|ed|ing)?|log(?:s|ged|ging)?|store(?:s|d|ing)?|execute(?:s|d|ing)?|run(?:s|ning)?|monitor(?:s|ed|ing)?|cop(?:y|ies|ied|ying)|read(?:s|ing)?)\b"
            r"|\bbackground\s+(process(?:es)?|task|job|service|sync|upload|send|transmit|monitor|logger|daemon|operation)\b",
            re.I,
        ),
    },
    {
        "id": "static-tool-chaining",
        "severity": "high",
        "reason": "Mentions forced tool ordering or instruction chaining.",
        "regex": re.compile(
            r"\bignore\s+previous\b"
            r"|\b(you\s+must|must)\s+(call|invoke|route)\b"
            r"|\bbefore\s+(using|calling|invoking)\s+(the\s+)?(other|external|file|email|search|tool|server)\b",
            re.I,
        ),
    },
    {
        "id": "static-host-modification",
        "severity": "high",
        "reason": "Mentions host-file edits, startup persistence, screenshots, SSH authorization, or shell subprocess actions.",
        "regex": re.compile(
            r"(?<![\w/])(/etc/(hosts|passwd|init\.d)|/boot/grub|authorized_keys|\.ssh/authorized_keys|sudo\s+reboot|"
            r"os\.system|subprocess\.(run|call|Popen)|scp\b|rsync\b|ssh\s+-L|gnome-screenshot|snippingtool|screencapture\b)(?![\w/])",
            re.I,
        ),
    },
)


STATIC_SCORES = {
    "critical": 0.95,
    "high": 0.80,
    "medium": 0.60,
    "low": 0.30,
}


def static_scan(text: str) -> list[dict[str, Any]]:
    findings: list[dict[str, Any]] = []
    for pattern in STATIC_PATTERNS:
        for match in pattern["regex"].finditer(text):
            start, end = match.span()
            context = text[max(0, start - 60) : min(len(text), end + 60)].strip()
            if is_benign_static_context(str(pattern["id"]), match.group(0), context):
                continue
            findings.append(
                {
                    "id": pattern["id"],
                    "severity": pattern["severity"],
                    "reason": pattern["reason"],
                    "matched_text": match.group(0),
                    "start": start,
                    "end": end,
                    "context": context,
                }
            )
            break
    return findings


def is_benign_static_context(finding_id: str, matched_text: str, context: str) -> bool:
    lower_context = context.lower()
    lower_match = matched_text.lower()
    if finding_id == "static-secret-handling":
        if "allowed-tools" in lower_context and "secret management" in lower_context:
            return True
        if "secret management" in lower_context and "secure credential storage" in lower_context:
            return True
    if finding_id == "static-exfiltration":
        alert_context = (
            "alerting.send" in lower_context
            or "pagerduty" in lower_context
            or "slack" in lower_context
        )
        if alert_context and ("secret rotation failed" in lower_context or "severity:" in lower_context):
            return True
        if "title:" in lower_match and "secret rotation failed" in lower_match:
            return True
    return False


def static_score(findings: list[dict[str, Any]]) -> float:
    return max((STATIC_SCORES.get(str(row.get("severity")), 0.0) for row in findings), default=0.0)


def choose_decision(risk_score: float, warn_threshold: float, block_threshold: float) -> str:
    if risk_score >= block_threshold:
        return "block"
    if risk_score >= warn_threshold:
        return "warn"
    return "allow"
