# Security Policy

intentprobe is a local research-preview scanner for AI agent tooling. It is
designed to flag risky MCP servers, Claude Code skills, package descriptions,
and tool metadata before installation.

## Supported versions

The `main` branch is the supported research-preview line.

## Privacy model

intentprobe runs locally and does not upload scan targets or scan results to an
intentprobe service. A model-backed scan may download the configured local base
model from Hugging Face on first use. After that, scans use the local model
cache.

Do not paste live secrets, private keys, tokens, customer data, or proprietary
tool code into public GitHub issues.

## Reporting scanner failures

Use the GitHub issue templates when possible:

- Missed detection: a malicious or suspicious tool was allowed.
- False positive: a benign tool was warned or blocked.

Useful reports include:

- intentprobe version or commit.
- command used.
- operating system, Python version, and whether the scan ran on CPU or GPU.
- redacted tool description, `package.json`, MCP config, or `SKILL.md`.
- actual scanner decision and expected decision.

Please replace real secrets with placeholders such as `REDACTED_API_KEY`,
`REDACTED_PRIVATE_URL`, or `REDACTED_CUSTOMER_NAME`.

## Vulnerability reports

If you find a way to make intentprobe execute untrusted code, leak private scan
contents, corrupt files, or silently bypass a documented block, open a private
security advisory on GitHub if available. If private advisories are unavailable,
open a minimal public issue without exploit details and mark it as a security
report.

## Scope

In scope:

- scanner bypasses with a reproducible poisoned tool description.
- unsafe filesystem behavior in `scan-path`.
- secret redaction failures in hook or target scanning output.
- packaging or install behavior that executes untrusted code unexpectedly.

Out of scope:

- claims that require live private credentials.
- attacks against Hugging Face, PyPI, npm, GitHub, or another third-party
  service.
- reports that only say "the model is imperfect" without a reproducible sample.
