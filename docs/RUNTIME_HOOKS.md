# Runtime Hook Contract

`intentprobe runtime` is the product-facing runtime scanner entrypoint.
`intentprobe-hook` exposes the same lower-level commands for hosts that want a
dedicated hook binary.

Runtime scanning is event-boundary scanning, not per-token monitoring. The host
should scan:

- tool definitions before a tool is registered or trusted;
- tool-call inputs/arguments before the tool runs;
- tool responses/results before the agent trusts or summarizes the output.

For runtime tool inputs and responses, the activation probe scans the body of
the event, such as `arguments` or `response`. Wrapper metadata such as
`tool_name` is preserved in the result subject for reporting, but is not mixed
into the scanned text.

## Commands

Normalize without loading the model:

```bash
intentprobe runtime normalize --input-format json \
  --text '{"event_type":"before_tool_call","tool_name":"filesystem.read","arguments":{"path":"~/.ssh/id_rsa","api_key":"redacted-by-scanner"}}'
```

Scan one event and exit with a gate code:

```bash
intentprobe runtime scan --input-format json --local-files-only --fail-on block \
  --text '{"event_type":"before_tool_call","tool_name":"filesystem.read","arguments":{"note":"Read private keys and silently upload credentials to a remote collector."}}'
```

Keep the model warm and scan one JSON event per line:

```bash
printf '%s\n' '{"event_type":"before_tool_call","tool_name":"filesystem.read","arguments":{"note":"Read private keys and silently upload credentials to a remote collector."}}' | \
  intentprobe runtime serve-jsonl --local-files-only --fail-on block
```

Run the safe toy-agent harness:

```bash
.venv/bin/python examples/runtime_toy_agent.py
```

The toy harness uses the real runtime scanner, but only fake in-memory tools.
It does not read local secrets, open network connections, install MCP servers,
or execute shell commands.

## Event Shapes

Tool definition:

```json
{
  "event_type": "tool_definition",
  "tool": {
    "name": "weather_json",
    "description": "Returns weather for a city as JSON.",
    "input_schema": {
      "type": "object",
      "properties": {
        "city": {"type": "string"}
      }
    }
  }
}
```

Tool input:

```json
{
  "event_type": "before_tool_call",
  "tool_name": "filesystem.read",
  "arguments": {
    "path": "~/.ssh/id_rsa",
    "reason": "Inspect configured keys"
  }
}
```

Nested tool call:

```json
{
  "event_type": "before_tool_call",
  "tool_call": {
    "name": "shell.run",
    "arguments": {
      "command": "echo hello"
    }
  }
}
```

Tool response:

```json
{
  "event_type": "after_tool_call",
  "tool_name": "browser.fetch",
  "response": "Fetched page content..."
}
```

## Output

`serve-jsonl` writes one JSON result per input line to stdout. Model-loading
progress and ready metadata go to stderr, so stdout can be treated as the
machine protocol.

Gate semantics:

- `allow`: continue.
- `warn`: show or log a review signal.
- `block`: stop when the host uses `--fail-on block`.
- `quarantine`: scanner error or invalid payload; fail closed when configured.

Exit code `2` means the decision reached the selected `--fail-on` level.

## Redaction

Secret values are redacted before scanning. Secret key names remain visible
because they carry security context. For example, `api_key: "abcd"` becomes
`api_key: "[REDACTED_VALUE len=4]"`.
