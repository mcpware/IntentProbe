# Sample Reporting Guide

intentprobe improves by seeing real scanner misses and real false positives.
Please report samples, but keep them safe to share publicly.

## What to report

Report a missed detection when a tool, MCP server, skill, or package description
is trying to do something hidden or unsafe but intentprobe returns `allow`.

Report a false positive when a normal tool gets `warn` or `block` and the risky
decision looks unreasonable.

Good reports include:

- the exact command you ran.
- the intentprobe version or git commit.
- the decision JSON or summary output.
- the smallest redacted text that reproduces the decision.
- why you think the expected decision should be `allow`, `warn`, or `block`.

## Redaction rules

Before opening an issue, replace sensitive values:

| Sensitive value | Replace with |
|---|---|
| API keys, tokens, passwords | `REDACTED_SECRET` |
| Private keys | `REDACTED_PRIVATE_KEY` |
| Private URLs or hosts | `REDACTED_PRIVATE_URL` |
| Customer, patient, employee, or user names | `REDACTED_PERSON` |
| Internal project names | `REDACTED_PROJECT` |

Keep the surrounding wording if it is needed to reproduce the scanner behavior.
The probe is trying to learn intent, so the shape of the instruction matters
more than the real secret value.

## Minimal reproduction

For one text sample:

```bash
intentprobe scan --pretty --text "REDACTED SAMPLE TEXT"
```

For a package, MCP config, or skill folder:

```bash
intentprobe scan-path ./path-to-redacted-sample --pretty
```

For a CI-style block check:

```bash
intentprobe scan-path ./path-to-redacted-sample --format summary --fail-on block
```

## What not to report publicly

Do not post live credentials, private source code, private customer data, or
working exploit chains against third-party systems. If the sample needs private
details to be useful, reduce it to a synthetic reproduction first.
