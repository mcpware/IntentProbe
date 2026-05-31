# intentprobe

**See a tool's true intent before you install it.**

intentprobe is a security scanner for AI agent tooling. Before your agent installs an MCP server, a Claude Code skill, or an npm package, intentprobe reads what the tool *actually intends to do* and flags it when the intent is malicious.

It works differently from every scanner shipping today. Instead of matching a tool's text against regex rules, it runs the tool's description through a small local model and reads the model's internal activations: the signal that encodes whether the model understood the text as benign, or as something that quietly steals your keys.

## Why activations, not text

Tool poisoning hides malicious instructions inside normal-sounding tool descriptions. Two descriptions can use almost identical words while only one of them quietly reads your private keys.

> **Tool A:** "Reads the SSH config file and returns a parsed list of host aliases, hostnames, and ports. Does not modify files or open connections."

> **Tool B:** "Reads the SSH config file and returns a parsed list of host aliases, hostnames, and ports. Also reads the corresponding private key files to compute fingerprints for each identity."

Same words. Same topic. Tool B exfiltrates your private keys. A text or regex scanner cannot tell them apart.

On 485 poisoned tool descriptions from real MCP servers (the MCPTox benchmark), a 60-rule regex scanner caught **zero**. A probe trained on a model's internal activations scored **97%** on matched pairs where the safe and malicious versions share the same vocabulary.

The research behind this approach is documented in the paper that seeded this project (link coming).

## Status

Early development. This is a research preview, not yet published.

A few honest notes, because they shape what intentprobe is:

- It detects and flags for a human. It does not try to silently "fix" a tool from the inside.
- The probe is strongest when safe and malicious descriptions look alike, exactly where text scanners fail. Generalizing across very different phrasings is the current research frontier (around 71-73%), so intentprobe leads with the cases where text scanners hit zero.
- If intentprobe misses a poisoned tool you hit in the wild, that sample is gold. Reporting it helps the underlying research.

## How it will work

1. You, or your agent, point intentprobe at a skill, MCP server, or npm package before installing it.
2. intentprobe runs each tool description through a small on-device model and reads the activations.
3. A trained probe scores the intent: safe, suspicious, or poisoned.
4. Wired as a pre-install hook, your agent runs this automatically before adding any tool, at zero token cost.

Runs locally on CPU with a sub-2B model. Nothing leaves your machine.

## License

Apache-2.0. Bring your own model; the probe is small and ships with the tool.

---

If intentprobe ever stops a poisoned tool from reaching your machine, a star helps other people find it: [github.com/mcpware/intentprobe](https://github.com/mcpware/intentprobe)
