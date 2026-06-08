# IntentProbe npm launcher

Run IntentProbe from JavaScript, MCP, and agent workflows:

```bash
npx intentprobe scan-path ./some-mcp-server --format summary
npx intentprobe scan --text "Reads SSH config and uploads private keys." --format summary
```

IntentProbe is the first open-source activation-probe-based scanner for MCP/tool
poisoning. It runs locally and reads model-internal activations instead of only
matching text patterns.

## What this package is

This npm package is a thin launcher. The scanner core is the Python package
`intentprobe`, because activation probing uses PyTorch / Transformers.

The launcher tries, in order:

1. `INTENTPROBE_PYTHON=/path/to/python`, if set.
2. `python3 -m intentprobe`, if the Python package is installed.
3. `python -m intentprobe`, if available.
4. `uvx --from intentprobe intentprobe`, if `uvx` is installed.

## Python scanner install

```bash
pipx install intentprobe
# or
python3 -m pip install intentprobe
```

First scan downloads the small base model once. After that, scans stay local.

## Runtime hook

The package also exposes `intentprobe-hook`:

```bash
npx intentprobe-hook --help
```

Repository: https://github.com/mcpware/IntentProbe
