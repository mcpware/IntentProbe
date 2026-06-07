"""Hook CLI entrypoint for MCP/tool/skill scanner gates."""

from __future__ import annotations

from research.activation_scanner_hook import main


if __name__ == "__main__":
    raise SystemExit(main())
