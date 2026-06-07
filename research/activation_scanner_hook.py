#!/usr/bin/env python3
"""Compatibility wrapper for the product scanner hook CLI."""

from __future__ import annotations

from intentprobe.scanner.hook import *  # noqa: F401,F403
from intentprobe.scanner.hook import main


if __name__ == "__main__":
    raise SystemExit(main())
