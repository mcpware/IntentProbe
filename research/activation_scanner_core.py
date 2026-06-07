#!/usr/bin/env python3
"""Compatibility wrapper for the product scanner runtime.

The canonical runtime now lives in :mod:`intentprobe.scanner.core`. This module
keeps old research reproduction commands working.
"""

from __future__ import annotations

from intentprobe.scanner.core import *  # noqa: F401,F403
from intentprobe.scanner.core import main


if __name__ == "__main__":
    raise SystemExit(main())
