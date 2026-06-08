# PyPI Release Notes

IntentProbe's Python package is the primary scanner package. The npm package is
a thin launcher that depends on this package being available first.

## Package

- PyPI package name: `intentprobe`
- current version: `0.1.1`
- Python: `>=3.10,<3.14`
- entry points: `intentprobe`, `intentprobe-hook`

## Local build gate

The release wheel should be built from a freshly extracted sdist, not directly
from the live repo checkout. The repo may have an ignored `build/` directory,
and a direct wheel build can accidentally reuse stale files.

The local release gate for `0.1.1` was run on 2026-06-08. This is a
README/PyPI-description sync patch after the root README moved to the primary
`python3 -m pip install intentprobe` install path.

- `python -m build --sdist` from the repo: passed.
- extract `intentprobe-0.1.1.tar.gz`, then `python -m build --wheel`: passed.
- `twine check` on sdist and wheel: passed.
- `check-wheel-contents`: passed.
- wheel contains `metadata.json`, `probe_weights.npz`, `targets.py`, and entry
  points.
- sdist contains release docs, runtime docs, operator decision docs, evidence
  packet, examples, research fixtures, and scanner artifact.
- no-deps wheel smoke import passed: package version `0.1.1`, metadata file
  present, probe weights present.

## Registry status

Published on 2026-06-08:

- PyPI release: <https://pypi.org/project/intentprobe/0.1.0/>
- PyPI release: <https://pypi.org/project/intentprobe/0.1.1/>
- `python3 -m pip index versions intentprobe` shows `0.1.1`.
- Fresh PyPI install smoke passed: `intentprobe --version` returned
  `activation-scanner-core-2026-06-03-static-v3`.
- Installed package version was `0.1.1`.
- Bundled `metadata.json` and `probe_weights.npz` were present.

## Publish command

After PyPI auth is configured:

```bash
. /tmp/intentprobe-release-paths.env
/tmp/intentprobe-release-tools/bin/python -m twine upload \
  "$sdist_dist"/intentprobe-0.1.1.tar.gz \
  "$wheel_dist"/intentprobe-0.1.1-py3-none-any.whl
```

Do not upload from old `dist/` files unless they were rebuilt during the same
release gate.

## Post-publish verification

```bash
python3 -m pip index versions intentprobe
python3 -m venv /tmp/intentprobe-pypi-smoke
/tmp/intentprobe-pypi-smoke/bin/python -m pip install --upgrade pip
/tmp/intentprobe-pypi-smoke/bin/python -m pip install intentprobe==0.1.1
/tmp/intentprobe-pypi-smoke/bin/intentprobe --help
/tmp/intentprobe-pypi-smoke/bin/intentprobe doctor --pretty
```

The npm launcher is not the public install path until npm auth is refreshed and
the npm package is published.
