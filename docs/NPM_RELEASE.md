# npm release notes

IntentProbe's npm package is a thin launcher for JavaScript, MCP, and agent
users who expect `npx intentprobe ...`.

The scanner core stays in Python. Do not move Torch, Transformers, or probe
artifacts into npm.

Publish the Python package first, then publish npm. The npm launcher defaults
to `uvx --from intentprobe intentprobe ...` when a local Python install is not
available, so the PyPI package must exist before `npx intentprobe ...` is a
good public first-run experience.

## Package

- npm package name: `intentprobe`
- package directory: `npm/`
- binaries: `intentprobe`, `intentprobe-hook`
- current version: `0.1.0`

## Publish checklist

From the repository root:

```bash
npm view intentprobe name version dist-tags --json
npm whoami
npm --prefix npm run check
INTENTPROBE_PYTHON=.venv/bin/python node npm/bin/intentprobe.mjs --help
npm --prefix npm run pack:dry
npm publish ./npm
```

After publish, verify:

```bash
npm view intentprobe name version dist-tags --json
npx intentprobe --help
```

Only after the package is live should the root README advertise `npx
intentprobe ...` as the primary install path.

## Auth blocker seen locally

`npm whoami` returned `ENEEDAUTH`, so publishing needs an npm login or npm
automation token before this can go live.
