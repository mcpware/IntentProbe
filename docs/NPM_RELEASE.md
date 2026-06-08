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

## Local gate status seen on 2026-06-08

Passed:

- `INTENTPROBE_PYTHON=.venv/bin/python node npm/bin/intentprobe.mjs --help`
- `npm --prefix npm run check`
- `npm --prefix npm run pack:dry`
- `npm publish ./npm --dry-run`

Observed but time-bound:

- `npm view intentprobe name version dist-tags --json` returned `E404` on
  2026-06-08. Treat this as a local check, not a durable public claim about name
  availability.

## Auth blocker seen locally

`npm whoami` returned `ENEEDAUTH`, so publishing needs an npm login or npm
automation token before this can go live.

The dry-run publish can still show the package tarball and publish target, but
the real publish requires npm auth.

Update from 2026-06-08:

- PyPI `intentprobe==0.1.1` is live and passed a fresh install smoke.
- The npm token from the credential vault decrypted successfully, but
  `npm whoami` returned `E401 Unauthorized`.
- Do not retry npm publish with the same token. Create or refresh an npm
  automation token, then rerun `npm whoami` and `npm publish ./npm --access
  public`.
