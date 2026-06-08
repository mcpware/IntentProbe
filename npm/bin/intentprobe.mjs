#!/usr/bin/env node
import { spawn, spawnSync } from "node:child_process";
import { basename } from "node:path";

const PACKAGE_NAME = "intentprobe";
const invokedAs = basename(process.argv[1] || "intentprobe");
const entrypoint = invokedAs === "intentprobe-hook" ? "intentprobe-hook" : "intentprobe";
const pythonModule = entrypoint === "intentprobe-hook" ? "intentprobe.hook" : "intentprobe";
const args = process.argv.slice(2);

function canRun(command, commandArgs) {
  const result = spawnSync(command, commandArgs, { stdio: "ignore" });
  return !result.error && result.status === 0;
}

function pythonHasIntentProbe(python) {
  return canRun(python, ["-c", "import intentprobe"]);
}

function findPythonWithIntentProbe() {
  const candidates = [
    process.env.INTENTPROBE_PYTHON,
    "python3",
    "python"
  ].filter(Boolean);

  for (const python of candidates) {
    if (pythonHasIntentProbe(python)) {
      return python;
    }
  }

  return null;
}

function run(command, commandArgs) {
  const child = spawn(command, commandArgs, {
    stdio: "inherit",
    env: process.env
  });

  child.on("error", (error) => {
    console.error(`intentprobe: failed to start ${command}: ${error.message}`);
    process.exit(127);
  });

  child.on("exit", (code, signal) => {
    if (signal) {
      process.kill(process.pid, signal);
      return;
    }
    process.exit(code ?? 1);
  });
}

const python = findPythonWithIntentProbe();
if (python) {
  run(python, ["-m", pythonModule, ...args]);
} else if (canRun("uvx", ["--version"])) {
  const source = process.env.INTENTPROBE_UVX_SOURCE || PACKAGE_NAME;
  run("uvx", ["--from", source, entrypoint, ...args]);
} else {
  console.error(`IntentProbe's npm package is a thin launcher for the Python scanner.

Could not find a Python environment with the "intentprobe" module, and "uvx"
is not available for one-shot execution.

Install the Python scanner first:

  pipx install intentprobe
  # or
  python3 -m pip install intentprobe

Then run:

  npx intentprobe scan-path ./some-mcp-server --format summary

For a custom environment:

  INTENTPROBE_PYTHON=/path/to/python npx intentprobe --help
`);
  process.exit(127);
}
