"""Keep the approved architecture image intact and in the preprint build."""

from __future__ import annotations

import hashlib
import importlib.util
from pathlib import Path
import struct

import pytest


ROOT = Path(__file__).resolve().parents[1]
IMAGE = ROOT / "docs" / "intentprobe-architecture-overview.png"
EXPECTED_SHA256 = "2e248922bdd4fb484cd9fbaf089c7cef96b69fa5618bd857387dde9494973d08"


def test_approved_figure_is_preserved_with_its_provenance() -> None:
    """Do not silently change the pixels Nicole approved."""
    content = IMAGE.read_bytes()
    assert content[:8] == b"\x89PNG\r\n\x1a\n"
    assert struct.unpack(">II", content[16:24]) == (1536, 1024)
    assert hashlib.sha256(content).hexdigest() == EXPECTED_SHA256
    note = (
        ROOT / "research/multitask-activation-benchmark/preprint/FIGURE_PROVENANCE.md"
    ).read_text(encoding="utf-8")
    assert EXPECTED_SHA256 in note
    assert "ARCHITECTURE_FIGURE_PROMPT.txt" in note


def test_article_and_readme_use_the_approved_figure() -> None:
    """The public entry points must show the image selected for this release."""
    for source in (
        ROOT / "README.md",
        ROOT / "docs/ONE_LOCAL_MODEL_PASS_MANY_TASK_SPECIFIC_PROBES.md",
        ROOT / "docs/index.md",
    ):
        assert IMAGE.name in source.read_text(encoding="utf-8"), source


def test_preprint_build_resolves_the_approved_image_locally(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Build from local assets, without retrieving a stale main-branch image."""
    pytest.importorskip("markdown")
    pytest.importorskip("weasyprint")
    script = ROOT / "scripts/build_architecture_preprint.py"
    spec = importlib.util.spec_from_file_location("architecture_preprint", script)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    rendered = {}

    class CaptureHTML:
        """Capture the render input without fetching anything."""

        def __init__(self, *, string: str, base_url: str) -> None:
            rendered["html"] = string
            rendered["base_url"] = base_url

        def write_pdf(self, output: Path) -> None:
            rendered["output"] = output

    monkeypatch.setattr(module, "HTML", CaptureHTML)
    output = tmp_path / "review.pdf"
    module.build(output, None)
    html = rendered["html"]
    assert IMAGE.name in html
    assert "https://raw.githubusercontent.com/" not in html
    assert rendered["base_url"] == str(ROOT / "docs")
    assert rendered["output"] == output
    assert "Figure 1" in html
