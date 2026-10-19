"""Keep the two papers separately citable and their PDF links publicly usable."""

from __future__ import annotations

import importlib.util
from pathlib import Path
import pytest


ROOT = Path(__file__).resolve().parents[1]
BENCH = ROOT / "research/multitask-activation-benchmark"
ARCHITECTURE_DOI = "10.5281/zenodo.23467859"
EMPIRICAL_DOI = "10.5281/zenodo.23471117"
EVIDENCE_REF = "44d1b6934f9e7a4cce153bb05b80e276d36ad180"


@pytest.fixture()
def renderer(monkeypatch: pytest.MonkeyPatch):
    """Import the standalone renderer with its sibling source module available."""
    pytest.importorskip("markdown")
    pytest.importorskip("weasyprint")
    monkeypatch.syspath_prepend(str(ROOT / "scripts"))
    spec = importlib.util.spec_from_file_location(
        "multitask_preprint", ROOT / "scripts/build_multitask_preprint.py"
    )
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_papers_cross_reference_distinct_dois() -> None:
    architecture = (ROOT / "docs/ONE_LOCAL_MODEL_PASS_MANY_TASK_SPECIFIC_PROBES.md").read_text()
    empirical = (BENCH / "paper/DRAFT.md").read_text()
    assert ARCHITECTURE_DOI != EMPIRICAL_DOI
    assert f"https://doi.org/{EMPIRICAL_DOI}" in architecture
    assert f"https://doi.org/{ARCHITECTURE_DOI}" in empirical
    assert f"**DOI:** [{EMPIRICAL_DOI}]" in empirical
    assert "## References" in empirical
    assert EMPIRICAL_DOI in (BENCH / "paper/intentprobe-empirical.bib").read_text()


def test_relative_pdf_links_resolve_to_public_evidence(renderer) -> None:
    link = renderer.resolve_target(
        "../experiments/ONE-FORWARD-MULTIHEAD-EXP001/result.json", evidence_ref=EVIDENCE_REF
    )
    assert link.startswith(f"https://github.com/mcpware/IntentProbe/blob/{EVIDENCE_REF}/")
    assert link.endswith("ONE-FORWARD-MULTIHEAD-EXP001/result.json")
    assert renderer.resolve_target("#references", evidence_ref=EVIDENCE_REF) == "#references"
    assert renderer.resolve_target(
        f"https://doi.org/{ARCHITECTURE_DOI}", evidence_ref=EVIDENCE_REF
    ).startswith("https:")
    with pytest.raises(ValueError):
        renderer.resolve_target("../../../../../../outside-repo.md", evidence_ref=EVIDENCE_REF)
    with pytest.raises(ValueError):
        renderer.resolve_target("https://example.com/unreviewed.png", evidence_ref=EVIDENCE_REF, image=True)
    with pytest.raises(ValueError):
        renderer.resolve_target("DERIVED_RESULTS.json", evidence_ref="review/temporary")
    for private_target in ("file:///home/private/report.md", "javascript:alert(1)", "?local=1"):
        with pytest.raises(ValueError):
            renderer.resolve_target(private_target, evidence_ref=EVIDENCE_REF)
    with pytest.raises(FileNotFoundError):
        renderer.resolve_target("missing-evidence-target.json", evidence_ref=EVIDENCE_REF)


def test_empirical_pdf_uses_the_approved_local_figure(
    renderer, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    captured = {}

    class CaptureHTML:
        def __init__(self, *, string: str, base_url: str) -> None:
            captured.update(html=string, base_url=base_url)

        def write_pdf(self, output: Path) -> None:
            captured["output"] = output

    monkeypatch.setattr(renderer, "HTML", CaptureHTML)
    output = tmp_path / "empirical.pdf"
    renderer.build(output, EVIDENCE_REF)
    text = captured["html"]
    assert "Wan Sheng Leung" in text
    assert EMPIRICAL_DOI in text and ARCHITECTURE_DOI in text
    assert (ROOT / "docs/intentprobe-architecture-overview.png").as_uri() in text
    assert (BENCH / "paper/outputs/figures/latency-panels.svg").as_uri() in text
    assert (BENCH / "paper/outputs/figures/selected-readout-windows.svg").as_uri() in text
    assert text.count("<figure>") == 3
    assert 'href="file:' not in text
    assert 'src="https:' not in text
    assert captured["output"] == output


def test_empirical_abstract_leads_with_scaling_without_merging_panels() -> None:
    source = (BENCH / "paper/DRAFT.md").read_text()
    abstract = source.split("## Abstract", 1)[1].split("## 1. Introduction", 1)[0]
    assert 180 <= len(abstract.split()) <= 250
    assert abstract.index("68.63") < abstract.index("24-cell")
    assert "80 held-out texts" in abstract and "ten-input panel" in abstract
    assert "repeated-forward lower bound" in abstract
    assert "Native heldout AUPRC" in abstract
    assert "motivate the broader project" not in abstract
    assert "motivate the broader project" in source.split("### 3.2", 1)[1]
