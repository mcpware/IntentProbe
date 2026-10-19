#!/usr/bin/env python3
"""Render the canonical architecture article as an attributed preprint PDF."""

from __future__ import annotations

import argparse
import html
from pathlib import Path

import markdown
from weasyprint import HTML


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "docs" / "ONE_LOCAL_MODEL_PASS_MANY_TASK_SPECIFIC_PROBES.md"
TITLE = "IntentProbe: One Local Model Pass, Many Task-Specific Probes"
AUTHOR = "Wan Sheng Leung"
# Figures are referenced by public raw URLs so the article renders on GitHub and Pages.
# The architecture illustration is pinned to its immutable asset commit; the latency figure
# tracks main. Both resolve to local bytes when the PDF is rendered.
ARCHITECTURE_ASSET_COMMIT = "ade288c568161beba2906c2dada3def2422757bf"
REMOTE_DOC_PREFIXES = (
    f"https://raw.githubusercontent.com/mcpware/IntentProbe/{ARCHITECTURE_ASSET_COMMIT}/docs/",
    "https://raw.githubusercontent.com/mcpware/IntentProbe/main/docs/",
)

CSS = """
@page {
  size: A4;
  margin: 18mm 18mm 19mm;
  @bottom-left { content: 'IntentProbe · architecture preprint v0.1.3'; font-size: 8pt; color: #66717e; }
  @bottom-right { content: counter(page) ' / ' counter(pages); font-size: 8pt; color: #66717e; }
}
body { font-family: 'DejaVu Sans', sans-serif; font-size: 10pt; line-height: 1.5; color: #203040; }
h1 { font-size: 23pt; line-height: 1.15; color: #17324d; margin: 0 0 14pt; }
h2 { font-size: 14pt; color: #17324d; margin: 17pt 0 7pt; break-after: avoid; }
p { margin: 0 0 9pt; orphans: 3; widows: 3; }
a { color: #285e9b; text-decoration: none; overflow-wrap: anywhere; }
img { display: block; max-width: 100%; height: auto; margin: 10pt auto; }
figure { margin: 11pt 0 13pt; break-inside: avoid; page-break-inside: avoid; }
figure img { margin: 0 auto 6pt; max-height: 112mm; }
figcaption { font-size: 8.4pt; line-height: 1.45; color: #3f4c5a; text-align: left; }
table { border-collapse: collapse; width: 100%; font-size: 8.5pt; margin: 9pt 0 12pt; }
thead { display: table-header-group; }
th, td { border: 0.5pt solid #d7e0e8; padding: 6pt; vertical-align: top; overflow-wrap: anywhere; }
th { background: #eaf2f8; text-align: left; }
tr { break-inside: avoid; }
li { margin-bottom: 3pt; }
code { font-family: 'DejaVu Sans Mono', monospace; font-size: 8.5pt; }
pre { white-space: pre-wrap; overflow-wrap: anywhere; }
.edition { font-size: 9pt; color: #66717e; margin: 0 0 8pt; }
"""


def build(output: Path, doi: str | None) -> None:
    """Render source text and local figures without fetching remote image bytes."""
    source = SOURCE.read_text(encoding="utf-8")
    assert source.startswith("---\n")
    text = source.split("---", 2)[2].lstrip()
    local_docs = (ROOT / "docs").as_uri() + "/"
    for prefix in REMOTE_DOC_PREFIXES:
        text = text.replace(prefix, local_docs)
    body = markdown.markdown(text, extensions=["tables", "fenced_code"])
    citation = (
        f'<p class="edition">DOI: <a href="https://doi.org/{html.escape(doi)}">'
        f"{html.escape(doi)}</a></p>"
        if doi
        else ""
    )
    document = (
        '<!doctype html><html lang="en"><head><meta charset="utf-8">'
        f"<title>{html.escape(TITLE)}</title>"
        f'<meta name="author" content="{AUTHOR}">'
        f"<style>{CSS}</style></head><body>"
        '<p class="edition">Architecture preprint · 19 October 2026 · v0.1.3</p>'
        f"{citation}{body}</body></html>"
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    HTML(string=document, base_url=str(ROOT / "docs")).write_pdf(output)


def main() -> None:
    """Parse one output path and optional reserved DOI."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--doi")
    args = parser.parse_args()
    build(args.output.resolve(), args.doi)


if __name__ == "__main__":
    main()
