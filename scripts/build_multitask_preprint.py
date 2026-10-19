#!/usr/bin/env python3
"""Render the empirical manuscript with public evidence links and local figures."""

from __future__ import annotations

import argparse
import html
from html.parser import HTMLParser
from pathlib import Path
import re
from urllib.parse import quote, urlsplit, urlunsplit

import markdown
from weasyprint import HTML

from build_architecture_preprint import AUTHOR, CSS, ROOT


SOURCE = ROOT / "research/multitask-activation-benchmark/paper/DRAFT.md"
TITLE = "IntentProbe: Many Task-Specific Probes from a Single Frozen Small-Model Forward Pass"
DOI = "10.5281/zenodo.23471117"
EMPIRICAL_CSS = CSS.replace(
    "architecture preprint v0.1.3", "empirical preprint v0.1.3"
) + """
h1 { font-size: 22pt; }
table { font-size: 7.7pt; break-inside: avoid; }
th, td { padding: 4.5pt; overflow-wrap: normal; }
figure img { max-height: 108mm; }
"""


def resolve_target(target: str, *, evidence_ref: str, image: bool = False) -> str:
    """Keep evidence links public and load images from the checked-out repository."""
    parsed = urlsplit(target)
    if parsed.scheme or parsed.netloc:
        if image or parsed.scheme not in {"https", "http"} or not parsed.netloc:
            raise ValueError("Links must be public HTTP(S); figures must be checked-in local paths")
        return target
    if not parsed.path:
        if parsed.fragment and not parsed.query:
            return target
        raise ValueError("An evidence link must identify a public resource or document fragment")
    local = (SOURCE.parent / parsed.path).resolve()
    relative = local.relative_to(ROOT)
    if not local.exists():
        raise FileNotFoundError(local)
    if image:
        if not local.is_file():
            raise FileNotFoundError(local)
        return local.as_uri()
    if not re.fullmatch(r"[0-9a-f]{40}", evidence_ref):
        raise ValueError("Evidence links require an immutable full Git commit hash")
    path = f"/mcpware/IntentProbe/blob/{evidence_ref}/{quote(relative.as_posix())}"
    return urlunsplit(("https", "github.com", path, parsed.query, parsed.fragment))


class PublicEvidenceHTML(HTMLParser):
    """Rewrite Markdown-generated links without leaking local file paths into PDF links."""

    def __init__(self, evidence_ref: str) -> None:
        super().__init__(convert_charrefs=False)
        if not re.fullmatch(r"[0-9a-f]{40}", evidence_ref):
            raise ValueError("Evidence links require an immutable full Git commit hash")
        self.evidence_ref = evidence_ref
        self.parts: list[str] = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if tag not in {"a", "img"}:
            self.parts.append(self.get_starttag_text())
            return
        rewritten = []
        for key, value in attrs:
            if value is not None and (tag, key) in {("a", "href"), ("img", "src")}:
                value = resolve_target(value, evidence_ref=self.evidence_ref, image=tag == "img")
            rewritten.append(
                key if value is None else f'{key}="{html.escape(value, quote=True)}"'
            )
        self.parts.append(f"<{tag} {' '.join(rewritten)}>")

    def handle_startendtag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        self.handle_starttag(tag, attrs)

    def handle_endtag(self, tag: str) -> None:
        self.parts.append(f"</{tag}>")

    def handle_data(self, data: str) -> None:
        self.parts.append(data)

    def handle_entityref(self, name: str) -> None:
        self.parts.append(f"&{name};")

    def handle_charref(self, name: str) -> None:
        self.parts.append(f"&#{name};")

    def handle_comment(self, data: str) -> None:
        self.parts.append(f"<!--{data}-->")


def build(output: Path, evidence_ref: str) -> None:
    """Render the canonical manuscript; do not rewrite or regenerate its evidence."""
    body = markdown.markdown(
        SOURCE.read_text(encoding="utf-8"), extensions=["tables", "fenced_code"]
    )
    links = PublicEvidenceHTML(evidence_ref)
    links.feed(body)
    links.close()
    document = (
        '<!doctype html><html lang="en"><head><meta charset="utf-8">'
        f"<title>{html.escape(TITLE)}</title>"
        f'<meta name="author" content="{AUTHOR}">'
        f"<style>{EMPIRICAL_CSS}</style></head><body>"
        '<p class="edition">Empirical research preprint · 19 October 2026 · v0.1.3</p>'
        + "".join(links.parts)
        + "</body></html>"
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    HTML(string=document, base_url=str(SOURCE.parent)).write_pdf(output)


def main() -> None:
    """Parse the output path and render with installed dependencies."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--evidence-ref", required=True, help="Full immutable source commit hash")
    args = parser.parse_args()
    build(args.output.resolve(), args.evidence_ref)


if __name__ == "__main__":
    main()
