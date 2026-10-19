# Empirical preprint build — v0.1.3

Canonical source: `DRAFT.md`. Author: Wan Sheng Leung. Reserved DOI:
`10.5281/zenodo.23471117`. Publication remains a separately approved action.

The PDF's relative evidence links are pinned to source commit
`25bde58b0702d3853f66795ce6b5cf14ca795287`, which includes the reconciled manuscript, source ledger,
derived results, training/extraction code and original frozen experiment artifacts.
The final PDF and release checksum ledger are committed separately after rendering.

From the repository root, using the installed Markdown 3.10.2 and WeasyPrint 69.0:

`python3 scripts/build_multitask_preprint.py --output research/multitask-activation-benchmark/paper/intentprobe-empirical-v0.1.3.pdf --evidence-ref 25bde58b0702d3853f66795ce6b5cf14ca795287`

No model or experiment is rerun to build the PDF. The renderer reads the approved local
illustration and rejects absent relative targets, out-of-repository paths, mutable evidence
refs, remote images and non-public anchor schemes. The generated PDF contains public links,
not local filesystem paths. Font/rendering environment changes may alter PDF bytes; the
published release hash identifies the exact candidate bytes independently of layout reproduction.

The architecture companion has DOI `10.5281/zenodo.23467859`. Its own build command is in
`../preprint/README.md`. Both PDFs use the unchanged approved figure recorded in
`../preprint/FIGURE_PROVENANCE.md`.
