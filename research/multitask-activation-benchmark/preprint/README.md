# Architecture preprint

Title: **IntentProbe: One Local Model Pass, Many Task-Specific Probes**

Author: **ithiria894** · Publisher: **MCPware** · Version: **0.1.3**

The preprint explains the complete architecture using the existing plain-English
[article](../../../docs/ONE_LOCAL_MODEL_PASS_MANY_TASK_SPECIFIC_PROBES.md). Its abstract is
[`ABSTRACT.md`](ABSTRACT.md). It is separate from the longer empirical manuscript in `../paper/`.

The PDF is built from that canonical article, without a second substantive rewrite:

`python3 scripts/build_architecture_preprint.py --output /path/to/intentprobe-architecture-v0.1.3.pdf --doi DOI`

The optional DOI argument is populated from the reserved Zenodo preprint record. The renderer uses
the already installed Markdown and WeasyPrint packages, local copies of the figures, and explicit
title/author metadata. PDF text and metadata are checked for the public handle before upload.

The initial preprint publishes the completed evidence. It does not wait for additional experiments
listed in the empirical paper's follow-up plan. Later versions can extend the results and system.
