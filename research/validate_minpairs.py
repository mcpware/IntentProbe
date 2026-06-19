#!/usr/bin/env python3
"""Build-time gate for the matched-vocabulary minimal-pair set (minpairs-v1).

Every pair must pass ALL gates or the build fails (exit 1). This is the
methodology guard the REDO-SPEC promises: it makes "matched vocabulary"
a checkable property, not a launch-copy claim.

Gates (per pair):
  1. schema      — required fields present, non-empty
  2. jaccard     — token-set Jaccard(clean, poison) >= JACCARD_MIN
  3. length      — |len(clean) - len(poison)| <= LEN_TOL chars
  4. framing     — no "tell" word appears on only one side of a pair
                   (kills the reassurance/additive confound: clean "does not"
                    / poison "also" that a word-counter scored 88.5% on)
Corpus gate:
  5. no-leak     — no clean/poison text appears in the existing training pool
  6. duplicates  — no exact-duplicate text across the whole set

Usage:
  python3 validate_minpairs.py research/datasets/minpairs_v1_seed.json [--jaccard 0.75]
Exit 0 = all pairs pass; exit 1 = at least one violation (prints which).
"""
from __future__ import annotations
import argparse, glob, json, os, re
from collections import Counter

JACCARD_MIN = 0.75
LEN_TOL = 15

# words that signal "safe-sounding" (clean-leaning) or "capability-expanding"
# (poison-leaning). If one appears on only ONE side of a pair, the probe could
# cheat on framing instead of intent. Forbid asymmetric tells.
REASSURANCE = ["does not", "doesn't", "do not", "read-only", "read only",
               "without", "never", "purely", "only parses", "just parses"]
ADDITIVE = ["also", "additionally", "automatically", "in addition",
            "as well", "furthermore", "moreover", "as a side effect"]
TELLS = REASSURANCE + ADDITIVE


def toks(s: str) -> set[str]:
    return set(re.findall(r"[a-z0-9]+", s.lower()))


def jaccard(a: str, b: str) -> float:
    A, B = toks(a), toks(b)
    return len(A & B) / len(A | B) if (A | B) else 0.0


def norm(s: str) -> str:
    return re.sub(r"\s+", " ", s.strip().lower())


def load_training_texts(datasets_dir: str, exclude_basenames: set[str]) -> set[str]:
    """Best-effort: collect every description-like string from existing dataset
    JSONs (so we can assert minpairs are NOT already in the training pool)."""
    texts: set[str] = set()
    for path in glob.glob(os.path.join(datasets_dir, "*.json")):
        base = os.path.basename(path)
        # the minpairs set IS the held-out eval set, not training data — never
        # treat its own sibling files as "training pool" (would false-positive).
        if base in exclude_basenames or base.startswith("minpairs_v1"):
            continue
        try:
            data = json.load(open(path))
        except Exception:
            continue
        stack = [data]
        while stack:
            x = stack.pop()
            if isinstance(x, str):
                if len(x) > 20:
                    texts.add(norm(x))
            elif isinstance(x, dict):
                for k in ("text", "description", "content", "tool_description", "skill"):
                    if isinstance(x.get(k), str):
                        texts.add(norm(x[k]))
                stack.extend(x.values())
            elif isinstance(x, list):
                stack.extend(x)
    return texts


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("path")
    ap.add_argument("--jaccard", type=float, default=JACCARD_MIN)
    ap.add_argument("--len-tol", type=int, default=LEN_TOL)
    args = ap.parse_args()

    pairs = json.load(open(args.path))
    datasets_dir = os.path.dirname(os.path.abspath(args.path))
    train_texts = load_training_texts(datasets_dir, {os.path.basename(args.path)})

    seen_text: dict[str, str] = {}
    fams = Counter()
    jac_vals = []
    violations = []
    rows = []

    for i, p in enumerate(pairs):
        pid = p.get("pair_id", f"#{i}")
        fams[p.get("family", "?")] += 1
        # gate 1 schema
        for f in ("pair_id", "family", "clean", "poison"):
            if not p.get(f):
                violations.append(f"[{pid}] schema: missing/empty '{f}'")
        clean, poison = p.get("clean", ""), p.get("poison", "")
        if not (clean and poison):
            continue
        # gate 2 jaccard
        j = jaccard(clean, poison)
        jac_vals.append(j)
        if j < args.jaccard:
            violations.append(f"[{pid}] jaccard {j:.3f} < {args.jaccard}")
        # gate 3 length
        dl = abs(len(clean) - len(poison))
        if dl > args.len_tol:
            violations.append(f"[{pid}] length diff {dl} > {args.len_tol} chars")
        # gate 4 framing tells (asymmetric)
        lc, lp = clean.lower(), poison.lower()
        for t in TELLS:
            if (t in lc) != (t in lp):
                side = "clean" if t in lc else "poison"
                violations.append(f"[{pid}] framing tell '{t}' only in {side}")
        # gate 5 no-leak
        for side, txt in (("clean", clean), ("poison", poison)):
            if norm(txt) in train_texts:
                violations.append(f"[{pid}] no-leak: {side} text is in the training pool")
        # gate 6 duplicates
        for side, txt in (("clean", clean), ("poison", poison)):
            n = norm(txt)
            if n in seen_text:
                violations.append(f"[{pid}] duplicate {side} text (also in {seen_text[n]})")
            seen_text[n] = pid
        rows.append((pid, p.get("family", "?"), j, dl))

    # report
    print(f"== minpairs validation: {args.path} ==")
    print(f"pairs={len(pairs)} families={dict(fams)}")
    if jac_vals:
        jac_vals.sort()
        mean = sum(jac_vals) / len(jac_vals)
        print(f"jaccard: mean={mean:.3f} min={jac_vals[0]:.3f} "
              f"median={jac_vals[len(jac_vals)//2]:.3f} max={jac_vals[-1]:.3f} "
              f"(gate >= {args.jaccard})")
        n_pass_j = sum(1 for v in jac_vals if v >= args.jaccard)
        print(f"jaccard gate: {n_pass_j}/{len(jac_vals)} pairs pass")
    print("\nper-pair [pair_id | family | jaccard | len_diff]:")
    for pid, fam, j, dl in rows:
        mark = "ok " if j >= args.jaccard and dl <= args.len_tol else "XX "
        print(f"  {mark}{pid:28s} {fam:22s} J={j:.3f} dlen={dl}")

    if violations:
        print(f"\nFAIL — {len(violations)} violation(s):")
        for v in violations:
            print(f"  - {v}")
        return 1
    print(f"\nPASS — all {len(pairs)} pairs clear every gate.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
