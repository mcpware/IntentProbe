#!/usr/bin/env python3
"""Audit LUFY's historical triple-annotated XLSX release.

The input directory must be the historical ``Dataset`` directory from commit
e0695216d52a38ef39ec2f91708d324877e328bd.  This script never downloads data;
it reads an explicitly supplied checkout and emits aggregate statistics only.
"""

from __future__ import annotations

import argparse
import json
import re
import unicodedata
from collections import Counter
from pathlib import Path
from typing import Iterable

import pandas as pd


def normalize_text(value: object) -> str:
    text = unicodedata.normalize("NFKC", str(value))
    return re.sub(r"\s+", " ", text).strip()


def fleiss_kappa(rows: list[list[int]]) -> float:
    if not rows:
        raise ValueError("Cannot calculate Fleiss kappa with no rows")
    raters = len(rows[0])
    if raters < 2 or any(len(row) != raters for row in rows):
        raise ValueError("Every row must have the same number of at least two raters")

    category_totals: Counter[int] = Counter()
    item_agreements: list[float] = []
    for row in rows:
        counts = Counter(row)
        category_totals.update(counts)
        item_agreements.append(
            (sum(count * count for count in counts.values()) - raters)
            / (raters * (raters - 1))
        )

    observed = sum(item_agreements) / len(item_agreements)
    proportions = [
        category_totals[label] / (len(rows) * raters) for label in (0, 1)
    ]
    expected = sum(value * value for value in proportions)
    return (observed - expected) / (1 - expected)


def user_rows(workbook: Path, user_name: str) -> list[tuple[str, int]]:
    frame = pd.read_excel(workbook, sheet_name="Conversation")
    # One historical workbook names the first column ``Unnamed: 0``.  The
    # schema position is stable even when that header is not.
    label_column = frame.columns[0]
    selected = frame[
        (frame["speaker"].astype(str) == user_name)
        & frame[label_column].isin([0, 1])
    ]
    return [
        (normalize_text(text), int(label))
        for text, label in zip(selected["utterance"], selected[label_column])
    ]


def audit_system(system_dir: Path) -> tuple[dict[str, object], list[list[int]], list[str]]:
    votes: list[list[int]] = []
    canonical_texts: list[str] = []
    users: set[str] = set()
    session_count = 0

    for user_dir in sorted(path for path in system_dir.iterdir() if path.is_dir()):
        users.add(user_dir.name)
        for session_dir in sorted(path for path in user_dir.iterdir() if path.is_dir()):
            session_count += 1
            annotations = [
                user_rows(workbook, user_dir.name)
                for workbook in sorted(session_dir.glob("*.xlsx"))
            ]
            if len(annotations) != 3:
                raise ValueError(f"Expected three annotators in {session_dir}")
            text_sequences = [[text for text, _ in rows] for rows in annotations]
            if not all(sequence == text_sequences[0] for sequence in text_sequences[1:]):
                raise ValueError(f"Annotator utterances do not align in {session_dir}")

            canonical_texts.extend(text_sequences[0])
            votes.extend(
                [[item[1] for item in annotation_items] for annotation_items in zip(*annotations)]
            )

    majority = [int(sum(row) >= 2) for row in votes]
    unanimous = sum(len(set(row)) == 1 for row in votes)
    result: dict[str, object] = {
        "system": system_dir.name,
        "workbooks": len(list(system_dir.rglob("*.xlsx"))),
        "users": len(users),
        "sessions": session_count,
        "human_user_turns": len(votes),
        "majority_skip": majority.count(0),
        "majority_save": majority.count(1),
        "unanimous": unanimous,
        "two_to_one_split": len(votes) - unanimous,
        "positive_prevalence": majority.count(1) / len(majority),
        "fleiss_kappa": fleiss_kappa(votes),
    }
    return result, votes, canonical_texts


def multiset_overlap(left: Iterable[str], right: Iterable[str]) -> int:
    return sum((Counter(left) & Counter(right)).values())


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset_dir", type=Path)
    parser.add_argument("--current-turns", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    if not args.dataset_dir.is_dir():
        raise SystemExit(f"Historical Dataset directory not found: {args.dataset_dir}")

    systems: list[dict[str, object]] = []
    all_votes: list[list[int]] = []
    historical_lufy_texts: list[str] = []
    for system_dir in sorted(path for path in args.dataset_dir.iterdir() if path.is_dir()):
        result, votes, texts = audit_system(system_dir)
        systems.append(result)
        all_votes.extend(votes)
        if system_dir.name == "LUFY":
            historical_lufy_texts = texts

    majority = [int(sum(row) >= 2) for row in all_votes]
    unanimous = sum(len(set(row)) == 1 for row in all_votes)
    report: dict[str, object] = {
        "historical_commit": "e0695216d52a38ef39ec2f91708d324877e328bd",
        "systems": systems,
        "total": {
            "workbooks": len(list(args.dataset_dir.rglob("*.xlsx"))),
            "human_user_turns": len(all_votes),
            "majority_skip": majority.count(0),
            "majority_save": majority.count(1),
            "unanimous": unanimous,
            "two_to_one_split": len(all_votes) - unanimous,
            "positive_prevalence": majority.count(1) / len(majority),
            "fleiss_kappa": fleiss_kappa(all_votes),
        },
    }

    if args.current_turns:
        current_user_texts: list[str] = []
        with args.current_turns.open(encoding="utf-8") as handle:
            for line in handle:
                item = json.loads(line)
                if item.get("role") == "user":
                    current_user_texts.append(normalize_text(item["content"]))
        overlap = multiset_overlap(historical_lufy_texts, current_user_texts)
        report["current_release_comparison"] = {
            "historical_lufy_user_turns": len(historical_lufy_texts),
            "current_lufy_user_turns": len(current_user_texts),
            "exact_normalized_multiset_overlap": overlap,
            "historical_overlap_rate": overlap / len(historical_lufy_texts),
            "current_overlap_rate": overlap / len(current_user_texts),
        }

    rendered = json.dumps(report, indent=2, ensure_ascii=False) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered, encoding="utf-8")
    print(rendered, end="")


if __name__ == "__main__":
    main()
