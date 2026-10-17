#!/usr/bin/env python3
"""Build the row-aligned LUFY majority-vote external evaluation set."""

from __future__ import annotations

import hashlib
import json
import re
import subprocess
import unicodedata
from pathlib import Path

import pandas as pd


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
RAW_REPO = ROOT / "data" / "raw" / "github" / "lufy-historical-annotations"
RAW = RAW_REPO / "Dataset"
OUTPUT = ROOT / "data" / "processed" / "exp003" / "lufy_majority.parquet"
MANIFEST = HERE / "dataset_manifest.json"
EXPECTED_COMMIT = "e0695216d52a38ef39ec2f91708d324877e328bd"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def normalize_text(value: object) -> str:
    text = unicodedata.normalize("NFKC", str(value))
    return re.sub(r"\s+", " ", text).strip()


def read_user_rows(workbook: Path, user_name: str) -> list[tuple[str, int]]:
    frame = pd.read_excel(workbook, sheet_name="Conversation")
    label_column = frame.columns[0]
    selected = frame[
        (frame["speaker"].astype(str) == user_name)
        & frame[label_column].isin([0, 1])
    ]
    return [
        (normalize_text(text), int(label))
        for text, label in zip(selected["utterance"], selected[label_column])
    ]


def main() -> None:
    revision = subprocess.run(
        ["git", "-C", str(RAW_REPO), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    if revision != EXPECTED_COMMIT:
        raise ValueError(f"Expected LUFY commit {EXPECTED_COMMIT}, found {revision}")

    records: list[dict[str, object]] = []
    for system_dir in sorted(path for path in RAW.iterdir() if path.is_dir()):
        for user_dir in sorted(path for path in system_dir.iterdir() if path.is_dir()):
            for session_dir in sorted(path for path in user_dir.iterdir() if path.is_dir()):
                annotations = [
                    read_user_rows(workbook, user_dir.name)
                    for workbook in sorted(session_dir.glob("*.xlsx"))
                ]
                if len(annotations) != 3:
                    raise ValueError(f"Expected three annotators in {session_dir}")
                text_sequences = [[text for text, _ in rows] for rows in annotations]
                if not all(sequence == text_sequences[0] for sequence in text_sequences[1:]):
                    raise ValueError(f"Annotator utterances do not align in {session_dir}")
                session_name = session_dir.name
                for turn_index, items in enumerate(zip(*annotations)):
                    votes = [item[1] for item in items]
                    text = items[0][0]
                    records.append(
                        {
                            "row_id": (
                                f"{system_dir.name}/{user_dir.name}/{session_name}/{turn_index:04d}"
                            ),
                            "system": system_dir.name,
                            "user_name": user_dir.name,
                            "session": session_name,
                            "turn_index": turn_index,
                            "text": text,
                            "rendered": f"USER_UTTERANCE\n{text}\n\nMEMORY_WRITE_DECISION",
                            "vote_0": votes[0],
                            "vote_1": votes[1],
                            "vote_2": votes[2],
                            "label": int(sum(votes) >= 2),
                            "unanimous": len(set(votes)) == 1,
                        }
                    )

    frame = pd.DataFrame.from_records(records)
    if len(frame) != 6470 or frame["row_id"].duplicated().any():
        raise ValueError("LUFY row contract changed")
    if frame["user_name"].nunique() != 17 or int(frame["label"].sum()) != 332:
        raise ValueError("LUFY user or label counts changed")
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(OUTPUT, index=False)

    manifest = {
        "experiment_id": "MEM-EXP003",
        "status": "BUILT",
        "source_revision": revision,
        "workbooks": len(list(RAW.rglob("*.xlsx"))),
        "rows": len(frame),
        "users": int(frame["user_name"].nunique()),
        "systems": frame["system"].value_counts().sort_index().to_dict(),
        "majority_skip": int((frame["label"] == 0).sum()),
        "majority_save": int((frame["label"] == 1).sum()),
        "unanimous": int(frame["unanimous"].sum()),
        "two_to_one_split": int((~frame["unanimous"]).sum()),
        "output": str(OUTPUT.relative_to(ROOT)),
        "output_bytes": OUTPUT.stat().st_size,
        "output_sha256": sha256(OUTPUT),
    }
    MANIFEST.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
