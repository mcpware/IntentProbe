#!/usr/bin/env python3
"""Fit the EXP002 matched surface-text baseline."""

from __future__ import annotations

import argparse
import json
import resource
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

from metrics_common import binary_metrics


HERE = Path(__file__).resolve().parent
PROJECT = HERE.parents[1]
DATA = PROJECT / "data" / "processed" / "exp002"
CONFIG = HERE / "config.json"


def main() -> None:
    """Train only on the frozen screen train and score screen validation."""
    argparse.ArgumentParser(description=__doc__).parse_args()
    started = time.perf_counter()
    config = json.loads(CONFIG.read_text(encoding="utf-8"))
    train = pd.read_parquet(DATA / "train.parquet", columns=["rendered", "label"])
    validation = pd.read_parquet(DATA / "validation.parquet", columns=["rendered", "label"])
    train_y = train["label"].map({"deny": 0, "approve": 1}).to_numpy(dtype=np.int8)
    validation_y = validation["label"].map({"deny": 0, "approve": 1}).to_numpy(dtype=np.int8)
    if np.isnan(train_y).any() or np.isnan(validation_y).any():
        raise ValueError("Unexpected label")

    vectorizer = TfidfVectorizer(
        ngram_range=(1, 2),
        min_df=2,
        max_features=50000,
        sublinear_tf=True,
        strip_accents="unicode",
        dtype=np.float32,
    )
    train_x = vectorizer.fit_transform(train["rendered"])
    validation_x = vectorizer.transform(validation["rendered"])
    model = LogisticRegression(
        C=1.0,
        class_weight="balanced",
        solver="liblinear",
        max_iter=1000,
        random_state=config["sampling_seed"],
    )
    model.fit(train_x, train_y)
    scores = model.predict_proba(validation_x)[:, 1]
    result = {
        "experiment_id": config["experiment_id"],
        "baseline": "word_1_2_tfidf_logistic",
        "completed_at": datetime.now(timezone.utc).isoformat(),
        "train_rows": int(len(train)),
        "validation_rows": int(len(validation)),
        "vocabulary_size": int(len(vectorizer.vocabulary_)),
        "metrics": binary_metrics(validation_y, scores, config["sampling_seed"]),
        "elapsed_seconds": time.perf_counter() - started,
        "peak_rss_kib": int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss),
    }
    (HERE / "surface_baseline.json").write_text(
        json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
