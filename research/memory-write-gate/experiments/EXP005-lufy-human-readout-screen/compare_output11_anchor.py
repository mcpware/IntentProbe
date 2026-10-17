#!/usr/bin/env python3
"""Post-test diagnostic: compare the selected readout with frozen output 11."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, balanced_accuracy_score, roc_auc_score


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
DATA = ROOT / "data" / "processed" / "exp003" / "lufy_majority.parquet"
CONFIG = HERE / "config.json"
SCREEN = HERE / "screen.py"
TEST_SCORES = HERE / "test_scores.npz"
EXP003_ANALYSIS = HERE.parent / "EXP003-lufy-human-transfer" / "analyze.py"
OUTPUT = HERE / "posthoc_output11_anchor.json"


def load_module(path: Path, name: str):
    """Load one experiment module by path."""
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def metrics(y: np.ndarray, score: np.ndarray) -> dict[str, float]:
    """Compute the same ranking and threshold metrics as the main screen."""
    return {
        "auprc": float(average_precision_score(y, score)),
        "auroc": float(roc_auc_score(y, score)),
        "balanced_accuracy_at_0_5": float(balanced_accuracy_score(y, score >= 0.5)),
    }


def main() -> None:
    """Fit output 11 on development users and compare on the already-open test."""
    screen = load_module(SCREEN, "memory_exp005_screen")
    bootstrap = load_module(EXP003_ANALYSIS, "memory_exp003_analysis")
    config = json.loads(CONFIG.read_text(encoding="utf-8"))
    frame = pd.read_parquet(DATA)
    y = frame["label"].to_numpy(dtype=np.int8)
    user_ids = frame["user_name"].astype(str).map(screen.anonymized_user_id).to_numpy()
    test_mask = np.isin(user_ids, np.asarray(config["split"]["heldout_test_user_ids"]))
    development_indices = np.flatnonzero(~test_mask)
    test_indices = np.flatnonzero(test_mask)
    anchor = screen.CachedActivationProbe(
        cache_dir=str(HERE / "cache"),
        pooling="masked_mean",
        start_layer=11,
        end_layer=11,
    ).fit(development_indices[:, None], y[development_indices])
    anchor_score = anchor.predict_proba(test_indices[:, None])[:, 1]
    with np.load(TEST_SCORES, allow_pickle=False) as archive:
        winner_score = archive["activation"].astype(float)
        test_y = archive["label"].astype(np.int8)
    test_groups = user_ids[test_indices]
    assert np.array_equal(test_y, y[test_indices])

    row_weights = bootstrap.group_bootstrap_row_weights(test_groups)
    anchor_auprc, anchor_auroc = bootstrap.weighted_rank_bootstrap(
        test_y, anchor_score, row_weights
    )
    winner_auprc, winner_auroc = bootstrap.weighted_rank_bootstrap(
        test_y, winner_score, row_weights
    )
    anchor_result = metrics(test_y, anchor_score)
    winner_result = metrics(test_y, winner_score)
    result = {
        "experiment_id": "MEM-EXP005-POSTHOC-OUTPUT11",
        "status": "POST_TEST_DIAGNOSTIC_NOT_CONFIRMATORY",
        "test_rows": len(test_y),
        "test_users": int(len(np.unique(test_groups))),
        "selected_masked_mean_outputs_7_8": winner_result,
        "frozen_masked_mean_output_11": {
            **anchor_result,
            "auprc_user_bootstrap_95ci": bootstrap.percentile(anchor_auprc),
            "auroc_user_bootstrap_95ci": bootstrap.percentile(anchor_auroc),
        },
        "selected_minus_output11": {
            "auprc_difference": winner_result["auprc"] - anchor_result["auprc"],
            "auprc_difference_user_bootstrap_95ci": bootstrap.percentile(
                winner_auprc - anchor_auprc
            ),
            "auroc_difference": winner_result["auroc"] - anchor_result["auroc"],
            "auroc_difference_user_bootstrap_95ci": bootstrap.percentile(
                winner_auroc - anchor_auroc
            ),
            "resamples": 2000,
            "unit": "held-out user",
            "seed": 20261017
        },
        "claim_boundary": "Output 11 was evaluated after the held-out test had been opened for the development-selected winner. This comparison diagnoses readout engineering and is not a second confirmatory test."
    }
    OUTPUT.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
