# MiniCPM5 Authorization Probe — EXP002 Research Artifact

This 28 KB artifact is a class-weighted logistic head plus train-only normalization statistics for
masked-mean MiniCPM5 hidden outputs 13 and 14. MiniCPM5 itself is not included.

Status: **research only; standalone send-gate calibration failed**.

On a separate 3,000-row silver test, the probe reached 0.8223 AUPRC and 0.8137 AUROC. A matched
word/bigram TF-IDF model trained on the same 1,600 rows reached 0.8079 AUPRC and 0.7897 AUROC. The
paired AUROC gain was supported by its 95% bootstrap interval, but the AUPRC interval crossed zero.
TF-IDF also had slightly better recall in the oracle <=1% and <=5% false-approve regions.

Most importantly, validation-selected activation thresholds did not calibrate across datasets. The
threshold intended for <=5% false approvals produced 9.49% held-out false approvals, and the <=1%
threshold produced 5.07%. Do not use this artifact to authorize, send, submit, purchase, delete, or
perform another consequential action. A deterministic authorization state machine must bind exact
payload, scope, recipient, timing, and approval facts.

See `metadata.json`, `../../experiments/EXP002-activation-screen/heldout_result.json`, and the EXP002
README for provenance, exact metrics, uncertainty, and limitations.
