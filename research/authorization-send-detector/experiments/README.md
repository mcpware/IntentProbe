# Experiments

One experiment directory represents one frozen hypothesis.

Required contents:

- `config.json` — source revisions, renderer, split fingerprint, model revision, layers/pooling/head;
- `environment.txt` — Python and package versions plus hardware;
- `metrics.json` — point estimates, sample sizes, confidence intervals, calibration;
- `run.log` — success, failure, or interruption;
- `README.md` — hypothesis, result, leakage checks, and lesson learned.

Do not reuse a final-test result to choose the next layer, pooling rule, threshold, or model.
