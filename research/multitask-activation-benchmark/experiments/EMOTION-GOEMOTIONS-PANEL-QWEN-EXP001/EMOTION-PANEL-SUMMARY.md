# GoEmotions: Qwen four-emotion activation-readout panel

All targets use identical frozen text rows, splits, backbone activations, candidate family,
and matched word-plus-character TF-IDF baseline protocol. Each readout was selected on
validation AUPRC before its one-shot test evaluation.

| Target | Selected readout | Depth | Test positives | Activation AUPRC | TF-IDF AUPRC | Difference (95% CI) |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| admiration | masked_mean 2--4 | 0.125 | 504 | 0.6014 | 0.6593 | -0.0580 [-0.0904, -0.0264] |
| anger | masked_mean 5--5 | 0.208 | 198 | 0.3232 | 0.4254 | -0.1022 [-0.1596, -0.0466] |
| curiosity | masked_mean 3--3 | 0.125 | 283 | 0.4143 | 0.4309 | -0.0166 [-0.0601, +0.0259] |
| sadness | masked_mean 16--16 | 0.667 | 156 | 0.3373 | 0.4823 | -0.1450 [-0.2132, -0.0690] |

## Descriptive readout pattern

Pooling winners: {'masked_mean': 4}. Selected normalized depth ranges from 0.125 to 0.667. These within-backbone task differences are descriptive; the six-backbone crossed matrix is required before attributing the pattern to task rather than one architecture.
