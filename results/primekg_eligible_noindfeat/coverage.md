# Coverage and degree analysis

Evaluable = held-out disease with at least one positive in the partition and a non-empty candidate pool. Diseases are bucketed by their degree in the training graph (tertiles) to expose uneven coverage.

## Coverage per split

| setting | test_pairs | test_diseases | evaluable_diseases | test_drugs | candidate_universe | median_pos_per_disease | median_test_disease_degree | median_test_drug_degree | test_diseases_zero_train_labels |
|---|---|---|---|---|---|---|---|---|---|
| random | 1408 | 643 | 643 | 738 | 1801 | 1.0000 | 32.0000 | 893.0000 | 84 |
| drug | 1403 | 640 | 640 | 270 | 270 | 1.0000 | 29.0000 | 851.0000 | 67 |
| disease | 1531 | 205 | 205 | 782 | 1801 | 3.0000 | 17.0000 | 854.0000 | 205 |

## Recall@10 by test-disease degree bucket (mean ± std over seeds)

| setting | model | degree_bucket | n_diseases | recall10_mean | recall10_std | seeds |
|---|---|---|---|---|---|---|
| random | sage | [1, 16] | 203 | 0.1187 | 0.0086 | 3 |
| random | sage | [16, 55] | 225 | 0.1426 | 0.0043 | 3 |
| random | sage | [55, 1183] | 215 | 0.1319 | 0.0121 | 3 |
| drug | sage | [1, 15] | 200 | 0.0175 | 0.0050 | 3 |
| drug | sage | [15, 52] | 223 | 0.0113 | 0.0052 | 3 |
| drug | sage | [52, 942] | 217 | 0.0916 | 0.0031 | 3 |
| disease | sage | [1, 8] | 66 | 0.0000 | 0.0000 | 3 |
| disease | sage | [8, 35] | 70 | 0.0115 | 0.0002 | 3 |
| disease | sage | [35, 1168] | 69 | 0.0033 | 0.0011 | 3 |
