# Coverage and degree analysis

Evaluable = held-out disease with at least one positive in the partition and a non-empty candidate pool. Diseases are bucketed by their degree in the training graph (tertiles) to expose uneven coverage.

## Coverage per split

| setting | test_pairs | test_diseases | evaluable_diseases | test_drugs | candidate_universe | median_pos_per_disease | median_test_disease_degree | median_test_drug_degree | test_diseases_zero_train_labels |
|---|---|---|---|---|---|---|---|---|---|
| random | 1408 | 643 | 643 | 738 | 7957 | 1.0000 | 32.0000 | 893.0000 | 84 |
| drug | 1403 | 640 | 640 | 270 | 270 | 1.0000 | 29.0000 | 851.0000 | 67 |
| disease | 1531 | 205 | 205 | 782 | 7957 | 3.0000 | 17.0000 | 854.0000 | 205 |

## Recall@10 by test-disease degree bucket (mean ± std over seeds)

| setting | model | degree_bucket | n_diseases | recall10_mean | recall10_std | seeds |
|---|---|---|---|---|---|---|
| random | popularity | [1, 16] | 203 | 0.1168 | 0.0000 | 1 |
| random | popularity | [16, 55] | 225 | 0.1585 | 0.0000 | 1 |
| random | popularity | [55, 1183] | 215 | 0.1258 | 0.0000 | 1 |
| random | sage | [1, 16] | 203 | 0.1174 | 0.0009 | 3 |
| random | sage | [16, 55] | 225 | 0.1630 | 0.0038 | 3 |
| random | sage | [55, 1183] | 215 | 0.1221 | 0.0069 | 3 |
| drug | popularity | [1, 15] | 200 | 0.0300 | 0.0000 | 1 |
| drug | popularity | [15, 52] | 223 | 0.0127 | 0.0000 | 1 |
| drug | popularity | [52, 942] | 217 | 0.0845 | 0.0000 | 1 |
| drug | sage | [1, 15] | 200 | 0.0250 | 0.0050 | 3 |
| drug | sage | [15, 52] | 223 | 0.0150 | 0.0080 | 3 |
| drug | sage | [52, 942] | 217 | 0.0886 | 0.0278 | 3 |
| disease | popularity | [1, 8] | 66 | 0.0707 | 0.0000 | 1 |
| disease | popularity | [8, 35] | 70 | 0.0805 | 0.0000 | 1 |
| disease | popularity | [35, 1168] | 69 | 0.0951 | 0.0000 | 1 |
| disease | sage | [1, 8] | 66 | 0.0157 | 0.0098 | 3 |
| disease | sage | [8, 35] | 70 | 0.0499 | 0.0268 | 3 |
| disease | sage | [35, 1168] | 69 | 0.0911 | 0.0059 | 3 |
