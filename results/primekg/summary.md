# Results summary

Metrics measure recovery of withheld documented indications against unlabeled pairs; they do not measure therapeutic efficacy.

## Setting: disease

Test pairs: 1531, evaluable diseases (queries): 205, mean candidate pool: 7957

| model | n seeds | ap | recall@10 | auroc | precision@10 | ndcg@10 | recall@20 | params | train s | peak MB | pairs/s |
|---|---|---|---|---|---|---|---|---|---|---|---|
| popularity | 1 | 0.6361 ± 0.0000 | 0.0823 ± 0.0000 | 0.9301 ± 0.0000 | 0.0844 ± 0.0000 | 0.1007 ± 0.0000 | 0.1626 ± 0.0000 | 0 | 0.0 | nan | 1806379 |
| sage | 3 | 0.3950 ± 0.0306 | 0.0528 ± 0.0116 | 0.8521 ± 0.0305 | 0.0567 ± 0.0092 | 0.0681 ± 0.0121 | 0.0848 ± 0.0182 | 6769089 | 65.9 | 2675 | 2439864 |

## Setting: drug

Test pairs: 1403, evaluable diseases (queries): 640, mean candidate pool: 270

| model | n seeds | ap | recall@10 | auroc | precision@10 | ndcg@10 | recall@20 | params | train s | peak MB | pairs/s |
|---|---|---|---|---|---|---|---|---|---|---|---|
| popularity | 1 | 0.0969 ± 0.0000 | 0.0425 ± 0.0000 | 0.5446 ± 0.0000 | 0.0069 ± 0.0000 | 0.0209 ± 0.0000 | 0.0591 ± 0.0000 | 0 | 0.0 | nan | 530890 |
| sage | 3 | 0.1097 ± 0.0047 | 0.0431 ± 0.0058 | 0.5637 ± 0.0046 | 0.0081 ± 0.0009 | 0.0221 ± 0.0013 | 0.0833 ± 0.0123 | 6769089 | 64.0 | 2675 | 314016 |

## Setting: random

Test pairs: 1408, evaluable diseases (queries): 643, mean candidate pool: 7947

| model | n seeds | ap | recall@10 | auroc | precision@10 | ndcg@10 | recall@20 | params | train s | peak MB | pairs/s |
|---|---|---|---|---|---|---|---|---|---|---|---|
| popularity | 1 | 0.6193 ± 0.0000 | 0.1344 ± 0.0000 | 0.9265 ± 0.0000 | 0.0297 ± 0.0000 | 0.1312 ± 0.0000 | 0.2265 ± 0.0000 | 0 | 0.0 | nan | 1697574 |
| sage | 3 | 0.5728 ± 0.0039 | 0.1349 ± 0.0030 | 0.9025 ± 0.0083 | 0.0295 ± 0.0005 | 0.1284 ± 0.0009 | 0.1979 ± 0.0046 | 6769089 | 108.6 | 2675 | 3136748 |
