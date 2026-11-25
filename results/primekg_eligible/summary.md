# Results summary

Metrics measure recovery of withheld documented indications against unlabeled pairs; they do not measure therapeutic efficacy.

## Setting: disease

Test pairs: 1531, evaluable diseases (queries): 205, mean candidate pool: 1801

| model | n seeds | ap | recall@10 | auroc | precision@10 | ndcg@10 | recall@20 | params | train s | peak MB | pairs/s |
|---|---|---|---|---|---|---|---|---|---|---|---|
| popularity | 1 | 0.3626 ± 0.0000 | 0.0823 ± 0.0000 | 0.7416 ± 0.0000 | 0.0844 ± 0.0000 | 0.1007 ± 0.0000 | 0.1626 ± 0.0000 | 0 | 0.0 | nan | 4150131 |
| sage | 3 | 0.1741 ± 0.0122 | 0.0533 ± 0.0117 | 0.6035 ± 0.0162 | 0.0569 ± 0.0092 | 0.0684 ± 0.0121 | 0.0848 ± 0.0182 | 6769089 | 62.1 | 2674 | 913242 |

## Setting: drug

Test pairs: 1403, evaluable diseases (queries): 640, mean candidate pool: 270

| model | n seeds | ap | recall@10 | auroc | precision@10 | ndcg@10 | recall@20 | params | train s | peak MB | pairs/s |
|---|---|---|---|---|---|---|---|---|---|---|---|
| popularity | 1 | 0.0969 ± 0.0000 | 0.0425 ± 0.0000 | 0.5446 ± 0.0000 | 0.0069 ± 0.0000 | 0.0209 ± 0.0000 | 0.0591 ± 0.0000 | 0 | 0.0 | nan | 1010145 |
| sage | 3 | 0.1097 ± 0.0047 | 0.0431 ± 0.0058 | 0.5637 ± 0.0046 | 0.0081 ± 0.0009 | 0.0221 ± 0.0013 | 0.0833 ± 0.0123 | 6769089 | 64.1 | 2675 | 293289 |

## Setting: random

Test pairs: 1408, evaluable diseases (queries): 643, mean candidate pool: 1791

| model | n seeds | ap | recall@10 | auroc | precision@10 | ndcg@10 | recall@20 | params | train s | peak MB | pairs/s |
|---|---|---|---|---|---|---|---|---|---|---|---|
| popularity | 1 | 0.3605 ± 0.0000 | 0.1344 ± 0.0000 | 0.7239 ± 0.0000 | 0.0297 ± 0.0000 | 0.1312 ± 0.0000 | 0.2265 ± 0.0000 | 0 | 0.0 | nan | 3915301 |
| sage | 3 | 0.2909 ± 0.0060 | 0.1341 ± 0.0017 | 0.6835 ± 0.0009 | 0.0294 ± 0.0003 | 0.1287 ± 0.0008 | 0.1938 ± 0.0044 | 6769089 | 88.5 | 2674 | 1404308 |
