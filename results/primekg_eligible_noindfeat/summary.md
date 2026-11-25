# Results summary

Metrics measure recovery of withheld documented indications against unlabeled pairs; they do not measure therapeutic efficacy.

## Setting: disease

Test pairs: 1531, evaluable diseases (queries): 205, mean candidate pool: 1801

| model | n seeds | ap | recall@10 | auroc | precision@10 | ndcg@10 | recall@20 | params | train s | peak MB | pairs/s |
|---|---|---|---|---|---|---|---|---|---|---|---|
| sage | 3 | 0.1161 ± 0.0011 | 0.0050 ± 0.0004 | 0.5714 ± 0.0076 | 0.0037 ± 0.0005 | 0.0047 ± 0.0004 | 0.0160 ± 0.0010 | 6766529 | 52.0 | 2670 | 887798 |

## Setting: drug

Test pairs: 1403, evaluable diseases (queries): 640, mean candidate pool: 270

| model | n seeds | ap | recall@10 | auroc | precision@10 | ndcg@10 | recall@20 | params | train s | peak MB | pairs/s |
|---|---|---|---|---|---|---|---|---|---|---|---|
| sage | 3 | 0.1138 ± 0.0045 | 0.0404 ± 0.0012 | 0.5796 ± 0.0056 | 0.0078 ± 0.0001 | 0.0189 ± 0.0013 | 0.0900 ± 0.0097 | 6766529 | 127.9 | 2676 | 314204 |

## Setting: random

Test pairs: 1408, evaluable diseases (queries): 643, mean candidate pool: 1791

| model | n seeds | ap | recall@10 | auroc | precision@10 | ndcg@10 | recall@20 | params | train s | peak MB | pairs/s |
|---|---|---|---|---|---|---|---|---|---|---|---|
| sage | 3 | 0.3586 ± 0.0073 | 0.1315 ± 0.0041 | 0.7457 ± 0.0010 | 0.0295 ± 0.0014 | 0.0979 ± 0.0029 | 0.1776 ± 0.0048 | 6766529 | 475.7 | 2673 | 1455610 |
