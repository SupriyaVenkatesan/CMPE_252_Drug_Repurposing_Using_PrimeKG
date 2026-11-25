# Candidate-pool diagnostic: the drug-eligibility artefact

How much of AP/AUROC is explained by drug *eligibility* alone?

Only 1,801 of PrimeKG's 7,957 drug-type nodes ever carry an indication, but the unlabeled
pool is drawn from all of them. A scorer that merely answers "is this drug ever indicated
for anything?" can therefore separate most positives from most unlabeled pairs without
using the disease at all.

**Lift = AP / prevalence.** A random scorer has lift 1.0 whatever the base rate, so lift is
comparable across pools while raw AP is not.

## Pool composition and AP decomposition

| setting | cand_drugs | eligible | unlabeled_pairs | unlab_drug_never_indicated_% | oracle_ap | oracle_auroc | oracle_lift | pop_ap_full | pop_lift_full | pop_ap_eligible | pop_lift_eligible | prevalence_full | prevalence_eligible |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| random | 1801 | 1801 | 14080 | 0.0000 | 0.0888 | 0.4986 | 0.9766 | 0.3605 | 3.9656 | 0.3605 | 3.9656 | 0.0909 | 0.0909 |
| drug | 270 | 270 | 14030 | 0.0000 | 0.0818 | 0.4577 | 0.9001 | 0.0969 | 1.0658 | 0.0969 | 1.0658 | 0.0909 | 0.0909 |
| disease | 1801 | 1801 | 15310 | 0.0000 | 0.0869 | 0.4936 | 0.9555 | 0.3626 | 3.9891 | 0.3626 | 3.9891 | 0.0909 | 0.0909 |

## Ranking metrics (unaffected by the unlabeled sample, affected by the candidate set)

| setting | metric | all_drugs | eligible_only | oracle_all_drugs |
|---|---|---|---|---|
| random | popularity Recall@10 | 0.1344 | 0.1344 | 0.0031 |
| drug | popularity Recall@10 | 0.0425 | 0.0425 | 0.0268 |
| disease | popularity Recall@10 | 0.0823 | 0.0823 | 0.0010 |
