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
| random | 7957 | 1801 | 14080 | 77.8196 | 0.3040 | 0.8895 | 3.3439 | 0.6193 | 6.8121 | 0.6257 | 2.0135 | 0.0909 | 0.3107 |
| drug | 270 | 270 | 14030 | 0.0000 | 0.0818 | 0.4577 | 0.9001 | 0.0969 | 1.0658 | 0.0969 | 1.0658 | 0.0909 | 0.0909 |
| disease | 7957 | 1801 | 15310 | 77.9229 | 0.2987 | 0.8889 | 3.2855 | 0.6361 | 6.9969 | 0.6425 | 2.0610 | 0.0909 | 0.3117 |

## Ranking metrics (unaffected by the unlabeled sample, affected by the candidate set)

| setting | metric | all_drugs | eligible_only | oracle_all_drugs |
|---|---|---|---|---|
| random | popularity Recall@10 | 0.1344 | 0.1344 | 0.0031 |
| drug | popularity Recall@10 | 0.0425 | 0.0425 | 0.0268 |
| disease | popularity Recall@10 | 0.0823 | 0.0823 | 0.0010 |
