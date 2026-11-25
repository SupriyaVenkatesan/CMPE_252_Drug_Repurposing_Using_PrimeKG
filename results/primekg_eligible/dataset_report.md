# Dataset pin, schema and usage terms

## Pinned snapshot

| field | value |
|---|---|
| dataset | PrimeKG (Harvard Dataverse doi:10.7910/DVN/IXA7BM) |
| file | kg.csv |
| source URL | https://dataverse.harvard.edu/api/access/datafile/6180620 |
| download date (UTC) | 2026-11-17T02:08:21 |
| SHA-256 | `0a5a010fcd88dcec572a573d98c26c1c244a11567092b6161d3ded611cfb5c69` |
| size (bytes) | 981751236 |
| nodes.csv SHA-256 | `6a1b8ce946278a786f4df8b9260de9851ff6d93a1a407849861ffb554f74f8e4` |
| edges.csv SHA-256 | `8330aafab00f3fdcd9bae08c38e6960d3a6f79642537f0265cc4c05671d8fcce` |

## Usage terms

* The PrimeKG GitHub repository's MIT licence covers **software and tools only**.
* Dataset use is governed by the terms on the Harvard Dataverse record (doi:10.7910/DVN/IXA7BM) and by the licences of PrimeKG's upstream sources. Record the Dataverse terms text here when the file is downloaded.
* Provenance note: Dataset terms are separate from the MIT software licence of the PrimeKG repo; see the Dataverse record for data terms.

No patient records, paid data or the original HGTDR dataset are used.

## Raw -> clean counts

| stage | count |
|---|---|
| raw rows | 8100498 |
| invalid rows (logged to invalid_rows.csv) | 0 |
| rows after invalid removal | 8100498 |
| duplicate / reverse-copy rows removed | 4050434 |
| unique canonical edges | 4050064 |
| unique indication pairs | 9388 |
| drugs with >=1 indication | 1801 |
| diseases with >=1 indication | 1363 |

Missing values per column (raw): none

## Node types

| node type | count |
|---|---|
| biological_process | 28642 |
| gene_protein | 27610 |
| disease | 17080 |
| effect_phenotype | 15311 |
| anatomy | 14033 |
| molecular_function | 11169 |
| drug | 7957 |
| cellular_component | 4176 |
| pathway | 2516 |
| exposure | 818 |

The combined `gene/protein` type is kept as released (sanitised to `gene_protein`). No new node types or inferred associations are created.

## Edge types (canonical direction; the graph adds `rev_*` copies)

| edge type | count |
|---|---|
| `anatomy -[anatomy_protein_present]-> gene_protein` | 1518203 |
| `drug -[drug_drug]-> drug` | 1336314 |
| `gene_protein -[protein_protein]-> gene_protein` | 321075 |
| `disease -[disease_phenotype_positive]-> effect_phenotype` | 150317 |
| `biological_process -[bioprocess_protein]-> gene_protein` | 144805 |
| `cellular_component -[cellcomp_protein]-> gene_protein` | 83402 |
| `disease -[disease_protein]-> gene_protein` | 80411 |
| `gene_protein -[molfunc_protein]-> molecular_function` | 69530 |
| `drug -[drug_effect]-> effect_phenotype` | 64784 |
| `biological_process -[bioprocess_bioprocess]-> biological_process` | 52886 |
| `gene_protein -[pathway_protein]-> pathway` | 42646 |
| `disease -[disease_disease]-> disease` | 32194 |
| `drug -[contraindication]-> disease` | 30675 |
| `drug -[drug_protein]-> gene_protein` | 25468 |
| `anatomy -[anatomy_protein_absent]-> gene_protein` | 19887 |
| `effect_phenotype -[phenotype_phenotype]-> effect_phenotype` | 18736 |
| `anatomy -[anatomy_anatomy]-> anatomy` | 14032 |
| `molecular_function -[molfunc_molfunc]-> molecular_function` | 13574 |
| `drug -[indication]-> disease` | 9388 |
| `cellular_component -[cellcomp_cellcomp]-> cellular_component` | 4845 |
| `effect_phenotype -[phenotype_protein]-> gene_protein` | 3330 |
| `drug -[off_label_use]-> disease` | 2568 |
| `pathway -[pathway_pathway]-> pathway` | 2535 |
| `disease -[exposure_disease]-> exposure` | 2304 |
| `exposure -[exposure_exposure]-> exposure` | 2070 |
| `biological_process -[exposure_bioprocess]-> exposure` | 1625 |
| `exposure -[exposure_protein]-> gene_protein` | 1212 |
| `disease -[disease_phenotype_negative]-> effect_phenotype` | 1193 |
| `exposure -[exposure_molfunc]-> molecular_function` | 45 |
| `cellular_component -[exposure_cellcomp]-> exposure` | 10 |

## Drug-disease relations

| relation | edges | overlap with indication pairs | used in graph? |
|---|---|---|---|
| contraindication | 30675 | 87 | no (treatment-label proxy) |
| off_label_use | 2568 | 125 | no (treatment-label proxy) |
| indication | 9388 | - | training partition only |

Contraindication and off-label relations are never merged into indication labels.
