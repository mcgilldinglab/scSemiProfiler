# Current code rules

This file describes the current `refactor` implementation and the invariants
that future optimization work must preserve.

## Output invariants

- Do not change seeds, epoch counts, mini-stage weights, learning rates, KNN
  membership, PCA dimensions, or representative-selection formulas in an
  exact-output optimization.
- Generated arrays, model state dictionaries, training histories, status files,
  and every retained AnnData field must compare value-for-value with upstream.
- The sole approved structural difference is the removal of unused dense
  `sample_sc/*.h5ad:obsm['adj']`; `fast_cellgraph` still returns the logical CSR
  adjacency for callers that need it.
- Existing target inference files are authoritative resume markers and must be
  skipped before any representative preparation or model loading.

## Current data flow

- `scprocess` computes pathway scores with a sparse membership matrix but writes
  the same full `geneset_scores/*.npy` compatibility files.
- `setdata` memory-maps those files, materializes only `hvset.npy` columns as
  float32, and never reconstructs the removed dense adjacency.
- `scinfer` prepares each representative once and reuses the prepared AnnData,
  feature variances, and one loaded `processed_bulkdata.h5ad` across targets.
- `myVAE.sample` applies the expression bound once; inference exposes the
  resulting CPU tensor to NumPy without a second full-size copy.
- Active selection keeps 100 PCA components, the existing randomized solver,
  and the existing patient-level FAISS query order.

## Required verification

Run from the repository root with the `scsemi` environment:

```bash
export LD_LIBRARY_PATH=/mnt/md2/bowen/miniconda3/envs/scsemi/lib:/usr/local/cuda-12.1/lib64
/mnt/md2/bowen/miniconda3/envs/scsemi/bin/python equivalence/regression_checks.py .
SCSEMI_PYTHON=/mnt/md2/bowen/miniconda3/envs/scsemi/bin/python equivalence/run_equivalence.sh
```

Do not describe a change as equivalent unless both commands pass. Approximate
KNN, reduced PCA, mixed precision, altered batch sizes, and reduced epochs require
a separate methodological validation and are not covered by this rule.
