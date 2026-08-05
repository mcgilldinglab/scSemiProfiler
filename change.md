# Change log

## 2026-08-05 - exact-output memory and rerun optimizations

- Fixed `tgtinfer` so an existing inferred target returns before loading data or
  models.
- Reused each representative's prepared AnnData and the cohort bulk AnnData
  across targets in `scinfer`.
- Memory-mapped full pathway-score files and removed redundant AnnData, tensor,
  NumPy, and threshold-mask copies.
- Kept pathway membership sparse during score calculation and preallocated the
  combined score matrix used for HV-pathway selection.
- Reused the already loaded bulk AnnData during active selection. A batched
  FAISS/PCA-copy candidate was profiled but not retained because it showed no
  stable whole-stage speedup and increased peak memory.
- Added focused regression checks; the full equivalence suite remains the
  release gate for values, model checkpoints, histories, and selections.

Validation on the bundled 12-sample cohort passed on 2026-08-05: 19 unit arrays,
12 preprocessing files, 55 inference files, and 57 selection files matched
upstream value-for-value. The only difference was the already approved removal
of dense `obsm['adj']` from two representative h5ad files. Measured preprocessing
was 218.375 s upstream versus 11.353 s current; fast inference was 178.619 s
versus 92.774 s. Selection timing was load-sensitive and is not claimed as a
whole-stage speedup from this single run.

After the batched selection candidate was removed, the final one-read selection
implementation was rerun from the validated inference tree and again matched all
57 upstream selection files; its observed wall time in that run was 23.110 s.

## 2026-08-05 - sparse cell graph (`cf1fe51`)

- Kept the KNN graph in CSR form and stopped persisting the unused dense
  `obsm['adj']` matrix.
- Preserved `neighbors`, `neighborx`, `selfw`, training outputs, and selections.

## Earlier refactor (`8b2e7da`)

- Removed repeated pathway parsing, quadratic scans, training-step disk logging,
  GPU/CPU round trips, and inference checkpoint reloads while preserving outputs.
