# scSemiProfiler performance refactor

Baseline: upstream `c2aea02`. Every numerical change below is value-preserving —
no algorithm swapped, no tolerance loosened, no epoch count touched. The sparse
cell-graph follow-up deliberately removes one unused output field,
`sample_sc/*.h5ad:obsm['adj']`; all remaining values stay exact.

The A/B suite compares every remaining output element-by-element, including model
checkpoints and training-history pickles, and permits only the declared removal of
the dense adjacency field.

## Environment

Upstream pins torch 1.12.1+cu113, which cannot target sm_89 (RTX 40-series). The
combination used for the measurements below was python 3.9, scvi-tools 1.0.4,
torch 2.1.2+cu121, pytorch-lightning 2.1.1, faiss-cpu 1.7.4, numpy 1.26.2,
anndata 0.10.3, scanpy 1.9.6, and jax/jaxlib pinned to 0.4.19 (chex 0.1.7 breaks
on newer jax).

`getGeneSetMatrix` reads `genesets/` relative to the working directory, so the
pipeline has to be run from the repository root.

## Verifying

```bash
equivalence/run_equivalence.sh
```

Runs the pipeline twice from identical seeds — once from a pristine worktree of
the branch point, once from this tree — and diffs every output other than the
declared `obsm['adj']` deletion. Exits non-zero on any other difference.
`SCSEMI_PYTHON`, `SCSEMI_BASELINE` and `SCSEMI_DEVICE` override the interpreter,
the baseline checkout and the torch device.

`equivalence/bench_step.py` is the interleaved training-loop benchmark used for
the training rows in the table.

## Measured

Example cohort: 12 samples, 41 553 cells, 6 030 genes, 2 representatives.
RTX 4090, torch 2.1.2+cu121, scvi-tools 1.0.4.

| Stage | Before | After | Speedup |
|---|---|---|---|
| `scprocess` (preprocess 2 representatives) | 153.8 s | 13.5 s | **11.4×** |
| training loop, 60 epochs — wall | 29.4 s | 13.9 s | **2.1×** |
| training loop, 60 epochs — CPU | 81.8 s | 12.6 s | **6.5×** |
| `tgtinfer` (1 target, full 1350 epochs, end to end) | 525.5 s | 343.3 s | 1.5× |
| `activeselection` | 20.9 s | 12.9 s | 1.6× |
| `fast_cellgraph` (1200 cells) | 1.66 s | 0.17 s | **9.6×** |
| `getGeneSetMatrix` (6030 real symbols) | 3.29 s | 0.63 s | 5.2× |
| `best_patient` (220-sample cohort) | 31 ms | 3 ms | **10×** |
| RRHO quadrant (max_rank 60) | 11.3 s | 1.87 s | 6.1× |

The preprocessing and selection wins grow with cohort size — several of the
fixes below were quadratic or worse in cells or samples.

### Sparse cell-graph follow-up

The follow-up A/B run compared this complete branch with upstream `c2aea02`; it
is not an isolated measurement of the adjacency change alone. On the shared test
host, `scprocess` was 219.788 s versus 11.213 s and fast `scinfer` was 167.678 s
versus 92.905 s. The 1,200-cell logical-adjacency probe was 2.929 s versus 0.468 s.

The persistent-size reduction is attributable exactly to the removed uncompressed
float64 adjacency datasets:

| Representative | Cells | Baseline h5ad | Sparse h5ad | Removed |
|---|---:|---:|---:|---:|
| `BGCV09_CV0279` | 2,720 | 191,395,928 B | 132,208,360 B | 59,187,568 B |
| `MH9143426` | 7,953 | 891,521,519 B | 385,519,479 B | 506,002,040 B |

The full suite passed 19 unit arrays and compared 12 process, 55 inference, and
57 selection files. The only structural difference was the declared removal of
`obsm['adj']` from the two representative h5ads; every remaining value matched.

### On the training numbers

The measurement host was shared. During the `tgtinfer` pair its load average
was ~79 and the GPU sat at 2–20% utilisation, so end-to-end wall clock partly measures
contention: the *same* pristine `tgtinfer` measured 400.4 s on an idle box and
525.5 s under load — a spread wider than the effect being measured. Treat the
1.5× end-to-end figure as indicative only.

The training-loop rows come from `equivalence/bench_step.py`, which runs the same
fixed workload alternately from both trees and reports the best of three — the
least-contended sample is the closest estimate of true cost. That measurement is
stable (refactored: 13.90 / 14.05 / 14.45 s) where the baseline is not
(29.4–36.9 s), which is itself the finding: the original loop is CPU-bound, so it
degrades under load while the refactored one barely moves.

**CPU time is the honest headline: 81.8 s → 12.6 s for the same 60 epochs.** The
original burned ~2.8 cores of host CPU per step formatting loss tensors into a
log file, syncing the device to do it, and shuttling a (cells × features) matrix
across the PCIe bus; the refactored loop stays under one core. On a shared
machine that also means it stops competing with everyone else's jobs.

Note the benchmark exercises pretrain-1-style training (default batch size 128),
which is where the step count actually is: pretrain 1 is 400 epochs × ~22
steps/epoch, while the five inference mini-stages run one step per epoch because
`batch_size` covers the whole sample. That is why a single target logged 13 268
training steps.

## What was slow, and why

### `singlecell_process.py`

- **`getGeneSetMatrix` was called once per sample inside the loop.** It re-parses
  the 1.2 MB GMT and rebuilds the full 2922 × n_genes pathway matrix every time,
  but it only depends on `var.index`, which is identical for every sample. Hoisted
  out of the loop. Also hoisted `np.array(gene_set_matrix).T` and the denominator,
  which were rebuilt per sample (a 145 MB copy each).
- **`gen_tf_gene_table` used `gene_names.index(x)`** — a linear scan of the gene
  list for every pathway target, so O(pathways × targets × genes). Replaced with a
  dict built once. First-occurrence semantics preserved for duplicate gene names.
- **`fast_cellgraph`'s neighbour extraction was O(cells²).** For each cell it
  rescanned the nonzero index list from position zero until it found that cell's
  block. The nonzero list is already row-major and column-ascending, which is the
  order the scan collected, so it reshapes directly into the neighbour table.
- **`fast_cellgraph` allocated a second n×n matrix** (`np.identity(n) * diagw`)
  just to add a constant to the diagonal, and used Python loops to read and to
  clear that diagonal. Now an in-place diagonal write, `np.diag`, and
  `np.fill_diagonal`.
- **The neighbour aggregation materialised a (cells, k-1, genes) array** and took
  `log1p` of all of it. Now `log1p` runs once on (cells, genes) and the gather-sum
  is chunked, so peak memory is bounded. Per-row reduction order is unchanged.
- **`hvmask` used `gene in numpy_array`**, a linear scan per gene — quadratic in
  gene count. Now a set.

### `vaegan.py` (this is the module the pipeline actually runs)

- **A debug trace was appended to `vaegan_train.txt` on every training step.**
  Formatting the loss tensors (`str(loss)`) forces a GPU→CPU sync each step, and
  the file grows without bound across runs — a short test run produced 2.2 MB.
  It is now off by default; set `SCSEMI_TRAIN_LOG=<path>` to restore it verbatim.
  **This is the one deliberate behavioural difference in the refactor.**
- **The per-feature loss weights were rebuilt every step** — a full sort of the
  feature-variance vector plus two `torch.tensor()` copies — despite depending
  only on values fixed at construction. Cached.
- **The discriminator's fake batch round-tripped through the CPU.**
  `sample()` pulled the full (cells × features) matrix to host memory,
  thresholded it there, and the caller shipped it straight back to the GPU. A new
  `to_cpu=False` argument keeps it on device; the thresholding is elementwise so
  the values are unchanged.
- Bulk target slicing and device transfer cached instead of redone per step.

### `inference.py`

- **`unisemi0`…`unisemi5` were six near-identical copies** differing only in a
  bulk-weight constant and which `tmp/` file they wrote and re-read. Collapsed
  into one `_run_ministage`. The mini-stage schedule, including the `ministages`
  gating, is preserved exactly.
- **Each mini-stage wrote its weights to disk and the next one read them back.**
  The state dict is now handed over in memory. Checkpoints are still written, so
  the on-disk result is unchanged.
- **`os.listdir()` was used for existence checks inside loops** — listing a
  growing directory once per target, then scanning the returned list. Now
  `os.path.exists`.
- `unisemi5` was unreachable and referenced an undefined `name`; removed.

### `representative_selection.py`

- **`best_patient` recomputed both distances inside an O(n²) Python loop**, so
  every (candidate, member) pair redid a full vector norm. Distances are now
  computed per candidate in one batched call. The accumulation stays a Python
  loop *on purpose*: numpy's scalar `**0.5` promotes float32 to float64 while the
  array form does not, and the running sum must round in the original order.
  `_bulk_distances` reproduces both details exactly.
- **It also deep-copied a full label vector per candidate** and kept them all,
  though only the winner's is ever read. Now built once, for the winner.
- **`compute_cluster_heterogeneity` rebuilt the faiss index on every patient**,
  even though the matrix being searched — the representative's cells — is the
  same throughout the loop. Built once.

### `utils.py`

- **`rrho_plot` had the same doubly-nested loop written out four times**, each
  calling a 200-digit mpmath hypergeometric tail. Factored into `_rrho_matrix`.
- **`mpmath.fac` is now memoised.** The RRHO loops evaluate the same handful of
  factorials once per (rank1, rank2, i) triple. mpmath is deterministic at fixed
  precision, so the cached value is what the call would have returned.
- **`set(lista[:rank1])` was rebuilt in the inner loop** though it only depends on
  the outer index — an extra factor of R. Hoisted. The `union` array was computed
  every iteration but only read by a NaN diagnostic; now computed only when that
  fires.
- **`assemble_cohort` re-read the entire bulk `.h5ad` from disk on every loop
  iteration**, and appended per-cell scalars one at a time through pandas.
- **`get_cohort_sc` / `get_eg_representatives` built their cell mask with a
  Python loop** doing a pandas scalar lookup per cell against a list. Now
  `.isin()`. They also densified the whole matrix to compute `exp(x)-1`; since
  `exp(0)-1 == 0` the stored values can be transformed in place instead.

## Notes

- `fast_generator.py` and `oldinference.py` are **dead code** — nothing imports
  `oldinference`, and `fast_generator` is only imported by it. `inference.py`
  does `from .vaegan import *`, so `vaegan.py` is the live model. The same
  optimisations were applied to `fast_generator.py` to keep the two in sync, but
  they do not affect the pipeline.
- `obsm['adj']` was loaded and converted by `setdata`, but the returned tensor was
  never consumed by the model; `neighborx` is the registered model input. The
  sparse follow-up removes this dead field, stops the loader conversion, and keeps
  the KNN graph in CSR form throughout `fast_cellgraph`. This changes the h5ad
  structure while preserving the logical adjacency returned by `fast_cellgraph`.
- `scinfer` still writes `pretrain1time.txt` into the current working directory.
