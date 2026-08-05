"""Exercise the refactored pure functions on inputs the 12-sample example barely
touches, and dump the results for a byte-wise A/B against the pristine checkout.

usage: unit_probe.py <pkg_root> <out.npz>
"""
import os
import sys
import time
import pickle

pkg_root = os.path.abspath(sys.argv[1])
out_path = os.path.abspath(sys.argv[2])
os.chdir(pkg_root)
sys.path.insert(0, pkg_root)

import numpy as np
import anndata

import scSemiProfiler  # noqa: F401  (ensures package import works)
from scSemiProfiler.singlecell_process import fast_cellgraph, gen_tf_gene_table, getGeneSetMatrix
from scSemiProfiler.representative_selection import best_patient, compute_cluster_heterogeneity
from scSemiProfiler import utils

res = {}
timings = {}


def timed(key, fn):
    t0 = time.perf_counter()
    val = fn()
    timings[key] = time.perf_counter() - t0
    return val


def dense_adj(adata, adj):
    """Read the logical adjacency from either implementation."""
    if 'adj' in adata.obsm:
        return np.asarray(adata.obsm['adj'])
    if hasattr(adj, 'toarray'):
        return adj.toarray()
    return np.asarray(adj)


# ---- fast_cellgraph on a bigger, denser sample than the example provides ----
rng = np.random.default_rng(7)
X = (rng.random((1200, 300), dtype=np.float32) * 40).astype(np.float32)
X[X < 20] = 0.0
ad = anndata.AnnData(X.copy())
ad.obs['cellidx'] = range(ad.shape[0])
ad2, adj = timed('fast_cellgraph', lambda: fast_cellgraph(ad, k=15, diagw=1.0))
res['cg_adj'] = dense_adj(ad2, adj)
res['cg_neighbors'] = np.asarray(ad2.obsm['neighbors'])
res['cg_neighborx'] = np.asarray(ad2.obsm['neighborx'])
res['cg_selfw'] = np.asarray(ad2.obs['selfw'].values)

# k where the graph is small relative to the data, and a non-unit diag weight
ad3 = anndata.AnnData(X[:200].copy())
ad3.obs['cellidx'] = range(200)
ad4, adj4 = fast_cellgraph(ad3, k=5, diagw=0.3)
res['cg2_adj'] = dense_adj(ad4, adj4)
res['cg2_neighbors'] = np.asarray(ad4.obsm['neighbors'])
res['cg2_neighborx'] = np.asarray(ad4.obsm['neighborx'])

# ---- gene set matrix ----
genes = ['GENE%d' % i for i in range(4000)]
gsm, keys = timed('getGeneSetMatrix',
                  lambda: getGeneSetMatrix('c2.cp.v7.4.symbols.gmt',
                                           [g.upper() for g in genes], 'genesets/'))
res['gsm'] = np.asarray(gsm)
res['gsm_keys'] = np.array(list(keys), dtype=object)

# real symbols, so the pathway tables are not all empty
real = list(anndata.read_h5ad('example_data/bulkdata.h5ad').var.index)
gsm2, _ = timed('getGeneSetMatrix_real',
                lambda: getGeneSetMatrix('c2.cp.v7.4.symbols.gmt',
                                         [g.upper() for g in real], 'genesets/'))
res['gsm_real'] = np.asarray(gsm2)

# duplicated gene names: index() keeps the first hit, the dict must too
dup = ['A', 'B', 'A', 'C', 'B']
tbl = gen_tf_gene_table(dup, ['P1', 'P2'], {'P1': ['A', 'C'], 'P2': ['B', 'ZZ']})
res['gtt_dup'] = np.array([tbl['P1'], tbl['P2']])

# ---- active learning on a 220-sample cohort ----
n_samples = 220
reduced_bulk = (rng.standard_normal((n_samples, 100))).astype(np.float32)
cluster_labels = list(rng.integers(0, 4, n_samples))
representatives = [int(np.where(np.array(cluster_labels) == c)[0][0]) for c in range(4)]

bestp, newlabels, hets = timed(
    'best_patient',
    lambda: best_patient(cluster_labels=list(cluster_labels),
                         representatives=list(representatives),
                         reduced_bulk=reduced_bulk, cluster_num=1, new_num=4))
res['bp_bestp'] = np.array([bestp])
res['bp_labels'] = np.array(newlabels)
res['bp_hets'] = np.asarray(hets)

xdim = [rng.standard_normal((60, 100)).astype(np.float32) for _ in range(n_samples)]
pseudobulk = np.abs(rng.standard_normal((n_samples, 300)).astype(np.float32))
semis = [np.abs(rng.standard_normal((60, 300)).astype(np.float32)) for _ in range(n_samples)]
het, unc, up = timed(
    'cluster_heterogeneity',
    lambda: compute_cluster_heterogeneity(
        cluster_number=1, reduced_bulk=reduced_bulk, representatives=representatives,
        cluster_labels=cluster_labels, xdim=xdim, pseudobulk=pseudobulk, semis=semis,
        discount_rate=1, semi_dis_rate=1))
res['ch_het'] = np.array([het])
res['ch_unc'] = np.asarray(unc, dtype='float64')
res['ch_up'] = np.array([up])

# ---- RRHO / hypergeometric tail ----
res['hypert'] = np.array([float(utils.hypert(6000, n1, n2, k))
                          for n1, n2, k in [(100, 100, 97), (500, 300, 12),
                                            (300, 40, 0), (250, 250, 250)]])
def _legacy_rrho(lista, listb, max_rank, population, upper):
    """The pristine checkout has no helper; replay its inline loop verbatim."""
    m = np.zeros((max_rank, max_rank))
    for rank1 in range(1, max_rank + 1):
        for rank2 in range(1, max_rank + 1):
            t1 = set(lista[:rank1])
            t2 = set(listb[:rank2])
            overlap = len(t1.intersection(t2))
            p = float(utils.hypert(population, rank1, rank2, overlap))
            if p < upper:
                p = upper
            m[rank1 - 1, rank2 - 1] = -np.log10(p)
    return m


la = ['G%d' % i for i in range(120)]
lb = ['G%d' % i for i in rng.permutation(200)[:120]]
_rrho_fn = getattr(utils, '_rrho_matrix', _legacy_rrho)
res['rrho'] = timed('rrho_matrix', lambda: _rrho_fn(la, lb, 60, 6000, 1e-50))

np.savez(out_path, **res)
with open(out_path + '.timings.pkl', 'wb') as fh:
    pickle.dump(timings, fh)
print('[[TIMINGS]]', {k: round(v, 3) for k, v in timings.items()})
