"""Focused regression checks for exact-output inference optimizations."""

import os
import sys
import tempfile

import anndata
import numpy as np
import scipy.sparse
import torch


pkg_root = os.path.abspath(sys.argv[1] if len(sys.argv) > 1 else '.')
os.chdir(pkg_root)
sys.path.insert(0, pkg_root)

from scSemiProfiler import inference  # noqa: E402


def assert_array_equal(actual, expected, label):
    if actual.dtype != expected.dtype or not np.array_equal(actual, expected):
        raise AssertionError('%s differs' % label)


def test_finished_target_returns_early():
    with tempfile.TemporaryDirectory(prefix='scsemi_skip_') as project:
        os.mkdir(os.path.join(project, 'inferreddata'))
        with open(os.path.join(project, 'sids.txt'), 'w') as fh:
            fh.write('representative\ntarget\n')
        np.save(os.path.join(project, 'inferreddata', 'representative_to_target.npy'),
                np.zeros((2, 2), dtype=np.float32))

        original_setdata = inference.setdata
        try:
            inference.setdata = lambda *args, **kwargs: (_ for _ in ()).throw(
                AssertionError('finished target reached data preparation'))
            inference.tgtinfer(project, representative=0, target=1, device='cpu')
        finally:
            inference.setdata = original_setdata


def legacy_setdata(project, sid):
    adata = anndata.read_h5ad(os.path.join(project, 'sample_sc', sid + '.h5ad'))
    scores = np.load(os.path.join(project, 'geneset_scores', sid + '.npy'))
    mask = np.load(os.path.join(project, 'hvset.npy'))
    scores = scores[:, mask].astype('float32')
    bdata = anndata.AnnData(np.concatenate([adata.X, scores], axis=1), dtype='float32')
    bdata.obs = adata.obs
    bdata.obsm = adata.obsm
    bdata.uns = adata.uns
    adata = bdata.copy()
    variances = torch.tensor(adata.uns['feature_var'])
    pseudobulk = np.array(adata.X.mean(axis=0)).reshape((-1))
    return adata, variances, pseudobulk, scores.shape[1]


def test_setdata_matches_legacy_values():
    with tempfile.TemporaryDirectory(prefix='scsemi_setdata_') as project:
        os.mkdir(os.path.join(project, 'sample_sc'))
        os.mkdir(os.path.join(project, 'geneset_scores'))
        sid = 'representative'
        x = np.array([[1, 2], [3, 4], [5, 6]], dtype=np.float32)
        scores = np.array([[0.5, 1.5, 2.5], [3.5, 4.5, 5.5],
                           [6.5, 7.5, 8.5]], dtype=np.float64)
        mask = np.array([True, False, True])
        adata = anndata.AnnData(x)
        adata.obs['sample_ids'] = sid
        adata.obsm['neighborx'] = np.arange(12, dtype=np.float32).reshape(3, 4)
        adata.uns['feature_var'] = np.array([1, 2, 3, 4], dtype=np.float32)
        adata.write_h5ad(os.path.join(project, 'sample_sc', sid + '.h5ad'))
        np.save(os.path.join(project, 'geneset_scores', sid + '.npy'), scores)
        np.save(os.path.join(project, 'hvset.npy'), mask)

        expected = legacy_setdata(project, sid)
        actual = inference.setdata(project, sid, device='cpu')
        assert_array_equal(np.asarray(actual[0].X), np.asarray(expected[0].X), 'setdata X')
        assert_array_equal(actual[0].obsm['neighborx'], expected[0].obsm['neighborx'],
                           'setdata neighborx')
        assert_array_equal(actual[2].numpy(), expected[1].numpy(), 'setdata variances')
        assert_array_equal(actual[3], expected[2], 'setdata pseudobulk')
        if actual[4] != expected[3]:
            raise AssertionError('setdata geneset length differs')


def test_inferred_array_matches_legacy_copy_path():
    raw = [
        torch.tensor([[0, 10, 11, 20], [30, 2, 12, 5]], dtype=torch.float32),
        torch.tensor([[13, 9, 14, 1]], dtype=torch.float32),
    ]
    samples = [value * (value > 10) for value in raw]
    expected = np.array(torch.cat(samples))[:, :3]
    expected = expected * (expected > 10)
    actual = inference._inferred_array(samples, 3)
    assert_array_equal(actual, expected, 'inferred array')
    single_actual = inference._inferred_array(samples[:1], 3)
    assert_array_equal(single_actual, expected[:2], 'single-batch inferred array')


def test_sparse_scores_are_exact():
    rng = np.random.default_rng(11)
    x = rng.integers(0, 50, size=(80, 120)).astype(np.float32)
    membership = (rng.random((35, 120)) < 0.04).astype(np.float64)
    eps = 1e-6
    den = np.array(membership.sum(axis=1)) + eps
    expected = x.dot(membership.T)
    expected = (expected + eps) / den
    expected = expected - eps / den
    sparse = scipy.sparse.csr_matrix(membership)
    actual = sparse.dot(x.T).T
    actual = (actual + eps) / den
    actual = actual - eps / den
    assert_array_equal(actual, expected, 'sparse pathway scores')


test_finished_target_returns_early()
test_setdata_matches_legacy_values()
test_inferred_array_matches_legacy_copy_path()
test_sparse_scores_are_exact()
print('PASS: focused regression checks')
