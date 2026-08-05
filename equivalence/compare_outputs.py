"""Byte/value-exact comparison of two scSemiProfiler project output trees.

usage: compare_outputs.py <dir_a> <dir_b>
Exit code 0 only if every file matches exactly.
"""
import os
import sys
import pickle

import numpy as np
import anndata
import torch
import pandas as pd

IGNORE = {'_timings.json'}

a_root, b_root = os.path.abspath(sys.argv[1]), os.path.abspath(sys.argv[2])
problems = []
checked = 0


def rel_files(root):
    out = set()
    for dirpath, _, names in os.walk(root):
        for n in names:
            p = os.path.relpath(os.path.join(dirpath, n), root)
            if os.path.basename(p) not in IGNORE:
                out.add(p)
    return out


def arrays_equal(x, y):
    x, y = np.asarray(x), np.asarray(y)
    if x.shape != y.shape:
        return False, 'shape %s vs %s' % (x.shape, y.shape)
    if x.dtype != y.dtype:
        return False, 'dtype %s vs %s' % (x.dtype, y.dtype)
    if x.dtype.kind in 'fc':
        eq = np.array_equal(x, y) or (np.isnan(x) == np.isnan(y)).all() and np.array_equal(
            x[~np.isnan(x)], y[~np.isnan(y)])
        if not eq:
            d = np.abs(x.astype('float64') - y.astype('float64'))
            return False, 'max|diff|=%.6g  n_diff=%d/%d' % (np.nanmax(d), int((x != y).sum()), x.size)
        return True, ''
    if not np.array_equal(x, y):
        return False, 'n_diff=%d/%d' % (int((x != y).sum()), x.size)
    return True, ''


def cmp_frames(da, db, label, rel):
    if list(da.columns) != list(db.columns):
        problems.append('%s: %s columns differ %s vs %s' % (rel, label, list(da.columns), list(db.columns)))
        return
    if not da.index.equals(db.index):
        problems.append('%s: %s index differs' % (rel, label))
        return
    for c in da.columns:
        va, vb = da[c], db[c]
        if isinstance(va.dtype, pd.CategoricalDtype) or va.dtype == object:
            if not (va.astype(str).values == vb.astype(str).values).all():
                problems.append('%s: %s[%s] differs' % (rel, label, c))
        else:
            ok, msg = arrays_equal(va.values, vb.values)
            if not ok:
                problems.append('%s: %s[%s] %s' % (rel, label, c, msg))


def cmp_h5ad(pa, pb, rel):
    A, B = anndata.read_h5ad(pa), anndata.read_h5ad(pb)
    xa = A.X.toarray() if hasattr(A.X, 'toarray') else np.asarray(A.X)
    xb = B.X.toarray() if hasattr(B.X, 'toarray') else np.asarray(B.X)
    ok, msg = arrays_equal(xa, xb)
    if not ok:
        problems.append('%s: X %s' % (rel, msg))
    cmp_frames(A.obs, B.obs, 'obs', rel)
    cmp_frames(A.var, B.var, 'var', rel)
    if set(A.obsm.keys()) != set(B.obsm.keys()):
        problems.append('%s: obsm keys %s vs %s' % (rel, sorted(A.obsm), sorted(B.obsm)))
    for k in set(A.obsm) & set(B.obsm):
        ok, msg = arrays_equal(A.obsm[k], B.obsm[k])
        if not ok:
            problems.append('%s: obsm[%s] %s' % (rel, k, msg))
    if set(A.uns.keys()) != set(B.uns.keys()):
        problems.append('%s: uns keys %s vs %s' % (rel, sorted(A.uns), sorted(B.uns)))
    for k in set(A.uns) & set(B.uns):
        cmp_uns(A.uns[k], B.uns[k], '%s: uns[%s]' % (rel, k))


def cmp_uns(va, vb, label):
    if isinstance(va, dict) or isinstance(vb, dict):
        if not (isinstance(va, dict) and isinstance(vb, dict)) or set(va) != set(vb):
            problems.append('%s: dict structure differs' % label)
            return
        for k in va:
            cmp_uns(va[k], vb[k], '%s[%s]' % (label, k))
    elif isinstance(va, (np.ndarray, list, tuple)):
        ok, msg = arrays_equal(va, vb)
        if not ok:
            problems.append('%s: %s' % (label, msg))
    elif va != vb:
        problems.append('%s: %r vs %r' % (label, va, vb))


def cmp_state_dict(pa, pb, rel):
    A = torch.load(pa, map_location='cpu')
    B = torch.load(pb, map_location='cpu')
    if set(A) != set(B):
        problems.append('%s: state_dict keys differ' % rel)
        return
    for k in A:
        ok, msg = arrays_equal(A[k].numpy(), B[k].numpy())
        if not ok:
            problems.append('%s: %s %s' % (rel, k, msg))


def cmp_pickle(pa, pb, rel):
    with open(pa, 'rb') as fh:
        A = pickle.load(fh)
    with open(pb, 'rb') as fh:
        B = pickle.load(fh)
    if set(A) != set(B):
        problems.append('%s: history keys differ' % rel)
        return
    for k in A:
        va, vb = A[k], B[k]
        va = va.values if hasattr(va, 'values') else np.asarray(va)
        vb = vb.values if hasattr(vb, 'values') else np.asarray(vb)
        ok, msg = arrays_equal(va, vb)
        if not ok:
            problems.append('%s: history[%s] %s' % (rel, k, msg))


fa, fb = rel_files(a_root), rel_files(b_root)
only_a, only_b = sorted(fa - fb), sorted(fb - fa)
for p in only_a:
    problems.append('only in A: %s' % p)
for p in only_b:
    problems.append('only in B: %s' % p)

for rel in sorted(fa & fb):
    pa, pb = os.path.join(a_root, rel), os.path.join(b_root, rel)
    checked += 1
    try:
        if rel.endswith('.npy'):
            ok, msg = arrays_equal(np.load(pa, allow_pickle=True), np.load(pb, allow_pickle=True))
            if not ok:
                problems.append('%s: %s' % (rel, msg))
        elif rel.endswith('.h5ad'):
            cmp_h5ad(pa, pb, rel)
        elif rel.endswith('.pkl'):
            cmp_pickle(pa, pb, rel)
        elif rel.endswith('.txt') or rel.endswith('.json'):
            with open(pa) as f1, open(pb) as f2:
                if f1.read() != f2.read():
                    problems.append('%s: text differs' % rel)
        else:
            cmp_state_dict(pa, pb, rel)
    except Exception as exc:  # noqa: BLE001
        problems.append('%s: comparison failed: %r' % (rel, exc))

print('compared %d files' % checked)
if problems:
    print('MISMATCH (%d)' % len(problems))
    for p in problems:
        print('  -', p)
    sys.exit(1)
print('ALL OUTPUTS IDENTICAL')
