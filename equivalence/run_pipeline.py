"""Run the scSemiProfiler example pipeline from a given package root.

Used to A/B a pristine checkout against the refactored one: same seeds, same
inputs, outputs written to an isolated directory that can be diffed byte-wise.

usage: run_pipeline.py <pkg_root> <out_dir> <phases> [--fast]
  phases: comma list of init,process,infer,select
"""
import os
import sys
import json
import time

pkg_root = os.path.abspath(sys.argv[1])
out_dir = os.path.abspath(sys.argv[2])
phases = sys.argv[3].split(',')
fast = '--fast' in sys.argv

os.chdir(pkg_root)
sys.path.insert(0, pkg_root)

import numpy as np
import torch
import scvi


def reseed():
    """Reset every RNG the pipeline can touch, so each phase starts identically."""
    scvi.settings.seed = 0
    np.random.seed(0)
    torch.manual_seed(0)
    torch.cuda.manual_seed_all(0)


import scSemiProfiler as semi

assert os.path.dirname(os.path.abspath(semi.__file__)) == os.path.join(pkg_root, 'scSemiProfiler'), \
    'imported the wrong scSemiProfiler: %s' % semi.__file__

timings = {}


def phase(fn_name, fn):
    reseed()
    t0 = time.perf_counter()
    fn()
    dt = time.perf_counter() - t0
    timings[fn_name] = dt
    print('[[TIMING]] %s %.3f' % (fn_name, dt), flush=True)


device = os.environ.get('SCSEMI_DEVICE', 'cuda:0')

if 'init' in phases:
    phase('initsetup', lambda: semi.initsetup(
        out_dir, os.path.join(pkg_root, 'example_data/bulkdata.h5ad'),
        logged=False, normed=True, geneselection=False, batch=2))
    phase('get_eg_representatives', lambda: semi.get_eg_representatives(out_dir))

if 'process' in phases:
    phase('scprocess', lambda: semi.scprocess(
        name=out_dir, singlecell=out_dir + '/representative_sc.h5ad',
        normed=True, logged=False, cellfilter=False, threshold=1e-3,
        geneset=True, weight=0.5, k=15))

if 'infer' in phases:
    kw = {}
    if fast:
        kw = dict(pretrain1vae=3, pretrain1gan=2, pretrain2vae=3, pretrain2gan=2,
                  inferepochs=3)
    phase('scinfer', lambda: semi.scinfer(
        out_dir, out_dir + '/status/init_representatives.txt',
        out_dir + '/status/init_cluster_labels.txt', bulktype='pseudobulk',
        device=device, **kw))

if 'tgt' in phases:
    # Full default epoch budget for one target: 100+300 pretrain1, 50+150
    # pretrain2, 5 x 150 inference. This is the per-target cost of a real run.
    phase('tgtinfer', lambda: semi.tgtinfer(
        name=out_dir, representative=6, target=1, bulktype='pseudobulk',
        device=device))

if 'select' in phases:
    phase('activeselection', lambda: semi.activeselection(
        out_dir, out_dir + '/status/init_representatives.txt',
        out_dir + '/status/init_cluster_labels.txt',
        batch=2, lambdasc=1.0, lambdapb=1.0))

with open(os.path.join(out_dir, '_timings.json'), 'w') as fh:
    json.dump(timings, fh, indent=2)
print('[[DONE]]', json.dumps(timings), flush=True)
