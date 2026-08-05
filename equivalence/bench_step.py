"""Interleaved A/B of the training loop itself.

Wall-clock timing on a loaded box is dominated by contention, so this runs the
same fixed training workload alternately from both checkouts, several times, and
reports the MINIMUM per repeat -- the least-contended sample is the closest
estimate of the true cost.

usage: bench_step.py <proj_dir> <epochs> <repeats>
"""
import os
import subprocess
import sys
import json

proj = os.path.abspath(sys.argv[1])
epochs = sys.argv[2] if len(sys.argv) > 2 else '60'
repeats = int(sys.argv[3]) if len(sys.argv) > 3 else 3

HERE = os.path.dirname(os.path.abspath(__file__))
NEW = os.path.dirname(HERE)
BASE = os.environ.get('SCSEMI_BASELINE')
if not BASE:
    sys.exit('set SCSEMI_BASELINE to a pristine checkout of the baseline commit '
             '(run_equivalence.sh creates one as a git worktree)')
PYTHON = os.environ.get('SCSEMI_PYTHON', sys.executable)
ROOTS = {'baseline': BASE, 'refactored': NEW}

WORKER = r'''
import os, sys, time
pkg, proj, epochs = sys.argv[1], sys.argv[2], int(sys.argv[3])
os.chdir(pkg); sys.path.insert(0, pkg)
import numpy as np, torch, scvi
scvi.settings.seed = 0
from scSemiProfiler.inference import setdata
from scSemiProfiler.vaegan import fastgenerator
device = os.environ.get('SCSEMI_DEVICE', 'cuda:0')
sid = open(proj + '/sids.txt').read().split()[6]
adata, adj, variances, bulk, geneset_len = setdata(proj, sid, device, 15, 1.0)
model = fastgenerator(variances, None, geneset_len, adata,
                      n_hidden=256, n_latent=32, dropout_rate=0)
# warm up cudnn/allocator so the measured window is steady state
model.train(max_epochs=2, plan_kwargs={'lr':1e-3,'lr2':1e-3,'kappa':4.0}, use_gpu=device)
torch.cuda.synchronize()
t0 = time.perf_counter(); c0 = time.process_time()
model.train(max_epochs=epochs, plan_kwargs={'lr':1e-3,'lr2':1e-3,'kappa':4.0}, use_gpu=device)
torch.cuda.synchronize()
print('RESULT %.4f %.4f' % (time.perf_counter()-t0, time.process_time()-c0))
'''

wpath = os.path.join(HERE, '_bench_worker.py')
with open(wpath, 'w') as fh:
    fh.write(WORKER)

results = {k: [] for k in ROOTS}
for r in range(repeats):
    for name, root in ROOTS.items():
        env = dict(os.environ)
        env['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
        out = subprocess.run(
            [PYTHON, wpath, root, proj, epochs],
            capture_output=True, text=True, env=env, cwd=root)
        line = [l for l in out.stdout.split('\n') if l.startswith('RESULT')]
        if not line:
            print(name, 'FAILED'); print(out.stdout[-2000:]); print(out.stderr[-2000:]); sys.exit(1)
        wall, cpu = (float(x) for x in line[0].split()[1:])
        results[name].append((wall, cpu))
        print('  rep%d %-11s wall %7.2fs  cpu %7.2fs' % (r, name, wall, cpu), flush=True)

print()
print('%d epochs, best of %d (min = least-contended sample)' % (int(epochs), repeats))
for name in ROOTS:
    w = min(x[0] for x in results[name]); c = min(x[1] for x in results[name])
    print('  %-11s wall %7.2fs   cpu %7.2fs' % (name, w, c))
bw = min(x[0] for x in results['baseline']); nw = min(x[0] for x in results['refactored'])
bc = min(x[1] for x in results['baseline']); nc = min(x[1] for x in results['refactored'])
print('  speedup     wall %6.2fx   cpu %6.2fx' % (bw / nw, bc / nc))
