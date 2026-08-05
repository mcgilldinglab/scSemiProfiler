#!/bin/bash
# A/B this tree against the pre-refactor commit.
#
# Every stage is run twice from identical seeds -- once against an untouched
# checkout of the baseline commit, once against this tree -- and every produced
# file is compared value-for-value. Exits non-zero if anything differs.
#
#   ./run_equivalence.sh [workdir]
#
# Environment:
#   SCSEMI_PYTHON    interpreter to use (default: python)
#   SCSEMI_BASELINE  pristine checkout to compare against. If unset, a git
#                    worktree of SCSEMI_BASELINE_REF is created under workdir.
#   SCSEMI_BASELINE_REF  baseline commit (default: the branch point vs origin/main)
#   SCSEMI_DEVICE    torch device for the inference phase (default: cuda:0)
set -u

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
NEW="$(dirname "$HERE")"
WORK="${1:-${TMPDIR:-/tmp}/scsemi_equivalence}"
PY="${SCSEMI_PYTHON:-python}"

export CUBLAS_WORKSPACE_CONFIG=:4096:8
export PYTHONHASHSEED=0
export SCSEMI_DEVICE="${SCSEMI_DEVICE:-cuda:0}"

mkdir -p "$WORK"

# Resolve the pristine side. A worktree of the branch point is the default so
# the suite is self-contained for anyone who checks this branch out.
BASE="${SCSEMI_BASELINE:-}"
if [ -z "$BASE" ]; then
    REF="${SCSEMI_BASELINE_REF:-$(git -C "$NEW" merge-base HEAD origin/main 2>/dev/null)}"
    if [ -z "$REF" ]; then
        echo "cannot determine the baseline commit; set SCSEMI_BASELINE_REF or SCSEMI_BASELINE" >&2
        exit 2
    fi
    BASE="$WORK/baseline"
    if [ ! -d "$BASE" ]; then
        git -C "$NEW" worktree add --detach "$BASE" "$REF" >/dev/null || exit 2
    fi
    echo "baseline: worktree of $REF at $BASE"
fi
if [ ! -d "$BASE/scSemiProfiler" ]; then
    echo "baseline checkout not found at $BASE" >&2
    exit 2
fi

# The pipeline reads genesets/ and example_data/ relative to the working
# directory, so the baseline checkout needs them too.
for d in genesets example_data; do
    [ -e "$BASE/$d" ] || ln -s "$NEW/$d" "$BASE/$d"
done

fail=0
banner() { echo; echo "=== $* ==="; }

# ---- unit level: pure functions, on inputs larger than the bundled example ----
banner "unit-level probe"
"$PY" "$HERE/unit_probe.py" "$BASE" "$WORK/unit_base.npz" || fail=1
"$PY" "$HERE/unit_probe.py" "$NEW"  "$WORK/unit_new.npz"  || fail=1
"$PY" - "$WORK/unit_base.npz" "$WORK/unit_new.npz" <<'EOF' || fail=1
import sys, numpy as np
a, b = np.load(sys.argv[1], allow_pickle=True), np.load(sys.argv[2], allow_pickle=True)
bad = [k for k in a.files
       if not (list(a[k]) == list(b[k]) if a[k].dtype == object
               else (a[k].dtype == b[k].dtype and np.array_equal(a[k], b[k])))]
print('unit arrays compared: %d' % len(a.files))
print('MISMATCH: %s' % bad if bad else 'ALL UNIT RESULTS IDENTICAL')
sys.exit(1 if bad else 0)
EOF

# ---- pipeline level ----
for phase_name in "init,process" "infer" "select"; do
    banner "pipeline: $phase_name"
    tag=$(echo "$phase_name" | tr ',' '_')
    b="$WORK/base_$tag"; n="$WORK/new_$tag"
    rm -rf "$b" "$n"
    case "$phase_name" in
        init,process) : ;;                                   # starts from scratch
        infer)   cp -r "$WORK/base_init_process" "$b"; cp -r "$WORK/new_init_process" "$n" ;;
        select)  cp -r "$WORK/base_infer" "$b";        cp -r "$WORK/new_infer" "$n" ;;
    esac
    extra=""
    [ "$phase_name" = "infer" ] && extra="--fast"
    "$PY" "$HERE/run_pipeline.py" "$BASE" "$b" "$phase_name" $extra || fail=1
    "$PY" "$HERE/run_pipeline.py" "$NEW"  "$n" "$phase_name" $extra || fail=1
    "$PY" "$HERE/compare_outputs.py" "$b" "$n" || fail=1
done

banner "result"
if [ "$fail" -eq 0 ]; then
    echo "PASS: refactored tree reproduces baseline output exactly"
else
    echo "FAIL: differences found (see above)"
fi
exit "$fail"
