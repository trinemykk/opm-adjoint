#!/bin/bash
# Verification of the COUPLED-SYSTEM adjoint solve (--adjoint-linear-solver=
# system_cprt): the wells stay in the transposed system
#
#     [ A^T  B^T ] [ lambda_r ]   [ rhs ]
#     [ C^T  D^T ] [ nu_w     ] = [  0  ]
#
# instead of being folded through the Schur complement A -= C^T D^-1 B. The
# point of the exercise is that this runs WITHOUT
# --matrix-add-well-contributions, which the Schur path requires on any deck
# with multi-perforation wells.
#
# Both runs share one recorded forward archive, so the comparison isolates the
# linear solve: any difference is the solver, not the forward trajectory.
# The reference is system_umfpack - a direct factorization of the same
# flattened coupled system - which is the exact oracle for this path, the way
# umfpack is for the Schur-folded one.
#
# The first gradient run deliberately passes no --matrix-add-well-contributions
# at all, checking that flow_adjoint defaults it correctly for system solvers.
#
# Usage:
#   run-adjoint-system-cprt-test.sh <flow_adjoint> <deck.DATA> <output-dir> [rel-tol]
#
# METRIC=field (as for the other large-deck comparisons) normalizes the
# difference by the largest reference gradient instead of per entry, which is
# the meaningful measure once most entries are ~0.

set -u

FLOW=${1:?usage: $0 <flow_adjoint> <deck.DATA> <output-dir> [rel-tol]}
DECK=${2:?missing deck}
OUTDIR=${3:?missing output dir}
RELTOL=${4:-1e-9}

DECKNAME=$(basename "${DECK}" .DATA)
# Fixed time stepping: an adaptive sequence would differ between runs and
# swamp the solver comparison.
STRICT="--threads-per-process=1 --enable-storage-cache=false \
 --enable-adaptive-time-stepping=false \
 --tolerance-cnv=1e-7 --tolerance-cnv-relaxed=1e-7 \
 --tolerance-mb=1e-9 --tolerance-mb-relaxed=1e-9 \
 --adjoint-objective=pressure-average"

mkdir -p "${OUTDIR}/run"

echo "=== forward (recording, no well contributions in the matrix)"
${FLOW} "${DECK}" --output-dir="${OUTDIR}/run" ${STRICT} \
    --matrix-add-well-contributions=false --adjoint-save=true \
    > "${OUTDIR}/fwd.log" 2>&1 \
    || { echo "forward FAILED"; tail -20 "${OUTDIR}/fwd.log"; exit 1; }

status=0
for solver in system_umfpack system_cprt; do
    # system_cprt runs without the flag on purpose: flow_adjoint must default
    # --matrix-add-well-contributions to false for coupled-system solvers.
    extra="--matrix-add-well-contributions=false"
    if [ "${solver}" = "system_cprt" ]; then
        extra=""
    fi
    ${FLOW} "${DECK}" --output-dir="${OUTDIR}/run" ${STRICT} ${extra} \
        --adjoint-mode=gradient --adjoint-linear-solver="${solver}" \
        > "${OUTDIR}/grad_${solver}.log" 2>&1 \
        || { echo "${solver}: gradient run FAILED"
             tail -20 "${OUTDIR}/grad_${solver}.log"; status=1; continue; }
    for kind in PV TRANS; do
        cp "${OUTDIR}/run/${DECKNAME}.ADJOINT_GRADIENTS_${kind}.txt" \
           "${OUTDIR}/${kind}_${solver}.txt"
    done
    iters=$(grep -o "Adjoint linear solver.*avg [0-9.]*)" \
                 "${OUTDIR}/grad_${solver}.log" || true)
    echo "${solver}: OK ${iters:+(${iters})}"
done

if [ ${status} -ne 0 ]; then
    echo "adjoint coupled-system comparison: FAIL (a run did not complete)"
    exit 1
fi

for kind in PV TRANS; do
    result=$(METRIC="${METRIC:-entrywise}" python3 - \
                       "${OUTDIR}/${kind}_system_umfpack.txt" \
                       "${OUTDIR}/${kind}_system_cprt.txt" "${RELTOL}" << 'EOF'
import os, sys
ref, test, tol = sys.argv[1], sys.argv[2], float(sys.argv[3])
metric = os.environ.get("METRIC", "entrywise")
def load(p):
    return [[float(v) for v in l.split()] for l in open(p)
            if not l.startswith('#')]
a, b = load(ref), load(test)
if metric == "field":
    scale = max((abs(x) for ra in a for x in ra), default=0.0) or 1e-30
    worst = max((abs(x - y)
                 for ra, rb in zip(a, b) for x, y in zip(ra, rb)),
                default=0.0) / scale
else:
    worst = max((abs(x - y) / max(abs(x), abs(y), 1e-30)
                 for ra, rb in zip(a, b) for x, y in zip(ra, rb)), default=0.0)
print(f"{worst:.3e}")
sys.exit(0 if worst <= tol and len(a) == len(b) else 1)
EOF
)
    if [ $? -eq 0 ]; then
        echo "system_cprt/${kind} vs system_umfpack: OK (max rel diff ${result})"
    else
        echo "system_cprt/${kind} vs system_umfpack: FAIL (max rel diff ${result})"
        status=1
    fi
done

if [ ${status} -eq 0 ]; then
    echo "adjoint coupled-system comparison: PASS (rel tol ${RELTOL})"
else
    echo "adjoint coupled-system comparison: FAIL"
fi
exit ${status}
