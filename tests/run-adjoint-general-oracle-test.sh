#!/bin/bash
# Oracle check for the general (stencil-local) gradient mechanism: the
# general method must reproduce the hand-derived analytic gradients (which
# are themselves FD-verified) component by component. Runs one forward
# recording, then a gradient sweep with each method, and diffs the per-family
# gradient files.
#
# Usage:
#   run-adjoint-general-oracle-test.sh <flow_adjoint> <deck> <outdir>
#                                      [reltol] [method]
# method: 'general' (finite difference, default) or 'general-ad' (seeded AD,
# which reproduces the analytic gradient to ~machine precision).

set -u

FLOW=${1:?usage: $0 <flow_adjoint> <deck> <outdir> [reltol] [method]}
DECK=${2:?missing deck}
OUTDIR=${3:?missing outdir}
RELTOL=${4:-1e-5}
METHOD=${5:-general}

CASE=$(basename "${DECK}" .DATA)
STRICT="--enable-storage-cache=false --enable-adaptive-time-stepping=false \
 --threads-per-process=1 --tolerance-cnv=1e-9 --tolerance-mb=1e-11 \
 --adjoint-objective=pressure-average"

mkdir -p "${OUTDIR}"

${FLOW} "${DECK}" --output-dir="${OUTDIR}" ${STRICT} \
    --adjoint-file="${OUTDIR}/a" --adjoint-save=true > "${OUTDIR}/fwd.log" 2>&1 \
    || { echo "forward failed"; tail -3 "${OUTDIR}/fwd.log"; exit 1; }

run_method() { # <method> <suffix>
    ${FLOW} "${DECK}" --output-dir="${OUTDIR}" ${STRICT} \
        --adjoint-file="${OUTDIR}/a" --adjoint-mode=gradient \
        --adjoint-gradient-method="$1" > "${OUTDIR}/grad_$2.log" 2>&1 \
        || { echo "gradient ($1) failed"; tail -3 "${OUTDIR}/grad_$2.log"; exit 1; }
    for kind in PV TRANS PERM; do
        f="${OUTDIR}/${CASE}.ADJOINT_GRADIENTS_${kind}.txt"
        [ -f "${f}" ] && cp "${f}" "${OUTDIR}/${kind}_$2.txt"
    done
}

run_method analytic an
run_method "${METHOD}" gen

status=0
for kind in PV TRANS PERM; do
    a="${OUTDIR}/${kind}_an.txt"
    g="${OUTDIR}/${kind}_gen.txt"
    [ -f "${a}" ] && [ -f "${g}" ] || continue
    out=$(python3 - "${a}" "${g}" "${RELTOL}" "${kind}" << 'EOF'
import sys
a, g, tol, kind = sys.argv[1], sys.argv[2], float(sys.argv[3]), sys.argv[4]
def vals(p):
    out = []
    for ln in open(p):
        s = ln.strip()
        if not s or s.startswith('#'):
            continue
        out.append(float(s.split()[-1]))   # last column is the gradient
    return out
va, vg = vals(a), vals(g)
# Infinity-norm relative error: max abs diff normalized by the family's
# largest magnitude. This is the right metric for comparing gradient
# vectors - a per-component relative diff is ill-conditioned on the many
# near-zero components (e.g. PERMY/PERMZ in a near-1D flow).
maxabs = max((abs(x) for x in va), default=0.0)
maxdiff = max((abs(x - y) for x, y in zip(va, vg)), default=0.0)
rel = maxdiff / maxabs if maxabs > 0 else 0.0
ok = rel < tol
print(f"{kind}: {len(va)} params, inf-norm rel diff {rel:.2e}  "
      f"{'PASS' if ok else 'FAIL'}")
sys.exit(0 if ok else 1)
EOF
)
    echo "${out}"
    echo "${out}" | grep -q PASS || status=1
done

if [ ${status} -eq 0 ]; then
    echo "general-vs-analytic oracle: PASS (rel tol ${RELTOL})"
else
    echo "general-vs-analytic oracle: FAIL"
fi
exit ${status}
