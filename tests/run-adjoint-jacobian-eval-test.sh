#!/bin/bash
# Validation of the general AD assembly (Step A): the standalone evaluator
# must reproduce the production linearizer's JACOBIAN (the forward
# "analytic derivative" dResidual/dstate, before the well Schur complement),
# not just the residual.
#
# Records a forward run, then replays with --adjoint-residual-eval-check
# (which now checks both the residual and the Jacobian at every converged
# substep). Passes when the Jacobian matches the linearizer bitwise
# (absolute differences at the float-order ~1e-14 level).
#
# Usage:
#   run-adjoint-jacobian-eval-test.sh <flow_adjoint> <deck> <output-dir>

set -u

FLOW=${1:?usage: $0 <flow_adjoint> <deck> <output-dir>}
DECK=${2:?missing deck}
OUTDIR=${3:?missing output dir}

COMMON_ARGS="--threads-per-process=1 --enable-storage-cache=false"

mkdir -p "${OUTDIR}"

${FLOW} "${DECK}" --output-dir="${OUTDIR}" ${COMMON_ARGS} \
    --adjoint-save=true > "${OUTDIR}/forward.log" 2>&1
status=$?
if [ ${status} -ne 0 ]; then
    echo "FAILURE: forward run exited with status ${status}"
    exit ${status}
fi

${FLOW} "${DECK}" --output-dir="${OUTDIR}" ${COMMON_ARGS} \
    --adjoint-mode=replay --adjoint-residual-eval-check=true \
    > "${OUTDIR}/replay.log" 2>&1
status=$?

grep -E "Jacobian-evaluator check" "${OUTDIR}/replay.log"

if [ ${status} -eq 0 ] && grep -q "Jacobian-evaluator check.*(bitwise)" "${OUTDIR}/replay.log"; then
    echo "Jacobian-evaluator vs linearizeDomain: PASS (bitwise)"
    exit 0
else
    echo "Jacobian-evaluator vs linearizeDomain: FAIL (see ${OUTDIR}/replay.log)"
    exit 1
fi
