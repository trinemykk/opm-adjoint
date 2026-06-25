#!/bin/bash
# Validation of the standalone AdjointResidualEvaluator (Step 1).
#
# Records a forward run, then replays it with --adjoint-residual-eval-check,
# which recomputes the residual one cell at a time with the residual-only
# evaluator and compares it to the production linearizer residual at the
# converged point of every substep. The evaluator passes when it reproduces
# linearize_cell + linearize_source_terms to the replay tolerance
# (bitwise: absolute differences at the 1e-14 machine-epsilon level).
#
# Usage:
#   run-adjoint-residual-eval-test.sh <flow_adjoint> <deck> <output-dir>

set -u

FLOW=${1:?usage: $0 <flow_adjoint> <deck> <output-dir>}
DECK=${2:?missing deck}
OUTDIR=${3:?missing output dir}

COMMON_ARGS="--threads-per-process=1 --enable-storage-cache=false"

mkdir -p "${OUTDIR}"

echo "=== forward run with adjoint recording"
${FLOW} "${DECK}" --output-dir="${OUTDIR}" ${COMMON_ARGS} \
    --adjoint-save=true > "${OUTDIR}/forward.log" 2>&1
status=$?
if [ ${status} -ne 0 ]; then
    echo "FAILURE: forward run exited with status ${status} (log: ${OUTDIR}/forward.log)"
    exit ${status}
fi

echo "=== backward replay with residual-evaluator check"
${FLOW} "${DECK}" --output-dir="${OUTDIR}" ${COMMON_ARGS} \
    --adjoint-mode=replay --adjoint-residual-eval-check=true \
    > "${OUTDIR}/replay.log" 2>&1
status=$?

grep -E "residual-evaluator check" "${OUTDIR}/replay.log"

if [ ${status} -eq 0 ] && grep -q "residual-evaluator check.*(bitwise)" "${OUTDIR}/replay.log"; then
    echo "residual-evaluator vs linearizeDomain: PASS (bitwise)"
    exit 0
else
    echo "residual-evaluator vs linearizeDomain: FAIL (see ${OUTDIR}/replay.log)"
    exit 1
fi
