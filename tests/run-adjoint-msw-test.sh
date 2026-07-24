#!/bin/bash
# Simple multisegment-well (MSW) smoke test for the adjoint. Establishes the
# current, supported behaviour (and guards it):
#   1. an MSW deck records a forward run (per-substep snapshots) fine;
#   2. a RESERVOIR objective (pressure-average) yields gradients with MSW wells
#      present -- the reservoir-parameter path does not need the well-equation
#      derivative;
#   3. a WELL objective (bhp) is currently rejected: MSW well-objective
#      gradients are unsupported (StandardWell-only). This is the limitation the
#      --use-multisegment-well=always conversion + MSW well-adjoint (task #16)
#      will lift; until then the run must fail with a clear message (not a
#      silent wrong answer).
#
# usage: run-adjoint-msw-test.sh <flow_adjoint> <SPE1CASE2_MSW.DATA> <outdir>
set -u
FLOW=${1:?flow_adjoint}
DECK=${2:?SPE1CASE2_MSW.DATA}
OUT=${3:?outdir}
S="--threads-per-process=1 --enable-storage-cache=false --enable-adaptive-time-stepping=false"
rm -rf "$OUT"; mkdir -p "$OUT"
status=0

# 1. record
$FLOW "$DECK" --output-dir="$OUT" $S --adjoint-save=true --adjoint-file="$OUT/a" \
    --adjoint-objective=pressure-average > "$OUT/rec.log" 2>&1
nsub=$(grep -oi "[0-9]* substeps stored" "$OUT/rec.log" | grep -o "[0-9]*")
if [ "${nsub:-0}" -gt 0 ]; then echo "MSW record: $nsub substeps (PASS)"
else echo "FAIL: MSW record produced no substeps"; status=1; fi

# 2. reservoir objective works
$FLOW "$DECK" --output-dir="$OUT" $S --adjoint-mode=gradient --adjoint-file="$OUT/a" \
    --adjoint-objective=pressure-average > "$OUT/grad_pa.log" 2>&1
if [ $? -eq 0 ] && [ -s "$OUT"/*.ADJOINT_GRADIENTS_PV.txt ]; then
    echo "MSW + reservoir objective: PV gradients produced (PASS)"
else echo "FAIL: MSW + pressure-average did not produce reservoir gradients"; status=1; fi

# 3. well objective currently rejected with a clear message
$FLOW "$DECK" --output-dir="$OUT" $S --adjoint-mode=gradient --adjoint-file="$OUT/a" \
    --adjoint-objective=bhp:PROD > "$OUT/grad_bhp.log" 2>&1
rc=$?
if [ $rc -ne 0 ] && grep -q "StandardWells only" "$OUT/grad_bhp.log"; then
    echo "MSW + well objective: rejected with 'StandardWells only' (PASS, expected until #16)"
else
    echo "FAIL: MSW + bhp objective did not cleanly reject (rc=$rc)"; status=1
fi

[ $status -eq 0 ] && echo "adjoint MSW smoke test: PASS" || echo "adjoint MSW smoke test: FAIL"
exit $status
