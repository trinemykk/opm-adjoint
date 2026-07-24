#!/bin/bash
# FD verification of the MultisegmentWell well-adjoint (task #16) on an MSW deck
# (SPE1CASE2_MSW: PROD & INJ are multisegment wells).
#
# MSW solves the segment pressures implicitly, so — unlike a StandardWell — the
# wellbore hydrostatic head is differentiable and the well-objective gradient is
# exact (this is the fix-free route for the frozen-head bias, task #24). We probe
# it with the bhp objective, which routes entirely through the well-equation
# adjoint (segment-system D^-T solve + B/C accumulation).
#
# Perturb all 300 PORO uniformly:  poro = 0.3*(1 +- eps)  ->  pvmult = 1 +- eps.
# A uniform direction avoids the random-direction DISGAS saturation-kink FD
# plateau, so the FD converges to the (eps-independent) adjoint value.
# Assert:  |FD - adjoint| / |adjoint|  <  TOL  for bhp:PROD.
#
# usage: run-adjoint-fd-msw-wellobj-test.sh <flow_adjoint> <SPE1CASE2_MSW.DATA> <outdir>
set -u
FLOW=${1:?flow_adjoint}
DECK=${2:?SPE1CASE2_MSW.DATA}
OUT=${3:?outdir}
EPS=${4:-1e-5}
TOL=${5:-1e-2}
rm -rf "$OUT"; mkdir -p "$OUT"

STRICT="--threads-per-process=1 --enable-storage-cache=false \
 --enable-adaptive-time-stepping=false \
 --tolerance-cnv=1e-9 --tolerance-cnv-relaxed=1e-9 \
 --tolerance-mb=1e-11 --tolerance-mb-relaxed=1e-11"

# base / plus / minus decks with uniformly scaled PORO
python3 - "$DECK" "$OUT" "$EPS" << 'EOF'
import sys, os
src, out, eps = sys.argv[1], sys.argv[2], float(sys.argv[3])
text = open(src).read()
assert '300*0.3' in text, "expected inline '300*0.3' PORO in the MSW deck"
for tag, sgn in (('base', 0.0), ('plus', eps), ('minus', -eps)):
    val = 0.3 * (1.0 + sgn)
    os.makedirs(f'{out}/deck_{tag}', exist_ok=True)
    open(f'{out}/deck_{tag}/SPE1CASE2_MSW.DATA', 'w').write(
        text.replace('300*0.3', f'300*{val:.16f}'))
EOF

for tag in base plus minus; do
    mkdir -p "$OUT/run_$tag"
    $FLOW "$OUT/deck_$tag/SPE1CASE2_MSW.DATA" --output-dir="$OUT/run_$tag" $STRICT \
        --adjoint-save=true --adjoint-file="$OUT/run_$tag/a" > "$OUT/run_$tag/fwd.log" 2>&1
done

getJ() { grep -o "Adjoint objective J = [0-9eE.+-]*" "$1" | awk '{print $5}'; }

OBJ=bhp:PROD
$FLOW "$OUT/deck_base/SPE1CASE2_MSW.DATA" --output-dir="$OUT/run_base" $STRICT \
    --adjoint-mode=gradient --adjoint-file="$OUT/run_base/a" \
    --adjoint-gradient-method=general-ad --adjoint-objective="$OBJ" \
    > "$OUT/grad.log" 2>&1
rc=$?
if [ $rc -ne 0 ] || grep -q "StandardWells only" "$OUT/grad.log"; then
    echo "FAIL: MSW bhp gradient did not run (rc=$rc)"; exit 1
fi

for s in plus minus; do
    $FLOW "$OUT/deck_$s/SPE1CASE2_MSW.DATA" --output-dir="$OUT/run_$s" $STRICT \
        --adjoint-mode=objective --adjoint-file="$OUT/run_$s/a" \
        --adjoint-objective="$OBJ" > "$OUT/obj_$s.log" 2>&1
done
JP=$(getJ "$OUT/obj_plus.log"); JM=$(getJ "$OUT/obj_minus.log")

python3 -c "
g = [float(l.split()[0]) for l in open('$OUT/run_base/SPE1CASE2_MSW.ADJOINT_GRADIENTS_PV.txt').read().splitlines()[1:]]
adj = sum(g)                      # uniform direction d_i = 1
fd  = ($JP - $JM) / (2*$EPS)
err = abs(fd-adj)/max(abs(fd), abs(adj), 1e-300)
print('MSW bhp:PROD  FD=%.6e  adjoint=%.6e  rel.err=%.2e (tol %.0e)' % (fd, adj, err, $TOL))
import sys; sys.exit(0 if err < $TOL else 1)
"
rc=$?
[ $rc -eq 0 ] && echo "adjoint MSW well-objective FD test: PASS" \
             || echo "adjoint MSW well-objective FD test: FAIL"
exit $rc
