#!/bin/bash
# DRSDT cross-step term: root-cause regression for the bhp-objective
# FD-vs-adjoint deviation on SPE1CASE1.
#
# SPE1CASE1 has "DRSDT 0" (dissolved-gas may not increase). Where it binds it
# caps Rs_k against the step-(k-1) Rs — an explicit cross-step dependence the
# adjoint's reservoir Bdiag does not carry, so dJ/dm is approximate. This test
# demonstrates that by toggling DRSDT:
#
#   nodrs : DRSDT removed  -> the bhp dJ/dPORO gradient is EXACT (hard assert).
#           This is the positive regression: it proves the well/bhp adjoint
#           coupling is correct on a real DISGAS deck (the coverage gap that
#           the all-on-MODEL_1D_DEBUG FD suite left open).
#   base  : DRSDT 0 kept   -> a nonzero deviation is EXPECTED and printed
#           (~3% at cell 144); this is the documented gap (STATUS.md), to be
#           closed by adding the DRSDT cross-step term to Bdiag. If a future
#           fix makes this ~0 too, relax/repurpose this line.
#
# Strict tolerances are mandatory (an inexact Jacobian still lets forward
# Newton converge; a loose adjoint reduction hides the truth).
#
# usage: run-adjoint-drsdt-crossterm-test.sh <flow_adjoint> <SPE1CASE1.DATA> <outdir> [cell] [reltol]
set -u
FLOW=${1:?flow_adjoint}
SPE1=${2:?SPE1CASE1.DATA}
OUT=${3:?outdir}
CELL=${4:-144}
RELTOL=${5:-2e-3}
mkdir -p "$OUT"

STRICT="--threads-per-process=1 --enable-storage-cache=false \
 --enable-adaptive-time-stepping=false \
 --tolerance-cnv=1e-10 --tolerance-cnv-relaxed=1e-10 \
 --tolerance-mb=1e-12 --tolerance-mb-relaxed=1e-12 --adjoint-objective=bhp:PROD"

mkdeck(){ # <dst> <pmult> <keep|remove>
  python3 - "$SPE1" "$1" "$2" "$3" "$CELL" << 'EOF'
import sys
src,dst,pm,drs,idx = sys.argv[1],sys.argv[2],float(sys.argv[3]),sys.argv[4],int(sys.argv[5])
t = open(src).read()
poro=[0.3]*300; poro[idx]=0.3*pm
assert '300*0.3' in t, 'unexpected SPE1 PORO layout'
t = t.replace('300*0.3','\n'+'\n'.join(f'  {p:.16f}' for p in poro))
if drs=='remove':
    assert 'DRSDT\n 0 /\n' in t, 'unexpected DRSDT layout'
    t = t.replace('DRSDT\n 0 /\n','')
open(dst,'w').write(t)
EOF
}
getJ(){ grep -o "Adjoint objective J = [0-9eE.+-]*" "$1" | awk '{print $5}'; }

# prints the FD-vs-adjoint relative deviation at cell $CELL for a DRSDT setting
deviation(){ # <tag> <keep|remove>
  local tag=$1 drs=$2
  mkdir -p "$OUT/$tag/b"; mkdeck "$OUT/$tag/b/S.DATA" 1.0 "$drs"
  $FLOW "$OUT/$tag/b/S.DATA" --output-dir="$OUT/$tag/rb" $STRICT --adjoint-save=true >/dev/null 2>&1
  $FLOW "$OUT/$tag/b/S.DATA" --output-dir="$OUT/$tag/rb" $STRICT --adjoint-mode=gradient \
      --adjoint-gradient-method=general-ad --adjoint-linear-solver-reduction=1e-16 >/dev/null 2>&1
  local ADJ; ADJ=$(awk -v r=$((CELL+2)) 'NR==r{print $1}' "$OUT/$tag/rb/S.ADJOINT_GRADIENTS_PV.txt")
  local H=1e-2 JP JM pm
  for s in p m; do
    [ "$s" = p ] && pm=$(python3 -c "print(1+$H)") || pm=$(python3 -c "print(1-$H)")
    mkdir -p "$OUT/$tag/f_$s"; mkdeck "$OUT/$tag/f_$s/S.DATA" "$pm" "$drs"
    $FLOW "$OUT/$tag/f_$s/S.DATA" --output-dir="$OUT/$tag/rf_$s" $STRICT --adjoint-save=true >/dev/null 2>&1
    $FLOW "$OUT/$tag/f_$s/S.DATA" --output-dir="$OUT/$tag/rf_$s" $STRICT --adjoint-mode=objective \
        > "$OUT/$tag/o_$s.log" 2>&1
    [ "$s" = p ] && JP=$(getJ "$OUT/$tag/o_p.log") || JM=$(getJ "$OUT/$tag/o_m.log")
  done
  python3 -c "
fd=($JP-$JM)/(2*$H*0.3); adj=$ADJ/0.3
print(repr((fd-adj)/fd if fd else 0.0))"
}

echo "=== cell $CELL, objective bhp:PROD ==="
DEV_NODRS=$(deviation nodrs remove)
DEV_BASE=$(deviation base keep)
printf "no-DRSDT (Rs free)   rel.dev = %s   (must be < %s)\n" "$DEV_NODRS" "$RELTOL"
printf "DRSDT 0  (Rs capped) rel.dev = %s   (documented gap, expected ~0.029)\n" "$DEV_BASE"

status=0
python3 -c "import sys; sys.exit(0 if abs($DEV_NODRS) < $RELTOL else 1)" || {
    echo "FAIL: no-DRSDT bhp gradient is not exact (well/bhp adjoint coupling broken)"; status=1; }
python3 -c "import sys; sys.exit(0 if abs($DEV_BASE) > 5*$RELTOL else 1)" || {
    echo "NOTE: DRSDT-0 deviation shrank — the cross-step term may have been fixed;"
    echo "      if so, update this test to assert exactness for both cases."; }
[ $status -eq 0 ] && echo "DRSDT cross-term regression: PASS (no-DRSDT exact; DRSDT-0 gap present)"
exit $status
