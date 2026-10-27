#!/bin/bash
# DIAGNOSTIC (not wired into ctest): localizes the bhp-objective FD-vs-adjoint
# deviation. See STATUS.md "Well-objective (bhp) gradient deviation".
set -u
OPM_ROOT=${OPM_ROOT:-$(cd "$(dirname "$0")/../.." && pwd)}   # dir with the sibling repos
FLOW=${FLOW:-$OPM_ROOT/opm-adjoint/build/flow_adjoint}
SPE1=${SPE1:-$OPM_ROOT/opm-tests/spe1/SPE1CASE1.DATA}
OUT=$1; CELL=${2:-144}; OBJ=${3:-bhp:PROD}
mkdir -p $OUT
STRICT="--threads-per-process=1 --enable-storage-cache=false \
 --enable-adaptive-time-stepping=false \
 --tolerance-cnv=1e-10 --tolerance-cnv-relaxed=1e-10 \
 --tolerance-mb=1e-12 --tolerance-mb-relaxed=1e-12 --adjoint-objective=$OBJ"
mkdeck(){ python3 - "$SPE1" "$1" "$2" "$3" << 'EOF'
import sys
src,dst,idx,pm=sys.argv[1],sys.argv[2],int(sys.argv[3]),float(sys.argv[4])
t=open(src).read(); poro=[0.3]*300; poro[idx]=0.3*pm
open(dst,'w').write(t.replace('300*0.3','\n'+'\n'.join(f'  {p:.16f}' for p in poro)))
EOF
}
getJ(){ grep -o "Adjoint objective J = [0-9eE.+-]*" "$1"|awk '{print $5}'; }
mkdir -p $OUT/b; mkdeck $OUT/b/S.DATA $CELL 1.0
$FLOW $OUT/b/S.DATA --output-dir=$OUT/rb $STRICT --adjoint-save=true >/dev/null 2>&1
$FLOW $OUT/b/S.DATA --output-dir=$OUT/rb $STRICT --adjoint-mode=gradient \
  --adjoint-gradient-method=general-ad >/dev/null 2>&1
ADJ=$(awk -v r=$((CELL+2)) 'NR==r{print $1}' $OUT/rb/S.ADJOINT_GRADIENTS_PV.txt)
printf "cell %d obj %s  adjoint dJ/dpvmult = %s\n" $CELL "$OBJ" "$ADJ"
for H in 4e-2 2e-2 1e-2 5e-3 2e-3 1e-3; do
  for s in p m; do
    [ $s = p ] && pm=$(python3 -c "print(1+$H)") || pm=$(python3 -c "print(1-$H)")
    mkdir -p $OUT/f_${H}_$s; mkdeck $OUT/f_${H}_$s/S.DATA $CELL $pm
    $FLOW $OUT/f_${H}_$s/S.DATA --output-dir=$OUT/rf_${H}_$s $STRICT --adjoint-save=true >/dev/null 2>&1
    $FLOW $OUT/f_${H}_$s/S.DATA --output-dir=$OUT/rf_${H}_$s $STRICT --adjoint-mode=objective >$OUT/o_${H}_$s.log 2>&1
    [ $s = p ] && JP=$(getJ $OUT/o_${H}_p.log) || JM=$(getJ $OUT/o_${H}_m.log)
  done
  python3 -c "
fd=($JP-$JM)/(2*$H); adj=$ADJ
print(f'  h=$H  FD={fd:+.6e}  rel.dev={(fd-adj)/fd:+.4f}')"
done
