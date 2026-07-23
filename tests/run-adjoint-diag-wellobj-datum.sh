#!/bin/bash
# DIAGNOSTIC (not wired into ctest): localizes the bhp-objective FD-vs-adjoint
# deviation. See STATUS.md "Well-objective (bhp) gradient deviation".
# Discriminating experiment: does the connection hydrostatic head (the
# frozen explicit perf_pressure_diffs_) cause the bhp-objective deviation?
# Compare adjoint-vs-FD dJ/dPORO on a single NON-perforated cell (5,5,2)
# under bhp:PROD, for two decks:
#   datum=8400  -> dz=0  -> perf_pressure_diffs_ = 0 (no explicit head)
#   datum=8335  -> dz=65ft-> nonzero explicit head (cross-step term active)
# If the head term is the cause, the deviation should differ markedly.
set -u
FLOW=/Users/hnil/Documents/OPM/opm_clean/builds/release/opm-adjoint/flow_adjoint
SPE1=/Users/hnil/Documents/OPM/opm_clean/opm-tests/spe1/SPE1CASE1.DATA
OUT=${1:?outdir}
H=${2:-5e-3}
mkdir -p $OUT

STRICT="--threads-per-process=1 --enable-storage-cache=false \
 --enable-adaptive-time-stepping=false \
 --tolerance-cnv=1e-9 --tolerance-cnv-relaxed=1e-9 \
 --tolerance-mb=1e-11 --tolerance-mb-relaxed=1e-11 \
 --adjoint-objective=bhp:PROD"

# cell (5,5,2): cart idx = 4 + 4*10 + 1*100 = 144
IDX=144

make_deck() { # <dst> <datum> <poromult>  (poromult applied to cell 144 via MULTIPLY on PORO? no: PORO edit)
    local dst=$1 datum=$2 pmult=$3
    mkdir -p $dst
    python3 - "$SPE1" "$dst/SPE1CASE1.DATA" "$datum" "$pmult" << 'EOF'
import sys
src,dst,datum,pmult = sys.argv[1],sys.argv[2],sys.argv[3],float(sys.argv[4])
t = open(src).read()
# set PROD datum depth (WELSPECS item 5 = 8400)
t = t.replace("\t'PROD'\t'G1'\t10\t10\t8400\t'OIL' /",
              f"\t'PROD'\t'G1'\t10\t10\t{datum}\t'OIL' /")
# expand PORO to per-cell and scale cell 144
assert '300*0.3' in t
poro=[0.3]*300; poro[144]=0.3*pmult
t=t.replace('300*0.3','\n'+'\n'.join(f'  {p:.16f}' for p in poro))
open(dst,'w').write(t)
EOF
}

getJ(){ grep -o "Adjoint objective J = [0-9eE.+-]*" "$1" | awk '{print $5}'; }

run_case() { # <datum> <tag>
    local datum=$1 tag=$2
    make_deck $OUT/base_$tag $datum 1.0
    $FLOW $OUT/base_$tag/SPE1CASE1.DATA --output-dir=$OUT/rb_$tag $STRICT \
        --adjoint-save=true > $OUT/fwd_$tag.log 2>&1
    $FLOW $OUT/base_$tag/SPE1CASE1.DATA --output-dir=$OUT/rb_$tag $STRICT \
        --adjoint-mode=gradient --adjoint-gradient-method=general-ad \
        > $OUT/grad_$tag.log 2>&1
    # cell 144 is data row 145 (row 1 = header)
    ADJ=$(awk 'NR==146{print $1}' $OUT/rb_$tag/SPE1CASE1.ADJOINT_GRADIENTS_PV.txt)
    for s in p m; do
        [ $s = p ] && pm=$(python3 -c "print(1+$H)") || pm=$(python3 -c "print(1-$H)")
        make_deck $OUT/fd_${tag}_$s $datum $pm
        $FLOW $OUT/fd_${tag}_$s/SPE1CASE1.DATA --output-dir=$OUT/rfd_${tag}_$s $STRICT \
            --adjoint-save=true > /dev/null 2>&1
        $FLOW $OUT/fd_${tag}_$s/SPE1CASE1.DATA --output-dir=$OUT/rfd_${tag}_$s $STRICT \
            --adjoint-mode=objective > $OUT/obj_${tag}_$s.log 2>&1
        [ $s = p ] && JP=$(getJ $OUT/obj_${tag}_p.log) || JM=$(getJ $OUT/obj_${tag}_m.log)
    done
    python3 -c "
fd=($JP-$JM)/(2*$H*0.3)     # dJ/dporo (dporo=H*0.3)
adj=$ADJ/0.3                 # pvmult grad -> dJ/dporo
rel=(fd-adj)/fd if fd else float('nan')
print(f'datum=$datum ($tag): FD={fd:.6e} adj={adj:.6e} rel.dev={rel:+.4f}')
"
}

echo "cell (5,5,2) non-perforated, objective bhp:PROD"
run_case 8400 "dz0"    # perf_pressure_diffs_ = 0
run_case 8335 "dz65"   # nonzero explicit head
