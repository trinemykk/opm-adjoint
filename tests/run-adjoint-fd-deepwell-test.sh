#!/bin/bash
# The StandardWell frozen-hydrostatic-head gradient bias, and its removal by the
# multisegment-well conversion, on a controlled deck (opm-tests DEEPWELL):
# dead oil + dry gas + water (no DISGAS/DRSDT/VAPPARS/hysteresis), single-well
# control (no groups), a free-gas cap making the wellbore density strongly
# state-dependent, and a producer perforating a ~95 m column. The ONLY
# explicitly-updated quantity is the well-side frozen head (+F0+B).
#
# Uniform PORO perturbation (pvmult 1 +- eps); objective bhp:PROD. Compare the
# adjoint sum_i g_pvmult_i to central FD (J+ - J-)/(2 eps), for the StandardWell
# and the converted-MSW models. Assert: the MSW gradient FD-matches (< MSW_TOL),
# and the StandardWell gradient is clearly worse (its bias is real), i.e.
# err_stw >= FACTOR * err_msw.
#
# usage: run-adjoint-fd-deepwell-test.sh <flow_adjoint> <DEEPWELL.DATA> <outdir>
set -u
FLOW=${1:?flow_adjoint}
DECK=${2:?DEEPWELL.DATA}
OUT=${3:?outdir}
EPS=${4:-1e-5}
MSW_TOL=${5:-2e-2}
FACTOR=${6:-3.0}
rm -rf "$OUT"; mkdir -p "$OUT"

STRICT="--threads-per-process=1 --enable-storage-cache=false \
 --enable-adaptive-time-stepping=false --check-satfunc-consistency=false \
 --tolerance-cnv=1e-9 --tolerance-cnv-relaxed=1e-9 \
 --tolerance-mb=1e-11 --tolerance-mb-relaxed=1e-11"

# base/plus/minus decks, PORO scaled uniformly
python3 - "$DECK" "$OUT" "$EPS" << 'EOF'
import sys, os
src, out, eps = sys.argv[1], sys.argv[2], float(sys.argv[3])
text = open(src).read()
assert '180*0.2' in text, "expected inline '180*0.2' PORO in DEEPWELL.DATA"
for tag, sgn in (('base', 0.0), ('plus', eps), ('minus', -eps)):
    val = 0.2 * (1.0 + sgn)
    os.makedirs(f'{out}/deck_{tag}', exist_ok=True)
    open(f'{out}/deck_{tag}/DEEPWELL.DATA', 'w').write(
        text.replace('180*0.2', f'180*{val:.16f}'))
EOF

getJ() { grep -o "Adjoint objective J = [0-9eE.+-]*" "$1" | awk '{print $5}'; }

# run one model (extra flags in $2) and return |FD-adjoint|/|adjoint|
run_model() {
    local tag="$1"; local extra="$2"
    for s in base plus minus; do
        mkdir -p "$OUT/${tag}_$s"
        $FLOW "$OUT/deck_$s/DEEPWELL.DATA" --output-dir="$OUT/${tag}_$s" $STRICT $extra \
            --adjoint-save=true --adjoint-file="$OUT/${tag}_$s/a" \
            --adjoint-objective=bhp:PROD > "$OUT/${tag}_$s/rec.log" 2>&1
    done
    $FLOW "$OUT/deck_base/DEEPWELL.DATA" --output-dir="$OUT/${tag}_base" $STRICT $extra \
        --adjoint-mode=gradient --adjoint-file="$OUT/${tag}_base/a" \
        --adjoint-gradient-method=general-ad --adjoint-objective=bhp:PROD \
        > "$OUT/${tag}_grad.log" 2>&1
    for s in plus minus; do
        $FLOW "$OUT/deck_$s/DEEPWELL.DATA" --output-dir="$OUT/${tag}_$s" $STRICT $extra \
            --adjoint-mode=objective --adjoint-file="$OUT/${tag}_$s/a" \
            --adjoint-objective=bhp:PROD > "$OUT/${tag}_obj_$s.log" 2>&1
    done
    local JP JM
    JP=$(getJ "$OUT/${tag}_obj_plus.log"); JM=$(getJ "$OUT/${tag}_obj_minus.log")
    python3 -c "
g=[float(l.split()[0]) for l in open('$OUT/${tag}_base/DEEPWELL.ADJOINT_GRADIENTS_PV.txt').read().splitlines()[1:]]
adj=sum(g); fd=($JP-$JM)/(2*$EPS)
print('%.10e' % (abs(fd-adj)/max(abs(fd),abs(adj),1e-300)))
"
}

err_stw=$(run_model stw "")
err_msw=$(run_model msw "--convert-to-multisegment-well=per-connection")
echo "StandardWell  bhp:PROD  rel.err = $err_stw"
echo "converted-MSW bhp:PROD  rel.err = $err_msw   (tol $MSW_TOL)"

python3 -c "
es,em=float('$err_stw'),float('$err_msw')
ok = (em < $MSW_TOL) and (es >= $FACTOR*em)
print('MSW is %.1fx more accurate than StandardWell' % (es/max(em,1e-300)))
import sys; sys.exit(0 if ok else 1)
"
rc=$?
[ $rc -eq 0 ] && echo "adjoint DEEPWELL frozen-head-fix test: PASS" \
             || echo "adjoint DEEPWELL frozen-head-fix test: FAIL"
exit $rc
