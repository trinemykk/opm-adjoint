#!/bin/bash
# Random-direction derivative test for well objectives on SPE1CASE1.
# Perturb ALL 300 PORO values along a fixed random direction d:
#   poro_i = 0.3*(1 +- eps*d_i)  ->  pvmult_i = 1 +- eps*d_i
# FD (J+ - J-)/(2 eps) vs adjoint sum_i g_pvmult_i d_i, per objective.
# PROD switches ORAT->BHP at ~day 973; windows isolate single-mode periods.
set -eu
FLOW=/Users/hnil/Documents/OPM/opm_clean/builds/release/opm-adjoint/flow_adjoint
DECK=/Users/hnil/Documents/OPM/opm_clean/opm-tests/spe1/SPE1CASE1.DATA
OUT=${1:?outdir}
EPS=${2:-1e-3}
mkdir -p $OUT

STRICT="--threads-per-process=1 --enable-storage-cache=false \
 --enable-adaptive-time-stepping=false \
 --tolerance-cnv=1e-9 --tolerance-cnv-relaxed=1e-9 \
 --tolerance-mb=1e-11 --tolerance-mb-relaxed=1e-11"

# fixed random direction, and the three decks
python3 - "$DECK" "$OUT" "$EPS" << 'EOF'
import random, sys
src, out, eps = sys.argv[1], sys.argv[2], float(sys.argv[3])
random.seed(42)
d = [random.uniform(-1.0, 1.0) for _ in range(300)]
open(f'{out}/direction.txt', 'w').write('\n'.join(repr(x) for x in d))
text = open(src).read()
assert '300*0.3' in text
for tag, sgn in (('base', 0.0), ('plus', eps), ('minus', -eps)):
    poro = '\n'.join(f'  {0.3*(1.0+sgn*di):.16f}' for di in d)
    import os
    os.makedirs(f'{out}/deck_{tag}', exist_ok=True)
    open(f'{out}/deck_{tag}/SPE1CASE1.DATA', 'w').write(
        text.replace('300*0.3', '\n' + poro))
EOF

# three forwards (archives are objective-independent)
for tag in base plus minus; do
    mkdir -p $OUT/run_$tag
    $FLOW $OUT/deck_$tag/SPE1CASE1.DATA --output-dir=$OUT/run_$tag $STRICT \
        --adjoint-save=true > $OUT/run_$tag/fwd.log 2>&1
done

getJ() { grep -o "Adjoint objective J = [0-9eE.+-]*" "$1" | awk '{print $5}'; }

OBJECTIVES=(
  "bhp:PROD"
  "bhp:PROD:t0=0:t1=900"
  "rate:PROD:oil"
  "rate:PROD:oil:t0=1100"
  "rate:PROD:gas:t0=0:t1=900"
  "pressure-average"
)

printf "%-28s | %15s %15s | %9s\n" objective "FD g.d" "adjoint g.d" rel.err
for OBJ in "${OBJECTIVES[@]}"; do
    tag=$(echo "$OBJ" | tr ':=' '__')
    # adjoint gradient on base archive
    $FLOW $OUT/deck_base/SPE1CASE1.DATA --output-dir=$OUT/run_base $STRICT \
        --adjoint-mode=gradient --adjoint-gradient-method=general-ad \
        --adjoint-objective="$OBJ" > $OUT/grad_$tag.log 2>&1
    cp $OUT/run_base/SPE1CASE1.ADJOINT_GRADIENTS_PV.txt $OUT/pv_$tag.txt
    # J on perturbed archives
    for s in plus minus; do
        $FLOW $OUT/deck_$s/SPE1CASE1.DATA --output-dir=$OUT/run_$s $STRICT \
            --adjoint-mode=objective --adjoint-objective="$OBJ" \
            > $OUT/obj_${tag}_$s.log 2>&1
    done
    JP=$(getJ $OUT/obj_${tag}_plus.log); JM=$(getJ $OUT/obj_${tag}_minus.log)
    python3 -c "
d = [float(l) for l in open('$OUT/direction.txt')]
g = [float(l.split()[0]) for l in open('$OUT/pv_$tag.txt').read().splitlines()[1:]]
adj = sum(gi*di for gi, di in zip(g, d))
fd = ($JP - $JM) / (2*$EPS)
err = abs(fd-adj)/max(abs(fd), abs(adj), 1e-300)
print(f'%-28s | %15.6e %15.6e | %9.2e' % ('$OBJ', fd, adj, err))
"
done