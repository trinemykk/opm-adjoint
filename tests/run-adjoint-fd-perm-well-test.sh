#!/bin/bash
# dCTF/dperm well-coupling term (B1): on a defaulted-CTF deck the perm gradient
# at a PERFORATED cell must now match the finite difference (which includes the
# well-index/Peaceman contribution). Before B1 this cell was ~95 % off.
#
# Perturbs PERMX by a single-cell MULTIPLY at the producer/injector perforation
# and at a non-perforated control cell, and asserts adjoint == FD.
#
# usage: run-adjoint-fd-perm-well-test.sh <flow_adjoint> <spe1_dir> <outdir> [reltol]
set -u
FLOW=${1:?flow_adjoint}
SPE1DIR=${2:?spe1 dir}
OUT=${3:?outdir}
RELTOL=${4:-2e-2}
DECK=SPE1CASE1
MD_TO_SI=9.869233e-16
H=1e-2
mkdir -p "$OUT"

STRICT="--threads-per-process=1 --enable-storage-cache=false \
 --enable-adaptive-time-stepping=false \
 --tolerance-cnv=1e-9 --tolerance-cnv-relaxed=1e-9 \
 --tolerance-mb=1e-11 --tolerance-mb-relaxed=1e-11 \
 --adjoint-objective=pressure-average"

# name I J K cart-index
CELLS="prod_perf 10 10 3 299
control 5 5 2 144"

make_deck(){ # <dst> <I> <J> <K> <mult>
    mkdir -p "$1"
    python3 - "$SPE1DIR/$DECK.DATA" "$1/$DECK.DATA" "$2" "$3" "$4" "$5" << 'EOF'
import sys
src,dst,I,J,K,mult = sys.argv[1],sys.argv[2],*sys.argv[3:6],sys.argv[6]
t=open(src).read()
mp=f"MULTIPLY\n  'PERMX' {mult} {I} {I} {J} {J} {K} {K} /\n/\n"
assert '\nPROPS' in t
i=t.index('\nPROPS'); t=t[:i]+'\n'+mp+t[i:]
open(dst,'w').write(t)
EOF
}
getJ(){ grep -o "Adjoint objective J = [0-9eE.+-]*" "$1"|awk '{print $5}'; }

# base adjoint gradient
mkdir -p "$OUT/base"; make_deck "$OUT/base/d" 1 1 1 1.0   # harmless no-op multiply
$FLOW "$OUT/base/d/$DECK.DATA" --output-dir="$OUT/base" $STRICT --adjoint-save=true >/dev/null 2>&1
$FLOW "$OUT/base/d/$DECK.DATA" --output-dir="$OUT/base" $STRICT --adjoint-mode=gradient \
    --adjoint-gradient-method=general-ad >/dev/null 2>&1
GRAD="$OUT/base/$DECK.ADJOINT_GRADIENTS_PERM.txt"
[ -s "$GRAD" ] || { echo "FAIL: no gradient file"; exit 1; }

status=0
echo "$CELLS" | while read -r name I J K idx; do
    for s in p m; do
        [ "$s" = p ] && mult=$(python3 -c "print(1+$H)") || mult=$(python3 -c "print(1-$H)")
        make_deck "$OUT/${name}_$s/d" "$I" "$J" "$K" "$mult"
        $FLOW "$OUT/${name}_$s/d/$DECK.DATA" --output-dir="$OUT/${name}_$s" $STRICT --adjoint-save=true >/dev/null 2>&1
        $FLOW "$OUT/${name}_$s/d/$DECK.DATA" --output-dir="$OUT/${name}_$s" $STRICT --adjoint-mode=objective >"$OUT/${name}_$s.log" 2>&1
        [ "$s" = p ] && JP=$(getJ "$OUT/${name}_p.log") || JM=$(getJ "$OUT/${name}_m.log")
    done
    KMD=$(python3 -c "print({1:500,2:50,3:200}[$K])")
    read rel ok << EOF
$(python3 -c "
fd=($JP-$JM)/(2*$H*$KMD)
adj=float(open('$GRAD').read().splitlines()[1+$idx].split()[0])*$MD_TO_SI
rel=abs(fd-adj)/max(abs(fd),abs(adj),1e-300)
print(rel, 1 if rel < $RELTOL else 0)")
EOF
    printf "cell %-10s (%d): rel.err %.2e  %s\n" "$name" "$idx" "$rel" \
        "$([ "$ok" = 1 ] && echo PASS || echo FAIL)"
    [ "$ok" = 1 ] || echo "PERMWELL_FAIL"
done | tee "$OUT/summary.txt"

grep -q PERMWELL_FAIL "$OUT/summary.txt" && { echo "dCTF/dperm well term: FAIL"; exit 1; }
echo "dCTF/dperm well term: PASS"
