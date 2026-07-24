#!/bin/bash
# Slim archive equivalence: --adjoint-slim=true stores dynamic state only
# (skipping the static Eclipse schedule reconstructed at replay), so it must
# produce BYTE-IDENTICAL gradients to the full snapshot while being smaller.
# Run on SPE1_GRPCTRL (group control + guide rates) — the state most likely to
# be lost if the slim format dropped something needed.
#
# usage: run-adjoint-slim-test.sh <flow_adjoint> <deck.DATA> <outdir>
set -u
FLOW=${1:?flow_adjoint}
DECK=${2:?deck}
OUT=${3:?outdir}
S="--threads-per-process=1 --enable-storage-cache=false \
 --enable-adaptive-time-stepping=false \
 --tolerance-cnv=1e-9 --tolerance-mb=1e-11 --adjoint-objective=pressure-average"
NAME=$(basename "$DECK" .DATA)

run() { # <dir> <slim 0|1>
    local d=$1 slim=$2
    rm -rf "$d"; mkdir -p "$d"
    $FLOW "$DECK" --output-dir="$d" $S --adjoint-file="$d/a" \
        --adjoint-save=true --adjoint-slim="$slim" > "$d/f.log" 2>&1
    $FLOW "$DECK" --output-dir="$d" $S --adjoint-file="$d/a" \
        --adjoint-mode=gradient --adjoint-gradient-method=general-ad > "$d/g.log" 2>&1
}
run "$OUT/full" 0
run "$OUT/slim" 1

status=0
for fam in PV TRANS PERM WELLCTRL_BYMODE; do
    f="$OUT/full/$NAME.ADJOINT_GRADIENTS_$fam.txt"
    s="$OUT/slim/$NAME.ADJOINT_GRADIENTS_$fam.txt"
    [ -s "$f" ] || continue
    if diff -q "$f" "$s" > /dev/null 2>&1; then
        echo "$fam: slim == full (byte-identical)"
    else
        echo "FAIL: $fam slim differs from full"; status=1
    fi
done

fullB=$(du -sk "$OUT/full/a" | cut -f1)
slimB=$(du -sk "$OUT/slim/a" | cut -f1)
echo "archive: full ${fullB} KB, slim ${slimB} KB"
if [ "$slimB" -ge "$fullB" ]; then
    echo "FAIL: slim archive not smaller than full"; status=1
fi

[ $status -eq 0 ] && echo "slim archive equivalence: PASS"
exit $status
