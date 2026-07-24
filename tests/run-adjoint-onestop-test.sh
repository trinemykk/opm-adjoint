#!/bin/bash
# One-stop mode: --adjoint-mode=onestop runs the forward record and the
# backward sweep in a SINGLE process, sharing states through an in-memory
# archive (no files). It must give BYTE-IDENTICAL gradients to the standard
# two-process (file-archive) run, and write no state archive.
#
# usage: run-adjoint-onestop-test.sh <flow_adjoint> <SPE1CASE1.DATA> <outdir>
set -u
FLOW=${1:?flow_adjoint}
DECK=${2:?SPE1CASE1.DATA}
OUT=${3:?outdir}
NAME=$(basename "$DECK" .DATA)
S="--threads-per-process=1 --enable-storage-cache=false \
 --enable-adaptive-time-stepping=false --adjoint-objective=bhp:PROD"
rm -rf "$OUT/two" "$OUT/one"; mkdir -p "$OUT/two" "$OUT/one"

# two-process baseline (file archive)
$FLOW "$DECK" --output-dir="$OUT/two" $S --adjoint-file="$OUT/two/a" \
    --adjoint-save=true > "$OUT/two/f.log" 2>&1
$FLOW "$DECK" --output-dir="$OUT/two" $S --adjoint-file="$OUT/two/a" \
    --adjoint-mode=gradient > "$OUT/two/g.log" 2>&1

# one-stop (single invocation, in-memory)
$FLOW "$DECK" --output-dir="$OUT/one" $S --adjoint-mode=onestop > "$OUT/one/g.log" 2>&1
rc=$?

status=0
[ $rc -eq 0 ] || { echo "FAIL: one-stop exited $rc"; tail -3 "$OUT/one/g.log"; status=1; }

for fam in PV TRANS PERM WELLCTRL_BYMODE; do
    f="$OUT/two/$NAME.ADJOINT_GRADIENTS_$fam.txt"
    o="$OUT/one/$NAME.ADJOINT_GRADIENTS_$fam.txt"
    [ -s "$f" ] || continue
    if diff -q "$f" "$o" > /dev/null 2>&1; then
        echo "$fam: one-stop == two-process (byte-identical)"
    else
        echo "FAIL: $fam one-stop differs from two-process"; status=1
    fi
done

# one-stop must not write any per-substep state archive
if [ -e "$OUT/one/$NAME.ADJOINT.h5" ] || [ -d "$OUT/one/a" ]; then
    echo "FAIL: one-stop wrote a state archive (should be in-memory only)"; status=1
else
    echo "one-stop wrote no state archive (in-memory)"
fi

[ $status -eq 0 ] && echo "one-stop equivalence: PASS"
exit $status
