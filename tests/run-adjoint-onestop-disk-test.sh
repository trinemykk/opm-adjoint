#!/bin/bash
# One-stop with a DISK archive: --adjoint-mode=onestop --adjoint-file=<path>
# runs forward + backward sweep in a SINGLE process while streaming the tape to
# disk instead of holding it in RAM. Gradients and J must be BYTE-IDENTICAL to
# both the in-memory one-stop run and the two-process file-archive run, for the
# HDF5 and the directory-store backend alike.
#
# usage: run-adjoint-onestop-disk-test.sh <flow_adjoint> <SPE1CASE1.DATA> <outdir>
set -u
FLOW=${1:?flow_adjoint}
DECK=${2:?SPE1CASE1.DATA}
OUT=${3:?outdir}
NAME=$(basename "$DECK" .DATA)
S="--threads-per-process=1 --enable-storage-cache=false \
 --enable-adaptive-time-stepping=false --adjoint-objective=bhp:PROD"
rm -rf "$OUT/two" "$OUT/mem" "$OUT/h5" "$OUT/dir"
mkdir -p "$OUT/two" "$OUT/mem" "$OUT/h5" "$OUT/dir"

# two-process baseline (forward records to a file, second run sweeps)
$FLOW "$DECK" --output-dir="$OUT/two" $S --adjoint-file="$OUT/two/a" \
    --adjoint-save=true > "$OUT/two/f.log" 2>&1
$FLOW "$DECK" --output-dir="$OUT/two" $S --adjoint-file="$OUT/two/a" \
    --adjoint-mode=gradient > "$OUT/two/g.log" 2>&1

# one-stop, in-memory (no --adjoint-file)
$FLOW "$DECK" --output-dir="$OUT/mem" $S --adjoint-mode=onestop \
    > "$OUT/mem/g.log" 2>&1
[ $? -eq 0 ] || { echo "FAIL: one-stop in-memory exited nonzero"; tail -3 "$OUT/mem/g.log"; exit 1; }

status=0
# one-stop on disk, both backends: .h5 selects HDF5, anything else the
# dependency-free directory store.
run_disk() {  # <subdir> <archive path>
    local d=$1 path=$2
    $FLOW "$DECK" --output-dir="$OUT/$d" $S --adjoint-mode=onestop \
        --adjoint-file="$path" > "$OUT/$d/g.log" 2>&1
    local rc=$?
    [ $rc -eq 0 ] || { echo "FAIL: one-stop disk ($d) exited $rc"; tail -3 "$OUT/$d/g.log"; status=1; return; }
    [ -e "$path" ] || { echo "FAIL: one-stop disk ($d) wrote no archive at $path"; status=1; return; }
    echo "one-stop disk ($d): archive written to $path"
    for fam in PV TRANS PERM WELLCTRL_BYMODE; do
        local ref="$OUT/two/$NAME.ADJOINT_GRADIENTS_$fam.txt"
        local got="$OUT/$d/$NAME.ADJOINT_GRADIENTS_$fam.txt"
        [ -s "$ref" ] || continue
        if diff -q "$ref" "$got" > /dev/null 2>&1; then
            echo "  $fam: byte-identical to the two-process run"
        else
            echo "FAIL: $fam differs from the two-process run ($d)"; status=1
        fi
        local mem="$OUT/mem/$NAME.ADJOINT_GRADIENTS_$fam.txt"
        if ! diff -q "$mem" "$got" > /dev/null 2>&1; then
            echo "FAIL: $fam differs from the in-memory one-stop run ($d)"; status=1
        fi
    done
    # the objective must come out of the same sweep, so compare it too
    local jref jgot
    jref=$(grep -o '"value":[^},]*' "$OUT/two/$NAME.ADJOINT_GRADIENTS.json" | head -1)
    jgot=$(grep -o '"value":[^},]*' "$OUT/$d/$NAME.ADJOINT_GRADIENTS.json" | head -1)
    if [ -n "$jref" ] && [ "$jref" = "$jgot" ]; then
        echo "  objective J identical ($jgot)"
    else
        echo "FAIL: objective J differs ($d): '$jgot' vs '$jref'"; status=1
    fi
}

run_disk h5  "$OUT/h5/a.h5"
run_disk dir "$OUT/dir/a"

[ $status -eq 0 ] && echo "one-stop disk equivalence: PASS"
exit $status
