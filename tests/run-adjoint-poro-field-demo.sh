#!/bin/bash
# A6 — PORO end-to-end field-like demo on SPE9 (9000 cells, 26 wells).
#
# Exercises the *practical field configuration* end to end and checks it against
# finite differences on a real, multi-well, DISGAS deck:
#   * a perturbed-PORO "truth" run produces a reference summary;
#   * the base run computes dJ/dPORO for ALL 9000 cells in ONE process via
#       --adjoint-mode=onestop  (forward record + backward sweep, in-memory)
#       --adjoint-slim=true      (dynamic-state-only snapshots)
#       --adjoint-config=<json>  (matchref objective, wildcard PRODU*, window)
#       general-AD gradients (the default) + cartesian-keyed CSV output;
#   * a handful of cells are FD spot-checked: central difference of J vs the
#     adjoint dJ/dPORO = (dJ/dpvmult)/poro read from the CSV.
# It also reports wall-clock and archive footprint — the "where to put the
# effort" signal at ~field scale.
#
# usage: run-adjoint-poro-field-demo.sh <flow_adjoint> <SPE9_CP_SHORT.DATA> <outdir>
#                                       [delta] [reltol] [amp]
set -u

FLOW=${1:?usage: $0 <flow_adjoint> <SPE9_CP_SHORT.DATA> <outdir> [delta] [reltol] [amp]}
DECK=${2:?missing SPE9_CP_SHORT.DATA}
OUT=${3:?missing outdir}
DELTA=${4:-2e-4}
RELTOL=${5:-3e-2}
AMP=${6:-0.08}
NTOP=${7:-6}                          # FD-check the NTOP cells of largest |grad|
HERE=$(cd "$(dirname "$0")" && pwd)
PY="$HERE/spe9_poro_field.py"
DECKDIR=$(cd "$(dirname "$DECK")" && pwd)
DECKNAME=$(basename "$DECK")
CASE=$(basename "$DECK" .DATA)

STRICT="--threads-per-process=1 --enable-storage-cache=false \
 --enable-adaptive-time-stepping=false \
 --tolerance-cnv=1e-8 --tolerance-cnv-relaxed=1e-8 \
 --tolerance-mb=1e-10 --tolerance-mb-relaxed=1e-10 \
 --linear-solver=cpr_trueimpes --linear-solver-reduction=1e-9"

rm -rf "$OUT"; mkdir -p "$OUT"

# ---- build a deck dir with an explicit flat PORO array (variant applied) ------
# $1 = destination dir, remaining args = spe9_poro_field.py subcommand for BASE.INC
# build_deck <dst> <subcmd> [extra args for the subcmd, after src/dst]
build_deck() {
    local dst=$1 sub=$2; shift 2
    mkdir -p "$dst"
    cp "$DECKDIR"/*.DATA "$DECKDIR"/*.INC "$DECKDIR"/*.GRDECL "$DECKDIR"/*.SCH \
       "$DECKDIR"/PERMVALUES.DATA "$DECKDIR"/TOPSVALUES.DATA "$dst"/ 2>/dev/null
    python3 "$PY" "$sub" "$DECKDIR/BASE.INC" "$dst/BASE.INC" "$@" \
        || { echo "poro field build failed"; exit 1; }
}
run_fwd() {  # <deckdir> <rundir> <extra args...>
    local dd=$1 rd=$2; shift 2
    mkdir -p "$rd"
    $FLOW "$dd/$DECKNAME" --output-dir="$rd" $STRICT "$@" > "$rd/run.log" 2>&1
}
get_J() { grep -o "Adjoint objective J = [0-9eE.+-]*" "$1" | tail -1 | awk '{print $5}'; }

echo "=== 1. truth run (PORO * smooth field, amp $AMP) -> reference summary"
# 'truth' takes the amp as its 2nd positional (before src/dst); pass via subcmd:
mkdir -p "$OUT/deck_truth"
cp "$DECKDIR"/*.DATA "$DECKDIR"/*.INC "$DECKDIR"/*.GRDECL "$DECKDIR"/*.SCH "$OUT/deck_truth"/ 2>/dev/null
python3 "$PY" truth "$DECKDIR/BASE.INC" "$OUT/deck_truth/BASE.INC" "$AMP" \
    || { echo "truth build failed"; exit 1; }
run_fwd "$OUT/deck_truth" "$OUT/truth" || { echo "truth run failed"; tail "$OUT/truth/run.log"; exit 1; }
# secure the reference summary OUTSIDE any adjoint output-dir (those truncate it)
mkdir -p "$OUT/ref"
cp "$OUT/truth/$CASE".SMSPEC "$OUT/truth/$CASE".UNSMRY "$OUT/ref"/ \
    || { echo "no reference summary"; exit 1; }
REF="$OUT/ref/$CASE"

echo "=== 2. base deck (flat PORO) + JSON config"
build_deck "$OUT/deck_base" expand
CFG="$OUT/adjoint.json"
cat > "$CFG" <<JSON
{
  "schema_version": 1,
  "mode": "onestop",
  "archive": { "slim": true },
  "gradient": { "method": "general-ad" },
  "objective": {
    "terms": [
      { "kind": "matchref", "phase": "gas", "wells": ["PRODU*"],
        "reference_case": "$REF", "weight": 1.0, "window_days": [0, 200] }
    ]
  }
}
JSON

echo "=== 3. adjoint one-stop (fwd record + backward sweep, in-memory)"
mkdir -p "$OUT/base"
T0=$(python3 -c 'import time;print(time.time())')
$FLOW "$OUT/deck_base/$DECKNAME" --output-dir="$OUT/base" $STRICT \
    --adjoint-config="$CFG" > "$OUT/base/onestop.log" 2>&1
RC=$?
T1=$(python3 -c 'import time;print(time.time())')
[ $RC -eq 0 ] || { echo "one-stop run failed (rc=$RC)"; tail -20 "$OUT/base/onestop.log"; exit 1; }
WALL=$(python3 -c "print(f'{$T1-$T0:.1f}')")
CSV="$OUT/base/$CASE.ADJOINT_GRADIENTS_PORO.csv"
JSON="$OUT/base/$CASE.ADJOINT_GRADIENTS.json"
[ -s "$CSV" ] || { echo "no PORO gradient CSV produced"; exit 1; }
J0=$(python3 -c "import json;print(json.load(open('$JSON'))['objective']['value'])")
NGRAD=$(awk -F, 'NR>1{n++} END{print n}' "$CSV")
echo "    one-stop wall-clock ${WALL}s, misfit J = ${J0}, dJ/dPORO for ${NGRAD} cells"

# FD is only trustworthy where the gradient carries real signal; a random cell
# may have a near-zero gradient (central-difference cancellation makes its
# relative error meaningless). So verify the NTOP cells of largest |gradient|.
FDCELLS=$(awk -F, 'NR>1{a=$2<0?-$2:$2; print a, $1}' "$CSV" | sort -rn | head -"$NTOP" | awk '{print $2}')
GMAX=$(awk -F, 'NR>1{a=$2<0?-$2:$2; if(a>m)m=a} END{print m}' "$CSV")
echo "=== 4. FD spot-check of the top $NTOP cells by |grad| (delta $DELTA, reltol $RELTOL)"
# The string form of the config objective; must match the JSON terms exactly.
OBJ="matchref:PRODU*:gas:$REF:w=1.0:t0=0:t1=200"
# J of a perturbed deck, computed by the simulator itself (definitionally
# consistent with the adjoint): record a slim forward, then replay it in
# objective-only mode (the replay does no linear solves, so it is cheap).
eval_J() {  # <deckdir> <rundir>  -> echoes J on stdout
    local dd=$1 rd=$2
    mkdir -p "$rd"
    # full archive for the objective replay (proven path; slim+objective-only
    # replay has a Schedule::getWell lookup bug — see STATUS.md follow-up).
    $FLOW "$dd/$DECKNAME" --output-dir="$rd" $STRICT \
        --adjoint-save=true --adjoint-file="$rd/a" \
        --adjoint-objective="$OBJ" > "$rd/rec.log" 2>&1 || return 1
    $FLOW "$dd/$DECKNAME" --output-dir="$rd" $STRICT --adjoint-mode=objective \
        --adjoint-file="$rd/a" --adjoint-objective="$OBJ" > "$rd/obj.log" 2>&1 || return 1
    get_J "$rd/obj.log"
}
status=0
printf "  %-7s %-14s %-14s %-10s %-6s\n" cell FD adjoint rel.err frac
for c in $FDCELLS; do
    poro0=$(python3 "$PY" poro "$DECKDIR/BASE.INC" "$c")
    JP=""; JM=""
    for s in + -; do
        d=$(python3 -c "print($DELTA if '$s'=='+' else -$DELTA)")
        build_deck "$OUT/fd_${c}_${s}" perturb "$c" "$d"
        J=$(eval_J "$OUT/fd_${c}_${s}" "$OUT/fd_${c}_${s}") \
            || { echo "FD run ${c}${s} failed"; tail -5 "$OUT/fd_${c}_${s}/rec.log" "$OUT/fd_${c}_${s}/obj.log" 2>/dev/null; exit 1; }
        [ "$s" = "+" ] && JP=$J || JM=$J
    done
    # adjoint dJ/dPORO[c] = (dJ/dpvmult[c]) / poro0[c], CSV keyed by cartesian idx
    read FD ADJ ERR FRAC OK <<EOF
$(python3 -c "
import csv
val=None
for r in csv.reader(open('$CSV')):
    if r and r[0].isdigit() and int(r[0])==$c: val=float(r[1]); break
fd=($JP-$JM)/(2*$DELTA)
adj=val/$poro0
err=abs(fd-adj)/max(abs(fd),abs(adj),1e-300)
print(fd,adj,err,abs(val)/$GMAX,1 if err<$RELTOL else 0)")
EOF
    printf "  %-7s %-14.6e %-14.6e %-10.2e %-6.2f %s\n" "$c" "$FD" "$ADJ" "$ERR" "$FRAC" \
        "$([ "$OK" = 1 ] && echo ok || echo FAIL)"
    [ "$OK" = 1 ] || status=1
done

echo "=== 5. footprint / effort signal"
# re-record slim to disk once, just to report the tape size at this scale
run_fwd "$OUT/deck_base" "$OUT/tape" --adjoint-save=true --adjoint-slim=true \
    --adjoint-file="$OUT/tape/a" --adjoint-objective="$OBJ" > /dev/null 2>&1
NSUB=$(grep -oi "[0-9]* substeps stored" "$OUT/tape/run.log" | grep -o "[0-9]*")
TAPE=$(du -sk "$OUT/tape/a" 2>/dev/null | awk '{print $1}')
python3 -c "
ncell=9000; nsub=int('${NSUB:-0}'); tape=float('${TAPE:-0}')
per=tape/nsub if nsub else 0
bpc=per*1024/ncell if ncell else 0
print(f'  cells {ncell}, substeps {nsub}, slim tape {tape/1024:.1f} MiB ({per:.0f} KiB/substep, {bpc:.0f} B/cell)')
print(f'  extrapolated slim tape @1e5 cells x1000 substeps: {bpc*1e5*1000/1e9:.0f} GB (-> disk); @1e6: {bpc*1e6*1000/1e9:.0f} GB')
"

[ $status -eq 0 ] && echo "SPE9 PORO field demo: PASS" || echo "SPE9 PORO field demo: FD MISMATCH"
exit $status
