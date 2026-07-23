#!/bin/bash
# --adjoint-config JSON: a config file must produce exactly the same gradient
# as the equivalent --adjoint-* flag string, and an explicit CLI flag must
# override its JSON counterpart.
#
# usage: run-adjoint-config-test.sh <flow_adjoint> <SPE1CASE1.DATA> <outdir>
set -u
FLOW=${1:?flow_adjoint}
DECK=${2:?SPE1CASE1.DATA}
OUT=${3:?outdir}
mkdir -p "$OUT/ref" "$OUT/flag" "$OUT/json"
S="--threads-per-process=1 --enable-storage-cache=false --enable-adaptive-time-stepping=false"

# reference summary (must live outside the gradient run dirs)
$FLOW "$DECK" --output-dir="$OUT/ref" $S > /dev/null 2>&1
REF="$OUT/ref/SPE1CASE1"

# JSON config: weighted, windowed, wildcard matchref via general-ad.
cat > "$OUT/cfg.json" << EOF
{
  "schema_version": 1,
  "gradient": { "method": "general-ad" },
  "objective": {
    "terms": [
      { "kind": "matchref", "wells": ["PROD*"], "phase": "gas",
        "reference_case": "$REF", "weight": 2.0, "window_days": [0, 1800] }
    ]
  }
}
EOF
CLIOBJ="matchsum:$REF:PROD.gas*2.0:t0=0:t1=1800"

grad() { # <dir> <extra args...>
    local d=$1; shift
    $FLOW "$DECK" --output-dir="$d" $S --adjoint-save=true "$@" > "$d/f.log" 2>&1
    $FLOW "$DECK" --output-dir="$d" $S --adjoint-mode=gradient "$@" > "$d/g.log" 2>&1
}
grad "$OUT/flag" --adjoint-objective="$CLIOBJ" --adjoint-gradient-method=general-ad
grad "$OUT/json" --adjoint-config="$OUT/cfg.json"

status=0
FPV="$OUT/flag/SPE1CASE1.ADJOINT_GRADIENTS_PV.txt"
JPV="$OUT/json/SPE1CASE1.ADJOINT_GRADIENTS_PV.txt"
if [ ! -s "$FPV" ] || [ ! -s "$JPV" ]; then
    echo "FAIL: a gradient run produced no PV file"; tail -3 "$OUT/json/g.log"; exit 1
fi
if diff -q "$FPV" "$JPV" > /dev/null; then
    echo "config == flags: identical PV gradient"
else
    echo "FAIL: JSON config gradient differs from the equivalent flag string"; status=1
fi

# CLI flag overrides the JSON objective: pressure-average must change J.
grad "$OUT/json" --adjoint-config="$OUT/cfg.json" --adjoint-objective=pressure-average
JMATCH=$(grep -oE 'J = [0-9eE.+-]+' "$OUT/json/g.log" | tail -1)
grad "$OUT/flag" --adjoint-objective=pressure-average
JPAVG=$(grep -oE 'J = [0-9eE.+-]+' "$OUT/flag/g.log" | tail -1)
if [ "$JMATCH" = "$JPAVG" ] && [ -n "$JPAVG" ]; then
    echo "CLI overrides JSON: pressure-average J matches ($JPAVG)"
else
    echo "FAIL: CLI --adjoint-objective did not override JSON (got '$JMATCH' vs '$JPAVG')"
    status=1
fi

[ $status -eq 0 ] && echo "adjoint --adjoint-config: PASS"
exit $status
