#!/bin/bash
# Smoke test for the simple standard-well -> multisegment-well converter
# (--convert-to-multisegment-well, opm-common makeMultiSegmentWellPerConnection).
#
# Converting a plain standard-well deck must produce working MSW wells: the run
# completes, no well is shut for non-convergence, and the field oil production
# reproduces the standard-well run closely (the only intended difference is the
# wellbore hydrostatic head, now solved implicitly).
#
# usage: run-adjoint-convert-msw-test.sh <flow_adjoint> <deck.DATA> <outdir>
set -u
FLOW=${1:?flow_adjoint}
DECK=${2:?deck}
OUT=${3:?outdir}
SUM="$(dirname "$FLOW")/../opm-common/bin/summary"
rm -rf "$OUT"; mkdir -p "$OUT/stw" "$OUT/msw"
COMMON="--threads-per-process=1 --enable-adaptive-time-stepping=false"
status=0

$FLOW "$DECK" --output-dir="$OUT/stw" $COMMON > "$OUT/stw.log" 2>&1
rc_stw=$?
$FLOW "$DECK" --output-dir="$OUT/msw" $COMMON --convert-to-multisegment-well=per-connection > "$OUT/msw.log" 2>&1
rc_msw=$?

if [ $rc_stw -ne 0 ] || [ $rc_msw -ne 0 ]; then
    echo "FAIL: run did not complete (stw=$rc_stw msw=$rc_msw)"; status=1
fi

shut=$(grep -c "will be shut" "$OUT/msw.log")
if [ "${shut:-0}" -eq 0 ]; then
    echo "converted MSW: no wells shut (PASS)"
else
    echo "FAIL: $shut converted MSW well(s) shut for non-convergence"; status=1
fi

# Compare a field oil production rate (FOPR) if the summary tool and vector are
# available (best-effort: a deck without the vector just skips this check).
if [ -x "$SUM" ]; then
    CS=$(ls "$OUT"/stw/*.UNSMRY 2>/dev/null | sed 's/.UNSMRY//' | head -1)
    CM=$(ls "$OUT"/msw/*.UNSMRY 2>/dev/null | sed 's/.UNSMRY//' | head -1)
    JS=$("$SUM" "$CS" FOPR 2>/dev/null | grep -E '[0-9]' | tail -1 | tr -d ' ')
    JM=$("$SUM" "$CM" FOPR 2>/dev/null | grep -E '[0-9]' | tail -1 | tr -d ' ')
    if [ -n "$JS" ] && [ -n "$JM" ]; then
        if python3 -c "
s,m=abs(float('$JS')),abs(float('$JM'))
rel=abs(s-m)/max(s,1e-30)
print('final FOPR stw=%.6e msw=%.6e rel.diff=%.2e' % (s,m,rel))
import sys; sys.exit(0 if rel < 2e-2 else 1)
"; then :; else
            echo "FAIL: converted-MSW FOPR differs > 2% from standard well"; status=1
        fi
    else
        echo "(FOPR not in summary; skipping production comparison)"
    fi
fi

[ $status -eq 0 ] && echo "adjoint convert-to-MSW smoke test: PASS" \
                  || echo "adjoint convert-to-MSW smoke test: FAIL"
exit $status
