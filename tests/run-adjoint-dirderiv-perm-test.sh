#!/bin/bash
# Larger-ish-case PERM gradient verification by directional FD on a real DISGAS
# deck (SPE1CASE1), isolated from the two known adjoint gaps so it does NOT
# depend on their fixes:
#   - DRSDT is stripped (the DRSDT-0 cross-step Rs cap; task #18) — otherwise
#     it contaminates the perm gradient direction-dependently.
#   - the directional-FD harness masks perforated cells (the dCTF/dK well term;
#     task B1) — the cell-based perm gradient there is knowingly incomplete.
# What remains is the reservoir perm gradient (half-trans chain rule), which is
# exact — this test guards it end-to-end at deck scale via random directions.
#
# usage: run-adjoint-dirderiv-perm-test.sh <flow_adjoint> <SPE1CASE1.DATA> <outdir> [eps] [reltol] [ndir]
set -u
FLOW=${1:?flow_adjoint}
SPE1=${2:?SPE1CASE1.DATA}
OUT=${3:?outdir}
EPS=${4:-1e-6}
RELTOL=${5:-5e-3}
NDIR=${6:-3}
HERE=$(cd "$(dirname "$0")" && pwd)
mkdir -p "$OUT/deck"

# copy the deck with DRSDT removed
python3 - "$SPE1" "$OUT/deck/SPE1CASE1.DATA" << 'EOF'
import sys
t = open(sys.argv[1]).read()
assert 'DRSDT\n 0 /\n' in t, 'unexpected DRSDT layout in SPE1CASE1'
open(sys.argv[2], 'w').write(t.replace('DRSDT\n 0 /\n', ''))
EOF

python3 "$HERE/run-adjoint-dirderiv-general.py" \
    "$FLOW" "$OUT/deck/SPE1CASE1.DATA" "$OUT/run" \
    perm pressure-average "$EPS" "$RELTOL" "$NDIR" 1
