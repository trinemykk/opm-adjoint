#!/bin/bash
# B0: quantify the missing dCTF/dK well term in dJ/dPERMX on SPE1CASE1
# (defaulted CTF -> Peaceman, perm-derived).
#
# For selected cells, central-difference J w.r.t. a PERMX multiplier
# (MULTIPLY box on the single cell) and compare with the adjoint
# dJ/dPERMX at that cell. On perforated cells the FD includes the
# dCTF/dK route; the adjoint (today) does not -> gap = well term.
#
# Measured 2026-07-23 (flow_adjoint @ ab5163d, general-ad, h=1e-2, cnv 1e-9):
#   objective=pressure-average:
#     prod_perf (10,10,3): FD -3.220e4  adjoint -1.433e3  -> gap = 95.6% of FD
#     inj_perf  (1,1,1):   FD -3.429e1  adjoint -3.406e1  -> gap =  0.7%
#     control   (5,5,2):   FD  6.800e1  adjoint  6.811e1  -> gap = -0.2% (noise)
#   objective=bhp:PROD: prod_perf gap 94.3%; note an additional ~4-6% systematic
#   FD-vs-adjoint deviation on ALL cells with this objective (control-switch
#   kinks: SPE1 PROD switches ORAT<->BHP; documented v1 approximation).
# => the dCTF/dK well term DOMINATES dJ/dperm at drawdown-limited producer
#    perforations; it is negligible for rate-controlled injectors. See STATUS.md.
#
# usage: run-adjoint-ctf-gap-quantify.sh <flow_adjoint> <spe1_dir> <outdir> [objective] [h_rel]

set -u
FLOW=${1:?flow_adjoint}
SPE1DIR=${2:?spe1 dir}
OUTDIR=${3:?outdir}
OBJECTIVE=${4:-bhp:PROD}
H=${5:-1e-3}

DECK=SPE1CASE1
MD_TO_SI=9.869233e-16

# cell list: "name I J K permX_mD"  (cartesian idx = (I-1)+(J-1)*10+(K-1)*100)
CELLS="prod_perf 10 10 3 200
inj_perf 1 1 1 500
control 5 5 2 50"

STRICT="--threads-per-process=1 --enable-storage-cache=false \
 --enable-adaptive-time-stepping=false \
 --tolerance-cnv=1e-9 --tolerance-cnv-relaxed=1e-9 \
 --tolerance-mb=1e-11 --tolerance-mb-relaxed=1e-11 \
 --adjoint-objective=${OBJECTIVE}"

mkdir -p "${OUTDIR}"

make_deck() { # <dstdir> <I> <J> <K> <mult>  - copy deck, MULTIPLY one cell's PERMX
    local dst=$1 I=$2 J=$3 K=$4 mult=$5
    mkdir -p "${dst}"
    python3 - "${SPE1DIR}/${DECK}.DATA" "${dst}/${DECK}.DATA" "$I" "$J" "$K" "$mult" << 'EOF'
import sys
src, dst, I, J, K, mult = sys.argv[1], sys.argv[2], *sys.argv[3:6], sys.argv[6]
lines = open(src).readlines()
out, done = [], False
for i, line in enumerate(lines):
    out.append(line)
    # insert right after the PERMZ record's closing data line (before ECHO)
    if not done and line.strip().startswith('ECHO'):
        out.insert(-1, f"MULTIPLY\n  'PERMX' {mult} {I} {I} {J} {J} {K} {K} /\n/\n\n")
        done = True
assert done, 'no ECHO anchor found in GRID section'
open(dst, 'w').writelines(out)
EOF
}

run_J() { # <deckdir> <rundir> -> J on stdout
    local deckdir=$1 rundir=$2
    mkdir -p "${rundir}"
    ${FLOW} "${deckdir}/${DECK}.DATA" --output-dir="${rundir}" ${STRICT} \
        --adjoint-save=true > "${rundir}/fwd.log" 2>&1 || { echo FAIL; return; }
    ${FLOW} "${deckdir}/${DECK}.DATA" --output-dir="${rundir}" ${STRICT} \
        --adjoint-mode=objective > "${rundir}/obj.log" 2>&1 || { echo FAIL; return; }
    grep -o "Adjoint objective J = [0-9eE.+-]*" "${rundir}/obj.log" | awk '{print $5}'
}

echo "=== base: forward + adjoint gradient (objective ${OBJECTIVE})"
BASE=${OUTDIR}/base
mkdir -p "${BASE}"
${FLOW} "${SPE1DIR}/${DECK}.DATA" --output-dir="${BASE}" ${STRICT} \
    --adjoint-save=true > "${BASE}/fwd.log" 2>&1 || { echo "base fwd failed"; exit 1; }
${FLOW} "${SPE1DIR}/${DECK}.DATA" --output-dir="${BASE}" ${STRICT} \
    --adjoint-mode=gradient --adjoint-gradient-method=general-ad \
    > "${BASE}/grad.log" 2>&1 || { echo "gradient failed"; exit 1; }
GRADFILE=${BASE}/${DECK}.ADJOINT_GRADIENTS_PERM.txt
[ -f "${GRADFILE}" ] || { echo "no ${GRADFILE}"; exit 1; }

echo ""
printf "%-10s %5s | %15s %15s | %12s %9s\n" cell cart "FD_dJ/dK[/mD]" "ADJ_dJ/dK[/mD]" "gap(FD-ADJ)" "gap/FD"
echo "${CELLS}" | while read -r name I J K KMD; do
    idx=$(( (I-1) + (J-1)*10 + (K-1)*100 ))
    for sign in p m; do
        [ "$sign" = p ] && mult=$(python3 -c "print(1+${H})") || mult=$(python3 -c "print(1-${H})")
        d=${OUTDIR}/fd_${name}_${sign}
        make_deck "${d}/deck" "$I" "$J" "$K" "$mult"
        J_out=$(run_J "${d}/deck" "${d}/run")
        [ "$J_out" = "FAIL" ] && { echo "FD run ${name} ${sign} FAILED (see ${d}/run)"; exit 1; }
        [ "$sign" = p ] && JP=$J_out || JM=$J_out
    done
    python3 -c "
import sys
jp, jm, h, kmd, idx = $JP, $JM, $H, $KMD, $idx
fd = (jp - jm) / (2*h*kmd)                       # dJ/dK per mD (dK = h*K)
rows = [l.split() for l in open('$GRADFILE').read().splitlines()[1:]]
adj = float(rows[idx][0]) * $MD_TO_SI            # per m^2 -> per mD
gap = fd - adj
rel = gap/fd if fd != 0 else float('nan')
print(f'%-10s %5d | %15.6e %15.6e | %12.4e %9.3f' % ('$name', idx, fd, adj, gap, rel))
"
done
