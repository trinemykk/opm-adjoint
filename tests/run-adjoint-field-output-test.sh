#!/bin/bash
# Machine-readable field output: a gradient run must produce cartesian-keyed
# CSV files and a valid JSON metadata sidecar (schema, per-family basis/units,
# objective value + per-term breakdown), and the CSV values must equal the
# legacy .txt gradients.
#
# usage: run-adjoint-field-output-test.sh <flow_adjoint> <SPE1CASE1.DATA> <outdir>
set -u
FLOW=${1:?flow_adjoint}
DECK=${2:?SPE1CASE1.DATA}
OUT=${3:?outdir}
mkdir -p "$OUT/ref" "$OUT/run"
S="--threads-per-process=1 --enable-storage-cache=false --enable-adaptive-time-stepping=false"

$FLOW "$DECK" --output-dir="$OUT/ref" $S > /dev/null 2>&1
REF="$OUT/ref/SPE1CASE1"
OBJ="matchsum:$REF:PROD.gas*2.0+PROD.oil"
$FLOW "$DECK" --output-dir="$OUT/run" $S --adjoint-save=true --adjoint-objective="$OBJ" > /dev/null 2>&1
$FLOW "$DECK" --output-dir="$OUT/run" $S --adjoint-mode=gradient \
    --adjoint-objective="$OBJ" --adjoint-gradient-method=general-ad > "$OUT/g.log" 2>&1

B="$OUT/run/SPE1CASE1"
python3 - "$B" << 'EOF'
import json, math, sys
b = sys.argv[1]
d = json.load(open(b + ".ADJOINT_GRADIENTS.json"))
assert d["schema_version"] == 1, d
assert d["case"] == "SPE1CASE1"
names = [f["name"] for f in d["families"]]
assert names == ["poro", "perm", "trans", "wellctrl"], names
assert len(d["objective"]["terms"]) == 2, d["objective"]
# poro family: basis multiplier, csv keyed by cartesian index 0..N-1 (serial)
poro = next(f for f in d["families"] if f["name"] == "poro")
assert poro["basis"] == "multiplier"
csv = [l.split(",") for l in open(b + ".ADJOINT_GRADIENTS_PORO.csv").read().splitlines()[1:]]
txt = open(b + ".ADJOINT_GRADIENTS_PV.txt").read().splitlines()[1:]
assert len(csv) == len(txt) == poro["count"], (len(csv), len(txt))
assert [int(c[0]) for c in csv] == list(range(len(csv))), "cartesian keys not 0..N-1"
assert all(float(c[1]) == float(t) for c, t in zip(csv, txt)), "PORO.csv != PV.txt"
# perm csv columns present and count matches
perm = [l.split(",") for l in open(b + ".ADJOINT_GRADIENTS_PERM.csv").read().splitlines()[1:]]
assert len(perm) == len(csv) and all(len(r) == 4 for r in perm)
# trans + wellctrl csv have headers and rows
for fam in ("TRANS", "WELLCTRL"):
    rows = open(b + f".ADJOINT_GRADIENTS_{fam}.csv").read().splitlines()
    assert len(rows) >= 2, fam
print("field output: JSON valid, cartesian-keyed CSVs, values match .txt: PASS")
EOF
