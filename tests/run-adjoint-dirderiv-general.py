#!/usr/bin/env python3
"""Deck-general directional-derivative check of the adjoint gradient, sized for
LARGER cases where per-cell FD is infeasible.

For a gradient vector g (dJ/dθ over a parameter family) and a random direction
d, the directional derivative g·d must equal the central difference
(J(θ+εd) − J(θ−εd)) / 2ε. Each direction costs only two extra forward runs, so
this scales to field decks (SPE9, Norne) where the per-cell cell×cell FD does
not.

Deck-general by construction: the perturbation is applied as an injected MULTPV
field (a per-cell pore-volume multiplier), so no assumption is made about how
PORO/PERM are specified (inline or via INCLUDE). The pv-multiplier is exactly
the basis of ADJOINT_GRADIENTS_PV.txt (dJ/d(pvmult) at pvmult=1), so g·d and the
MULTPV FD are in the same basis — no porosity-division fix-ups.

Families (extensible — add one dict entry to support another):
  pv : perturb the pore-volume multiplier via MULTPV; gradient = dJ/d(pvmult).

Only all-active decks are supported (asserted: #gradient rows == nx*ny*nz); add
an ACTNUM->cartesian map to lift that. Strict tolerances are mandatory (see the
adjoint strict-tolerance note): a loose forward tolerance or adjoint reduction
hides the true error.

Caveat: on decks with a *binding* explicitly-updated cross-step quantity
(DRSDT/DRVDT rate cap, hysteresis, ...) some directions show the known adjoint
cross-step gap (a few %), so the adjoint is not exact there yet — use a deck
without such a term (e.g. SPE9) for a clean verification, or expect that gap.
Small eps is required: J is nonlinear in pore volume and DISGAS saturation kinks
add a chord offset at moderate eps (validated: eps 1e-6 → rel.err ~1e-3 on
SPE1/SPE9; eps 1e-4 → ~1e-2). Increasing eps makes it worse, not better.

Usage:
  run-adjoint-dirderiv-general.py <flow_adjoint> <deck.DATA> <outdir>
      [family=pv] [objective=pressure-average] [eps] [reltol] [ndir] [seed]
"""
import os, re, sys, shutil, subprocess, random

flow, deck_path, outdir = sys.argv[1:4]
family    = sys.argv[4] if len(sys.argv) > 4 else "pv"
objective = sys.argv[5] if len(sys.argv) > 5 else "pressure-average"
# eps must be small: J is nonlinear in pore volume and DISGAS saturation kinks
# add a chord offset at moderate eps (SPE1: ~1e-3 at eps 1e-4, collapsing to
# ~1e-6 at eps 1e-6). 1e-6 is the validated sweet spot (roundoff-safe with
# strict forward tolerances). reltol 5e-3 tolerates residual kink noise.
eps    = float(sys.argv[6]) if len(sys.argv) > 6 else 1e-6
reltol = float(sys.argv[7]) if len(sys.argv) > 7 else 5e-3
ndir   = int(sys.argv[8])   if len(sys.argv) > 8 else 2
seed   = int(sys.argv[9])   if len(sys.argv) > 9 else 1

STRICT = ["--enable-storage-cache=false", "--enable-adaptive-time-stepping=false",
          "--threads-per-process=1",
          "--tolerance-cnv=1e-10", "--tolerance-cnv-relaxed=1e-10",
          "--tolerance-mb=1e-12", "--tolerance-mb-relaxed=1e-12",
          f"--adjoint-objective={objective}"]

FAMILY = {
    # perturb pv-multiplier via MULTPV; gradient file already in that basis.
    "pv": dict(grad="ADJOINT_GRADIENTS_PV.txt", col=0, keyword="MULTPV"),
}
if family not in FAMILY:
    print(f"unsupported family '{family}' (have: {', '.join(FAMILY)})"); sys.exit(2)
spec = FAMILY[family]

deck_dir  = os.path.dirname(os.path.abspath(deck_path))
deck_name = os.path.splitext(os.path.basename(deck_path))[0]
SECTIONS  = ("EDIT", "PROPS", "REGIONS", "SOLUTION", "SCHEDULE")

def read_grad_col(path, col):
    vals = []
    for ln in open(path):
        s = ln.strip()
        if not s or s.startswith("#"):
            continue
        try:
            vals.append(float(s.split()[col]))
        except (ValueError, IndexError):
            pass
    return vals

def deck_dimens(dirpath):
    """nx*ny*nz from DIMENS, scanning the deck and its .INC/.GRDECL files."""
    for root, _, files in os.walk(dirpath):
        for fn in files:
            if not fn.upper().endswith((".DATA", ".INC", ".GRDECL")):
                continue
            txt = open(os.path.join(root, fn), errors="ignore").read()
            m = re.search(r"^\s*DIMENS\s*\n\s*([0-9]+)\s+([0-9]+)\s+([0-9]+)",
                          txt, re.MULTILINE)
            if m:
                return int(m[1]) * int(m[2]) * int(m[3])
    raise RuntimeError("DIMENS not found in deck")

def deck_file_chain(dirpath, main_name):
    """Files (main deck + transitive INCLUDEs) belonging to this deck, so we
    never touch sibling decks that share the directory."""
    seen, stack = [], [os.path.join(dirpath, f"{main_name}.DATA")]
    while stack:
        p = stack.pop()
        if p in seen or not os.path.isfile(p):
            continue
        seen.append(p)
        txt = open(p, errors="ignore").read()
        for m in re.finditer(r"^\s*INCLUDE\s*(?:--.*)?\n\s*['\"]?([^'\"/\s]+)"
                             r"['\"]?\s*/", txt, re.MULTILINE):
            stack.append(os.path.normpath(os.path.join(os.path.dirname(p), m[1])))
    return seen

def inject_multpv(dstdir, main_name, values):
    """Insert 'MULTPV <values> /' at the end of the GRID section, in whichever
    file of THIS deck's include chain holds the GRID keyword."""
    body = spec["keyword"] + "\n" + \
           "\n".join(" ".join(f"{v:.10f}" for v in values[i:i+8])
                     for i in range(0, len(values), 8)) + "\n/\n"
    for p in deck_file_chain(dstdir, main_name):
        lines = open(p, errors="ignore").read().splitlines(keepends=True)
        grid_at = next((i for i, l in enumerate(lines)
                        if l.strip().upper() == "GRID"), None)
        if grid_at is None:
            continue
        end = next((i for i in range(grid_at + 1, len(lines))
                    if lines[i].strip().upper() in SECTIONS), len(lines))
        lines.insert(end, body)
        open(p, "w").writelines(lines)
        return
    raise RuntimeError("GRID section not found in deck's include chain")

def run(deckdir, save):
    deck = os.path.join(deckdir, f"{deck_name}.DATA")
    arch = os.path.join(deckdir, "a")
    mode = "--adjoint-save=true" if save else "--adjoint-mode=objective"
    r = subprocess.run([flow, deck, f"--output-dir={deckdir}", *STRICT,
                        f"--adjoint-file={arch}", mode],
                       capture_output=True, text=True)
    return r

def objective_J(deckdir):
    run(deckdir, save=True)                         # record
    r = run(deckdir, save=False)                    # evaluate J
    m = re.search(r"Adjoint objective J = ([0-9eE.+-]+)", r.stdout)
    if not m:
        print(r.stdout[-800:]); raise RuntimeError("no objective J")
    return float(m.group(1))

# --- base gradient (general-ad) --------------------------------------------
os.makedirs(outdir, exist_ok=True)
base = os.path.join(outdir, "base")
shutil.rmtree(base, ignore_errors=True); shutil.copytree(deck_dir, base)
run(base, save=True)
subprocess.run([flow, os.path.join(base, f"{deck_name}.DATA"),
                f"--output-dir={base}", *STRICT,
                f"--adjoint-file={os.path.join(base,'a')}",
                "--adjoint-mode=gradient",
                "--adjoint-gradient-method=general-ad"],
               capture_output=True, text=True)
gradpath = os.path.join(base, f"{deck_name}.{spec['grad']}")
if not os.path.exists(gradpath):
    print("no gradient file"); sys.exit(1)
g = read_grad_col(gradpath, spec["col"])
ncell = deck_dimens(deck_dir)
n = len(g)
print(f"deck={deck_name} family={family} objective={objective}")
print(f"  {n} gradient rows, grid {ncell} cells, eps={eps}, reltol={reltol}")
if n != ncell:
    print(f"FAIL: {n} active gradient rows != {ncell} grid cells; this harness "
          f"supports all-active decks only (add an ACTNUM->cartesian map)")
    sys.exit(1)

rng = random.Random(seed)
ok = True
for k in range(ndir):
    d = [rng.uniform(-1.0, 1.0) for _ in range(n)]
    gd = sum(gi * di for gi, di in zip(g, d))
    Js = {}
    for sgn, tag in ((+1, "p"), (-1, "m")):
        dd = os.path.join(outdir, f"dir{k}_{tag}")
        shutil.rmtree(dd, ignore_errors=True); shutil.copytree(deck_dir, dd)
        inject_multpv(dd, deck_name, [1.0 + sgn * eps * di for di in d])
        Js[tag] = objective_J(dd)
    fd = (Js["p"] - Js["m"]) / (2.0 * eps)
    rel = abs(gd - fd) / max(abs(gd), abs(fd), 1e-300)
    ok &= rel < reltol
    print(f"  dir {k}: g.d {gd: .8e}  FD {fd: .8e}  rel.err {rel:.2e}  "
          f"{'PASS' if rel < reltol else 'FAIL'}")

print(f"directional-derivative ({family}, {objective}): {'PASS' if ok else 'FAIL'}")
sys.exit(0 if ok else 1)
