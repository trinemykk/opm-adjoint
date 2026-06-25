#!/usr/bin/env python3
"""Random directional-derivative check of the general (stencil-local)
parameter gradient, computed for ALL parameters in one backward sweep.

For a gradient vector g (dJ/dθ for every parameter of a family) and a random
direction d, the directional derivative g·d must match the central finite
difference (J(θ+ε d) − J(θ−ε d)) / (2ε). One FD pair validates the whole
gradient vector along d; a few random directions give high confidence without
the per-component cell×cell cost.

The gradient is produced by --adjoint-gradient-method=general (the mechanism
under test); the FD perturbs the underlying deck field and re-runs the
objective. Implemented family: pv (porosity / pore-volume multiplier).

Usage:
  run-adjoint-dirderiv-test.py <flow_adjoint> <inputfiles-dir> <deckname>
                               <family> <outdir> [eps] [reltol] [ndir] [seed]
"""
import os, re, sys, shutil, subprocess, random

flow, inputdir, deckname, family, outdir = sys.argv[1:6]
eps    = float(sys.argv[6]) if len(sys.argv) > 6 else 1e-4
reltol = float(sys.argv[7]) if len(sys.argv) > 7 else 1e-3
ndir   = int(sys.argv[8])   if len(sys.argv) > 8 else 2
seed   = int(sys.argv[9])   if len(sys.argv) > 9 else 1

STRICT = ["--enable-storage-cache=false", "--enable-adaptive-time-stepping=false",
          "--threads-per-process=1", "--tolerance-cnv=1e-9", "--tolerance-mb=1e-11",
          "--adjoint-objective=pressure-average"]

os.makedirs(outdir, exist_ok=True)

# Deck field perturbed for each family, and how the adjoint gradient maps to
# the perturbed field.
#   pv:   gradient file is dJ/d(pv-multiplier); storage ∝ porosity so
#         dJ/dporo_i = g_i / poro_i; the FD perturbs poro.txt.
#   perm: gradient file is dJ/dK in SI (per m^2); permx.txt is in mD, so
#         dJ/dK_mD = g_i * 9.869233e-16; the FD perturbs permx.txt.
DARCY = 9.869232667160128e-16  # 1 mD in m^2
FIELD = {
    "pv":   dict(field="poro.txt",  grad="ADJOINT_GRADIENTS_PV.txt",   col=0,
                 xform=lambda g, fv: g / fv),
    "perm": dict(field="permx.txt", grad="ADJOINT_GRADIENTS_PERM.txt", col=0,
                 xform=lambda g, fv: g * DARCY),
}
if family not in FIELD:
    print(f"unsupported family '{family}'"); sys.exit(2)
spec = FIELD[family]
fieldfile, gradfile, gcol = spec["field"], spec["grad"], spec["col"]

def read_col(path, col=0, skip_hash=True):
    vals = []
    for ln in open(path):
        s = ln.strip()
        if not s or (skip_hash and s.startswith("#")):
            continue
        if s.startswith(fieldfile.split('.')[0].upper()) or s == "/":
            continue
        try:
            vals.append(float(s.split()[col]))
        except ValueError:
            pass
    return vals

def write_field(srcdir, dstdir, newvals):
    shutil.rmtree(dstdir, ignore_errors=True)
    shutil.copytree(srcdir, dstdir)
    src = os.path.join(srcdir, fieldfile)
    out, n = [], 0
    for ln in open(src):
        s = ln.strip()
        if s and s != "/" and not s.upper().startswith(fieldfile.split('.')[0].upper()):
            ln = f"{newvals[n]:.16e}\n"; n += 1
        out.append(ln)
    open(os.path.join(dstdir, fieldfile), "w").writelines(out)

def run_J(rundir):
    deck = os.path.join(rundir, f"{deckname}.DATA")
    arch = os.path.join(rundir, "a")
    subprocess.run([flow, deck, f"--output-dir={rundir}", *STRICT,
                    f"--adjoint-file={arch}", "--adjoint-save=true"],
                   capture_output=True, text=True)
    r = subprocess.run([flow, deck, f"--output-dir={rundir}", *STRICT,
                        f"--adjoint-file={arch}", "--adjoint-mode=objective"],
                       capture_output=True, text=True)
    m = re.search(r"Adjoint objective J = ([0-9eE.+-]+)", r.stdout)
    if not m:
        print(r.stdout[-500:]); raise RuntimeError("no objective J")
    return float(m.group(1))

# 1) base gradient with the GENERAL method (the mechanism under test).
base = os.path.join(outdir, "base")
shutil.rmtree(base, ignore_errors=True); shutil.copytree(inputdir, base)
deck = os.path.join(base, f"{deckname}.DATA")
arch = os.path.join(base, "a")
subprocess.run([flow, deck, f"--output-dir={base}", *STRICT,
                f"--adjoint-file={arch}", "--adjoint-save=true"],
               capture_output=True, text=True)
g = subprocess.run([flow, deck, f"--output-dir={base}", *STRICT,
                    f"--adjoint-file={arch}", "--adjoint-mode=gradient",
                    "--adjoint-gradient-method=general"],
                   capture_output=True, text=True)
gradpath = os.path.join(base, f"{deckname}.{gradfile}")
if not os.path.exists(gradpath):
    print(g.stdout[-800:]); print("no gradient file"); sys.exit(1)

g_raw = read_col(gradpath, gcol)             # raw adjoint gradient (file basis)
field0 = read_col(os.path.join(inputdir, fieldfile))
# map the adjoint gradient to the basis of the perturbed deck field
g_field = [spec["xform"](g, f) for g, f in zip(g_raw, field0)]
n = len(g_field)
print(f"family={family}: {n} parameters, eps={eps}, reltol={reltol}")

rng = random.Random(seed)
ok = True
for kdir in range(ndir):
    d = [rng.uniform(-1.0, 1.0) for _ in range(n)]
    gd = sum(gi * di for gi, di in zip(g_field, d))      # g·d (analytic dir-deriv)
    plus  = [f + eps * di for f, di in zip(field0, d)]
    minus = [f - eps * di for f, di in zip(field0, d)]
    write_field(inputdir, os.path.join(outdir, f"dir{kdir}_p"), plus)
    write_field(inputdir, os.path.join(outdir, f"dir{kdir}_m"), minus)
    Jp = run_J(os.path.join(outdir, f"dir{kdir}_p"))
    Jm = run_J(os.path.join(outdir, f"dir{kdir}_m"))
    fd = (Jp - Jm) / (2.0 * eps)
    rel = abs(gd - fd) / max(abs(gd), abs(fd), 1e-300)
    status = "PASS" if rel < reltol else "FAIL"
    if rel >= reltol:
        ok = False
    print(f"  dir {kdir}: g.d {gd: .8e}  FD {fd: .8e}  rel.err {rel:.2e}  {status}")

print(f"directional-derivative ({family}, general): {'PASS' if ok else 'FAIL'}")
sys.exit(0 if ok else 1)
