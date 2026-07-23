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

DARCY = 9.869232667160128e-16  # 1 mD in m^2
FAMILY = {
    # kind "multpv": perturb the pv-multiplier via a standalone MULTPV field
    #   (base multiplier is 1, so the gradient is already in the perturbation
    #   basis). kind "array": perturb a deck field (e.g. PERMX) multiplicatively
    #   by rewriting its array; the gradient (SI) maps to the multiplier basis
    #   as g_i * base_i * gscale.
    "pv":   dict(grad="ADJOINT_GRADIENTS_PV.txt",   gcol=0, kind="multpv"),
    # PERM: B1 added the dCTF/dK well term, so the perm gradient is now correct
    # at perforations too — no masking needed. (Set mask_perf=True to isolate the
    # reservoir gradient if the well term is disabled via --adjoint-perm-well-term.)
    "perm": dict(grad="ADJOINT_GRADIENTS_PERM.txt", gcol=0, kind="array",
                 keyword="PERMX", gscale=DARCY, mask_perf=False),
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

def deck_dims(dirpath):
    """(nx, ny, nz) from DIMENS, scanning the deck and its .INC/.GRDECL files."""
    for root, _, files in os.walk(dirpath):
        for fn in files:
            if not fn.upper().endswith((".DATA", ".INC", ".GRDECL")):
                continue
            txt = open(os.path.join(root, fn), errors="ignore").read()
            m = re.search(r"^\s*DIMENS\s*\n\s*([0-9]+)\s+([0-9]+)\s+([0-9]+)",
                          txt, re.MULTILINE)
            if m:
                return int(m[1]), int(m[2]), int(m[3])
    raise RuntimeError("DIMENS not found in deck")

def perforated_cells(dirpath, dims):
    """Cartesian indices of COMPDAT connections (explicit I J K1 K2), so the
    perm test can skip cells whose gradient needs the dCTF/dK well term."""
    nx, ny, _ = dims
    cells = set()
    for p in deck_file_chain(dirpath, deck_name):
        lines = open(p, errors="ignore").read().splitlines()
        i = 0
        while i < len(lines):
            if lines[i].strip().upper().startswith("COMPDAT"):
                for j in range(i + 1, len(lines)):
                    s = lines[j].split("--")[0].strip()
                    if s.startswith("/") or not s:
                        if s.startswith("/"):
                            break
                        continue
                    toks = s.rstrip("/").split()
                    # 'WELL' I J K1 K2 ... ; skip if I/J defaulted (1*)
                    if len(toks) >= 5 and toks[1] not in ("1*", "*") \
                       and toks[2] not in ("1*", "*"):
                        try:
                            I, J, K1, K2 = (int(toks[1]), int(toks[2]),
                                            int(toks[3]), int(toks[4]))
                        except ValueError:
                            continue
                        for K in range(K1, K2 + 1):
                            cells.add((I-1) + (J-1)*nx + (K-1)*nx*ny)
                i = j
            i += 1
    return cells

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
    body = "MULTPV\n" + \
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

def parse_array(dstdir, keyword):
    """Per-cell values of an ECL keyword array (expands N*val, skips comments)
    from THIS deck's include chain. Returns None if not found."""
    for p in deck_file_chain(dstdir, deck_name):
        lines = open(p, errors="ignore").read().splitlines()
        i = next((k for k, l in enumerate(lines)
                  if l.strip().upper() == keyword), None)
        if i is None:
            continue
        vals = []
        for j in range(i + 1, len(lines)):
            s = lines[j].split("--")[0].strip()
            if not s:
                continue
            done = s.endswith("/")
            for tok in s.rstrip("/").split():
                if "*" in tok:
                    c, v = tok.split("*"); vals += [float(v)] * int(c)
                else:
                    vals.append(float(tok))
            if done:
                break
        return vals
    return None

def array_body(keyword, values):
    return keyword + "\n" + \
           "\n".join(" ".join(f"{v:.8f}" for v in values[i:i+8])
                     for i in range(0, len(values), 8)) + "\n/\n"

def ensure_array(dstdir, keyword, values):
    r"""Set a keyword array to `values`: rewrite it in place if present, else
    inject it at the end of the GRID section (for fields that default, e.g.
    PERMY/PERMZ following PERMX — pinning them isolates the PERMX perturbation)."""
    body = array_body(keyword, values)
    for p in deck_file_chain(dstdir, deck_name):
        lines = open(p, errors="ignore").read().splitlines(keepends=True)
        i = next((k for k, l in enumerate(lines)
                  if l.strip().upper() == keyword), None)
        if i is None:
            continue
        j = i + 1
        while j < len(lines) and "/" not in lines[j].split("--")[0]:
            j += 1
        lines[i:j+1] = [body]
        open(p, "w").writelines(lines)
        return
    inject_in_grid(dstdir, body)

def inject_in_grid(dstdir, body):
    for p in deck_file_chain(dstdir, deck_name):
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
    raise RuntimeError("GRID section not found for array injection")

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
g = read_grad_col(gradpath, spec["gcol"])
dims = deck_dims(deck_dir)
ncell = dims[0] * dims[1] * dims[2]
n = len(g)
print(f"deck={deck_name} family={family} objective={objective}")
print(f"  {n} gradient rows, grid {ncell} cells, eps={eps}, reltol={reltol}")
if n != ncell:
    print(f"FAIL: {n} active gradient rows != {ncell} grid cells; this harness "
          f"supports all-active decks only (add an ACTNUM->cartesian map)")
    sys.exit(1)

# Per-family base field and gradient mapped to the perturbation-multiplier
# basis, so g_basis·d is comparable to the FD of a (1 + eps·d) perturbation.
perm_pin = []   # (keyword, base) for PERMY/PERMZ, pinned so only PERMX varies
if spec["kind"] == "multpv":
    base_vals = [1.0] * n
    g_basis   = g                                   # already dJ/d(pvmult)
else:  # array (perm): perturb PERMX only, pin PERMY/PERMZ to isolate dJ/dKx
    base_vals = parse_array(deck_dir, spec["keyword"])
    if base_vals is None or len(base_vals) != n:
        got = 0 if base_vals is None else len(base_vals)
        print(f"FAIL: could not parse {spec['keyword']} array ({got} vs {n} "
              f"cells) — field may use COPY/BOX (not yet supported)")
        sys.exit(1)
    # PERMY/PERMZ default to PERMX when absent; rewriting PERMX would move them
    # too. Pin each to its base (parsed, or PERMX if it defaults) so the FD is
    # purely dJ/dPERMX, matching gradient column 0.
    for kw in ("PERMY", "PERMZ"):
        b = parse_array(deck_dir, kw)
        perm_pin.append((kw, b if (b and len(b) == n) else base_vals))
    # dJ/d(mult_i) = dJ/dPERMX_i[SI] * PERMX_i[m^2]; PERMX_i[m^2]=base*gscale
    g_basis = [gi * bi * spec["gscale"] for gi, bi in zip(g, base_vals)]

def apply_perturbation(dd, mult):
    if spec["kind"] == "multpv":
        inject_multpv(dd, deck_name, mult)
    else:
        ensure_array(dd, spec["keyword"], [b * m for b, m in zip(base_vals, mult)])
        for kw, b in perm_pin:
            ensure_array(dd, kw, b)

# Cells to hold fixed (d = 0): perforations for perm (dCTF/dK gap, task B1).
masked = perforated_cells(deck_dir, dims) if spec.get("mask_perf") else set()
if masked:
    print(f"  masking {len(masked)} perforated cell(s) (perm dCTF/dK gap, B1)")

rng = random.Random(seed)
ok = True
for k in range(ndir):
    d = [0.0 if i in masked else rng.uniform(-1.0, 1.0) for i in range(n)]
    gd = sum(gb * di for gb, di in zip(g_basis, d))
    Js = {}
    for sgn, tag in ((+1, "p"), (-1, "m")):
        dd = os.path.join(outdir, f"dir{k}_{tag}")
        shutil.rmtree(dd, ignore_errors=True); shutil.copytree(deck_dir, dd)
        apply_perturbation(dd, [1.0 + sgn * eps * di for di in d])
        Js[tag] = objective_J(dd)
    fd = (Js["p"] - Js["m"]) / (2.0 * eps)
    rel = abs(gd - fd) / max(abs(gd), abs(fd), 1e-300)
    ok &= rel < reltol
    print(f"  dir {k}: g.d {gd: .8e}  FD {fd: .8e}  rel.err {rel:.2e}  "
          f"{'PASS' if rel < reltol else 'FAIL'}")

print(f"directional-derivative ({family}, {objective}): {'PASS' if ok else 'FAIL'}")
sys.exit(0 if ok else 1)
