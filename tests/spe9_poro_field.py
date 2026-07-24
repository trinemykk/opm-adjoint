#!/usr/bin/env python3
"""Helper for the SPE9 PORO field-HM demo (adjoint A6).

The stock SPE9 BASE.INC specifies PORO as 15 constant layers (600 cells each).
For a per-cell history-matching demo we need an *explicit* flat PORO array so
individual cells can be perturbed (truth field, FD spot-check). This tool reads
the original BASE.INC, expands its PORO block into a flat 9000-value array, and
writes a new BASE.INC with the requested variant of that array.

Grid: DIMENS 24 25 15 -> 9000 cells, natural order c = i + 24*j + 600*k.

Subcommands (all read the ORIGINAL base.inc and write a new one):
  expand  <base.inc> <out.inc>                 flat base PORO (no change)
  truth   <base.inc> <out.inc> [amp]           flat base * smooth field (default amp 0.08)
  perturb <base.inc> <out.inc> <cell> <delta>  flat base with PORO[cell] += delta
  poro    <base.inc> <cell>                     print base PORO of a cell (stdout)
"""
import math
import sys

NX, NY, NZ = 24, 25, 15
NCELL = NX * NY * NZ


def read_base_poro(path):
    """Return the flat list of NCELL base porosities from a BASE.INC."""
    toks = []
    with open(path) as f:
        lines = f.read().splitlines()
    inblock = False
    for ln in lines:
        s = ln.split("--", 1)[0].strip()          # drop comments
        if not inblock:
            if s == "PORO":
                inblock = True
            continue
        if not s:
            continue
        done = s.endswith("/")
        s = s.rstrip("/").strip()
        for t in s.split():
            if "*" in t:
                n, v = t.split("*")
                toks.extend([float(v)] * int(n))
            elif t:
                toks.append(float(t))
        if done:
            break
    if len(toks) != NCELL:
        sys.exit(f"expected {NCELL} PORO values, parsed {len(toks)}")
    return toks


def smooth_field(base, amp):
    """Deterministic smooth multiplicative truth field, clipped physical."""
    out = []
    for c in range(NCELL):
        i = c % NX
        j = (c // NX) % NY
        k = c // (NX * NY)
        f = 1.0 + amp * math.sin(math.pi * (i + 0.5) / NX) \
                      * math.cos(math.pi * (j + 0.5) / NY) \
                      * (1.0 if k % 2 == 0 else -1.0)
        out.append(min(0.4, max(0.02, base[c] * f)))
    return out


def write_base_with_poro(src, dst, poro):
    """Copy BASE.INC, replacing the PORO block with an explicit flat array."""
    with open(src) as f:
        lines = f.read().splitlines()
    out = []
    i = 0
    while i < len(lines):
        if lines[i].split("--", 1)[0].strip() == "PORO":
            out.append("PORO")
            # emit 8 values per line, terminate with '/'
            for r in range(0, NCELL, 8):
                out.append("  " + " ".join(f"{v:.8f}" for v in poro[r:r + 8]))
            out[-1] += " /"
            # skip the original block up to and including its '/'
            i += 1
            while i < len(lines) and not lines[i].split("--", 1)[0].rstrip().endswith("/"):
                i += 1
            i += 1  # past the terminating '/'
            continue
        out.append(lines[i])
        i += 1
    with open(dst, "w") as f:
        f.write("\n".join(out) + "\n")


def main(argv):
    if len(argv) < 3:
        sys.exit(__doc__)
    cmd, src = argv[1], argv[2]
    base = read_base_poro(src)

    if cmd == "poro":
        print(repr(base[int(argv[3])]))
    elif cmd == "expand":
        write_base_with_poro(src, argv[3], base)
    elif cmd == "truth":
        amp = float(argv[4]) if len(argv) > 4 else 0.08
        write_base_with_poro(src, argv[3], smooth_field(base, amp))
    elif cmd == "perturb":
        cell, delta = int(argv[4]), float(argv[5])
        p = list(base)
        p[cell] = p[cell] + delta
        write_base_with_poro(src, argv[3], p)
    else:
        sys.exit(f"unknown subcommand {cmd}")


if __name__ == "__main__":
    main(sys.argv)
