# Adjoint paths, and the road to a practical field-scale adjoint

_Written 2026-06-27 at commit `7f285b1` (suite 28/28). Purpose: the decision basis for the
next phase — a "practical" optimal, flexible adjoint, AD-first, integrable in
history-matching workflows, sized for large(r) field cases. A plan is written from this._

## 1. The gradient paths as they exist today

One backward sweep (λ-solves once per substep) feeds all paths; they differ only in how
`dR/dθ` is obtained and contracted.

| path | selector | mechanism | families | accuracy | SPE9 gradient phase |
|---|---|---|---|---|---|
| **analytic** | default | hand-derived closed forms (storage linear in PV, flux linear in trans, half-trans chain rule for perm) | PV, trans, perm, well controls | exact; FD-verified | 0.036 s |
| **general (FD)** | `--adjoint-gradient-method=general` | term-targeted centered FD on the scale factor (`--adjoint-fd-step`) | PV, trans, perm | ~2e-9 floor | 0.360 s |
| **general-AD** | `general-ad` | term-targeted **one-slot seeded `Evaluation`** — exact `dR/dθ` in one pass | PV, trans, perm | machine precision vs analytic (1e-15…1e-13) | 0.200 s |
| **endpoint whole** | `--adjoint-endpoints=…` (default method) | perturb material law, **full re-linearization** per (param, cell), contract λ_r **and** λ_w | SWL…SGU, KR*, PC* | FD (~1e-8 vs FD tests) | cell×N (14.6 s on SPE1) |
| **endpoint stencil** | `--adjoint-endpoint-method=stencil` | perturb material law, rebuild **one cell's IQ** via public `update()`, evaluate `{i}∪nb(i)`; whole-domain fallback for perforated cells | same | agrees with whole to 5e-11 | cell×~7 (0.26 s on SPE1, **56×**) |
| **well controls** | always | `dJ/du = −Σ λ_w[last]` (control-equation row), tagged **per active mode** | rate/BHP targets per mode | exact (row derivative is −1); FD-verified | negligible |

Supporting substrate (not gradients, but what makes them trustworthy and extensible):
- **Replay**: recreates every converged forward linearization bitwise (SPE1/SPE9/m1d/
  SPE1_GRPCTRL; Norne end-to-end, ~90 % bitwise).
- **Evaluator**: standalone residual **and Jacobian** reproduction, bitwise vs the
  linearizer, from the GPU-shared kernels — both off the model IQ cache and off the
  **separate adjoint-owned IQ array** (public `update()`, no simulator change).
- MPI (exact transposed operator), iterative solvers (cprt serial / ilu0 parallel),
  FD-verified golden regression set, 28 ctests.

**The key status fact: "AD, not FD" is already achieved for PERM/PORO/TRANS.** The
general-AD path is exact, cheap, and one code path. FD survives *only* in the end-point
family, where exact AD is blocked by the `Scalar`-typed material-law scaled points
(`EclEpsScalingPoints`) — a deep, GPU-shared upstream change (C.4). End points are also
the *least* important family for classical history matching.

## 2. Known gaps that matter for field use (honesty list)

Ranked for a PERM/PORO history-matching target:

1. **`dCTF/dK` well-index coupling missing from `dJ/dperm`** — on real decks the
   connection factor is perm-derived (defaulted CTF, Peaceman/Kh), so the perm gradient
   at perforated cells misses the well-equation term. Warned at runtime since `c616f50`;
   **magnitude never quantified**. This is potentially the top correctness item for field
   perm-HM — perforated cells are exactly where perm matters most. First step of any
   plan: quantify by FD on a perforated cell (SPE1/SPE9, defaulted CTF).
2. **Field-deck replay edges**: report-step-0 startup (model2-like), well-event/hysteresis
   boundaries (Norne ~10 % of substeps at 1e-3…0.46), WGRUPCON/voidage targets (Stage-2
   fix designed, unimplemented). Group-control gradients themselves are FD-verified
   (SPE1_GRPCTRL) — the risk is replay drift on messy schedules, which the T2/T3
   comparisons detect but do not fix.
3. **Archive size / IO**: "write too much" stores a full simulator snapshot per substep
   (Norne with systems: 23 GB). Practical field loops need a slim mode (solution vectors
   + well/group state only) and/or the planned **one-stop mode** (forward sweep →
   backward sweep in one process, states kept in memory, Jutul's `AdjointPackedResult`
   pattern) — designed in `adjoint_plan.md`, never implemented.
4. Guide-rate/group and hysteresis/DRSDT cross-terms (warned, negligible on tested decks).
5. Endpoint exact-AD (deep; stencil-FD is a correct, fast stand-in).

## 3. The input/output layer (workflow integration)

What exists — grown organically as CLI strings:
- **In**: `--adjoint-save`, `--adjoint-mode=replay|gradient|objective`,
  `--adjoint-objective=pressure-average|bhp:W|rate:W:p|match:W:p:target|matchref:W:p:REFCASE|matchsum:REF:W.p+…`,
  `--adjoint-endpoints=…`, method/solver/step flags. Observed data = a reference case's
  summary read via `ESmry` (units auto-converted; must live outside the run's output dir).
- **Out**: per-cell/per-face **text files**
  `<CASE>.ADJOINT_GRADIENTS_{PV,TRANS,PERM,WELLCTRL,WELLCTRL_BYMODE,ENDPOINT_*}.txt`,
  `ADJOINT_LAMBDA_WELLS.txt`, and `J` on the log. Cell order = cartesian-sorted interior
  cells; PERM in SI (per m²) with the mD conversion in the header.

What a history-matching workflow (ERT/Everest-style loop, or a Python optimizer)
actually needs — the gaps:
1. **A config file instead of flag strings** — the JSON sketched in `adjoint_plan.md`
   (`--adjoint-config`): objective terms with **weights** (the `weight` field exists in
   code but is not settable from the CLI), time **windows**, wildcard well lists,
   parameter selection (which families, which regions/boxes), solver, output format.
2. **Machine-readable gradient output**: one structured file (HDF5 group in the existing
   archive, or npz/csv) keyed by **cartesian index**, with metadata: parameter basis
   (multiplier vs value — optimizers need to know the gradient is `dJ/d(mult)` at
   mult=1), units, objective value **per term**, deck/case id, run parameters. Text
   files stay for humans.
3. **One-stop gradient mode** (`flow_adjoint CASE --adjoint-config=…` → J + gradients in
   one process) — halves IO per optimizer iteration and removes the two-binary-stages
   foot-gun (summary truncation, storage-cache mismatches).
4. Conventions: stable exit codes, gradient sign convention documented (minimize J),
   and a versioned schema so a driver can detect incompatibility.

**JutulDarcy as the reference UX and the cross-check**: it has been (at least partly)
used on real fields; its `case + objective → sens[:porosity]…` interface is the model
for (1)–(3). Our T6 bridge exists (`jutul/`, SPE1 pattern/sign agreement; forwards
differ by well model). For field validation, running the *same* config through both and
comparing PORO/PERM sensitivities on a shared deck is the strongest available check
beyond FD — worth keeping in the plan as a validation channel, not CI.

## 4. Recommendation: restrict and harden

**Yes — restrict the practical scope to PORO (pvmult) + PERM (and trans as its raw
form), and optimize for large(r) field cases.** Reasoning:
- These are the canonical HM parameters, and they are the *finished* part: exact AD,
  one-sweep, MPI-parallel, machine-precision-verified, cheap (gradient accumulation is
  2 % of the sweep; the solve dominates).
- Everything that actually blocks field use is **not** a missing parameter family:
  it is the `dCTF/dK` term (quantify first), field-deck replay robustness, archive
  size/one-stop, and the I/O layer. Those are where the effort should go.
- End-point AD (the one deep upstream change) is deferred without loss: stencil-FD
  covers end points correctly when someone needs them.

Suggested plan shape (for the coming planning session, in order):
1. Quantify `dCTF/dK` on a defaulted-CTF deck; if material, add the well-row perm term
   (the machinery — λ_w, B/C/D access, perforation loop — all exists).
2. Flip defaults to the proven fast paths (`general-ad`, `stencil`) after a Norne-scale
   shakeout; keep analytic as oracle.
3. I/O layer: JSON config + structured gradient output (+ per-term J, weights, windows).
4. One-stop mode + slim archive (the two together make an optimizer iteration ≈ one
   forward + one backward, minimal disk).
5. Field-deck hardening pass on Norne-class decks (report-0, well events; WGRUPCON
   Stage-2 if group-controlled fields are in scope).
6. JutulDarcy side-by-side on one shared field-like case (validation channel).

## 5. Fresh conversation?

Recommended: **yes, plan and execute the next phase in a fresh Claude Code session.**
This session has been compacted several times and is far past its natural size; a fresh
one starts with full context budget for reading field decks and the I/O design. The
handoff is already in place: this file + `STATUS.md` + `adjoint_code_assessment.md` in
the repo, and the auto-memory (`adjoint-project-state`) carries the full decision
history. Suggested opener for the new session: *"Read opm-adjoint/adjoint_paths_summary.md
and STATUS.md, then plan the practical field-HM phase (§4)"* — and enter plan mode from
there.
