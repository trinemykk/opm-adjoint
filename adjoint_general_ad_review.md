# Checkpoint review: toward a fully-general AD assembly

_Status after Step A (commit `6eec6c4`). This document is the review the next
plan is written from — it states what the general AD substrate now is, how
parameters are differentiated through it, and exactly where each quantity's
restriction lies._

## 1. What exists now

`AdjointResidualEvaluator<TypeTag>` (opm-adjoint, header-only) is a faithful
re-implementation of the forward TPFA black-oil assembly, built **only** on the
GPU-shared static kernels (`computeFlux` / `computeStorage<Eval>` /
`computeSource` in `blackoillocalresidualtpfa.hh`) and the model's existing
`Evaluation` intensive quantities. No new kernel code; no Evaluation-type change
deep in the simulator; all public APIs.

Verified against the production linearizer at every converged replay substep
(`--adjoint-residual-eval-check`):

| reproduced | how | result (m1d / SPE1) |
|---|---|---|
| residual `R(x)` | `evalResidual` (value pass) | bitwise (abs ~1e-14) |
| Jacobian `dR/dstate` | `evalJacobian` mirrors `linearize_cell` + `setResAndJacobi` | **bitwise, rel 0.0** (7/7, 123/123) |

Parameter gradients already ride on this substrate, validated to machine
precision vs the analytic closed forms (PV/trans/perm), with two derivative
mechanisms in production:

- **post-hoc scale** on a linear input — PV (`pvScale` on storage), trans
  (`transScale` on the flux value; flux is exactly linear in trans);
- **in-loop recompute** from geometric factors — perm (seed the directional-K
  factor → rescale the linear half-trans → harmonic-mean trans).

Both are exposed as FD (`--adjoint-gradient-method=general`) and exact one-pass
AD (`general-ad`, seed = `Evaluation<Scalar,1>`).

## 2. The three differentiation modes (and the fourth, restricted, one)

| mode | mechanism | quantities | status |
|---|---|---|---|
| **(a) post-hoc scale** | multiply a kernel input/output by a seeded scale; exact when the residual is linear in it | PV multiplier, transmissibility multiplier | **done** (FD + AD) |
| **(b) in-loop recompute** | recompute a precomputed geometric factor from exposed primitives with the parameter seeded | permeability (via `Transmissibility::halfTransmissibility`) | **done** (FD + AD) |
| **(c) IQ rebuild via public `update`** | build a *local* IQ with a chosen `Evaluation` by seeding the primary variables, through the ElementContext-free `IntensiveQuantities::update(problem, priVars, globalIdx, timeIdx)` (`blackoilintensivequantities.hh:760`) — **no simulator change** | reservoir **state** (the full Jacobian); anything entering only through primary variables | **available** (state Jacobian reproduced; not yet driven by a seeded-priVars pass) |
| **(d) restricted / deeper change** | parameter enters through a precomputed object the kernels treat as a constant, or through equations outside the cell residual | end points, well CTF/WPI, defaulted THPRES, diffusion | **FD or neglected** (see §3) |

## 3. Per-quantity restriction map

Cross-referenced with the precomputed-value provenance table in
`adjoint_plan.md` (§ flexible parameter linearizer) and the kernel signatures.

| quantity | enters through | differentiable by | restriction |
|---|---|---|---|
| **reservoir state** | IQ (primary variables) | (c) seed priVars → exact Jacobian | none — `IntensiveQuantities::update` is public; Step A already reproduces `dR/dstate` bitwise |
| **pore volume / MULTPV** | storage term (linear) | (a) pvScale | none |
| **transmissibility / MULT** | flux term (linear) | (a) transScale | none |
| **permeability** | `nbInfo.trans` via half-trans (linear in K) | (b) recompute harmonic trans | reservoir part exact; the **CTF/WPI** part (perm → well index) is separate, see below |
| **end-point scaling** (SWL…, KR*, PC*) | relperm/Pc inside `IntensiveQuantities::update` (material law) | FD today (`EndpointPerturber` rebuilds scaled points as `Scalar`, recomputes the local IQ) | **exact AD blocked**: needs `Evaluation`-typed material-law scaled-point construction (the C.4 upstream item). Also the **well-row** term (relperm reaches perforation rates) and the **initial-state** term (Pc shifts the equilibrated transition zone) — the latter needs explicit-init decks |
| **well CTF / WPI (Peaceman, Kh)** | the **well equations**, not the cell residual | — | **out of the cell evaluator**; `dCTF/dK` reaches `dJ/dperm` only through the well rows; neglected today, needs the well-equation derivative |
| **well control target** | the well control-equation row | already done (`dJ/du = -Σ λ_w[last]`) | none (separate path, not the cell evaluator) |
| **threshold pressure THPRES** | flux term (subtracted potential) | (a)-like if a parameter | constant when deck-given; **if defaulted** (from initial equil) the initial-state dependence is ignored — warn |
| **rock-compaction multipliers** | porosity/trans vs pressure history | (c) state pass | state-dependent, captured by the Jacobian; no separate parameter in v1 |
| **diffusion / dispersion** | extra flux terms | — | out of scope (isothermal black-oil), already a replay restriction |

**The two genuine blockers for "fully general":**
1. **Material-law parameter AD** (end points): the kernels are `Evaluation`-generic,
   but the *material-law parameter objects* (`EclEpsScalingPoints`) are `Scalar`.
   Exact endpoint AD needs them `Evaluation`-typed (or a per-seed rebuild). Until
   then end points stay FD (already correct, just `cell × Ncells` and FD-noisy).
2. **Well-equation coupling**: quantities that change perforation rates (end
   points, perm-via-CTF) also perturb the well residual `R_w`. The cell evaluator
   captures the reservoir-side source (via `computeSource`) but not the well-row
   term `λ_w · dR_w/dθ`. Needs the evaluator extended to the well rows.

## 4. Recommendation for the next plan

The assembly is now general enough that **state and any "scale/recompute"
parameter are exact**; the remaining work is (1) moving end points onto this
stencil-local mechanism and (2) closing the two blockers. Suggested order:

1. **End points onto the general mechanism, FD first** — reuse `EndpointPerturber`
   as a parameter family on the stencil-local evaluator (replace the current
   whole-domain `cell × Ncells` FD with `cell × ~7`), and add the **well-row
   term** for perforated cells. Biggest practical win; no deep change.
2. **Combined / seeded-priVars AD pass (optional)** — drive a state-seeded pass
   through `IntensiveQuantities::update` to assemble the Jacobian (and, batched,
   `Evaluation<numEq+P>` to get state + P parameters in one pass) entirely from
   the evaluator, decoupling it from the model's cached IQ. Mostly a performance
   / self-containment step; correctness already proven by Step A.
3. **Material-law parameter AD (deepest)** — make `EclEpsScalingPoints`
   construction `Evaluation`-typed (likely an upstream opm-common/opm-simulators
   change) to get exact endpoint `dR/dθ` in one pass and retire the FD there.

Items 1–2 stay inside `opm-adjoint` with public APIs; item 3 is the one that
warrants an upstream micro-PR. **The next plan should start at item 1.**

## 5. Notes for the generic single-slot AD work

**A. Where the AD seed must be set is *not* the primary variables.** The forward
code always seeds AD at the primary variables: `BlackOilPrimaryVariables::makeEvaluation`
returns `createVariable(value, varIdx)` (slot = the state variable) when
`timeIdx == linearizationType.time`, else `createConstant(value)` (frozen). That is why
forward AD computes `dR/dstate`. For a *parameter* gradient the differentiation variable
is the parameter, so the **state must be frozen** (`createConstant`, which is width-agnostic
— works for `Eval<1>`) and the **seed set where the parameter enters**.

**B. The kernels decide which parameters can be seeded *through* them.**
- `computeStorage<LhsEval>` **decays** its IQ inputs (`saturation`, `invB`, `porosity`) to
  `LhsEval` — i.e. takes the value. `calculateFluxes_` hard-casts `const Scalar trans =
  nbInfo.trans`. ⇒ **porosity, transmissibility and permeability cannot carry an AD seed
  through the kernels**; they are differentiated by **post-hoc scale on the kernel output**
  (storage/flux are linear in them). This is the existing single-slot `general-ad` path and
  is, by the kernel structure, the *only* way — porosity/perm/trans are **done** as single-slot
  Eval<1> generic AD.
- The interior-upwind **`mobility(phaseIdx, facedir)` is kept as `Evaluation`** in the flux
  (not decayed). ⇒ **mobility (relperm → end points) is the one quantity that flows as AD
  through the kernel**, so the "frozen-state IQ + seed in the IQ" mechanism applies precisely
  to it. That is the genuinely new generic-AD work, and the harder one (deferred).

**C. `LinearizationType` is a lever but not the mechanism.** Its `time` field gates
`createVariable` vs `createConstant` (so it *can* freeze the state), and it was introduced
partly to support other/sequential derivative schemes (`VarType::pressure`, `seqtransport`).
But it is **not flexible enough** to express "freeze the state and seed an arbitrary
parameter at its entry point" — a dedicated seeding seam in the IQ/material-law update is
what the parameter-AD work needs.

**D. Permeability appears in *well* parameters too.** The well connection transmissibility
factor (CTF / WPI, Peaceman ∝ `Kh`) depends on permeability, so `dJ/dperm` has a contribution
**through the well equations** (the perforation rate uses the perm-dependent well index). The
cell-evaluator perm gradient does **not** include this `dCTF/dK` term (it lives in the well
rows, like the endpoint well-row term). Flag when perm gradients are taken on decks where the
well index is perm-derived — the gradient is otherwise missing the well-coupling part.
