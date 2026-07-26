# opm-adjoint — code-reading guide & design summary

_Last updated: 2026-07-24. This file is meant to be **kept in sync with the code**._
_When you change the source, update the matching section here (see
[Keeping this doc current](#12-keeping-this-doc-current) at the bottom). Line numbers
are `file:line` at the time of writing — treat them as a hint, not gospel; grep the
method name if it has drifted._

This document is the answer to two questions:

1. **"How do I read this code?"** — [§3 Reading order](#3-reading-order-the-shortest-path-through-the-code)
   gives a numbered path from the entry point through the backward sweep.
2. **"What is the current design?"** — [§4](#4-architecture-at-a-glance)–[§10](#10-the-opm-simulators-hooks)
   describe the architecture file by file and trace the data flow.

For **status / what-works / how-to-run**, read [`STATUS.md`](STATUS.md) instead — this
guide deliberately does not duplicate the objective/derivative tables or the run
recipes. For **why decisions were made** and dead ends already tried, read
[`adjoint_plan.md`](adjoint_plan.md) and the `adjoint-project-state` auto-memory.

---

## 1. What the code computes (the one-paragraph version)

`flow_adjoint` computes the gradient `dJ/dθ` of a scalar objective `J` (a well-curve
misfit, a time-integrated BHP/rate, or a mean reservoir pressure) with respect to
simulation parameters `θ` — **porosity / pore volume, transmissibility, permeability,
saturation end-point scaling, and well-control targets** — using the **discrete
adjoint** of the blackoil forward simulation. It is a separate downstream OPM module
(the `opm-flowgeomechanics` pattern) that builds against a **near-stock
`opm-simulators`**: all forward-run recording is done through virtual-lifecycle
overrides in a derived problem class, and the backward sweep uses only public
simulator APIs plus a handful of small accessor commits on the `adjoint-hooks` branch
(see [§10](#10-the-opm-simulators-hooks)).

## 2. The math, just enough to read the code

The forward simulation solves, for each accepted substep `k = 0 … N-1`, a converged
nonlinear residual

```
R_k( x_k , x_{k-1} ) = 0          (reservoir + well unknowns)
```

where `x_k` is the state at the end of substep `k`. **v1 forces
`--enable-storage-cache=false`** precisely so that `R_k` is a pure function of the two
adjacent states `(x_k, x_{k-1})` — this is what makes a single re-linearization per
substep sufficient. The objective is a per-substep sum `J = Σ_k J_k(x_k, xw_k)`.

The **discrete adjoint** introduces a Lagrange multiplier `λ_k` per substep, solved
**backward** (`k = N-1 … 0`) from the transposed linearized system:

```
Aᵀ_k · λ_k  =  −(∂J_k/∂x_k)ᵀ  −  Bdiagᵀ_{k+1} · λ_{k+1}
```

- `A_k = ∂R_k/∂x_k` is the converged forward Jacobian (reservoir block after the
  well **Schur complement** `A ← A − CᵀD⁻¹B` is folded in).
- `Bdiag_{k+1} = ∂R_{k+1}/∂x_k` is the **cross-step** coupling block (the storage
  term links step `k+1` back to state `x_k`). It carries `λ_{k+1}` into step `k`'s
  right-hand side.
- Well unknowns are eliminated locally (`D` is block-diagonal per well); the well
  adjoint `λ_w` is recovered afterwards from `λ` and used for both well-control
  gradients and the objective RHS.

Once every `λ_k` is known, each parameter gradient is a contraction `g += λ_k · (∂R_k/∂θ)`
accumulated across the sweep. The three ways `∂R_k/∂θ` is obtained are the crux of
the design — see [§7](#7-the-three-gradient-mechanisms).

> **Known approximation:** the cross-step block `Bdiag` currently carries only the
> reservoir **storage** term. Explicit cross-step couplings (notably the `DRSDT`/`DRVDT`
> Rs/Rv caps) are **not** in `Bdiag`, which is the documented source of the SPE1 `bhp`
> gradient deviation. See [§11](#11-known-approximations--where-they-live-in-the-code).

## 3. Reading order (the shortest path through the code)

Read these in order; each builds on the last. Paths are under `opm/adjoint/` unless
noted.

| # | File | Why read it here |
|---|---|---|
| 1 | [`examples/flow_adjoint.cpp`](examples/flow_adjoint.cpp) | The **entry point**. One binary, four run modes dispatched from `main()`. Shows the TypeTag wiring and the `--adjoint-mode` switch. |
| 2 | [`AdjointFlowProblem.hpp`](opm/adjoint/AdjointFlowProblem.hpp) | The **forward hook**. See how recording piggybacks on `beginEpisode`/`endTimeStep` without touching the simulator. |
| 3 | [`AdjointRecorder.hpp`](opm/adjoint/AdjointRecorder.hpp) | **What gets captured** each substep (state snapshots, optionally the converged system). |
| 4 | [`AdjointMeta.hpp`](opm/adjoint/AdjointMeta.hpp) + [`AdjointStorage.hpp`](opm/adjoint/AdjointStorage.hpp) | The **archive contract**: the per-substep metadata list and the three storage backends. Skim; come back for detail. |
| 5 | [`AdjointReplay.hpp`](opm/adjoint/AdjointReplay.hpp) | **Backward re-linearization.** `replayStep()` recreates each converged system bitwise. This is Milestone A and the foundation of everything else. |
| 6 | [`AdjointSolver.hpp`](opm/adjoint/AdjointSolver.hpp) | **The core.** `run()` is the backward sweep: RHS build → Schur → λ solve → gradient accumulation. Read `run()` top-to-bottom first, then dive into the `accumulate*_` helpers. |
| 7 | [`AdjointLinearSolver.hpp`](opm/adjoint/AdjointLinearSolver.hpp) + [`transposeMatrix.hpp`](opm/adjoint/transposeMatrix.hpp) | **The transposed solve** `Aᵀλ = rhs`, serial + MPI. |
| 8 | [`AdjointObjective.hpp`](opm/adjoint/AdjointObjective.hpp) | **`J` and `∂J/∂x`.** Where objectives are defined and their RHS terms produced. |
| 9 | [`AdjointResidualEvaluator.hpp`](opm/adjoint/AdjointResidualEvaluator.hpp) | The **stencil-local residual/Jacobian evaluator** behind the `general`/`general-ad` gradient paths and the replay self-checks. |
| 10 | The rest, on demand | `AdjointEndpointGradients.hpp`, `AdjointParallel.hpp`, `AdjointObjectiveSpec.hpp` / `AdjointConfigFile.hpp` / `AdjointParameters.*`, `AdjointDeckWarnings.hpp`, `AdjointWellModel.hpp`, `AdjointSystemIO.hpp`. |

**If you only have 20 minutes:** read `flow_adjoint.cpp` `main()`, then
`AdjointSolver::run()` ([`AdjointSolver.hpp:192`](opm/adjoint/AdjointSolver.hpp)) with
this guide's [§6](#6-the-backward-sweep-in-detail-adjointsolverrun) open beside it.

## 4. Architecture at a glance

Everything is a header-only `template<class TypeTag>` in `namespace Opm` (TypeTag-free
spec/registry helpers live in `Opm::Adjoint` / `Opm::detail`). The only `.cpp` in the
module is `AdjointParameters.cpp` (parameter registration).

The three-stage data flow, all stages sharing one `AdjointArchive`:

```
   ┌─────────────────────── forward run (records) ───────────────────────┐
   │  FlowMain time loop                                                  │
   │    AdjointFlowProblem::beginEpisode  ─▶ recordInitial (episode 0)    │
   │    AdjointFlowProblem::endTimeStep   ─▶ recordSubstep(k)  ──┐        │
   └────────────────────────────────────────────────────────────┼────────┘
                                                                 ▼
                                                       AdjointArchive
                                              (HDF5 file │ directory │ in-RAM)
                                                                 ▲
   ┌──────────────── backward sweep (reads, k = N-1 … 0) ───────┼────────┐
   │  AdjointSolver::run                                          │        │
   │    AdjointReplay::replayStep(k) ─▶ recreate converged A_k,   │        │
   │        Bdiag_k, well D/B/C   ◀──── loadSnapshot_(k), (k-1) ──┘        │
   │    build rhs (objective ∂J/∂x + cross-term Bdiagᵀλ_{k+1})            │
   │    Schur: A ← A − CᵀD⁻¹B                                             │
   │    AdjointLinearSolver::solveTransposed  ─▶ λ_k                      │
   │    computeWellAdjoints_ + accumulate{Pv,Trans,Perm,Endpoint}…        │
   └──────────────────────────────────────────────────────────────────────┘
                                     ▼
              <CASE>.ADJOINT_GRADIENTS_{PV,TRANS,PERM,WELLCTRL,…}.txt
              <CASE>.ADJOINT_GRADIENTS.{csv,json}   (machine-readable)
```

`--adjoint-mode=onestop` fuses the two halves into one process by pointing the
recorder at an **in-RAM** archive (`AdjointFlowProblem::oneStopArchive()`), so no files
are written and the large archive is not re-read per optimizer iteration.

### Module file map

| File | Role | Core symbol |
|---|---|---|
| `examples/flow_adjoint.cpp` | entry point, TypeTag, mode dispatch | `AdjointMain`, `main()` |
| `AdjointFlowProblem.hpp` | forward-run recording hook | `AdjointFlowProblem` |
| `AdjointRecorder.hpp` | per-substep state/system capture | `AdjointRecorder` |
| `AdjointReplay.hpp` | backward re-linearization (Milestone A) | `AdjointReplay` |
| `AdjointSolver.hpp` | backward sweep + gradient accumulation (**core, 1639 lines**) | `AdjointSolver` |
| `AdjointObjective.hpp` | `J` and `∂J/∂x`, `∂J/∂xw` | `AdjointObjectiveFunction` |
| `AdjointResidualEvaluator.hpp` | stencil-local residual/Jacobian | `AdjointResidualEvaluator` |
| `AdjointLinearSolver.hpp` | transposed linear solve (serial+MPI) | `AdjointLinearSolver`, `TransposedParallelOperator` |
| `transposeMatrix.hpp` | explicit BCRS transposition | `transposeBlockMatrix*` |
| `AdjointStorage.hpp` | archive backends (dir / RAM / HDF5) | `AdjointArchive` |
| `AdjointMeta.hpp` | archive metadata + group paths | `AdjointMeta`, `AdjointGroups` |
| `AdjointSystemIO.hpp` | matrix/vector dump + compare | `AdjointMatrixDump`, `compare()` |
| `AdjointWellModel.hpp` | WGState serialization shim | `normalizeWGStateForSerialization` |
| `AdjointParallel.hpp` | MPI context, gathers, reductions | `AdjointParallel` |
| `AdjointEndpointGradients.hpp` | end-point keyword registry + perturber | `EndpointPerturber`, `endpointParamRegistry()` |
| `AdjointObjectiveSpec.hpp` | TypeTag-free objective description + parser | `Adjoint::ObjectiveSpec` |
| `AdjointConfigFile.hpp` | JSON `--adjoint-config` loader | `loadAdjointConfigFile()` |
| `AdjointParameters.{hpp,cpp}` | `--adjoint-*` registry + resolved `AdjointConfig` | `Opm::AdjointConfig` |
| `AdjointDeckWarnings.hpp` | end-of-run "what's approximate on this deck" scan | `warnAdjointDeckApproximations()` |

## 5. The forward side — recording

### 5.1 `AdjointFlowProblem` — the hook ([`AdjointFlowProblem.hpp`](opm/adjoint/AdjointFlowProblem.hpp))

`AdjointFlowProblem<TypeTag> : public FlowProblemBlackoil<TypeTag>` (`:53`). This is the
"geomech extension pattern": **no simulator changes** — recording is injected purely
through overrides.

- **ctor `:59`** calls `transmissibilities_.setStoreHalfTrans(true)` (`:65`) before
  `finishInit()` so the half-transmissibilities survive for the permeability chain rule.
- **`beginEpisode()` `:84`** lazily constructs the `AdjointRecorder` (`:95`/`:99`) and, at
  episode 0, snapshots the initial state (`recorder_->recordInitial()` `:102`).
- **`endTimeStep()` `:110`** fires once per **accepted** substep and calls
  `recorder_->recordSubstep(...)` (`:115`), tracking `substepInEpisode_`.
- **`setSubStepReport()` `:126`** and **`finalizeOutput()` `:135`** *name-hide* (not
  override) their non-virtual parents — they are called on the concrete type — to
  capture the Newton-iteration count and to flush the archive.
- **`oneStopArchive()` `:78`** is a `static shared_ptr<AdjointArchive>&`; when non-null the
  recorder writes into that shared in-RAM archive (the one-stop path set up in
  `flow_adjoint.cpp:126`).

### 5.2 `AdjointRecorder` — what is captured ([`AdjointRecorder.hpp`](opm/adjoint/AdjointRecorder.hpp))

- **`recordInitial()` `:106`** normalizes the well/group state
  (`normalizeWGStateForSerialization`, see §9) then writes the `initial` snapshot.
- **`writeSnapshot_(group)` `:119`** is the serialization primitive and is **symmetric
  with `AdjointReplay::loadSnapshot_`**. In **full** mode it serializes the whole
  `simulator_` (`:129`, via eWoms `Simulator::serializeOp`, which captures model +
  problem + vanguard/summaryState). In **slim** mode it writes only the dynamic pieces
  (model, problem, summary, udq, action `:123-127`) and lets replay reconstruct the
  static Eclipse Schedule from the deck.
- **`recordSubstep(...)` `:139`** writes the substep snapshot, optionally the converged
  system (`recordSystem_` `:215`, dumps `linearizer.residual()`/`.jacobian()` when
  `--adjoint-save-system`), and appends an `AdjointSubstepMeta`.
- **`finalize()` `:170`** writes `meta_` and a text parameter dump.

### 5.3 The archive ([`AdjointStorage.hpp`](opm/adjoint/AdjointStorage.hpp), [`AdjointMeta.hpp`](opm/adjoint/AdjointMeta.hpp))

`AdjointArchive` (`AdjointStorage.hpp:211`) dispatches to one of three backends, all
using the same `Serializer<MemPacker>` byte encoding as OPM's own `--save-step`:

- **`AdjointRawDirSerializer` `:65`** — one `.bin` per dataset in a directory tree
  (`--adjoint-file=<dir>` with no `.h5` suffix; dependency-free; per-rank subdirs in MPI).
- **`AdjointMemStore` `:162`** — a `group/dset → buffer` RAM map (the one-stop backend;
  byte-identical buffers, only the medium differs).
- **`HDF5Serializer`** — when `HAVE_HDF5`; the default `<CASE>.ADJOINT.h5` (parallel HDF5,
  per-rank datasets).

`AdjointMeta` (`AdjointMeta.hpp:69`) holds `schemaVersion` (currently `1`, `:71`),
flags (`systemSaved`, `storageCacheEnabled`, `slim`, `caseName`), and a
`vector<AdjointSubstepMeta>`. Each `AdjointSubstepMeta` (`:37`) records one accepted
substep with `solution(0)=x_k`, `solution(1)=x_{k-1}`, times, `dt`, and Newton count.
`AdjointGroups` (`:102`) maps to archive paths: `initial()`, `substep(k)`, `meta()`.

## 6. The backward sweep in detail — `AdjointSolver::run()`

This is the heart of the module: [`AdjointSolver.hpp:192`](opm/adjoint/AdjointSolver.hpp),
class `AdjointSolver<TypeTag>` (`:84`). Key member type `PEval =
DenseAd::Evaluation<Scalar,1>` (`:100`) is the **one-slot seeded parameter AD** used by
the exact `general-ad` path.

The **ctor** (`:113`) builds `replay_`, `parallel_`, `objective_`; calls
`replay_.setComputeBdiag(true)`; configures the transposed `linearSolver_` (umfpack →
ilu0 fallback in parallel `:131`); parses the requested endpoint keywords (`:145`); and
sets `generalAd_` / `generalGradients_` from `--adjoint-gradient-method`.

**`run()` loops `for k = N-1 … 0`** (`:224`). Per substep:

1. **Replay** `replay_.replayStep(k, …)` (`:228`) recreates the converged system for
   substep `k` — Jacobian `A_k`, the cross-step `Bdiag_k`, and the well matrices
   `D/B/C` — bitwise (timed into `timing_.replay`).
2. On the first iteration, `checkWellContributionSparsity_()` (`:234`) aborts early with
   a clear message if multi-perforation wells were run without
   `--matrix-add-well-contributions`.
3. **Objective value** accumulates `objective_.stepValue(...)` (`:240`).
4. **RHS build** (`:247-257`):
   - `rhs = 0`; `objective_.addStepGradient(...)` adds the reservoir `∂J/∂x_r` (`:249`);
     `rhs *= -1`.
   - `addWellObjectiveRhs_(dt, rhs)` (`:251`) adds the well-objective contribution
     `Bᵀ D⁻ᵀ ∂J/∂x_w` (a small dense per-well transposed solve, scattered to perforated
     cells).
   - **cross-step term** `rhs_i -= Bdiagᵀ_{k+1,i} · λ_{k+1,i}` via
     `bdiagPrev[i].usmtv(...)` (`:255`).
5. **Schur reduction** (`:263`): `wellModel().addWellContributions(jacobian)` folds
   `A ← A − CᵀD⁻¹B` in place (timed `timing_.schur`).
6. **λ solve** (`:269`): `linearSolver_.solveTransposed(jacobian.istlMatrix(), rhs,
   lambda)`, then `parallel_.makeConsistent(lambda)` (`:273`) (timed `timing_.solve`).
7. **Gradient accumulation** (`:277-286`):
   - `computeWellAdjoints_(k, dt, lambda)` (`:277`) recovers `λ_w` and accumulates
     **well-control** gradients.
   - `accumulateGeneralGradients_` (`:279`) **or** `accumulateGradients_` (`:281`) for
     PV/TRANS/PERM (method-dependent, see §7).
   - `accumulatePermWellTerm_` (`:283`) adds the `dCTF/dperm` well-coupling term.
   - `accumulateEndpointGradients_` (`:286`) for end-point scaling.
8. **Carry over** `bdiagPrev = replay_.bdiag()` and `lambdaPrev = lambda` (`:289`) into
   the next (earlier) substep.

After the loop: `reportTiming_`, `writeResults_(objectiveValue)` (text +
`writeFieldOutput_` CSV/JSON), and `warnAdjointDeckApproximations`.

### Well adjoint recovery — `computeWellAdjoints_` (`:469`)

`λ_w = −D⁻ᵀ ( ∂J/∂x_w + Σ_perf C·λ_r )`. Because only one control mode binds per well
per substep and its equation is always the **last** well row, the well-control gradient
is simply `dJ/d(target) = −λ_w[last]`. It is accumulated both lumped (`controlGradient_`
`:521`) and **per active mode** (`controlGradientByMode_` `:533`, tagged by the converged
production/injection cmode) — the per-mode split is the correctness fix for wells that
switch control mode mid-run.

## 7. The three gradient mechanisms

`∂R_k/∂θ` is obtained three different ways depending on how `θ` enters the residual.
This is the single most important design fact to internalize.

### (1) Analytic / seeded-AD — PV, TRANS, PERM

Storage is **linear** in the PV multiplier and the TPFA flux is **linear** in the
transmissibility multiplier, so `∂R/∂θ` is either a closed-form inner product or an
exact one-pass seeded-AD derivative — **no extra residual evaluation, no FD step.**

- **`accumulateGradients_` `:538`** — the `analytic` oracle: `accumulatePvAnalytic_`
  `:571` (`g += λ·(V/dt)(S(x_k)−S(x_{k-1}))`) and `accumulateTransAnalytic_` `:857`
  (`g += (λ_I−λ_J)·flux_f`). PERM is a chain-rule post-process of the trans gradient
  through half-transmissibilities (`analyticPermLocalGradient_` `:1546`).
- **`accumulateGeneralGradients_(dt, lambda, useAd)` `:553`** — the default
  `general`/`general-ad` path, driven by the stencil-local
  `AdjointResidualEvaluator`: `accumulatePvGeneral_` `:594` (diagonal `evalStorageTerm`),
  `accumulateTransGeneral_` `:683` (two directed `evalFaceTerm` per face), and
  `accumulatePermGeneral_` `:772` (per cell/direction harmonic-mean half-trans ratio).
  `seed_()` `:562` produces the one-slot AD scale (`value 1, d/dθ = 1`). `general-ad`
  reproduces the analytic oracle to machine precision (SPE9: PV 6e-16, PERM 1e-15).

The evaluator computes gradients for **all** parameters of a family in one backward
sweep at cost `cell × ~7 stencil` (not `cell × Ncells`), because each parameter touches
only a cell's own residual plus its TPFA neighbours.

### (2) Residual finite-difference — end-point scaling

The saturation end points (`SWL … SGU`, `KRW/KRG/KRORW/KRORG`, `PCW/PCG`) enter through
the material-law scaling with **no closed-form `∂R/∂θ`**. So for each requested end point
of each affected cell the code perturbs the scaled point by `±h`
([`AdjointEndpointGradients.hpp`](opm/adjoint/AdjointEndpointGradients.hpp),
`EndpointPerturber::apply` `:135`), rebuilds the cell's intensive quantities at `x_k`
and `x_{k-1}` via the public `IntensiveQuantities::update`, re-evaluates the local
residual, and central-differences it — contracted with **both** `λ_r` and `λ_w`
(relperm reaches the perforation source term).

- **`accumulateEndpointGradients_` `:910`** dispatches to
  `accumulateEndpointGradientsStencil_` `:1165` (default `stencil`: perturb only cell
  `i`, touch `{i} ∪ nb(i)`; ~56× faster) or `accumulateEndpointGradientsWhole_` `:990`
  (whole-domain re-linearize, the `whole` oracle). Perforated cells fall back to the
  whole-domain contraction so the `λ_w·dR_w` term is captured.

### (3) Well-equation row — well-control targets

The control target appears directly in the last well-equation row (`R = rate|bhp −
target`), so `dJ/du = −Σ_k λ_w,k[last]` with **no residual re-evaluation** — computed
inside `computeWellAdjoints_` (§6).

**One hybrid case: `dCTF/dperm`.** At defaulted-CTF perforations the CTF enters the well
residual and reservoir source **linearly** (`Tw = CTF·trans_mult`, `cq =
−Tw·mob·drawdown`). `accumulatePermWellTerm_` `:1054` perturbs the well index (via the
`WellInterfaceGeneric::wellIndex()` hook), re-linearizes, contracts `λ_r·R_r + λ_w·R_w`
on both sides, and chains to perm through the dominant Peaceman term (`d(ln CTF)/dK =
1/(2K)`). This closes the SPE1 producer-perforation gap from 95.6 % to 0.2 %.

## 8. Objectives — `AdjointObjectiveFunction` ([`AdjointObjective.hpp`](opm/adjoint/AdjointObjective.hpp))

`AdjointObjectiveFunction<TypeTag>` (`:88`) implements the per-substep sum `J = Σ_k J_k`.
The three public entry points the solver uses:

- **`stepValue(sim, k, N, dt)` `:190`** — the `J_k` contribution: `PressureAverage`
  (mean cell pressure at the **final** substep `:194`), `WellBhp` (`dt·bhp` `:211`), or a
  sum of `RateTerms` (`:219`).
- **`addStepGradient(sim, k, N, gradient)` `:241`** — the reservoir RHS `∂J_k/∂x_r`
  (non-zero only for `PressureAverage`, at the pressure index `:256`).
- **`termWeight(sim, term, dt, q)` `:277`** — the `∂J/∂q` weight for rate-family terms
  (`Rate` / `MatchConst` / `MatchRef` `:284`).

Rate terms (`RateTerm` `:101`) carry a well, phase/component index, weight, optional
reference `ESmry` case and time window, and a lazily-loaded observed curve
(`loadObservations_` `:373` → WOPR/WWPR/WGPR with unit→SI conversion `:399`). Well names
may be shell patterns, expanded against the schedule at construction (`expandWells_`
`:447`).

The objective **description** is TypeTag-free
([`AdjointObjectiveSpec.hpp`](opm/adjoint/AdjointObjectiveSpec.hpp),
`Adjoint::ObjectiveSpec`), shared by the CLI-string parser (`ObjectiveSpec::parse`
`:112`, handling the `:w=/:t0=/:t1=` suffix tokens and the `matchsum` multi-term grammar)
and the JSON loader ([`AdjointConfigFile.hpp`](opm/adjoint/AdjointConfigFile.hpp)
`detail::parseObjective` `:95`).

## 9. Supporting machinery

- **`AdjointResidualEvaluator`** ([`AdjointResidualEvaluator.hpp`](opm/adjoint/AdjointResidualEvaluator.hpp),
  class `:62`) — a value-type-generic reimplementation of
  `TpfaLinearizer::linearize_cell + linearize_source_terms` that reads only a cell's own
  IQ plus its TPFA-neighbour IQs. `evalStorageTerm` `:163` (PV), `evalFaceTerm` `:196`
  (trans/perm, scale applied post-hoc to the flux value — exact because the flux is
  linear in trans), and `evalJacobian` `:240` (the full reservoir Jacobian, used by the
  replay self-check). The nested `SeparateIQ` `:328` is an adjoint-owned IQ array so the
  evaluator never disturbs the model's cache.
- **`AdjointLinearSolver`** ([`AdjointLinearSolver.hpp`](opm/adjoint/AdjointLinearSolver.hpp),
  class `:204`) — `solveTransposed` `:264` picks UMFPACK on an explicit transpose
  (`solveDirect_` `:318`) or a FlexibleSolver on the transpose (`solveSerial_` `:373`:
  ilu0 / cpr / **cprt** / a JSON config, with quasi-IMPES weights). The MPI path
  (`solveParallel_` `:405`) uses `TransposedParallelOperator` `:110` — an **exact** `Aᵀx`
  via scattering over the forward matrix's complete interior rows plus
  `addOwnerCopyToOwnerCopy`, because a plain local transpose drops cross-rank coupling
  (this was a real bug: correct at np=2, wrong at np≥3). Transposes are cached and
  refilled values-only across the sweep (`transposeBlockMatrixInto`,
  [`transposeMatrix.hpp:116`](opm/adjoint/transposeMatrix.hpp)) since the pattern is
  constant.
- **`AdjointParallel`** ([`AdjointParallel.hpp`](opm/adjoint/AdjointParallel.hpp),
  class `:66`) — interior/ghost masks, cartesian-index map, `ownsFace` (counts each
  global face once), reductions, and the rank-0 gathers (`gatherCellField`,
  `gatherCellIds`, `gatherLines`) that make serial and parallel output byte-identical.
  `makeConsistent` `:281` (`copyOwnerToAll`) is applied to `λ` after each solve.
- **`AdjointWellModel`** ([`AdjointWellModel.hpp`](opm/adjoint/AdjointWellModel.hpp)) —
  just `normalizeWGStateForSerialization` `:69`, which commits + `updateNupcolWGState`
  so the report-0 snapshot round-trips (the `updateNupcolWGState` member is protected;
  it is reached via a member-pointer trick in `detail::WGStateNormalizer` `:48`).
- **`AdjointSystemIO`** ([`AdjointSystemIO.hpp`](opm/adjoint/AdjointSystemIO.hpp)) —
  `AdjointMatrixDump`/`AdjointVectorDump` (serializable CSR / block vector) and the
  `compare()` functions with the relative/absolute-floor logic behind the bitwise replay
  checks.
- **`AdjointDeckWarnings`** ([`AdjointDeckWarnings.hpp`](opm/adjoint/AdjointDeckWarnings.hpp)) —
  `warnAdjointDeckApproximations` `:60` scans the deck at end-of-run and prints, per
  category, every quantity the adjoint does not handle exactly: *approx-gradient*
  (hysteresis, DRSDT/DRVDT incl. 0, VAPPARS, guide-rate group control, defaulted-CTF
  perm, well explicit quantities), *approx-replay* (RESV/VREP/REIN voidage group
  control), *neglected* (aquifers, extended network). Called from both replay and solver.
- **`AdjointParameters`** ([`AdjointParameters.hpp`](opm/adjoint/AdjointParameters.hpp) +
  `.cpp`) — the `--adjoint-*` parameter registry (note the current defaults:
  `--adjoint-linear-solver=umfpack`, `--adjoint-gradient-method=general-ad`,
  `--adjoint-endpoint-method=whole`, `--adjoint-perm-well-term=true`) and the resolved
  `Opm::AdjointConfig` bundle (`fromParameters` `:167`, JSON base < CLI override).

## 10. The opm-simulators hooks

The module is *near*-stock: it needs a small set of accessor commits on the
`opm-simulators` **`adjoint-hooks`** branch (master + these, HEAD `9b07194fe`). Keep this
list current — it is the module's entire coupling to upstream:

1. `B/C/D` getters on `StandardWellEquations` (expose the well Jacobian blocks for the
   Schur elimination and `λ_w` recovery).
2. A `SummaryState` **assembly seam** on `WellInterface` (`assemblySummaryState` /
   `setAssemblySummaryState`) — Stage-2 keystone routing the assembly's control
   derivation through a settable pointer (default = live summaryState; byte-identical
   forward).
3. Half-transmissibility exposure on `Transmissibility` (`setStoreHalfTrans`) — for the
   permeability chain rule.
4. `BlackoilWellModel::assembleWellEqGivenControls(dt)` — assembles the well residual
   from **restored** controls without re-deriving them (the replay's final linearization
   uses this so group-controlled and shut-well decks replay without re-running the
   forward-only control advance).
5. Mutable `WellInterfaceGeneric::wellIndex()` — for the `dCTF/dperm` perturbation.

`opm-common` and `opm-grid` stay on their normal branches.

## 11. Known approximations — where they live in the code

These are the deliberate v1 simplifications. Each is flagged at runtime by
`warnAdjointDeckApproximations`. See STATUS.md for the quantitative story.

| Approximation | Where | Effect |
|---|---|---|
| **`Bdiag` = reservoir storage only** — explicit cross-step terms (`DRSDT`/`DRVDT` Rs/Rv caps) missing | `AdjointReplay::captureBdiag_` (`AdjointReplay.hpp:327`) | The documented SPE1 `bhp` gradient deviation (~3 % single-cell). Fix = add the capped-cell `dR_k/dRs_{k-1}` term to `Bdiag`. |
| **End-point initial-state cross-term** not included | `AdjointEndpointGradients.hpp` (docstring), endpoint accumulators | FD end-point checks require explicitly-initialized decks (no equilibration path). |
| **Voidage (VREP/RESV/REIN) group control** replay-inexact | well assembly / `AdjointDeckWarnings` | Fix designed (WellAssemblyState Stage-2, `adjoint_refactoring.md`), not implemented. |
| **Multisegment wells** — reservoir replays bitwise, but any *well-objective* gradient aborts | `AdjointSolver.hpp:415` (`dynamic_cast<StandardWell>`) | MSW D/B/C elimination is a separate task. |
| **Aquifers / extended network** neglected | flagged only | Out of v1 scope. |

## 12. Keeping this doc current

This file is intended to be edited alongside the code. Practical rules:

- **When you rename or move a method**, update the `file:line` and symbol references in
  the relevant section. The high-churn spots are [§6](#6-the-backward-sweep-in-detail-adjointsolverrun)
  (`AdjointSolver::run` and the `accumulate*_` helpers) and [§7](#7-the-three-gradient-mechanisms).
- **When you add a gradient family or objective**, add a row to the file map ([§4](#module-file-map)),
  and extend [§7](#7-the-three-gradient-mechanisms) / [§8](#8-objectives--adjointobjectivefunction-adjointobjectivehpp).
- **When you add or remove an opm-simulators hook**, update [§10](#10-the-opm-simulators-hooks) —
  that list is the module's whole upstream contract and must not drift.
- **When you close or discover an approximation**, move the row in [§11](#11-known-approximations--where-they-live-in-the-code)
  and keep it consistent with `warnAdjointDeckApproximations` and STATUS.md.
- **When the archive schema changes**, bump `AdjointMeta::currentSchemaVersion` and note
  it in [§5.3](#53-the-archive-adjointstoragehpp-adjointmetahpp).
- Line numbers drift; **symbol names are the durable anchor** — if a line reference looks
  wrong, `grep` the method name. Update the `_Last updated_` date at the top when you
  make a substantive pass.

Related docs (do not duplicate them here): [`STATUS.md`](STATUS.md) (status + run
recipes), [`adjoint_plan.md`](adjoint_plan.md) (design rationale + PR sequencing),
[`adjoint_status.md`](adjoint_status.md) (feature catalog + explicitly-updated-quantities
table), [`adjoint_refactoring.md`](adjoint_refactoring.md) (WellAssemblyState Stage-1/2/3),
[`adjoint_testing.md`](adjoint_testing.md), [`adjoint_general_ad_review.md`](adjoint_general_ad_review.md),
[`adjoint_code_assessment.md`](adjoint_code_assessment.md) (code review findings).
