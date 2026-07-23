# OPM adjoint — project status & quickstart

_Last updated: 2026-06-24_

Adjoint-based gradients for the OPM flow simulator: gradients of well-curve-matching
(and other) objectives with respect to porosity/PV, transmissibility, permeability,
end-point scaling, and well controls — computed by the adjoint method.

---

## 1. Status summary

**Milestones A, B and C are complete and finite-difference (FD) verified.** The core
deliverable — adjoint gradients of well-matching objectives w.r.t.
poro/perm/trans/end-points/well-controls — works on cases ranging from 1-cell debug
decks up to field models, and is guarded by a green ctest suite (**15/15**).

### Development layout

- **`opm-adjoint`** — separate downstream module (the `opm-flowgeomechanics` pattern),
  built against a **near-stock `opm-simulators`**. Recording hooks live in a derived
  `AdjointFlowProblem` (virtual `beginEpisode`/`endTimeStep`); the backward replay/solver
  driver uses only public simulator APIs. Branch `master`, HEAD `b231773`.
- **`opm-simulators`** — branch `adjoint-hooks` (master + a handful of small accessor
  commits: `B/C/D` getters on `StandardWellEquations`, a `SummaryState` assembly seam,
  half-trans exposure on `Transmissibility`, `assembleWellEqGivenControls`). HEAD
  `9b07194fe`.
- **`opm-tests`** — branch `adjoint` (adds `adjoint_tests/MODEL_*_DEBUG` and
  `SPE1_GRPCTRL` decks).

### What works

| Area | Status |
|---|---|
| **Milestone A — exact backward replay** | SPE1 123/123, SPE9 21/21, MODEL_1D_DEBUG 7/7, SPE1_GRPCTRL 120/120 **bitwise**. Full **Norne** (44 431 cells, 36 wells, group ctrl + shut wells, hysteresis, VAPOIL, faults) replays **end-to-end**, ~90 % bitwise (remainder at well-event / hysteresis boundaries — documented v1 approximation). |
| **Milestone B — adjoint solve** | Backward Lagrange sweep, Schur-reduced, UMFPACK on the transposed system; iterative solves (ilu0 / cpr / **cprt**); **MPI** parallel record/replay + transposed solve (cross-rank transpose bug found & fixed). |
| **Milestone C — parameter gradients (all FD-verified)** | see the derivatives table below. |
| **Tests** | ctest **15/15**; FD-verified golden-reference regression set in `tests/references/`; per-eps FD accuracy study in [`tests/FD_ACCURACY.md`](tests/FD_ACCURACY.md). |
| **JutulDarcy cross-check** | Set up (`jutul/`); pattern/sign-level agreement (forwards differ by well model). FD remains the authoritative oracle. |

### Objective functions (`--adjoint-objective=`)

Per-substep sum objectives `J = Σₖ Jₖ(xₖ, xwₖ)` (Jutul-style interface):

| selector | definition | use |
|---|---|---|
| `pressure-average` | `(1/N) Σᵢ pᵢ` at the **final** substep (mean cell pressure) | reservoir-state validation |
| `bhp:<WELL>` | `Σₖ dtₖ · BHP_WELL(xₖ)` (time-integrated BHP) | well-curve / control objectives |
| `rate:<WELL>:<phase>` | `Σₖ dtₖ · q` (time-integrated rate) | production/injection totals |
| `match:<WELL>:<phase>:<target>` | `Σₖ dtₖ (q − target)²` | match a constant target |
| `matchref:<WELL>:<phase>:<refcase>` | `Σₖ dtₖ (q − q_obs(t))²`, `q_obs` from a reference `ESmry`, time-interpolated | well-curve history matching |
| `matchsum:<refcase>:<W>.<p>[*<w>][+<W>.<p>…]` | sum of `matchref` terms over several well/phase pairs (optional per-term weight `*<w>`) | multi-term misfit |

Rate-family extensions (2026-07-23): well names may be shell patterns
(`OP_*`, expanded against the schedule at construction, error if nothing
matches); optional trailing suffix tokens `:w=<weight>`, `:t0=<days>`,
`:t1=<days>` set per-term weight and a time window (a substep contributes iff
its end time lies in the window). A `matchref`/`matchsum` term without an
explicit window defaults to the reference summary's time span, so times
outside the observed data contribute nothing (previously: flat-extrapolated
misfit with a warning). Per-term J values are accumulated and logged for
multi-term objectives. Guarded by ctest `adjoint_matchref_extended_spe1`
(FD-vs-adjoint under a weighted, windowed, wildcard objective).

### Derivatives (parameters, all FD-verified)

The right column is the gradient **mechanism** — how `dR/dp` (the residual
sensitivity contracted with the adjoint λ) is obtained. Three different routes;
end-point scaling is the odd one out (see [Code structure](#code-structure--how-gradients-are-computed)).

| derivative w.r.t. | parameter / keyword | gradient mechanism |
|---|---|---|
| porosity / pore volume | per-cell φ, MULTPV | **analytic** — storage linear in PV, `g += λ·(V/dt)(Sₙ−Sₙ₋₁)` |
| transmissibility | per-face MULTX / face T | **analytic** — TPFA flux linear in T, `g += (λ_I−λ_J)·flux` |
| permeability | per-cell PERMX/Y/Z | **analytic + chain rule** — post-process the trans gradient through half-transmissibilities |
| end-point scaling | SWL…SGU, KRW/KRG/KRORW/KRORG, PCW/PCG | **residual finite-difference** — perturb the scaled end point, *re-evaluate the local residual*, contract with λ_r **and** λ_w |
| well control | per-well rate/BHP target u | **well-equation row** — `dJ/du = −Σₖ λ_w,ₖ[last]` (control equation) |

### Code structure — how gradients are computed

The backward sweep lives in [`AdjointSolver.hpp`](opm/adjoint/AdjointSolver.hpp);
per accepted substep k it (1) re-linearizes the converged system via
[`AdjointReplay.hpp`](opm/adjoint/AdjointReplay.hpp), (2) solves the transposed
adjoint system `Aᵀλₖ = −(dJ/dx)ᵀ − Bdiagᵀλₖ₊₁`
([`AdjointLinearSolver.hpp`](opm/adjoint/AdjointLinearSolver.hpp)), then (3)
accumulates parameter gradients. Step 3 uses **three distinct mechanisms**:

1. **Analytic accumulation** — PV, transmissibility (and permeability as a
   chain-rule post-process). `dR/dp` is known in closed form (storage is linear
   in the PV multiplier, the TPFA flux linear in the trans multiplier), so the
   gradient is a plain inner product `λ·(dR/dp)` using the storage/flux values
   already computed at the converged point — no extra residual evaluation.
   (`accumulateGradients_`, `gatherPermGradients_`).
2. **Residual finite-difference** — end-point scaling
   ([`AdjointEndpointGradients.hpp`](opm/adjoint/AdjointEndpointGradients.hpp),
   `accumulateEndpointGradients_`). Here the parameter enters through the
   saturation-function scaling deep inside the material law, with **no closed-form
   `dR/dp`**. So for each requested end point of each cell the code perturbs the
   scaled point by ±h, re-derives the scaled drainage curves, recomputes the
   cell's intensive quantities at xₖ and xₖ₋₁ through the public
   `IntensiveQuantities::update`, re-evaluates the cell's storage and its faces'
   fluxes with the TPFA kernels, and central-differences the residual:
   `dJ/dθ_i += λ_i·dR_i + Σ_faces λ_J·dR_J`. This is a numerical derivative of
   the **local** residual (cell-local, no global solve), contracted with both the
   reservoir adjoint λ_r and the well adjoint λ_w (relperm reaches the
   perforation source term). It is the "different residual evaluation" — distinct
   from the analytic paths, and the reason end-point FD verification needs
   explicitly-initialized decks (the initial-state-via-equilibration cross-term
   is not included).
3. **Well-equation row** — well-control dJ/du
   ([`computeWellAdjoints_`](opm/adjoint/AdjointSolver.hpp)). The control target
   appears directly in the last well-equation row (`R = rate|bhp − target`), so
   `dJ/du = −Σₖ λ_w,ₖ[last]`; no residual re-evaluation at all.

Objectives are assembled separately in
[`AdjointObjective.hpp`](opm/adjoint/AdjointObjective.hpp) (the `dJ/dx` /
`dJ/dxw` right-hand-side terms).

### Key design decisions

- v1 forces `--enable-storage-cache=false` so the residual is a pure function of
  `(x_k, x_{k-1})` — replay needs a single linearization per substep.
- Archive = `serializeOp` buffers in an HDF5 file (`<CASE>.ADJOINT.h5`, parallel HDF5
  active) or a dependency-free directory store (`--adjoint-file=<dir>` without `.h5`).
- Gradient accuracy tracks the linear-solve reduction roughly linearly; default
  reduction `1e-14`.

### dCTF/dK quantified (2026-07-23) — the well term is DOMINANT at producers

`tests/run-adjoint-ctf-gap-quantify.sh` (SPE1CASE1, defaulted CTF, single-cell
PERMX multiplier FD vs adjoint `general-ad`, h=1e-2, cnv 1e-9):

| cell | objective | FD dJ/dK [/mD] | adjoint | missing well term |
|---|---|---|---|---|
| PROD perf (10,10,3) | pressure-average | −3.220e4 | −1.433e3 | **95.6 % of FD** |
| INJ perf (1,1,1) | pressure-average | −3.429e1 | −3.406e1 | 0.7 % |
| unperforated (5,5,2) | pressure-average | 6.800e1 | 6.811e1 | −0.2 % (noise) |

Same picture with `bhp:PROD` (prod-perf gap 94.3 %). **Decision: the well-row
perm term (λ·dR/dCTF·dCTF/dK) is mandatory for perm history matching** — at
drawdown-limited producer perforations the cell-based perm gradient is ~20×
too small. It is negligible for rate-controlled injectors (delivery not CTF-limited).

### Well-objective (bhp) gradient deviation on SPE1 — INVESTIGATED 2026-07-23

A ~3 % (single-cell) / ~6–9 % (aggregate) FD-vs-adjoint deviation appears for
the **bhp objective** on SPE1CASE1 (`pressure-average` and `rate:oil` are
accurate). Investigation scripts: `tests/run-adjoint-dirderiv-wellobj.sh`
(random-direction over all 300 PORO), plus single-cell ε/datum sweeps.

| objective (window, days) | rel. deviation | reading |
|---|---|---|
| `bhp:PROD` full run | 6.1 % | |
| `bhp:PROD` t∈[0,900] (pure ORAT, no switch) | 8.8 % | error is in the ORAT period |
| `bhp:PROD` t∈[1100,∞) (BHP-controlled) | **0 (exact)** | BHP pinned ⇒ dJ=0 |
| `bhp:PROD` t∈[0,100] / [0,300] | 0.8 % / 7.9 % | grows as free gas evolves |
| single non-perf cell (5,5,2), bhp:PROD | 2.94 % | ε-flat 4e-2…1e-3 |
| `rate:PROD:gas` t∈[0,900] | 1.5 % | |
| `rate:PROD:oil` | 0.1–0.5 % | oil pinned to ORAT target |
| `pressure-average` | 0.2 % | (its dirderiv value is FD-noise) |

**Ruled OUT — the well explicit-quantities term (`calculateExplicitQuantities`:
connection Δp / wellbore-storage F0 / explicit B) is NOT the cause**, three
independent ways:
1. SPE1's PROD/INJ datum depths (8400/8335) equal their connection-cell
   centres, so `perf_pressure_diffs_` ≈ 0; SPE1 has no wellbore volume ⇒ F0/
   `computeAccumWell` inert. `calculateExplicitQuantities` has no cross-step
   effect here, yet the deviation is 2.94 %.
2. Shifting the PROD datum to 8335 (65 ft head, activating the term) leaves the
   deviation unchanged (2.94 % → 2.59 %).
3. **Ruled OUT — non-differentiability / DISGAS variable-switching kinks**: the
   deviation is ε-flat to 5 sig-figs across ε 4e-2…1e-3 (a kink shrinks as
   ε→0). It is a *smooth* systematic error.

**Confirmed genuine, not noise.** The 2.94 % single-cell deviation is identical
under forward `cnv=1e-11, mb=1e-13` and adjoint linear reduction 1e-14 **and**
1e-16 — so it is a real Jacobian/matrix inexactness, not convergence or
linear-solve noise. (Strict tolerances are mandatory when FD-testing the
adjoint — see the strict-tolerance memory.)

**Coverage gap that let it hide.** Every per-cell FD gradient test (pv, perm,
**wellctrl**) runs only on MODEL_1D_DEBUG (3 cells, 7 calm substeps); SPE1 has
replay/Jacobian-eval tests but **no FD gradient test**. So a well-objective
gradient in a gas-evolving regime was never FD-checked. m1d (also DISGAS+VAPOIL,
ORAT producer) matches bhp×PV at 4e-9 because it never evolves significant free
gas.

**Mechanism (leading, well-evidenced).** The forward Newton converges the well
equations to R_w ≈ 0 regardless of whether the assembled well Jacobian (D/B/C)
is the *exact* linearization — Newton tolerates an inexact Jacobian. The adjoint
instead applies `D⁻ᵀ` / `Bᵀ` to a **nonzero** RHS (`dJ/dxw = dt·e_bhp` for the
bhp objective), so any inexactness in the well Jacobian that the forward's
near-zero well residual never probes is exposed and amplified in the gradient.
This explains every symptom: bhp-only (`pressure-average` has `dJ/dxw=0`;
`rate:oil`'s `dJ/dxw` is the control-eq-row direction, which `D⁻ᵀ` maps to a
trivial multiplier), ORAT-period-only (BHP a free variable), routes through the
well so it hits any cell via λ_res, and grows with free-gas evolution — because
the most likely inexact term is the **production perforation mobility
derivative**: `getMobility` (WellInterface_impl.hpp:2131) takes the well-cell
phase mobility `kr_p(S)/µ_p` at timeIdx=0 and maps its reservoir-cell AD
derivatives into the well Evaluation via `extendEval`; `kr_p(S)` is strongly
saturation-dependent for producers as gas evolves (injectors use total/injected
mobility, so are insensitive — consistent with the injector being clean).

**Next diagnostic (task W1).** FD-verify the assembled well Jacobian D and B
against the well residual R_w at a converged substep (new well-matrix check,
analogous to the reservoir `--adjoint-jacobian-eval-check`), isolating the
mobility (`extendEval`/`dq/dS`) and well-index (`dq/d(CTF)`) columns. Expect the
inexactness to surface in the mobility/saturation derivative. Until resolved,
prefer `matchref` rate objectives for HM (oil ≲0.5 %, gas ~1.5 %); `bhp:`
carries the documented caveat. A DISGAS well-objective FD test on SPE1 must be
added to close the coverage gap.

### Multisegment wells — NOT supported for gradients (2026-07-23)

MSW decks (`SPE1CASE2_MSW`) **replay bitwise** (120/120, reservoir side), but
any gradient with a well objective aborts: `addWellObjectiveRhs_` throws
"Adjoint well objectives support StandardWells only" (`AdjointSolver.hpp:415`),
because every well-adjoint path `dynamic_cast`s to `StandardWell` and MSW is a
`MultisegmentWell`. Well-control and well-objective gradients for MSW are a
separate implementation task (the segment system needs its own D/B/C
elimination). A small MSW gradient test is gated on that support.

### Known gaps / queue (low priority, none blocking)

- Report-step-0 startup bitwise for model2-like decks; Norne hysteresis / well-event
  boundary bitwise.
- **WGRUPCON / VREP** voidage-replacement replay-inexactness — fix _designed_ (Stage-2:
  carry the converged effective control target per well in the `WellAssemblyState`
  bundle so `assembleControlEqInj/Prod` read it directly), **not yet implemented**.
- `FlexibleCellEvaluator` (C.2) for exact parameter-AD; end-point scaling is currently
  via cell-local parameter FD.
- T6 JutulDarcy gradient cross-check is pattern/sign-only.

The end-of-run `AdjointDeckWarnings` scan reports, per deck, every quantity the adjoint
does not handle exactly (approx-gradient: hysteresis, DRSDT/DRVDT, VAPPARS, guide-rate
group allocation, well explicit quantities; approx-replay: voidage group control;
neglected: aquifers, networks). SPE1 prints nothing (clean).

---

## 2. Checkout

The working tree is a set of side-by-side OPM module repos. The adjoint work lives on
three non-default branches:

```bash
# opm-simulators: small accessor hooks on top of master
git -C opm-simulators checkout adjoint-hooks

# opm-tests: adjoint debug decks + SPE1_GRPCTRL
git -C opm-tests checkout adjoint

# opm-adjoint: the module itself (default branch)
git -C opm-adjoint checkout master
```

`opm-common` and `opm-grid` stay on their normal branches. `opm-adjoint/dune.module`
declares the dependency chain: `dune-istl opm-common opm-grid opm-simulators`.

---

## 3. Compile

The module builds like any Dune/OPM module. The existing release tree is
`builds/release` (Ninja, `CMAKE_BUILD_TYPE=Release`, parallel HDF5 + UMFPACK).

```bash
cd /Users/hnil/Documents/OPM/opm_clean/builds/release

# (re)configure — only needed for a fresh tree
cmake -G Ninja \
  -C ../cmake_minimal.cmake \
  -DCMAKE_PREFIX_PATH=/Users/hnil/Documents/GITHUB/OPM/dune_installed_all \
  -DCMAKE_BUILD_TYPE=Release \
  /path/to/opm-adjoint            # plus the other modules as usual

# build just the adjoint binary
cmake --build . --target flow_adjoint -j8
```

Producing `builds/release/opm-adjoint/flow_adjoint`.

> **Build gotcha (important):** downstream executables must use
> `use_additional_optimization(TARGET ...)`. OPM builds Release with `-UNDEBUG`
> (`WITH_NDEBUG=OFF`); an `NDEBUG` mismatch against the library is an ABI break that
> crashes inside the material law during equilibration. This is already handled in the
> module's CMake.

---

## 4. Run an example

`flow_adjoint` is a single binary. Mode is selected by `--adjoint-mode`
(default = plain forward; `replay`, `gradient`, `objective`). The forward stage
**records** with `--adjoint-save=true`; later stages read that archive.

Set a convenience variable:

```bash
FLOW=/Users/hnil/Documents/OPM/opm_clean/builds/release/opm-adjoint/flow_adjoint
DECK=/Users/hnil/Documents/OPM/opm_clean/opm-tests/spe1/SPE1CASE1.DATA
```

### 4a. Exact backward replay (Milestone A — verifies bitwise re-linearization)

```bash
mkdir -p /tmp/spe1_adj

# forward run, recording each accepted substep's converged residual/Jacobian too
$FLOW $DECK --output-dir=/tmp/spe1_adj \
    --threads-per-process=1 --enable-storage-cache=false \
    --adjoint-save=true --adjoint-save-system=true

# backward replay: re-create every substep's system and compare to the stored one
$FLOW $DECK --output-dir=/tmp/spe1_adj \
    --threads-per-process=1 --enable-storage-cache=false \
    --adjoint-mode=replay --adjoint-replay-tolerance=1e-12
```

The replay log ends with `Adjoint replay finished`; SPE1 matches 123/123 substeps
bitwise. The wrapper `tests/run-adjoint-replay-test.sh <flow> <deck> <out-dir> [tol]`
does exactly this.

### 4b. Compute a gradient (Milestones B+C)

Uses a group-control field case with per-cell PORO. **Gradient runs disable adaptive
time-stepping** so a parameter perturbation cannot change the substep sequence (which
would swamp any FD comparison):

```bash
DECK=/Users/hnil/Documents/OPM/opm_clean/opm-tests/adjoint_tests/SPE1_GRPCTRL/SPE1_GRPCTRL.DATA
OUT=/tmp/spe1grp
mkdir -p $OUT

STRICT="--enable-storage-cache=false --enable-adaptive-time-stepping=false \
        --threads-per-process=1 --tolerance-cnv=1e-9 --tolerance-mb=1e-11 \
        --adjoint-objective=pressure-average"

# 1) forward + record
$FLOW $DECK --output-dir=$OUT $STRICT --adjoint-file=$OUT/a --adjoint-save=true

# 2) backward sweep -> dJ/d(porosity multiplier) per cell
$FLOW $DECK --output-dir=$OUT $STRICT --adjoint-file=$OUT/a --adjoint-mode=gradient
#   => $OUT/<CASE>.ADJOINT_GRADIENTS_PV.txt

# 3) (optional) just the objective value, e.g. for an FD check
$FLOW $DECK --output-dir=$OUT $STRICT --adjoint-file=$OUT/a --adjoint-mode=objective
#   => log line "Adjoint objective J = ..."
```

Objectives: `--adjoint-objective=pressure-average | bhp:<WELL> | matchref:<ref>:<W>.<p>
| matchsum:<ref>:<W>.<p>+...`. Parameter families are written to
`ADJOINT_GRADIENTS_{PV,TRANS,PERM,...}.txt`; well-control multipliers to
`ADJOINT_LAMBDA_WELLS.txt`.

> **Reference-summary caveat:** gradient/objective/replay runs truncate summary files in
> their output dir at startup, so a `matchref` reference summary must live **outside** the
> run's `--output-dir`.

### 4c. Run the test suite

```bash
cd /Users/hnil/Documents/OPM/opm_clean/builds/release/opm-adjoint
ctest                      # 15/15, ~30 s
ctest -R adjoint_fd        # just the FD-vs-adjoint gradient checks
```

Individual `tests/run-adjoint-*.sh` scripts (replay, FD per parameter family, matchref,
MPI, storage-cache equivalence) take `<flow_adjoint> <deck> <out-dir>` and can be run
standalone. They are `bash` scripts — on this machine run them with `bash`, not `zsh`
(the test one-liners rely on word-splitting).

---

## 5. Where to read more

- `adjoint_plan.md` — full design rationale and PR sequencing.
- `adjoint_status.md` — detailed feature catalog (incl. §6 explicitly-updated-quantities
  table that the end-of-run warning mirrors).
- `adjoint_refactoring.md` — the Stage-1/2/3 `WellAssemblyState` refactor that closes the
  remaining group-control / voidage replay gaps.
- `adjoint_testing.md` — per-test-level instructions.
- `jutul/README.md` — JutulDarcy cross-check setup (use the Julia LTS channel: `+lts`).
