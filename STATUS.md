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
decks up to field models, and is guarded by a ctest suite (52 tests).

### Development layout

- **`opm-adjoint`** — separate downstream module (the `opm-flowgeomechanics` pattern),
  built against a **near-stock `opm-simulators`**. Recording hooks live in a derived
  `AdjointFlowProblem` (virtual `beginEpisode`/`endTimeStep`); the backward replay/solver
  driver uses only public simulator APIs. Branch `master`, HEAD `b231773`.
- **`opm-simulators`** — branch `adjoint-hooks` (master + a handful of small accessor
  commits: `B/C/D` getters on `StandardWellEquations`, a `SummaryState` assembly seam,
  half-trans exposure on `Transmissibility`, `assembleWellEqGivenControls`). HEAD
  `9b07194fe`.
- **`opm-tests`** — plain `master` (SPE1/SPE9 decks). The adjoint-specific decks
  (`MODEL_*_DEBUG`, `SPE1_GRPCTRL`, `DEEPWELL`) ship with the module in `doc/tutorials/`.

### What works

| Area | Status |
|---|---|
| **Milestone A — exact backward replay** | SPE1 123/123, SPE9 21/21, MODEL_1D_DEBUG 7/7, SPE1_GRPCTRL 120/120 **bitwise**. Full **Norne** (44 431 cells, 36 wells, group ctrl + shut wells, hysteresis, VAPOIL, faults) replays **end-to-end**, ~90 % bitwise (remainder at well-event / hysteresis boundaries — documented v1 approximation). |
| **Milestone B — adjoint solve** | Backward Lagrange sweep, Schur-reduced, UMFPACK on the transposed system; iterative solves (ilu0 / cpr / **cprt**); **MPI** parallel record/replay + transposed solve (cross-rank transpose bug found & fixed). |
| **Milestone C — parameter gradients (all FD-verified)** | see the derivatives table below. |
| **Tests** | ctest (52 tests); FD-verified golden-reference regression set in `tests/references/`; per-eps FD accuracy study in [`tests/FD_ACCURACY.md`](tests/FD_ACCURACY.md). |

### One-stop mode (`--adjoint-mode=onestop`)

Runs the forward record and the backward sweep in a **single process**, sharing
the recorded states through an **in-memory archive** (no files, no re-reading a
large archive per optimizer iteration). Low-risk: the in-memory backend
(`AdjointMemStore`) reuses the exact `Serializer<MemPacker>` packing of the
directory store — byte-identical buffers, only the medium changes. Verified
byte-identical to the two-process run (ctest `adjoint_onestop_spe1`), zero state
files. **Combine with `--adjoint-slim=true`** for the practical field config:
in-RAM, minimal footprint, byte-identical. shared_ptr archive ctors on
Recorder/Replay/Solver; `AdjointFlowProblem::oneStopArchive()` injects it into
the forward recorder.

**One-stop with a disk archive (2026-08-18).** `--adjoint-mode=onestop
--adjoint-file=<path>` keeps the single process but streams the tape to disk
instead of holding it in RAM: the forward leg opens the archive for writing, the
sweep reopens it read-only. Both backends (`.h5`/`.hdf5` → HDF5, otherwise the
directory store), gradients + objective + field output unchanged. This is the
field configuration — one simulator initialization *and* ~forward memory —
because the in-memory store is undeflated (~230 B/cell/substep resident), which
caps the RAM variant well below field size. Byte-identical to both the in-memory
one-stop and the two-process run on SPE1 (PV/TRANS/PERM/WELLCTRL_BYMODE and J,
both backends), gated by `adjoint_onestop_disk_spe1`. SPE9 (9 000 cells, 21
substeps): max RSS **241 → 189 MiB**, wall clock 2.70 → 2.86 s; the saving is the
tape, so it grows with cells × substeps. Two limits: `--adjoint-checkpoint-every>1`
is refused in one-stop (only the two-run path installs the segment-recompute
driver), and one-stop **aborts under MPI** — a pre-existing `std::bad_alloc` in
the second `FlowMain`, present on the in-memory path too, so parallel runs still
use the two-process path.

### Slim archive (`--adjoint-slim=true`)

The per-substep snapshot serialized the full Eclipse Schedule (via the
vanguard) every substep — static and reconstructed from the deck at replay, so
pure redundancy (124 KB/substep on SPE1; far more on field decks). Slim mode
stores only the **dynamic** state (model solution, problem, summary/UDQ/action);
replay reconstructs the schedule from the deck and sets time/episode from meta
(it already did both). **Byte-identical to the full snapshot** — verified as a
hard gate (`adjoint_slim_grpctrl`, on the group-control/guide-rate deck), all
gradient families bit-for-bit, SPE1 archive 27M→11M (2.5×; larger on field
decks). The full snapshot stays the **default proven path**; slim is opt-in
(JSON `archive.slim`). **Not** for runtime schedule-modifying decks
(ACTIONX/UDQ-driven) — a startup warning fires; use the full archive there.
General-AD is verified exact for PORO/PERM at field scale (SPE9 oracle: PV
6e-16, PERM 1e-15), so `--adjoint-gradient-method=general-ad` is the
recommended fast generic path (analytic remains the default oracle).

**Slim's saving shrinks with cell count**: on
SPE1 (300 cells) it drops 57% of the snapshot (the schedule dominates), but on
SPE9 (9000 cells) only **11%** — the per-cell `problem.bin` (history state,
~152 B/cell) + `model.bin` (~62 B/cell) dominate and slim cannot touch them.
Measured slim floor ≈ **230 B/cell/substep**. Consequence for field scale: an
all-in-RAM tape is not viable (1 M-cell HM ≈ 100–450 GB); the field path is
**disk-streaming** (what the A5 byte-budgeted spill converges to). Slim is a
schedule-heavy-deck win, not the field-scale lever.

**Known gap:** slim + `--adjoint-mode=objective` aborts in `Schedule::getWell`
during `evaluateObjectiveOnly` (gradient / one-stop replay on slim is fine).
Use the full archive for standalone objective-only replays until fixed
(follow-up filed).

### Configuration file (`--adjoint-config=<file>.json`)

A JSON config file (2026-07-23) replaces the grown `--adjoint-*` flag strings and
adds a **structured objective** (per-term wells/phase/weight/time-window,
wildcards) the CLI string cannot express. It is the *base* configuration; any
`--adjoint-*` flag set explicitly on the command line **overrides** its JSON
counterpart. Schema v1 (sections optional): `mode`, `archive` {path, save,
save_system}, `gradient` {method, fd_step, endpoint_method}, `solver`
{linear_solver, reduction, max_iter}, `endpoints`, `objective` (either
`{kind: pressure-average}`, `{kind: bhp, well, window_days}`, or `{terms: [...]}`
with per-term kind/wells/phase/reference_case/target_sm3_per_day/weight/
window_days). Loader `AdjointConfigFile.hpp` (boost::property_tree); the
TypeTag-free `AdjointObjectiveSpec.hpp` is shared by the CLI parser and the
loader. Guarded by ctest `adjoint_config_spe1` (config == equivalent flags;
CLI overrides JSON). The JSON `mode` is now honoured by the run-mode dispatch
(the argv peek in `flow_adjoint.cpp` falls back to the config's `mode` when no
`--adjoint-mode` is on the command line), so a config file is **self-contained**
— ERT/Everest can hand over only the JSON and still select `onestop`.

### Memory & performance (measured)

**Memory — the constraint that decides adjoint model size is the backward linear
solver, not storage.** Measured max RSS:

| | forward | backward umfpack | backward cprt |
|---|---:|---:|---:|
| SPE9 (9 k cells) | 156 MiB | **4103 MiB (26×)** | **356 MiB (2.3×)** |
| **Norne (44 k cells)** | 699 MiB | ~40 GB → **OOM** (extrap.) | **778 MiB (1.1×)** |

umfpack's LU fill scales super-linearly → fatal at field scale; **cprt keeps the adjoint
at ~forward memory**. The backward solver now **defaults to `auto`** (umfpack ≤ 2000
cells as the exact small-case oracle, cprt above) — so field runs get cprt without a
flag. (Gated by `adjoint_solver_spe9`: cprt vs umfpack agree to <1e-7 field-normalized.)

**Coupled-system alternative: `--adjoint-linear-solver=system_cprt` (2026-07-26).** Keeps
the wells *in* the transposed system `[Aᵀ Bᵀ; Cᵀ Dᵀ]` rather than folding them through the
Schur complement, so it needs **no `--matrix-add-well-contributions`** — the flag the
Schur path forces on every multi-perforation deck. Reuses opm's `system_cpr` stack
(`linalg/system/`) with its reservoir stage flipped `cpr`→`cprt`; the well blocks come from
the public `addBCDMatrix`/`WellMatrixMerger` hooks, so there is **no new well code and no
opm-simulators change**. The adjoint's right-hand side is already Schur-shifted
(`g_r − BᵀD⁻ᵀg_w`), so the well block of the rhs is zero and `λ_r` — plus the whole
well-adjoint recovery — is unchanged. Accuracy vs its direct oracle `system_umfpack`:
**8e-12** (SPE9), **7e-12** field-normalized (MSW); on SPE1 it beats the Schur path against
`umfpack` (4e-10 vs 9.6e-7) because it never forms `CᵀD⁻¹B`. Costs a few extra iterations
(SPE9 13.0/solve vs 9.5, SPE1 7.5 vs 6.2). Standard 3-phase blackoil only
(`SystemTypes.hpp` fixes 3 reservoir / 4 well dofs). **Parallel too**: `Aᵀ` from
`buildConsistentParallelTranspose`, `SystemParOp` over `MultiCommunicator(resComm,
JacComm)`, well blocks rank-local (wells are never split), same `copyOwnerToAll` on the CPR
weights. **Iterations are flat in rank count** — SPE1 8.4 at np=2/3/4 (7.6 serial), SPE9
13.0/12.8/12.9 at np=1/2/4 — and parallel-vs-serial gradients agree to 1–2e-7 (SPE1) and
3–8e-6 (SPE9) field-normalized. Gated by `adjoint_system_cprt_{spe1,msw_spe1,spe9}` and
`adjoint_mpi_system_cprt_{spe1,spe9}`.

**Norne parallel replay unblocked (2026-07-26).** Parallel adjoint replay of progressively
drilled decks aborted with `Error deserializing WellState: size mismatch` on *every* solver
(so: not the coupled system). `BlackoilWellModelGeneric::prepareDeserialize` skipped
`wellState().resize(...)` when a rank had no local wells at that report step, leaving the
previous call's state in place; harmless for a forward restart, fatal walking a run
backwards. Fixed by resizing unconditionally — the one opm-simulators change in this work.
Norne np=2 now matches `cprt`+Schur to **1.4e-10** on the same archive.

**Standalone systems.** `--adjoint-dump-system=<dir>` writes one substep's `A/B/C/D` +
rhs + λ as full-precision MatrixMarket (loadable in MATLAB/MRST/Julia/scipy), and
`adjoint_system_solve <dir> [config.json]` rebuilds the transposed system and solves it
without the simulator — it reproduces the simulator's λ exactly, so preconditioner
settings can be tried against a real field system in milliseconds instead of a full sweep.

**Wall-clock — rebuilt 2026-07-25; the old "≈3× forward, gradient-74%" story is
obsolete.** A chain of fixes made the backward sweep ~forward-cost and linearly scaling:

- **Analytic dCTF/dperm well term** (was the dominant cost — 2 whole-domain
  re-linearizations per perforation per substep = 96 % of the SPE9 sweep; now an exact
  O(perforations) analytic contraction, `Σ cq·(λ_w−λ_r)`, guarded by
  `OPM_ADJOINT_PERMWELL_METHOD=verify`). **SPE9 sweep 24 s → 1 s.**
- **cprt preconditioner reuse** across the sweep (build the FlexibleSolver once, then
  `preconditioner().update()`; Norne AMG setup 1.16 s → 0.39 s). Escape hatch
  `OPM_ADJOINT_REBUILD_PRECOND=1`.
- **WS6a**: dropped a dead per-substep snapshot load (3 → 2 loads/substep).

**Norne (44 k cells, 24 substeps): backward sweep 8.9 s vs forward 14.3 s = 0.62×
forward.** Split: transposed solve 52 %, replay (re-linearize) 28 %, gradient families
20 % (objective-independent), perm-well-term ~0. Was ~71 s before these fixes (the
inert-umfpack default). All three surviving components are ~O(cells×substeps), so the
sweep **scales linearly** — one gradient now costs ≈ forward + 0.6×forward sweep +
recording (well under 2× forward). At 0.1–1 M cells the binding constraint is no longer
compute but **tape I/O / disk** (230 B/cell/substep → tens–hundreds of GB; stream to
disk). Timing detail prints to the `.DBG`/`.PRT` (`Adjoint sweep timing`, the gradients
breakdown, and `Adjoint recording timing`).

**Two correctness rules to keep straight** (2026-07-24):

1. **Replay must be *exact* (bitwise).** A non-zero replay residual is not an
   approximation to tolerate — it means the recorded storage is incomplete or the
   re-initialization at replay is wrong (some history state is not stored/reconstructed
   bit-for-bit). It is a bug to fix, not a tuning knob.
2. **The adjoint is not correct without the explicit-update cross-step equations.**
   Explicitly-updated quantities (DRSDT/DRVDT, hysteresis, guide rates, …) make step-k's
   residual depend on step-(k−1) state; the backward `Bdiag` currently omits that
   derivative (task #18). So even with exact replay, gradients on decks where those
   quantities bind are wrong until #18 lands.

**Consequence — the first real gradient-accuracy case must avoid both problems.** Full
Norne replay drifts to ~1.0 residual at hysteresis/well-event boundaries (missing
stored/reinit'd state) *and* binds explicit-update quantities. A cleaned Norne removes
both sources. Two decks now exist (`opm-tests/norne`, uncommitted):

- `NORNE_ATW2013_ADJSHORT.DATA` — the stock SHORT deck with its 1-step `END` removed,
  ~24–35 substeps, full 44 k cells (~a few s forward).
- `NORNE_ATW2013_ADJSHORT_CLEAN.DATA` — same with **tracers and hysteresis commented
  out** (`-- [adjoint-clean]` markers). Measured over 35 timesteps: field pressure
  changes **0.003 %**, well rates **< 0.25 %** (tracers passive; hysteresis barely
  activates) — *behaviourally the same field* — while the replay residual drops **4.5×**
  (7e-3 → 1.5e-3), because the un-stored hysteresis state is no longer in play. The
  remaining 1.5e-3 is well-event/group-control state that is *still* not stored/reinit
  exactly — a real bug to chase toward bitwise replay, not an accepted floor. OLYMPUS is
  a still-simpler fallback. Forward wells run on **RESV with constraints**; the accuracy
  objective is **well rates (± BHP)** — the A6 matchref machinery already covers it.

### Field-scale PORO history-match demo (A6, `adjoint_poro_field_demo_spe9`)

End-to-end exercise of the **practical field configuration** on SPE9 (9000
cells, 26 wells, DISGAS), ctest label `field`:

- a perturbed-PORO **truth** run makes a reference summary;
- the base run computes **dJ/dPORO for all 9000 cells in one process** via
  `--adjoint-mode=onestop` + `--adjoint-config` (matchref, wildcard `PRODU*`
  gas-rate, time window) + general-AD + cartesian-keyed CSV output;
- the dominant-sensitivity cells are **FD-verified**: the top-6 cells by
  |gradient| match central differences to **0.03 %–1.5 %** (delta 2e-4). Random
  cells are *not* checked — near-zero gradients make the FD relative error
  meaningless (central-difference cancellation), so the check targets the cells
  that carry signal.

Solvers (per field guidance): forward `cpr_trueimpes` (~2 s/run vs ~24 s for
umfpack — umfpack refactorizes every Newton iteration, no symbolic reuse);
backward stays **umfpack** (direct, accurate — proven for testing). `cprt`
backward was re-validated here and is **essentially exact**: normalized by the
field-max gradient it agrees with umfpack to **1.9e-9** (the earlier "10 %" was
the *relative* metric on numerically-zero cells — misleading; use the
field-normalized metric). Timing on SPE9: cprt gradient run 25.6 s vs umfpack
32.4 s (~20 % on the whole run — init dominates, so the sweep-only speedup is
larger). So cprt is a sound accurate backward solver already; its bigger payoff
at field scale is **reusing the AMG hierarchy across the N transposed solves**
of the backward sweep. **hypre**: `USE_HYPRE=ON` in the CMake cache but the
opm-simulators `PreconditionerFactory` in this build does **not** register
`hypre` (aborts: available = amg cpr cprt dilu …). Testing hypre needs
opm-simulators rebuilt with hypre registration + a ≥10⁵-cell case to matter.

### Objective functions (`--adjoint-objective=`)

Per-substep sum objectives `J = Σₖ Jₖ(xₖ, xwₖ)`:

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

Same picture with `bhp:PROD` (prod-perf gap 94.3 %). **This gap is now closed**
(dCTF/dperm term below); it was negligible for rate-controlled injectors anyway.

### dCTF/dperm well-coupling term IMPLEMENTED (B1, 2026-07-23)

`accumulatePermWellTerm_` (AdjointSolver.hpp) adds the well-equation
contribution to `dJ/dperm` at defaulted-CTF perforations. The CTF enters the
reservoir source and well residual **linearly** (Tw = CTF·trans_mult,
cq = −Tw·mob·drawdown), so perturbing the well index (new mutable
`WellInterfaceGeneric::wellIndex()` hook) and re-linearizing, the whole-domain
contraction `λ_r·R_r + λ_w·R_w` (reused from the endpoint path) captures BOTH
sides; central-differencing gives `dJ/d(ln CTF)`. Chained to perm via the
dominant Peaceman term `Kh = √(K_a·K_b)·h` ⇒ `d(ln CTF)/dK_a = 1/(2·K_a)` for
the two directions perpendicular to the connection (the weak r0 dependence is
neglected in v1 — residual < 0.2 % at the SPE1 perf). Explicit-CTF connections
have `dCTF/dK = 0` and are skipped (no-op for explicit-CTF decks). Added to both
the analytic and general perm paths. **Result: SPE1 producer-perforation
FD-vs-adjoint 95.6 % → 0.2 %**; the perm directional test now passes UN-masked.
Flag `--adjoint-perm-well-term` (default on) / JSON `gradient.perm_well_term`.
ctest `adjoint_fd_perm_well_spe1`. Cost: 2 whole-domain re-linearizations per
perforation per substep — cheap (perfs ≪ cells); the AD/linearity path (plan
§B1) would remove the re-linearization for very-many-perforation field decks.

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

**ROOT CAUSE (2026-07-23): the `DRSDT 0` explicit cross-step Rs cap.** Removing
DRSDT from SPE1 drops the single-cell bhp dJ/dPORO deviation from 2.94 % to
**2e-6 (exact)**; ctest `adjoint_drsdt_crossterm_spe1` guards this. `DRSDT 0`
caps `Rs_k ≤ Rs_{k-1}` wherever it binds (near the gas injector) — an explicit
dependence of the step-k residual on the step-(k-1) Rs. Replay reproduces the
*within-step* linearization bitwise, but the adjoint's *cross-step* coupling
(reservoir storage Bdiag) does not carry this term, so dJ/dm is approximate.
Per-objective at cell 144: bhp 2.94 %→0, pressure-average 0.2 %→0.2 % (diluted
over 300 cells), gas-rate ~0.4 %. The earlier "inexact well Jacobian / mobility
derivative" hypothesis is **refuted** — with DRSDT removed the well/bhp adjoint
is exact, so D/B/C were always correct; the mobility and well-index derivatives
are fine. bhp merely *exposed* the reservoir cross-step gap most (the producer
BHP is sensitive to the near-injector Rs field; pressure-average dilutes it).

Confirmed genuine, not noise: the 2.94 % is identical under forward
`cnv=1e-11, mb=1e-13` and adjoint reduction 1e-14 **and** 1e-16. (Strict
tolerances are mandatory when FD-testing the adjoint — see the strict-tolerance
memory.) `AdjointDeckWarnings` now flags `DRSDT`/`DRVDT` whenever active
(including value 0); the prior code wrongly excluded DRSDT 0 as "exact".

**Field-HM relevance:** DRSDT (often 0 or small) is common in real decks to
control gas dissolution, so this cross-step term matters for practical HM. Fix
(task): add the DRSDT/DRVDT cross-step contribution to Bdiag — at binding cells
`dR_k/dRs_{k-1}` equals the within-step `dR_k/dRs_k` (since `Rs_k = Rs_{k-1}`
there), detectable from the converged capped-cell set.

**Coverage gap that let it hide.** Every per-cell FD gradient test (pv, perm,
**wellctrl**) ran only on MODEL_1D_DEBUG (3 cells, 7 calm substeps); SPE1 had
replay/Jacobian-eval tests but **no FD gradient test**, so a gradient in a
gas-evolving, DRSDT-capped regime was never FD-checked. m1d (also DISGAS+VAPOIL)
matches bhp×PV at 4e-9 — it has no binding DRSDT cap. Now closed by
`adjoint_drsdt_crossterm_spe1` (well/bhp gradient FD-verified exact on SPE1 with
DRSDT off; DRSDT-0 gap documented).

**Investigation trail (for the record).** Two earlier hypotheses were tested and
refuted before DRSDT: (1) the well explicit-quantities term (connection Δp / F0)
— ruled out (SPE1 datums give zero head; no wellbore volume; head-shift
invariant). (2) an inexact well Jacobian / production mobility derivative
(`getMobility` → `extendEval`) exposed by the adjoint's nonzero `dJ/dxw` RHS — a
plausible mechanism, but refuted by the DRSDT toggle (well/bhp gradient is exact
once DRSDT is removed, so D/B/C were always correct). The lesson worth keeping:
a *reservoir* explicit cross-step term (DRSDT) can present as a *well-objective*
error because the producer BHP is the quantity most sensitive to it.

### Multisegment wells — NOT supported for gradients (2026-07-23)

MSW decks (`SPE1CASE2_MSW`) **replay bitwise** (120/120, reservoir side), but
any gradient with a well objective aborts: `addWellObjectiveRhs_` throws
"Adjoint well objectives support StandardWells only" (`AdjointSolver.hpp:415`),
because every well-adjoint path `dynamic_cast`s to `StandardWell` and MSW is a
`MultisegmentWell`. Well-control and well-objective gradients for MSW are a
separate implementation task (the segment system needs its own D/B/C
elimination). A small MSW gradient test is gated on that support.

### MPI / parallel adjoint — np up to 4 verified against serial

Recording, replay and the transposed solve are all parallel. Each rank
stores and reads back its own partition: the HDF5 archive uses
PROCESS_SPLIT (one dataset per rank, the same mechanism as flow's
save/load-step) and the directory store gets a per-rank sub-directory.

**The transpose is the subtle part.** The forward simulation assembles
complete *rows* for interior cells, but the adjoint system needs
complete *columns* - and a plain local transpose drops the cross-rank
coupling `A_{ki}` where `k` is interior on another rank (the local
matrix's overlap rows do not carry it). This is a real bug, not
theoretical: with a plain local transpose the gradients are correct for
np=2 (a clean split) but **wrong for np>=3** (PV off by ~10%, and with
the cross-rank term disabled entirely the np=3 PV gradient is off by
120%). The fix (`TransposedParallelOperator`) applies the exact
transposed action: it scatters `A^T` over the forward matrix's complete
interior rows, writing partial sums into both interior and overlap
entries, then accumulates the overlap contributions back onto their
owners with `addOwnerCopyToOwnerCopy` - i.e. it communicates the missing
column part. The (incomplete) local transpose is kept only as the
preconditioner matrix (opm's parallel ILU0); the solver is BiCGSTAB.
lambda is made consistent (copyOwnerToAll) after each solve so the
flux/Bdiag cross terms that read a neighbor's lambda are correct.
Per-cell and per-face gradients are gathered onto rank 0 (sorted by
cartesian id); serial output is byte-identical to the pre-MPI behavior
(the gather is the identity in serial).

```bash
tests/run-adjoint-mpi-test.sh <flow_adjoint> <deck.DATA> <outdir> [nprocs] [rel-tol]
```

Compares the np=1 and np=N PV gradient as a sorted value multiset
(ordering-agnostic, since the partitioned grid reorders cells).
**--enable-adaptive-time-stepping=false is required**: otherwise the
parallel forward run can take a different substep sequence than serial
(parallel linear solves converge slightly differently -> different
chopping), which changes the trajectory and makes the gradients
legitimately differ - this masquerades as an adjoint error. Under fixed
stepping, SPE1 PV gradients match serial to **1e-5 (np=2), 6e-6 (np=3),
9e-6 (np=4)**; the transmissibility gradient matches to 4.5e-8
normalized by max|g| (its sorted relative diff looks larger only because
of near-zero faces). ctest `adjoint_mpi_spe1` (np=2, auto-skips when no
MPI launcher is found).

**Linear solver in parallel.** The operator is exact, so the
preconditioner only affects convergence rate; parallel uses opm's
parallel ILU0 (block-Jacobi over ranks) regardless of the requested
`--adjoint-linear-solver` (which then only sets tol/maxiter). UMFPACK is
single-rank and falls back to ilu0 in parallel. In serial, cprt remains
the fastest correct choice. Run a parallel adjoint manually with e.g.:
```bash
mpirun -np 4 flow_adjoint CASE.DATA --enable-storage-cache=false \
    --enable-adaptive-time-stepping=false --adjoint-save=true
mpirun -np 4 flow_adjoint CASE.DATA --enable-storage-cache=false \
    --enable-adaptive-time-stepping=false --adjoint-mode=gradient
```

### Known gaps / queue (low priority, none blocking)

- Report-step-0 startup bitwise for model2-like decks; Norne hysteresis / well-event
  boundary bitwise.
- **WGRUPCON / VREP** voidage-replacement replay-inexactness — fix _designed_ (Stage-2:
  carry the converged effective control target per well in the `WellAssemblyState`
  bundle so `assembleControlEqInj/Prod` read it directly), **not yet implemented**.
- `FlexibleCellEvaluator` (C.2) for exact parameter-AD; end-point scaling is currently
  via cell-local parameter FD.

The end-of-run `AdjointDeckWarnings` scan reports, per deck, every quantity the adjoint
does not handle exactly (approx-gradient: hysteresis, DRSDT/DRVDT, VAPPARS, guide-rate
group allocation, well explicit quantities; approx-replay: voidage group control;
neglected: aquifers, networks). SPE1 prints nothing (clean).

### Explicitly-updated quantities — what the end-of-run warning means

A number of quantities are updated *explicitly* in the forward run
(between iterations or steps) rather than being solved implicitly. The
adjoint handles each of three ways. The end-of-run scan
(`AdjointDeckWarnings.hpp`, printed after every gradient/replay run)
detects the ones present in the deck and warns; the tags below match the
warning tags.

| Quantity | Where updated | Replay (J) | Gradient (dJ/dm) | Warning tag |
|---|---|---|---|---|
| Saturation hysteresis (HYSTER) | `endTimeStep` (max/min sat in materialLawManager) | **exact** (state serialized) | cross-term neglected | `approx-gradient` |
| DRSDT / DRVDT (active, max>0) | `endTimeStep` (mixControls_ max Rs/Rv rate) | **exact** (cache-free storage) | cross-term neglected | `approx-gradient` |
| VAPPARS | `endTimeStep` | **exact** | cross-term neglected | `approx-gradient` |
| DRSDT/DRVDT **0** (frozen Rs/Rv) | — | **exact** | **exact** (no rate-dependent term) | none (not flagged) |
| Rock compaction (poro/trans mult from pressure history) | `endTimeStep` | exact (serialized) | cross-term neglected | *(not yet auto-detected)* |
| Max-oil / max-water saturation tracking (drives VAPPARS) | `updateMaxOilSaturation_` in `endTimeStep` | exact | neglected | via `VAPPARS` (`approx-gradient`) |
| Well explicit quantities (F0_ wellbore storage, connection hydrostatic pressure drop from prior-state perf densities, explicit B) | `prepareTimeStep` / `computePropertiesForPressures` (`StandardWellConnections.cpp:230`) | exact (recomputed from state k−1) | known cross-term neglected | `approx-gradient` (standing note, printed whenever any other item fires) |
| Guide-rate group control allocation (target shared among members by guide rates from well potentials) | `BlackoilWellModelGeneric.cpp:1595` `computePotentials` → `updateGuideRates` | **exact** (converged guide rates restored, verified bitwise on gconprod/gconinje/grupcntl/model4) | d(guide-rate)/d(state) cross-term neglected — **negligible/zero for single-member or guide-rate-insensitive groups** (why `adjoint_fd_grpctrl` on SPE1_GRPCTRL still passes); can be material when ≥2 members share a target | `approx-gradient` |
| **Reservoir-volume / voidage-replacement group control (RESV/VREP/REIN)** | `updateWellControlsAndNetwork` (group voidage target) | **not bitwise** — the effective control target is re-derived from reservoir voidage during the replay assembly | approximate | `approx-replay` |
| Single-rate group control (ORAT/GRAT/WRAT/LRAT, GRUP) | `updateWellControlsAndNetwork` | **exact** (verified bitwise on gconprod/gconinje/grupcntl/model4) | exact apart from the guide-rate cross-term above | guide-rate row |
| Well control-mode switching | during Newton | exact (converged controls in WGState) | kinks at switches are real objective non-smoothness, not a bug | none |
| Aquifers | aquifer `endTimeStep` | **inexact** (aquifer contributions not assembled in replay) | neglected | `neglected` |
| Extended network (NODEPROP/BRANPROP) | `updateWellControlsAndNetwork` | inexact | neglected | `neglected` |

The `approx-gradient` group is the principled v1 trade-off: the forward
*state* is serialized so the residual/Jacobian (hence J) is reproduced
bitwise, but the cross-step derivative of the explicitly-updated quantity
is dropped from dJ/dm. The `approx-replay` and `neglected` groups are
where the replay itself is not bitwise — the voidage case is the
`WellAssemblyState` Stage-2 fix listed under *Known gaps* above.

---

## 2. Checkout

The OPM module repos sit side by side in one directory (called `$OPM` below, e.g.
`~/OPM`), each with its own `build/` folder. The adjoint work lives on two
non-default branches:

```bash
# opm-simulators: small accessor hooks on top of master
git -C opm-simulators checkout adjoint-hooks

# opm-adjoint: the module itself (default branch)
git -C opm-adjoint checkout master
```

`opm-common`, `opm-grid` and `opm-tests` stay on their normal branches. The
adjoint-specific test decks are in `opm-adjoint/doc/tutorials/` (see its README); the
SPE1/SPE9 tests use the standard opm-tests decks and are skipped if opm-tests is
missing. `opm-adjoint/dune.module`
declares the dependency chain: `dune-istl opm-common opm-grid opm-simulators`.

---

## 3. Compile

Build opm-common, opm-grid and opm-simulators first, each in its own `build/` folder,
then opm-adjoint the same way. Recipe used on macOS (Apple Silicon, DUNE and the other
dependencies from Homebrew):

```bash
OPM=~/OPM                          # directory holding the sibling repos
mkdir -p $OPM/opm-adjoint/build && cd $OPM/opm-adjoint/build

cmake -DUSE_GPU_BRIDGE=0 \
      -DCMAKE_OSX_DEPLOYMENT_TARGET=26.0 \
      -DWITH_NDEBUG=1 \
      -DCMAKE_BUILD_TYPE=Release \
      -Dopm-common_DIR=$OPM/opm-common/build \
      -Dopm-grid_DIR=$OPM/opm-grid/build \
      -Dopm-simulators_DIR=$OPM/opm-simulators/build \
      ..

make -j12                          # or: make -j12 flow_adjoint
```

This produces `build/flow_adjoint` and the unit-test binaries in `build/bin/`. Notes:

- Use the **same** `-DWITH_NDEBUG`, build type and deployment target as for the other
  modules (set the deployment target to your macOS version).
- The explicit `-D<module>_DIR` flags are needed: the module does not find opm-grid and
  opm-simulators by sibling search on its own.
- Configure in a fresh `build/` — a `CMakeCache.txt` from another machine breaks it.
- HDF5 is optional. Without it the archive is a plain directory instead of one `.h5`
  file, and the HDF5-specific test fails.

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
OPM=~/OPM                          # directory holding the sibling repos
FLOW=$OPM/opm-adjoint/build/flow_adjoint
DECK=$OPM/opm-tests/spe1/SPE1CASE1.DATA
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
DECK=$OPM/opm-adjoint/doc/tutorials/SPE1_GRPCTRL/SPE1_GRPCTRL.DATA
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
cd $OPM/opm-adjoint/build
ctest                      # 52 tests
ctest -R adjoint_fd        # just the FD-vs-adjoint gradient checks
```

Individual `tests/run-adjoint-*.sh` scripts (replay, FD per parameter family, matchref,
MPI, storage-cache equivalence) take `<flow_adjoint> <deck> <out-dir>` and can be run
standalone. They are `bash` scripts — on this machine run them with `bash`, not `zsh`
(the test one-liners rely on word-splitting).

### 4d. Test levels step by step

Run these in order when something breaks; a failure at a lower level explains failures
above it. `$FLOW` and `$OPM` as in §4.

**T1 — storage/IO unit tests**

```bash
$OPM/opm-adjoint/build/bin/test_AdjointStorage      # archive roundtrip, dump comparison
$OPM/opm-adjoint/build/bin/test_transposeMatrix     # transpose vs umtv, UMFPACK transposed solve
```
Expected: `*** No errors detected` from both.

**T0.5 — determinism (run before blaming the replay)**

```bash
for i in 1 2; do
  $FLOW $OPM/opm-tests/spe1/SPE1CASE1.DATA --output-dir=/tmp/det$i \
      --threads-per-process=1 --enable-storage-cache=false
done
cmp /tmp/det1/SPE1CASE1.UNSMRY /tmp/det2/SPE1CASE1.UNSMRY && echo DETERMINISTIC
```

**T0.7 — storage cache vs no-cache equivalence**

```bash
cd $OPM/opm-adjoint
bash tests/run-storage-cache-equivalence.sh $FLOW \
    $OPM/opm-tests/spe1/SPE1CASE1.DATA /tmp/t07
```
Expected: `UNSMRY: bitwise identical`, `UNRST: bitwise identical`, PASS. The same check
is worth running on a DRSDT deck (`spe02/SPE02-RADIAL-02.DATA`) and a ROCKCOMP deck
(`model6/0B_ROCKTAB_MODEL6.DATA`); any deviation beyond rounding is a storage-term bug
independent of the adjoint. Pass opm-common's `compareECL` as 4th argument for a
tolerance-based comparison instead of bitwise.

**T2/T3 — backward replay (Milestone A acceptance)**

```bash
cd $OPM/opm-adjoint
bash tests/run-adjoint-replay-test.sh $FLOW \
    $OPM/opm-tests/spe1/SPE1CASE1.DATA /tmp/replay_spe1
bash tests/run-adjoint-replay-test.sh $FLOW \
    doc/tutorials/MODEL_1D_DEBUG/MODEL_1D_DEBUG.DATA /tmp/replay_m1d
```
The manual two-step version is §4a. The forward run writes the archive:
```
# -> /tmp/adj/SPE1CASE1.ADJOINT/  (or .h5 with HDF5)
#    adjoint/initial/simulator.bin            initial snapshot
#    adjoint/substep/<k>/simulator.bin        full state at end of substep k
#    adjoint/substep/<k>/wgstate.bin          well/group state alone
#    adjoint/substep/<k>/system/{residual,jacobian}.bin
#    adjoint/meta.bin, adjoint/parameters
```
(one `.h5` file instead of the directory when built with HDF5).

Per-substep log lines (PRT/DBG and stdout):
```
replay substep  122 (report step 119, dt 31 days): residual rel diff 0.000e+00 (abs 0.000e+00), Jacobian rel diff 0.000e+00 (abs 0.000e+00)
...
Adjoint replay finished: 123/123 substeps within tolerance 1.0e-12 (...)
```
Exit code 0 = every substep within tolerance. Tolerances:
`--adjoint-replay-tolerance` (relative, default 1e-12) and
`--adjoint-replay-abs-tolerance` (absolute noise floor, default 1e-9 —
entry differences below it are rounding noise and excluded from the
relative criterion).

What the replay does per substep k (see `AdjointReplay::replayStep`):
restore snapshot k−1 → mirror the report-step context
(`startNextEpisode`/`beginEpisode`) → mirror `prepareStep`
(`advanceTimeLevel`, `setTime`, `setTimeStepSize`, `beginTimeStep`) →
iteration-0 assembly (well `prepareTimeStep` explicit quantities from state
k−1, exactly like forward) → install x_k + WellState_k → final linearization
→ compare with stored system. Afterwards the linearizer holds the converged
system of substep k — the state Milestone B's adjoint solve starts from.

**Interpreting failures**

- `WellState size mismatch` on reading a snapshot: archive recorded with an
  older build — re-run the forward simulation.
- Residual rel diff large but abs diff ~1e-12: rounding noise from forward
  retry (chopped step) paths; covered by the abs-tolerance floor.
- Large diffs on decks with group control, aquifers or networks: check the
  end-of-run warning against the *Explicitly-updated quantities* table in §1
  before suspecting a bug.

**Forward-run validity warnings.** With `--adjoint-save=true` the recorder warns when
the storage cache is enabled, more than 1 thread is used, or system dumps are disabled.

---

## 5. Where to read more

- `adjoint_code_guide.md` — code-reading guide: source reading order, backward-sweep
  data flow, gradient mechanisms, opm-simulators hook list.
- `adjoint_plan.md` — full design rationale, literature and PR sequencing.
- `tests/FD_ACCURACY.md` — adjoint-vs-FD accuracy as a function of the FD step.
- `doc/tutorials/README.md` — the adjoint test decks shipped with the module.
