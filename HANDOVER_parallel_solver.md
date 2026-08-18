# Handover — parallel transposed (adjoint) linear solve

**Date:** 2026-07-25. **Purpose:** start a fresh session on the one thing blocking
parallel field-scale adjoint gradients: **there is no scalable parallel preconditioner
for the transposed adjoint system.** The serial path is fast and exact (`cprt`); the
parallel path silently falls back to block-ILU0, which is slow, rank-degrading, and only
~1e-4 accurate at Norne scale. This thread builds the parallel analogue of serial `cprt`.

Orthogonal to the correctness thread (`HANDOVER.md`, #22/#24) and the now-largely-done
performance thread (`HANDOVER_optimization.md`, auto-memory `adjoint-performance-optimization`).

## Read first
1. Auto-memory `adjoint-performance-optimization` — what the serial optimization did and
   the current sweep profile (serial Norne sweep 8.9 s = 0.62× forward, linearly scaling).
2. `STATUS.md` "Memory & performance" — updated 2026-07-25.
3. This machine's MPI build: `~/OPM_CLEAN/opm_docs/build/toolchain.md`
   (Open MPI 5.0.9 brew; the release tree is MPI-capable — `HAVE_MPI` is a compile flag,
   not a `config.h` entry, and `flow_adjoint` links `libmpi`).

## The problem (measured 2026-07-25, Norne 44 431 cells, 24 substeps, slim tape)

Serial `cprt` is the reference: **16 iterations/solve, 4.2 s, gradients exact.** In
parallel the transposed solve is the ONLY thing that breaks — replay, gradient
accumulation, and tape I/O are per-rank local and scale fine.

| run | iters/solve | gradient vs serial cprt |
|---|---:|---|
| serial `cprt` (transposed CPR + AMG) | **16** | reference |
| serial `cpr` (non-transposed CPR on Aᵀ) | 166 | — (10× slower, wrong preconditioner) |
| np=2 (block-ILU0), reduction 1e-14 | **204** | **2.15e-04** |
| np=2, reduction 1e-8 | 110 | 7.8e-04 |
| np=4, reduction 1e-14, default maxiter 300 | >300 → **diverges/aborts** | fails |

Three distinct defects, in priority order:

1. **Parallel discards CPR/AMG.** `AdjointLinearSolver::setComm`
   (`AdjointLinearSolver.hpp:295`) logs *"the requested 'X' affects only tolerance/iteration
   limits in parallel"* — `solveParallel_` (`:448`) always builds
   `ParallelOverlappingILU0` (`:466`) + BiCGSTAB, regardless of cpr/cprt/ilu0. Block-ILU0
   is ~13× weaker than cprt (204 vs 16 iters/solve) and needs maxiter well above the
   default 300, which is why np≥2 aborts out of the box.
2. **It anti-scales.** More ranks → more subdomains → weaker block-ILU0: np=4 fails where
   np=2 barely converges. The one component that should speed up with ranks gets worse.
3. **Accuracy floors at ~2e-4** even at reduction 1e-14 — NOT machine precision. This is a
   correctness red flag beyond tolerance (see "Open question A" below). `adjoint_mpi_spe1`
   misses it because SPE1 is tiny (matches ~5e-6 there); at Norne scale it is marginal.

## The goal

A **parallel transposed CPR**: pressure/coarse solve by a parallel AMG (opm's parallel
`amg`, or **hypre BoomerAMG** — now built, see below) + a parallel smoother, applied to
the **exact transposed operator** already implemented. This is the parallel analogue of the
serial `cprt` path, which is the proven template.

## Where the code is (all in `opm/adjoint/AdjointLinearSolver.hpp`)

- **`class TransposedParallelOperator` (`:113`)** — applies the EXACT transposed action Aᵀ
  by scattering over the forward matrix's complete interior rows and accumulating overlap
  contributions (`scatterTransposed_`, `:156`; `applyscaleadd`). This is the operator the
  new preconditioner must wrap. **Reuse it as-is** (but see Open question A — verify it is
  truly exact).
- **`configure()` (`:216`)** — builds the FlexibleSolver property tree from the spec; sets
  `usesCprWeights_`/`transposedWeights_` (`:259-261`). For `cprt`:
  `setupCPR("cpr_quasiimpes")` + `preconditioner.type=cprt` +
  `PressureTransferPolicy<transpose=true>` + quasi-IMPES weights with `transpose=true`
  (file header, `:34`). **This is the recipe the parallel version must reproduce on the
  parallel operator.**
- **`solveSerial_` (`:396`)** — the working serial template: FlexibleSolver on
  `cachedTranspose_` with the transposed quasi-impes weights (`:414`). Note WS3 now caches
  the solver and calls `preconditioner().update()` across substeps — do the same in parallel.
- **`solveParallel_` (`:448`)** — the code to replace/extend. Currently
  `ParallelOverlappingILU0` + `Dune::BiCGSTABSolver`. Needs a parallel CPR preconditioner.
- **`setComm` (`:295`)** — where the parallel branch decides the preconditioner; currently
  forces the block-ILU0 message. Update once a real parallel CPR exists.

## Approach (suggested)

1. **First, resolve Open question A (accuracy floor)** — it gates everything. Until a
   parallel solve can reach serial gradients to ~1e-9, a better preconditioner just
   converges *faster to the wrong answer*.
2. **Parallel CPR two ways to try:**
   - **(a) opm's parallel FlexibleSolver CPR** on the `TransposedParallelOperator`. The
     blocker is that FlexibleSolver's `PreconditionerFactory` is pre-instantiated only for
     opm's own operator types (noted in `solveParallel_`'s comment), so a custom operator
     may not slot in directly — check whether the transposed operator can be expressed as
     (or adapted to) an opm `WellModelGhostLastMatrixAdapter`-like type the factory knows.
   - **(b) hand-assembled parallel CPR**: parallel pressure restriction + a parallel AMG
     coarse solve (opm `amg` or hypre) + a parallel smoother (the existing block-ILU0 as
     the fine smoother is fine), mirroring `solveSerial_`'s cprt structure but with the
     parallel operator/comm. More code, but avoids the factory-instantiation wall.
3. **hypre is the natural parallel coarse solver** — it is MPI-native and now built &
   registered (opm-adjoint CMakeLists PUBLIC-links `HYPRE::HYPRE`; `HyprePreconditioner`
   symbol is in the binary). But hypre BoomerAMG on the TRANSPOSED system needs tuning
   (open WS4 item in `adjoint_storage_sizing.md`: it stalls on the serial transposed Norne;
   the dune `amg` coarse converges on the same system, so it is settings, not the
   transpose). `tests/adjoint-cprt-hypre.json` is the config (tight tol 1e-14 — do NOT
   loosen: the adjoint is a single linear solve with no outer Newton loop to absorb error).
4. **Keep the tight reduction.** The default `AdjointLinearSolverReduction=1e-14` is
   required for gradient accuracy (λ error multiplies dR/dm directly). A weak parallel
   preconditioner that can't reach it is not usable — that is the whole point.

## Solver guidance (from the serial work, carries over)

- Serial fastest/correct: `cprt` (auto-selected for >2000 cells). `cpr` on Aᵀ is 10× worse.
  umfpack single-rank only.
- `cprt` "false-converges in parallel" (CLAUDE.md) — understand WHY as part of this work;
  it likely means the transposed CPR weights/pressure policy are computed inconsistently
  across ranks (a partition/overlap issue), which may be the same root as the 2e-4 floor.
- WS3 preconditioner reuse (`preconditioner().update()`) and its escape hatch
  `OPM_ADJOINT_REBUILD_PRECOND=1` should extend to the parallel path.

## Open questions to answer

- **A (blocker): is the 2e-4 floor the operator or the preconditioner?** Test: solve Aᵀλ=b
  in parallel with a DIRECT check — compare `TransposedParallelOperator.apply(λ_serial)`
  against `b` (residual of the serial solution under the parallel operator). If that
  residual is ~1e-4, the operator is inexact (fix `scatterTransposed_`/overlap
  accumulation). If it is ~1e-14, the operator is fine and the floor is the
  preconditioner/BiCGSTAB stopping on the preconditioned residual (switch to a true-residual
  stopping test, or GMRES). Do this on np=2 Norne first.
- **B: does the exact transposed operator need the full matrix or just interior rows?** The
  operator comment says it uses "complete interior rows"; confirm the overlap accumulation
  (`addOwnerCopyToOwnerCopy`) is exact for a column-complete Aᵀ.
- **C: partition sensitivity** — Zoltan partitioning changes cell ownership; the gradient
  must be partition-independent (the mpi test compares as a sorted multiset). Confirm the
  new preconditioner keeps that.

## Build / run / measure

```bash
cd /Users/hnil/OPM_CLEAN/opm_adjoint/builds/release && cmake --build . --target flow_adjoint -j8
FLOW=$PWD/opm-adjoint/flow_adjoint
DECK=$PWD/../../opm-tests/norne/NORNE_ATW2013_ADJSHORT_CLEAN.DATA
STRICT="--threads-per-process=1 --enable-storage-cache=false --enable-adaptive-time-stepping=false --adjoint-objective=pressure-average"
# forward record (directory store => per-rank subdirs), then backward, on N ranks:
mpirun -np 2 --oversubscribe $FLOW $DECK $STRICT --adjoint-save=true --adjoint-slim=true --adjoint-file=OUT/tape
mpirun -np 2 --oversubscribe $FLOW $DECK $STRICT --adjoint-mode=gradient --adjoint-file=OUT/tape --adjoint-linear-solver=ilu0 --adjoint-linear-solver-max-iter=1000
```
- Reusable measurement scripts (this session): `scratchpad/par.sh` (np=1/2/4 sweep +
  cross-np PV multiset compare) and `scratchpad/par2.sh` (np=2 ilu0 at several reductions
  vs serial cprt). Recreate from this handover if the scratchpad is gone.
- Solver diagnostics print to `.PRT`: `Adjoint linear solver '…': N iterations …`. NOTE
  the WS3 `setup/apply` split is only instrumented in `solveSerial_`; extend to
  `solveParallel_` when you touch it (shows 0.000s in parallel today).
- Gradients: directory store writes one per-rank subdir; concat + `sort -g` the
  `*.ADJOINT_GRADIENTS_PV.txt` for a partition-independent multiset compare vs serial.

## Validation / tests

- `adjoint_mpi_spe1` (`tests/run-adjoint-mpi-test.sh`) is the existing gate — but SPE1 is
  too small to catch the Norne-scale accuracy floor (it passes at ~5e-6). **Add a larger
  parallel gate** (SPE9 or Norne) comparing parallel gradients to serial cprt at a tight
  field-normalized tolerance once the solve is fixed. Mirror the field-normalized metric
  added to `tests/run-adjoint-solver-test.sh` (env `METRIC=field`).
- Keep the serial oracle paths untouched: umfpack (small) and cprt (validated) remain the
  references the parallel solve is checked against.

## Coordination / uniformity notes

- **Same working tree as the correctness thread** (#22/#24). Shared files touched by this
  session's perf work: `AdjointSolver.hpp` (auto-select ctor + analytic perm-well term +
  timing), `AdjointReplay.hpp` (WS6a dead-load drop + loadSnapshot timing),
  `AdjointLinearSolver.hpp` (WS3 reuse + timing), `AdjointParameters.hpp` (`linearSolver`
  default `"auto"`), `CMakeLists.txt`, `tests/run-adjoint-solver-test.sh`,
  `tests/adjoint-cprt-hypre.json`. All coexist cleanly (build + 41/41 fast tests green as
  of 2026-07-25).
- **Conventions to keep uniform:** debug/safeguard toggles are env vars with the
  `OPM_ADJOINT_*` prefix (`OPM_ADJOINT_REBUILD_PRECOND`, `OPM_ADJOINT_PERMWELL_METHOD`);
  production config is `--adjoint-*` params registered in `AdjointParameters`. Timing
  accumulators are `*Seconds_` members surfaced through `AdjointSolver::reportTiming_`.
- The parallel accuracy floor (Open question A) may share a root cause with the
  correctness thread's replay work — a parallel-partition inconsistency shows up in both.
  Coordinate if #22 touches parallel replay/overlap.
