# opm-adjoint — session guide

Adjoint-based gradients for the OPM flow simulator: `dJ/dθ` of well-matching (and
other) objectives w.r.t. **porosity/PV, transmissibility, permeability**, end-point
scaling, and well controls. Separate downstream module (the `opm-flowgeomechanics`
pattern), built against a near-stock `opm-simulators`.

## Read first (in this order)

1. `adjoint_paths_summary.md` — decision basis for the current **practical field-HM
   phase**; §4 is the plan shape. Newest strategic doc.
2. `STATUS.md` — status + build/run quickstart + objective/derivative tables.
3. `adjoint_code_guide.md` — **code-reading guide + current-design summary**: reading
   order through the source, the backward-sweep data flow, the three gradient
   mechanisms, and the opm-simulators hook list. Start here to understand *how the code
   is structured*.
4. `adjoint_code_assessment.md` — fresh code review (findings F1–F7, perf P1–P3).
5. Auto-memory `adjoint-project-state` — full decision/commit history.

6. `HANDOVER_storage.md` — **the current field-scale blocker.** The backward solve is
   cheap now (~1.2× forward memory, ~1.5× forward time); the archive is not: full Norne
   needs ~7.3 GB of snapshots per forward run. Measured attribution (`problem` = 89.5 % at
   field scale), ranked work items and verification plan live there.

Background (read on demand): `adjoint_plan.md` (design + PR sequencing),
`adjoint_status.md` (feature catalog + explicitly-updated-quantities table),
`adjoint_refactoring.md` (WellAssemblyState Stage-1/2/3), `adjoint_testing.md`,
`adjoint_general_ad_review.md`, `jutul/README.md`.

## Current phase (2026-06-27 →): practical field-scale adjoint

Scope restricted to **PORO (pvmult) + PERM** (+ trans as raw form) — the finished,
exact-AD, one-sweep, MPI-parallel families. Plan order (see summary §4):
1. **Quantify `dCTF/dK`** on a defaulted-CTF deck (SPE1/SPE9), FD on a perforated cell;
   add the well-row perm term (`λ_w·dR_w/dperm`) if material. **Do this first.**
2. Flip defaults to `general-ad` + `stencil` after a Norne shakeout; keep analytic oracle.
3. I/O layer: JSON `--adjoint-config` + machine-readable cartesian-keyed gradient output.
4. One-stop mode (fwd+bwd in one process, in-memory) + slim archive.
5. Field-deck hardening (report-0, well events, WGRUPCON Stage-2).
6. JutulDarcy side-by-side validation on a shared field case.

## Branches

```
opm-adjoint     parallel-cprt  # the module
opm-simulators  adjoint-hooks  # upstream/master + ONE commit (half transmissibilities)
opm-tests       adjoint        # adjoint_tests/MODEL_*_DEBUG, SPE1_GRPCTRL, NORNE_*_ADJSHORT*
opm-common      adjoint        # == upstream/master, no local patches
opm-grid        adjoint        # == upstream/master, no local patches
```

**Synced to upstream 2026-08-10.** Nearly the whole hook set was merged upstream, so
the downstream patch set went from 13 commits to **1**: `flow: optionally store one-sided
half transmissibilities`. Everything else the adjoint needs is now stock API.

Upstream renamed several hooks during review — when re-syncing, expect this class of
change and grep for the old names:

| was (ours) | now (upstream) |
|---|---|
| `Bmatrix()` / `Cmatrix()` / `Dmatrix()` | `getB()` / `getC()` / `getD()` (both StandardWell and MSW) |
| `assembleWellEqGivenControls` | `assembleWellEqAtRestoredState` |
| `--convert-to-multisegment-well=true` | `=per-connection` (string, validated) |
| `makeSimpleMultiSegmentWell` (in-place) | `makeMultiSegmentWellPerConnection` (returns the well) |

Dropped as dead code: the `setAssemblySummaryState` seam — defined, never called, no
consumer anywhere. Do not resurrect it without a caller.

Rebase gotcha seen twice: `git rebase` **silently skipped** commits as "previously
applied" on a false patch-id match when upstream had a *renamed* equivalent, once
deleting files that were not actually upstream. Always check `git rev-list --count` and
that the symbols still exist afterwards.

## Build

```bash
cd /Users/hnil/OPM_CLEAN/opm_adjoint/builds/release
cmake --build . --target flow_adjoint -j8
```
Release tree: Ninja, `CMAKE_BUILD_TYPE=Release`, parallel HDF5 + UMFPACK. Produces
`builds/release/opm-adjoint/flow_adjoint` (single binary; mode via `--adjoint-mode`).

> **Gotcha:** downstream executables need `use_additional_optimization(TARGET ...)` —
> OPM builds Release with `-UNDEBUG`; an NDEBUG mismatch is an ABI break that crashes in
> the material law during equilibration. Already handled in the module CMake.

## Run (see STATUS.md §4 for full recipes)

```bash
FLOW=/Users/hnil/OPM_CLEAN/opm_adjoint/builds/release/opm-adjoint/flow_adjoint
# forward + record, then backward sweep:
$FLOW $DECK --output-dir=$OUT --adjoint-file=$OUT/a --adjoint-save=true   $STRICT
$FLOW $DECK --output-dir=$OUT --adjoint-file=$OUT/a --adjoint-mode=gradient $STRICT
# STRICT = --enable-storage-cache=false --enable-adaptive-time-stepping=false \
#          --threads-per-process=1 --tolerance-cnv=1e-9 --tolerance-mb=1e-11
```
Gradient methods: `--adjoint-gradient-method=analytic|general|general-ad`.
Objectives: `--adjoint-objective=pressure-average|bhp:W|rate:W:p|match…|matchref…|matchsum…`.

## Test

```bash
cd /Users/hnil/OPM_CLEAN/opm_adjoint/builds/release/opm-adjoint
ctest              # 28/28, ~30 s
ctest -R adjoint_fd   # just the FD-vs-adjoint gradient checks
```

## Gotchas (hard-won)

- **Gradient/FD runs require `--enable-adaptive-time-stepping=false`** — else a parameter
  perturbation changes the substep sequence and swamps any FD comparison.
- **`matchref` reference summaries must live OUTSIDE `--output-dir`** — gradient/objective/
  replay runs truncate summary files in their output dir at startup.
- **Multi-perf wells** need `--matrix-add-well-contributions=true` (now injected by default
  in `flow_adjoint main()`, with `--linear-solver=ilu0`) — *unless* you use
  `--adjoint-linear-solver=system_cprt`, see below.
- **Coupled-system backward solve (`--adjoint-linear-solver=system_cprt`)** keeps the wells
  IN the transposed system `[Aᵀ Bᵀ; Cᵀ Dᵀ]` instead of folding them through the Schur
  complement, so it runs **without `--matrix-add-well-contributions`** (flow_adjoint defaults
  the flag to false for `system_*` solvers). Built on opm's `system_cpr` stack
  (`opm-simulators/opm/simulators/linalg/system/`): the well blocks come from the public
  `BlackoilWellModel::addBCDMatrix` + `WellMatrixMerger`, all four blocks are transposed with
  the existing `transposeBlockMatrixTo` (already rectangular-safe), and the *same*
  `SystemPreconditioner` is reused with its reservoir stage flipped `cpr`→`cprt`. **No new
  well code, no new template instantiations, no opm-simulators changes.** It needs no `cprwt`
  because `setupSystemCPR` sets `add_wells=false` in that stage, so it uses the ordinary
  transpose-capable `PressureTransferPolicy`. **Serial and parallel**; standard 3-phase
  blackoil only (`SystemTypes.hpp` hardcodes 3 reservoir / 4 well dofs). In parallel `Aᵀ`
  comes from `buildConsistentParallelTranspose` (ghost rows → identity) and the operator is
  `SystemParOp` over `MultiCommunicator(resComm, JacComm)`; the well blocks need no
  communication because a well is never split across ranks
  (`AllowDistributedWells=false`), and the CPR weights get the same `copyOwnerToAll` as the
  reservoir-only path. Iterations are flat in rank count (SPE1 8.4 at np=2/3/4 vs 7.6
  serial; SPE9 13.0/12.8/12.9 at np=1/2/4). **Why the well rhs is zero:** the
  adjoint's rhs is already the Schur-shifted `g_r − BᵀD⁻ᵀg_w`, and
  `[Aᵀ Bᵀ; Cᵀ Dᵀ][λ_r; ν_w] = [that; 0]` has the same `λ_r` — so `computeWellAdjoints_` and
  the whole rhs assembly are untouched. Accuracy: matches its direct oracle
  (`system_umfpack`) to ~8e-12 on SPE9; on SPE1 it beats the Schur path against `umfpack`
  (4e-10 vs 9.6e-7) because it never forms `CᵀD⁻¹B`. Costs a few more iterations
  (SPE9 13.0 vs 9.5, SPE1 7.5 vs 6.2).
- **Norne parallel replay: fixed (2026-07-26), was an upstream bug.** Parallel adjoint
  replay of a deck whose wells are drilled progressively aborted with
  `Error deserializing WellState: size mismatch` (`WellState.hpp:347`) — on *both* the
  `cprt`+Schur and `system_cprt` paths, so it was never solver-related.
  `BlackoilWellModelGeneric::prepareDeserialize` only called `wellState().resize(...)`
  `if (!wells_ecl_.empty())`; on a rank with no local wells at that report step the
  PREVIOUS call's well state survived, so it no longer described `report_step`. A forward
  restart never notices (one step into a fresh model), but walking a run *backwards* does:
  a rank goes from "has wells" to "has none" moving to earlier steps, and the stale
  non-empty state meets a stored empty one. SPE1/SPE9 have all wells from step 0 and never
  hit it. Fix: resize unconditionally (`resize` handles an empty well list). Verified on
  Norne np=2 — `system_cprt` and `cprt`+Schur agree to **1.4e-10** on the same archive;
  serial-vs-parallel is 8.3e-4, which is the forward trajectory (J itself differs 1.2e-6),
  not the adjoint. Debug aid: `OPM_ADJOINT_WELLSTATE_DIAG=1` prints recorded vs
  reconstructed well-state sizes per rank/substep.
- **Dump a system and solve it standalone:** `--adjoint-dump-system=<dir>` writes
  `A/B/C/D/rhs/lambda.mm` (standard MatrixMarket, full precision — loadable in
  MATLAB/MRST/Julia/scipy) for one substep, and `adjoint_system_solve <dir> [config.json]`
  rebuilds the transposed system and solves it outside the simulator. Reproduces the
  simulator's λ exactly, so preconditioner settings can be tried on a real field system in
  milliseconds instead of a whole sweep. NB `Dune::readMatrixMarket` cannot load
  rectangular-block matrices (its `readSparseEntries` asserts `row/bcols < N()` where it
  means `row/brows`, and OPM Release builds with `-UNDEBUG`); `matrixMarketIO.hpp` reads
  scalar and re-blocks instead.
- **Solver choice:** serial fastest = `cprt`; parallel auto-selects `cprt` too and now
  matches serial convergence (SPE9: 9.4 iters/solve at np2 and np4, vs serial 9.5). Parallel
  builds the exact consistent transpose `M=Aᵀ` (`buildConsistentParallelTranspose`, verified
  to machine precision; `OPM_ADJOINT_VERIFY_TRANSPOSE=1`), wraps it in
  `GhostLastMatrixAdapter`, and runs opm's parallel `cprt`. **Key fix:** `cprt`'s coarse
  assembly (`PressureTransferPolicy` transpose branch) indexes weights by *column*, which for
  an owned row's ghost columns needs the ghost's OWNER weights — so the weights vector is made
  owner-consistent with `copyOwnerToAll` in the parallel weights lambda before the CPR builds
  its coarse matrix. Without that, ghost weights (from M's identity-invalidated ghost rows)
  are wrong → cross-rank-inconsistent coarse operator → BiCGSTAB false-converges (the real
  cause of the old "cprt falsely converges in parallel"; it was a missing communication, not
  an inherent limit). umfpack single-rank; parallel auto-selects `cprt`. Default reduction
  `1e-14`. Coarse solver amg≈hypre at ≤44k (both correct). Massive improvement over the old
  block-ILU0 fallback (204 iters, diverged at np≥4). **Parallel works on NNC/fault decks
  (Norne) too:** the matrix sparsity includes every grid neighbour but the parallel overlap
  omits zero-transmissibility NNCs, so a few cross-rank couplings are structurally present
  with an off-rank partner — their blocks are exactly zero, so `buildConsistentParallelTranspose`
  drops them (verified `|block|=0`; transpose matches the oracle to ~8e-17 on Norne). A
  *nonzero* unplaceable coupling (diffusion/thermal on a below-trans NNC) throws rather than
  being silently dropped. (Norne fixed-step forward needs loose tolerances +
  `--newton-max-iterations`>20; serial-vs-parallel end-to-end then differs ~2e-4 from the
  forward trajectory, not the adjoint.)
- **Test scripts are bash, not zsh** — the one-liners rely on word-splitting; run with
  `bash`. macOS `/bin/bash` is 3.2 (no `${VAR^^}`); no `timeout` (use `gtimeout`).
- v1 forces `--enable-storage-cache=false` (residual = pure `R(x_k, x_{k-1})`).
