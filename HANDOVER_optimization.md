# Handover — adjoint performance & memory optimization

**Date:** 2026-07-24. **Purpose:** start a fresh session on the *optimization* thread of
opm-adjoint. Orthogonal to the correctness/field-validation thread (`HANDOVER.md`) — different
files, no shared blockers, safe to run in parallel.

## Read first
1. `adjoint_storage_sizing.md` — the measured memory/performance analysis (this thread's basis).
2. `STATUS.md` — "Memory & performance (measured)" section.
3. Auto-memory `adjoint-project-state` — measured numbers + decisions.

## The key measured facts (why this thread exists)
- **Memory: the binding constraint is the backward linear solver, not storage.** SPE9 (9 k cells):
  forward 156 MiB; backward **umfpack 4.1 GB (26×)**, **cprt 356 MiB (2.3×)**. Norne (44 k, measured):
  forward 699 MiB, backward **cprt 778 MiB (1.1×)**; umfpack extrapolates to ~40 GB (OOM). ⇒ with an
  **iterative backward solver the adjoint runs at ~forward memory** — the enabler for 0.1–1 M cells.
- **Recording adds ~0 RAM**; the tape is streamed one substep at a time. Recording *write* adds ~45 s
  at Norne scale (4.5 GB full tape; slim ≈ ⅓). New `Adjoint recording timing` line lands in the
  .DBG/.PRT next to `Adjoint sweep timing`.
- **Wall-clock overhead ≈ 3× a forward run** (Norne). Sweep split is **objective-dependent**:
  SPE9/matchref (per-substep) gradient-accumulation 74%; Norne/pressure-avg transposed-solve 53%.
- **Each backward step is heavy, not "used too many times"**: re-linearize ≈ 8× a forward assembly
  (it reconstructs/reinitializes state), and each transposed solve rebuilds its preconditioner.
  The 147 backward assemblies vs the forward's 940 confirm it's per-call cost, not count.

## Optimization backlog, in priority order (impact × ease)
1. **Slim / one-stop recording** — cut the ~45 s write to ~15 s (slim) or ~0 (one-stop in-RAM;
   Norne slim tape ~1.5 GB fits). *Easy, low-risk, already exists (byte-identical). Biggest cheap win.*
   Files: `AdjointRecorder.hpp`, `AdjointStorage.hpp`. Task **A5** = byte-budgeted spill (turns
   one-stop into disk-streaming when the RAM budget is exceeded — the field default; ~50 lines in
   the in-memory backend + a forced-tiny-budget byte-identity test).
2. **Default the backward solver to cprt for field** (task **#21**) — umfpack 26× RAM vs cprt 2.3×;
   keep umfpack as the small-case oracle; consider size-based auto-selection. cprt validated
   essentially exact (1.9e-9 field-normalized vs umfpack). Files: `AdjointParameters.*`,
   `AdjointSolver.hpp`.
3. **Gradient-accumulation reuse** — for a per-substep well-rate objective this is ~74% of the sweep.
   general-AD re-seeds/re-evaluates per cell each substep; reuse the seeded intensive-quantities
   across the PV/TRANS/PERM families (seed once, accumulate all), or use the analytic path where it
   exists. Files: `AdjointSolver.hpp` (accumulateGeneralGradients_), `AdjointResidualEvaluator.hpp`.
4. **Transposed-solve preconditioner / AMG-hierarchy reuse** — the sweep does N solves on
   same-sparsity matrices; build the cprt/hypre hierarchy once and reuse (currently rebuilt each
   substep). Files: `AdjointLinearSolver.hpp`. **hypre coarse-solve config exists**:
   `tests/adjoint-cprt-hypre.json` — but it needs opm-simulators rebuilt with `HAVE_HYPRE`
   (`USE_HYPRE=ON` in the cache but `HAVE_HYPRE` absent from `config.h`, so `hypre` is unregistered).
5. **Cheaper replay reinit** — the 8×-a-forward-assembly per step is deserialize + IQ recompute;
   lighter reinit / skip full IQ recompute where consecutive substeps share state. *Couples to the
   bitwise-replay work (#22 in the correctness thread) — coordinate.*

## Solver guidance (from the user)
Forward: `cpr_trueimpes` (pure cpr; = cprw if not, but cprw is unregistered in this build).
Backward: `cprt` (validated accurate; hierarchy reuse is the win). umfpack for testing only.
hypre as the CPR **coarse (pressure) solve** — needs the `HAVE_HYPRE` rebuild + a ≥10⁵-cell case
to matter.

## Build / measure
```bash
cd /Users/hnil/OPM_CLEAN/opm_adjoint/builds/release && cmake --build . --target flow_adjoint -j8
```
Measure max RSS with `/usr/bin/time -l` (macOS). Timing breakdowns print to the .DBG/.PRT at the end
of gradient runs. Use `--enable-adaptive-time-stepping=false --enable-storage-cache=false
--threads-per-process=1`, forward `--linear-solver=cpr_trueimpes`, backward
`--adjoint-linear-solver=cprt`.

## Gotcha
Bash test/measure scripts are **bash, not zsh** — inline `$VAR` in the shell tool does not
word-split; put multi-flag strings in a bash script file.

## Suggested first step
**A5 (byte-budgeted spill) + flip backward default to cprt (#21)** — together they make the field
config (one-stop + slim + cprt) run at ~forward memory and streaming IO, the "fast, not
memory/file-intensive" target. Both are localized and byte-identity-testable against the proven path.
