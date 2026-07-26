# Adjoint memory, storage and performance

**Status:** measured note, 2026-07-24. Numbers from `flow_adjoint` on this machine
(Apple M4 Pro, serial, `--enable-storage-cache=false`). Covers three separate
questions that were being conflated: **peak RAM** (can adjoint run the same model
sizes as forward?), **tape storage** (RAM vs disk), and **wall-clock overhead**.

## TL;DR — the binding constraint is the backward *solver*, not the tape

- **The first thing that limits adjoint model size is the backward linear solver's
  memory, not storage.** On SPE9 (9 000 cells) the backward sweep peaks at **4.1 GB
  with umfpack** (direct LU of the transposed, well-coupled Jacobian) vs **156 MB for
  the forward run** — a **26×** blowup. The recorded tape is only 43 MB and is
  streamed one substep at a time, so it is *not* the driver.
- **Switch the backward solve to an iterative solver and the blowup disappears**:
  cprt **356 MB**, ilu0 **347 MB** — about **2.3× the forward**, not 26×. umfpack's
  fill-in scales super-linearly, so at field scale it is fatal (Norne ≈ tens of GB,
  1 M-cell impossible); cprt/ilu0/hypre-coarse scale ~linearly like the forward solve.
  **=> With an iterative backward solver, adjoint runs at essentially forward-model
  size.** This is the enabler for the 0.1–1 M-cell target.
- **Recording is nearly free** in both RAM and time: it adds ~0 MB peak RSS (the
  snapshot is serialized and handed to the archive, not held) and, on these cases,
  <1 ms/snapshot. "IO is not free" — true — but at these sizes it is a rounding error
  next to the sweep; the note quantifies it rather than assuming it away.
- **Backward wall-clock (rebuilt 2026-07-25 — the numbers below in "Performance" are
  the ORIGINAL profile; superseded).** The sweep was found to be dominated not by
  gradient accumulation but by the **dCTF/dperm well term** doing whole-domain
  re-linearizations per perforation per substep (96 % of the SPE9 sweep). Made analytic
  (exact, O(perforations)); with that + cprt-default + preconditioner reuse + a dead-load
  removal, the **SPE9 sweep went 24 s → 1 s and Norne 71 s → 8.9 s (0.62× forward)**, and
  the sweep now scales ~linearly. See STATUS.md "Memory & performance" for the current
  profile; the per-phase figures in the "Performance" section below are retained as the
  pre-optimization baseline.
- **Tape storage is a real but second-order constraint** (details below): the
  two-process/disk path streams it at ~forward memory; only the in-RAM one-stop must
  hold the whole tape, so one-stop is a dev/small-case convenience and the field path
  is record-to-disk (what the A5 spill formalises). Norne (~44 k cells) is a *small*
  field; targets are **0.1–1 M cells, tens–hundreds of wells, hundreds–thousands of
  substeps**.

## Peak memory (measured, `/usr/bin/time -l` max RSS)

SPE9, 9 000 active cells, 21 substeps:

| run | wall | max RSS | ×forward |
|---|---:|---:|---:|
| forward only | 2.7 s | **156 MiB** | 1.0 |
| forward + record (full **or** slim) | 2.6 s | ~155 MiB | ~1.0 |
| backward, **umfpack** (2-proc, tape from disk) | 32 s | **4103 MiB** | **26×** |
| backward, **cprt** | 27 s | **356 MiB** | 2.3× |
| backward, **ilu0** | 28 s | 347 MiB | 2.2× |
| one-stop (fwd+tape+bwd in RAM), umfpack | 34 s | 5491 MiB | 35× |

Norne, **44 431 active cells**, 147 substeps — real field case, **measured** (not
extrapolated), pressure-average objective, cprt backward:

| run | wall | max RSS | ×forward |
|---|---:|---:|---:|
| forward only | 57 s | **699 MiB** | 1.0 |
| forward + record (full, 4.5 GB tape) | 102 s | 756 MiB | 1.08 |
| **backward, cprt** | 63 s | **778 MiB** | **1.11** |
| backward, umfpack (extrapolated, ~O(n¹·⁵) fill) | — | ~40 GB → **OOM** | ~57× |

Reading: **the full Norne adjoint runs at 1.1× the forward memory with cprt** — you do
*not* need a smaller model. umfpack at this size would need tens of GB (likely OOM).
**Recording costs ~0 extra RAM** (756 vs 699 MiB) but ~45 s of extra *time* here
(writing the 4.5 GB full tape — "IO is not free" at field scale; slim ≈ 1.5 GB would
roughly a third that). Use umfpack only as the small-case verification oracle; default
field runs to an iterative backward solver (`--adjoint-linear-solver=cprt`, or the
hypre-coarse CPR config `tests/adjoint-cprt-hypre.json` once hypre is compiled in).
(NB: Norne gradients here carry the documented ~90 %-bitwise replay drift at
well-event/hysteresis boundaries — a v1 approximation; this run measures cost, not
gradient accuracy, which is the Norne-shakeout task.)

## Performance: where the wall-clock goes

The backward sweep is already instrumented; the breakdown (and the new recording
timing) print to the `.PRT`/`.DBG` at the end of the run. SPE9, umfpack backward,
31.8 s total:

```
replay (re-linearize)    0.47 s   1.5%
schur  (well contrib.)   0.00 s   0.0%
solve  (transposed)      7.69 s  24.2%    <- solver: also the memory driver
gradients (pv/trans/w)  23.62 s  74.3%    <- general-AD accumulation: the time driver
endpoints (FD)           0.00 s   0.0%
```

The split is **objective-dependent**. Norne (147 substeps, cprt, *pressure-average*
objective — only the final substep contributes to the rhs) inverts it:

```
replay (re-linearize)   18.68 s  30.4%
solve  (transposed)     32.75 s  53.3%    <- solve dominates here
gradients (pv/trans/w)   9.96 s  16.2%    <- cheap: rhs is zero except last substep
```

A per-substep objective (matchref) makes gradient accumulation dominate; a
final-substep objective (pressure-average) makes the solve dominate. Both knobs matter.

### The forward run itself, for comparison

The simulator prints its own phase breakdown at the end of every run; putting it next
to the adjoint sweep makes the components comparable (the adjoint sweep is essentially
*one Jacobian re-linearization + one transposed solve per recorded substep*, plus
gradient accumulation, which the forward has no analogue for):

| phase | Norne fwd (57.5 s, 150 steps) | SPE9 fwd (2.7 s, 21 steps) |
|---|---:|---:|
| assembly (linearize) | 15.4 s (27 %) | 0.51 s (19 %) |
| linear solve | 21.5 s (37 %) | 1.77 s (65 %) |
| props/update (IQ) | 6.6 s (11 %) | 0.29 s (11 %) |
| pre/post step | 13.7 s (24 %) | 0.02 s (1 %) |
| output write | 0.09 s | — |

Component-by-component (Norne):

- **adjoint re-linearize 18.7 s ≈ forward assembly 15.4 s** — both re-assemble the
  reservoir Jacobian; the adjoint does it once per substep (147×), the forward once per
  Newton iteration (≈940 total). Per-linearization they match.
- **adjoint transposed solve 32.8 s vs forward linear solve 21.5 s** — the adjoint does
  one transposed solve per substep (147) at tight reduction; the forward does ≈940
  looser iterative solves. Same order; the adjoint's per-solve cost is a bit higher
  (tight tol, no warm start), and this is where cprt hierarchy reuse would help most.
- **gradient accumulation 10 s has no forward analogue** — it is the price of the
  derivative, objective-dependent (16 % here, 74 % on SPE9/matchref).

- **Overhead factor:** one adjoint gradient (record + backward) ≈ **3× a forward run**
  on Norne (102 s record + 63 s sweep vs 57 s forward) and ≈ **10–12×** on SPE9 (the
  per-substep matchref objective loads the gradient phase). The sweep + record, not any
  single piece, is the cost. Rule of thumb for the field target: **budget one gradient
  at ≈ 3 forward runs** (a per-substep rate/BHP objective pushes that up via the
  gradient phase; a direct solver pushes memory, not this factor).
- **Memory knob = solver** (§ above). **Speed knob = gradient accumulation** (74 %):
  the general-AD path re-seeds and re-evaluates per cell each substep. Analytic
  accumulation is cheaper but per-family; a future speed win is reusing the AD
  seeding/IQ across families, or the analytic path where it exists.
- **cprt hierarchy reuse** would cut the solve further: the sweep does N transposed
  solves on matrices with the *same sparsity*, so the AMG hierarchy (cprt/hypre setup)
  can be built once and reused — currently rebuilt each substep.
- **Recording/IO** (new `Adjoint recording timing` line): SPE1 121 snapshots in
  0.073 s (0.6 ms each). Small, but it is now measured per run, so a field deck where
  IO *does* bite will show it.

## hypre for the coarse (pressure) solve

`tests/adjoint-cprt-hypre.json` configures the backward solver as transposed CPR
(`cprt`) with **hypre BoomerAMG as the coarse/pressure solve** — intended as the
memory-light replacement for umfpack at field scale.

**Now built and registered (2026-07, this machine).** `USE_HYPRE=ON` + `USE_MPI=ON`
and the opm-adjoint `CMakeLists` links `HYPRE::HYPRE` **PUBLIC**, so the target's
`HAVE_HYPRE=1` interface definition reaches `flow_adjoint` (the `HyprePreconditioner`
symbol is in the binary and the config runs — no more "not registered" abort). Note
`HAVE_HYPRE` is a *compile definition carried by the imported target*, not a `config.h`
entry, which is why grepping `config.h` for it finds nothing.

**But measured: hypre-coarse is NOT the default and does not yet pay off here.** Two
findings from running it:
1. **Tolerance trap (fixed):** the json's outer `tol` must be tight (**1e-14**, matching
   the default adjoint reduction). The earlier `tol: 1e-2` let the backward solve stop
   early → **~38% gradient error vs umfpack**. The shipped json is now `tol 1e-14`,
   `maxiter 300`.
2. **hypre-coarse loses to the built-in dune-istl AMG coarse of the plain `cprt` preset
   at every reachable scale:** SPE9 (9k) it is ~15% slower (0.52 s vs 0.45 s, both
   accurate to ~1e-13); **Norne (44k) the transposed system fails to converge to 1e-14
   (stalls ~5e-9 in 300 iters) and aborts.** Even the json-based `amg` coarse
   (`coarsesolver.preconditioner.type: amg`, which does converge) is slower than the
   `cprt` preset on Norne (6.5 s vs 4.6 s).

**Conclusion:** keep the default `auto`/`cprt` path — it wins at ≤44k. hypre only becomes
worthwhile with (a) a ≥10⁵-cell case *and* (b) BoomerAMG options tuned for the transposed
adjoint system (strong-threshold / coarsening / relaxation); it is not a drop-in win as
written. The `amg` coarse json is the working (if slower) stand-in
(`coarsesolver.preconditioner.type: amg`).

**Why strict tolerance is non-negotiable for the backward solve (general principle).** The
adjoint sweep is a *single linear solve* per substep — there is no outer nonlinear loop to
absorb an inaccurate solve, so the linear reduction *is* the gradient accuracy (`λ` error
multiplies `dR/dm` straight into `dJ/dm`). Contrast the forward run, where each Newton step's
linear solve can be loose because the surrounding nonlinear iteration corrects it. So any
adjoint linear-solver config (cprt preset or json) must use a tight reduction (default
`1e-14`); the json's original `1e-2` produced ~38% gradient error. Corollary worth testing:
where the substep's nonlinear coupling is weak (near-linear residual), the solve should
converge to strict tolerance cheaply, so a tight target costs little — another reason not to
loosen it.

**OPEN — investigate coarse/AMG settings for the *transposed* system.** The transposed CPR
system is not spectrally identical to the forward pressure system BoomerAMG's defaults are
tuned for; the Norne stall (~5e-9, no convergence to 1e-14) is most likely mis-tuned
coarsening/smoothing on the transpose, not a fundamental limit. Worth a dedicated sweep of
BoomerAMG strong-threshold, coarsening (HMIS/PMIS vs Falgout), relaxation (hybrid-GS vs
`ℓ1`), and cycle type on the transposed operator before judging hypre unusable at field
scale. The built-in dune-istl AMG coarse converges on the same system, which is evidence the
problem is hypre *settings*, not the transposed formulation.

---

## Tape storage (RAM vs disk) — the second-order constraint

## What one snapshot contains

The recorder (`AdjointRecorder`) stores, per accepted substep, everything the backward
sweep re-linearizes against: the primary solution at that step, and the
history-dependent problem state that the residual `R(x_k, x_{k-1})` reads but does not
carry as a primary (hysteresis, max pressures/saturations, Rs/Rv trackers, guide
rates, summary/UDQ/action state). Two record formats:

| format | what it stores per substep | proven? |
|---|---|---|
| **full** | the *entire* simulator via `serializeOp` — one blob incl. the **static Eclipse schedule** + time scalars, plus a separate WGState | yes (same path as `--save-step`) |
| **slim** | dynamic state only: `model` (solution), `problem` (history state), `summary`/`udq`/`action`. Schedule + scalars reconstructed from the deck at replay | yes on non-schedule-modifying decks; gated by a byte-identity test vs full |

Slim is **not** allowed on decks that mutate the schedule at runtime (ACTIONX / UDQ-driven
WELOPEN/WELTARG) — there the reconstructed schedule would differ. Those decks must use full.

## Measured footprint (directory store, byte-exact `du`)

| case | cells | substeps | full KiB/substep | slim KiB/substep | slim saving |
|---|---:|---:|---:|---:|---:|
| SPE1 | 300 | 120 | 225.8 | 96.9 | 57% |
| SPE9 (short) | 9 000 | 21 | 2372.4 | 2126.9 | **11%** |

Per-dataset breakdown of one SPE9 substep:

```
full:  simulator.bin 2156 KiB  (includes static schedule)  + wgstate 120 KiB
slim:  problem.bin   1336 KiB  (per-cell history state)     <- field-scale bottleneck
       model.bin      548 KiB  (primary solution, 3 eq)
       summary.bin    144 KiB  udq/action ~8 KiB
```

Reading the two rows together tells the real story:

- **SPE1:** the schedule (120 report steps, 4 wells) is large relative to 300 cells, so
  dropping it in slim nearly halves the snapshot.
- **SPE9:** at 9 000 cells the per-cell `problem.bin` + `model.bin` dominate; the
  schedule slim removes is now noise. **Slim's win collapses to 11%, and it keeps
  shrinking as cell count grows.** Per cell: slim ≈ 231 B, of which ~152 B is
  history state (`problem`) and ~62 B is the solution (`model`); the raw 3-primary
  solution is only 24 B, so even `model.bin` carries ~2.5× indexing/flag overhead.

The consequence: **the per-cell dynamic state is an irreducible floor that neither slim
nor any schedule trick can lower.** Only storing *less often* (checkpointing) or storing
*elsewhere* (disk) changes the field-scale picture.

## Scaling to real fields (slim, 230 B/cell/substep)

Tape size = `230 B × n_cells × n_substeps`. Per-substep and total-tape:

| cells | KiB/substep | tape @ 500 substeps | tape @ 2000 substeps |
|---|---:|---:|---:|
| 9 000 (SPE9) | 2.0 MiB | 1.0 GB | 4.0 GB |
| 100 000 | 22 MiB | **11 GB** | **44 GB** |
| 500 000 | 110 MiB | 55 GB | 220 GB |
| 1 000 000 | 220 MiB | **110 GB** | **440 GB** |

(Full adds the schedule per substep — deck-dependent, grows with wells × report steps,
so on a many-well field deck the full tape is materially larger than the slim column.)

A 20–30 year history match on a 0.1–1 M-cell field lands in the **tens to hundreds of
GB** of tape. That is the number that decides the storage strategy.

## Do we need disk? Yes — for field.

Three regimes, in order of increasing case size:

1. **All-in-RAM (current one-stop, `--adjoint-mode=onestop`).** Forward records into an
   in-memory archive; the backward sweep reads it in the same process; zero files.
   Fastest, simplest, no IO. **Viable only while the whole tape fits in RAM** — i.e.
   SPE-scale to a few GB. Good for development, CI, unit-scale gradient checks. **Not a
   field path**: a 1 M-cell tape (110–440 GB) will not sit in a compute node's RAM
   alongside the live simulator.

2. **Stream to disk (the field default).** Write each substep to disk during the forward
   run; the backward sweep consumes newest-first and can free as it goes, so **peak RAM
   is O(1–2 substeps)** while disk holds the full tape. IO is one sequential write + one
   sequential read of the tape — on HPC scratch this is a small fraction of the forward
   simulation's own compute and IO. This is what makes 0.1–1 M-cell adjoints practical.
   **This is what `A5`'s byte-budgeted spill converges to**: when the RAM budget is set
   below the tape size, "spill the oldest substeps" *is* disk streaming, with an in-RAM
   tail for free when the case is small. So the spill is not polish — it is the
   mechanism that lets one binary cover regimes 1 and 2 with one code path.

3. **Checkpoint + recompute (future, if disk-bound).** Store only sparse checkpoints and
   recompute forward segments during the backward sweep (Griewank/Revolve). Trades
   simulation compute for storage — the right tool only when even disk is constrained
   (very large cells × very many substeps). Not needed yet; noted for completeness.

### Why not just store restart files like the simulator does?

The simulator's `UNRST` keeps *report-step* states for post-processing. The adjoint needs
**every accepted substep** (the backward recursion couples consecutive substeps through
`R(x_k, x_{k-1})`), plus history state the restart doesn't carry in adjoint-usable form.
So the adjoint tape is inherently finer-grained and larger per step than a restart file —
another reason it belongs on scratch disk, streamed, not in RAM.

## Recommendation

- **Default the field path to disk-streaming.** The two-process record-to-disk path
  already does this at O(1–2 substeps) peak RAM — so field-scale disk placement exists
  today. Keep the pure in-RAM one-stop as the small-case / dev default.
- **A5 / WS2 (byte-budgeted spill) does NOT reduce storage** and is DEFERRED (2026-07-25
  decision). It only relocates the *one-stop* mode's in-RAM tape to disk above a budget —
  a code-path-unification convenience over the already-existing two-process disk stream.
  Total bytes stored are unchanged (still 230 B/cell/substep). Because it sits on the
  storage↔initialization/replay-reinit path ("wrong storing → hard-to-find bugs") for
  zero storage payoff, it is not worth the correctness risk while performance is adequate.
  Revisit only if a single-process in-RAM run is specifically needed on a case whose tape
  exceeds node RAM.
- **Do not invest further in slimming `problem.bin`** as the field lever — risk-heavy and,
  per the SPE9 crossover, buys little at scale. Gate any trim by the full-vs-slim
  byte-identity test.
- **Checkpoint-recompute is the only real "reduce storage significantly" lever** (store
  O(√N)/O(log N) checkpoints, recompute forward segments — Griewank/Revolve). It is also
  the most correctness-critical: the recompute must be *bitwise-exact*, so it is coupled
  to the bitwise-replay work (#22) and is a deliberate future effort, not a quick win.
  Only pursue when a concrete case is genuinely disk-bound.

## Open follow-ups

- `A5`: byte-budgeted spill in the in-memory backend (turns regime 1 into regime 2). ~50
  lines; the tail-in-RAM/rest-on-disk split; verified by a forced-tiny-budget spill test
  giving byte-identical gradients.
- Measure a real many-well, many-report-step deck (Norne) to get the *full*-format
  schedule contribution at field well counts, and confirm the 230 B/cell/substep slim
  floor holds at 10⁵–10⁶ cells.
