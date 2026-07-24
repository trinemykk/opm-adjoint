# Adjoint storage sizing and the case for disk

**Status:** measured note, 2026-07-24. Numbers from `flow_adjoint` on this machine
(directory store, `--enable-storage-cache=false`, non-adaptive). Supersedes the
optimistic "slim fits in RAM" framing in earlier notes for the field-scale case.

## TL;DR

- A forward run stores a **tape**: one state snapshot per *accepted substep*, needed
  by the backward sweep. Its size is `bytes_per_cell × n_cells × n_substeps`.
- Measured **slim** cost is **~230 bytes/cell/substep** for black-oil (3 eq). Full
  (proven) snapshots add the static schedule on top — dominant on small decks,
  minor at field scale.
- **Slim only helps small / schedule-heavy decks.** At 9 000 cells it already saves
  just 11%, because the per-cell dynamic state — not the schedule — dominates.
- **For real fields, an all-in-RAM tape is not viable**, matching the intuition that
  drove this note. A 1 M-cell history match wants **100–450 GB** of tape. That must
  live on disk (streamed), or be avoided by checkpoint-recompute. The in-RAM one-stop
  archive is a **development / small-case** convenience, not the field path.
- Norne (~44 k active cells) is a *small* field. Target field sizes are **0.1–1 M
  cells with tens to hundreds of wells** and 20–30 years of monthly-plus reporting,
  i.e. **hundreds to a few thousand substeps**. The tape grows with the product.

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

- **Default the field path to disk-streaming**, with the RAM budget as the knob:
  in-RAM below the budget, spill above it (regimes 1→2 on one code path). Keep the
  pure in-RAM one-stop as the small-case / dev default.
- **Do not invest further in slimming `problem.bin`** as the field lever — it is
  risk-heavy (the user's "wrong storing introduces hard-to-find bugs") and, per the
  SPE9 crossover, buys little at scale. Any such trim must stay gated by the
  full-vs-slim byte-identity test. The field lever is *where* the tape lives (disk),
  not *how small* each snapshot is.
- **Checkpoint-recompute** only if a concrete case is disk-bound.

## Open follow-ups

- `A5`: byte-budgeted spill in the in-memory backend (turns regime 1 into regime 2). ~50
  lines; the tail-in-RAM/rest-on-disk split; verified by a forced-tiny-budget spill test
  giving byte-identical gradients.
- Measure a real many-well, many-report-step deck (Norne) to get the *full*-format
  schedule contribution at field well counts, and confirm the 230 B/cell/substep slim
  floor holds at 10⁵–10⁶ cells.
