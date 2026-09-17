# Handover — reducing adjoint archive storage

**Goal:** make field-scale adjoint runs practical. The backward solve is no longer the
constraint (it runs at ~1.2× forward memory and ~1.5× forward time). **The archive is.**
A full Norne history match currently needs **~7.3 GB of snapshots per forward run**, and a
history-match loop repeats that every iteration.

Status of everything else: 51/51 tests green, serial + parallel, `system_cprt` shipped.
See `CLAUDE.md` and `STATUS.md` for the solver side; this document is only about storage.

---

## 0. RESOLVED 2026-07-27 — the archive was never compressed

**The primary target is met by a one-line upstream fix; §1–§3 below are the pre-fix
analysis, kept for the attribution tables.**

`HDF5File` has always applied deflate-1 + chunking to every dataset
(`getCompression`, opm-simulators/opm/simulators/utils/HDF5File.cpp), but it was guarded by
`#if H5_VERS_MINOR > 8` — added 2023 (`52008d523`) to exclude HDF5 ≤ 1.8, and correct while
HDF5 was 1.x. **HDF5 2.x restarts the minor number at 0**, so on HDF5 2.1.1 (this machine)
the guard is false and compression was silently off. Verified before the fix: `h5ls -v`
reported contiguous datasets at 100.00 % utilization with no filter.

Fixed by replacing both occurrences with `OPM_HDF5_AFTER_1_8`
(`H5_VERS_MAJOR > 1 || H5_VERS_MINOR > 8`) plus a `size > 0` guard before `H5Pset_chunk`
(a zero chunk dimension is an error, and parallel runs legitimately produce empty per-rank
datasets). The same stale guard also selected the pre-1.10 `H5Gget_objinfo` branch in
`groupExists`. **Upstreamed as OPM/opm-simulators#7233 (merged 2026-07-29)**; any older
opm-simulators on HDF5 2.x still writes uncompressed files.

### Measured effect (Norne, 44 431 cells, 24 report steps, `NORNE_ATW2013_ADJSHORT*`)

| deck | format | before | after | ratio |
|---|---|---:|---:|---:|
| with tracers | full | 741.7 MiB | **85.0 MiB** | 8.7× |
| with tracers | slim | 692.7 MiB | **64.5 MiB** | 10.7× |
| no tracers | full | 266.7 MiB | 80.6 MiB | 3.3× |
| no tracers | slim | 217.6 MiB | 60.2 MiB | 3.6× |

**Against the right yardstick — what the run already writes.** One Norne restart step
(`NORNE_ATW2013.UNRST`, 18 float32 cell arrays) is **3.20 MB = 72 B/cell**. A late-time
(substep 23) compressed slim snapshot is **3.36 MB = 1.05× one restart step**, versus 29.7 MiB
= 9.3× before. Projected full 250-substep history: **~0.84 GB, versus ~7.3 GB** — the §7
success criterion, met without dropping any state.

Cost: recording went 11.8 → 79.1 ms/snapshot (deflate at ~366 MB/s), but it writes ~9× fewer
bytes; forward wall clock rose 16.2 → 19.0 s on this deck, where the archive lands on a local
SSD. On slower/shared storage the IO saving dominates and recording gets cheaper.

**Do not use the byte-shuffle filter.** Measured 6.2× vs plain deflate's 8.8× — this data is
zero-runs and repeated defaults, not smooth float fields. deflate level 1 is also the right
level (level 6 buys 11 % more for 2.2× the compression time).

### Verification of the fix
- `ctest` 51/51.
- **Bitwise replay from a compressed archive** (SPE9, `--adjoint-save-system=true`):
  `21/21 substeps within tolerance 1.0e-12, worst residual rel diff 0.000e+00, worst Jacobian
  rel diff 0.000e+00`.
- **np=2 Norne**: collective filtered writes work; both per-rank datasets show
  `Filter-0: deflate-1`, and the parallel gradient sweep completes.

### Detailed storage use, per component

Measured on the late-time snapshot (substep 23) of the slim archive, Norne 44 431 cells.
Everything here is **per cell**, so it carries directly to other model sizes.

| component | uncompressed | compressed | ratio | what it is |
|---|---:|---:|---:|---|
| `model` | 62.0 B/cell | 31.8 B/cell | 1.95× | 2 time levels × (3 doubles + 5 `uint8` meanings + `uint16` pvtRegionIdx) |
| `problem`, no tracers | 140.0 B/cell | 29.3 B/cell | 4.78× | 130 material-law + ~10 `maxOilSaturation_`/well model |
| `problem`, 7 tracers | 588.1 B/cell | 42.6 B/cell | 13.81× | + 448 B/cell of tracer state |
| `summary` | ~50 KB/snapshot | — | — | per well/keyword, **not** per cell |
| **total, no tracers** | **202.0 B/cell** | **61.1 B/cell** | **3.31×** | |
| **total, 7 tracers** | **650.1 B/cell** | **74.4 B/cell** | **8.74×** | |

Yardstick: one Norne restart step is 72.0 B/cell. A compressed adjoint snapshot is
**0.85× (no tracers) to 1.03× (tracers) one restart step** — i.e. recording the adjoint tape
now costs about what writing a restart file at every step costs.

The ratios are very unequal, and that matters for what to optimise next. `model` is the
solution itself, high-entropy, and compresses only 1.95×; it is also already provably minimal.
`problem` compresses 4.8–13.8× because it is dominated by constant or slowly-varying per-cell
tuples. **Post-compression `model` is 52 % of a tracer-free snapshot** — the composition has
inverted, and no serialization trimming can touch it. Checkpointing is the only remaining
lever there.

### Compression and decompression cost, measured on Norne

Write side (full format, tracer deck, 29.7 MiB/snapshot uncompressed):

| | before | after |
|---|---:|---:|
| recording, fused serialize+write | 11.8 ms/snapshot | 79.1 ms/snapshot |
| forward wall clock | 16.16 s | 19.03 s (+17.8 %) |
| bytes written | 741.7 MiB | 85.0 MiB |

The 67 ms delta over 29.7 MiB is **441 MB/s effective deflate-1 throughput**.

Read side (backward sweep, 2 snapshot loads per substep, 48 loads):

| | uncompressed | compressed |
|---|---:|---:|
| `loadSnapshot` total | 0.976 s (12 % of sweep) | 1.472 s (17 % of sweep) |
| per load | 20.3 ms | 30.7 ms |
| sweep total | 8.10 s | 8.47 s (+4.6 %) |

The 10.3 ms delta per load over 29.7 MiB is **2.87 GB/s effective inflate throughput** —
decompression is ~6.5× cheaper than compression, as expected for deflate.

So on this machine, with the archive on a local SSD, compression is a **17.8 % forward
penalty and a 4.6 % sweep penalty** in exchange for 8.7× fewer bytes. On shared or network
scratch, where write bandwidth is the constraint rather than CPU, the trade flips and
recording gets faster.

### Extrapolation to larger models

Every quantity above is per cell and the substep count is set by the schedule, not the grid,
so extrapolation is linear in cells. Tracer-free (the realistic field-HM configuration):

| cells | snapshot uncompr. | snapshot compr. | deflate | inflate (×2/substep) |
|---:|---:|---:|---:|---:|
| 44 431 (Norne) | 9.0 MB | 2.7 MB | 20 ms | 6 ms |
| 100 000 | 20.2 MB | 6.1 MB | 46 ms | 14 ms |
| 500 000 | 101.0 MB | 30.5 MB | 229 ms | 70 ms |
| **1 000 000** | **202.0 MB** | **61.1 MB** | **458 ms** | **141 ms** |

Whole archive, compressed / uncompressed:

| cells | 500 substeps | 2000 substeps |
|---:|---:|---:|
| 44 431 | 1.4 / 4.5 GB | 5.4 / 18.0 GB |
| 100 000 | 3.1 / 10.1 GB | 12.2 / 40.4 GB |
| 500 000 | 15.3 / 50.5 GB | 61.1 / 202.0 GB |
| **1 000 000** | **30.5 / 101.0 GB** | **122.2 / 404.0 GB** |

Reading these:

- **A 1 M-cell, 500-substep history match needs ~30 GB of scratch per forward run**, against
  ~101 GB before the fix. That is ordinary HPC scratch, and it is the same order as writing a
  restart file at every step (72 MB/step × 500 = 36 GB), which is an accepted cost.
- **2000 substeps at 1 M cells (~122 GB) is where this stops being comfortable.** That is the
  regime where checkpoint/recompute (§3 Task 4) finally earns its complexity — and note that
  `model`, the part checkpointing addresses, is exactly the part compression cannot help.
- **CPU cost of compression stops mattering at scale, because it is per rank.** 458 ms/snapshot
  serial at 1 M cells is 229 s over a 500-substep record — but a 1 M-cell case runs on tens of
  ranks, and each rank deflates only its own partition: ~4 s of wall clock at 64 ranks.
  Decompression (141 ms/substep serial, ~2 ms at 64 ranks) is noise against a sweep that is
  dominated by one transposed solve per substep.
- **The aggregate ratio degrades as models get cleaner.** 8.74× on Norne is inflated by the
  tracers; a tracer-free field model sits at ~3.3×, and that is bounded below by `model`'s
  1.95×. Budget 3–3.5×, not 9×.
- **Parallel duplicates overlap cells.** Measured: the same case at np=2 is 69.4 MiB versus
  60.2 MiB serial, **+15 %**, because each rank stores its overlap region. That fraction grows
  with rank count (it scales with total partition surface area), so at np=64 on a 1 M-cell model
  the archive will be materially larger than the serial extrapolation above. **This is
  un-quantified beyond np=2 and is the main open risk in these projections.**

### Is the write cost acceptable? Yes.

Isolated properly (slim, tracer-free, `--adjoint-save` off vs on, two runs each):

| | forward wall |
|---|---:|
| no recording | 14.23 / 14.37 s |
| record + deflate | 15.71 / 15.59 s |

**+9.4 %**, of which the recorder's own fused serialize+write is 1.10 s (44 ms/snapshot). The
earlier +17.8 % figure was the full-format tracer deck — a 3× larger blob — so 9.4 % is the
number for the realistic field configuration. It is CPU, not IO, and it is per rank, so it
shrinks with rank count while the IO saving does not.

### Is compression per-write or cross-timestep? **Per-write only.**

`H5Pset_chunk(dcpl, 1, &size)` sets one chunk equal to the whole dataset, and each dataset is
one (substep, component) pair. So every snapshot is deflated in complete isolation.
Confirmed empirically: concatenating two consecutive snapshots and deflating gives
2 601 212 B against 2 601 158 B for compressing them separately — **no gain at all**, because
deflate's window is 32 KB and the blobs are megabytes.

That leaves real redundancy on the table. Between substeps 22 and 23 of the tracer-free deck,
**93.2 % of `problem` bytes and 61.8 % of `model` bytes are byte-identical**.

### What compression options exist, and what each is worth

Measured on two consecutive snapshots; "vs today" is against independent deflate-1.

| scheme | `model` | `problem` | note |
|---|---:|---:|---|
| deflate-1 independent (today) | 1.00× | 1.00× | |
| deflate-6 independent | 1.01× | 1.07× | 2.2× the CPU |
| zstd-3 independent | 1.05× | 1.13× | needs filter plugin 32015 |
| zstd-19 independent | 1.10× | 1.24× | far too slow to record with |
| **zstd-3 `--long`, concatenated** | **1.40×** | 1.21× | cross-timestep |
| **zstd-19 `--long`, concatenated** | **1.47×** | **1.71×** | cross-timestep |
| XOR-delta vs previous + deflate-1 | 1.03× | 1.34× | cross-timestep |
| shuffle + XOR-delta + deflate-1 | 0.84× | 0.95× | **worse** |

Codec throughput on one 9.0 MB snapshot: deflate-1 214 MB/s compress / 1400 MB/s decompress;
zstd-1 226 / 579 MB/s at a 1.08× better ratio (these zstd figures include subprocess spawn, so
they understate it). Byte-shuffle loses in every combination — this data is zero-runs and
repeated per-cell tuples, not smooth float fields.

Reading the table: **switching codec buys ~1.1×; going cross-timestep buys 1.3–1.7×.** Almost
all of zstd's advantage here is the long window, not the codec — independent zstd-3 gains only
1.05× on `model`, but zstd-3 `--long` over the concatenation gains 1.40×.

**Recommendation: do neither now.** Each route has a real cost:
- *zstd* is not in stock HDF5. It needs the H5Z-ZSTD plugin (filter id 32015) present wherever
  the archive is read — `h5ls`/`h5dump` included. That is a portability regression for a 1.1×
  codec gain.
- *Cross-timestep* requires either chunks spanning several substeps (an archive restructure) or
  explicit delta encoding in the adjoint layer. Both sit squarely on the
  storage↔replay-reinitialisation path that `adjoint_storage_sizing.md` flags as
  "wrong storing → hard-to-find bugs", for ~1.5×.

Revisit only in the 1 M-cell × 2000-substep regime (§ extrapolation). Note that even there,
**checkpointing is the better tool than more compression**: the component that compresses worst
is `model` (1.95×), and `model` is exactly what checkpointing eliminates.

### What compression does and does not solve
- **Solves the hysteresis cost.** `EclHysteresisTwoPhaseLawParams` is compiled in
  unconditionally, so every deck pays 130 B/cell (2 × (8 doubles + 1 bool)) even with no
  `EHYSTR` — and with hysteresis off the values never change, so it is the *same 65-byte tuple
  repeated per cell*. SPE9's `problem` dataset compresses **26.3×** (1 369 528 → 52 003 B) for
  exactly this reason. No hysteresis-specific optimisation is needed.
- **Does not solve `model`.** The primary solution is genuinely high-entropy: 2.75 → 1.41 MB,
  only **1.95×**. Post-compression the composition flips — `model` is now **42 %** of a snapshot
  and `problem` 56 %. `model` is already provably minimal (62.0 B/cell measured = exactly
  2 time levels × (3 doubles + 5 `uint8` meanings + `uint16` pvtRegionIdx)), so the only lever
  left on it is checkpointing.

---

## 1. Measured baseline

Machine: Apple M4 Pro, Release build, serial, HDF5 archive. All figures measured
2026-07-27 with the current code, not extrapolated unless labelled.

### Per-snapshot cost

| deck | active cells | snapshot | B/cell |
|---|---:|---:|---:|
| SPE1CASE1 | 300 | 87 KiB | ~290 |
| SPE9_CP_SHORT | 9 000 | 2.07 MiB | ~230 |
| **NORNE (truncated, 54 steps)** | **44 431** | **27.8 MiB** | **~660** |

### Whole-archive cost

| deck | substeps | archive | notes |
|---|---:|---:|---|
| SPE9_CP_SHORT | 21 | 49 MiB | slim: 44 MiB (−10 %) |
| NORNE (truncated) | 54 | 1 656 MiB | slim: 1 531 MiB (−8 %) |
| NORNE (full, ~250 steps) | ~250 | **~7.3 GB** | projection from the measured 27.8 MiB/snapshot |

Recording *time* is not a problem: 1.5 ms/snapshot, ~2 % of the forward run. Peak RSS
during recording is +6 %. **Only bytes on disk are the issue.**

### Where the bytes are

The full archive writes one opaque blob per substep, so it tells you nothing. The **slim**
archive (`--adjoint-slim=true`) writes components as separate datasets — use it to
attribute cost:

```bash
h5ls -r CASE.ADJOINT.h5 | grep "substep/10/"
```

| component | SPE9 (9 k) | share | NORNE (44 k) | share | scaling |
|---|---:|---:|---:|---:|---|
| `problem` | 1.37 MB | 65 % | **26.11 MB** | **89.5 %** | 151 → **588 B/cell** |
| `model` | 0.56 MB | 27 % | 2.75 MB | 9.4 % | **62 B/cell (constant)** |
| `summary` | 0.15 MB | 7 % | 0.28 MB | 1.0 % | ~fixed per deck |
| `udq`, `action` | 72 B | ~0 % | 72 B | ~0 % | — |

**Two facts drive everything below:**

1. **`problem` is the target.** It is 65 % of a small deck and **89.5 %** of a field deck.
2. **Its per-cell cost is not constant — it grows with enabled physics** (151 B/cell on
   SPE9 → 588 B/cell on Norne, ~73 doubles per cell). `model` by contrast is a clean
   62 B/cell on both decks. So the growth is *optional per-cell arrays and sub-objects*,
   not the primary variables.

`--adjoint-slim` only buys 8–10 %: it drops the static schedule, which was never the bulk.
Do not expect more from it.

---

## 2. What is actually being serialized

`FlowProblemBlackoil::serializeOp` (opm-simulators/opm/simulators/flow/):

```cpp
serializer(static_cast<FlowProblemType&>(*this));   // -> FlowProblem, below
serializer(mixControls_);                           // DRSDT/DRVDT lastRs_, lastRv_
serializer(*eclWriter_);                            // <-- ECL OUTPUT writer
```

`FlowProblem::serializeOp`:

```cpp
serializer(static_cast<BaseType&>(*this));  // -> FlowGenericProblem
serializer(drift_);                         // GlobalEqVector, numCells * numEq
serializer(wellModel_);
serializer(aquiferModel_);
serializer(tracerModel_);
serializer(*materialLawManager_);           // hysteresis state, per cell
```

`FlowGenericProblem::serializeOp`: `maxOilSaturation_`, `polymer_`, `maxWaterSaturation_`,
`minRefPressure_`, `overburdenPressure_`, `solventSaturation_`, `solventRsw_`,
`bioeffects_` — each a per-cell vector *when the feature is enabled*, empty otherwise.
This is exactly the mechanism behind the 151 → 588 B/cell jump.

**RESOLVED 2026-07-27 — measured, not suspected.** Obtained by deck differencing
(`NORNE_ATW2013_ADJSHORT` vs `..._NOTRACER` vs `..._CLEAN`), no instrumentation needed.

| member | Norne B/cell | share of `problem` | note |
|---|---:|---:|---|
| `tracerModel_` | **448.0** | **76.3 %** | 7 water tracers × 64 B; measured as the exact 587.5 − 139.5 difference |
| `*materialLawManager_` | 130.0 | 22.1 % | 2 × (8 doubles + 1 bool); **paid on every deck**, hysteresis or not |
| `maxOilSaturation_` + `wellModel_` + rest | 9.5 | 1.6 % | |

`problem` = 587.5 B/cell with tracers, **139.5 B/cell without**. `model` = 62.0 B/cell on both.

Three of the four original suspects were wrong:

- **`*eclWriter_` is not a storage item.** `EclWriter::serializeOp` →
  `GenericOutputBlackoilModule::serializeOp` serializes **only `initialInplace_`** — a
  `region-set → phase → region → double` map, ~15 KB, never O(cells). It is write-once and has
  no reader reachable during replay (the sweep never calls `evalSummaryState`). Worth removing
  for cleanliness — an *output* writer has no business in restart serialization, and it does
  feed `FOE`/`ROE`/`ROEW` into `SummaryState` where UDQ/ACTIONX can read them in a forward run
  — but it buys no bytes.
- **`drift_` is already empty.** `EnableDriftCompensation` defaults false, and
  `EnableDriftCompensationTemp`'s `true` default is only applied for the
  `SequentialImplicitThermal` energy module, so blackoil never resizes `drift_`. The 24 B/cell
  was never being paid.
- **`aquiferModel_` is empty** — Norne has no `AQUCT`/`AQUANCON`.
- **`materialLawManager_` was right in size but wrong in cause**: it is 130 B/cell on *both*
  decks, not a Norne-specific jump, because the hysteresis wrapper is a compile-time type.

**Tracers are provably inert to the flow solution.** The only non-output reader of tracer state
in opm-simulators is `assignWellTracerRates_` (fills a `data::Wells` report struct); nothing in
the residual, intensive quantities or well control touches it. Confirmed end-to-end: the tracer
and no-tracer decks give **byte-identical** `ADJOINT_GRADIENTS_{PV,TRANS,PERM,WELLCTRL,
WELLCTRL_BYMODE}.txt` and identical `J = 2.5329917305375706e+07`.

**But after compression the tracer trim is worth far less than 76 %.** At these 24 report steps
the tracers have barely spread, so they compress away: 85.0 → 80.6 MiB full, 64.5 → 60.2 MiB
slim, i.e. ~5 %. Their *compressed* cost does grow steeply with simulated time — 87 KB at
substep 0 to 591 KB at substep 23, still rising — so on a full history the trim recovers value.
Judge it on a long deck, not this one.

---

## 3. Work items, in priority order

> **Status 2026-07-27.** Task 3 (compression) is **done** and met the goal on its own — see §0.
> Task 1 (attribution) is **done** by deck differencing, no instrumentation — see §2.
> Task 2 (stop archiving unused state) is **deferred**: it is safe and proven inert, but
> compression already absorbs ~95 % of its value on a short deck; re-measure on a long deck
> before spending the upstream change. Task 5 (retire slim/full) is untouched.
> Separately landed: drift compensation is now **warned at record and refused at replay**
> (`AdjointRecorder::checkConfiguration_`, `AdjointReplay` ctor) — its accumulated residual
> correction has no term in the adjoint recursion, so supporting it would mean silently wrong
> gradients. Gated on `EnableDriftCompensation` only; the `Temp` variant's `true` default
> applies solely to the sequential-thermal energy module.

### Task 1 — Attribute the `problem` blob (do this first, ~half a day)

Do not optimise blind. Add a temporary instrumented write that serializes each member of
the `problem` chain into its own dataset (mirroring what slim already does one level up),
run SPE9 **and** Norne, and read the sizes with `h5ls -r`.

Placement: a debug branch in `AdjointRecorder::writeSnapshot_`
(`opm/adjoint/AdjointRecorder.hpp`), gated by an env var like the existing
`OPM_ADJOINT_WELLSTATE_DIAG`. Serializing sub-objects individually needs the same
`archive_->write(obj, group, name)` call already used there — each member that has a
`serializeOp` can be written directly.

**Deliverable:** a table like §1's but one level deeper, for both decks. Everything below
should be re-prioritised against it.

### Task 2 — Stop archiving what replay does not read (the likely big win)

The adjoint needs enough state to *re-linearize* at the converged point. That is much less
than a full restart image. Candidates to exclude, cheapest first:

- `*eclWriter_` — output only.
- `drift_` — 24 B/cell.
- `tracerModel_`, `aquiferModel_` — only if the objective and gradients do not touch them.

Placement: this is the interesting design question. Two routes:

- **(a) In opm-adjoint (preferred).** Write the specific pieces replay needs instead of the
  whole `problem`, extending the slim path. Self-contained, no upstream change, and the
  slim/full split already establishes the pattern. Risk: it must track upstream changes to
  `FlowProblem` members.
- **(b) Upstream serialization tags.** Give `serializeOp` a notion of "restart image" vs
  "replay image". Cleaner long-term, much wider blast radius, needs opm-simulators buy-in.

Start with (a). Expected: if `eclWriter_` + `drift_` + unused sub-models are the bulk, this
alone could be the difference between 7.3 GB and ~1 GB on Norne.

### Task 3 — Compression

HDF5 supports transparent gzip/szip filters per dataset. Reservoir state is smooth and
should compress well. This is orthogonal to Tasks 1–2 and multiplies with them.

Placement: `AdjointStorage.hpp` (the HDF5 backend), at dataset creation. Measure both the
size win and the recording-time cost — there is 2 % of forward runtime of headroom before
recording stops being free.

### Task 4 — Checkpoint/recompute instead of storing every substep
> **Superseded by §8 (L1/L2)**, which has the measured storage/time trade-off table, the
> solver-hint scheme that makes the recompute cheap, and the gates to clear first.

The classic adjoint trade: store every N-th state and re-run the forward between
checkpoints during the backward sweep. Trades storage for time at a known ratio
(Griewank). Only worth it if Tasks 1–3 fall short, because it makes the sweep several
times more expensive and complicates the replay driver considerably.

Placement: `AdjointReplay.hpp` + `AdjointRecorder.hpp`. This is a substantially larger
change than 1–3 — treat it as a separate project.

### Task 5 — Reconsider `--adjoint-slim`

Once Task 2 lands, slim's 8–10 % may be subsumed. Either fold it in or retire the flag so
there is one archive format instead of two.

---

## 4. How to verify you have not broken anything

Storage changes are **replay-correctness** changes. The existing gates are the safety net:

```bash
cd builds/release/opm-adjoint && ctest          # 51 tests, ~3 min
```

The ones that matter most here:

| test | what it catches |
|---|---|
| `adjoint_replay_*` | re-linearization no longer reproduces the forward |
| `adjoint_slim_grpctrl` | slim vs full divergence on a group-control deck |
| `adjoint_gradient_regression` | gradient drift |
| `adjoint_fd_*` | gradients vs finite differences |
| `adjoint_mpi_*` | per-rank archive round-trip |

**Bitwise replay is the real criterion** — Milestone A established that replay reproduces
the forward Jacobian exactly. If a dropped field degrades that, an FD test may still pass
while gradients quietly lose digits. Run with `--adjoint-save-system=true` and check the
replay comparison reports zero mismatch, not just "tests pass".

Field-scale check (not in ctest, needs a truncated Norne — see §5):

```bash
# gradients must be unchanged to ~1e-12 before/after a storage change
diff <(...) <(...)   # compare ADJOINT_GRADIENTS_PV.txt
```

---

## 5. Reproducing the field case

Full Norne's fixed-step forward fails to converge around report step 107, so use a
truncated schedule:

```bash
cp -R opm-tests/norne/INCLUDE $SCRATCH/
head -2214 $SCRATCH/INCLUDE/BC0407_HIST01122006.SCH > $SCRATCH/INCLUDE/SHORT.SCH
echo END >> $SCRATCH/INCLUDE/SHORT.SCH
sed 's|BC0407_HIST01122006.SCH|SHORT.SCH|' opm-tests/norne/NORNE_ATW2013.DATA \
  > $SCRATCH/NORNE_SHORT.DATA
```

That gives 54 report steps, 44 431 active cells, 1 656 MiB archive. Run it with:

```bash
flow_adjoint $SCRATCH/NORNE_SHORT.DATA --output-dir=$OUT \
  --threads-per-process=1 --enable-storage-cache=false \
  --enable-adaptive-time-stepping=false --newton-max-iterations=40 \
  --matrix-add-well-contributions=false \
  --adjoint-objective=pressure-average --adjoint-save=true
```

Alternatively run the full deck with adaptive stepping (it converges), but then serial and
parallel trajectories differ and only same-rank-count comparisons are meaningful.

---

## 6. Traps

- **`numLocalWellsEnd()` counts wells at the END of the schedule**, not at the current
  step. I misread this once and chased the wrong hypothesis for an hour. Use
  `wellState().size()`.
- **Norne has zero local wells on some ranks at early report steps.** That is legitimate,
  and it is what exposed the `prepareDeserialize` bug fixed in this branch. Any new
  per-rank archive logic must handle "this rank has nothing" without going stale.
- **Gradient/FD runs need `--enable-adaptive-time-stepping=false`**, or a perturbation
  changes the substep sequence and swamps the comparison.
- **`matchref` reference summaries must live outside `--output-dir`** — gradient runs
  truncate summary files there at startup.
- **Test scripts are bash, not zsh**; macOS `/bin/bash` is 3.2 and there is no `timeout`.
- Debug aid already in place: `OPM_ADJOINT_WELLSTATE_DIAG=1` prints recorded vs
  reconstructed well-state sizes per rank and substep — the model to copy for Task 1.

---

## 7. Success criterion

Norne full history (~250 substeps) under **1 GB** would make field-scale history matching
routine, versus ~7.3 GB today. Based on §1, that needs roughly a 7× reduction, and
`problem` at 89.5 % is where all of it has to come from.

**Met 2026-07-27** by the compression fix alone (§0): projected ~0.84 GB with tracers, ~0.69 GB
without. The section below is for the regimes that are *still* constrained.

---

## 7b. Where checkpointing stands, and what it still needs (2026-07-27)

**Judgement: checkpointing is not needed for the models this adjoint currently targets.** Compression
alone brought full Norne to ~0.84 GB (1.05× one restart step), and the tracer trim and the material-law
guard are still unspent. At Norne scale the archive is no longer the constraint, so the default stays
`--adjoint-checkpoint-every=1` and both the recorder and the sweep warn when it is not.

It **is** the only lever that scales, so it stays on the roadmap for larger models — 0.5–1 M cells with
500–2000 substeps, where the compressed archive is 15–122 GB and per-snapshot size is floored by
`model`'s 1.95× compression. What exists today works on SPE1/SPE9 (see §8 L1 STATUS) and is a usable
foundation, not a finished feature.

**What a field-ready version needs beyond what is there:**

1. **Per-substep branch metadata — the discrete state, not just the continuous one.** The recompute has
   to land on the *same branch* of the code the recording took. That means storing the enumerated
   decisions per substep and bumping the schema version when they are added: realised well and group
   control modes (`WellStatus`, `injection_cmode`, `production_cmode`, `Group::ProductionCMode` /
   `InjectionCMode`), and the primary-variable meanings (`WaterMeaning`/`PressureMeaning`/`GasMeaning`,
   which already ride along inside `model`). These are a handful of bytes per well per substep —
   ~150 B/substep for Norne's 36 wells against 2.4 MB for a snapshot — and they buy two things at once:
   the recompute can *seed* the right branch instead of rediscovering it by switching, and a discrete
   mismatch afterwards is a **hard divergence guard needing no tolerance**, far sharper than comparing
   Newton counts. The substep-0 failure above is exactly the class of bug this catches early.
2. **Lossy-compressed warm-start values.** Store `float32` of the solution at every substep (~31 B/cell,
   ~15 % of a snapshot) as the Newton initial guess. Newton then starts ~7 digits from the answer and
   should converge in 1–2 iterations instead of 5–15, which is what turns Revolve's "3 extra forward
   sweeps" into a fraction of one. Lossy is fine here: it is only a starting point, and the solve still
   converges to full tolerance, so the state is accurate rather than `float32`-accurate.
3. The rest of §8 L1/L2: Revolve scheduling, and deflating `AdjointMemStore` to reclaim the 3.72×
   RAM-vs-disk asymmetry that makes segment RAM the binding constraint at scale.

Both (1) and (2) change the Newton iterate path, so they trade bitwise reproduction for
tolerance-bounded agreement — which must be *measured* against the M=1 reference, not assumed.

---

## 8. Optimization plan for further reduction

Nothing here is needed for Norne. The target regime is **0.5–1 M cells × 500–2000 substeps**,
where the compressed archive is 15–122 GB (§0 extrapolation). Ranked by payoff per unit of
correctness risk.

### L1 — Checkpoint/recompute (Griewank–Revolve). The only lever that scales.

> **IN PROGRESS.** The recording half is **done and tested**; the recompute half is not.
> Landed: `--adjoint-checkpoint-every=M` (default 1 = unchanged), the payload gate in
> `AdjointRecorder::recordSubstep`, `AdjointSubstepMeta::hasPayload` +
> `AdjointMeta::checkpointEvery` (schema version bumped 1 → 2), hard refusal of adaptive
> stepping and TUNING when M > 1, and a clear replay-side refusal of a checkpointed archive
> until the segment driver exists. `ctest` 51/51.
>
> | case | M=1 | M>1 | ratio |
> |---|---:|---:|---:|
> | SPE9_CP_SHORT (21 substeps) | 9.01 MiB | 2.24 MiB (M=4) | 4.0× |
> | SPE9_CP_SHORT | 9.01 MiB | 0.05 MiB (M=100, = one-stop) | 180× |
> | NORNE_ADJSHORT_NOTRACER (24 substeps) | 60.22 MiB | **11.66 MiB (M=5)** | **5.16×** |
>
> Projected full Norne (250 substeps) at M=5: **~0.12 GB**, against 0.84 GB at M=1 and ~7.3 GB
> before any of this work — a **~61× total reduction**. The forward run is marginally *faster*
> (15.47 s vs 16.20 s) since it writes fewer snapshots; the cost moves to the sweep, which is
> where the not-yet-written recompute will pay it.
>
> **Also landed:**
> - `AdjointRecorder` segment mode — `globalIndexOffset` + `segmentMode` ctor args, so a recomputed
>   segment writes **global** group numbers (the sweep reads by global k) and stores every substep
>   (a segment must be fully materialized because `replayStep(k)` needs k and k-1). Segment mode
>   skips the guards, banner and provenance text so it doesn't repeat them per segment.
> - `AdjointFlowProblem::resetRecorder(archive, offset)` / `clearRecorder()` / `hasRecorder()` — the
>   fix for the silent no-record hazard: `recorderInitialized_` latches on the first `beginEpisode()`,
>   which in gradient mode declines a recorder permanently, so a segment recompute would otherwise
>   record nothing and only fail later as a missing dataset.
> - `AdjointReplay::setSegmentArchive()` + `sourceArchive_(k)` — payload lookup switches archive per
>   substep, keyed on the recorded `hasPayload`. Checkpoints and `initial` come from the recorded
>   archive, recomputed substeps from the segment archive. **Nothing is copied between archives and
>   there is no ambiguity**; a stale or wrong segment archive surfaces as a missing dataset, not as
>   silently wrong state. `compareWithStored_` now also gates the residual/Jacobian comparison on
>   `hasPayload`, since the system dumps ride along with the payload.
> - **Upstream (`adjoint-hooks`):** `SimulatorFullyImplicit::setRestoreStateHook(std::function<void(int)>)`,
>   invoked in `runStep` at exactly the OPMRST restore position — after `startNextEpisode`/
>   `setEpisodeIndex`, before the solve. Inactive unless installed; the callback owns the whole
>   restore including `prepareDeserialize` and IQ invalidation, as the OPMRST path does.
>
> ### STATUS: works end-to-end on SPE1/SPE9. Shipped behind a warning, NOT the field path.
>
> `AdjointSegmentSource` re-runs the missing substeps from the preceding checkpoint and hands them to
> the sweep. Equivalence against `--adjoint-checkpoint-every=1`:
>
> | deck | M | PV / TRANS / PERM | WELLCTRL | WELLCTRL_BYMODE |
> |---|---|---|---|---|
> | SPE9_CP_SHORT (21 substeps) | 3, 4, 100 | **byte-identical** | byte-identical | byte-identical |
> | SPE1CASE1 | 3, 4, 100 | **byte-identical** | byte-identical | agrees to ~1e-12 |
>
> J is identical to all 17 digits on both decks at every M, and the recompute costs exactly one extra
> forward pass (N substeps re-run, independent of M). The SPE1 `WELLCTRL_BYMODE` residue is 1.3e-12 to
> 9.6e-12 relative — floating-point summation order, not a trajectory difference; note the per-well
> totals in `WELLCTRL` are byte-identical, so it is the by-mode split that reassociates. The families
> this project actually targets (PORO/PV, PERM, TRANS) are byte-identical on both decks.
>
> **Both the recorder and the sweep now log a warning** that this path is under development and
> verified on SPE1/SPE9, serial, fixed-step only. `--adjoint-checkpoint-every` defaults to 1, so
> nothing changes unless it is asked for. **Norne and np=2 are untested.**
>
> #### Two bugs found and fixed getting there
>
> 1. **Segment format.** The segment recorder took `slim` from the *sweeping* process's
>    `--adjoint-slim` rather than from the recording, so a sweep invoked without the flag wrote a
>    `simulator` blob while the reader wanted `model`/`problem` — surfacing as
>    "missing dataset /adjoint/substep/20/model". It now takes the recorded `AdjointMeta` and matches
>    its format, and forces `saveSystem` off since a recomputed substep has nothing to compare against.
> 2. **Substep 0 could not be recomputed.** Gradients deviated 3.8e-4 in norm, *identically* for
>    M = 2, 7 and 21 — the M-independence is what localized it. Per-substep replay residuals showed
>    substeps 1..20 bitwise exact and **substep 0 at 1.782e+00 instead of 4.872e-09**: re-running the
>    first report step from the `initial` snapshot does not reproduce `R(x_0, x_-1)`, because the
>    snapshot that comes back has the wrong `x_{k-1}` slot. Since substep 1 was exact, the *solve* was
>    right — only the stored history was wrong. Fixed by always checkpointing substep 0 (one extra
>    snapshot out of N/M), and `AdjointSegmentSource` now starts each segment at its first genuinely
>    missing substep and leads from the nearest stored one, so no segment leans on `initial`.
>    The root cause of the `initial`-lead history slot is **not** diagnosed; it is avoided.
>
> A **pre-existing, unrelated crash** turned up while testing: passing both `--linear-solver=ilu0`
> and `--matrix-add-well-contributions=true` segfaults in `Main`'s constructor (`initMPI` /
> `opal_argv_join`) — the argv-injection block in `flow_adjoint.cpp` main mishandles options that are
> already present. Either flag alone is fine. Worth fixing on its own.
>
> #### Design (for reference) — restore `c_j` through the hook, `resetRecorder` onto a fresh
> in-memory archive, `FlowMain::executeStep()` for `(c_j, c_{j+1}]`, then `setSegmentArchive`, sweep,
> `clearRecorder`. Two wrinkles found while designing it, not yet handled:
> 1. **`runStep` writes ECL output** (`writeOutput(true)`, `_impl.hpp:466-472`) and the hook fires on
>    *every* report step. So the driver needs a one-shot latch on the hook, and the recompute must not
>    re-emit ECL output — either force `--enable-ecl-output=false` for a checkpointed sweep or suppress
>    the write for recomputed steps. Left unresolved on purpose; it is a correctness/cleanliness issue
>    for the output files, not for the gradients.
> 2. `evaluateObjectiveOnly` (`AdjointSolver.hpp:215-233`) is a forward-order loop needing every
>    payload — route it through the segment source or refuse it when M > 1.

Store the converged state only every M-th substep. When the backward sweep needs substep `k`
and it was not stored, **re-run the forward simulation** from the nearest earlier checkpoint up
to `k`. Storage traded for forward compute. This is the only option that reduces the *number*
of snapshots rather than the size of each, so it is also the only one that helps with `model`,
which compresses just 1.95×.

Two-level form: load checkpoint `j`, run forward `j…j+M` holding those M states in RAM, reverse
through them, discard, move to the previous segment. Each substep is then recomputed exactly
once, so the whole sweep costs one extra forward pass. Revolve is the optimal schedule: with
`c` checkpoints and `r` extra forward sweeps you can reverse `N` substeps whenever
`N ≤ C(c+r, r)`, so storage falls roughly logarithmically as you allow more recomputation.

| | snapshots | storage | extra forward |
|---|---:|---:|---|
| **Norne, 250 substeps @ 2.7 MB** | | | |
| every substep (today) | 250 | 0.68 GB | 0 |
| uniform, M=16 | 31 | 0.08 GB | 1 sweep |
| revolve, r=2 | 21 | 0.06 GB | 2 sweeps |
| **1 M cells, 2000 substeps @ 61.1 MB** | | | |
| every substep | 2000 | 122.2 GB | 0 |
| uniform, M=45 | 89 | 5.44 GB | 1 sweep |
| revolve, r=2 | 62 | 3.79 GB | 2 sweeps |
| revolve, r=3 | 21 | 1.28 GB | 3 sweeps |

#### How storage and time scale with M (measured 2026-07-27, Norne ADJSHORT_NOTRACER, 24 substeps)

| M | archive MiB | snapshots | forward s | recording s | peak RSS MiB |
|---:|---:|---:|---:|---:|---:|
| 1 | 60.22 | 25 | 15.70 | 1.106 | 639 |
| 2 | 31.29 | 13 | | | 636 |
| 3 | 21.57 | 9 | | | 632 |
| 4 | 16.73 | 7 | 14.85 | 0.313 | 637 |
| 6 | 11.88 | 5 | | | 637 |
| 8 | 9.46 | 4 | | | 638 |
| 12 | 7.02 | 3 | | | 635 |
| 24 | 4.59 | 2 | 14.70 | 0.086 | 636 |
| 48 | 1.95 | 1 | | | 629 |

Storage is **exactly** `floor(N/M) + 1` snapshots × 2.41 MiB — perfectly linear, no overhead term.
Recording time falls in proportion (1.106 → 0.086 s), so the +9.4 % forward penalty at M=1 becomes
~+1 % at M=24. Peak RSS is flat: recording streams, it does not accumulate.

**Cost model for uniform two-level:**

| quantity | scaling |
|---|---|
| disk | `floor(N/M) + 1` snapshots — *measured exact* |
| RAM during the sweep | `M + 1` states (the segment held in memory) |
| extra forward work | **~1 pass, independent of M** — each substep is recomputed exactly once |

The time-independence is worth being explicit about: within uniform two-level, **raising M buys disk
for free in compute terms** and costs only RAM. Time only enters the trade if RAM forces you to
re-run a segment per reversed step (O(M/2) passes, O(1) RAM) — which is what Revolve optimises.

**The binding constraint at scale is RAM, not disk, because of a 3.72× asymmetry.** A snapshot is
2.41 MB on disk (deflated) but **8.98 MB in RAM** — the in-memory segment archive holds raw packed
`MemPacker` buffers with no compression. So segment RAM grows 3.7× faster than disk shrinks, and the
optimum sits well below the naive `M = √N`.

Projection to 1 M cells (61.1 MB compressed / 202 MB packed per snapshot):

| N | M=1 disk | M=√N | disk at √N | **RAM at √N** | RAM budget 8 GB | RAM budget 2 GB |
|---:|---:|---:|---:|---:|---|---|
| 500 | 30.6 GB | 22 | 1.41 GB | 4.65 GB | M≤38 → 0.86 GB | M≤8 → 3.85 GB |
| 2000 | 122.3 GB | 44 | 2.81 GB | **9.09 GB** | M≤38 → 3.24 GB | M≤8 → 15.3 GB |

So 122 GB → ~3 GB is reachable with an 8 GB segment buffer and one extra forward pass. Even a modest
2 GB buffer gets 122 GB → 15 GB.

**Cheap follow-on worth doing:** deflate the buffers in `AdjointMemStore` (it stores
`std::vector<char>` verbatim). At the measured 441 MB/s that reclaims the full 3.7× of segment RAM for
a few percent of sweep time, moving the optimum back toward `√N` and roughly halving the disk figures
above. Contained change, no correctness surface.

**Placement — and this is much less invasive than it sounds.** Almost every primitive already
exists; the work is composition, not new machinery, and it lands in one new driver class rather
than spreading through the existing code.

Already there:

| primitive | where |
|---|---|
| public per-report-step forward advance | `SimulatorFullyImplicit::runStep(SimulatorTimer&)` (`SimulatorFullyImplicit.hpp:273`) |
| **restore-then-continue, already inside `runStep`** | the `--load-step` path: `if (serializer_.shouldLoad()) { prepareDeserialize(); loadState(); invalidateAndUpdateIntensiveQuantities(0); }` (`SimulatorFullyImplicit_impl.hpp:328-332`) — the same sequence as `AdjointReplay::loadSnapshot_`, and one-shot (`loadState()` resets `loadStep_ = -1`, `SimulatorSerializer.cpp:191`) |
| a whole tape held in RAM and handed to the sweep | `AdjointArchive::makeInMemory()` + the injected-archive `AdjointRecorder`/`AdjointSolver` constructors — i.e. **one-stop mode already is a segment recompute, just with one segment covering the whole run** |
| automatic per-substep recording of a recomputed segment | `AdjointFlowProblem::endTimeStep` already fires `recordSubstep` for every accepted substep — point the recorder at an in-memory archive and a recomputed segment records itself with no new code |
| the hints | `AdjointSubstepMeta` (`dt`, `startTime`, `reportStep`, `newtonIterations`), `assembleWellEqGivenControls` |

So a checkpoint is exactly a current snapshot, and a segment recompute is exactly the existing
one-stop driver applied to a sub-range. What actually has to be written:

1. **One `if` in `AdjointRecorder::recordSubstep`** — write the payload only on checkpoint
   substeps. Keep writing `AdjointSubstepMeta` for *every* substep; the sweep needs `dt` and
   `reportStep` throughout, and it is a handful of bytes.
2. **A segment driver** (the one genuinely new class): restore checkpoint → construct an
   in-memory recorder → `runStep` M times → hand that archive to the sweep for those substeps →
   discard, move to the previous segment. This is `runOneStop` (`examples/flow_adjoint.cpp:118`)
   generalised from "the whole run" to "a sub-range", so use that as the template.
3. **Schedule bookkeeping** — uniform M is trivial arithmetic; Revolve is a known ~200-line
   algorithm, and only worth it once uniform is proven.

One real constraint falls out of reusing the `--load-step` restore: it loads at a **report-step**
group boundary, while adjoint substeps are finer. So place checkpoints on report-step boundaries.
That is natural anyway, and it keeps the whole restore path stock rather than bespoke.

**Gates before starting — the main one is now MEASURED and it PASSES (2026-07-27).**

The open risk was: a recompute re-runs the **nonlinear solve**, including the iterative linear solver,
which is not bitwise reproducible in general. Tested directly with stock `--save-step`/`--load-step`
plus `--adjoint-save-system=true`, comparing the archive's numeric dumps:

| deck | restart at | result |
|---|---|---|
| SPE9_CP_SHORT (21 substeps) | report step 10 | **all 11 restarted substeps bitwise identical** — residual, Jacobian and `model` |
| NORNE_ATW2013_ADJSHORT (24 substeps, 36 wells, hysteresis, group control, 7 tracers) | report step 12 | **residual and Jacobian bitwise identical**; `model` bitwise identical for all 12 substeps |

**So checkpoint/recompute can be exact and the bitwise equivalence gate is attainable.**

**One legitimate, inert difference — do not mistake it for a failure.** On Norne the `problem` blob
differs by ~582–631 bytes out of 26 MB, in a 772-byte span starting at offset ≈355 648 — right where
`wellModel_` begins, just past `maxOilSaturation_` (44 431 × 8 B) and the empty vectors. Decoding shows
length-prefixed **ASCII well/group names**: the full run has `D-2H`, `D-1H` where the restarted run has
`D2-DUMMY`, `MANI-F`. That is `guideRate_` — accumulated, name-keyed guide-rate history whose population
and ordering legitimately differ when a run is continued rather than run from step 0.

It is inert for the sweep *by construction*: `replayStep` calls `assembleWellEqGivenControls`, which
assembles from the restored converged controls **without** re-deriving the control/group/network/
guide-rate state — which is why the Jacobian is bitwise identical despite the difference.

Two consequences:
1. **The equivalence gate must compare gradient files, never snapshot bytes** — besides the `shared_ptr`
   address issue, this measured `guideRate_` difference would fail a byte comparison spuriously.
2. **Risk flagged for the recompute, not the sweep.** A forward recompute *does* use guide rates to derive
   group targets. Here the solution still reproduced bitwise, but a deck whose group targets are
   guide-rate-binding could diverge — exactly what the recorded-control metadata below is for, with
   "record per-substep guide rates too" as the documented escalation.

Remaining gates unchanged: fixed-step only (adaptive stepping would pick different substeps *and* the tape
lacks the controller state), `--enable-tuning=false`, and a hard divergence guard (L2.3).

### L2 — Solver hints for the recompute. Makes L1 affordable.

The recompute is not a blind forward run: we already know a great deal about the trajectory it
must follow, and some of it is already in the archive. Five hints, in order of value.

1. **A cheap low-precision solution as the Newton initial guess.** Store `float32` of the
   solution at *every* substep (31 B/cell) alongside sparse full checkpoints. Newton then starts
   ~7 digits from the answer and should converge in 1–2 iterations instead of the recorded 5–15,
   so the "r extra forward sweeps" in L1's table collapse to a fraction of a sweep. Cost is
   31 B/cell against today's 202 B/cell — ~15 % — and it compresses too. This is the single
   change that makes checkpointing cheap rather than merely small.
   *Note the distinction:* the hint is only a starting point; the solve still converges to full
   tolerance, so the resulting state is accurate, not `float32`-accurate. (Storing `float32` and
   skipping the solve entirely is a different, lossy option — 2× only, and it perturbs the
   Jacobian at ~1e-7. Not worth it next to L1.)
2. **The recorded substep sequence.** `AdjointSubstepMeta` already carries `startTime` and `dt`
   per substep. Forcing the recompute onto exactly those steps is both a convergence aid (no
   time-step search) and a correctness requirement.
3. **The recorded Newton iteration count** — already stored as
   `AdjointSubstepMeta::newtonIterations`. Use it as the **divergence guard** L1 needs: if a
   recomputed substep needs materially more iterations than recorded, the trajectory has
   diverged and the sweep must abort rather than silently continue with a different state.
   Cheap, and it converts the main silent-failure mode into a loud one.
4. **The recorded well controls.** A recompute can pick a different control-switching path than
   the forward run did — this is the same hazard `replayStep` already documents. The mechanism
   to avoid it exists: `assembleWellEqGivenControls` assembles from the restored converged
   controls instead of re-deriving them, precisely because re-deriving does not reproduce the
   forward's values. A segment recompute needs the same treatment.
5. **Preconditioner / AMG hierarchy reuse.** The sparsity is constant across substeps and the
   sweep already reuses the hierarchy (see `HANDOVER_optimization.md`); the same reuse applies
   across a recomputed segment.

**The catch, stated plainly:** hints 1 and 4 change the Newton iterate path, so they forfeit
*bitwise* reproduction even though they preserve accuracy. The converged state then differs by
O(nonlinear tolerance) — with the strict recipe's `--tolerance-cnv=1e-9` that is far below the
~1e-6 serial-vs-parallel trajectory difference already accepted on Norne, so it is defensible.
But it must be *demonstrated*, not assumed: recompute a segment with and without the hint and
compare gradients, the same way `adjoint-testing-strict-tolerance` demands strict tolerances
before believing an FD comparison. Hints 2, 3 and 5 are free of this concern.

### L3 — Drop tracer state (deferred Task 2)
Safe and proven inert (§2), worth 76 % uncompressed but only ~5 % compressed on a 24-step deck.
Its compressed cost grows 87 KB → 591 KB over 24 substeps and is still rising, so re-measure on
a long deck. Cheap to do once someone is already touching the serialization path.

### L4 — Cross-timestep compression
1.3–1.7× (§0). Needs either chunks spanning substeps or explicit delta encoding, both on the
storage↔replay-reinit path. **Note it is partly redundant with L1**: checkpointing removes most
of the consecutive snapshots whose similarity L4 exploits. Do not do both.

### L5 — Codec swap to zstd
~1.1× (§0), at the cost of an H5Z-ZSTD plugin dependency for every reader. Not worth it alone;
reconsider only if bundled with L4, which needs the long window anyway.

### Combined projection, 1 M cells × 2000 substeps
122 GB today → **1.3 GB** with L1 at r=3 → and with L2.1 that 3-sweep recompute cost drops
towards one sweep. That is the target configuration for this section, and it is a separate
project, not an increment.
