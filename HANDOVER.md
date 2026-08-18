# Handover — adjoint field-scale correctness & validation

**Date:** 2026-07-24. **Purpose:** start a fresh session on the *correctness /
field-validation* thread of opm-adjoint. (Performance/optimization work is a separate,
orthogonal thread — see `HANDOVER_optimization.md`.)

## Read first (in order)
1. `CLAUDE.md` — session guide, build/test/run, branches, gotchas.
2. `STATUS.md` — status + objective/derivative tables + the memory/performance section.
3. `adjoint_storage_sizing.md` — measured memory/performance (the "adjoint runs at ~forward
   memory with cprt" story).
4. Auto-memory `adjoint-project-state` — full decision/commit history (this is the deepest record).

## Where things stand (one paragraph)
The reservoir-parameter adjoint (**PORO/PV, transmissibility, permeability**) is **validated
at field scale**: SPE9 PORO history-match demo FD-matches to 0.03–1.5% (`adjoint_poro_field_demo_spe9`),
and on a real 44 431-cell Norne the **reservoir** gradient (pressure-average objective) matches
FD to ~2–4% (near the replay floor). The **well-objective** gradients (BHP, rate) are **not yet
correct on field decks**: a consistent ~12% bias, root-caused to the frozen wellbore gravity head.

## Validated vs open (correctness)

| item | state |
|---|---|
| PORO/PV, TRANS, PERM gradients | ✅ FD-verified, incl. 44k-cell Norne (pressure-avg 2–4%) |
| SPE9 PORO HM demo (A6) | ✅ `adjoint_poro_field_demo_spe9`, 0.03–1.5% |
| well-objective (BHP/rate) at field scale | ❌ ~12% bias — see #24 below |

## The active investigation: well-objective gradient bias (~12% on Norne)
Root-caused (task **#24**, resolved = *warn, not fix*): the **wellbore gravity head**
`perf_pressure_diffs_[perf] = dz·ρ·g` is a frozen `Scalar` computed in
`StandardWell::calculateExplicitQuantities` (once per timestep), enters the drawdown
(`well_pressure = bhp + pressure_diff(perf)`) → `cq = -Tw·mob·drawdown`, so its state
derivative is absent from the well Jacobian the adjoint reuses. Negligible for thin/single-perf
wells (SPE1/SPE9 exact), up to ~10% on BHP/rate objectives for thick multi-perf wells.
**Not practically fixable** (density couples to all near-well reservoir state; legacy of the
reference simulator) → resolution is the strengthened warning in `AdjointDeckWarnings.hpp`.

How it was isolated (skeptical, controlled — I was wrong twice, controls caught it):
- **NOT eps/phase-kinks** (bhp identical at eps 1e-6 and 1e-5);
- **NOT DRSDT/VAPPARS** (removing them left the gap unchanged);
- **NOT the core PV gradient** (pressure-average is clean);
- **NOT replay drift** (residual ~1.5e-3 → only ~0.15%, ~80× too small);
- **CAUTION**: `--enable-gravity=false` is a *global* toggle (reservoir + equil + well head),
  so that test is confounded — the *code mechanism* is the evidence, not that test.

## Related open correctness tasks (see the task list / memory for full text)
- **#18** — general cross-step explicit-quantity term in the adjoint Bdiag (DRSDT/DRVDT/hysteresis/
  guide-rates). The well gravity head (#24) is a *well* instance of this class.
- **#22** — make replay **bitwise-exact** on well-event/group-control decks (a nonzero replay
  residual = incomplete storage or wrong reinit; a bug, not an approximation). Full Norne
  drifts to ~1.0 at hysteresis/well-event boundaries; cleaned Norne to ~1.5e-3.
- **#16** — MSW well-adjoint support (currently `bhp` on MSW throws "StandardWells only";
  reservoir gradients on MSW already work — see `adjoint_msw_smoke_spe1`).
- **#25** — `--use-multisegment-well=false|true|always` STW→MSW converter (the fix-free route
  for the gravity head; needs #16). Building blocks + injection point are in the task text.
- **#20** — slim + `--adjoint-mode=objective` crashes (`Schedule::getWell`); use gradient-mode.
- **#23** — full-archive record crashes mid-run on Norne; use `--adjoint-slim=true`.

## Test decks & harness (for reproducing the field FD checks)
- **Norne test decks** (in `opm-tests/norne`, **uncommitted** — commit to the adjoint branch if
  keeping): `NORNE_ATW2013_ADJSHORT.DATA` (stock SHORT with its 1-step `END` removed → ~24–35
  substeps, full 44 k cells) and `NORNE_ATW2013_ADJSHORT_CLEAN.DATA` (tracers + hysteresis
  commented `-- [adjoint-clean]`; behaviourally identical: pressure 0.003%, rates <0.25%; replay
  drift 4.5× lower). **Gotcha:** the stock SHORT schedule has `END` at line 207 of
  `INCLUDE/BC0407_HIST01122006_SHORT.SCH` capping it to 1 step — must remove to run multi-step.
- **Directional-FD method** (per-cell FD infeasible at 44 k): random direction `d`, check `g·d`
  vs central FD `(J(θ+εd)−J(θ−εd))/2ε`. Handle ACTNUM via the **cartesian-keyed CSV**
  (`*.ADJOINT_GRADIENTS_PORO.csv` gives cartesian_index → dJ/dpvmult); perturb via injected
  MULTPV over the full grid. Use **slim record + gradient-mode + read J from the JSON sidecar**
  (`objective.value`) — full archive and objective-mode both crash on Norne. Keep **default**
  relaxed tolerances (non-adaptive can't chop → needs the fallback). The scratch harness
  `norne_fd.py` implemented all this but lives in the session scratchpad (not persisted) —
  recreate from this description or from the memory record.

## Build / test / run
```bash
cd /Users/hnil/OPM_CLEAN/opm_adjoint/builds/release
cmake --build . --target flow_adjoint -j8
cd opm-adjoint && ctest              # fast suite; `-LE field` excludes the ~90s SPE9 demo
```
FD/gradient runs need `--enable-adaptive-time-stepping=false --enable-storage-cache=false
--threads-per-process=1`. Forward solver `cpr_trueimpes` (~12× faster than umfpack); backward
`--adjoint-linear-solver=cprt` for field (umfpack OOMs). Branches: opm-adjoint `master`,
opm-simulators `adjoint-hooks`, opm-tests `adjoint`.

## Gotchas
- **Bash test scripts are bash, not zsh** — inline `$VAR` in the shell tool does NOT word-split;
  put multi-flag command strings in a bash script file (cost real time repeatedly).
- Norne wells run on **RESV**; BHP is the free response. Objective focus = **well rates ± BHP**.
- Replay must be bitwise-exact; a nonzero residual is a storage/reinit bug (#22), not a tolerance.

## Suggested immediate next steps
1. **#16 + #25** together — MSW well-adjoint + the STW→MSW converter — the real path to correct
   well-objective gradients on gravity-dominated wells (and independently confirms #24).
2. Or **#22** — chase the ~1.5e-3 cleaned-Norne replay residual to bitwise (well-event/group-control
   state storage/reinit).
3. Or make the MSW well-objective rejection exit cleanly (it currently aborts, `AdjointSolver.hpp:429`).
