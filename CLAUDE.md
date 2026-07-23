# opm-adjoint — session guide

Adjoint-based gradients for the OPM flow simulator: `dJ/dθ` of well-matching (and
other) objectives w.r.t. **porosity/PV, transmissibility, permeability**, end-point
scaling, and well controls. Separate downstream module (the `opm-flowgeomechanics`
pattern), built against a near-stock `opm-simulators`.

## Read first (in this order)

1. `adjoint_paths_summary.md` — decision basis for the current **practical field-HM
   phase**; §4 is the plan shape. Newest strategic doc.
2. `STATUS.md` — status + build/run quickstart + objective/derivative tables.
3. `adjoint_code_assessment.md` — fresh code review (findings F1–F7, perf P1–P3).
4. Auto-memory `adjoint-project-state` — full decision/commit history.

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
opm-adjoint     master        # the module (HEAD ab5163d)
opm-simulators  adjoint-hooks  # master + 4 small accessor commits
opm-tests       adjoint        # adjoint_tests/MODEL_*_DEBUG, SPE1_GRPCTRL
```
`opm-common` / `opm-grid` stay on their normal branches.

## Build

```bash
cd /Users/hnil/Documents/OPM/opm_clean/builds/release
cmake --build . --target flow_adjoint -j8
```
Release tree: Ninja, `CMAKE_BUILD_TYPE=Release`, parallel HDF5 + UMFPACK. Produces
`builds/release/opm-adjoint/flow_adjoint` (single binary; mode via `--adjoint-mode`).

> **Gotcha:** downstream executables need `use_additional_optimization(TARGET ...)` —
> OPM builds Release with `-UNDEBUG`; an NDEBUG mismatch is an ABI break that crashes in
> the material law during equilibration. Already handled in the module CMake.

## Run (see STATUS.md §4 for full recipes)

```bash
FLOW=/Users/hnil/Documents/OPM/opm_clean/builds/release/opm-adjoint/flow_adjoint
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
cd /Users/hnil/Documents/OPM/opm_clean/builds/release/opm-adjoint
ctest              # 28/28, ~30 s
ctest -R adjoint_fd   # just the FD-vs-adjoint gradient checks
```

## Gotchas (hard-won)

- **Gradient/FD runs require `--enable-adaptive-time-stepping=false`** — else a parameter
  perturbation changes the substep sequence and swamps any FD comparison.
- **`matchref` reference summaries must live OUTSIDE `--output-dir`** — gradient/objective/
  replay runs truncate summary files in their output dir at startup.
- **Multi-perf wells** need `--matrix-add-well-contributions=true` (now injected by default
  in `flow_adjoint main()`, with `--linear-solver=ilu0`).
- **Solver choice:** serial fastest = `cprt`; parallel = `cpr`/`ilu0` (cprt falsely
  converges in parallel); umfpack single-rank. Default reduction `1e-14`.
- **Test scripts are bash, not zsh** — the one-liners rely on word-splitting; run with
  `bash`. macOS `/bin/bash` is 3.2 (no `${VAR^^}`); no `timeout` (use `gtimeout`).
- v1 forces `--enable-storage-cache=false` (residual = pure `R(x_k, x_{k-1})`).
