# opm-adjoint

Adjoint-based gradients for the OPM flow simulator, developed as a
downstream module against **unmodified opm-simulators master** (the
opm-flowgeomechanics extension pattern): the recording hooks live in a
derived `AdjointFlowProblem` (virtual `beginEpisode`/`endTimeStep`), the
well-state normalization in a derived `AdjointWellModel`, and the
backward replay driver only uses public simulator APIs.

Single binary `flow_adjoint`:

    # forward run with per-substep recording
    flow_adjoint CASE.DATA --adjoint-save=true --adjoint-save-system=true \
        --enable-storage-cache=false --threads-per-process=1
    # backward replay + verification against the stored systems
    flow_adjoint CASE.DATA --adjoint-mode=replay \
        --enable-storage-cache=false --threads-per-process=1

## Contents

| Path | What it is |
|---|---|
| `examples/flow_adjoint.cpp` | The `flow_adjoint` binary: blackoil TPFA Flow plus the adjoint problem, parameters and run modes |
| `examples/adjoint_system_solve.cpp` | Solves a dumped adjoint system (`--adjoint-dump-system`) outside the simulator |
| `opm/adjoint/` | The module itself (header-only apart from `AdjointParameters.cpp`), see below |
| `tests/` | Unit tests, script-driven ctests, FD reference gradients (`tests/references/`) |
| `doc/tutorials/` | Small decks used by the tests and examples |

Main parts of `opm/adjoint/`:

| Files | Role |
|---|---|
| `AdjointFlowProblem`, `AdjointWellModel` | Hooks into the forward run |
| `AdjointRecorder`, `AdjointStorage`, `AdjointMeta` | What is recorded per substep, and the archive (directory or HDF5) |
| `AdjointReplay` | Re-linearizes recorded substeps backwards and checks them against the stored systems |
| `AdjointSolver`, `AdjointLinearSolver`, `*Transpose*` | Backward sweep: transposed solves and gradient accumulation |
| `AdjointObjective*` | Objective functions (`pressure-average`, `bhp`, `rate`, `match`, `matchref`, `matchsum`) |
| `AdjointEndpointGradients`, `AdjointResidualEvaluator` | End-point scaling gradients and the general (stencil-local) gradient path |
| `AdjointDeckWarnings` | End-of-run report of deck features the adjoint handles only approximately |

## Building

Build opm-common, opm-grid and opm-simulators (plain `master`) first, each in its own
`build/` folder, with the repos side by side; then this module the same way. On macOS
with DUNE from Homebrew:

```bash
OPM=~/OPM                          # directory holding the sibling repos
mkdir -p $OPM/opm-adjoint/build && cd $OPM/opm-adjoint/build
cmake -DUSE_GPU_BRIDGE=0 -DCMAKE_OSX_DEPLOYMENT_TARGET=26.0 \
      -DWITH_NDEBUG=1 -DCMAKE_BUILD_TYPE=Release \
      -Dopm-common_DIR=$OPM/opm-common/build \
      -Dopm-grid_DIR=$OPM/opm-grid/build \
      -Dopm-simulators_DIR=$OPM/opm-simulators/build \
      ..
make -j12
```

This gives `build/flow_adjoint`. Use the same `WITH_NDEBUG`, build type and deployment
target as for the other modules; an `NDEBUG` mismatch is an ABI break that crashes at
run time. The `-D<module>_DIR` flags are needed, since the module does not find
opm-grid and opm-simulators on its own. HDF5 is optional (without it the archive is a
directory). More detail in `STATUS.md` §3.

## OPM Flow interface

`flow_adjoint` is OPM Flow with its own TypeTag (`FlowAdjointProblem`, inheriting
`FlowProblemTPFA`), so it reads the same decks and takes all normal Flow options.
Without `--adjoint-mode` it is a plain forward run.

- **Forward run:** `AdjointFlowProblem` overrides `beginEpisode`/`endTimeStep` and, with
  `--adjoint-save=true`, stores each accepted substep (state, well/group state, and
  optionally the residual and Jacobian) in an archive.
- **Backward run:** `--adjoint-mode=replay|objective|gradient` reloads the archive and
  walks the substeps backwards through Flow's own linearizer and well model, so the
  adjoint uses exactly the forward discretization. `--adjoint-mode=onestop` does the
  forward and backward run in one process.
- **Output:** gradients are written next to the normal Flow output as
  `CASE.ADJOINT_GRADIENTS_{PV,TRANS,PERM,WELLCTRL}.txt`, plus `.csv` files and a
  `.json` summary.

All adjoint options start with `--adjoint-`; `flow_adjoint --help` lists them, and
`--adjoint-config=<file>.json` sets them from a file. The opm-simulators functions the
module relies on are listed in `adjoint_code_guide.md` §10.

## Tests

```bash
cd build && ctest
```

- **Unit tests:** `AdjointStorage` (archive round-trip), `transposeMatrix` and
  `systemTranspose` (transposed matrices and solves), and `adjoint_parallel_transpose`
  (parallel transpose on 2–4 ranks; only built with MPI).
- **Script tests** (`tests/run-adjoint-*.sh`): replay is bitwise, and adjoint gradients
  agree with finite differences across parameter families, objectives, solvers and
  run modes. Tests on the decks in `doc/tutorials/` always run; tests on SPE1/SPE9 run
  when opm-tests is found next to this repository.

Step-by-step test levels and how to read failures: `STATUS.md` §4d.

## Tutorials

There are no step-by-step tutorials yet. `doc/tutorials/` holds small example decks
(see its README), and `STATUS.md` §4 walks through a full example. A quick gradient on
the 3-cell deck, from the repository root:

```bash
D=doc/tutorials/MODEL_1D_DEBUG/MODEL_1D_DEBUG.DATA
# forward run, recording the archive
build/flow_adjoint $D --output-dir=/tmp/m1d \
    --enable-storage-cache=false --enable-adaptive-time-stepping=false --adjoint-save=true
# backward sweep: dJ/d(pore-volume multiplier) per cell
build/flow_adjoint $D --output-dir=/tmp/m1d \
    --enable-storage-cache=false --enable-adaptive-time-stepping=false \
    --adjoint-mode=gradient --adjoint-objective=pressure-average
cat /tmp/m1d/MODEL_1D_DEBUG.ADJOINT_GRADIENTS_PV.txt
```

## Documentation

Documentation (in this repository):

- `STATUS.md` — status, build/run quickstart, test levels, objectives and derivatives
- `adjoint_code_guide.md` — code-reading guide, design summary, background and literature
- `tests/FD_ACCURACY.md` — adjoint-vs-FD accuracy over the FD step
- `doc/tutorials/` — adjoint test decks used by the ctests

## Contributors

- Halvor M. Nilsen (SINTEF Digital) — author and maintainer;
  upstream repository <https://github.com/hnil/opm-adjoint>
- Trine Mykkeltvedt — build fixes against current OPM master, documentation cleanup
