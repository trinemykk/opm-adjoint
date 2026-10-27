# opm-adjoint
Adjoint-based gradients for the OPM Flow simulator: gradients of well-matching and other objectives with respect to
porosity/pore volume, transmissibility, permeability, saturation end points and well controls.

## Contains
The opm-adjoint module includes:
- Interface to OPM Flow simulator (`flow_adjoint`), built against unmodified opm-simulators
- Recording of forward runs and exact backward replay of every time step
- Adjoint solver with direct and iterative transposed linear solvers, serial and MPI
- Objective functions for pressure, bottom-hole pressure, well rates and history matching against a reference run
- Standalone solver for dumped adjoint systems (`adjoint_system_solve`)
- Documentation: [STATUS.md](STATUS.md) (status, examples, test levels) and
  [adjoint_code_guide.md](adjoint_code_guide.md) (code-reading guide, design, literature)

## Build instructions
opm-adjoint depends on:

- [opm-common](https://github.com/OPM/opm-common)
- [opm-grid](https://github.com/OPM/opm-grid)
- [opm-simulators](https://github.com/OPM/opm-simulators)

Follow the [build instructions on the OPM home page](https://opm-project.org/?page_id=36).

After the prerequisites are in place, build the opm-adjoint module by running:
```shell
mkdir build/
cd build/
cmake -Dopm-common_DIR=<path>/opm-common/build \
      -Dopm-grid_DIR=<path>/opm-grid/build \
      -Dopm-simulators_DIR=<path>/opm-simulators/build ..
make
```

Use the same build type and `WITH_NDEBUG` setting as for the other OPM modules. See [STATUS.md](STATUS.md) §3 for a
complete macOS example.

## OPM Flow interface
The executable `flow_adjoint` is OPM Flow with an interface to the adjoint solver. It reads the same decks and takes
the same options as `flow`; without adjoint options it is a normal forward run. The adjoint options all start with
`--adjoint-` (see `flow_adjoint --help`):
```shell
# forward run, records an archive
flow_adjoint CASE.DATA --enable-storage-cache=false --adjoint-save=true
# backward sweep, writes the gradients
flow_adjoint CASE.DATA --enable-storage-cache=false \
             --adjoint-mode=gradient --adjoint-objective=pressure-average
```

Other modes are `replay` (verify the recorded run), `objective` (objective value only) and `onestop` (forward and
backward in one run). Gradients are written as `CASE.ADJOINT_GRADIENTS_*.txt` next to the normal Flow output.

See [tutorials section](#tutorials) for usage.

## Tutorials
Example decks are in [doc/tutorials](doc/tutorials/README.md), and [STATUS.md](STATUS.md) §4 walks through a complete
example. A first gradient on the 3-cell model, from the repository root:
```shell
D=doc/tutorials/MODEL_1D_DEBUG/MODEL_1D_DEBUG.DATA
build/flow_adjoint $D --output-dir=/tmp/m1d \
    --enable-storage-cache=false --enable-adaptive-time-stepping=false --adjoint-save=true
build/flow_adjoint $D --output-dir=/tmp/m1d \
    --enable-storage-cache=false --enable-adaptive-time-stepping=false \
    --adjoint-mode=gradient --adjoint-objective=pressure-average
cat /tmp/m1d/MODEL_1D_DEBUG.ADJOINT_GRADIENTS_PV.txt
```

## Tests
Unit tests (`test_AdjointStorage`, `test_transposeMatrix`, `test_systemTranspose`, and `test_parallelTranspose` with
MPI) and script tests (`tests/run-adjoint-*.sh`) are run with
[CTest](https://cmake.org/cmake/help/book/mastering-cmake/chapter/Testing%20With%20CMake%20and%20CTest.html):
```shell
cd build/
ctest
```
The script tests check that the replay is exact and that the adjoint gradients agree with finite differences. Tests on
the decks in [doc/tutorials](doc/tutorials/) always run; tests on the SPE1/SPE9 decks need
[opm-tests](https://github.com/OPM/opm-tests) next to this repository.

## Contributors
opm-adjoint was developed by:

- [Halvor M. Nilsen](https://github.com/hnil/) ([upstream repository](https://github.com/hnil/opm-adjoint))
