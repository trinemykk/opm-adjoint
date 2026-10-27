# Adjoint test and tutorial decks

Small decks written for the adjoint work. The ctests in `CMakeLists.txt` use them
directly, so the adjoint-specific tests run without a special opm-tests branch. The
SPE1/SPE9 tests still use the standard decks from opm-tests.

| Deck | Size | Used for |
|---|---|---|
| `MODEL_1D_DEBUG/MODEL_1D_DEBUG.DATA` | 3×1×1 cells | Most FD-vs-adjoint tests (PV, TRANS, PERM, end points, well controls), replay, regression references in `tests/references/` |
| `MODEL_2D_DEBUG_3/MODEL_2D_DEBUG_3.DATA` | 3×1×3 cells | Multi-layer debug case; not used by any test yet |
| `MODEL_3D_DEBUG_1/MODEL_3D_DEBUG_1.DATA` | 10×10×1 cells, 1 injector + 4 producers | Multi-well debug case; not used by any test yet |
| `SPE1_GRPCTRL/SPE1_GRPCTRL.DATA` | SPE1 | SPE1 under GCONPROD group control: group-control gradient and slim-archive tests |
| `DEEPWELL/DEEPWELL.DATA` | 3×3×20 cells | Deep producer, no other explicitly-updated quantities: isolates the frozen wellbore-head bias of StandardWell vs MSW |
| `norne/NORNE_ATW2013_ADJSHORT_NOTRACER.DATA` | Norne | Short Norne with tracers off (hysteresis kept), for archive-size and replay studies |

The `MODEL_*` decks are plain-text includes generated with MRST; each test script
copies the whole folder before perturbing e.g. `poro.txt`, so keep only input files
in these folders.

**Norne** needs the include files of the full Norne model. Copy the deck next to the
stock one and run it from there:

```bash
cp NORNE_ATW2013_ADJSHORT_NOTRACER.DATA $OPM/opm-tests/norne/
```

## Origin and licence

Copied from the `adjoint` branch of <https://github.com/hnil/opm-tests>
(`adjoint_tests/` and `norne/`, last commit `92e82454`); the 2018 simulator output
files, restart dumps and `adjoint_results.txt` references from that branch were left
out. Like the rest of opm-tests, the decks are made available under the Open Database
License (<http://opendatacommons.org/licenses/odbl/1.0/>), with
individual contents under the Database Contents License
(<http://opendatacommons.org/licenses/dbcl/1.0/>).
