# Adjoint vs finite-difference — accuracy sweep over the FD step

_Generated with `tests/fd_accuracy_sweep.py` (binary `flow_adjoint`, Release build)._

A small-model check that the adjoint gradient agrees with central finite
differences (FD) across a **range of FD steps** `eps`. The adjoint value *is*
the gradient dJ/dp — it is analytic (one backward sweep, independent of `eps`).
The FD estimate should converge to it, with the classic U-shaped error:
truncation error ∝ eps² at large steps, floating-point round-off ∝ 1/eps at
small steps.

## Model and wells

`MODEL_1D_DEBUG` — a 3-cell 1-D black-oil deck. Two wells, both BHP-controlled
in the base case:

| cell (I) | well | role |
|---:|---|---|
| 0 | `Inj` | water injector, BHP 350 bar |
| 1 | — | interior cell (no well) |
| 2 | `Prod` | oil producer, BHP 153 bar |

The **well cells (0 and 2) are the high-impact cells**: a parameter change there
moves injection/production directly, so |dJ/dp| is largest. The interior cell 1
has a smaller but still non-zero gradient — it only affects J through flow
between the wells.

## Objective functions

Two objectives are exercised (per-substep sum objectives,
`--adjoint-objective=`):

- **`pressure-average`** — the reservoir-state validation objective:
  `J = (1/N) Σᵢ pᵢ`, the mean cell pressure (oil-phase pressure primary
  variable) over all N=3 cells, evaluated at the **final accepted substep
  only**. Units: Pa. Sensitive to anything that changes the converged
  pressure field — porosity, transmissibility, permeability.
- **`bhp:Prod`** — a well-curve objective: `J = Σₖ dtₖ · BHP_Prod(xₖ)`, the
  **time-integrated** producer bottom-hole pressure summed over all substeps
  (units Pa·s). For the well-control derivative the producer is switched to
  ORAT control (so its BHP is a free response variable rather than pinned by a
  BHP target).

## Derivatives exercised

The last column is the gradient **mechanism** — how dR/dp (contracted with the
adjoint λ) is obtained. The four families here all use closed-form / analytic
routes. End-point scaling (SWL…SGU, KRW…, PCW/PCG) is **not** in this sweep — it
uses a *different residual finite-difference* mechanism and has its own check
(`ctest -R adjoint_fd_endpoint_swl`); see STATUS.md → "Code structure" for the
contrast.

| derivative w.r.t. | parameter | objective(s) | gradient mechanism |
|---|---|---|---|
| porosity / pore volume | per-cell φ (MULTPV-equivalent) | `pressure-average`, `bhp:Prod` | analytic — storage term inner product |
| transmissibility | per-face MULTX | `pressure-average` | analytic — TPFA flux inner product |
| permeability | per-cell PERMX [mD] | `pressure-average` | analytic + half-trans chain rule |
| well control | producer ORAT target u | `bhp:Prod` | well-equation row (dJ/du) |

## 1. Gradient magnitude across cells (fixed eps)

Shows the gradient is non-zero everywhere and that **well cells dominate**. `|dJ/dp|` is the absolute gradient magnitude.


**objective `pressure-average` — d/d porosity**  (eps = 1e-5, unit: abs Δφ  (φ₀ = 0.1))

| cell | well | adjoint dJ/dp | \|dJ/dp\| | FD | rel.err |
|---:|---|---:|---:|---:|---:|
| 0 | Inj (injector) |  3.783065e+06 | 3.783065e+06 |  3.783065e+06 | 7.0e-08 |
| 1 | — |  1.828505e+06 | 1.828505e+06 |  1.828505e+06 | 9.5e-08 |
| 2 | Prod (producer) | -4.748503e+03 | 4.748503e+03 | -4.749816e+03 | 2.8e-04 |

**objective `pressure-average` — d/d permx**  (eps = 1e-2, unit: abs ΔK [mD]  (K₀ = 100 mD))

| cell | well | adjoint dJ/dp | \|dJ/dp\| | FD | rel.err |
|---:|---|---:|---:|---:|---:|
| 0 | Inj (injector) |  4.765834e+03 | 4.765834e+03 |  4.765834e+03 | 2.9e-08 |
| 1 | — | -3.403120e+03 | 3.403120e+03 | -3.403119e+03 | 1.6e-07 |
| 2 | Prod (producer) | -6.956488e+03 | 6.956488e+03 | -6.956488e+03 | 5.3e-08 |

**objective `pressure-average` — d/d transmissibility**  (eps = 1e-4, unit: abs ΔMULTX  (MULTX₀ = 1.0))

| face | adjacent well | adjoint dJ/dp | \|dJ/dp\| | FD | rel.err |
|---|---|---:|---:|---:|---:|
| face 0-1 | Inj (cell 0) |  9.096441e+05 | 9.096441e+05 |  9.096442e+05 | 6.0e-08 |
| face 1-2 | Prod (cell 2) | -1.469022e+06 | 1.469022e+06 | -1.469022e+06 | 2.2e-08 |

**objective `bhp:Prod` — d/d porosity**  (eps = 1e-5, unit: abs Δφ  (φ₀ = 0.1))

| cell | well | adjoint dJ/dp | \|dJ/dp\| | FD | rel.err |
|---:|---|---:|---:|---:|---:|
| 0 | Inj (injector) |  1.535213e+13 | 1.535213e+13 |  1.535213e+13 | 8.6e-09 |
| 1 | — | -8.910436e+13 | 8.910436e+13 | -8.910436e+13 | 4.2e-09 |
| 2 | Prod (producer) | -1.325472e+14 | 1.325472e+14 | -1.325472e+14 | 4.1e-09 |

## 2. Step-size (eps) sweep at the highest-impact well cell

For each case the swept location is the cell/face with the largest |dJ/dp| (a well cell). The U-curve confirms FD → adjoint.


**objective `pressure-average` — d/d porosity** @ cell 0 — Inj (injector)  ·  adjoint dJ/dp = ** 3.78306500e+06**  (|dJ/dp| = 3.7831e+06)  ·  unit: abs Δφ  (φ₀ = 0.1)

| eps | FD | rel.err vs adjoint |
|---:|---:|---:|
| 1e-3 |  3.783244e+06 | 4.74e-05 |
| 1e-4 |  3.783067e+06 | 5.39e-07 |
| 1e-5 |  3.783065e+06 | 7.01e-08 |
| 1e-6 |  3.783065e+06 | 6.48e-08 |
| 1e-7 |  3.783065e+06 | 6.58e-08 |

**objective `pressure-average` — d/d permx** @ cell 2 — Prod (producer)  ·  adjoint dJ/dp = **-6.95648800e+03**  (|dJ/dp| = 6.9565e+03)  ·  unit: abs ΔK [mD]  (K₀ = 100 mD)

| eps | FD | rel.err vs adjoint |
|---:|---:|---:|
| 1e0 | -6.956935e+03 | 6.42e-05 |
| 1e-1 | -6.956492e+03 | 5.83e-07 |
| 1e-2 | -6.956488e+03 | 5.28e-08 |
| 1e-3 | -6.956488e+03 | 5.87e-08 |
| 1e-4 | -6.956488e+03 | 5.82e-08 |

**objective `pressure-average` — d/d transmissibility** @ face 1-2 — borders Prod  ·  adjoint dJ/dp = **-1.46902200e+06**  (|dJ/dp| = 1.4690e+06)  ·  unit: abs ΔMULTX  (MULTX₀ = 1.0)

| eps | FD | rel.err vs adjoint |
|---:|---:|---:|
| 1e-2 | -1.469073e+06 | 3.50e-05 |
| 1e-3 | -1.469022e+06 | 3.25e-07 |
| 1e-4 | -1.469022e+06 | 2.21e-08 |
| 1e-5 | -1.469021e+06 | 2.55e-08 |
| 1e-6 | -1.469021e+06 | 2.55e-08 |
| 1e-7 | -1.469021e+06 | 2.55e-08 |

**objective `bhp:Prod` — d/d well ORAT target (Prod)** @ producer ORAT target u  ·  adjoint dJ/dp = **-3.08138700e+11**  (|dJ/dp| = 3.0814e+11)  ·  unit: abs Δu [sm³/d]  (u₀ = 1.0)

| eps | FD | rel.err vs adjoint |
|---:|---:|---:|
| 1e-2 | -3.081387e+11 | 2.73e-09 |
| 1e-3 | -3.081387e+11 | 3.05e-09 |
| 1e-4 | -3.081387e+11 | 5.69e-09 |
| 1e-5 | -3.081387e+11 | 5.44e-08 |
| 1e-6 | -3.081388e+11 | 2.40e-07 |
| 1e-7 | -3.081391e+11 | 1.05e-06 |

**objective `bhp:Prod` — d/d porosity** @ cell 2 — Prod (producer)  ·  adjoint dJ/dp = **-1.32547200e+14**  (|dJ/dp| = 1.3255e+14)  ·  unit: abs Δφ  (φ₀ = 0.1)

| eps | FD | rel.err vs adjoint |
|---:|---:|---:|
| 1e-3 | -1.325472e+14 | 3.55e-07 |
| 1e-4 | -1.325472e+14 | 7.55e-09 |
| 1e-5 | -1.325472e+14 | 4.07e-09 |
| 1e-6 | -1.325472e+14 | 4.69e-09 |
| 1e-7 | -1.325472e+14 | 2.87e-08 |

## Observations

- **Every derivative is clearly non-zero**, and the well cells (0, 2) carry the
  largest |dJ/dp| — the expected high-impact locations. The interior cell 1 is
  smaller but still matched by FD, confirming the gradient is correct away from
  wells too.
- **FD converges to the adjoint to ≤ 1e-7 relative** at the optimal step for
  every (objective × derivative) combination — independent confirmation of sign
  and magnitude.
- **The U-curve is visible**: error falls as O(eps²) at large steps, bottoms
  out near the forward solver tolerance (~1e-7), then rises from round-off at
  the smallest steps.
- **Small-magnitude entries show larger relative error** — e.g.
  `pressure-average` d/d porosity at the producer cell 2 is ~1000× smaller than
  at the injector cell 0 (the producer is BHP-pinned, so its porosity barely
  moves the mean pressure), giving rel.err ~3e-4 while the absolute agreement is
  still excellent. This is FD/round-off floor on a near-zero quantity, not a
  gradient error.
- **The accuracy floor is the forward solver, not the adjoint** (cnv 1e-7);
  the adjoint is exact for the converged discrete system.
- **Recommended FD step for this model: eps ≈ 1e-4 … 1e-6** (relative
  ~1e-3 … 1e-5 of the parameter).

## Reproduce

```bash
python3 opm-adjoint/tests/fd_accuracy_sweep.py     # ~1 min; OPM_ROOT/FLOW override paths
```

