# opm-adjoint — code assessment (fresh review)

_Date: 2026-06-26. Reviewed at commit `7c40447` (per-mode well-control gradients).
Method: full re-read of the module headers (`AdjointObjective`, `AdjointLinearSolver`,
`AdjointRecorder`, `AdjointParallel`, `AdjointSolver`, `AdjointResidualEvaluator`,
`AdjointReplay`, `transposeMatrix`, parameters/IO skimmed), cross-checked against the
opm-simulators call sites. Suite re-run at review time: **27/27 green**._

## 1. Overall state

The module is in good shape: ~5.1k lines across 17 headers, mission accomplished on
Milestones A/B/C plus the general/AD gradient mechanism. What sets it apart is the
**verification density** — nearly every claim is backed by a bitwise or FD test:

| pillar | evidence |
|---|---|
| replay = forward assembly | bitwise on m1d/SPE1/SPE9/SPE1_GRPCTRL; Norne ~90 % bitwise end-to-end |
| standalone evaluator | residual **and** Jacobian bitwise vs `linearizeDomain` (m1d, SPE1, SPE9) |
| separate adjoint IQ array | Jacobian bitwise (reuses forward physics via public `update()`) |
| parameter gradients | analytic vs general (FD) vs general-ad (machine precision); random directional FD; per-family FD tests; golden regression set |
| solvers | umfpack/ilu0/cpr/cprt agreement; MPI np=2..4 vs serial; exact transposed operator A/B-proven |

Architecture is clean: downstream module, public-API-only against near-stock
opm-simulators (4-commit `adjoint-hooks` branch), every mechanism behind a flag with the
proven default. No confirmed wrong-result bug was found in the tested configurations. The
findings below are **latent defects, robustness gaps, and improvement opportunities**,
ranked.

> **Status update (same day):** the findings below have been acted on —
> F1 `27ee860`, F3+F5 `f5c45e0`, F6 `fccc75c`/`b0432b6`, F2 `faed6f9` (deprecation
> note + BYMODE-sum consistency check; the multi-mode FD test remains open),
> F4 `559aadc` (`--adjoint-fd-step`), F7's warning promotion `c616f50`
> (fires on SPE1's defaulted CTF, silent on m1d's explicit CTF), and the P1
> improvement `80abce6` (stencil-local endpoint gradients,
> `--adjoint-endpoint-method=stencil`: FD-verified, 4.9e-11 vs whole on SPE1,
> endpoint phase 14.58 s → 0.26 s = 56×; default stays `whole` per policy).
> Still open: multi-mode WELLCTRL FD test, P2 face-targeted evaluations,
> UMFPACK factorization reuse, FD/AD struct unification, and the F7 catalog
> items (WGRUPCON Stage-2, endpoint well-row exact term, upstream endpoint AD).

## 2. Findings — possible errors (ranked by risk)

### F1. `TransposedParallelOperator::scatterTransposed_` can iterate past `end()` (latent UB)
`AdjointLinearSolver.hpp:155`: `for (auto row = forward_.begin(); row.index() < interiorSize_; ++row)`
— if a rank has **no overlap rows** (`interiorSize_ == numRows`), the loop increments to
`end()` and then calls `row.index()` on it: undefined behavior. Masked in practice because
every np≥2 rank normally has ghost rows, but a degenerate partition (tiny domain, many
ranks) can hit it. **Fix: `row != forward_.end() && row.index() < interiorSize_`.** Same
file: `computeInteriorSize_` compares `attribute() == 1` (magic constant for
`OwnerOverlapCopyAttributeSet::owner`) and assumes owner-first contiguous (ghost-last)
ordering — true for flow, but use the enum and assert contiguity.

### F2. Lumped `WELLCTRL` total is ill-defined under mode switching (mitigated, keep an eye)
The per-mode breakdown (`WELLCTRL_BYMODE`, commit `7c40447`) fixed the substance, but the
legacy `ADJOINT_GRADIENTS_WELLCTRL.txt` still sums derivatives w.r.t. *different targets*
when a well switches control (SPE1 PROD: BHP + ORAT into one number). Any downstream use
of the lumped file on a mode-switching deck is silently wrong. **Suggest: deprecation note
in the file header, and an FD test for the multi-mode case** (perturb the BHP target and
the ORAT target separately on a deck where both are active; compare against the BYMODE
entries). Note also that under group control the mode tag is `GRUP` — the derivative is
then w.r.t. the *effective group-derived* target, which deserves a doc sentence.

### F3. `AdjointParallel` constructor: unchecked `std::any_cast` pointer
`AdjointParallel.hpp:103-106`: `parinfo` from `std::any_cast<ParallelISTLInformation>(&...)`
is dereferenced without a null check. Fails only for grid managers that don't populate the
parallel info (not CpGrid), but the failure mode would be a segfault instead of a clear
error. One-line guard + `OPM_THROW`.

### F4. FD noise floor of the `general` (non-AD) path is fixed by a hard-coded `h = 1e-7`
`accumulatePvGeneral_` / `accumulateTransGeneral_` / `accumulatePermGeneral_` all use
`h = 1e-7` on an O(1) multiplier → roundoff floor ~2e-9 relative (visible as the 1e-8
oracle diffs on SPE1). Correct but not parameterized; a deck with extreme magnitudes could
degrade silently. Low priority since `general-ad` is exact and preferred — but either
expose `--adjoint-fd-step` or log the step with the gradient output.

### F5. Objective edge behaviors (silent, documented-by-test only)
- `observedRate_` extrapolates **flat** outside the reference summary's time range — a
  mismatched reference window silently biases the misfit; worth one warning on first
  clamp.
- `stepValue`/`loadObservations_` assume the well exists at `episodeIndex()` and the phase
  is active — throws `map::at`-style errors deep in the sweep for malformed specs;
  fine, but an upfront validation pass at construction would fail faster.
- Producer sign convention (`sign = -1`) and the three hard-coded rate units are correct
  for the tested decks; `SM3/DAY|MSCF/DAY|STB/DAY` is not exhaustive (e.g. `RM3/DAY`
  voidage keywords would throw — acceptable, message is clear).

### F6. Stale/incorrect operator feedback
- `AdjointRecorder::checkConfiguration_` still warns "more than one MPI process is
  untested; replay is serial-first" — **stale**: parallel record/replay/gradients are
  implemented and CI-tested (np=2 ctest). Misleads users into serial runs.
- `AdjointResidualEvaluator::NoPerturb` is dead code (never instantiated) — remove or use
  it as the documented default perturbation.
- Editor droppings `#tmp.txt#`, `.#tmp.txt` in the repo root; `tests/FD_ACCURACY.md` and
  `tests/fd_accuracy_sweep.py` are still untracked — commit or ignore.

### F7. Known, documented approximation gaps (not new, listed for completeness)
These are correctly warned about at runtime (`AdjointDeckWarnings`) but remain the real
accuracy limits: guide-rate/group cross-terms and voidage replay (WGRUPCON Stage-2 fix
designed, unimplemented); endpoint initial-state term under equilibration; **`dCTF/dK`
well-index perm coupling missing from `dJ/dperm`** (review doc §5D — flag fires only in
docs, not yet in `AdjointDeckWarnings` — promote it); hysteresis/DRSDT cross-terms;
report-0 startup residuals on model2-like decks.

## 3. Improvements — performance

### P1. Stencil-local endpoint FD (largest win remaining)
`accumulateEndpointGradients_` still re-linearizes the **whole domain** per (parameter,
cell): `2·P·N` full assemblies per substep. The `SeparateIQ` array + `EndpointPerturber`
partial-override (endpoints only touch relperm/mobility) enable a `cell×~7` version with
a whole-domain fallback for perforated cells — ~`N/7`× cheaper, no material-law change.
All the pieces exist; this is the natural next implementation step.

### P2. Face-targeted general evaluations
`evalCellResidualReservoir` recomputes **all** faces of a cell even when the perturbation
touches one face. In `accumulateTransGeneral_/AD_` and the perm loops this costs ~deg(≈7)×
more Evaluation-kernel flux calls than needed (SPE9 general-ad oracle: 19 s vs ~1 s
analytic). A `evalFaceResidualPair(face, perturb)` variant plus reusing the storage term
across the per-face loop would close most of the gap.

### P3. Cheap wins
- UMFPACK refactorizes from scratch every substep — the sparsity is fixed; reuse the
  symbolic factorization across the sweep.
- `checkResidualEvaluator_`/`checkJacobianEvaluator_` construct `SeparateIQ` and full
  matrix copies per substep — hoist to the run level (check-only paths, but Norne-scale
  checks would thank you).
- `applyscaleadd` allocates a temporary vector per call.
- `PermPerturb*/TransPerturb*/PvPerturb*` exist in near-identical FD and AD flavors —
  unify on a template over the scale type (the evaluator already is generic).
- Multi-slot batching (`Eval<K>`): seed K parameters per evaluation for ~K× fewer passes
  (planned Phase-4 knob; only after profiling says the passes matter).

## 4. Improvements — capability (ordered)

1. **Endpoint stencil-local FD** (P1) with the **well-row term** (`λ_w·dR_w/dθ`) for
   perforated cells — closes both the cost and the correctness gap for endpoints.
2. **Promote the perm/CTF warning** into `AdjointDeckWarnings` (F7) and add the
   **multi-mode WELLCTRL FD test** (F2).
3. **WGRUPCON Stage-2** (carry the converged effective control target in the
   `WellAssemblyState` bundle) — makes voidage replay bitwise; design done.
4. Endpoint exact AD via `Evaluation`-typed material-law scaled points — upstream change,
   parked deliberately (review doc §3F); revisit only with upstream appetite.
5. Norne hysteresis/well-event boundary bitwise + model2 report-0 startup — long tail.

## 5. Bottom line

No wrong-result defect found in tested configurations; the three code-level fixes worth
doing now are **F1 (iterator UB guard), F3 (any_cast guard), F6 (stale MPI warning +
dead code + untracked files)** — all small. The one substantive risk to users is **F2**
(legacy lumped WELLCTRL on mode-switching decks) — mitigated by BYMODE but worth the
deprecation note + test. The dominant remaining inefficiency is the whole-domain endpoint
FD (P1), and the dominant known accuracy gaps are the documented cross-terms (F7), of
which the perm/CTF well coupling is the one most likely to surprise a user doing perm
history matching.
