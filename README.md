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

Documentation (in this repository):

- `STATUS.md` — status, build/run quickstart, test levels, objectives and derivatives
- `adjoint_code_guide.md` — code-reading guide and design summary
- `adjoint_plan.md` — design rationale, literature, PR sequencing
- `tests/FD_ACCURACY.md` — adjoint-vs-FD accuracy over the FD step
- `doc/tutorials/` — adjoint test decks used by the ctests
