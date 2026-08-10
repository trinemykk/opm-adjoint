/*
  Copyright 2026 SINTEF Digital, Mathematics and Cybernetics.

  This file is part of the Open Porous Media project (OPM).

  OPM is free software: you can redistribute it and/or modify
  it under the terms of the GNU General Public License as published by
  the Free Software Foundation, either version 3 of the License, or
  (at your option) any later version.

  OPM is distributed in the hope that it will be useful,
  but WITHOUT ANY WARRANTY; without even the implied warranty of
  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
  GNU General Public License for more details.

  You should have received a copy of the GNU General Public License
  along with OPM.  If not, see <http://www.gnu.org/licenses/>.
*/
/*!
 * \file
 * \brief Backward replay of a recorded forward simulation.
 *
 * For every accepted substep k (walked backwards), the converged
 * linearization of the forward run is recreated from the stored snapshots
 * and compared against the stored residual/Jacobian (when present). This
 * is test level T2/T3 of the adjoint plan and the foundation the adjoint
 * solve builds on: after replayStep(k) returns, the linearizer holds the
 * converged system of forward substep k.
 *
 * v1 restrictions: serial runs, --enable-storage-cache=false (the replay
 * relies on the cache-free storage term, where the residual is a pure
 * function R(x_k, x_{k-1})).
 */
#ifndef OPM_ADJOINT_REPLAY_HPP
#define OPM_ADJOINT_REPLAY_HPP

#include <opm/common/ErrorMacros.hpp>
#include <opm/common/OpmLog/OpmLog.hpp>

#include <opm/models/discretization/common/fvbaseparameters.hh>
#include <opm/models/utils/parametersystem.hpp>
#include <opm/models/utils/propertysystem.hh>

#include <opm/adjoint/AdjointMeta.hpp>
#include <opm/adjoint/AdjointParameters.hpp>
#include <opm/adjoint/AdjointStorage.hpp>
#include <opm/adjoint/AdjointDeckWarnings.hpp>
#include <opm/adjoint/AdjointSystemIO.hpp>
#include <opm/adjoint/AdjointResidualEvaluator.hpp>
#include <opm/simulators/flow/FlowProblemParameters.hpp>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <type_traits>

#include <dune/common/fmatrix.hh>

#include <vector>

#include <fmt/format.h>

#include <memory>
#include <stdexcept>
#include <string>

namespace Opm {

template<class TypeTag>
class AdjointReplay
{
public:
    using Simulator = GetPropType<TypeTag, Properties::Simulator>;
    using SolutionVector = GetPropType<TypeTag, Properties::SolutionVector>;
    using Scalar = GetPropType<TypeTag, Properties::Scalar>;
    using LocalResidual = GetPropType<TypeTag, Properties::LocalResidual>;
    using Evaluation = GetPropType<TypeTag, Properties::Evaluation>;

    static constexpr int numEq = getPropValue<TypeTag, Properties::NumEq>();
    //! Dense block of dR_k/dx_{k-1} for one cell (block-diagonal cross term).
    using BdiagBlock = Dune::FieldMatrix<Scalar, numEq, numEq>;
    using Matrix = std::remove_cvref_t<
        decltype(std::declval<Simulator&>().model().linearizer()
                     .jacobian().istlMatrix())>;

    explicit AdjointReplay(Simulator& simulator)
        : AdjointReplay(simulator, nullptr)
    {}

    //! \brief One-stop replay: read from a shared (in-memory) archive that the
    //!        forward recorder in the same process just filled.
    AdjointReplay(Simulator& simulator,
                  std::shared_ptr<AdjointArchive> injectedArchive)
        : simulator_(simulator)
        , config_(AdjointConfig::fromParameters())
    {
        if (Parameters::Get<Parameters::EnableStorageCache>()) {
            OPM_THROW(std::runtime_error,
                      "Adjoint replay requires the cache-free storage term; "
                      "run with --enable-storage-cache=false (and record the "
                      "forward run with the same option).");
        }
        if (Parameters::Get<Parameters::EnableDriftCompensation>()) {
            // See the matching warning in AdjointRecorder::checkConfiguration_.
            // Drift compensation feeds an accumulated correction into the
            // residual, which the adjoint recursion does not model, so the
            // sweep would return quietly wrong gradients. Refuse rather than
            // approximate. Deliberately not gated on
            // EnableDriftCompensationTemp: its true default only takes effect
            // for the sequential-thermal energy module.
            OPM_THROW(std::runtime_error,
                      "Adjoint replay does not support drift compensation: the "
                      "accumulated residual correction is not part of the "
                      "adjoint recursion, so the gradients would be wrong. "
                      "Re-run the forward simulation and the sweep with "
                      "--enable-drift-compensation=false.");
        }

        std::string path = "(in-memory, one-stop)";
        if (injectedArchive) {
            archive_ = std::move(injectedArchive);
        } else {
            path = config_.path;
            if (path.empty()) {
                const auto& ioConfig = simulator_.vanguard().eclState().getIOConfig();
                path = AdjointArchive::defaultPath(ioConfig.getOutputDir(),
                                                   simulator_.vanguard().caseName());
            }
            archive_ = std::make_shared<AdjointArchive>(
                path, AdjointArchive::Mode::Read,
                simulator_.vanguard().gridView().comm());
        }
        archive_->read(meta_, AdjointGroups::meta(), "meta");

        if (meta_.schemaVersion != AdjointMeta::currentSchemaVersion) {
            OPM_THROW(std::runtime_error,
                      "Adjoint archive schema version " +
                      std::to_string(meta_.schemaVersion) + " does not match " +
                      std::to_string(AdjointMeta::currentSchemaVersion));
        }
        if (!meta_.allPayloadsPresent()) {
            // Recorded with --adjoint-checkpoint-every>1: the substeps between
            // checkpoints must be recomputed and supplied via
            // setSegmentArchive(). sourceArchive_() throws precisely where one
            // is missing, so this only has to say what the archive is.
            OpmLog::warning(fmt::format(
                "Adjoint archive is checkpointed (stride {}): {} of {} substeps "
                "carry no stored state and are recomputed during the sweep. "
                "This path is UNDER DEVELOPMENT - verified on SPE1/SPE9, serial, "
                "fixed-step only.",
                meta_.checkpointEvery,
                std::count_if(meta_.substeps.begin(), meta_.substeps.end(),
                              [](const AdjointSubstepMeta& m)
                              { return !m.hasPayload; }),
                meta_.substeps.size()));
        }
        if (meta_.storageCacheEnabled) {
            OpmLog::warning("The forward run was recorded with the storage "
                            "cache enabled; exact replay is not expected. "
                            "Re-run the forward simulation with "
                            "--enable-storage-cache=false.");
        }

        OpmLog::info(fmt::format("Adjoint replay: {} substeps in archive {}",
                                 meta_.substeps.size(), path));
    }

    //! \brief Walk backwards over all substeps, re-linearize and compare.
    //! \return EXIT_SUCCESS when every substep matched within tolerance.
    int run()
    {
        const int numSubsteps = static_cast<int>(meta_.substeps.size());
        int failed = 0;
        double worstResidual = 0.0;
        double worstJacobian = 0.0;

        for (int k = numSubsteps - 1; k >= 0; --k) {
            if (!replayStep(k, worstResidual, worstJacobian)) {
                ++failed;
            }
        }

        if (meta_.systemSaved) {
            OpmLog::info(fmt::format(
                "Adjoint replay finished: {}/{} substeps within tolerance {:.1e} "
                "(worst residual rel diff {:.3e}, worst Jacobian rel diff {:.3e})",
                numSubsteps - failed, numSubsteps, config_.replayTolerance,
                worstResidual, worstJacobian));
        } else {
            OpmLog::info("Adjoint replay finished (no stored systems to compare "
                         "against; record with --adjoint-save-system=true for "
                         "verification)");
        }
        if (config_.residualEvalCheck) {
            OpmLog::info(fmt::format(
                "Adjoint residual-evaluator check: worst rel diff vs "
                "linearizer {:.3e} over {} substeps ({})",
                worstResidualEval_, numSubsteps,
                residualEvalFailed_ ? "FAIL" : "bitwise"));
            OpmLog::info(fmt::format(
                "Adjoint Jacobian-evaluator check: worst rel diff vs "
                "linearizer {:.3e} over {} substeps ({})",
                worstJacobianEval_, numSubsteps,
                jacobianEvalFailed_ ? "FAIL" : "bitwise"));
        }
        warnAdjointDeckApproximations(simulator_);
        const bool ok = (failed == 0) && !residualEvalFailed_ &&
                        !jacobianEvalFailed_;
        return ok ? EXIT_SUCCESS : EXIT_FAILURE;
    }

    //! \brief Recreate the converged linearization of forward substep \p k.
    //!
    //! On successful return the linearizer holds the converged residual and
    //! Jacobian of substep k (the state the adjoint solve needs).
    //! \return false if a stored system exists and the comparison exceeded
    //!         the tolerance.
    bool replayStep(int k, double& worstResidual, double& worstJacobian)
    {
        const auto& meta = meta_.substeps[k];
        const auto& schedule = simulator_.vanguard().schedule();

        // Restore the state at the end of substep k-1 (the initial state for
        // k == 0). WS6a: a former "Phase A" load of snapshot k was here to
        // extract its converged solution, but that value was discarded and the
        // snapshot is fully re-loaded below (after the Bdiag capture), and this
        // k-1 load overwrites all deserialized state anyway — so the extra load
        // was dead work (one of three per substep) and is removed. Guarded by
        // the replay-residual comparison (compareWithStored_) staying at floor.
        const int prevReport = (k > 0) ? meta_.substeps[k - 1].reportStep
                                       : meta.reportStep;
        loadSnapshot_(k - 1, prevReport);

        // Report-step context, mirroring SimulatorFullyImplicit::runStep and
        // the --load-step path (episode setup after the restore, then
        // beginEpisode to rebuild the well container from the restored
        // well/group state).
        //
        // NOTE: for substeps in the middle of a report step this re-applies
        // the start-of-episode schedule processing, which the forward run
        // performed only once per report step (with the state of the last
        // substep of the previous report step). For simple decks
        // (constant controls) this is idempotent; control switching during
        // an episode is a documented v1 risk - the T2/T3 comparisons below
        // detect any resulting drift.
        simulator_.startNextEpisode(simulator_.startTime() +
                                        schedule.seconds(meta.reportStep),
                                    schedule.stepLength(meta.reportStep));
        simulator_.setEpisodeIndex(meta.reportStep);
        problem_().beginEpisode();

        // prepareStep mirror (NonlinearSystem::prepareStep).
        model_().advanceTimeLevel();
        simulator_.setTime(meta.startTime);
        simulator_.setTimeStepSize(meta.dt);
        problem_().resetIterationForNewTimestep();
        problem_().beginTimeStep();

        // Iteration-0 assembly: triggers the well model's per-step setup
        // (prepareTimeStep -> explicit quantities from state k-1) exactly
        // like the forward run's initial linearization.
        problem_().beginIteration();
        model_().linearizer().linearizeDomain();
        problem_().endIteration();
        problem_().markTimestepInitialized();
        problem_().advanceIteration();

        // Milestone B cross term: at this point solution(0) = x_{k-1} and
        // the intensive quantities at time index 0 carry derivatives with
        // respect to x_{k-1}, so the block-diagonal
        // Bdiag_k = dR_k/dx_{k-1} = -(V/dt) dStorage/dx|x_{k-1}
        // is a cell-local evaluation (fluxes and sources are purely
        // implicit in the TPFA residual).
        if (computeBdiagEnabled_) {
            captureBdiag_(meta.dt);
        }

        // Install the FULL converged state of substep k. Restoring the
        // whole simulator snapshot (not just the WGState) brings back the
        // well-model guide rates and the summary state in addition to the
        // well/group state - all of which the control-equation assembly
        // reads (getGroupProductionControl uses the group state AND the
        // guide rates). The snapshot stores solution(0)=x_k and
        // solution(1)=x_{k-1}, so the storage term R(x_k, x_{k-1}) is
        // reproduced; the Bdiag cross term was already captured above.
        loadSnapshot_(k, meta.reportStep);
        simulator_.setTime(meta.startTime);
        simulator_.setTimeStepSize(meta.dt);

        // Final linearization at the converged point. Assemble the well
        // equations from the restored converged controls WITHOUT
        // re-deriving the control/group/network/guide-rate state (which
        // does not reproduce the forward's converged values and crashes on
        // shut wells); recompute only the well primary variables.
        wellModel_().assembleWellEqAtRestoredState(meta.dt);
        model_().linearizer().linearizeDomain();

        // Optional: validate the standalone evaluator against the
        // just-assembled production linearizer - residual (Step 1) and the
        // full Jacobian (Step A, general AD assembly).
        if (config_.residualEvalCheck) {
            checkResidualEvaluator_(k);
            checkJacobianEvaluator_(k);
        }

        return compareWithStored_(k, worstResidual, worstJacobian);
    }

    const AdjointMeta& meta() const
    { return meta_; }

    //! \brief Enable capturing of the cross-term blocks during replay.
    void setComputeBdiag(bool enable)
    { computeBdiagEnabled_ = enable; }

    //! \brief Point the reader at the archive holding a recomputed segment.
    //!
    //! Substeps whose `hasPayload` is false are read from here instead of the
    //! recorded archive; pass nullptr when the segment is retired. Only the
    //! non-checkpoint substeps are affected, so a wrong or stale segment
    //! archive surfaces as a missing dataset rather than as silently wrong
    //! state.
    void setSegmentArchive(std::shared_ptr<AdjointArchive> archive)
    { segmentArchive_ = std::move(archive); }

    //! \brief Cross-term blocks dR_k/dx_{k-1} of the last replayed substep.
    const std::vector<BdiagBlock>& bdiag() const
    { return bdiag_; }

    //! \brief Full restore of snapshot \p k (k == -1 means the initial
    //!        state). Public for objective-only evaluation.
    void loadSnapshot(int k, int reportStepForWells)
    { loadSnapshot_(k, reportStepForWells); }

    Simulator& simulator()
    { return simulator_; }

    //! \brief WS6 instrumentation: total time in loadSnapshot_ (deserialize +
    //!        IQ update) and the number of such calls across the sweep. Three
    //!        loads per replayStep today; the Phase-A load is the WS6a target.
    double loadSnapshotSeconds() const { return loadSnapshotSeconds_; }
    int loadSnapshotCalls() const { return loadSnapshotCalls_; }

private:
    auto& model_()
    { return simulator_.model(); }

    auto& problem_()
    { return simulator_.problem(); }

    auto& wellModel_()
    { return simulator_.problem().wellModel(); }

    //! \brief Which archive holds substep \p k's payload.
    //!
    //! Checkpoints (and the initial state) live in the recorded archive; the
    //! substeps in between were never written and come from the segment archive
    //! the driver just filled by recomputing them. `hasPayload` in the recorded
    //! meta is the authority, so there is no ambiguity and nothing to copy
    //! between archives.
    AdjointArchive& sourceArchive_(int k)
    {
        const bool stored = (k < 0) || meta_.substeps[k].hasPayload;
        if (stored) {
            return *archive_;
        }
        if (!segmentArchive_) {
            OPM_THROW(std::runtime_error, fmt::format(
                      "Substep {} has no stored state (the archive was recorded "
                      "with --adjoint-checkpoint-every={}) and no recomputed "
                      "segment is available. Either re-record with "
                      "--adjoint-checkpoint-every=1 or run a mode that drives "
                      "the segment recompute.", k, meta_.checkpointEvery));
        }
        return *segmentArchive_;
    }

    //! \brief Full restore of snapshot \p k (k == -1 means the initial state).
    void loadSnapshot_(int k, int reportStepForWells)
    {
        const auto t0 = std::chrono::steady_clock::now();
        ++loadSnapshotCalls_;
        wellModel_().prepareDeserialize(reportStepForWells);
        // OPM_ADJOINT_WELLSTATE_DIAG=1 prints the reconstructed well state
        // next to its recorded counterpart (AdjointRecorder prints the "REC"
        // side). An archive read fails with a bare "size mismatch", so
        // comparing the two per rank and substep is the way to localize a
        // record/replay well-set divergence.
        if (std::getenv("OPM_ADJOINT_WELLSTATE_DIAG")) {
            const auto& wm = wellModel_();
            std::cerr << fmt::format(
                "[wellstate-diag] rank {} substep {} reportStep {}: "
                "localWells={} wellStateSize={}\n",
                simulator_.vanguard().grid().comm().rank(), k,
                reportStepForWells, wm.numLocalWellsEnd(),
                wm.wellState().size());
        }
        const std::string group = (k < 0) ? AdjointGroups::initial()
                                          : AdjointGroups::substep(k);
        AdjointArchive& src = sourceArchive_(k);
        if (meta_.slim) {
            // Symmetric with AdjointRecorder::writeSnapshot_: restore the
            // dynamic state; the schedule stays the deck-reconstructed one and
            // time/episode are set from meta by the caller (replayStep).
            auto& v = simulator_.vanguard();
            src.read(simulator_.model(),   group, "model");
            src.read(simulator_.problem(), group, "problem");
            src.read(v.summaryState(),     group, "summary");
            src.read(v.udqState(),         group, "udq");
            src.read(v.actionState(),      group, "action");
        } else {
            src.read(simulator_, group, "simulator");
        }
        model_().invalidateAndUpdateIntensiveQuantities(/*timeIdx=*/0);
        loadSnapshotSeconds_ += std::chrono::duration<double>(
            std::chrono::steady_clock::now() - t0).count();
    }

    void captureBdiag_(double dt)
    {
        const std::size_t numCells = model_().numGridDof();
        bdiag_.resize(numCells);
        for (std::size_t globI = 0; globI < numCells; ++globI) {
            const auto& intQuants = model_().intensiveQuantities(globI, /*timeIdx=*/0);
            Dune::FieldVector<Evaluation, numEq> storage;
            LocalResidual::template computeStorage<Evaluation>(storage, intQuants);
            const Scalar factor = -model_().dofTotalVolume(globI) / dt;
            for (int eq = 0; eq < numEq; ++eq) {
                for (int pv = 0; pv < numEq; ++pv) {
                    bdiag_[globI][eq][pv] = factor * storage[eq].derivative(pv);
                }
            }
        }
    }

    //! \brief Recompute the residual with the standalone evaluator and
    //!        compare it to the production linearizer residual (which holds
    //!        the converged system after the final linearizeDomain). A pass
    //!        means the residual-only evaluator reproduces linearize_cell +
    //!        linearize_source_terms bitwise.
    void checkResidualEvaluator_(int k)
    {
        AdjointResidualEvaluator<TypeTag> evaluator(simulator_);
        const auto& live = model_().linearizer().residual();
        const std::size_t numCells = model_().numGridDof();

        typename AdjointResidualEvaluator<TypeTag>::EqVector res(0.0);
        double maxAbs = 0.0;
        double maxRelSignificant = 0.0; // rel diff only where abs clears the floor
        for (std::size_t i = 0; i < numCells; ++i) {
            evaluator.evalCellResidual(static_cast<unsigned>(i), res);
            for (int eq = 0; eq < numEq; ++eq) {
                const double a = res[eq];
                const double b = live[i][eq];
                const double absDiff = std::abs(a - b);
                maxAbs = std::max(maxAbs, absDiff);
                // At convergence the residual is ~0, so a relative criterion
                // alone is ill-conditioned. Mirror the replay comparison: an
                // entry only matters once its absolute difference clears the
                // noise floor; then judge it on the relative tolerance.
                if (absDiff > config_.replayAbsTolerance) {
                    const double denom = std::max(std::abs(a), std::abs(b));
                    const double rel = (denom > 0.0) ? absDiff / denom : 0.0;
                    maxRelSignificant = std::max(maxRelSignificant, rel);
                }
            }
        }
        worstResidualEval_ = std::max(worstResidualEval_, maxRelSignificant);
        const bool fail = maxRelSignificant > config_.replayTolerance;
        if (fail) {
            residualEvalFailed_ = true;
        }
        OpmLog::info(fmt::format(
            "residual-eval substep {:4d}: max abs diff {:.3e}, "
            "max rel diff (above {:.0e} floor) {:.3e} vs linearizer{}",
            k, maxAbs, config_.replayAbsTolerance, maxRelSignificant,
            fail ? "  <-- EXCEEDS TOLERANCE" : ""));
    }

    //! \brief Reproduce the converged reservoir Jacobian with the standalone
    //!        evaluator (the forward "analytic derivative" dR/dstate, before
    //!        the well Schur complement) and compare it to the production
    //!        linearizer matrix. A pass means the general AD assembly
    //!        reproduces linearize_cell's Jacobian bitwise.
    void checkJacobianEvaluator_(int k)
    {
        AdjointResidualEvaluator<TypeTag> evaluator(simulator_);
        auto& liveMat = model_().linearizer().jacobian().istlMatrix();
        const auto refDump = AdjointMatrixDump::from(liveMat);

        // Scratch matrix hoisted across substeps (the sparsity is constant;
        // evalJacobian zeroes and refills all values).
        if (!checkMat_) {
            checkMat_ = std::make_unique<Matrix>(liveMat);
        }

        // (1) model-cache IQ source (Step A).
        evaluator.evalJacobian(*checkMat_);
        const auto diff = compare(refDump, AdjointMatrixDump::from(*checkMat_),
                                  1e-300, config_.replayAbsTolerance);

        // (2) separate adjoint-owned IQ array, populated via the public
        //     IntensiveQuantities::update() (reuses the forward physics) — the
        //     substrate the adjoint manipulates. Must reproduce the same
        //     Jacobian. The array object is hoisted; the per-substep rebuild
        //     (IQ updates at the restored state) is the point of the check.
        if (!sepIQ_) {
            sepIQ_ = std::make_unique<
                typename AdjointResidualEvaluator<TypeTag>::SeparateIQ>(
                simulator_);
        } else {
            sepIQ_->rebuild();
        }
        evaluator.evalJacobian(*checkMat_, *sepIQ_);
        const auto sepDiff = compare(refDump,
                                     AdjointMatrixDump::from(*checkMat_),
                                     1e-300, config_.replayAbsTolerance);

        worstJacobianEval_ = std::max({worstJacobianEval_, diff.maxRelDiff,
                                       sepDiff.maxRelDiff});
        const bool fail = !diff.withinTolerance(config_.replayTolerance) ||
                          !sepDiff.withinTolerance(config_.replayTolerance);
        if (fail) {
            jacobianEvalFailed_ = true;
        }
        OpmLog::info(fmt::format(
            "jacobian-eval substep {:4d}: model-IQ rel {:.3e} (abs {:.3e}), "
            "separate-IQ rel {:.3e} (abs {:.3e}) vs linearizer{}",
            k, diff.maxRelDiff, diff.maxAbsDiff,
            sepDiff.maxRelDiff, sepDiff.maxAbsDiff,
            fail ? "  <-- EXCEEDS TOLERANCE" : ""));
    }

    bool compareWithStored_(int k, double& worstResidual, double& worstJacobian)
    {
        const auto& meta = meta_.substeps[k];
        auto liveResidual = AdjointVectorDump::from(model_().linearizer().residual());

        // The residual/Jacobian dumps ride along with the state payload, so a
        // recomputed substep has none to compare against - report its residual
        // norm instead, exactly as an archive recorded without systems does.
        if (!meta_.systemSaved || !meta_.substeps[k].hasPayload) {
            double norm = 0.0;
            for (const double value : liveResidual.values) {
                norm = std::max(norm, std::abs(value));
            }
            OpmLog::info(fmt::format(
                "replay substep {:4d} (report step {:3d}, dt {:.4g} days): "
                "max |residual| = {:.3e}", k, meta.reportStep,
                meta.dt / 86400.0, norm));
            return true;
        }

        const std::string group = AdjointGroups::substep(k) + "/system";

        AdjointVectorDump storedResidual;
        archive_->read(storedResidual, group, "residual");
        const auto residualDiff = compare(storedResidual, liveResidual,
                                          1e-300, config_.replayAbsTolerance);

        AdjointMatrixDump storedJacobian;
        archive_->read(storedJacobian, group, "jacobian");
        auto liveJacobian =
            AdjointMatrixDump::from(model_().linearizer().jacobian().istlMatrix());
        const auto jacobianDiff = compare(storedJacobian, liveJacobian,
                                          1e-300, config_.replayAbsTolerance);

        worstResidual = std::max(worstResidual, residualDiff.maxRelDiff);
        worstJacobian = std::max(worstJacobian, jacobianDiff.maxRelDiff);

        const bool pass = residualDiff.withinTolerance(config_.replayTolerance) &&
                          jacobianDiff.withinTolerance(config_.replayTolerance);

        OpmLog::info(fmt::format(
            "replay substep {:4d} (report step {:3d}, dt {:.4g} days): "
            "residual rel diff {:.3e} (abs {:.3e}), "
            "Jacobian rel diff {:.3e} (abs {:.3e}){}{}",
            k, meta.reportStep, meta.dt / 86400.0,
            residualDiff.maxRelDiff, residualDiff.maxAbsDiff,
            jacobianDiff.maxRelDiff, jacobianDiff.maxAbsDiff,
            pass ? "" : "  <-- EXCEEDS TOLERANCE",
            residualDiff.structureMatches && jacobianDiff.structureMatches
                ? "" : "  (structure mismatch)"));

        return pass;
    }

    Simulator& simulator_;
    AdjointConfig config_;
    std::shared_ptr<AdjointArchive> archive_;
    //! Holds the recomputed states of the segment currently being swept; only
    //! consulted for substeps the recording skipped.
    std::shared_ptr<AdjointArchive> segmentArchive_;
    AdjointMeta meta_;
    bool computeBdiagEnabled_{false};
    std::vector<BdiagBlock> bdiag_;
    double worstResidualEval_{0.0};
    bool residualEvalFailed_{false};
    double worstJacobianEval_{0.0};
    bool jacobianEvalFailed_{false};
    //! Hoisted scratch for the per-substep evaluator checks.
    std::unique_ptr<Matrix> checkMat_;
    std::unique_ptr<typename AdjointResidualEvaluator<TypeTag>::SeparateIQ>
        sepIQ_;
    double loadSnapshotSeconds_{0.0};  //!< WS6 instrumentation
    int loadSnapshotCalls_{0};
};

} // namespace Opm

#endif // OPM_ADJOINT_REPLAY_HPP
