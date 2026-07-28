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
 * \brief Flow problem variant that records per-substep adjoint state.
 *
 * All recording hooks live in lifecycle overrides of the problem class
 * (the geomech extension pattern), so the module builds against
 * unmodified opm-simulators master:
 *  - beginEpisode(): create the recorder lazily; snapshot the initial
 *    state after the wells of report step 0 are set up;
 *  - endTimeStep(): called exactly once per *accepted* substep (both in
 *    the adaptive substep loop and the non-adaptive path) — record the
 *    substep snapshot after the parent has finished its end-of-step
 *    processing;
 *  - setSubStepReport()/finalizeOutput() are non-virtual upstream but are
 *    always invoked on the concrete Problem type from templated code, so
 *    name hiding is sufficient.
 */
#ifndef OPM_ADJOINT_FLOW_PROBLEM_HPP
#define OPM_ADJOINT_FLOW_PROBLEM_HPP

#include <opm/models/utils/parametersystem.hpp>
#include <opm/models/utils/propertysystem.hh>

#include <opm/simulators/flow/FlowProblemBlackoil.hpp>
#include <opm/simulators/timestepping/SimulatorReport.hpp>

#include <opm/adjoint/AdjointParameters.hpp>
#include <opm/adjoint/AdjointRecorder.hpp>

#include <memory>

namespace Opm {

template<class TypeTag>
class AdjointFlowProblem : public FlowProblemBlackoil<TypeTag>
{
    using ParentType = FlowProblemBlackoil<TypeTag>;
    using Simulator = GetPropType<TypeTag, Properties::Simulator>;

public:
    explicit AdjointFlowProblem(Simulator& simulator)
        : ParentType(simulator)
    {
        // One-sided half transmissibilities are needed for the adjoint
        // permeability chain rule; must be enabled before finishInit()
        // triggers the transmissibility update.
        this->transmissibilities_.setStoreHalfTrans(true);
    }

    static void registerParameters()
    {
        ParentType::registerParameters();
        registerAdjointParameters();
    }

    //! \brief One-stop archive: when set (by the onestop driver before the
    //!        forward run), the recorder writes into this shared in-memory
    //!        archive so the backward sweep in the same process reads it
    //!        directly — no files. Cleared after the forward run.
    static std::shared_ptr<AdjointArchive>& oneStopArchive()
    {
        static std::shared_ptr<AdjointArchive> archive;
        return archive;
    }

    //! \brief Install a recorder for a recomputed segment, at sweep time.
    //!
    //! Needed because the beginEpisode() logic below is a ONE-SHOT:
    //! recorderInitialized_ latches on the first call, and in gradient/replay
    //! mode that call declines a recorder (AdjointMode is non-empty), for good
    //! reason — the sweep must not overwrite the archive it is reading. But a
    //! checkpoint recompute then runs the forward model on this same problem
    //! instance and would silently record NOTHING: endTimeStep() sees a null
    //! recorder_ and does nothing, and the sweep only finds out later, as a
    //! missing dataset. So the segment driver installs the recorder explicitly.
    //!
    //! \param archive In-memory archive holding this segment's states.
    //! \param globalIndexOffset Global index of the segment's first substep, so
    //!        the groups it writes match the global numbering the sweep reads.
    //! \param recordedMeta Metadata of the archive being swept; the segment must
    //!        be written in the same format the reader expects.
    void resetRecorder(std::shared_ptr<AdjointArchive> archive,
                       int globalIndexOffset,
                       const AdjointMeta& recordedMeta)
    {
        recorder_ = std::make_unique<AdjointRecorder<TypeTag>>(
            this->simulator(), std::move(archive), globalIndexOffset,
            &recordedMeta);
        recorderInitialized_ = true;
        // The segment's leading state is the checkpoint the driver restored, so
        // there is no "initial" snapshot to take here.
        initialRecorded_ = true;
        substepInEpisode_ = 0;
        // These report steps are being re-run, so their ECL output already
        // exists; see writeOutput() below.
        rerunning_ = true;
    }

    //! \brief Drop the segment recorder once its segment has been swept, so a
    //!        later replayStep cannot record into a stale archive.
    void clearRecorder()
    {
        recorder_.reset();
        rerunning_ = false;
    }

    //! \brief Suppress ECL output while a segment is being recomputed.
    //!
    //! A recompute drives the real forward step, and runStep ends every report
    //! step with writeOutput(true). Those report steps were already written by
    //! the forward run, so letting the rerun write again would append duplicate
    //! entries to the UNRST/summary of the very run we are differentiating.
    //! The sweep's product is gradients, not ECL output, so a rerun writes
    //! nothing. Only recomputed steps are affected: outside a segment
    //! (rerunning_ == false) output behaves exactly as before.
    void writeOutput(const bool verbose) override
    {
        if (rerunning_) {
            return;
        }
        ParentType::writeOutput(verbose);
    }

    //! \brief Whether a recorder is currently installed (segment driver debug).
    bool hasRecorder() const
    {
        return static_cast<bool>(recorder_);
    }

    void beginEpisode() override
    {
        ParentType::beginEpisode();
        substepInEpisode_ = 0;

        if (!recorderInitialized_) {
            recorderInitialized_ = true;
            // Record in forward runs; the replay/adjoint stages reuse this
            // problem class but must not overwrite the archive. In one-stop the
            // archive presence drives recording (mode is not empty there).
            if (oneStopArchive()) {
                recorder_ = std::make_unique<AdjointRecorder<TypeTag>>(
                    this->simulator(), oneStopArchive());
            } else if (Parameters::Get<Parameters::AdjointSave>() &&
                       Parameters::Get<Parameters::AdjointMode>().empty()) {
                recorder_ = std::make_unique<AdjointRecorder<TypeTag>>(this->simulator());
            }
        }
        if (recorder_ && !initialRecorded_ && this->simulator().episodeIndex() == 0) {
            // The well/group state is now in its per-episode condition,
            // so the snapshot is deserializable via prepareDeserialize(0).
            recorder_->recordInitial();
            initialRecorded_ = true;
        }
    }

    void endTimeStep() override
    {
        ParentType::endTimeStep();
        if (recorder_) {
            const auto& simulator = this->simulator();
            recorder_->recordSubstep(simulator.time(),
                                     simulator.timeStepSize(),
                                     substepInEpisode_,
                                     lastSubStepReport_);
            ++substepInEpisode_;
        }
    }

    //! \brief Capture the substep report (Newton iterations for the
    //!        archive metadata); hides the non-virtual parent method,
    //!        which is always called on the concrete Problem type.
    void setSubStepReport(const SimulatorReportSingle& report)
    {
        lastSubStepReport_ = report;
        ParentType::setSubStepReport(report);
    }

    //! \brief Write the archive metadata at the end of the run (hides the
    //!        non-virtual parent method; called from the simulator's
    //!        finalize on the concrete Problem type).
    void finalizeOutput()
    {
        if (recorder_) {
            recorder_->finalize();
            recorder_.reset();
        }
        ParentType::finalizeOutput();
    }

private:
    std::unique_ptr<AdjointRecorder<TypeTag>> recorder_;
    SimulatorReportSingle lastSubStepReport_{};
    int substepInEpisode_{0};
    bool recorderInitialized_{false};
    bool initialRecorded_{false};
    //! Re-running already-simulated report steps for a checkpoint segment; see
    //! writeOutput().
    bool rerunning_{false};
};

} // namespace Opm

#endif // OPM_ADJOINT_FLOW_PROBLEM_HPP
