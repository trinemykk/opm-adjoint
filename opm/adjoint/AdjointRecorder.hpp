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
 * \brief Records per-substep simulation state during a forward run for
 *        later adjoint/replay processing.
 *
 * Storage philosophy: write too much at first, optimize later. Every
 * accepted substep stores a full simulator snapshot (via the same
 * serializeOp machinery as --save-step) plus, optionally, the converged
 * residual and Jacobian for replay verification.
 */
#ifndef OPM_ADJOINT_RECORDER_HPP
#define OPM_ADJOINT_RECORDER_HPP

#include <opm/common/OpmLog/OpmLog.hpp>

#include <fmt/format.h>

#include <opm/models/discretization/common/fvbaseparameters.hh>
#include <opm/models/utils/parametersystem.hpp>
#include <opm/models/utils/propertysystem.hh>

#include <opm/adjoint/AdjointMeta.hpp>
#include <opm/adjoint/AdjointParameters.hpp>
#include <opm/adjoint/AdjointWellModel.hpp>
#include <opm/adjoint/AdjointStorage.hpp>
#include <opm/adjoint/AdjointSystemIO.hpp>
#include <opm/simulators/flow/FlowProblemParameters.hpp>
#include <opm/simulators/flow/SimulatorFullyImplicit.hpp>
#include <opm/simulators/timestepping/EclTimeSteppingParams.hpp>
#include <opm/simulators/timestepping/SimulatorReport.hpp>

#include <algorithm>
#include <chrono>
#include <memory>
#include <sstream>
#include <string>

namespace Opm {

template<class TypeTag>
class AdjointRecorder
{
public:
    using Simulator = GetPropType<TypeTag, Properties::Simulator>;

    //! \brief Owning recorder: creates the archive from the config path.
    explicit AdjointRecorder(Simulator& simulator)
        : AdjointRecorder(simulator, nullptr)
    {}

    //! \brief One-stop / segment recorder: records into a shared (in-memory)
    //!        archive so the backward sweep in the same process reads the
    //!        states directly.
    //! \param globalIndexOffset Global index of this recorder's first substep.
    //!        Non-zero for a segment recompute, whose local substep j is the
    //!        run's global substep offset+j — the sweep asks for groups by
    //!        global index, so the groups must be numbered that way.
    //! \param recordedMeta Non-null puts the recorder in SEGMENT mode: it is
    //!        recording a recomputed stretch rather than a forward run. Every
    //!        substep then gets a payload (the sweep needs both k and k-1, so a
    //!        segment must be fully materialized), the configuration warnings
    //!        are left to the original forward run, and — importantly — the
    //!        snapshot FORMAT is taken from the recording rather than from this
    //!        process's parameters. A sweep does not have to be invoked with the
    //!        same --adjoint-slim as the run it is differentiating, and the
    //!        reader follows the recorded meta, so the segment must match it.
    AdjointRecorder(Simulator& simulator,
                    std::shared_ptr<AdjointArchive> injectedArchive,
                    int globalIndexOffset = 0,
                    const AdjointMeta* recordedMeta = nullptr)
        : simulator_(simulator)
        , config_(AdjointConfig::fromParameters())
        , globalIndexOffset_(globalIndexOffset)
        , segmentMode_(recordedMeta != nullptr)
    {
        std::string path;
        if (injectedArchive) {
            archive_ = std::move(injectedArchive);
            // One-stop injects either the RAM store or a disk archive, so ask
            // the archive rather than assuming.
            path = archive_->path();
        } else {
            path = config_.path;
            if (path.empty()) {
                const auto& ioConfig = simulator_.vanguard().eclState().getIOConfig();
                path = AdjointArchive::defaultPath(ioConfig.getOutputDir(),
                                                   simulator_.vanguard().caseName());
            }
            archive_ = std::make_shared<AdjointArchive>(
                path, AdjointArchive::Mode::Write,
                simulator_.vanguard().gridView().comm());
        }

        meta_.systemSaved = config_.saveSystem;
        meta_.storageCacheEnabled = Parameters::Get<Parameters::EnableStorageCache>();
        meta_.slim = config_.slim;
        meta_.checkpointEvery = std::max(1, config_.checkpointEvery);
        meta_.caseName = simulator_.vanguard().caseName();

        if (segmentMode_) {
            // A segment stores every substep it recomputes, whatever stride the
            // original recording used.
            meta_.checkpointEvery = 1;
            // Match the recording's format, not this process's flags: the reader
            // dispatches on the recorded meta.slim, and a sweep need not have
            // been given the same --adjoint-slim as the forward run.
            config_.slim = recordedMeta->slim;
            meta_.slim = recordedMeta->slim;
            // Verification dumps belong to the recording; a recomputed substep
            // has nothing to compare against (see compareWithStored_).
            config_.saveSystem = false;
            meta_.systemSaved = false;
            return;   // no guards, no banner: this is not a fresh forward run
        }

        checkConfiguration_();
        const std::string backend =
            archive_->inMemory() ? "" : (" (" + archive_->backendTag() + ")");
        OpmLog::info("Adjoint recording enabled; archive: " + path + backend);
        if (config_.slim) {
            OpmLog::info("Adjoint slim archive: per-substep snapshots omit the "
                         "static schedule (reconstructed from the deck at "
                         "replay). Do NOT use on decks that modify the schedule "
                         "at runtime (ACTIONX/UDQ-driven WELOPEN, WELTARG, ...) "
                         "— use the full archive there.");
        }
    }

    //! \brief Snapshot the initial state; call after the well model has
    //!        been set up for report step 0 (beginEpisode) but before the
    //!        first solve.
    void recordInitial()
    {
        // Sync the internal WGState copies so the snapshot can later be
        // read against a prepareDeserialize()'d well model in replay.
        normalizeWGStateForSerialization(simulator_.problem().wellModel());
        const auto t0 = Clock::now();
        writeSnapshot_(AdjointGroups::initial());
        recordSeconds_ += secs_(t0);
    }

    //! \brief Write one snapshot. Full = the whole simulator (proven). Slim =
    //!        the dynamic state only (model solution, problem, summary/UDQ/
    //!        action), skipping the static schedule + time scalars that the
    //!        replay reconstructs from the deck and meta. Symmetric with
    //!        AdjointReplay::loadSnapshot_.
    void writeSnapshot_(const std::string& group)
    {
        if (config_.slim) {
            auto& v = simulator_.vanguard();
            archive_->write(simulator_.model(),   group, "model");
            archive_->write(simulator_.problem(), group, "problem");
            archive_->write(v.summaryState(),     group, "summary");
            archive_->write(v.udqState(),         group, "udq");
            archive_->write(v.actionState(),      group, "action");
        } else {
            archive_->write(simulator_, group, "simulator");
        }
    }

    //! \brief Record an accepted substep; call after problem.endTimeStep()
    //!        and before the substep timer advances.
    //! \param startTime Simulation time at the start of the substep [s].
    //! \param dt Substep length [s].
    //! \param substepInReport Substep counter within the current report step.
    //! \param report Solver report of this substep.
    void recordSubstep(double startTime,
                       double dt,
                       int substepInReport,
                       const SimulatorReportSingle& report)
    {
        const int k = globalIndexOffset_ + static_cast<int>(meta_.substeps.size());
        const std::string group = AdjointGroups::substep(k);

        const auto t0 = Clock::now();
        // Recording side of OPM_ADJOINT_WELLSTATE_DIAG; see the matching
        // print in AdjointReplay::loadSnapshot_. Note that numLocalWellsEnd()
        // counts this rank's wells at the END of the schedule, so it stays
        // constant - wellStateSize() is the one that tracks the current step.
        if (std::getenv("OPM_ADJOINT_WELLSTATE_DIAG")) {
            const auto& wm = simulator_.problem().wellModel();
            std::cerr << fmt::format(
                "[wellstate-diag REC] rank {} substep {} episode {}: "
                "localWells={} wellStateSize={}\n",
                simulator_.vanguard().grid().comm().rank(), k,
                simulator_.episodeIndex(), wm.numLocalWellsEnd(),
                wm.wellState().size());
        }
        // Under checkpointing only every M-th substep carries a payload; the
        // rest are recomputed forward from the preceding checkpoint during the
        // sweep. The meta list below stays DENSE either way — the sweep needs
        // dt/reportStep for every substep, and it costs a few bytes.
        const bool checkpoint = isCheckpoint_(k);
        if (checkpoint) {
            writeSnapshot_(group);
            if (!config_.slim) {
                // Retained for the full (proven) format; the well/group state is
                // already inside the problem snapshot, so slim omits this.
                auto& wellModel = simulator_.problem().wellModel();
                archive_->write(wellModel, group, "wgstate");
            }

            if (config_.saveSystem) {
                recordSystem_(group);
            }
        }
        recordSeconds_ += secs_(t0);

        AdjointSubstepMeta meta;
        meta.globalIdx = k;
        meta.reportStep = simulator_.episodeIndex();
        meta.substepInReport = substepInReport;
        meta.startTime = startTime;
        meta.dt = dt;
        meta.newtonIterations = report.total_newton_iterations;
        meta.hasPayload = checkpoint;
        meta_.substeps.push_back(meta);
    }

    //! \brief Write the archive metadata; call once at the end of the run.
    void finalize()
    {
        archive_->write(meta_, AdjointGroups::meta(), "meta");

        if (segmentMode_) {
            // One segment out of many: the provenance text belongs to the
            // forward run's archive, and per-segment log lines would drown the
            // sweep output. The sweep reads state by global index, not meta.
            return;
        }

        std::ostringstream params;
        Parameters::printValues(params);
        archive_->writeText(AdjointGroups::meta(), "parameters", params.str());

        // Recording-phase timing (serialize + archive write, fused). This is
        // the forward-side adjoint overhead; it lands in the .DBG/.PRT next to
        // the backward "Adjoint sweep timing" block so the whole adjoint cost
        // is visible at the end of the run. IO is not free — this quantifies it.
        const int n = static_cast<int>(meta_.substeps.size());
        const char* medium = archive_->inMemory() ? "in-memory"
                           : (archive_->usesHdf5() ? "HDF5" : "directory store");
        OpmLog::info(fmt::format(
            "Adjoint recording timing: {} snapshots in {:.3f} s "
            "({:.1f} ms/snapshot, {} archive)",
            n + 1, recordSeconds_,
            1000.0 * recordSeconds_ / std::max(1, n + 1), medium));
        const int stored = static_cast<int>(
            std::count_if(meta_.substeps.begin(), meta_.substeps.end(),
                          [](const AdjointSubstepMeta& m) { return m.hasPayload; }));
        if (meta_.checkpointEvery > 1) {
            OpmLog::info(fmt::format(
                "Adjoint recording finished: {} substeps, {} snapshots stored "
                "(checkpoint stride {}); the remaining {} are recomputed in the "
                "backward sweep",
                meta_.substeps.size(), stored + 1, meta_.checkpointEvery,
                static_cast<int>(meta_.substeps.size()) - stored));
        } else {
            OpmLog::info("Adjoint recording finished: " +
                         std::to_string(meta_.substeps.size()) + " substeps stored");
        }
    }

    int numRecordedSubsteps() const
    { return static_cast<int>(meta_.substeps.size()); }

private:
    //! \brief Validity guard, warnings only: the warning list is the
    //!        worklist; nothing is refused at this stage.
    void checkConfiguration_()
    {
        if (meta_.storageCacheEnabled) {
            OpmLog::warning("Adjoint recording: the storage cache is enabled "
                            "(--enable-storage-cache=true). The replay driver "
                            "currently assumes the cache-free storage term; run "
                            "the forward simulation with "
                            "--enable-storage-cache=false for exact replay.");
        }
        if (meta_.checkpointEvery > 1) {
            // The substeps without a payload are reproduced by re-running the
            // forward solve, so the recorded substep sequence has to be
            // reproducible. Adaptive stepping breaks that twice over: it can
            // pick different substeps on the recompute, and the tape does not
            // carry the controller's internal state (suggestedNextStep and the
            // PID error history live in SimulatorFullyImplicit, not Simulator).
            // TUNING is the same story - its updater clears schedule events, so
            // re-entering a report step silently skips the update.
            // Refused here rather than at replay: the archive would be unusable.
            if (Parameters::Get<Parameters::EnableAdaptiveTimeStepping>()) {
                OPM_THROW(std::runtime_error,
                          "--adjoint-checkpoint-every>1 requires fixed time "
                          "stepping: the substeps between checkpoints are "
                          "recomputed, which cannot reproduce an adaptively "
                          "chosen substep sequence. Re-run with "
                          "--enable-adaptive-time-stepping=false.");
            }
            if (Parameters::Get<Parameters::EnableTuning>()) {
                OPM_THROW(std::runtime_error,
                          "--adjoint-checkpoint-every>1 is not compatible with "
                          "--enable-tuning=true: applying TUNING clears schedule "
                          "events, so a recomputed report step would see a "
                          "different configuration than the recorded one.");
            }
            OpmLog::warning(fmt::format(
                "Adjoint checkpointing (--adjoint-checkpoint-every={}) is UNDER "
                "DEVELOPMENT. Storing every {}th substep and recomputing the "
                "rest is verified on SPE1/SPE9 only, serial, fixed-step. It is "
                "not needed for the model sizes this adjoint currently targets - "
                "use the default (=1) unless you are deliberately trading "
                "storage for forward compute on a large case.",
                meta_.checkpointEvery, meta_.checkpointEvery));
        }
        if (Parameters::Get<Parameters::EnableDriftCompensation>()) {
            // Drift compensation adds an accumulated mass-balance correction to
            // the residual (FlowProblemBlackoil::addToSourceDense), which makes
            // R depend on the whole history rather than on (x_k, x_{k-1}) alone.
            // The adjoint recursion has no term for that, so the gradients would
            // be silently wrong. Warned here, refused in AdjointReplay - the same
            // asymmetry as the storage cache above, so a forward run that also
            // wants its ECL output still completes.
            //
            // Only the non-Temp flag matters: EnableDriftCompensationTemp
            // defaults to true, but FlowProblem only reads it for the
            // SequentialImplicitThermal energy module, so testing it here would
            // refuse every default blackoil run.
            OpmLog::warning("Adjoint recording: drift compensation is enabled "
                            "(--enable-drift-compensation=true). Its accumulated "
                            "residual correction is not part of the adjoint "
                            "recursion, so the backward sweep will refuse this "
                            "archive. Re-run with "
                            "--enable-drift-compensation=false.");
        }
        if (simulator_.gridView().comm().size() > 1) {
            OpmLog::info("Adjoint recording: parallel run - the archive is "
                         "written per rank, so replay/gradient runs must use "
                         "the same number of MPI processes (and the same "
                         "partitioning).");
        }
        if (Parameters::Get<Parameters::ThreadsPerProcess>() > 1) {
            OpmLog::warning("Adjoint recording: more than one thread per process "
                            "may prevent bitwise reproduction in replay; use "
                            "--threads-per-process=1 for the strictest tests.");
        }
        if (!config_.saveSystem) {
            OpmLog::info("Adjoint recording: residual/Jacobian dumps disabled "
                         "(enable with --adjoint-save-system=true for replay "
                         "verification).");
        }
    }

    void recordSystem_(const std::string& group)
    {
        const auto& linearizer = simulator_.model().linearizer();
        auto residual = AdjointVectorDump::from(linearizer.residual());
        archive_->write(residual, group + "/system", "residual");
        auto jacobian = AdjointMatrixDump::from(linearizer.jacobian().istlMatrix());
        archive_->write(jacobian, group + "/system", "jacobian");
    }

    //! \brief Whether substep \p k carries a state payload.
    //!
    //! Checkpoints land at k = M-1, 2M-1, ..., so with c_0 = -1 (the always
    //! written `initial` group) every segment (c_j, c_{j+1}] holds exactly M
    //! substeps. M = 1 makes every substep a checkpoint, i.e. bit-for-bit the
    //! pre-checkpointing behaviour.
    bool isCheckpoint_(int k) const
    {
        if (segmentMode_) {
            return true;   // a segment must be fully materialized
        }
        const int m = meta_.checkpointEvery;
        // Substep 0 is always stored. Recomputing it means re-running the first
        // report step from the `initial` snapshot, and the snapshot that comes
        // back does not reproduce R(x_0, x_-1): its x_{k-1} slot is wrong, so
        // the replayed residual is O(1) instead of ~1e-9 while every later
        // substep is bitwise exact. Storing it costs one snapshot out of N/M and
        // removes the only segment boundary that starts from `initial`.
        return m <= 1 || k == 0 || ((k + 1) % m) == 0;
    }

    using Clock = std::chrono::steady_clock;
    static double secs_(Clock::time_point t0)
    {
        return std::chrono::duration<double>(Clock::now() - t0).count();
    }

    Simulator& simulator_;
    AdjointConfig config_;
    int globalIndexOffset_ = 0;    //!< global index of this recorder's substep 0
    bool segmentMode_ = false;     //!< recording a recomputed segment, not a run
    std::shared_ptr<AdjointArchive> archive_;
    AdjointMeta meta_;
    double recordSeconds_ = 0.0;   //!< wall time spent serializing + writing
};

} // namespace Opm

#endif // OPM_ADJOINT_RECORDER_HPP
