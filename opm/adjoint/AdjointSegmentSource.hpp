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
 * \brief Supplies the states of substeps a checkpointed recording skipped, by
 *        re-running the forward model from the preceding checkpoint.
 *
 * With --adjoint-checkpoint-every=M the archive holds a snapshot only every
 * M-th substep. The backward sweep still needs every one, so before replaying
 * substep k this materializes the whole segment k belongs to into an in-memory
 * archive and hands it to AdjointReplay.
 *
 * Segment algebra. Checkpoints sit at k = M-1, 2M-1, ..., and the always
 * written `initial` group is c_0 = -1, so segment s covers substeps
 * [s*M, s*M + M - 1] with leading state s*M - 1 — which is either `initial` or
 * a checkpoint, hence always stored. That matters because replayStep(k) loads
 * BOTH k-1 and k: for the first substep of a segment, k-1 is the leading state,
 * and for the rest both lie inside the segment. So a segment is self-contained
 * and no overlap handling is needed.
 *
 * The sweep walks k downwards, so segments are visited newest-first, but a
 * segment is always recomputed forward from its own checkpoint and is therefore
 * independent of the order they are visited in.
 */
#ifndef OPM_ADJOINT_SEGMENT_SOURCE_HPP
#define OPM_ADJOINT_SEGMENT_SOURCE_HPP

#include <opm/common/ErrorMacros.hpp>
#include <opm/common/OpmLog/OpmLog.hpp>

#include <opm/simulators/flow/FlowMain.hpp>

#include <opm/adjoint/AdjointMeta.hpp>
#include <opm/adjoint/AdjointReplay.hpp>
#include <opm/adjoint/AdjointStorage.hpp>

#include <fmt/format.h>

#include <algorithm>
#include <chrono>
#include <stdexcept>

namespace Opm {

template<class TypeTag>
class AdjointSegmentSource
{
public:
    //! \param flowMain Fully initialized driver (executeInitStep() already
    //!        run); its report-step stepping and timer are used to re-run
    //!        segments, on the SAME simulator the sweep re-linearizes on.
    //! \param replay The sweep's replay driver: used both to restore a
    //!        checkpoint and to receive the recomputed segment.
    AdjointSegmentSource(FlowMain<TypeTag>& flowMain,
                         AdjointReplay<TypeTag>& replay)
        : flowMain_(flowMain)
        , replay_(replay)
        , meta_(replay.meta())
    {
        stride_ = std::max(1, meta_.checkpointEvery);
        numSubsteps_ = static_cast<int>(meta_.substeps.size());
        if (flowMain_.getStepDriverPtr() == nullptr) {
            OPM_THROW(std::runtime_error,
                      "Adjoint segment recompute needs an initialized "
                      "simulator; call executeInitStep() first.");
        }
    }

    //! \brief Make the states replayStep(\p k) reads available.
    //!
    //! A no-op when both k and k-1 were stored, and when k's segment is already
    //! the materialized one — so a whole segment costs one recompute, not one
    //! per substep.
    void ensureAvailable(int k)
    {
        if (isStored_(k) && isStored_(k - 1)) {
            return;
        }
        const int segment = k / stride_;
        if (segment == materialized_) {
            return;
        }
        materialize_(segment);
        materialized_ = segment;
    }

    //! \brief Total wall time spent recomputing segments.
    double recomputeSeconds() const { return recomputeSeconds_; }
    //! \brief Number of segments recomputed, and forward substeps re-run.
    int segmentsRecomputed() const { return segments_; }
    int substepsRecomputed() const { return substeps_; }

private:
    bool isStored_(int k) const
    {
        return k < 0 || meta_.substeps[k].hasPayload;
    }

    //! Report step to hand prepareDeserialize when restoring state \p k;
    //! mirrors AdjointReplay::replayStep's choice for k-1.
    int reportStepFor_(int k) const
    {
        return (k >= 0) ? meta_.substeps[k].reportStep
                        : meta_.substeps[0].reportStep;
    }

    void materialize_(int segment)
    {
        const int last = std::min(segment * stride_ + stride_ - 1,
                                  numSubsteps_ - 1);
        // Start at the first substep of the segment that is actually missing,
        // rather than assuming it is the one right after segment*stride - 1.
        // Substep 0 is stored (see AdjointRecorder::isCheckpoint_), so segment 0
        // begins at 1 and leads from 0 — it never has to lean on `initial`.
        int first = segment * stride_;
        while (first <= last && isStored_(first)) {
            ++first;
        }
        if (first > last) {
            OPM_THROW(std::runtime_error, fmt::format(
                      "Adjoint segment {} has nothing to recompute (substeps "
                      "{}..{} are all stored)", segment,
                      segment * stride_, last));
        }
        const int lead = first - 1;
        if (!isStored_(lead)) {
            OPM_THROW(std::runtime_error, fmt::format(
                      "Adjoint segment {} would restart from substep {}, which "
                      "was not stored - the checkpoint layout is inconsistent "
                      "with the recorded stride {}.", segment, lead, stride_));
        }
        const auto t0 = std::chrono::steady_clock::now();

        auto* driver = flowMain_.getStepDriverPtr();
        auto& problem = flowMain_.getSimulatorPtr()->problem();

        // Fresh archive per segment: the previous one is dropped here, which is
        // what keeps peak memory at one segment rather than the whole tape.
        auto segmentArchive = AdjointArchive::makeInMemory();
        replay_.setSegmentArchive(nullptr);
        problem.resetRecorder(segmentArchive, first, meta_);

        // Restore the leading state at the start of the first re-run report
        // step, from inside runStep, where the OPMRST --load-step restore also
        // happens (after the episode is set up, before the solve). One-shot:
        // the hook fires on every report step, but only the first must be
        // preloaded — the rest continue from the recomputed trajectory.
        bool restored = false;
        driver->setRestoreStateHook([this, lead, &restored](int /*reportStep*/) {
            if (restored) {
                return;
            }
            restored = true;
            replay_.loadSnapshot(lead, this->reportStepFor_(lead));
        });

        auto* timer = flowMain_.getSimTimer();
        timer->setCurrentStepNum(meta_.substeps[first].reportStep);
        for (int n = first; n <= last; ++n) {
            if (!flowMain_.executeStep()) {
                driver->setRestoreStateHook({});
                problem.clearRecorder();
                OPM_THROW(std::runtime_error, fmt::format(
                          "Adjoint segment recompute stopped early at substep "
                          "{} (the schedule requested EXIT); the sweep cannot "
                          "continue without that state.", n));
            }
        }

        driver->setRestoreStateHook({});
        problem.clearRecorder();
        if (!restored) {
            OPM_THROW(std::runtime_error,
                      "Adjoint segment recompute never restored its checkpoint "
                      "- the restore hook was not invoked, so the re-run did "
                      "not follow the recorded trajectory.");
        }
        replay_.setSegmentArchive(segmentArchive);

        ++segments_;
        substeps_ += last - first + 1;
        recomputeSeconds_ += std::chrono::duration<double>(
            std::chrono::steady_clock::now() - t0).count();
        OpmLog::debug(fmt::format(
            "Adjoint segment {}: recomputed substeps {}..{} from checkpoint {}",
            segment, first, last, lead));
    }

    FlowMain<TypeTag>& flowMain_;
    AdjointReplay<TypeTag>& replay_;
    const AdjointMeta& meta_;
    int stride_ = 1;
    int numSubsteps_ = 0;
    int materialized_ = -1;        //!< segment currently in the segment archive
    int segments_ = 0;
    int substeps_ = 0;
    double recomputeSeconds_ = 0.0;
};

} // namespace Opm

#endif // OPM_ADJOINT_SEGMENT_SOURCE_HPP
