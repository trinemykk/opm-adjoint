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
 * \brief Metadata describing the contents of an adjoint archive.
 */
#ifndef OPM_ADJOINT_META_HPP
#define OPM_ADJOINT_META_HPP

#include <cstddef>
#include <string>
#include <vector>

namespace Opm {

//! \brief Per-substep record in the adjoint archive.
//!
//! One entry per *accepted* substep, in forward order; failed/chopped
//! attempts are never recorded. Substep k stores the simulator snapshot at
//! the END of the substep, i.e. solution(0) = x_k and solution(1) = x_{k-1}.
struct AdjointSubstepMeta
{
    int globalIdx{-1};        //!< global substep counter (0-based over whole run)
    int reportStep{-1};       //!< report step (episode index) this substep belongs to
    int substepInReport{-1};  //!< substep counter within the report step
    double startTime{0.0};    //!< simulation time at the start of the substep [s]
    double dt{0.0};           //!< substep length [s]
    unsigned newtonIterations{0}; //!< Newton iterations used in the forward solve
    //! \brief Whether the state payload for this substep is in the archive.
    //!
    //! False under checkpointing (--adjoint-checkpoint-every > 1): the meta list
    //! stays dense because the sweep needs dt/reportStep at every substep, but
    //! only checkpoints carry a snapshot; the rest are recomputed. Also gates
    //! the residual/Jacobian dumps, which only exist where a payload does.
    bool hasPayload{true};

    template<class Serializer>
    void serializeOp(Serializer& serializer)
    {
        serializer(globalIdx);
        serializer(reportStep);
        serializer(substepInReport);
        serializer(startTime);
        serializer(dt);
        serializer(newtonIterations);
        serializer(hasPayload);
    }

    bool operator==(const AdjointSubstepMeta& rhs) const
    {
        return globalIdx == rhs.globalIdx &&
               reportStep == rhs.reportStep &&
               substepInReport == rhs.substepInReport &&
               startTime == rhs.startTime &&
               dt == rhs.dt &&
               newtonIterations == rhs.newtonIterations &&
               hasPayload == rhs.hasPayload;
    }
};

//! \brief Archive-level metadata, written once at the end of the forward run.
struct AdjointMeta
{
    //! Bumped to 2 when hasPayload/checkpointEvery were added; the reader
    //! rejects anything else outright (AdjointReplay's constructor).
    static constexpr int currentSchemaVersion = 2;

    int schemaVersion{currentSchemaVersion};
    bool systemSaved{false};        //!< residual/Jacobian dumps present
    bool storageCacheEnabled{true}; //!< EnableStorageCache of the forward run
    bool slim{false};               //!< slim snapshots (no serialized schedule)
    //! \brief Snapshot stride: 1 = every substep (no recompute needed), M > 1 =
    //!        only every M-th substep carries a payload and the rest are
    //!        recomputed forward from the preceding checkpoint.
    int checkpointEvery{1};
    std::string caseName{};
    std::vector<AdjointSubstepMeta> substeps;

    template<class Serializer>
    void serializeOp(Serializer& serializer)
    {
        serializer(schemaVersion);
        serializer(systemSaved);
        serializer(storageCacheEnabled);
        serializer(slim);
        serializer(checkpointEvery);
        serializer(caseName);
        serializer(substeps);
    }

    bool operator==(const AdjointMeta& rhs) const
    {
        return schemaVersion == rhs.schemaVersion &&
               systemSaved == rhs.systemSaved &&
               storageCacheEnabled == rhs.storageCacheEnabled &&
               slim == rhs.slim &&
               checkpointEvery == rhs.checkpointEvery &&
               caseName == rhs.caseName &&
               substeps == rhs.substeps;
    }

    //! \brief True when every substep carries its own snapshot, i.e. the sweep
    //!        can read all state straight from the archive with no recompute.
    bool allPayloadsPresent() const
    {
        return checkpointEvery <= 1;
    }
};

namespace AdjointGroups {

//! Group holding the snapshot of the initial state (before the first substep).
inline std::string initial()
{ return "/adjoint/initial"; }

//! Group holding all data of accepted substep \p k.
inline std::string substep(int k)
{ return "/adjoint/substep/" + std::to_string(k); }

//! Group holding archive metadata.
inline std::string meta()
{ return "/adjoint"; }

} // namespace AdjointGroups

} // namespace Opm

#endif // OPM_ADJOINT_META_HPP
