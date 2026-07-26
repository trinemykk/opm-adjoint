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
 * \brief Runtime parameters controlling adjoint state recording and replay.
 *
 * These parameters are registered by all flow variants (via
 * SimulatorFullyImplicit::registerParameters) but everything is off by
 * default; a normal run is unaffected.
 */
#ifndef OPM_ADJOINT_PARAMETERS_HPP
#define OPM_ADJOINT_PARAMETERS_HPP

#include <opm/adjoint/AdjointObjectiveSpec.hpp>

#include <optional>
#include <string>

namespace Opm::Parameters {

//! \brief Path to a JSON configuration file (--adjoint-config). It provides
//!        base values for the adjoint settings and a structured objective
//!        (weights/windows/wildcards); any --adjoint-* flag set explicitly on
//!        the command line overrides its JSON counterpart. Empty disables.
struct AdjointConfig { static constexpr auto value = ""; };

//! \brief Record per-substep simulation state for later adjoint/replay runs.
struct AdjointSave { static constexpr bool value = false; };

//! \brief Store SLIM per-substep snapshots: the dynamic state only (model
//!        solution, problem, summary/UDQ/action state), skipping the static
//!        Eclipse schedule (reconstructed from the deck at replay). Much
//!        smaller archives; OFF by default (the full snapshot is the proven
//!        path). Not for decks that modify the schedule at runtime (ACTIONX
//!        that alters the schedule) — those need the full archive.
struct AdjointSlim { static constexpr bool value = false; };

//! \brief Additionally store the converged residual and Jacobian of every
//!        accepted substep ("write too much" verification data for the
//!        replay tests).
struct AdjointSaveSystem { static constexpr bool value = false; };

//! \brief Run mode of the simulator with respect to adjoint processing:
//!        "" (normal forward run) or "replay" (backward replay of a
//!        recorded forward run instead of the time loop).
struct AdjointMode { static constexpr auto value = ""; };

//! \brief Path of the adjoint archive. Empty means
//!        <output-dir>/<CASENAME>.ADJOINT[.h5]. A path ending in .h5/.hdf5
//!        selects the HDF5 backend, anything else a plain directory store.
struct AdjointFile { static constexpr auto value = ""; };

//! \brief Linear solver for the transposed adjoint systems:
//!        auto (default) | umfpack (direct) | ilu0 | cpr | cprt | <file>.json
//!        (FlexibleSolver configuration).
//!        "auto" selects by problem size: umfpack (exact) for small cases —
//!        the verification oracle — and cprt (transposed CPR, ~forward memory)
//!        above a modest cell count, because umfpack's LU fill is super-linear
//!        (26x forward RAM on SPE9, OOM at field scale). Parallel runs pick an
//!        iterative solver regardless (umfpack is single-rank). Any explicit
//!        value here overrides the auto choice.
struct AdjointLinearSolver { static constexpr auto value = "auto"; };

//! \brief Relative residual reduction for iterative adjoint solves.
//!        Tighter than forward solves: lambda accuracy multiplies dR/dm
//!        directly in the gradients.
struct AdjointLinearSolverReduction { static constexpr double value = 1e-14; };

//! \brief Iteration cap for iterative adjoint solves.
struct AdjointLinearSolverMaxIter { static constexpr int value = 300; };

//! \brief Verbosity for the adjoint linear solver (passed to
//!        FlexibleSolver; 0 = quiet).
struct AdjointLinearSolverVerbosity { static constexpr int value = 0; };

//! \brief Comma-separated list of end-point scaling parameters to
//!        compute gradients for (e.g. "SWL,KRW"); empty disables.
struct AdjointEndpoints { static constexpr auto value = ""; };

//! \brief Objective function for adjoint/gradient runs:
//!        "pressure-average" or "bhp:<WELL>".
struct AdjointObjective { static constexpr auto value = "pressure-average"; };

//! \brief Relative tolerance used when the replay driver compares the
//!        re-linearized systems against the stored ones.
struct AdjointReplayTolerance { static constexpr double value = 1e-12; };

//! \brief Absolute noise floor for the replay comparison: entries whose
//!        absolute difference is below this are counted as rounding noise
//!        and excluded from the relative criterion.
struct AdjointReplayAbsTolerance { static constexpr double value = 1e-9; };

//! \brief During replay, also recompute the residual with the standalone
//!        AdjointResidualEvaluator and verify it matches the production
//!        linearizer bitwise (validation of the residual-only evaluator).
struct AdjointResidualEvalCheck { static constexpr bool value = false; };

//! \brief How PV/trans/perm parameter gradients are accumulated:
//!        "general-ad" (seeded-AD stencil-local evaluation, the generic
//!        default — exact to machine precision, verified on SPE9), "general"
//!        (the same stencil-local mechanism by finite difference), or
//!        "analytic" (hand-derived closed forms; fastest, kept as the
//!        verification oracle).
struct AdjointGradientMethod { static constexpr auto value = "general-ad"; };

//! \brief Central-difference step for the finite-difference gradient
//!        paths (the 'general' method's multiplier perturbations and the
//!        end-point gradients). Relative to O(1) multipliers; the exact
//!        'general-ad' path does not use it.
struct AdjointFdStep { static constexpr double value = 1e-7; };

//! \brief Include the dJ/dperm well-coupling term at defaulted-CTF
//!        perforations (the connection transmissibility factor is
//!        perm-derived, so the perm gradient has a well-equation
//!        contribution the cell-based reservoir gradient omits). Default on;
//!        a no-op for explicit-CTF decks.
struct AdjointPermWellTerm { static constexpr bool value = true; };

//! \brief How the end-point FD gradients evaluate the perturbed residual:
//!        "whole" (full re-linearization per perturbation, the original
//!        cell x Ncells method, default) or "stencil" (stencil-local
//!        re-evaluation, cell x ~7, with a whole-domain fallback for
//!        perforated cells whose perturbation reaches the well equations).
struct AdjointEndpointMethod { static constexpr auto value = "whole"; };

} // namespace Opm::Parameters

namespace Opm {

//! \brief Settings for adjoint state recording, resolved from the
//!        parameter system.
struct AdjointConfig
{
    bool save{false};         //!< record per-substep state
    std::string mode{};       //!< "" (forward) or "replay"
    bool saveSystem{false};   //!< also dump residual/Jacobian per substep
    bool slim{false};         //!< slim snapshots (skip serialized schedule)
    std::string path{};       //!< archive path ("" = derive from case name)
    double replayTolerance{1e-12};
    double replayAbsTolerance{1e-9};
    bool residualEvalCheck{false}; //!< verify the standalone residual evaluator
    std::string gradientMethod{"general-ad"}; //!< general-ad|general|analytic
    double fdStep{1e-7};      //!< central-difference step for FD gradients
    std::string endpointMethod{"whole"}; //!< endpoint FD: "whole" or "stencil"
    bool permWellTerm{true};  //!< dCTF/dperm well-coupling term at perforations
    std::string objective{};  //!< objective specification (CLI string form)
    //! Structured objective from a JSON config; when set it takes precedence
    //! over the `objective` string (which cannot express per-term weights on
    //! mixed families). Empty => parse `objective`.
    std::optional<Adjoint::ObjectiveSpec> objectiveSpec{};
    std::string endpoints{};  //!< end-point parameter list
    std::string linearSolver{"auto"};  //!< adjoint linear solver spec; "auto"
                                       //!< = size-based (umfpack small / cprt large)
    double linearSolverReduction{1e-14};
    int linearSolverMaxIter{300};
    int linearSolverVerbosity{0};

    static AdjointConfig fromParameters();
};

//! \brief Register the adjoint runtime parameters.
void registerAdjointParameters();

} // namespace Opm

#endif // OPM_ADJOINT_PARAMETERS_HPP
