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

#include <config.h>
#include <opm/adjoint/AdjointParameters.hpp>
#include <opm/adjoint/AdjointConfigFile.hpp>

#include <opm/models/utils/parametersystem.hpp>

namespace Opm {

AdjointConfig AdjointConfig::fromParameters()
{
    AdjointConfig config;

    // A JSON config file provides the base values; any --adjoint-* flag set
    // explicitly on the command line overrides its JSON counterpart below
    // (Parameters::IsSet is true only for flags actually passed).
    const std::string configFile =
        Parameters::Get<Parameters::AdjointConfig>();
    if (!configFile.empty()) {
        loadAdjointConfigFile(configFile, config);
    }

    const auto override_ = [](auto& field, auto param) {
        using P = decltype(param);
        if (Parameters::IsSet<P>()) {
            field = Parameters::Get<P>();
        }
    };
    override_(config.save,             Parameters::AdjointSave{});
    override_(config.mode,             Parameters::AdjointMode{});
    override_(config.saveSystem,       Parameters::AdjointSaveSystem{});
    override_(config.slim,             Parameters::AdjointSlim{});
    override_(config.path,             Parameters::AdjointFile{});
    override_(config.replayTolerance,  Parameters::AdjointReplayTolerance{});
    override_(config.replayAbsTolerance, Parameters::AdjointReplayAbsTolerance{});
    override_(config.residualEvalCheck, Parameters::AdjointResidualEvalCheck{});
    override_(config.gradientMethod,   Parameters::AdjointGradientMethod{});
    override_(config.fdStep,           Parameters::AdjointFdStep{});
    override_(config.endpointMethod,   Parameters::AdjointEndpointMethod{});
    override_(config.permWellTerm,     Parameters::AdjointPermWellTerm{});
    override_(config.endpoints,        Parameters::AdjointEndpoints{});
    override_(config.linearSolver,     Parameters::AdjointLinearSolver{});
    override_(config.linearSolverReduction,
              Parameters::AdjointLinearSolverReduction{});
    override_(config.linearSolverMaxIter,
              Parameters::AdjointLinearSolverMaxIter{});
    override_(config.linearSolverVerbosity,
              Parameters::AdjointLinearSolverVerbosity{});

    // The objective string is overridden by an explicit CLI flag; a JSON
    // objective (structured) is used only when no CLI objective was given.
    if (Parameters::IsSet<Parameters::AdjointObjective>() ||
        !config.objectiveSpec.has_value()) {
        config.objective = Parameters::Get<Parameters::AdjointObjective>();
        if (Parameters::IsSet<Parameters::AdjointObjective>()) {
            config.objectiveSpec.reset();  // CLI wins
        }
    }
    return config;
}

void registerAdjointParameters()
{
    Parameters::Register<Parameters::AdjointConfig>
        ("Path to a JSON configuration file for the adjoint run. It provides "
         "base values (objective with weights/windows/wildcards, solver, "
         "archive, gradient method); explicit --adjoint-* flags override it");
    Parameters::Register<Parameters::AdjointSave>
        ("Record per-substep simulation state for adjoint/replay runs");
    Parameters::Register<Parameters::AdjointMode>
        ("Adjoint run mode: empty for a normal forward run, 'replay' to "
         "walk backwards over a recorded forward run and verify that its "
         "linearizations are reproduced");
    Parameters::Register<Parameters::AdjointSaveSystem>
        ("Also store the converged residual and Jacobian of every accepted "
         "substep in the adjoint archive (verification data for replay tests)");
    Parameters::Register<Parameters::AdjointSlim>
        ("Store slim per-substep snapshots (dynamic state only, skip the static "
         "Eclipse schedule reconstructed at replay); much smaller archives, off "
         "by default; not for runtime schedule-modifying (ACTIONX) decks");
    Parameters::Register<Parameters::AdjointFile>
        ("Path of the adjoint archive. Empty means <output-dir>/<CASENAME>.ADJOINT[.h5]. "
         "A path ending in .h5/.hdf5 selects the HDF5 backend, anything else "
         "a plain directory store");
    Parameters::Register<Parameters::AdjointReplayTolerance>
        ("Relative tolerance for comparing re-linearized systems against "
         "stored ones in adjoint replay");
    Parameters::Register<Parameters::AdjointLinearSolver>
        ("Linear solver for the transposed adjoint systems: umfpack "
         "(direct), ilu0, cpr, cprt, or a FlexibleSolver *.json file");
    Parameters::Register<Parameters::AdjointLinearSolverReduction>
        ("Relative residual reduction for iterative adjoint solves");
    Parameters::Register<Parameters::AdjointLinearSolverMaxIter>
        ("Maximum iterations for iterative adjoint solves");
    Parameters::Register<Parameters::AdjointLinearSolverVerbosity>
        ("Verbosity of the adjoint linear solver");
    Parameters::Register<Parameters::AdjointEndpoints>
        ("Comma-separated saturation end-point parameters for adjoint "
         "gradients (SWL, SWCR, KRW, ...); empty disables");
    Parameters::Register<Parameters::AdjointObjective>
        ("Objective function for adjoint gradient runs: 'pressure-average' "
         "or 'bhp:<WELL>' (time-integrated bottom-hole pressure)");
    Parameters::Register<Parameters::AdjointReplayAbsTolerance>
        ("Absolute noise floor for the adjoint replay comparison; entry "
         "differences below this value count as rounding noise");
    Parameters::Register<Parameters::AdjointResidualEvalCheck>
        ("During replay, also recompute the residual with the standalone "
         "AdjointResidualEvaluator and check it matches the linearizer "
         "(validation of the residual-only evaluator)");
    Parameters::Register<Parameters::AdjointGradientMethod>
        ("How PV/trans/perm gradients are accumulated: 'analytic' (closed "
         "forms, default), 'general' (stencil-local re-evaluation via the "
         "residual evaluator, finite difference), or 'general-ad' (the same "
         "stencil-local mechanism with seeded AD, exact dR/dθ in one pass)");
    Parameters::Register<Parameters::AdjointFdStep>
        ("Central-difference step for the finite-difference gradient paths "
         "('general' multiplier perturbations and end-point gradients); "
         "the exact 'general-ad' path ignores it");
    Parameters::Register<Parameters::AdjointEndpointMethod>
        ("End-point gradient evaluation: 'whole' (full re-linearization per "
         "perturbation, default) or 'stencil' (stencil-local re-evaluation, "
         "~Ncells/7 x faster, whole-domain fallback for perforated cells)");
    Parameters::Register<Parameters::AdjointPermWellTerm>
        ("Include the dJ/dperm well-coupling term (dCTF/dK) at defaulted-CTF "
         "perforations; default true, a no-op for explicit-CTF decks");
}

} // namespace Opm
