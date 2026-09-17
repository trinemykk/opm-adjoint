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
 * \brief flow_adjoint: blackoil simulator with adjoint state recording and
 *        backward replay, built as a downstream module against unmodified
 *        opm-simulators.
 *
 * Workflow (single binary):
 *   1. forward + record:
 *        flow_adjoint CASE.DATA --adjoint-save=true --adjoint-save-system=true \
 *            --enable-storage-cache=false --threads-per-process=1
 *   2. backward replay + verification:
 *        flow_adjoint CASE.DATA --adjoint-mode=replay \
 *            --enable-storage-cache=false --threads-per-process=1
 *
 * The TypeTag mirrors the production blackoil TPFA setup
 * (flow/flow_blackoil.cpp) and adds the adjoint problem/well-model
 * extensions via property overrides.
 */
#include "config.h"

#include <opm/material/common/ResetLocale.hpp>
#include <opm/grid/CpGrid.hpp>
#include <opm/simulators/flow/FlowMain.hpp>
#include <opm/simulators/flow/Main.hpp>
#include <opm/simulators/flow/TTagFlowProblemTPFA.hpp>

#include <opm/models/blackoil/blackoilconvectivemixingmodule.hh>
#include <opm/models/blackoil/blackoillocalresidualtpfa.hh>
#include <opm/models/discretization/common/tpfalinearizer.hh>

#include <opm/adjoint/AdjointConfigFile.hpp>
#include <opm/adjoint/AdjointFlowProblem.hpp>
#include <opm/adjoint/AdjointParameters.hpp>
#include <opm/adjoint/AdjointSegmentSource.hpp>
#include <opm/adjoint/AdjointReplay.hpp>
#include <opm/adjoint/AdjointSolver.hpp>
#include <opm/adjoint/AdjointStorage.hpp>

#include <cstring>
#include <memory>
#include <string>

namespace Opm::Properties {

namespace TTag {
struct FlowAdjointProblem {
    using InheritsFrom = std::tuple<FlowProblemTPFA>;
};
}

// Same discretization choices as the production blackoil simulator
// (flow/flow_blackoil.cpp).
template<class TypeTag>
struct Linearizer<TypeTag, TTag::FlowAdjointProblem>
{ using type = TpfaLinearizer<TypeTag>; };

template<class TypeTag>
struct LocalResidual<TypeTag, TTag::FlowAdjointProblem>
{ using type = BlackOilLocalResidualTPFA<TypeTag>; };

template<class TypeTag>
struct EnableDiffusion<TypeTag, TTag::FlowAdjointProblem>
{ static constexpr bool value = false; };

template<class TypeTag>
struct AvoidElementContext<TypeTag, TTag::FlowAdjointProblem>
{ static constexpr bool value = true; };

// Adjoint extension, hooked in via the property system (no
// opm-simulators changes).
template<class TypeTag>
struct Problem<TypeTag, TTag::FlowAdjointProblem>
{ using type = AdjointFlowProblem<TypeTag>; };

} // namespace Opm::Properties

namespace Opm {

//! \brief Main variant dispatching between the normal forward run (with
//!        recording handled by AdjointFlowProblem) and the backward
//!        replay driver.
class AdjointMain : public Main
{
public:
    using Main::Main;

    template<class TypeTag>
    int runForward()
    {
        int exitCode = EXIT_SUCCESS;
        if (initialize_<TypeTag>(exitCode)) {
            // Mirrors Main::dispatchStatic_.
            this->setupVanguard();
            FlowMain<TypeTag> flowMain(argc_, argv_, outputCout_, outputFiles_);
            exitCode = flowMain.execute();
        }
        return exitCode;
    }

    //! \brief One-stop: forward record + backward sweep (gradients and the
    //!        objective) in a single process, sharing one archive.
    //!
    //! The tape lives in RAM by default — no files, but the whole tape is
    //! resident (~230 B/cell/substep, undeflated), which caps the model size.
    //! `--adjoint-file=<path>` puts it on disk instead: the forward leg streams
    //! snapshots out as it goes, the sweep reopens the archive read-only. That
    //! keeps the resident set at ~forward memory and still costs only one
    //! simulator initialization, which is the field-scale configuration.
    template<class TypeTag>
    int runOneStop()
    {
        int exitCode = EXIT_SUCCESS;
        if (!initialize_<TypeTag>(exitCode)) {
            return exitCode;
        }
        // initialize_ has parsed the parameters, so the archive medium can be
        // decided here — before the forward leg builds its recorder.
        const auto config = AdjointConfig::fromParameters();
        const bool onDisk = !config.path.empty();
        if (config.checkpointEvery > 1) {
            // The sweep would need a segment-recompute driver for the substeps
            // between checkpoints; only runReplay installs one. Refuse up front
            // instead of failing mid-sweep as a missing dataset.
            OPM_THROW(std::runtime_error,
                      "--adjoint-mode=onestop does not support "
                      "--adjoint-checkpoint-every>1. Record and sweep as two "
                      "runs (--adjoint-save=true, then --adjoint-mode=gradient), "
                      "which drives the segment recompute.");
        }
        auto archive = onDisk
            ? std::make_shared<AdjointArchive>(config.path,
                                               AdjointArchive::Mode::Write,
                                               FlowGenericVanguard::comm())
            : AdjointArchive::makeInMemory();
        {
            // Forward run records into the shared archive.
            AdjointFlowProblem<TypeTag>::oneStopArchive() = archive;
            this->setupVanguard();
            FlowMain<TypeTag> flowMain(argc_, argv_, outputCout_, outputFiles_);
            exitCode = flowMain.execute();
            AdjointFlowProblem<TypeTag>::oneStopArchive() = nullptr;
            if (exitCode != EXIT_SUCCESS) {
                return exitCode;
            }
        }
        // The recorder released its reference in finalizeOutput() and the
        // forward FlowMain is out of scope, so this is the last one.
        if (onDisk) {
            // Reopen read-only: the write handle must be closed first (HDF5
            // holds the file open, and the meta dataset is only complete once
            // the recorder has finalized).
            archive.reset();
            archive = std::make_shared<AdjointArchive>(
                config.path, AdjointArchive::Mode::Read,
                FlowGenericVanguard::comm());
        }
        // Backward sweep: fresh production setup (no time loop), reading the
        // states from the shared archive still held here.
        FlowMain<TypeTag> flowMain(argc_, argv_, outputCout_, outputFiles_);
        const int status = flowMain.executeInitStep();
        if (status != EXIT_SUCCESS) {
            return status;
        }
        AdjointSolver<TypeTag> solver(*flowMain.getSimulatorPtr(), archive);
        return solver.run();
    }

    template<class TypeTag>
    int runReplay(const std::string& mode)
    {
        int exitCode = EXIT_SUCCESS;
        if (initialize_<TypeTag>(exitCode)) {
            this->setupVanguard();
            FlowMain<TypeTag> flowMain(argc_, argv_, outputCout_, outputFiles_);
            // Full production setup without running the time loop.
            const int status = flowMain.executeInitStep();
            if (status != EXIT_SUCCESS) {
                return status;
            }
            if (mode == "replay") {
                AdjointReplay<TypeTag> replay(*flowMain.getSimulatorPtr());
                exitCode = replay.run();
            } else if (mode == "gradient") {
                AdjointSolver<TypeTag> solver(*flowMain.getSimulatorPtr());
                // A checkpointed archive is missing the substeps between
                // checkpoints; the segment source re-runs them on this same
                // flowMain, which is why the driver has to install it (the
                // solver only holds the model Simulator, not the step driver).
                AdjointSegmentSource<TypeTag> segments(flowMain, solver.replay());
                solver.setSegmentProvider(
                    [&segments](int k) { segments.ensureAvailable(k); });
                exitCode = solver.run();
                if (segments.segmentsRecomputed() > 0) {
                    OpmLog::info(fmt::format(
                        "Adjoint checkpoint recompute: {} segments, {} forward "
                        "substeps re-run in {:.3f} s",
                        segments.segmentsRecomputed(),
                        segments.substepsRecomputed(),
                        segments.recomputeSeconds()));
                }
            } else { // "objective"
                AdjointSolver<TypeTag> solver(*flowMain.getSimulatorPtr());
                exitCode = solver.evaluateObjectiveOnly();
            }
        }
        return exitCode;
    }
};

} // namespace Opm

int main(int argc, char** argv)
{
    using TypeTag = Opm::Properties::TTag::FlowAdjointProblem;

    // The run mode must be known before the parameter system is set up,
    // so peek at the raw command line.
    std::string mode;
    std::string configPath;
    for (int i = 1; i < argc; ++i) {
        if (std::strncmp(argv[i], "--adjoint-mode=", 15) == 0) {
            mode = argv[i] + 15;
        } else if (std::strncmp(argv[i], "--adjoint-config=", 17) == 0) {
            configPath = argv[i] + 17;
        }
    }
    // A --adjoint-config JSON may declare the run mode, so a config file can be
    // self-contained (ERT/Everest hand over only the JSON). An explicit
    // --adjoint-mode on the command line always wins. A malformed config is
    // ignored here and reported later by the normal parameter-resolution path.
    if (mode.empty() && !configPath.empty()) {
        try {
            Opm::AdjointConfig cfg;
            Opm::loadAdjointConfigFile(configPath, cfg);
            mode = cfg.mode;
        } catch (...) { /* reported downstream */ }
    }

    // Adjoint-appropriate defaults, injected as command-line arguments so
    // they sit below any value the user passes explicitly (an injected
    // arg only takes effect when the user did not set that option):
    //   - the Schur step (addWellContributions) needs the well coupling
    //     in the matrix sparsity, so default --matrix-add-well-contributions
    //     on (single-perforation decks are unaffected; multi-perforation
    //     decks would otherwise crash without it). The coupled-system
    //     backward solve (--adjoint-linear-solver=system_cprt) keeps the
    //     wells in the system instead of folding them, so it needs the flag
    //     OFF - default it off there;
    //   - with that flag on, flow auto-promotes the forward solver "cpr"
    //     to "cprw" (CPR with the well system in the matrix), which is not
    //     registered in this build; "ilu0" is robust on any problem size
    //     (incl. the 3-cell test deck) and is not promoted, so default the
    //     forward solver to it. The adjoint backward solves use their own
    //     --adjoint-linear-solver and are unaffected.
    // Done via argv injection (not Parameters::SetDefault) because the
    // model/linalg parameters are registered after the problem, so a
    // SetDefault in the problem's registerParameters runs too early.
    std::vector<std::string> injected;
    auto hasOption = [argc, argv](const char* opt) {
        const std::size_t n = std::strlen(opt);
        for (int i = 1; i < argc; ++i) {
            if (std::strncmp(argv[i], opt, n) == 0) {
                return true;
            }
        }
        return false;
    };
    // The coupled-system backward solve wants the plain (unfolded) reservoir
    // sparsity. Detect it here rather than in the problem, because the flag
    // must be decided before the parameters are parsed.
    auto optionValueIs = [argc, argv](const char* opt, const char* value) {
        const std::string prefix = std::string(opt) + "=";
        for (int i = 1; i < argc; ++i) {
            const std::string arg = argv[i];
            if (arg.starts_with(prefix)) {
                return arg.compare(prefix.size(), std::string::npos, value) == 0;
            }
        }
        return false;
    };
    const bool coupledSystemSolve =
        optionValueIs("--adjoint-linear-solver", "system_cprt");
    if (!hasOption("--matrix-add-well-contributions")) {
        injected.emplace_back(coupledSystemSolve
                                  ? "--matrix-add-well-contributions=false"
                                  : "--matrix-add-well-contributions=true");
    }
    if (!hasOption("--linear-solver")) {
        injected.emplace_back("--linear-solver=ilu0");
    }

    std::vector<char*> args(argv, argv + argc);
    for (auto& opt : injected) {
        args.push_back(opt.data());
    }
    int newArgc = static_cast<int>(args.size());
    char** newArgv = args.data();

    auto mainObject = std::make_unique<Opm::AdjointMain>(newArgc, newArgv);
    int exitCode;
    if (mode.empty()) {
        exitCode = mainObject->runForward<TypeTag>();
    } else if (mode == "onestop") {
        exitCode = mainObject->runOneStop<TypeTag>();
    } else {
        exitCode = mainObject->runReplay<TypeTag>(mode);
    }
    // Destruct mainObject as the destructor calls MPI_Finalize!
    mainObject.reset();
    return exitCode;
}
