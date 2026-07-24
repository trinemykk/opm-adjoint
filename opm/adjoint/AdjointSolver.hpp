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
 * \brief Backward adjoint sweep with parameter-gradient accumulation.
 *
 * For each recorded substep k (walked backwards) the replay driver
 * recreates the converged linearization; this class then
 *  1. forms the Schur-reduced system matrix A_red = A - C^T D^-1 B by
 *     adding the well contributions (public BlackoilWellModel API),
 *  2. solves A_red^T lambda_k = -(dJ_k/dx_k)^T - Bdiag_{k+1}^T lambda_{k+1}
 *     with the direct transposed solver,
 *  3. accumulates the parameter gradients
 *       g_pv[i]   += lambda_k[i] . (V_i/dt_k) (S_i(x_k) - S_i(x_{k-1}))
 *       g_tmult[f] += (lambda_I - lambda_J) . flux_f
 *     (storage linear in the pore-volume multiplier; TPFA flux linear in
 *     the transmissibility multiplier; fluxes re-evaluated at the
 *     converged state with the public templated TPFA kernels).
 *
 * v1 restrictions: objective without explicit well-DOF dependence (the
 * well equations are eliminated exactly through the Schur complement),
 * isothermal, no diffusion/dispersion, serial.
 */
#ifndef OPM_ADJOINT_SOLVER_HPP
#define OPM_ADJOINT_SOLVER_HPP

#include <opm/common/ErrorMacros.hpp>
#include <opm/common/OpmLog/OpmLog.hpp>

#include <opm/models/utils/propertysystem.hh>

#include <opm/adjoint/AdjointEndpointGradients.hpp>
#include <opm/adjoint/AdjointLinearSolver.hpp>
#include <opm/adjoint/AdjointObjective.hpp>
#include <opm/adjoint/AdjointParallel.hpp>
#include <opm/adjoint/AdjointDeckWarnings.hpp>
#include <opm/adjoint/AdjointReplay.hpp>
#include <opm/adjoint/AdjointResidualEvaluator.hpp>

#include <opm/input/eclipse/EclipseState/Grid/FaceDir.hpp>

#include <opm/material/densead/Evaluation.hpp>
#include <opm/material/densead/Math.hpp>

#include <opm/simulators/wells/StandardWell.hpp>

#include <opm/input/eclipse/Schedule/Well/Connection.hpp>

#include <opm/input/eclipse/Schedule/Well/WellEnums.hpp>

#include <dune/common/fvector.hh>
#include <dune/istl/bvector.hh>

#include <fmt/format.h>

#include <array>
#include <chrono>
#include <fstream>
#include <map>
#include <string>
#include <type_traits>
#include <utility>
#include <vector>

namespace Opm {

template<class TypeTag>
class AdjointSolver
{
public:
    using Simulator = GetPropType<TypeTag, Properties::Simulator>;
    using Scalar = GetPropType<TypeTag, Properties::Scalar>;
    using Evaluation = GetPropType<TypeTag, Properties::Evaluation>;
    using LocalResidual = GetPropType<TypeTag, Properties::LocalResidual>;
    using FluidSystem = GetPropType<TypeTag, Properties::FluidSystem>;
    using SparseMatrixAdapter = GetPropType<TypeTag, Properties::SparseMatrixAdapter>;

    static constexpr int numEq = getPropValue<TypeTag, Properties::NumEq>();

    using IntensiveQuantities = GetPropType<TypeTag, Properties::IntensiveQuantities>;
    //! \brief One-derivative AD type: the parameter seed for exact, one-pass
    //!        dR/dθ (Step 3). The state is taken as scalar values, so the
    //!        single derivative slot carries only the parameter sensitivity.
    using PEval = DenseAd::Evaluation<Scalar, 1>;
    using Replay = AdjointReplay<TypeTag>;
    using BdiagBlock = typename Replay::BdiagBlock;
    using Matrix = std::remove_cvref_t<
        decltype(std::declval<Simulator&>().model().linearizer().jacobian().istlMatrix())>;
    using Vector = Dune::BlockVector<Dune::FieldVector<Scalar, numEq>>;

    explicit AdjointSolver(Simulator& simulator)
        : AdjointSolver(simulator, nullptr)
    {}

    //! \brief One-stop solver: read the recorded states from a shared
    //!        (in-memory) archive filled by the forward run in this process.
    AdjointSolver(Simulator& simulator,
                  std::shared_ptr<AdjointArchive> injectedArchive)
        : simulator_(simulator)
        , config_(AdjointConfig::fromParameters())
        , replay_(simulator, std::move(injectedArchive))
        , parallel_(simulator)
        , objective_(config_.objectiveSpec
                         ? *config_.objectiveSpec
                         : Adjoint::ObjectiveSpec::parse(config_.objective),
                     simulator.vanguard().schedule())
    {
        replay_.setComputeBdiag(true);
        objective_.setParallel(&parallel_);
        constexpr std::size_t pressureIndex =
            GetPropType<TypeTag, Properties::Indices>::pressureSwitchIdx;
        // umfpack is the serial default but single-rank; fall back to the
        // iterative parallel path (block-ILU0) when running on >1 rank.
        std::string linearSolver = config_.linearSolver;
        if (parallel_.parallel() && linearSolver == "umfpack") {
            linearSolver = "ilu0";
        }
        linearSolver_.configure(linearSolver,
                                config_.linearSolverReduction,
                                config_.linearSolverMaxIter,
                                config_.linearSolverVerbosity,
                                pressureIndex);
#if HAVE_MPI
        if (parallel_.parallel()) {
            linearSolver_.setComm(parallel_.istlComm(),
                                  &parallel_.overlapRows());
        }
#endif
        const std::string endpointSpec = config_.endpoints;
        std::size_t pos = 0;
        while (pos != std::string::npos && pos < endpointSpec.size()) {
            const auto comma = endpointSpec.find(',', pos);
            const std::string name = endpointSpec.substr(
                pos, comma == std::string::npos ? std::string::npos : comma - pos);
            if (!name.empty()) {
                endpointDefs_.push_back(&findEndpointParam<Scalar>(name));
            }
            pos = (comma == std::string::npos) ? std::string::npos : comma + 1;
        }
        if (!endpointDefs_.empty()) {
            OpmLog::info("Adjoint end-point gradients enabled for " +
                         endpointSpec);
        }
        generalAd_ = (config_.gradientMethod == "general-ad");
        generalGradients_ = (config_.gradientMethod == "general") || generalAd_;
        if (generalGradients_) {
            OpmLog::info(fmt::format(
                "Adjoint gradients: general (stencil-local re-evaluation, {})",
                generalAd_ ? "seeded AD, one pass" : "finite difference"));
        }
    }

    //! \brief Evaluate the objective on the recorded final state only.
    int evaluateObjectiveOnly()
    {
        const auto& meta = replay_.meta();
        const int numSubsteps = static_cast<int>(meta.substeps.size());
        if (numSubsteps == 0) {
            OpmLog::error("Adjoint archive contains no substeps");
            return EXIT_FAILURE;
        }
        Scalar value = 0.0;
        objective_.resetTermValues();
        for (int k = 0; k < numSubsteps; ++k) {
            const auto& step = meta.substeps[k];
            replay_.loadSnapshot(k, step.reportStep);
            value += objective_.stepValue(simulator_, k, numSubsteps, step.dt);
        }
        OpmLog::info(fmt::format("Adjoint objective J = {:.16e}", value));
        logTermValues_();
        return EXIT_SUCCESS;
    }

    //! \brief Backward sweep: solve all adjoint systems and accumulate
    //!        the parameter gradients.
    int run()
    {
        const auto& meta = replay_.meta();
        const int numSubsteps = static_cast<int>(meta.substeps.size());
        if (numSubsteps == 0) {
            OpmLog::error("Adjoint archive contains no substeps");
            return EXIT_FAILURE;
        }

        auto& model = simulator_.model();
        const std::size_t numCells = model.numGridDof();

        gradientPv_.assign(numCells, 0.0);
        gradientPermGeneral_.assign(numCells, {0.0, 0.0, 0.0});
        gradientPermWell_.assign(numCells, {0.0, 0.0, 0.0});

        Vector lambda(numCells);
        Vector lambdaPrev(numCells);
        lambdaPrev = 0.0;
        std::vector<BdiagBlock> bdiagPrev;

        Scalar objectiveValue = 0.0;
        objective_.resetTermValues();
        int replayFailures = 0;

        using Clock = std::chrono::steady_clock;
        auto tick = [] { return Clock::now(); };
        auto secs = [](Clock::time_point a, Clock::time_point b) {
            return std::chrono::duration<double>(b - a).count();
        };
        const auto sweepStart = tick();

        for (int k = numSubsteps - 1; k >= 0; --k) {
            double worstResidual = 0.0;
            double worstJacobian = 0.0;
            auto t0 = tick();
            if (!replay_.replayStep(k, worstResidual, worstJacobian)) {
                ++replayFailures;
            }
            timing_.replay += secs(t0, tick());
            // The well container is rebuilt by replayStep; check the Schur
            // sparsity requirement once, now that the wells exist.
            if (k == numSubsteps - 1) {
                checkWellContributionSparsity_();
            }
            const auto& step = meta.substeps[k];

            // Objective contribution of this substep (solution(0) = x_k).
            objectiveValue +=
                objective_.stepValue(simulator_, k, numSubsteps, step.dt);

            // Right-hand side:
            //   -(dJ_k/dx_r)^T + B^T D^-T (dJ_k/dx_w) - Bdiag_{k+1}^T lambda_{k+1}
            // (from stationarity of the Lagrangian in x_w and x_r; the well
            // equations are eliminated through the Schur complement).
            Vector rhs(numCells);
            rhs = 0.0;
            objective_.addStepGradient(simulator_, k, numSubsteps, rhs);
            rhs *= -1.0;
            addWellObjectiveRhs_(step.dt, rhs);
            if (k < numSubsteps - 1) {
                for (std::size_t i = 0; i < numCells; ++i) {
                    // rhs_i -= Bdiag_{k+1,i}^T lambda_{k+1,i}
                    bdiagPrev[i].usmtv(-1.0, lambdaPrev[i], rhs[i]);
                }
            }

            // Schur-reduced matrix: A -= C^T D^-1 B (in place on the
            // linearizer's matrix; it is re-assembled in the next
            // replayStep anyway).
            t0 = tick();
            auto& jacobian = model.linearizer().jacobian();
            simulator_.problem().wellModel().addWellContributions(jacobian);
            timing_.schur += secs(t0, tick());

            // Solve A_red^T lambda_k = rhs.
            t0 = tick();
            linearSolver_.solveTransposed(jacobian.istlMatrix(), rhs, lambda);
            // lambda is owner-correct after the solve; make it consistent
            // on ghost cells so the flux/Bdiag cross terms (which read a
            // neighbor's lambda, possibly a ghost) are correct.
            parallel_.makeConsistent(lambda);
            timing_.solve += secs(t0, tick());

            t0 = tick();
            computeWellAdjoints_(k, step.dt, lambda);
            if (generalGradients_) {
                accumulateGeneralGradients_(step.dt, lambda, generalAd_);
            } else {
                accumulateGradients_(step.dt, lambda);
            }
            accumulatePermWellTerm_(lambda);
            timing_.gradients += secs(t0, tick());
            t0 = tick();
            accumulateEndpointGradients_(step.dt, lambda);
            timing_.endpoints += secs(t0, tick());

            bdiagPrev = replay_.bdiag();
            lambdaPrev = lambda;
        }
        timing_.total = secs(sweepStart, tick());

        OpmLog::info(fmt::format(
            "Adjoint sweep finished: J = {:.16e}, {} substeps, {} replay "
            "verification failures", objectiveValue, numSubsteps, replayFailures));
        logTermValues_();
        if (linearSolver_.totalIterations() > 0) {
            OpmLog::info(fmt::format(
                "Adjoint linear solver '{}': {} iterations over {} solves "
                "(avg {:.1f})", linearSolver_.spec(),
                linearSolver_.totalIterations(), linearSolver_.numSolves(),
                double(linearSolver_.totalIterations()) /
                    std::max(1, linearSolver_.numSolves())));
        }
        reportTiming_(numSubsteps);

        writeResults_(objectiveValue);
        warnAdjointDeckApproximations(simulator_);
        return replayFailures == 0 ? EXIT_SUCCESS : EXIT_FAILURE;
    }

private:
    //! \brief Per-phase wall-clock breakdown of the backward sweep, to
    //!        guide where optimization effort is worthwhile.
    struct Timing
    {
        double replay = 0.0;     //!< re-linearization at the converged point
        double schur = 0.0;      //!< A -= C^T D^-1 B (addWellContributions)
        double solve = 0.0;      //!< transposed linear solve (+ consistency)
        double gradients = 0.0;  //!< well adjoints + PV/trans accumulation
        double endpoints = 0.0;  //!< end-point FD (0 unless --adjoint-endpoints)
        double total = 0.0;      //!< whole sweep
    };

    //! \brief The Schur step (addWellContributions) writes C^T D^-1 B
    //!        into the reservoir matrix, which couples every pair of cells
    //!        a well perforates. Those entries only exist in the matrix
    //!        sparsity when the run was set up with
    //!        --matrix-add-well-contributions=true. Without it, a well
    //!        with >1 perforation writes out of bounds (segfault), so we
    //!        check up front and fail with an actionable message.
    void checkWellContributionSparsity_()
    {
        if (Parameters::Get<Parameters::MatrixAddWellContributions>()) {
            return;
        }
        std::size_t maxPerf = 0;
        for (const auto& wellPtr :
             simulator_.problem().wellModel().localNonshutWells()) {
            maxPerf = std::max(maxPerf, wellPtr->cells().size());
        }
        maxPerf = parallel_.parallel()
                      ? static_cast<std::size_t>(parallel_.max(double(maxPerf)))
                      : maxPerf;
        if (maxPerf > 1) {
            OPM_THROW(std::runtime_error,
                      "Adjoint gradient runs with multi-perforation wells "
                      "need the well-coupling sparsity used by the Schur "
                      "complement, but --matrix-add-well-contributions=false "
                      "was passed explicitly. Remove that override (the "
                      "adjoint binary enables it by default).");
        }
    }

    void reportTiming_(int numSubsteps)
    {
        const double t = timing_.total > 0.0 ? timing_.total : 1.0;
        const double accounted = timing_.replay + timing_.schur +
                                 timing_.solve + timing_.gradients +
                                 timing_.endpoints;
        auto pct = [t](double x) { return 100.0 * x / t; };
        OpmLog::info(fmt::format(
            "Adjoint sweep timing ({} substeps, {:.2f} s total):\n"
            "  replay (re-linearize)   {:8.3f} s  ({:5.1f}%)\n"
            "  schur  (well contrib.)  {:8.3f} s  ({:5.1f}%)\n"
            "  solve  (transposed)     {:8.3f} s  ({:5.1f}%)\n"
            "  gradients (pv/trans/w)  {:8.3f} s  ({:5.1f}%)\n"
            "  endpoints (FD)          {:8.3f} s  ({:5.1f}%)\n"
            "  other                   {:8.3f} s  ({:5.1f}%)",
            numSubsteps, timing_.total,
            timing_.replay, pct(timing_.replay),
            timing_.schur, pct(timing_.schur),
            timing_.solve, pct(timing_.solve),
            timing_.gradients, pct(timing_.gradients),
            timing_.endpoints, pct(timing_.endpoints),
            timing_.total - accounted, pct(timing_.total - accounted)));
    }

    //! \brief dJ_k/dx_w of the objective for one StandardWell
    //!        (empty vector if the well is not involved).
    Dune::DynamicVector<Scalar>
    wellObjectiveGradient_(const StandardWell<TypeTag>& stdWell, double dt)
    {
        using Kind = typename AdjointObjectiveFunction<TypeTag>::Kind;
        const auto& eqns = stdWell.linSys();
        const std::size_t numWellEq = eqns.Dmatrix()[0][0].N();
        Dune::DynamicVector<Scalar> dJdxw(numWellEq, 0.0);

        if (objective_.kind() == Kind::WellBhp) {
            // BHP is the last well primary variable.
            dJdxw[numWellEq - 1] =
                objective_.bhpGradient(stdWell.name(), dt,
                                       simulator_.time() + dt);
        } else if (objective_.kind() == Kind::RateTerms) {
            // q_c is a function of the well primary variables; its
            // derivatives sit in the well slots of the EvalWell returned
            // by getQs (reservoir derivatives first, well derivatives at
            // numEq + j).
            for (const auto& term : objective_.rateTerms()) {
                if (term.well != stdWell.name()) {
                    continue;
                }
                const auto qs = stdWell.primaryVariables().getQs(
                    FluidSystem::canonicalToActiveCompIdx(term.canonicalCompIdx));
                const Scalar weight =
                    objective_.termWeight(simulator_, term, dt, qs.value());
                for (std::size_t j = 0; j < numWellEq; ++j) {
                    dJdxw[j] += weight * qs.derivative(numEq + j);
                }
            }
        }
        return dJdxw;
    }

    //! \brief Add the well-objective coupling B^T D^-T (dJ_k/dx_w) to the
    //!        adjoint right-hand side (StandardWells only).
    void addWellObjectiveRhs_(double dt, Vector& rhs)
    {
        const auto& wellModel = simulator_.problem().wellModel();
        for (const auto& wellPtr : wellModel.localNonshutWells()) {
            if (!objective_.involvesWell(wellPtr->name())) {
                continue;
            }
            const auto* stdWell =
                dynamic_cast<const StandardWell<TypeTag>*>(wellPtr.get());
            if (!stdWell) {
                OPM_THROW(std::runtime_error,
                          "Adjoint well objectives support StandardWells only");
            }
            const auto& eqns = stdWell->linSys();
            const auto& Dmat = eqns.Dmatrix()[0][0];
            const std::size_t numWellEq = Dmat.N();

            const auto dJdxw = wellObjectiveGradient_(*stdWell, dt);

            // y = D^-T dJdxw  (small dense transposed solve).
            Dune::DynamicMatrix<Scalar> Dt(numWellEq, numWellEq);
            for (std::size_t i = 0; i < numWellEq; ++i) {
                for (std::size_t j = 0; j < numWellEq; ++j) {
                    Dt[i][j] = Dmat[j][i];
                }
            }
            Dune::DynamicVector<Scalar> y(numWellEq, 0.0);
            Dt.solve(y, dJdxw);

            // rhs_cell += B[0][perf]^T y for all perforated cells.
            const auto& Bmat = eqns.Bmatrix();
            const auto& cells = stdWell->cells();
            for (auto colIt = Bmat[0].begin(); colIt != Bmat[0].end(); ++colIt) {
                const std::size_t perfIdx = colIt.index();
                const int cellIdx = cells[perfIdx];
                const auto& block = *colIt; // numWellEq x numEq
                for (int eq = 0; eq < numEq; ++eq) {
                    Scalar sum = 0.0;
                    for (std::size_t we = 0; we < numWellEq; ++we) {
                        sum += block[we][eq] * y[we];
                    }
                    rhs[cellIdx][eq] += sum;
                }
            }
        }
    }

    //! \brief Well adjoints of substep k:
    //!        lambda_w = -D^-T (dJ_k/dx_w + sum_perf C[0][perf] lambda_r[cell]),
    //!        from stationarity of the Lagrangian in x_w. Appended to the
    //!        lambda-well log (basis for later well-control gradients).
    void computeWellAdjoints_(int k, double dt, const Vector& lambda)
    {
        lambdaWells_.clear();
        const auto& wellModel = simulator_.problem().wellModel();
        for (const auto& wellPtr : wellModel.localNonshutWells()) {
            const auto* stdWell =
                dynamic_cast<const StandardWell<TypeTag>*>(wellPtr.get());
            if (!stdWell) {
                continue;
            }
            const auto& eqns = stdWell->linSys();
            const auto& Dmat = eqns.Dmatrix()[0][0];
            const std::size_t numWellEq = Dmat.N();

            auto work = wellObjectiveGradient_(*stdWell, dt);

            const auto& Cmat = eqns.Cmatrix();
            const auto& cells = stdWell->cells();
            for (auto colIt = Cmat[0].begin(); colIt != Cmat[0].end(); ++colIt) {
                const std::size_t perfIdx = colIt.index();
                const int cellIdx = cells[perfIdx];
                const auto& block = *colIt; // numWellEq x numEq
                for (std::size_t we = 0; we < numWellEq; ++we) {
                    for (int eq = 0; eq < numEq; ++eq) {
                        work[we] += block[we][eq] * lambda[cellIdx][eq];
                    }
                }
            }

            Dune::DynamicMatrix<Scalar> Dt(numWellEq, numWellEq);
            for (std::size_t i = 0; i < numWellEq; ++i) {
                for (std::size_t j = 0; j < numWellEq; ++j) {
                    Dt[i][j] = Dmat[j][i];
                }
            }
            Dune::DynamicVector<Scalar> lambdaW(numWellEq, 0.0);
            Dt.solve(lambdaW, work);
            lambdaW *= -1.0;
            lambdaWells_[stdWell->name()] = lambdaW;

            std::string line = std::to_string(k) + " " + stdWell->name();
            for (std::size_t j = 0; j < numWellEq; ++j) {
                line += " " + std::to_string(lambdaW[j]);
            }
            wellAdjointLog_.push_back(std::move(line));

            // Control-target gradient: the active control equation is the
            // last well-equation row with the convention
            // R_ctrl = (rate or bhp) - target, so dR_ctrl/d(target) = -1
            // and dJ/d(target) accumulates -lambda_w[last] over the
            // substeps where this control is active.
            const Scalar contribution = -lambdaW[numWellEq - 1];
            controlGradient_[stdWell->name()] += contribution;

            // Per-active-control-mode breakdown: only one control mode is
            // binding on a well at a given substep, but its row is always the
            // last well equation, so -lambda_w[last] is dJ/d(that mode's
            // target). Tagging by the converged active mode lets the one
            // backward sweep accumulate a separate derivative for EVERY
            // control that was active at some step (BHP, ORAT, WRAT, ...).
            const auto& ws = wellModel.wellState().well(stdWell->name());
            const std::string mode = stdWell->isProducer()
                ? WellProducerCMode2String(ws.production_cmode)
                : WellInjectorCMode2String(ws.injection_cmode);
            controlGradientByMode_[stdWell->name()][mode] += contribution;
        }
    }

    //! \brief Analytic parameter gradients (hand-derived closed forms).
    void accumulateGradients_(double dt, const Vector& lambda)
    {
        accumulatePvAnalytic_(dt, lambda);
        accumulateTransAnalytic_(dt, lambda);
    }

    //! \brief General parameter gradients: the same dJ/dθ for all parameters
    //!        of a family in this one backward sweep, obtained by
    //!        term-targeted re-evaluation (AdjointResidualEvaluator) instead
    //!        of hand-derived formulas. Each family evaluates only the
    //!        residual terms its parameter touches: PV the cell's storage,
    //!        trans/perm the affected face fluxes (both directed sides) —
    //!        not whole-cell residuals. \p useAd selects the exact one-slot
    //!        seeded-AD scale (general-ad) or the centered-FD scale pair
    //!        (general); the evaluation is otherwise identical.
    void accumulateGeneralGradients_(double dt, const Vector& lambda,
                                     bool useAd)
    {
        accumulatePvGeneral_(dt, lambda, useAd);
        accumulateTransGeneral_(dt, lambda, useAd);
        accumulatePermGeneral_(dt, lambda, useAd);
    }

    //! \brief A seeded one-derivative scale (value 1, d/dθ = 1).
    static PEval seed_()
    {
        PEval s(Scalar(1));
        s.setDerivative(0, Scalar(1));
        return s;
    }

    //! \brief PV-multiplier gradient, analytic: dR_i/dpvmult_i =
    //!        (V_i/dt)(S_i(x_k) - S_i(x_{k-1})).
    void accumulatePvAnalytic_(double dt, const Vector& lambda)
    {
        auto& model = simulator_.model();
        const std::size_t numCells = model.numGridDof();
        for (std::size_t i = 0; i < numCells; ++i) {
            Dune::FieldVector<Scalar, numEq> storageNew;
            Dune::FieldVector<Scalar, numEq> storageOld;
            LocalResidual::template computeStorage<Scalar>(
                storageNew, model.intensiveQuantities(i, /*timeIdx=*/0));
            LocalResidual::template computeStorage<Scalar>(
                storageOld, model.intensiveQuantities(i, /*timeIdx=*/1));
            const Scalar factor = model.dofTotalVolume(i) / dt;
            for (int eq = 0; eq < numEq; ++eq) {
                gradientPv_[i] += lambda[i][eq] * factor *
                                  (storageNew[eq] - storageOld[eq]);
            }
        }
    }

    //! \brief PV-multiplier gradient, general: the multiplier only scales
    //!        the cell's storage term (it enters neither flux nor source),
    //!        so evaluate exactly that term with the scale seeded (AD) or
    //!        centered (FD) and contract with lambda_i. Diagonal stencil.
    void accumulatePvGeneral_(double /*dt*/, const Vector& lambda, bool useAd)
    {
        const std::size_t numCells = simulator_.model().numGridDof();
        AdjointResidualEvaluator<TypeTag> evaluator(simulator_);
        using EqVec = typename AdjointResidualEvaluator<TypeTag>::EqVector;
        const Scalar h = config_.fdStep; // multiplier is O(1): absolute step
        Dune::FieldVector<PEval, numEq> resAd;
        EqVec resPlus(0.0);
        EqVec resMinus(0.0);
        for (std::size_t i = 0; i < numCells; ++i) {
            const unsigned cell = static_cast<unsigned>(i);
            Scalar g = 0.0;
            if (useAd) {
                evaluator.evalStorageTerm(cell, seed_(), resAd);
                for (int eq = 0; eq < numEq; ++eq) {
                    g += lambda[i][eq] * resAd[eq].derivative(0);
                }
            } else {
                evaluator.evalStorageTerm(cell, Scalar(1) + h, resPlus);
                evaluator.evalStorageTerm(cell, Scalar(1) - h, resMinus);
                for (int eq = 0; eq < numEq; ++eq) {
                    g += lambda[i][eq] * (resPlus[eq] - resMinus[eq])
                       / (2.0 * h);
                }
            }
            gradientPv_[i] += g;
        }
    }

    //! \brief Build the local face list (I < J) and size gradientTrans_ on
    //!        first use. In parallel a face shared with another rank is owned
    //!        by exactly one rank (ownsFace), so the gathered global trans
    //!        gradient counts each face once. Also records, per face, the
    //!        positions of BOTH directed neighbour entries (J in I's row and
    //!        I in J's row) so the targeted face evaluations can address
    //!        them without searching.
    void ensureFaces_()
    {
        if (!gradientTrans_.empty()) {
            return;
        }
        auto& model = simulator_.model();
        const std::size_t numCells = model.numGridDof();
        const auto& neighborInfo = model.linearizer().getNeighborInfo();
        auto findPos = [&neighborInfo](unsigned row, unsigned neighbor) {
            unsigned pos = 0;
            for (const auto& nb : neighborInfo[row]) {
                if (nb.neighbor == neighbor) {
                    return pos;
                }
                ++pos;
            }
            OPM_THROW(std::logic_error,
                      "Adjoint face list: neighbour table is not symmetric");
        };
        for (std::size_t globI = 0; globI < numCells; ++globI) {
            unsigned pos = 0;
            for (const auto& nbInfo : neighborInfo[globI]) {
                if (nbInfo.neighbor > globI) {
                    faces_.push_back({globI, nbInfo.neighbor});
                    faceDirs_.push_back(nbInfo.res_nbinfo.faceDir);
                    ownedFace_.push_back(
                        parallel_.ownsFace(globI, nbInfo.neighbor));
                    facePos_.push_back(
                        {pos, findPos(nbInfo.neighbor,
                                      static_cast<unsigned>(globI))});
                }
                ++pos;
            }
        }
        gradientTrans_.assign(faces_.size(), 0.0);
    }

    //! \brief The two directed neighbour entries of face \p f: first = the
    //!        (I -> J) entry in I's row, second = the (J -> I) entry in J's
    //!        row (I = faces_[f].first < J).
    template<class NeighborTable>
    auto faceEntries_(const NeighborTable& neighborInfo, std::size_t f) const
    {
        const auto globI = faces_[f].first;
        const unsigned globJ = faces_[f].second;
        return std::make_pair(&neighborInfo[globI][facePos_[f][0]],
                              &neighborInfo[globJ][facePos_[f][1]]);
    }

    //! \brief Trans-multiplier gradient, general: the multiplier touches
    //!        exactly the flux of face f in R_I and R_J, so evaluate the two
    //!        directed face terms with the scale seeded (AD) or centered
    //!        (FD) and contract with lambda_I, lambda_J.
    void accumulateTransGeneral_(double /*dt*/, const Vector& lambda,
                                 bool useAd)
    {
        ensureFaces_();
        AdjointResidualEvaluator<TypeTag> evaluator(simulator_);
        using EqVec = typename AdjointResidualEvaluator<TypeTag>::EqVector;
        const auto& neighborInfo =
            simulator_.model().linearizer().getNeighborInfo();
        const Scalar h = config_.fdStep;
        Dune::FieldVector<PEval, numEq> aI, aJ;
        EqVec rIp(0.0), rIm(0.0), rJp(0.0), rJm(0.0);
        for (std::size_t f = 0; f < faces_.size(); ++f) {
            const unsigned globI = static_cast<unsigned>(faces_[f].first);
            const unsigned globJ = faces_[f].second;
            const auto [nbIJ, nbJI] = faceEntries_(neighborInfo, f);
            Scalar g = 0.0;
            if (useAd) {
                const PEval s = seed_();
                evaluator.evalFaceTerm(globI, *nbIJ, s, aI);
                evaluator.evalFaceTerm(globJ, *nbJI, s, aJ);
                for (int eq = 0; eq < numEq; ++eq) {
                    g += lambda[globI][eq] * aI[eq].derivative(0) +
                         lambda[globJ][eq] * aJ[eq].derivative(0);
                }
            } else {
                evaluator.evalFaceTerm(globI, *nbIJ, Scalar(1) + h, rIp);
                evaluator.evalFaceTerm(globI, *nbIJ, Scalar(1) - h, rIm);
                evaluator.evalFaceTerm(globJ, *nbJI, Scalar(1) + h, rJp);
                evaluator.evalFaceTerm(globJ, *nbJI, Scalar(1) - h, rJm);
                for (int eq = 0; eq < numEq; ++eq) {
                    g += (lambda[globI][eq] * (rIp[eq] - rIm[eq]) +
                          lambda[globJ][eq] * (rJp[eq] - rJm[eq])) / (2.0 * h);
                }
            }
            gradientTrans_[f] += g;
        }
    }

    //! \brief A face incident to a cell: its index in faces_, the neighbour
    //!        cell and the permeability direction (X/Y/Z -> 0/1/2; NNCs are
    //!        skipped).
    struct IncidentFace
    {
        std::size_t face;
        unsigned neighbor;
        int dim;
    };

    //! \brief Build, once, the per-cell list of incident faces with their
    //!        permeability direction (from faces_/faceDirs_).
    void ensureCellIncident_()
    {
        if (!cellIncident_.empty()) {
            return;
        }
        ensureFaces_();
        const std::size_t numCells = simulator_.model().numGridDof();
        cellIncident_.assign(numCells, {});
        for (std::size_t f = 0; f < faces_.size(); ++f) {
            int dim;
            switch (faceDirs_[f]) {
            case FaceDir::XPlus:
            case FaceDir::XMinus: dim = 0; break;
            case FaceDir::YPlus:
            case FaceDir::YMinus: dim = 1; break;
            case FaceDir::ZPlus:
            case FaceDir::ZMinus: dim = 2; break;
            default: continue; // NNC: no single permeability direction
            }
            const auto globI = faces_[f].first;
            const auto globJ = faces_[f].second;
            cellIncident_[globI].push_back({f, globJ, dim});
            if (globJ < cellIncident_.size()) {
                cellIncident_[globJ].push_back(
                    {f, static_cast<unsigned>(globI), dim});
            }
        }
    }

    //! \brief Permeability gradient, general: perturbing K_c along one
    //!        direction rescales the (linear-in-K) half-transmissibility of
    //!        cell c on each incident face in that direction, and hence the
    //!        face's harmonic-mean trans by the ratio
    //!        t(s)/t(1) = (s·hc·ho/(s·hc+ho)) / (hc·ho/(hc+ho)).
    //!        Each affected face touches exactly its own flux in R_c and
    //!        R_n, so evaluate the two directed face terms per affected
    //!        face with the ratio seeded (AD, exact through the nonlinear
    //!        harmonic mean) or centered (FD), contract with lambda, and
    //!        recover dJ/dK from the factor derivative by dividing by K.
    void accumulatePermGeneral_(double /*dt*/, const Vector& lambda,
                                bool useAd)
    {
        ensureCellIncident_();
        const auto& trans = simulator_.problem().eclTransmissibilities();
        const std::size_t numCells = simulator_.model().numGridDof();
        AdjointResidualEvaluator<TypeTag> evaluator(simulator_);
        using EqVec = typename AdjointResidualEvaluator<TypeTag>::EqVector;
        const auto& neighborInfo =
            simulator_.model().linearizer().getNeighborInfo();
        const Scalar h = config_.fdStep;
        Dune::FieldVector<PEval, numEq> aC, aN;
        EqVec rCp(0.0), rCm(0.0), rNp(0.0), rNm(0.0);

        for (std::size_t c = 0; c < numCells; ++c) {
            if (!parallel_.interior(c)) {
                continue;
            }
            for (int dim = 0; dim < 3; ++dim) {
                const Scalar kc = trans.permeability(c)[dim][dim];
                if (kc <= 0.0) {
                    continue;
                }
                Scalar gd = 0.0;
                bool any = false;
                for (const auto& inc : cellIncident_[c]) {
                    if (inc.dim != dim) {
                        continue;
                    }
                    const Scalar hc = trans.halfTransmissibility(
                        static_cast<unsigned>(c), inc.neighbor);
                    const Scalar ho = trans.halfTransmissibility(
                        inc.neighbor, static_cast<unsigned>(c));
                    if (hc <= 0.0 || hc + ho <= 0.0) {
                        continue;
                    }
                    any = true;
                    // Directed entries of this face: which side is c?
                    const auto [nbFirst, nbSecond] =
                        faceEntries_(neighborInfo, inc.face);
                    const bool cIsFirst = (faces_[inc.face].first == c);
                    const auto* nbC = cIsFirst ? nbFirst : nbSecond;
                    const auto* nbN = cIsFirst ? nbSecond : nbFirst;
                    const Scalar tOld = hc * ho / (hc + ho);
                    if (useAd) {
                        const PEval s = seed_();
                        const PEval ratio =
                            ((s * hc) * ho / (s * hc + ho)) / tOld;
                        evaluator.evalFaceTerm(static_cast<unsigned>(c),
                                               *nbC, ratio, aC);
                        evaluator.evalFaceTerm(inc.neighbor, *nbN, ratio, aN);
                        for (int eq = 0; eq < numEq; ++eq) {
                            gd += lambda[c][eq] * aC[eq].derivative(0) +
                                  lambda[inc.neighbor][eq] *
                                      aN[eq].derivative(0);
                        }
                    } else {
                        auto scaleOf = [&](Scalar s) {
                            return ((s * hc) * ho / (s * hc + ho)) / tOld;
                        };
                        const Scalar sp = scaleOf(Scalar(1) + h);
                        const Scalar sm = scaleOf(Scalar(1) - h);
                        evaluator.evalFaceTerm(static_cast<unsigned>(c),
                                               *nbC, sp, rCp);
                        evaluator.evalFaceTerm(static_cast<unsigned>(c),
                                               *nbC, sm, rCm);
                        evaluator.evalFaceTerm(inc.neighbor, *nbN, sp, rNp);
                        evaluator.evalFaceTerm(inc.neighbor, *nbN, sm, rNm);
                        for (int eq = 0; eq < numEq; ++eq) {
                            gd += (lambda[c][eq] * (rCp[eq] - rCm[eq]) +
                                   lambda[inc.neighbor][eq] *
                                       (rNp[eq] - rNm[eq])) / (2.0 * h);
                        }
                    }
                }
                if (!any) {
                    continue;
                }
                // factor perturbation -> dJ/dK by dividing by K.
                gradientPermGeneral_[c][dim] += gd / kc;
            }
        }
    }

    //! \brief Transmissibility-multiplier gradient, analytic.
    void accumulateTransAnalytic_(double /*dt*/, const Vector& lambda)
    {
        auto& model = simulator_.model();
        const std::size_t numCells = model.numGridDof();

        // Transmissibility-multiplier gradient from the flux term:
        // dR_I/dtmult_f = flux_f (TPFA flux linear in trans), so
        // g_tmult[f] += (lambda_I - lambda_J) . flux_f. Fluxes are
        // re-evaluated at the converged state with the public TPFA
        // kernels, mirroring linearize_cell.
        const auto& neighborInfo = model.linearizer().getNeighborInfo();
        ensureFaces_();

        std::size_t faceIdx = 0;
        for (std::size_t globI = 0; globI < numCells; ++globI) {
            const auto& nbInfos = neighborInfo[globI];
            const auto& intQuantsIn = model.intensiveQuantities(globI, /*timeIdx=*/0);
            for (const auto& nbInfo : nbInfos) {
                const unsigned globJ = nbInfo.neighbor;
                if (globJ <= globI) {
                    continue;
                }
                // Computed for all local faces; output filters to owned
                // faces (TRANS) and interior endpoints (PERM).
                const auto& intQuantsEx = model.intensiveQuantities(globJ, /*timeIdx=*/0);
                Dune::FieldVector<Evaluation, numEq> flux(0.0);
                Dune::FieldVector<Evaluation, numEq> darcy(0.0);
                LocalResidual::computeFlux(flux, darcy, globI, globJ,
                                           intQuantsIn, intQuantsEx,
                                           nbInfo.res_nbinfo,
                                           simulator_.problem().moduleParams());
                for (int eq = 0; eq < numEq; ++eq) {
                    const Scalar fluxValue =
                        flux[eq].value() * nbInfo.res_nbinfo.faceArea;
                    gradientTrans_[faceIdx] +=
                        (lambda[globI][eq] - lambda[globJ][eq]) * fluxValue;
                }
                ++faceIdx;
            }
        }
    }

    //! \brief End-point scaling gradients by per-cell parameter
    //!        differences of the FULL residual (see
    //!        AdjointEndpointGradients.hpp for the perturbation mechanism).
    //!
    //! Unlike pore volume and transmissibility, the end points change the
    //! relative permeabilities and capillary pressure, which enter the
    //! perforation rates and the well equations as well as storage and
    //! flux. The perturbed residual is therefore obtained by re-running
    //! the converged-point linearization (well sources flow through
    //! problem.source), and the contraction includes the well rows:
    //!   dJ/dtheta_i += lambda_r . dR_r/dtheta_i + sum_w lambda_w . dR_w/dtheta_i
    void accumulateEndpointGradients_(double dt, const Vector& lambda)
    {
        if (endpointDefs_.empty()) {
            return;
        }
        if (gradientEndpoints_.empty()) {
            gradientEndpoints_.assign(
                endpointDefs_.size(),
                std::vector<Scalar>(simulator_.model().numGridDof(), 0.0));
        }
        if (config_.endpointMethod == "stencil") {
            accumulateEndpointGradientsStencil_(dt, lambda);
        } else {
            accumulateEndpointGradientsWhole_(dt, lambda);
        }
    }

    //! \brief Whole-domain perturbed contraction
    //!        lambda_r . R_r + sum_w lambda_w . R_w at the CURRENT
    //!        material-law state: full re-linearization (well sources flow
    //!        through problem.source, well equations re-assembled in
    //!        beginIteration), then contract. Used by the 'whole' endpoint
    //!        method for every cell and by the 'stencil' method as the
    //!        fallback for perforated cells (whose perturbation reaches the
    //!        well equations).
    Scalar endpointWholeContraction_(const Vector& lambda)
    {
        auto& model = simulator_.model();
        auto& problem = simulator_.problem();
        const std::size_t numCells = model.numGridDof();

        // both time levels: Pc/kr changes affect the new-time fluxes
        // and sources AND the old-time storage (phase pressure -> invB)
        model.invalidateAndUpdateIntensiveQuantities(/*timeIdx=*/0);
        model.invalidateAndUpdateIntensiveQuantities(/*timeIdx=*/1);
        problem.beginIteration();
        model.linearizer().linearizeDomain();
        problem.endIteration();

        Scalar value = 0.0;
        const auto& res = model.linearizer().residual();
        for (std::size_t i = 0; i < numCells; ++i) {
            for (int eq = 0; eq < numEq; ++eq) {
                value += lambda[i][eq] * res[i][eq];
            }
        }
        for (const auto& wellPtr : problem.wellModel().localNonshutWells()) {
            const auto* stdWell =
                dynamic_cast<const StandardWell<TypeTag>*>(wellPtr.get());
            if (!stdWell) {
                continue;
            }
            const auto lwIt = lambdaWells_.find(stdWell->name());
            if (lwIt == lambdaWells_.end()) {
                continue;
            }
            const auto& resWell = stdWell->linSys().residual();
            const auto& lw = lwIt->second;
            for (std::size_t j = 0; j < lw.size(); ++j) {
                value += lw[j] * resWell[0][j];
            }
        }
        return value;
    }

    //! \brief Restore the model to the unperturbed converged-linearization
    //!        state after whole-domain endpoint perturbations.
    void restoreConvergedLinearization_()
    {
        auto& model = simulator_.model();
        auto& problem = simulator_.problem();
        model.invalidateAndUpdateIntensiveQuantities(/*timeIdx=*/0);
        model.invalidateAndUpdateIntensiveQuantities(/*timeIdx=*/1);
        problem.beginIteration();
        model.linearizer().linearizeDomain();
        problem.endIteration();
    }

    //! \brief Original whole-domain endpoint method: 2 full
    //!        re-linearizations per (parameter, cell) - cell x Ncells work.
    void accumulateEndpointGradientsWhole_(double /*dt*/, const Vector& lambda)
    {
        auto& problem = simulator_.problem();
        const std::size_t numCells = simulator_.model().numGridDof();
        EndpointPerturber<TypeTag> perturber(problem);

        const Scalar h = config_.fdStep;
        for (std::size_t p = 0; p < endpointDefs_.size(); ++p) {
            const auto& def = *endpointDefs_[p];
            for (unsigned i = 0; i < numCells; ++i) {
                // Each interior cell is perturbed on exactly one rank; the
                // perturbation is rank-local, so the contraction (summed
                // over all local rows, incl. ghosts that share a face with
                // i) captures the full effect with no global reduction.
                if (!parallel_.interior(i)) {
                    continue;
                }
                perturber.apply(i, def, h);
                const Scalar plus = endpointWholeContraction_(lambda);
                perturber.apply(i, def, -h);
                const Scalar minus = endpointWholeContraction_(lambda);
                perturber.restore(i, def);

                gradientEndpoints_[p][i] += (plus - minus) / (2.0 * h);
            }
        }

        restoreConvergedLinearization_();
    }

    //! \brief Local cells perforated by any (standard) well.
    std::vector<bool> perforatedCells_() const
    {
        std::vector<bool> perforated(simulator_.model().numGridDof(), false);
        const auto& wellModel = simulator_.problem().wellModel();
        for (const auto& wellPtr : wellModel.localNonshutWells()) {
            const auto* stdWell =
                dynamic_cast<const StandardWell<TypeTag>*>(wellPtr.get());
            if (!stdWell) {
                continue;
            }
            for (const int cell : stdWell->cells()) {
                perforated[cell] = true;
            }
        }
        return perforated;
    }

    //! \brief Well-coupling permeability gradient (dCTF/dperm).
    //!
    //! At a defaulted-CTF perforation the connection transmissibility factor is
    //! perm-derived (Peaceman), so dJ/dperm has a term through the well
    //! equations that the cell-based (reservoir face) perm gradient omits.
    //! Measured on SPE1 to be ~95 % of dJ/dK at the producer perforation.
    //!
    //! Mechanism: the CTF enters the reservoir source and the well residual
    //! linearly (Tw = CTF*trans_mult, cq = -Tw*mob*drawdown). Perturbing the
    //! well index (= CTF) and re-linearizing, the whole-domain contraction
    //! lambda_r.R_r + lambda_w.R_w captures BOTH sides; central-differencing it
    //! gives dC/d(ln CTF) = dJ/d(ln CTF). The chain to perm uses the dominant
    //! Peaceman term Kh = sqrt(K_a K_b) h, so d(ln CTF)/dK_a = 1/(2 K_a) for the
    //! two directions perpendicular to the connection (the weaker r0 dependence
    //! is neglected in this v1). Explicit-CTF connections have dCTF/dK = 0 and
    //! are skipped.
    void accumulatePermWellTerm_(const Vector& lambda)
    {
        if (!config_.permWellTerm) {
            return;
        }
        const auto& trans = simulator_.problem().eclTransmissibilities();
        auto& wellModel = simulator_.problem().wellModel();
        const Scalar h = config_.fdStep;
        bool perturbed = false;

        for (const auto& wellPtr : wellModel.localNonshutWells()) {
            auto* stdWell = dynamic_cast<StandardWell<TypeTag>*>(wellPtr.get());
            if (!stdWell) {
                continue;
            }
            const auto& perfData = stdWell->perforationData();
            const auto& conns = stdWell->wellEcl().getConnections();
            const auto& cells = stdWell->cells();
            auto& wellIndex = stdWell->wellIndex();

            for (std::size_t perf = 0; perf < wellIndex.size(); ++perf) {
                const int cell = cells[perf];
                if (!parallel_.interior(cell)) {
                    continue;
                }
                const auto& conn = conns[perfData[perf].ecl_index];
                if (conn.kind() != Connection::CTFKind::Defaulted) {
                    continue;  // explicit CTF: dCTF/dK = 0
                }
                const Scalar wi0 = wellIndex[perf];
                if (!(wi0 > 0.0)) {
                    continue;
                }

                wellIndex[perf] = wi0 * (Scalar{1} + h);
                const Scalar cplus = endpointWholeContraction_(lambda);
                wellIndex[perf] = wi0 * (Scalar{1} - h);
                const Scalar cminus = endpointWholeContraction_(lambda);
                wellIndex[perf] = wi0;
                perturbed = true;

                const Scalar dCdlnCtf = (cplus - cminus) / (2.0 * h);

                // Directions perpendicular to the connection carry Kh.
                int a, b;
                switch (conn.dir()) {
                case Connection::Direction::X: a = 1; b = 2; break;
                case Connection::Direction::Y: a = 0; b = 2; break;
                default:                       a = 0; b = 1; break;  // Z
                }
                const Scalar ka = trans.permeability(cell)[a][a];
                const Scalar kb = trans.permeability(cell)[b][b];
                if (ka > 0.0) {
                    gradientPermWell_[cell][a] += dCdlnCtf / (2.0 * ka);
                }
                if (kb > 0.0) {
                    gradientPermWell_[cell][b] += dCdlnCtf / (2.0 * kb);
                }
            }
        }

        if (perturbed) {
            restoreConvergedLinearization_();
        }
    }

    //! \brief Add the accumulated dCTF/dperm well term into a local per-cell
    //!        permeability gradient (before the parallel gather).
    void addPermWellTerm_(std::vector<std::array<Scalar, 3>>& gradient) const
    {
        for (std::size_t c = 0; c < gradient.size() &&
                                c < gradientPermWell_.size(); ++c) {
            for (int d = 0; d < 3; ++d) {
                gradient[c][d] += gradientPermWell_[c][d];
            }
        }
    }

    //! \brief IQ source returning a locally-rebuilt (perturbed) IQ for one
    //!        cell and the model cache for every other cell.
    struct EndpointOverrideIQ
    {
        const Simulator& simulator;
        unsigned cell;
        const std::array<IntensiveQuantities, 2>* local;
        const IntensiveQuantities& operator()(unsigned c, unsigned t) const
        {
            return c == cell ? (*local)[t]
                             : simulator.model().intensiveQuantities(c, t);
        }
    };

    //! \brief Stencil-local endpoint method: an end point of cell i only
    //!        changes cell i's relperm/Pc, so the residual changes only in
    //!        {i} u neighbors(i) - cell x ~7 work instead of cell x Ncells.
    //!
    //! Mechanism per (parameter, non-perforated interior cell i): perturb the
    //! material law (EndpointPerturber), rebuild cell i's intensive
    //! quantities at BOTH time levels through the public
    //! IntensiveQuantities::update (the perturbed relperm/Pc flows in; both
    //! levels because Pc shifts phase pressures and hence the old-time
    //! storage), evaluate the residual of i and its neighbours with the
    //! model cache overridden for cell i only, contract with lambda, central
    //! difference. The model's IQ cache is never invalidated, so the
    //! neighbours read the unperturbed converged state throughout.
    //!
    //! Perforated cells fall back to the whole-domain contraction: their
    //! perturbation changes the perforation rates and hence the well
    //! equations (the lambda_w . dR_w term), which the cell evaluator does
    //! not carry. The fallback pass runs AFTER the stencil pass because the
    //! whole-domain re-linearizations leave the model IQ cache perturbed.
    void accumulateEndpointGradientsStencil_(double /*dt*/, const Vector& lambda)
    {
        auto& model = simulator_.model();
        auto& problem = simulator_.problem();
        const std::size_t numCells = model.numGridDof();
        EndpointPerturber<TypeTag> perturber(problem);
        AdjointResidualEvaluator<TypeTag> evaluator(simulator_);
        const auto& neighborInfo = model.linearizer().getNeighborInfo();
        const auto perforated = perforatedCells_();
        const Scalar h = config_.fdStep;

        std::array<IntensiveQuantities, 2> localIQ;
        using EqVec = typename AdjointResidualEvaluator<TypeTag>::EqVector;
        EqVec res(0.0);

        auto stencilContraction = [&](unsigned i) -> Scalar {
            for (unsigned t = 0; t < 2; ++t) {
                localIQ[t].update(problem, model.solution(t)[i], i, t);
            }
            const EndpointOverrideIQ iq{simulator_, i, &localIQ};
            Scalar value = 0.0;
            evaluator.evalCellResidual(i, iq, res);
            for (int eq = 0; eq < numEq; ++eq) {
                value += lambda[i][eq] * res[eq];
            }
            for (const auto& nb : neighborInfo[i]) {
                evaluator.evalCellResidual(nb.neighbor, iq, res);
                for (int eq = 0; eq < numEq; ++eq) {
                    value += lambda[nb.neighbor][eq] * res[eq];
                }
            }
            return value;
        };

        // Pass 1: stencil-local for non-perforated interior cells (the model
        // IQ cache stays pristine - nothing invalidates it here).
        for (std::size_t p = 0; p < endpointDefs_.size(); ++p) {
            const auto& def = *endpointDefs_[p];
            for (unsigned i = 0; i < numCells; ++i) {
                if (!parallel_.interior(i) || perforated[i]) {
                    continue;
                }
                perturber.apply(i, def, h);
                const Scalar plus = stencilContraction(i);
                perturber.apply(i, def, -h);
                const Scalar minus = stencilContraction(i);
                perturber.restore(i, def);
                gradientEndpoints_[p][i] += (plus - minus) / (2.0 * h);
            }
        }

        // Pass 2: whole-domain fallback for perforated interior cells.
        bool ranFallback = false;
        for (std::size_t p = 0; p < endpointDefs_.size(); ++p) {
            const auto& def = *endpointDefs_[p];
            for (unsigned i = 0; i < numCells; ++i) {
                if (!parallel_.interior(i) || !perforated[i]) {
                    continue;
                }
                ranFallback = true;
                perturber.apply(i, def, h);
                const Scalar plus = endpointWholeContraction_(lambda);
                perturber.apply(i, def, -h);
                const Scalar minus = endpointWholeContraction_(lambda);
                perturber.restore(i, def);
                gradientEndpoints_[p][i] += (plus - minus) / (2.0 * h);
            }
        }
        if (ranFallback) {
            restoreConvergedLinearization_();
        }
    }

    //! \brief Log the per-term breakdown of J (rate-family objectives with
    //!        more than one term; single-term J equals the total).
    void logTermValues_() const
    {
        const auto& values = objective_.termObjectiveValues();
        if (values.size() < 2) {
            return;
        }
        const auto& terms = objective_.rateTerms();
        for (std::size_t i = 0; i < values.size(); ++i) {
            OpmLog::info(fmt::format(
                "Adjoint objective term {:>3} {:<30} J_term = {:.16e} "
                "(weight {:g})", i, terms[i].label, values[i],
                terms[i].weight));
        }
    }

    void writeResults_(Scalar objectiveValue)
    {
        const auto& ioConfig = simulator_.vanguard().eclState().getIOConfig();
        const std::string base = ioConfig.getOutputDir() + "/" +
                                 simulator_.vanguard().caseName();

        // Per-cell fields: gather interior cells onto rank 0 (sorted by
        // global id in parallel; identity in serial, so serial output is
        // byte-identical to the pre-MPI behavior).
        const auto pvGathered = parallel_.gatherCellField(gradientPv_);
        // Cartesian ids matching the gathered per-cell order (for the
        // machine-readable output) and per-term objective values reduced
        // across ranks (the well of each term may live on another rank).
        const auto cellIds = parallel_.gatherCellIds();
        std::vector<Scalar> termValues;
        for (const Scalar v : objective_.termObjectiveValues()) {
            termValues.push_back(parallel_.parallel() ? parallel_.sum(v) : v);
        }
        std::vector<std::vector<Scalar>> endpointGathered;
        for (const auto& g : gradientEndpoints_) {
            endpointGathered.push_back(parallel_.gatherCellField(g));
        }
        // Transmissibility faces: owned faces only, with global cell ids.
        std::vector<std::string> transLines;
        for (std::size_t f = 0; f < faces_.size(); ++f) {
            if (!ownedFace_.empty() && !ownedFace_[f]) {
                continue;
            }
            std::ostringstream line;
            line.precision(16);
            line << faceId_(faces_[f].first) << " " << faceId_(faces_[f].second)
                 << " " << gradientTrans_[f];
            transLines.push_back(line.str());
        }
        transLines = parallel_.gatherLines(transLines);
        // Well outputs are naturally partitioned (a well lives on one rank).
        std::vector<std::string> wellCtrlLines;
        for (const auto& [well, value] : controlGradient_) {
            std::ostringstream line;
            line.precision(16);
            line << well << " " << value;
            wellCtrlLines.push_back(line.str());
        }
        wellCtrlLines = parallel_.gatherLines(wellCtrlLines);
        // Per-active-control-mode breakdown: one line per (well, mode), the
        // derivative w.r.t. that mode's target accumulated over the substeps
        // where it was the binding control.
        std::vector<std::string> wellCtrlByModeLines;
        for (const auto& [well, byMode] : controlGradientByMode_) {
            for (const auto& [mode, value] : byMode) {
                std::ostringstream line;
                line.precision(16);
                line << well << " " << mode << " " << value;
                wellCtrlByModeLines.push_back(line.str());
            }
        }
        wellCtrlByModeLines = parallel_.gatherLines(wellCtrlByModeLines);
        const auto lambdaLines = parallel_.gatherLines(wellAdjointLog_);

        // Permeability: the general method accumulates dJ/dK directly per
        // substep (gradientPermGeneral_); the analytic method post-processes
        // the accumulated trans gradient through the half-trans chain rule.
        std::vector<std::array<Scalar, 3>> permLocal = generalGradients_
            ? gradientPermGeneral_
            : analyticPermLocalGradient_();
        addPermWellTerm_(permLocal);  // dCTF/dperm at defaulted-CTF perforations
        const auto permGathered = gatherPermArray_(permLocal);

        if (parallel_.rank() != 0) {
            return;
        }

        {
            std::ofstream os(base + ".ADJOINT_GRADIENTS_PV.txt");
            os.precision(16);
            os << "# dJ/d(pvmult_i), one line per cell (J = " << objectiveValue
               << ")\n";
            for (const auto& value : pvGathered) {
                os << value << "\n";
            }
        }
        {
            std::ofstream os(base + ".ADJOINT_GRADIENTS_TRANS.txt");
            os.precision(16);
            os << "# dJ/d(tmult_f): cellI cellJ gradient (J = " << objectiveValue
               << ")\n";
            for (const auto& line : transLines) {
                os << line << "\n";
            }
        }
        {
            std::ofstream os(base + ".ADJOINT_GRADIENTS_PERM.txt");
            os.precision(16);
            os << "# dJ/dPERMX dJ/dPERMY dJ/dPERMZ per cell, SI (per m^2); "
                  "multiply by 9.869233e-16 for per-mD\n";
            for (const auto& g : permGathered) {
                os << g[0] << " " << g[1] << " " << g[2] << "\n";
            }
        }
        for (std::size_t p = 0; p < endpointDefs_.size(); ++p) {
            std::ofstream os(base + ".ADJOINT_GRADIENTS_ENDPOINT_" +
                             endpointDefs_[p]->name + ".txt");
            os.precision(16);
            os << "# dJ/d(" << endpointDefs_[p]->name
               << "_i), one line per cell\n";
            for (const auto& value : endpointGathered[p]) {
                os << value << "\n";
            }
        }
        {
            std::ofstream os(base + ".ADJOINT_GRADIENTS_WELLCTRL.txt");
            os.precision(16);
            os << "# dJ/d(active control target) per well, summed over all "
                  "substeps; SI units (rate targets: per m3/s, bhp targets: "
                  "per Pa)\n"
                  "# DEPRECATED for wells that switch control mode during the "
                  "run: this total then sums derivatives w.r.t. DIFFERENT "
                  "targets (e.g. a BHP and an ORAT target) into one number. "
                  "Use ADJOINT_GRADIENTS_WELLCTRL_BYMODE.txt instead.\n";
            for (const auto& line : wellCtrlLines) {
                os << line << "\n";
            }
        }
        {
            // Derivatives w.r.t. ALL controls that were active at some step,
            // from the one backward sweep: dJ/d(target) per (well, mode),
            // accumulated over the substeps where that mode was binding.
            std::ofstream os(base + ".ADJOINT_GRADIENTS_WELLCTRL_BYMODE.txt");
            os.precision(16);
            os << "# well mode dJ/d(target): derivative w.r.t. each active "
                  "control mode's target, accumulated over the substeps where "
                  "it was binding; SI units (rate: per m3/s, bhp: per Pa)\n"
                  "# NOTE: mode GRUP means the well followed a group-derived "
                  "target; the value is then the derivative w.r.t. that "
                  "EFFECTIVE per-well target, not w.r.t. the group keyword "
                  "item.\n";
            for (const auto& line : wellCtrlByModeLines) {
                os << line << "\n";
            }
        }
        {
            std::ofstream os(base + ".ADJOINT_LAMBDA_WELLS.txt");
            os << "# substep well lambda_w (one line per well per substep, "
                  "backward order)\n";
            for (const auto& line : lambdaLines) {
                os << line << "\n";
            }
        }
        std::string kinds = "PV,TRANS,PERM,WELLCTRL,WELLCTRL_BYMODE";
        for (const auto* def : endpointDefs_) {
            kinds += ",ENDPOINT_" + def->name;
        }
        OpmLog::info("Adjoint gradients written to " + base +
                     ".ADJOINT_GRADIENTS_{" + kinds + "}.txt");

        // Machine-readable, cartesian-keyed output for optimizers (ERT/
        // Everest): CSV columns + a JSON metadata sidecar.
        writeFieldOutput_(base, objectiveValue, cellIds, pvGathered,
                          permGathered, transLines, wellCtrlByModeLines,
                          termValues);
    }

    //! \brief Machine-readable gradient output for history-matching drivers.
    //!        CSV files keyed by cartesian cell index plus a JSON sidecar
    //!        carrying the parameter basis, units, objective value and
    //!        per-term breakdown. Rank 0 only. The human-readable .txt files
    //!        stay; this is the stable, documented interface.
    void writeFieldOutput_(const std::string& base,
                           Scalar objectiveValue,
                           const std::vector<long>& cellIds,
                           const std::vector<Scalar>& pv,
                           const std::vector<std::array<Scalar, 3>>& perm,
                           const std::vector<std::string>& transLines,
                           const std::vector<std::string>& wellByModeLines,
                           const std::vector<Scalar>& termValues)
    {
        const std::string caseName = base.substr(base.find_last_of('/') + 1);
        const Scalar mdToSi = 9.869233e-16;

        const std::string poroCsv = base + ".ADJOINT_GRADIENTS_PORO.csv";
        {
            std::ofstream os(poroCsv);
            os.precision(16);
            os << "cartesian_index,dJ_dpvmult\n";
            for (std::size_t i = 0; i < pv.size(); ++i) {
                os << cellIds[i] << "," << pv[i] << "\n";
            }
        }
        const std::string permCsv = base + ".ADJOINT_GRADIENTS_PERM.csv";
        {
            std::ofstream os(permCsv);
            os.precision(16);
            os << "cartesian_index,dJ_dPERMX,dJ_dPERMY,dJ_dPERMZ\n";
            for (std::size_t i = 0; i < perm.size(); ++i) {
                os << cellIds[i] << "," << perm[i][0] << "," << perm[i][1]
                   << "," << perm[i][2] << "\n";
            }
        }
        const std::string transCsv = base + ".ADJOINT_GRADIENTS_TRANS.csv";
        {
            std::ofstream os(transCsv);
            os.precision(16);
            os << "cartesian_i,cartesian_j,dJ_dtmult\n";
            for (const auto& line : transLines) {
                std::istringstream in(line);
                long i, j; Scalar g; in >> i >> j >> g;
                os << i << "," << j << "," << g << "\n";
            }
        }
        const std::string wcCsv = base + ".ADJOINT_GRADIENTS_WELLCTRL.csv";
        {
            std::ofstream os(wcCsv);
            os.precision(16);
            os << "well,mode,dJ_dtarget\n";
            for (const auto& line : wellByModeLines) {
                std::istringstream in(line);
                std::string well, mode, val; in >> well >> mode >> val;
                os << well << "," << mode << "," << val << "\n";
            }
        }

        // JSON metadata sidecar (hand-written; small and stable).
        std::ofstream js(base + ".ADJOINT_GRADIENTS.json");
        js.precision(16);
        const auto rel = [&](const std::string& p) {
            return p.substr(p.find_last_of('/') + 1);
        };
        js << "{\n"
           << "  \"schema_version\": 1,\n"
           << "  \"case\": \"" << caseName << "\",\n"
           << "  \"convention\": \"minimize J; gradient is dJ/d(parameter)\",\n"
           << "  \"objective\": { \"value\": " << objectiveValue;
        if (!termValues.empty()) {
            const auto& terms = objective_.rateTerms();
            js << ", \"terms\": [";
            for (std::size_t t = 0; t < termValues.size(); ++t) {
                js << (t ? ", " : "") << "{ \"label\": \"" << terms[t].label
                   << "\", \"value\": " << termValues[t]
                   << ", \"weight\": " << terms[t].weight << " }";
            }
            js << "]";
        }
        js << " },\n"
           << "  \"families\": [\n"
           << "    { \"name\": \"poro\", \"file\": \"" << rel(poroCsv)
           << "\", \"key\": \"cartesian_index\", \"basis\": \"multiplier\", "
              "\"note\": \"dJ/d(pvmult) at pvmult=1; dJ/dPORO = value / poro\", "
              "\"count\": " << pv.size() << " },\n"
           << "    { \"name\": \"perm\", \"file\": \"" << rel(permCsv)
           << "\", \"key\": \"cartesian_index\", \"basis\": \"value\", "
              "\"units\": \"1/m^2 (SI)\", \"mD_to_SI\": " << mdToSi
           << ", \"count\": " << perm.size() << " },\n"
           << "    { \"name\": \"trans\", \"file\": \"" << rel(transCsv)
           << "\", \"key\": \"cartesian_i,cartesian_j\", \"basis\": "
              "\"multiplier\", \"note\": \"dJ/d(tmult) per face at tmult=1\" },\n"
           << "    { \"name\": \"wellctrl\", \"file\": \"" << rel(wcCsv)
           << "\", \"key\": \"well,mode\", \"basis\": \"value\", "
              "\"units\": \"per m3/s (rate) or per Pa (bhp)\" }\n"
           << "  ]\n"
           << "}\n";

        OpmLog::info("Adjoint field output: " + base +
                     ".ADJOINT_GRADIENTS_{PORO,PERM,TRANS,WELLCTRL}.csv + .json");
    }

    //! \brief Global id for a transmissibility-face endpoint (cartesian
    //!        index in parallel; local index in serial so the serial
    //!        output stays byte-identical to the pre-MPI behavior).
    long faceId_(std::size_t localCell) const
    {
        // Always the cartesian (global) index — in serial too, so the
        // face/cell ids in the output are a stable grid identifier an
        // optimizer can map to, not a partition-dependent local index.
        return parallel_.cartesianIndex(localCell);
    }

    //! \brief Permeability chain rule, post-processed from the
    //!        transmissibility-multiplier gradients:
    //!        dJ/dK_in^dir = sum_faces g_tmult[f] (h_out/(h_in+h_out)) / K_in^dir
    //!        (T proportional to the harmonic mean of the one-sided half
    //!        transmissibilities, which are linear in the directional
    //!        permeability; multipliers cancel since g_tmult is the
    //!        multiplier-based gradient).
    //!
    //! In parallel each interior cell collects the contributions of all
    //! its incident local faces (the face's flux gradient is the same on
    //! both sides since lambda is made consistent), so the gathered
    //! per-cell result is correct without ghost-row reduction; ghost-cell
    //! accumulations are dropped on gather.
    //! \brief Analytic dJ/dK (local, pre-gather): the accumulated trans
    //!        gradient chain-ruled through the one-sided half-transmissibilities.
    std::vector<std::array<Scalar, 3>> analyticPermLocalGradient_()
    {
        const auto& trans = simulator_.problem().eclTransmissibilities();
        const std::size_t numCells = simulator_.model().numGridDof();
        std::vector<std::array<Scalar, 3>> gradient(numCells, {0.0, 0.0, 0.0});

        for (std::size_t f = 0; f < faces_.size(); ++f) {
            const auto [globI, globJ] = faces_[f];
            int dim;
            switch (faceDirs_[f]) {
            case FaceDir::XPlus:
            case FaceDir::XMinus: dim = 0; break;
            case FaceDir::YPlus:
            case FaceDir::YMinus: dim = 1; break;
            case FaceDir::ZPlus:
            case FaceDir::ZMinus: dim = 2; break;
            default:
                continue; // NNC: no single permeability direction
            }
            const Scalar hIn = trans.halfTransmissibility(globI, globJ);
            const Scalar hOut = trans.halfTransmissibility(globJ, globI);
            if (hIn + hOut <= 0.0) {
                continue;
            }
            const Scalar kIn = trans.permeability(globI)[dim][dim];
            const Scalar kOut = trans.permeability(globJ)[dim][dim];
            const Scalar g = gradientTrans_[f];
            // Add to each endpoint only if it is interior on this rank;
            // a ghost endpoint's contribution is added on its owner.
            if (kIn > 0.0 && parallel_.interior(globI)) {
                gradient[globI][dim] += g * (hOut / (hIn + hOut)) / kIn;
            }
            if (kOut > 0.0 && parallel_.interior(globJ)) {
                gradient[globJ][dim] += g * (hIn / (hIn + hOut)) / kOut;
            }
        }

        return gradient;
    }

    //! \brief Gather a per-cell 3-vector field (one column per permeability
    //!        direction) onto rank 0.
    std::vector<std::array<Scalar, 3>>
    gatherPermArray_(const std::vector<std::array<Scalar, 3>>& gradient)
    {
        const std::size_t numCells = gradient.size();
        std::array<std::vector<Scalar>, 3> comp;
        for (int d = 0; d < 3; ++d) {
            std::vector<Scalar> column(numCells);
            for (std::size_t i = 0; i < numCells; ++i) {
                column[i] = gradient[i][d];
            }
            comp[d] = parallel_.gatherCellField(column);
        }
        std::vector<std::array<Scalar, 3>> result(comp[0].size());
        for (std::size_t i = 0; i < result.size(); ++i) {
            result[i] = {comp[0][i], comp[1][i], comp[2][i]};
        }
        return result;
    }

    Simulator& simulator_;
    AdjointConfig config_;
    Replay replay_;
    AdjointParallel<TypeTag> parallel_;
    AdjointObjectiveFunction<TypeTag> objective_;
    AdjointLinearSolver<Matrix, Vector> linearSolver_;
    std::vector<bool> ownedFace_;

    std::vector<Scalar> gradientPv_;
    std::vector<std::pair<std::size_t, unsigned>> faces_;
    std::vector<FaceDir::DirEnum> faceDirs_;
    //! Per face: position of the (I->J) entry in I's neighbour row and of
    //! the (J->I) entry in J's row (I = faces_[f].first).
    std::vector<std::array<unsigned, 2>> facePos_;
    std::vector<Scalar> gradientTrans_;
    std::vector<std::string> wellAdjointLog_;
    std::map<std::string, Scalar> controlGradient_;
    //! well -> active-control-mode -> dJ/d(that mode's target).
    std::map<std::string, std::map<std::string, Scalar>> controlGradientByMode_;
    std::map<std::string, Dune::DynamicVector<Scalar>> lambdaWells_;
    std::vector<const EndpointParamDef<Scalar>*> endpointDefs_;
    std::vector<std::vector<Scalar>> gradientEndpoints_;
    bool generalGradients_{false}; //!< use the stencil-local general mechanism
    bool generalAd_{false};        //!< seeded AD (one pass) instead of FD
    std::vector<std::array<Scalar, 3>> gradientPermGeneral_; //!< general dJ/dK
    std::vector<std::array<Scalar, 3>> gradientPermWell_;    //!< dCTF/dK term
    std::vector<std::vector<IncidentFace>> cellIncident_; //!< per-cell faces+dir
    Timing timing_;
};

} // namespace Opm

#endif // OPM_ADJOINT_SOLVER_HPP
