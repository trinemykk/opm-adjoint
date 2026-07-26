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
 * \brief Linear solver for the per-timestep adjoint systems J^T lambda = rhs.
 *
 * The transpose is always formed explicitly (one structural pass plus a
 * value copy - negligible next to the solve), which makes every existing
 * solver and preconditioner usable unchanged. Two paths behind one
 * interface (--adjoint-linear-solver):
 *
 *  - "umfpack": direct solve, removes one error source while the adjoint
 *    recursion itself is validated (the small-case oracle; "auto" selects it
 *    below a modest cell count and cprt above — see AdjointParameters.hpp);
 *  - iterative (the plan's v1.5): FlexibleSolver on the transposed
 *    matrix, configured by preset name "ilu0" / "cpr" / "cprt" (flow's
 *    own property-tree defaults via setupILU/setupCPR) or a FlexibleSolver
 *    JSON file. "cprt" is CPR built for transposed systems
 *    (PressureTransferPolicy<transpose=true> + quasi-IMPES weights with
 *    transpose=true), i.e. the canonical adjoint preconditioner; "cpr"
 *    on the transposed matrix is kept for comparison. trueimpes weights
 *    need simulator internals and are not supported here - quasiimpes
 *    only.
 *
 * Parallel runs (serial keeps the FlexibleSolver path above): the
 * forward simulation assembles complete ROWS for interior cells, but the
 * adjoint needs complete COLUMNS, and a plain local transpose drops the
 * cross-rank couplings that a column needs (verified: gradients wrong for
 * np>=3). The parallel path builds the EXACT, parallel-consistent
 * transpose M = A^T explicitly (buildConsistentParallelTranspose,
 * parallelTransposeMatrix.hpp): each rank communicates its interior-row
 * blocks that sit in a remotely-owned column to the owner of that column,
 * so M's owned rows equal the serial transpose. M is then an ordinary
 * reservoir-shaped matrix with the standard 1-overlap structure, wrapped
 * in Opm::GhostLastMatrixAdapter and handed to opm's parallel
 * FlexibleSolver - so the SAME preconditioner as serial is honoured,
 * including the transposed CPR 'cprt' (parallel AMG/hypre coarse solve),
 * instead of a block-ILU0 fallback. UMFPACK stays single-rank (parallel
 * auto-selects cprt). The matrix-free TransposedParallelOperator is kept
 * as a correctness oracle (OPM_ADJOINT_VERIFY_TRANSPOSE=1).
 */
#ifndef OPM_ADJOINT_LINEAR_SOLVER_HPP
#define OPM_ADJOINT_LINEAR_SOLVER_HPP

#include <opm/common/ErrorMacros.hpp>
#include <opm/common/OpmLog/OpmLog.hpp>
#include <opm/simulators/linalg/PropertyTree.hpp>

#include <opm/simulators/linalg/FlexibleSolver.hpp>
#include <opm/simulators/linalg/WellOperators.hpp>
#include <opm/simulators/linalg/FlowLinearSolverParameters.hpp>
#include <opm/simulators/linalg/getQuasiImpesWeights.hpp>
#include <opm/simulators/linalg/setupPropertyTree.hpp>

#include <opm/adjoint/transposeMatrix.hpp>
#include <opm/adjoint/parallelTransposeMatrix.hpp>

#if HAVE_SUITESPARSE_UMFPACK
#include <dune/istl/umfpack.hh>
#endif

#include <dune/istl/operators.hh>
#include <dune/istl/solver.hh>
#include <dune/istl/solvers.hh>
#if HAVE_MPI
#include <opm/simulators/linalg/ParallelOverlappingILU0.hpp>
#include <dune/istl/owneroverlapcopy.hh>
#include <dune/istl/schwarz.hh>
#endif

#include <fmt/format.h>

#include <chrono>
#include <cmath>
#include <cstdlib>
#include <functional>
#include <stdexcept>
#include <string>

namespace Opm {

#if HAVE_MPI
//! \brief Exact parallel transposed operator y = A^T x.
//!
//! The forward simulation assembles complete *rows* for interior cells,
//! but the adjoint needs complete *columns* — and the local matrix's
//! overlap (ghost) rows do not carry the off-diagonal entries that
//! complete an interior column. A plain local transpose therefore loses
//! the cross-rank couplings A_{ki} where k is interior on another rank
//! (verified: wrong gradients for np>=3).
//!
//! This operator avoids the lossy transpose: it scatters A^T over the
//! complete interior rows of the FORWARD matrix
//!   y[i] += A[k][i]^T x[k]   for interior rows k, all columns i,
//! which writes partial sums into both interior and overlap entries of
//! y, then accumulates the overlap contributions onto their owners with
//! addOwnerCopyToOwnerCopy. The owner of each interior cell then holds
//! the full column sum. The (lossy) local transpose is kept only as the
//! preconditioner matrix (getmat) — an approximation is fine there since
//! the operator action is exact.
template<class Matrix, class Vector, class Comm>
class TransposedParallelOperator
    : public Dune::AssembledLinearOperator<Matrix, Vector, Vector>
{
public:
    using field_type = typename Vector::field_type;

    TransposedParallelOperator(const Matrix& forward,
                               const Matrix& localTranspose,
                               const Comm& comm)
        : forward_(forward)
        , transpose_(localTranspose)
        , comm_(comm)
        , interiorSize_(computeInteriorSize_(comm))
    {}

    Dune::SolverCategory::Category category() const override
    { return Dune::SolverCategory::overlapping; }

    void apply(const Vector& x, Vector& y) const override
    {
        y = 0;
        scatterTransposed_(x, y);
        // accumulate the cross-rank column contributions onto owners
        // (essential: without it the gradients are wrong for np>=3 -
        // the interior-row scatter alone gives an incomplete column)
        comm_.addOwnerCopyToOwnerCopy(y, y);
        // restrict to interior (ghost rows are owned elsewhere)
        for (std::size_t i = interiorSize_; i < y.size(); ++i) {
            y[i] = 0;
        }
    }

    void applyscaleadd(field_type alpha, const Vector& x, Vector& y) const override
    {
        tmp_.resize(y.size());
        apply(x, tmp_);
        y.axpy(alpha, tmp_);
    }

    const Matrix& getmat() const override
    { return transpose_; }

private:
    void scatterTransposed_(const Vector& x, Vector& y) const
    {
        // NB: the end() check must come first - a rank whose subdomain has
        // no overlap rows (interiorSize_ == numRows) would otherwise
        // evaluate row.index() on the end iterator (UB).
        const auto rowEnd = forward_.end();
        for (auto row = forward_.begin();
             row != rowEnd && row.index() < interiorSize_; ++row) {
            const auto k = row.index();
            for (auto col = (*row).begin(); col != (*row).end(); ++col) {
                // y[col] += A[k][col]^T x[k]
                (*col).umtv(x[k], y[col.index()]);
            }
        }
    }

    static std::size_t computeInteriorSize_(const Comm& comm)
    {
        // Relies on flow's ghost-last cell ordering: owner-attributed local
        // indices form the contiguous range [0, interiorSize).
        constexpr auto owner = Dune::OwnerOverlapCopyAttributeSet::owner;
        std::size_t is = 0;
        std::size_t numOwner = 0;
        const auto& indexSet = comm.indexSet();
        for (auto idx = indexSet.begin(); idx != indexSet.end(); ++idx) {
            if (idx->local().attribute() == owner) {
                is = std::max<std::size_t>(is, idx->local().local());
                ++numOwner;
            }
        }
        if (numOwner > 0 && numOwner != is + 1) {
            OPM_THROW(std::logic_error,
                      "Adjoint transposed operator requires ghost-last cell "
                      "ordering (owner indices contiguous from 0)");
        }
        return numOwner;
    }

    const Matrix& forward_;
    const Matrix& transpose_;
    const Comm& comm_;
    std::size_t interiorSize_;
    mutable Vector tmp_; //!< applyscaleadd scratch (avoid per-call alloc)
};
#endif // HAVE_MPI

//! \brief Solver for transposed systems A^T x = b.
//!
//! \tparam Matrix BCRS matrix with square dense blocks (double scalar).
//! \tparam Vector Block vector matching the matrix block size.
template<class Matrix, class Vector>
class AdjointLinearSolver
{
public:
    //! \brief Select solver and tolerances; must be called before solves.
    //!
    //! \param spec "umfpack", "ilu0", "cpr", "cprt", or a path to a
    //!             FlexibleSolver JSON configuration (*.json).
    //! \param pressureIndex pressure variable index for CPR-type
    //!                      preconditioners (blackoil: 1).
    void configure(const std::string& spec,
                   double reduction,
                   int maxIter,
                   int verbosity,
                   std::size_t pressureIndex)
    {
        spec_ = spec.empty() ? "umfpack" : spec;
        pressureIndex_ = pressureIndex;
        // WS3 cache is per (spec, sparsity); drop any prior solver so a
        // reconfigure rebuilds it. OPM_ADJOINT_REBUILD_PRECOND=1 forces a full
        // rebuild every substep (the debugging escape hatch — equivalent to the
        // pre-WS3 behaviour).
        cachedSerialSolver_.reset();
        cachedOp_.reset();
#if HAVE_MPI
        cachedParSolver_.reset();
        cachedParOp_.reset();
        if (const char* env = std::getenv("OPM_ADJOINT_VERIFY_TRANSPOSE")) {
            verifyTranspose_ = (env[0] == '1' || env[0] == 't' ||
                                env[0] == 'T' || env[0] == 'y');
        }
#endif
        if (const char* env = std::getenv("OPM_ADJOINT_REBUILD_PRECOND")) {
            reusePreconditioner_ = !(env[0] == '1' || env[0] == 't' ||
                                     env[0] == 'T' || env[0] == 'y');
        }
        if (spec_ == "umfpack") {
            return;
        }

        FlowLinearSolverParameters params; // reset() defaults
        params.linear_solver_reduction_ = reduction;
        params.linear_solver_maxiter_ = maxIter;
        params.linear_solver_verbosity_ = verbosity;

        if (spec_.size() > 5 && spec_.substr(spec_.size() - 5) == ".json") {
            prm_ = PropertyTree(spec_);
        } else if (spec_ == "ilu0") {
            prm_ = setupILU(spec_, params);
        } else if (spec_ == "cpr" || spec_ == "cprt") {
            prm_ = setupCPR("cpr_quasiimpes", params);
            prm_.put("preconditioner.type", spec_);
        } else {
            OPM_THROW(std::runtime_error,
                      "Unknown --adjoint-linear-solver '" + spec_ +
                      "' (use umfpack, ilu0, cpr, cprt or a "
                      "FlexibleSolver *.json file)");
        }

        const auto precType =
            prm_.get<std::string>("preconditioner.type", "");
        usesCprWeights_ = (precType == "cpr" || precType == "cprt" ||
                           precType == "cprw" || precType == "cprwt");
        transposedWeights_ = (precType == "cprt" || precType == "cprwt");
        if (usesCprWeights_) {
            const auto weightsType =
                prm_.get<std::string>("preconditioner.weight_type",
                                      "quasiimpes");
            if (weightsType != "quasiimpes") {
                OPM_THROW(std::runtime_error,
                          "Adjoint CPR supports quasiimpes weights only "
                          "(trueimpes needs forward-simulator internals)");
            }
        }
    }

    //! \brief Solve A^T x = b.
    //!
    //! The right-hand side is taken by value because solvers overwrite it.
    void solveTransposed(const Matrix& matrix, Vector rhs, Vector& solution)
    {
        solution.resize(rhs.size());
        solution = 0.0;

        if (spec_ == "umfpack") {
            solveDirect_(matrix, rhs, solution);
        } else {
            solveIterative_(matrix, rhs, solution);
        }
        ++numSolves_;
    }

#if HAVE_MPI
    //! \brief Enable the parallel solve path (ghost-last operator).
    //!
    //! \param comm ISTL communication (owner/overlap/copy)
    //! \param overlapRows local indices of the ghost rows
    void setComm(const Dune::OwnerOverlapCopyCommunication<int, int>* comm,
                 const std::vector<int>* overlapRows)
    {
        comm_ = comm;
        overlapRows_ = overlapRows;
        if (comm_ && spec_ == "umfpack") {
            OPM_THROW(std::runtime_error,
                      "--adjoint-linear-solver=umfpack is a single-rank "
                      "direct solver; parallel adjoint runs use an iterative "
                      "solver (pass any non-umfpack value, e.g. ilu0)");
        }
        // In parallel the operator wraps the exact, communicated transpose
        // M = A^T (buildConsistentParallelTranspose) in a standard
        // GhostLastMatrixAdapter, so the requested preconditioner - including
        // the transposed CPR 'cprt' - is honoured through opm's parallel
        // FlexibleSolver exactly as in serial (see solveParallel_).
        if (comm_) {
            OpmLog::info("Parallel adjoint solve uses the consistent "
                         "transposed operator with preconditioner '" +
                         spec_ + "'.");
        }
    }
#endif

    //! \brief Iterations of the iterative path summed over all solves
    //!        (0 for umfpack).
    int totalIterations() const
    { return totalIterations_; }

    int numSolves() const
    { return numSolves_; }

    const std::string& spec() const
    { return spec_; }

    //! \brief Wall-clock spent building the (iterative) solver/preconditioner
    //!        across all solves — the resetup that hierarchy reuse targets.
    double setupSeconds() const
    { return setupSeconds_; }

    //! \brief Wall-clock spent in the iterative apply (Krylov iterations)
    //!        across all solves.
    double applySeconds() const
    { return applySeconds_; }

private:
    void solveDirect_(const Matrix& matrix, Vector& rhs, Vector& solution)
    {
#if HAVE_SUITESPARSE_UMFPACK
        // Pattern is constant across the sweep: values-only refill after
        // the first solve.
        transposeBlockMatrixInto(matrix, cachedDirectTranspose_);

        Dune::UMFPack<TransposedMatrix<Matrix>> solver(cachedDirectTranspose_,
                                                       /*verbose=*/0);
        Dune::InverseOperatorResult result;
        solver.apply(solution, rhs, result);

        if (!result.converged) {
            OPM_THROW(std::runtime_error,
                      "UMFPACK failed to solve the transposed (adjoint) system");
        }
#else
        static_cast<void>(matrix);
        static_cast<void>(rhs);
        static_cast<void>(solution);
        OPM_THROW(std::runtime_error,
                  "--adjoint-linear-solver=umfpack requires SuiteSparse; "
                  "this build has no SuiteSparse support. Use an iterative "
                  "solver (ilu0, cpr, cprt or a *.json config).");
#endif
    }

    void solveIterative_(const Matrix& matrix, Vector& rhs, Vector& solution)
    {
        Dune::InverseOperatorResult result;
#if HAVE_MPI
        if (comm_) {
            solveParallel_(matrix, rhs, solution, result);
        } else
#endif
        {
            solveSerial_(matrix, rhs, solution, result);
        }

        if (!result.converged) {
            OPM_THROW(std::runtime_error,
                      fmt::format("Adjoint linear solver '{}' did not "
                                  "converge in {} iterations "
                                  "(reduction achieved {:.3e})",
                                  spec_, result.iterations,
                                  result.reduction));
        }
        totalIterations_ += result.iterations;
        OpmLog::debug(fmt::format("adjoint linear solve ({}): {} iterations, "
                                  "reduction {:.3e}",
                                  spec_, result.iterations, result.reduction));
    }

    //! \brief Serial iterative solve via FlexibleSolver on the explicit
    //!        local transpose (supports cpr/cprt/ilu0/json).
    void solveSerial_(const Matrix& matrix, Vector& rhs, Vector& solution,
                      Dune::InverseOperatorResult& result)
    {
        transposeBlockMatrixInto(matrix, cachedTranspose_);

        // WS3: build the solver + preconditioner once, then on later substeps
        // refresh it in place with preconditioner().update() (the same reuse the
        // forward ISTLSolver does between Newton iterations). cachedTranspose_
        // keeps the same sparsity across the sweep and is refilled in place, so
        // the cached operator's matrix reference stays valid. setup/apply are
        // timed separately so the realized saving is visible in the sweep log.
        using Clock = std::chrono::steady_clock;
        const bool rebuild = !reusePreconditioner_ || !cachedSerialSolver_;
        const auto tSetup0 = Clock::now();
        if (rebuild) {
            cachedSerialSolver_.reset();  // free before the operator it references
            cachedOp_ = std::make_unique<SerialOperator>(cachedTranspose_);
            cachedWeights_ = std::function<Vector()>{};
            if (usesCprWeights_) {
                const Matrix& matrixRef = cachedTranspose_;
                const bool transposeFlag = transposedWeights_;
                const std::size_t pIdx = pressureIndex_;
                cachedWeights_ = [&matrixRef, transposeFlag, pIdx]() {
                    return Amg::getQuasiImpesWeights<Matrix, Vector>(
                        matrixRef, pIdx, transposeFlag,
                        /*enable_thread_parallel=*/false);
                };
            }
            cachedSerialSolver_ =
                std::make_unique<Dune::FlexibleSolver<SerialOperator>>(
                    *cachedOp_, prm_, cachedWeights_, pressureIndex_);
        } else {
            // Same sparsity, new values: recompute the preconditioner (numeric
            // part) but keep the hierarchy the constructor built.
            cachedSerialSolver_->preconditioner().update();
        }
        setupSeconds_ += std::chrono::duration<double>(
            Clock::now() - tSetup0).count();
        const auto tApply0 = Clock::now();
        cachedSerialSolver_->apply(solution, rhs, result);
        applySeconds_ += std::chrono::duration<double>(
            Clock::now() - tApply0).count();
    }

#if HAVE_MPI
    //! \brief Parallel iterative solve on the EXACT, parallel-consistent
    //!        transpose M = A^T.
    //!
    //! buildConsistentParallelTranspose communicates the off-rank column
    //! couplings the local transpose drops, so M's owned rows equal the
    //! serial transpose. M is an ordinary reservoir-shaped matrix with the
    //! standard 1-overlap structure, so it wraps in the standard
    //! Opm::GhostLastMatrixAdapter and is handed to opm's parallel
    //! FlexibleSolver with the SAME property tree as the serial path -
    //! the transposed CPR 'cprt' preconditioner (parallel AMG/hypre coarse
    //! solve) is thus reused unchanged, instead of the previous block-ILU0
    //! fallback. WS3 caching (build once, preconditioner().update() after)
    //! mirrors solveSerial_.
    void solveParallel_(const Matrix& matrix, Vector& rhs, Vector& solution,
                        Dune::InverseOperatorResult& result)
    {
        using Comm = Dune::OwnerOverlapCopyCommunication<int, int>;
        using ParOp = Opm::GhostLastMatrixAdapter<Matrix, Vector, Vector, Comm>;

        buildConsistentParallelTranspose(matrix, *comm_, cachedTranspose_);
        invalidateRows_(cachedTranspose_, *overlapRows_); // ghost rows -> identity
        for (const int row : *overlapRows_) {
            rhs[row] = 0.0;
        }

        if (verifyTranspose_) {
            verifyTransposeExact_(matrix, cachedTranspose_);
        }

        using Clock = std::chrono::steady_clock;
        const bool rebuild = !reusePreconditioner_ || !cachedParSolver_;
        const auto tSetup0 = Clock::now();
        if (rebuild) {
            cachedParSolver_.reset(); // free before the operator it references
            cachedParOp_ = std::make_unique<ParOp>(cachedTranspose_, *comm_);
            cachedWeights_ = std::function<Vector()>{};
            if (usesCprWeights_) {
                const Matrix& matrixRef = cachedTranspose_;
                const bool transposeFlag = transposedWeights_;
                const std::size_t pIdx = pressureIndex_;
                const Comm* comm = comm_;
                cachedWeights_ = [&matrixRef, transposeFlag, pIdx, comm]() {
                    Vector w = Amg::getQuasiImpesWeights<Matrix, Vector>(
                        matrixRef, pIdx, transposeFlag,
                        /*enable_thread_parallel=*/false);
                    // cprt's coarse assembly indexes weights by COLUMN
                    // (PressureTransferPolicy transpose branch), which for an
                    // owned row's ghost columns needs the ghost cell's OWNER
                    // weights. They are computed locally from M's
                    // identity-invalidated ghost rows and are wrong, so copy the
                    // owners' weights onto the ghosts - else the coarse operator
                    // is cross-rank inconsistent and BiCGSTAB false-converges.
                    comm->copyOwnerToAll(w, w);
                    return w;
                };
            }
            cachedParSolver_ =
                std::make_unique<Dune::FlexibleSolver<ParOp>>(
                    *cachedParOp_, *comm_, prm_, cachedWeights_,
                    pressureIndex_);
        } else {
            cachedParSolver_->preconditioner().update();
        }
        setupSeconds_ += std::chrono::duration<double>(
            Clock::now() - tSetup0).count();
        const auto tApply0 = Clock::now();
        cachedParSolver_->apply(solution, rhs, result);
        applySeconds_ += std::chrono::duration<double>(
            Clock::now() - tApply0).count();
    }

    //! \brief Open-Question-A guard: check the consistent transpose M
    //!        reproduces the exact transposed action of the matrix-free
    //!        oracle (TransposedParallelOperator) for a random consistent x.
    //!        Enabled by OPM_ADJOINT_VERIFY_TRANSPOSE=1.
    void verifyTransposeExact_(const Matrix& forward, const Matrix& M)
    {
        using Comm = Dune::OwnerOverlapCopyCommunication<int, int>;
        using ParOp = Opm::GhostLastMatrixAdapter<Matrix, Vector, Vector, Comm>;
        constexpr int bs = static_cast<int>(Vector::block_type::dimension);

        Vector x(M.N());
        for (std::size_t i = 0; i < x.size(); ++i) {
            for (int j = 0; j < bs; ++j) {
                x[i][j] = std::sin(0.13 * (static_cast<double>(i) * bs + j) + 1.0);
            }
        }
        comm_->copyOwnerToAll(x, x);

        Matrix localT;
        transposeBlockMatrixInto(forward, localT);
        TransposedParallelOperator<Matrix, Vector, Comm> oracle(forward, localT,
                                                                *comm_);
        ParOp op(M, *comm_);

        Vector y1(M.N()), y2(M.N());
        op.apply(x, y1);
        oracle.apply(x, y2);
        y1 -= y2;
        const double num = comm_->norm(y1);
        const double den = std::max(comm_->norm(y2), 1e-300);
        const double rel = num / den;
        if (rel > 1e-10) {
            OpmLog::warning(fmt::format(
                "Adjoint parallel transpose verify: operator differs from "
                "oracle by relative {:.3e} (Open Question A: operator inexact)",
                rel));
        } else {
            OpmLog::info(fmt::format(
                "Adjoint parallel transpose verify: operator matches oracle "
                "(relative {:.3e})", rel));
        }
    }
#endif

    static void invalidateRows_(Matrix& matrix, const std::vector<int>& rows)
    {
        typename Matrix::block_type identity(0.0);
        for (int eq = 0; eq < static_cast<int>(Matrix::block_type::rows); ++eq) {
            identity[eq][eq] = 1.0;
        }
        for (const int row : rows) {
            matrix[row] = 0.0;
            matrix[row][row] = identity;
        }
    }

    // Cached transposes: the sparsity pattern is constant across the
    // backward sweep, so after the first solve only values are refilled.
#if HAVE_SUITESPARSE_UMFPACK
    TransposedMatrix<Matrix> cachedDirectTranspose_;
#endif
    Matrix cachedTranspose_;

    std::string spec_{"umfpack"};
    PropertyTree prm_{};
    bool usesCprWeights_ = false;
    bool transposedWeights_ = false;
    std::size_t pressureIndex_ = 1;
#if HAVE_MPI
    const Dune::OwnerOverlapCopyCommunication<int, int>* comm_ = nullptr;
    const std::vector<int>* overlapRows_ = nullptr;
    bool verifyTranspose_ = false; //!< OPM_ADJOINT_VERIFY_TRANSPOSE=1
#endif
    int totalIterations_ = 0;
    int numSolves_ = 0;
    double setupSeconds_ = 0.0;  //!< solver/preconditioner construction time
    double applySeconds_ = 0.0;  //!< iterative apply (Krylov) time

    // WS3: reuse the serial iterative solver across the sweep. Every backward
    // substep solves a system with the SAME sparsity (only cachedTranspose_'s
    // values change), so the preconditioner hierarchy (for cpr/cprt: the AMG
    // aggregation) can be built once and refreshed with preconditioner().update()
    // instead of reconstructing the FlexibleSolver each solve — reclaiming the
    // symbolic share of the setup cost (measured ~24% of the Norne solve). The
    // operator holds a reference to cachedTranspose_ (a member, refilled in
    // place), so its lifetime is stable. Declared before the solver so the
    // solver (which references the operator) is destroyed first.
    using SerialOperator = Dune::MatrixAdapter<Matrix, Vector, Vector>;
    std::unique_ptr<SerialOperator> cachedOp_;
    std::unique_ptr<Dune::FlexibleSolver<SerialOperator>> cachedSerialSolver_;
    std::function<Vector()> cachedWeights_;
    bool reusePreconditioner_ = true;  //!< off via OPM_ADJOINT_REBUILD_PRECOND=1

#if HAVE_MPI
    // Parallel analogue of the serial cache: the consistent transpose M is
    // wrapped in a GhostLastMatrixAdapter and solved by opm's parallel
    // FlexibleSolver (cprt). Same sparsity across the sweep, so the solver is
    // built once and refreshed with preconditioner().update(). Declared after
    // the serial members and destroyed before cachedTranspose_ (which the
    // operator references).
    using ParallelOperator =
        Opm::GhostLastMatrixAdapter<Matrix, Vector, Vector,
                                    Dune::OwnerOverlapCopyCommunication<int, int>>;
    std::unique_ptr<ParallelOperator> cachedParOp_;
    std::unique_ptr<Dune::FlexibleSolver<ParallelOperator>> cachedParSolver_;
#endif
};

} // namespace Opm

#endif // OPM_ADJOINT_LINEAR_SOLVER_HPP
