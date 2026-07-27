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
 * \brief Solve a dumped adjoint system standalone, outside the simulator.
 *
 * Reads a directory written by
 *
 *     flow_adjoint CASE.DATA --adjoint-mode=gradient \
 *         --adjoint-linear-solver=system_cprt --adjoint-dump-system=DIR
 *
 * rebuilds the transposed coupled system S^T = [A^T B^T; C^T D^T], and solves
 * it. Useful for:
 *
 *   - tuning the preconditioner on a *real field system* in seconds instead
 *     of re-running a whole forward+backward sweep;
 *   - reproducing a convergence failure from a single small directory;
 *   - checking a candidate configuration against the direct solution.
 *
 * A JSON FlexibleSolver configuration can be passed as the second argument to
 * override the built-in system_cprt tree; that is the knob to turn when
 * experimenting.
 *
 * Usage:
 *   adjoint_system_solve <dump-dir> [solver-config.json]
 */

#include <config.h>

#include <opm/adjoint/AdjointSystemTranspose.hpp>
#include <opm/adjoint/matrixMarketIO.hpp>

#include <opm/simulators/linalg/FlexibleSolver.hpp>
#include <opm/simulators/linalg/getQuasiImpesWeights.hpp>
#include <opm/simulators/linalg/system/SystemPreconditionerFactory.hpp>
#include <opm/simulators/linalg/system/SystemTypes.hpp>

#if HAVE_SUITESPARSE_UMFPACK
#include <dune/istl/umfpack.hh>
#endif

#include <chrono>
#include <cmath>
#include <cstddef>
#include <exception>
#include <filesystem>
#include <iostream>
#include <string>

namespace {

using Scalar = double;
constexpr auto _0 = Dune::Indices::_0;
constexpr auto _1 = Dune::Indices::_1;
constexpr std::size_t pressureIndex = 1;  // blackoil Indices::pressureSwitchIdx

double dot(const Opm::SystemVector<Scalar>& a,
           const Opm::SystemVector<Scalar>& b)
{
    return a[_0].dot(b[_0]) + a[_1].dot(b[_1]);
}

double norm(const Opm::SystemVector<Scalar>& v)
{
    return std::sqrt(dot(v, v));
}

} // anonymous namespace

int main(int argc, char** argv)
try {
    if (argc < 2) {
        std::cerr << "usage: " << argv[0]
                  << " <dump-dir> [solver-config.json]\n\n"
                     "  <dump-dir> is a directory written by\n"
                     "    flow_adjoint ... --adjoint-dump-system=<dir>\n";
        return EXIT_FAILURE;
    }
    const std::filesystem::path dir(argv[1]);

    // ---- read the forward blocks -----------------------------------------
    Opm::RRMatrix<Scalar> A;
    Opm::SystemWellBlocks<Scalar> forward;
    Opm::ResVector<Scalar> rhs;

    Opm::readBlockedMatrixMarket(A, (dir / "A.mm").string());
    Opm::readBlockedVectorMarket(rhs, (dir / "rhs.mm").string());
    const bool coupled = std::filesystem::exists(dir / "D.mm");
    if (!coupled) {
        std::cerr << "error: " << dir
                  << " holds a Schur-reduced system (no well blocks).\n"
                     "       Re-dump with --adjoint-linear-solver=system_cprt.\n";
        return EXIT_FAILURE;
    }
    Opm::readBlockedMatrixMarket(forward.B, (dir / "B.mm").string());
    Opm::readBlockedMatrixMarket(forward.C, (dir / "C.mm").string());
    Opm::readBlockedMatrixMarket(forward.D, (dir / "D.mm").string());

    const std::size_t numCells = A.N();
    const std::size_t numWellDofs = forward.D.N();
    std::cout << "system: " << numCells << " reservoir cells ("
              << numCells * Opm::numResDofs << " dofs), " << numWellDofs
              << " well blocks (" << numWellDofs * Opm::numWellDofs
              << " dofs)\n";

    // ---- transpose --------------------------------------------------------
    Opm::RRMatrix<Scalar> At;
    Opm::TransposedWellBlocks<Scalar> transposed;
    Opm::transposeBlockMatrixInto(A, At);
    Opm::transposeWellBlocks(forward, transposed);

    const auto S = Opm::makeSystemMatrix<Scalar>(A, forward.C, forward.B,
                                                 forward.D);
    const auto St = Opm::makeSystemMatrix<Scalar>(At, transposed.Bt,
                                                  transposed.Ct,
                                                  transposed.Dt);

    Opm::SystemVector<Scalar> b, x, work;
    auto sizeUp = [numCells, numWellDofs](Opm::SystemVector<Scalar>& v) {
        v[_0].resize(numCells);
        v[_1].resize(numWellDofs);
        v[_0] = 0.0;
        v[_1] = 0.0;
    };
    sizeUp(b);
    sizeUp(x);
    sizeUp(work);
    b[_0] = rhs;   // the well right-hand side is zero; see the plan/header

    // ---- sanity: <S^T y, z> == <y, S z> ----------------------------------
    {
        Opm::SystemVector<Scalar> y, z, Sz, Sty;
        sizeUp(y); sizeUp(z); sizeUp(Sz); sizeUp(Sty);
        for (std::size_t i = 0; i < numCells; ++i) {
            for (int j = 0; j < Opm::numResDofs; ++j) {
                y[_0][i][j] = 1.0 + 0.25 * static_cast<double>((i + j) % 7);
                z[_0][i][j] = 0.5 - 0.125 * static_cast<double>((3 * i + j) % 5);
            }
        }
        for (std::size_t i = 0; i < numWellDofs; ++i) {
            for (int j = 0; j < Opm::numWellDofs; ++j) {
                y[_1][i][j] = 1.0 - 0.1 * j;
                z[_1][i][j] = 0.3 + 0.2 * j;
            }
        }
        S.mv(z, Sz);
        St.mv(y, Sty);
        const double lhs = dot(y, Sz);
        const double rhsDot = dot(Sty, z);
        const double rel = std::abs(lhs - rhsDot) /
                           std::max(std::abs(lhs), 1e-300);
        std::cout << "transpose identity <y,Sz> vs <S^T y,z>: relative "
                  << "difference " << rel << "\n";
        if (rel > 1e-10) {
            std::cerr << "error: the transpose is wrong\n";
            return EXIT_FAILURE;
        }
    }

    // ---- solve ------------------------------------------------------------
    const auto prm = (argc > 2)
        ? Opm::PropertyTree(argv[2])
        : Opm::setupAdjointSystemCPR(/*reduction=*/1e-14, /*maxIter=*/300,
                                     /*verbosity=*/0);

    using SysOp = Opm::SystemSeqOp<Scalar>;
    SysOp op(St);
    std::function<Opm::SystemVector<Scalar>()> weights = [&At]() {
        Opm::SystemVector<Scalar> w;
        w[_0] = Opm::Amg::getQuasiImpesWeights<Opm::RRMatrix<Scalar>,
                                               Opm::ResVector<Scalar>>(
            At, pressureIndex, /*transpose=*/true,
            /*enable_thread_parallel=*/false);
        return w;
    };

    using Clock = std::chrono::steady_clock;
    const auto tSetup = Clock::now();
    Dune::FlexibleSolver<SysOp> solver(op, prm, weights, pressureIndex);
    const auto tApply = Clock::now();

    Opm::SystemVector<Scalar> rhsCopy(b);  // the solver overwrites it
    Dune::InverseOperatorResult result;
    solver.apply(x, rhsCopy, result);
    const auto tDone = Clock::now();

    auto seconds = [](auto a, auto b2) {
        return std::chrono::duration<double>(b2 - a).count();
    };
    std::cout << "solve: " << result.iterations << " iterations, converged="
              << result.converged << ", reduction " << result.reduction
              << "\n       setup " << seconds(tSetup, tApply) << " s, apply "
              << seconds(tApply, tDone) << " s\n";

    // True relative residual, independent of what the solver reports.
    work = b;
    Opm::SystemVector<Scalar> Sx;
    sizeUp(Sx);
    St.mv(x, Sx);
    work[_0] -= Sx[_0];
    work[_1] -= Sx[_1];
    std::cout << "relative residual ||S^T x - b|| / ||b|| = "
              << norm(work) / std::max(norm(b), 1e-300) << "\n";

#if HAVE_SUITESPARSE_UMFPACK
    // ---- compare against the direct solution ------------------------------
    {
        auto flat = Opm::flattenSystem(St);
        Dune::BlockVector<Dune::FieldVector<Scalar, 1>> flatRhs, flatSol;
        Opm::flattenSystemVector(b, flatRhs);
        flatSol.resize(flatRhs.size());
        flatSol = 0.0;

        Dune::UMFPack<Opm::FlatSystemMatrix<Scalar>> direct(flat, 0);
        Dune::InverseOperatorResult directResult;
        direct.apply(flatSol, flatRhs, directResult);

        Opm::SystemVector<Scalar> exact;
        sizeUp(exact);
        Opm::unflattenSystemVector(flatSol, exact);

        double maxDiff = 0.0;
        double scale = 0.0;
        for (std::size_t i = 0; i < numCells; ++i) {
            for (int j = 0; j < Opm::numResDofs; ++j) {
                maxDiff = std::max(maxDiff,
                                   std::abs(x[_0][i][j] - exact[_0][i][j]));
                scale = std::max(scale, std::abs(exact[_0][i][j]));
            }
        }
        std::cout << "vs direct solve: max reservoir difference "
                  << maxDiff / std::max(scale, 1e-300) << " (relative)\n";

        // The direct solution's residual is the floor imposed by the
        // conditioning/scaling of S^T; reporting it stops the iterative
        // solver's residual from looking worse than it is.
        Opm::SystemVector<Scalar> exactResidual(b), SExact;
        sizeUp(SExact);
        St.mv(exact, SExact);
        exactResidual[_0] -= SExact[_0];
        exactResidual[_1] -= SExact[_1];
        std::cout << "                 direct solve's own relative residual "
                  << norm(exactResidual) / std::max(norm(b), 1e-300) << "\n";
    }
#endif

    // ---- compare against the adjoint the simulator computed ---------------
    if (std::filesystem::exists(dir / "lambda.mm")) {
        Opm::ResVector<Scalar> lambda;
        Opm::readBlockedVectorMarket(lambda, (dir / "lambda.mm").string());
        double maxDiff = 0.0;
        double scale = 0.0;
        for (std::size_t i = 0; i < numCells; ++i) {
            for (int j = 0; j < Opm::numResDofs; ++j) {
                maxDiff = std::max(maxDiff, std::abs(x[_0][i][j] - lambda[i][j]));
                scale = std::max(scale, std::abs(lambda[i][j]));
            }
        }
        std::cout << "vs the simulator's lambda: max difference "
                  << maxDiff / std::max(scale, 1e-300) << " (relative)\n";
    }

    return result.converged ? EXIT_SUCCESS : EXIT_FAILURE;
}
catch (const std::exception& e) {
    std::cerr << "adjoint_system_solve failed: " << e.what() << "\n";
    return EXIT_FAILURE;
}
