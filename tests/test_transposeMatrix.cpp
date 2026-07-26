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

#include <opm/adjoint/AdjointLinearSolver.hpp>
#include <opm/adjoint/transposeMatrix.hpp>

#define BOOST_TEST_MODULE TransposeMatrixTest
#include <boost/test/unit_test.hpp>

#include <opm/simulators/linalg/matrixblock.hh>

#include <dune/common/fmatrix.hh>
#include <dune/common/fvector.hh>
#include <dune/istl/bcrsmatrix.hh>
#include <dune/istl/bvector.hh>
#include <dune/istl/matrixindexset.hh>

#include <cmath>
#include <cstddef>

namespace {

// 3x3 blocks (the blackoil numEq): AdjointLinearSolver's parallel path is
// pre-instantiated (opm-simulators FlexibleSolver/PreconditionerFactory) only
// for Opm::MatrixBlock<double,3,3>, so the solver test must use that type.
constexpr int BS = 3;
using Block = Opm::MatrixBlock<double, BS, BS>;
using Matrix = Dune::BCRSMatrix<Block>;
using Vector = Dune::BlockVector<Dune::FieldVector<double, BS>>;

// A small nonsymmetric block tri-diagonal matrix with full asymmetric blocks.
Matrix makeTestMatrix()
{
    constexpr int n = 5;
    Dune::MatrixIndexSet indices(n, n);
    for (int i = 0; i < n; ++i) {
        indices.add(i, i);
        if (i > 0) {
            indices.add(i, i - 1);
        }
        if (i < n - 1) {
            indices.add(i, i + 1);
        }
    }
    Matrix matrix;
    indices.exportIdx(matrix);

    double value = 1.0;
    for (auto row = matrix.begin(); row != matrix.end(); ++row) {
        for (auto entry = row->begin(); entry != row->end(); ++entry) {
            for (int i = 0; i < BS; ++i) {
                for (int j = 0; j < BS; ++j) {
                    (*entry)[i][j] = value;
                    value = std::fmod(value * 1.7 + 0.3, 5.0) + 0.1;
                }
            }
            // Make diagonal blocks strongly dominant so the matrix is
            // comfortably invertible for the solver test.
            if (entry.index() == row.index()) {
                for (int e = 0; e < BS; ++e) {
                    (*entry)[e][e] += 25.0;
                }
            }
        }
    }
    return matrix;
}

} // anonymous namespace

BOOST_AUTO_TEST_CASE(BlockTranspose)
{
    Block block;
    block[0][0] = 1.0; block[0][1] = 2.0;
    block[1][0] = 3.0; block[1][1] = 4.0;
    const auto transposed = Opm::transposeBlock(block);
    BOOST_CHECK_EQUAL(transposed[0][1], 3.0);
    BOOST_CHECK_EQUAL(transposed[1][0], 2.0);
    BOOST_CHECK_EQUAL(transposed[0][0], 1.0);
    BOOST_CHECK_EQUAL(transposed[1][1], 4.0);
}

BOOST_AUTO_TEST_CASE(MatrixTransposeEntries)
{
    const auto matrix = makeTestMatrix();
    const auto transposed = Opm::transposeBlockMatrix(matrix);

    BOOST_CHECK_EQUAL(transposed.N(), matrix.M());
    BOOST_CHECK_EQUAL(transposed.M(), matrix.N());
    BOOST_CHECK_EQUAL(transposed.nonzeroes(), matrix.nonzeroes());

    for (auto row = matrix.begin(); row != matrix.end(); ++row) {
        for (auto entry = row->begin(); entry != row->end(); ++entry) {
            const auto& original = *entry;
            const auto& mirrored = transposed[entry.index()][row.index()];
            for (int i = 0; i < BS; ++i) {
                for (int j = 0; j < BS; ++j) {
                    BOOST_CHECK_EQUAL(mirrored[i][j], original[j][i]);
                }
            }
        }
    }

    // Transposing twice gives back the original.
    const auto twice = Opm::transposeBlockMatrix(transposed);
    for (auto row = matrix.begin(); row != matrix.end(); ++row) {
        for (auto entry = row->begin(); entry != row->end(); ++entry) {
            const auto& original = *entry;
            const auto& restored = twice[row.index()][entry.index()];
            for (int i = 0; i < BS; ++i) {
                for (int j = 0; j < BS; ++j) {
                    BOOST_CHECK_EQUAL(restored[i][j], original[i][j]);
                }
            }
        }
    }
}

BOOST_AUTO_TEST_CASE(TransposeMatVecConsistency)
{
    // y1 = A^T x via the explicit transpose must match y2 = A^T x via umtv.
    const auto matrix = makeTestMatrix();
    const auto transposed = Opm::transposeBlockMatrix(matrix);

    Vector x(matrix.N());
    for (std::size_t i = 0; i < x.size(); ++i) {
        for (int j = 0; j < BS; ++j) {
            x[i][j] = 1.0 + static_cast<double>(i) - 0.5 * j;
        }
    }

    Vector viaExplicit(matrix.M());
    viaExplicit = 0.0;
    transposed.mv(x, viaExplicit);

    Vector viaUmtv(matrix.M());
    viaUmtv = 0.0;
    matrix.umtv(x, viaUmtv);

    for (std::size_t i = 0; i < viaExplicit.size(); ++i) {
        for (int j = 0; j < BS; ++j) {
            BOOST_CHECK_CLOSE(viaExplicit[i][j], viaUmtv[i][j], 1e-12);
        }
    }
}

#if HAVE_SUITESPARSE_UMFPACK
BOOST_AUTO_TEST_CASE(SolveTransposed)
{
    const auto matrix = makeTestMatrix();

    Vector rhs(matrix.M());
    for (std::size_t i = 0; i < rhs.size(); ++i) {
        for (int j = 0; j < BS; ++j) {
            rhs[i][j] = 1.0 + j * (static_cast<double>(i) - 2.0);
        }
    }

    Opm::AdjointLinearSolver<Matrix, Vector> solver;
    Vector solution;
    solver.solveTransposed(matrix, rhs, solution);

    // Check the residual of the transposed system: A^T x - b = 0.
    Vector residual(rhs);
    matrix.mmtv(solution, residual);   // residual -= A^T solution

    double norm = 0.0;
    for (std::size_t i = 0; i < residual.size(); ++i) {
        for (int j = 0; j < BS; ++j) {
            norm = std::max(norm, std::abs(residual[i][j]));
        }
    }
    BOOST_CHECK_SMALL(norm, 1e-10);
}
#endif
