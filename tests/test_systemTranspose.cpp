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
 * \brief Deck-free tests for the transposed coupled reservoir+well system.
 *
 * Builds a small but structurally faithful black-oil-like system
 * S = [A C; B D] (TPFA 3-equation reservoir on a 1D chain plus two
 * multi-perforation wells, merged through the real Opm::WellMatrixMerger),
 * then checks the four things the adjoint full-system solve depends on:
 *
 *   1. the block-shape mapping into the transposed SystemMatrix,
 *   2. the transpose itself, via <S^T y, x> == <y, S x>,
 *   3. MatrixMarket round-tripping of every block (incl. the rectangular
 *      4x3 / 3x4 well blocks),
 *   4. that `system_cprt` solves S^T z = b, cross-checked against a direct
 *      solve of the assembled system, and that the Schur-shifted right-hand
 *      side [g_r - B^T D^-T g_w; 0] gives the same reservoir adjoint as the
 *      unshifted [g_r; g_w].
 *
 * (4) is the algebraic identity the adjoint relies on to keep using its
 * existing right-hand side and its existing well-adjoint recovery.
 */

#include <config.h>

#include <opm/adjoint/AdjointSystemTranspose.hpp>
#include <opm/adjoint/matrixMarketIO.hpp>

#define BOOST_TEST_MODULE SystemTransposeTest
#include <boost/test/unit_test.hpp>

#include <opm/simulators/linalg/FlexibleSolver.hpp>
#include <opm/simulators/linalg/FlowLinearSolverParameters.hpp>
#include <opm/simulators/linalg/getQuasiImpesWeights.hpp>
#include <opm/simulators/linalg/setupPropertyTree.hpp>
#include <opm/simulators/linalg/system/SystemPreconditionerFactory.hpp>
#include <opm/simulators/linalg/system/SystemTypes.hpp>

#include <dune/common/dynmatrix.hh>
#include <dune/common/dynvector.hh>
#include <dune/istl/matrixindexset.hh>
#include <dune/istl/umfpack.hh>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <filesystem>
#include <string>
#include <vector>

namespace {

using Scalar = double;
constexpr int numRes = Opm::numResDofs;    // 3
constexpr int numWell = Opm::numWellDofs;  // 4
constexpr std::size_t pressureIndex = 1;   // blackoil Indices::pressureSwitchIdx

using RR = Opm::RRMatrix<Scalar>;
using RW = Opm::RWMatrix<Scalar>;
using WR = Opm::WRMatrix<Scalar>;
using WW = Opm::WWMatrix<Scalar>;
using ResVec = Opm::ResVector<Scalar>;
using WellVec = Opm::WellVector<Scalar>;
using SysVec = Opm::SystemVector<Scalar>;
using ScalarMatrix = Dune::BCRSMatrix<Dune::FieldMatrix<Scalar, 1, 1>>;

constexpr auto _0 = Dune::Indices::_0;
constexpr auto _1 = Dune::Indices::_1;

constexpr std::size_t numCells = 40;
const std::vector<std::vector<int>> wellPerfs = {{2, 9, 15, 21}, {5, 27, 33}};

//! \brief Cheap deterministic pseudo-random sequence (no <random> dependence
//!        on a particular library's generator, so the test is reproducible).
class Lcg
{
public:
    explicit Lcg(unsigned seed) : state_(seed) {}
    double next()
    {
        state_ = state_ * 1103515245u + 12345u;
        return static_cast<double>((state_ >> 16) & 0x7fffu) / 32768.0 - 0.5;
    }
private:
    unsigned state_;
};

// --------------------------------------------------------------------------
// A TPFA-shaped reservoir Jacobian on a 1D chain.
//
// The structure matters: CPR (and its transposed variant) only behaves like
// CPR if the matrix really has a pressure-elliptic sub-block. So each face
// contributes T * Flux with Flux's pressure column dominant, and each cell an
// accumulation block, exactly as a black-oil TPFA linearizer would build it.
// --------------------------------------------------------------------------
RR makeReservoirMatrix()
{
    Dune::FieldMatrix<Scalar, numRes, numRes> flux(0.0);
    // Pressure-driven flow: the pressure column carries the phase mobilities.
    flux[0][pressureIndex] = 1.0;
    flux[1][pressureIndex] = 1.6;
    flux[2][pressureIndex] = 0.4;
    // Weak upwind-saturation coupling in the non-pressure columns.
    for (int e = 0; e < numRes; ++e) {
        for (int v = 0; v < numRes; ++v) {
            if (v != static_cast<int>(pressureIndex)) {
                flux[e][v] = 0.03 * (1.0 + 0.5 * e - 0.25 * v);
            }
        }
    }

    Dune::FieldMatrix<Scalar, numRes, numRes> acc(0.0);
    acc[0][0] = 2.0;  acc[0][1] = 0.10; acc[0][2] = 0.30;
    acc[1][0] = 0.05; acc[1][1] = 3.0;  acc[1][2] = 0.20;
    acc[2][0] = 0.02; acc[2][1] = 0.15; acc[2][2] = 2.5;

    Dune::MatrixIndexSet indices(numCells, numCells);
    for (std::size_t i = 0; i < numCells; ++i) {
        indices.add(i, i);
        if (i > 0)             { indices.add(i, i - 1); }
        if (i + 1 < numCells)  { indices.add(i, i + 1); }
    }
    RR matrix;
    indices.exportIdx(matrix);
    matrix = 0.0;

    Lcg rng(12345);
    for (std::size_t i = 0; i < numCells; ++i) {
        for (std::size_t j : {i > 0 ? i - 1 : i, i + 1}) {
            if (j == i || j >= numCells) {
                continue;
            }
            // Heterogeneous but positive transmissibility.
            const double trans = 1.0 + 0.8 * rng.next();
            for (int e = 0; e < numRes; ++e) {
                for (int v = 0; v < numRes; ++v) {
                    matrix[i][j][e][v] -= trans * flux[e][v];
                    matrix[i][i][e][v] += trans * flux[e][v];
                }
            }
        }
        for (int e = 0; e < numRes; ++e) {
            for (int v = 0; v < numRes; ++v) {
                matrix[i][i][e][v] += acc[e][v] * (1.0 + 0.1 * rng.next());
            }
        }
    }
    return matrix;
}

//! \brief Per-well B/C/D blocks in the layout `addBCDMatrix` produces:
//!        B is 1 x nperf, C is nperf x 1, D is 1 x 1.
void makeWellBlocks(const std::vector<int>& perfs,
                    unsigned seed,
                    WR& wellB,
                    RW& wellC,
                    WW& wellD)
{
    const auto nperf = perfs.size();
    Lcg rng(seed);

    {
        Dune::MatrixIndexSet idx(1, nperf);
        for (std::size_t p = 0; p < nperf; ++p) {
            idx.add(0, p);
        }
        idx.exportIdx(wellB);
        wellB = 0.0;
        for (std::size_t p = 0; p < nperf; ++p) {
            for (int i = 0; i < numWell; ++i) {
                for (int j = 0; j < numRes; ++j) {
                    wellB[0][p][i][j] = 0.4 * rng.next();
                }
            }
        }
    }
    {
        Dune::MatrixIndexSet idx(nperf, 1);
        for (std::size_t p = 0; p < nperf; ++p) {
            idx.add(p, 0);
        }
        idx.exportIdx(wellC);
        wellC = 0.0;
        for (std::size_t p = 0; p < nperf; ++p) {
            for (int i = 0; i < numRes; ++i) {
                for (int j = 0; j < numWell; ++j) {
                    wellC[p][0][i][j] = 0.5 * rng.next();
                }
            }
        }
    }
    {
        Dune::MatrixIndexSet idx(1, 1);
        idx.add(0, 0);
        idx.exportIdx(wellD);
        wellD = 0.0;
        for (int i = 0; i < numWell; ++i) {
            for (int j = 0; j < numWell; ++j) {
                wellD[0][0][i][j] = 0.3 * rng.next();
            }
            wellD[0][0][i][i] += 4.0;  // comfortably invertible
        }
    }
}

//! \brief Build the merged forward well blocks through the real merger.
Opm::SystemWellBlocks<Scalar> makeMergedWellBlocks()
{
    std::vector<WR> perWellB(wellPerfs.size());
    std::vector<RW> perWellC(wellPerfs.size());
    std::vector<WW> perWellD(wellPerfs.size());
    Opm::SparseTable<int> wellCells;

    for (std::size_t w = 0; w < wellPerfs.size(); ++w) {
        makeWellBlocks(wellPerfs[w], 777u + 31u * static_cast<unsigned>(w),
                       perWellB[w], perWellC[w], perWellD[w]);
        wellCells.appendRow(wellPerfs[w].begin(), wellPerfs[w].end());
    }

    const Opm::WellMatrixMerger<Scalar> merger(numCells, perWellB, perWellC,
                                               perWellD, wellCells);
    Opm::SystemWellBlocks<Scalar> blocks;
    merger.buildMatrices(blocks.B, blocks.C, blocks.D);
    return blocks;
}

//! \brief Flatten a SystemMatrix into a scalar BCRS matrix, so a direct
//!        solver can be used as an independent reference.
ScalarMatrix assembleScalar(const Opm::SystemMatrix<Scalar>& S)
{
    const std::size_t nRes = S.A->N();
    const std::size_t nWell = S.D->N();
    const std::size_t dim = nRes * numRes + nWell * numWell;
    const std::size_t wellOffset = nRes * numRes;

    Dune::MatrixIndexSet indices(dim, dim);
    auto addBlocks = [&indices](const auto& block, std::size_t rowOff,
                                std::size_t colOff, int br, int bc) {
        for (auto row = block.begin(); row != block.end(); ++row) {
            for (auto entry = row->begin(); entry != row->end(); ++entry) {
                for (int i = 0; i < br; ++i) {
                    for (int j = 0; j < bc; ++j) {
                        indices.add(rowOff + row.index() * br + i,
                                    colOff + entry.index() * bc + j);
                    }
                }
            }
        }
    };
    addBlocks(*S.A, 0, 0, numRes, numRes);
    addBlocks(*S.C, 0, wellOffset, numRes, numWell);
    addBlocks(*S.B, wellOffset, 0, numWell, numRes);
    addBlocks(*S.D, wellOffset, wellOffset, numWell, numWell);

    ScalarMatrix flat;
    indices.exportIdx(flat);
    flat = 0.0;

    auto fillBlocks = [&flat](const auto& block, std::size_t rowOff,
                              std::size_t colOff, int br, int bc) {
        for (auto row = block.begin(); row != block.end(); ++row) {
            for (auto entry = row->begin(); entry != row->end(); ++entry) {
                for (int i = 0; i < br; ++i) {
                    for (int j = 0; j < bc; ++j) {
                        flat[rowOff + row.index() * br + i]
                            [colOff + entry.index() * bc + j][0][0] = (*entry)[i][j];
                    }
                }
            }
        }
    };
    fillBlocks(*S.A, 0, 0, numRes, numRes);
    fillBlocks(*S.C, 0, wellOffset, numRes, numWell);
    fillBlocks(*S.B, wellOffset, 0, numWell, numRes);
    fillBlocks(*S.D, wellOffset, wellOffset, numWell, numWell);
    return flat;
}

void fillPseudoRandom(SysVec& v, unsigned seed)
{
    Lcg rng(seed);
    for (std::size_t i = 0; i < v[_0].size(); ++i) {
        for (int j = 0; j < numRes; ++j) {
            v[_0][i][j] = rng.next();
        }
    }
    for (std::size_t i = 0; i < v[_1].size(); ++i) {
        for (int j = 0; j < numWell; ++j) {
            v[_1][i][j] = rng.next();
        }
    }
}

void resizeSystemVector(SysVec& v, std::size_t nRes, std::size_t nWell)
{
    v[_0].resize(nRes);
    v[_1].resize(nWell);
    v[_0] = 0.0;
    v[_1] = 0.0;
}

double dot(const SysVec& a, const SysVec& b)
{
    return a[_0].dot(b[_0]) + a[_1].dot(b[_1]);
}

//! \brief Solve S^T z = b with the system_cprt FlexibleSolver.
Dune::InverseOperatorResult solveSystemCprt(const Opm::SystemMatrix<Scalar>& St,
                                            const RR& transposedA,
                                            const SysVec& b,
                                            SysVec& z)
{
    using SysOp = Opm::SystemSeqOp<Scalar>;
    SysOp op(St);

    std::function<SysVec()> weights = [&transposedA]() {
        SysVec w;
        w[_0] = Opm::Amg::getQuasiImpesWeights<RR, ResVec>(
            transposedA, pressureIndex, /*transpose=*/true,
            /*enable_thread_parallel=*/false);
        return w;
    };

    // The very same tree the solver and the standalone example build.
    const auto prm = Opm::setupAdjointSystemCPR(1e-12, 200, 0);
    Dune::FlexibleSolver<SysOp> solver(op, prm, weights, pressureIndex);

    SysVec rhs(b);  // the solver overwrites its right-hand side
    Dune::InverseOperatorResult result;
    solver.apply(z, rhs, result);
    return result;
}

double relativeResidual(const Opm::SystemMatrix<Scalar>& S,
                        const SysVec& x,
                        const SysVec& b)
{
    SysVec residual(b);
    SysVec Sx;
    resizeSystemVector(Sx, b[_0].size(), b[_1].size());
    S.mv(x, Sx);
    residual[_0] -= Sx[_0];
    residual[_1] -= Sx[_1];
    const double bNorm = std::sqrt(dot(b, b));
    return std::sqrt(dot(residual, residual)) / std::max(bNorm, 1e-300);
}

//! \brief Dense D^-T g_w (D is small: one 4x4 block per standard well).
WellVec solveTransposedD(const WW& D, const WellVec& g)
{
    const std::size_t dim = D.N() * numWell;
    Dune::DynamicMatrix<Scalar> dense(dim, dim, 0.0);
    for (auto row = D.begin(); row != D.end(); ++row) {
        for (auto entry = row->begin(); entry != row->end(); ++entry) {
            for (int i = 0; i < numWell; ++i) {
                for (int j = 0; j < numWell; ++j) {
                    // transposed: (D^T)[col*n+j][row*n+i] = D[row][col][i][j]
                    dense[entry.index() * numWell + j][row.index() * numWell + i]
                        = (*entry)[i][j];
                }
            }
        }
    }
    Dune::DynamicVector<Scalar> b(dim), x(dim, 0.0);
    for (std::size_t i = 0; i < D.N(); ++i) {
        for (int j = 0; j < numWell; ++j) {
            b[i * numWell + j] = g[i][j];
        }
    }
    dense.solve(x, b);

    WellVec out(D.N());
    for (std::size_t i = 0; i < D.N(); ++i) {
        for (int j = 0; j < numWell; ++j) {
            out[i][j] = x[i * numWell + j];
        }
    }
    return out;
}

//! \brief The fixture: one forward system and its transpose, built once.
struct SystemFixture
{
    SystemFixture()
        : A(makeReservoirMatrix())
        , forward(makeMergedWellBlocks())
    {
        Opm::transposeBlockMatrixInto(A, At);
        Opm::transposeWellBlocks(forward, transposed);
        S = Opm::makeSystemMatrix<Scalar>(A, forward.C, forward.B, forward.D);
        St = Opm::makeSystemMatrix<Scalar>(At, transposed.Bt, transposed.Ct,
                                           transposed.Dt);
    }

    std::size_t numWellDofs() const { return forward.D.N(); }

    RR A;
    RR At;
    Opm::SystemWellBlocks<Scalar> forward;
    Opm::TransposedWellBlocks<Scalar> transposed;
    Opm::SystemMatrix<Scalar> S;
    Opm::SystemMatrix<Scalar> St;
};

} // anonymous namespace

BOOST_FIXTURE_TEST_SUITE(SystemTranspose, SystemFixture)

//! The block-shape mapping: B^T must land in the C slot and vice versa.
BOOST_AUTO_TEST_CASE(BlockShapes)
{
    const std::size_t nWell = numWellDofs();
    BOOST_CHECK_EQUAL(nWell, wellPerfs.size());  // one dof block per standard well

    BOOST_CHECK_EQUAL(forward.B.N(), nWell);
    BOOST_CHECK_EQUAL(forward.B.M(), numCells);
    BOOST_CHECK_EQUAL(forward.C.N(), numCells);
    BOOST_CHECK_EQUAL(forward.C.M(), nWell);

    // B^T has C's shape, C^T has B's shape - which is why S^T is again a
    // SystemMatrix and needs no new types.
    BOOST_CHECK_EQUAL(transposed.Bt.N(), numCells);
    BOOST_CHECK_EQUAL(transposed.Bt.M(), nWell);
    BOOST_CHECK_EQUAL(transposed.Ct.N(), nWell);
    BOOST_CHECK_EQUAL(transposed.Ct.M(), numCells);
    BOOST_CHECK_EQUAL(transposed.Dt.N(), nWell);
    BOOST_CHECK_EQUAL(transposed.Dt.M(), nWell);

    BOOST_CHECK_EQUAL(transposed.Bt.nonzeroes(), forward.B.nonzeroes());
    BOOST_CHECK_EQUAL(transposed.Ct.nonzeroes(), forward.C.nonzeroes());

    // Spot-check the entries of the rectangular blocks.
    for (auto row = forward.B.begin(); row != forward.B.end(); ++row) {
        for (auto entry = row->begin(); entry != row->end(); ++entry) {
            for (int i = 0; i < numWell; ++i) {
                for (int j = 0; j < numRes; ++j) {
                    BOOST_CHECK_EQUAL(transposed.Bt[entry.index()][row.index()][j][i],
                                      (*entry)[i][j]);
                }
            }
        }
    }
}

//! The transpose identity. This is the check that catches every possible
//! mis-mapping of a block into the wrong slot, without solving anything.
BOOST_AUTO_TEST_CASE(TransposeIdentity)
{
    const std::size_t nWell = numWellDofs();
    SysVec x, y, Sx, Sty;
    resizeSystemVector(x, numCells, nWell);
    resizeSystemVector(y, numCells, nWell);
    resizeSystemVector(Sx, numCells, nWell);
    resizeSystemVector(Sty, numCells, nWell);
    fillPseudoRandom(x, 424242u);
    fillPseudoRandom(y, 987654u);

    S.mv(x, Sx);
    St.mv(y, Sty);

    const double lhs = dot(y, Sx);    // <y, S x>
    const double rhs = dot(Sty, x);   // <S^T y, x>
    BOOST_CHECK_CLOSE(lhs, rhs, 1e-10);
    // Guard against the identity holding trivially on a near-zero system.
    BOOST_CHECK_GT(std::abs(lhs), 1e-6);
}

//! MatrixMarket round-trip, including the rectangular 4x3 and 3x4 blocks
//! that dune-istl's own blocked reader cannot load (see matrixMarketIO.hpp).
BOOST_AUTO_TEST_CASE(MatrixMarketRoundTrip)
{
    const auto dir = std::filesystem::temp_directory_path() /
                     "opm_adjoint_system_mm_test";
    std::filesystem::create_directories(dir);

    auto roundTrip = [&dir](const auto& original, const std::string& name) {
        using MatrixType = std::remove_cvref_t<decltype(original)>;
        const auto path = (dir / (name + ".mm")).string();
        Opm::writeMatrixMarketFile(original, path);
        MatrixType restored;
        Opm::readBlockedMatrixMarket(restored, path);

        BOOST_CHECK_EQUAL(restored.N(), original.N());
        BOOST_CHECK_EQUAL(restored.M(), original.M());
        // Exact, not approximate: the dump must reproduce the in-situ system
        // bit for bit, which is what max_digits10 buys us.
        for (auto row = original.begin(); row != original.end(); ++row) {
            for (auto entry = row->begin(); entry != row->end(); ++entry) {
                for (int i = 0; i < MatrixType::block_type::rows; ++i) {
                    for (int j = 0; j < MatrixType::block_type::cols; ++j) {
                        BOOST_CHECK_EQUAL(restored[row.index()][entry.index()][i][j],
                                          (*entry)[i][j]);
                    }
                }
            }
        }
    };

    roundTrip(A, "A");
    roundTrip(forward.B, "B");   // 4x3 blocks: the case the dune reader trips on
    roundTrip(forward.C, "C");   // 3x4 blocks
    roundTrip(forward.D, "D");

    // Vectors too.
    ResVec rhs(numCells);
    for (std::size_t i = 0; i < numCells; ++i) {
        for (int j = 0; j < numRes; ++j) {
            rhs[i][j] = 1.0 + 0.5 * i - j;
        }
    }
    const auto rhsPath = (dir / "rhs_res.mm").string();
    Opm::writeVectorMarketFile(rhs, rhsPath);
    ResVec restoredRhs;
    Opm::readBlockedVectorMarket(restoredRhs, rhsPath);
    BOOST_REQUIRE_EQUAL(restoredRhs.size(), rhs.size());
    for (std::size_t i = 0; i < rhs.size(); ++i) {
        for (int j = 0; j < numRes; ++j) {
            BOOST_CHECK_EQUAL(restoredRhs[i][j], rhs[i][j]);
        }
    }

    std::filesystem::remove_all(dir);
}

//! `system_cprt` must actually solve the transposed coupled system.
BOOST_AUTO_TEST_CASE(SystemCprtSolvesTransposedSystem)
{
    const std::size_t nWell = numWellDofs();
    SysVec b, z;
    resizeSystemVector(b, numCells, nWell);
    resizeSystemVector(z, numCells, nWell);
    fillPseudoRandom(b, 20260726u);

    const auto result = solveSystemCprt(St, At, b, z);
    BOOST_CHECK(result.converged);

    const double relRes = relativeResidual(St, z, b);
    BOOST_TEST_MESSAGE("system_cprt: " << result.iterations
                       << " iterations, relative residual " << relRes);
    BOOST_CHECK_SMALL(relRes, 1e-9);
}

#if HAVE_SUITESPARSE_UMFPACK
//! Cross-check against a direct solve of the assembled transposed system.
BOOST_AUTO_TEST_CASE(SystemCprtMatchesDirectSolve)
{
    const std::size_t nWell = numWellDofs();
    SysVec b, z;
    resizeSystemVector(b, numCells, nWell);
    resizeSystemVector(z, numCells, nWell);
    fillPseudoRandom(b, 13579u);

    solveSystemCprt(St, At, b, z);

    auto flat = assembleScalar(St);
    const std::size_t dim = numCells * numRes + nWell * numWell;
    Dune::BlockVector<Dune::FieldVector<Scalar, 1>> flatRhs(dim), flatSol(dim);
    flatSol = 0.0;
    for (std::size_t i = 0; i < numCells; ++i) {
        for (int j = 0; j < numRes; ++j) {
            flatRhs[i * numRes + j] = b[_0][i][j];
        }
    }
    for (std::size_t i = 0; i < nWell; ++i) {
        for (int j = 0; j < numWell; ++j) {
            flatRhs[numCells * numRes + i * numWell + j] = b[_1][i][j];
        }
    }

    Dune::UMFPack<ScalarMatrix> direct(flat, 0);
    Dune::InverseOperatorResult directResult;
    direct.apply(flatSol, flatRhs, directResult);

    double maxDiff = 0.0;
    double scale = 0.0;
    for (std::size_t i = 0; i < numCells; ++i) {
        for (int j = 0; j < numRes; ++j) {
            const double ref = flatSol[i * numRes + j];
            maxDiff = std::max(maxDiff, std::abs(z[_0][i][j] - ref));
            scale = std::max(scale, std::abs(ref));
        }
    }
    for (std::size_t i = 0; i < nWell; ++i) {
        for (int j = 0; j < numWell; ++j) {
            const double ref = flatSol[numCells * numRes + i * numWell + j];
            maxDiff = std::max(maxDiff, std::abs(z[_1][i][j] - ref));
            scale = std::max(scale, std::abs(ref));
        }
    }
    BOOST_CHECK_SMALL(maxDiff / std::max(scale, 1e-300), 1e-8);
}
#endif // HAVE_SUITESPARSE_UMFPACK

//! The algebraic identity the adjoint depends on: eliminating the well
//! right-hand side up front does not change the reservoir adjoint.
//!
//!   [A^T B^T; C^T D^T] [lambda_r; lambda_w] = [g_r; g_w]
//! and
//!   [A^T B^T; C^T D^T] [lambda_r; nu_w]     = [g_r - B^T D^-T g_w; 0]
//!
//! have the same lambda_r. The adjoint's existing right-hand side already is
//! the shifted one (addWellObjectiveRhs_ applies B^T D^-T dJ/dx_w), so it can
//! be handed to the coupled solve unchanged with a zero well block.
BOOST_AUTO_TEST_CASE(SchurShiftedRhsGivesSameReservoirAdjoint)
{
    const std::size_t nWell = numWellDofs();

    SysVec unshifted, shifted, zUnshifted, zShifted;
    resizeSystemVector(unshifted, numCells, nWell);
    resizeSystemVector(shifted, numCells, nWell);
    resizeSystemVector(zUnshifted, numCells, nWell);
    resizeSystemVector(zShifted, numCells, nWell);
    fillPseudoRandom(unshifted, 8080u);

    // mu_w = D^-T g_w ; shifted_r = g_r - B^T mu_w ; shifted_w = 0
    const auto mu = solveTransposedD(forward.D, unshifted[_1]);
    shifted[_0] = unshifted[_0];
    transposed.Bt.mmv(mu, shifted[_0]);   // shifted_r -= B^T mu
    shifted[_1] = 0.0;

    solveSystemCprt(St, At, unshifted, zUnshifted);
    solveSystemCprt(St, At, shifted, zShifted);

    double maxDiff = 0.0;
    double scale = 0.0;
    for (std::size_t i = 0; i < numCells; ++i) {
        for (int j = 0; j < numRes; ++j) {
            maxDiff = std::max(maxDiff,
                               std::abs(zUnshifted[_0][i][j] - zShifted[_0][i][j]));
            scale = std::max(scale, std::abs(zUnshifted[_0][i][j]));
        }
    }
    BOOST_CHECK_SMALL(maxDiff / std::max(scale, 1e-300), 1e-8);

    // And the discarded well part is recovered by lambda_w = mu_w + nu_w.
    double maxWellDiff = 0.0;
    double wellScale = 0.0;
    for (std::size_t i = 0; i < nWell; ++i) {
        for (int j = 0; j < numWell; ++j) {
            const double recovered = mu[i][j] + zShifted[_1][i][j];
            maxWellDiff = std::max(maxWellDiff,
                                   std::abs(recovered - zUnshifted[_1][i][j]));
            wellScale = std::max(wellScale, std::abs(zUnshifted[_1][i][j]));
        }
    }
    BOOST_CHECK_SMALL(maxWellDiff / std::max(wellScale, 1e-300), 1e-8);
}

BOOST_AUTO_TEST_SUITE_END()
