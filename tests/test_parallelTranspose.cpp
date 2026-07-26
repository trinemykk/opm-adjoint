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
 * \brief Standalone correctness gate for buildConsistentParallelTranspose.
 *
 * Decoupled from the simulator: build a distributed 1-D block-tridiagonal
 * matrix (ghost-last local ordering, 1 overlap layer, 3x3 blocks), assemble
 * the explicit parallel-consistent transpose M = A^T, and check that every
 * owned (non-ghost) row of M equals the plain serial transpose of the
 * global matrix to machine precision. The global matrix is known
 * analytically, so each rank verifies its own rows independently.
 *
 * The local forward matrix assembles ONLY its interior (owned) rows - ghost
 * rows are left as identity, exactly as flow's per-rank assembly leaves the
 * off-rank couplings out. A plain local transpose therefore cannot recover
 * A[k][i] for a ghost k of interior i; the routine under test must
 * communicate it. Run on >=2 ranks to exercise that path.
 */
#include <config.h>

#if HAVE_MPI

#include <opm/adjoint/parallelTransposeMatrix.hpp>
#include <opm/adjoint/transposeMatrix.hpp>

#include <opm/simulators/linalg/matrixblock.hh>

#include <dune/common/parallel/mpihelper.hh>
#include <dune/common/parallel/plocalindex.hh>
#include <dune/istl/bcrsmatrix.hh>
#include <dune/istl/matrixindexset.hh>
#include <dune/istl/owneroverlapcopy.hh>

#include <cmath>
#include <cstddef>
#include <exception>
#include <iostream>
#include <vector>

namespace {

constexpr int BS = 3;
using Block = Opm::MatrixBlock<double, BS, BS>;
using Matrix = Dune::BCRSMatrix<Block>;
using Comm = Dune::OwnerOverlapCopyCommunication<int, int>;
using GridAttributes = Dune::OwnerOverlapCopyAttributeSet;
using LocalIndex = Dune::ParallelLocalIndex<GridAttributes::AttributeSet>;

//! Analytic global block A[r][c] - distinct, asymmetric entries so that the
//! transpose is unmistakably different from the original.
Block globalBlock(int r, int c)
{
    Block b(0.0);
    for (int i = 0; i < BS; ++i) {
        for (int j = 0; j < BS; ++j) {
            b[i][j] = 1.0 + 1000.0 * r + 100.0 * c + 10.0 * i + j;
        }
    }
    return b;
}

bool blocksEqual(const Block& a, const Block& b, double tol = 1e-9)
{
    for (int i = 0; i < BS; ++i) {
        for (int j = 0; j < BS; ++j) {
            if (std::abs(a[i][j] - b[i][j]) > tol) {
                return false;
            }
        }
    }
    return true;
}

//! Guard test (np==2): an owned interior cell A on rank 0 couples to a cell B
//! owned by rank 1, where A is deliberately NOT in rank 1's index set — the
//! structural NNC asymmetry. With a ZERO coupling block the builder must drop
//! it cleanly (exact); with a NONZERO block it must THROW rather than silently
//! drop a real coupling. Returns true iff the observed behaviour is correct.
bool guardScenario(const Dune::MPIHelper& mpi, bool nonzero)
{
    const int rank = mpi.rank();
    Comm comm(mpi.getCommunication());
    comm.indexSet().beginResize();
    if (rank == 0) {
        comm.indexSet().add(100, LocalIndex(0, GridAttributes::owner, true));
        comm.indexSet().add(200, LocalIndex(1, GridAttributes::copy, true));
    } else if (rank == 1) {
        comm.indexSet().add(200, LocalIndex(0, GridAttributes::owner, true));
    }
    comm.indexSet().endResize();
    comm.remoteIndices().template rebuild<false>();

    Matrix forward;
    if (rank == 0) {
        Dune::MatrixIndexSet p(2, 2);
        p.add(0, 0);
        p.add(0, 1); // interior row A (100) couples to B (200, owned by rank 1)
        p.add(1, 1);
        p.exportIdx(forward);
        forward = 0.0;
        for (int e = 0; e < BS; ++e) {
            forward[0][0][e][e] = 1.0;
            forward[1][1][e][e] = 1.0;
        }
        if (nonzero) {
            forward[0][1] = globalBlock(100, 200); // a real cross-rank coupling
        } // else: block stays exactly zero (zero-trans NNC analogue)
    } else if (rank == 1) {
        Dune::MatrixIndexSet p(1, 1);
        p.add(0, 0);
        p.exportIdx(forward);
        forward = 0.0;
        for (int e = 0; e < BS; ++e) {
            forward[0][0][e][e] = 1.0;
        }
    }

    bool threw = false;
    try {
        Matrix transpose;
        Opm::buildConsistentParallelTranspose(forward, comm, transpose);
    } catch (const std::exception&) {
        threw = true; // the throw happens after the collective exchange
    }

    // Expected: nonzero → rank 1 (owner of B) throws, rank 0 does not; zero → no throw anywhere.
    int correctLocal;
    if (nonzero) {
        correctLocal = (rank == 1) ? (threw ? 1 : 0) : (threw ? 0 : 1);
    } else {
        correctLocal = threw ? 0 : 1;
    }
    return mpi.getCommunication().min(correctLocal) == 1;
}

} // anonymous namespace

int main(int argc, char** argv)
{
    const auto& mpi = Dune::MPIHelper::instance(argc, argv);
    const int rank = mpi.rank();
    const int nproc = mpi.size();

    // Global 1-D chain of N cells, block-tridiagonal (|r-c| <= 1).
    constexpr int N = 12;

    // Owner region [istart, iend) for this rank (contiguous split).
    const int base = N / nproc;
    const int rem = N % nproc;
    const int istart = rank * base + std::min(rank, rem);
    const int iend = istart + base + (rank < rem ? 1 : 0);
    const int interiorSize = iend - istart;

    // Ghost-last local ordering: owned globals first, then the (at most two)
    // overlap neighbours.
    std::vector<int> localToGlobal(interiorSize);
    for (int l = 0; l < interiorSize; ++l) {
        localToGlobal[l] = istart + l;
    }
    if (istart > 0) {
        localToGlobal.push_back(istart - 1); // left ghost
    }
    if (iend < N) {
        localToGlobal.push_back(iend); // right ghost
    }
    const int nLoc = static_cast<int>(localToGlobal.size());

    std::vector<int> globalToLocal(N, -1);
    for (int l = 0; l < nLoc; ++l) {
        globalToLocal[localToGlobal[l]] = l;
    }

    // --- local forward matrix (interior rows complete; ghost rows identity) --
    Dune::MatrixIndexSet pattern(nLoc, nLoc);
    for (int l = 0; l < interiorSize; ++l) {
        const int g = localToGlobal[l];
        for (int gn = g - 1; gn <= g + 1; ++gn) {
            if (gn >= 0 && gn < N) {
                pattern.add(l, globalToLocal[gn]);
            }
        }
    }
    for (int l = interiorSize; l < nLoc; ++l) {
        pattern.add(l, l); // ghost row: identity placeholder (unused by builder)
    }
    Matrix forward;
    pattern.exportIdx(forward);
    forward = 0.0;
    for (int l = 0; l < interiorSize; ++l) {
        const int g = localToGlobal[l];
        for (int gn = g - 1; gn <= g + 1; ++gn) {
            if (gn >= 0 && gn < N) {
                forward[l][globalToLocal[gn]] = globalBlock(g, gn);
            }
        }
    }
    for (int l = interiorSize; l < nLoc; ++l) {
        for (int e = 0; e < BS; ++e) {
            forward[l][l][e][e] = 1.0;
        }
    }

    // --- parallel communication (owner/copy index set + remote indices) -----
    Comm comm(mpi.getCommunication());
    comm.indexSet().beginResize();
    for (int l = 0; l < nLoc; ++l) {
        const auto attr = (l < interiorSize) ? GridAttributes::owner
                                             : GridAttributes::copy;
        comm.indexSet().add(localToGlobal[l],
                            LocalIndex(l, attr, /*isPublic=*/true));
    }
    comm.indexSet().endResize();
    comm.remoteIndices().template rebuild<false>();

    // --- routine under test -------------------------------------------------
    Matrix transpose;
    Opm::buildConsistentParallelTranspose(forward, comm, transpose);

    // --- verify owned rows against the serial transpose ---------------------
    int localFailures = 0;
    for (int l = 0; l < interiorSize; ++l) {
        const int gi = localToGlobal[l];

        // Expected column set for row gi of A^T: the neighbours of gi.
        int expected = 0;
        for (int gj = gi - 1; gj <= gi + 1; ++gj) {
            if (gj >= 0 && gj < N) {
                ++expected;
            }
        }

        int seen = 0;
        for (auto col = transpose[l].begin(); col != transpose[l].end(); ++col) {
            const int gc = localToGlobal[col.index()];
            ++seen;
            // (A^T)[gi][gc] = (A[gc][gi])^T
            const Block ref = Opm::transposeBlock(globalBlock(gc, gi));
            if (std::abs(gc - gi) > 1 || !blocksEqual(*col, ref)) {
                ++localFailures;
                std::cerr << "rank " << rank << ": M[" << gi << "][" << gc
                          << "] wrong\n";
            }
        }
        if (seen != expected) {
            ++localFailures;
            std::cerr << "rank " << rank << ": row " << gi << " has " << seen
                      << " cols, expected " << expected << "\n";
        }
    }

    const int totalFailures = mpi.getCommunication().sum(localFailures);
    if (rank == 0) {
        if (totalFailures == 0) {
            std::cout << "parallel transpose (np=" << nproc
                      << ", 3x3 blocks): PASS\n";
        } else {
            std::cout << "parallel transpose (np=" << nproc
                      << "): FAIL (" << totalFailures << " mismatches)\n";
        }
    }

    // Guard test (np==2 only): zero-block unplaceable coupling → dropped;
    // nonzero-block unplaceable coupling → throws (no silent physics loss).
    bool guardOk = true;
    if (nproc == 2) {
        const bool dropOk = guardScenario(mpi, /*nonzero=*/false);
        const bool throwOk = guardScenario(mpi, /*nonzero=*/true);
        guardOk = dropOk && throwOk;
        if (rank == 0) {
            std::cout << "parallel transpose guard (zero→drop, nonzero→throw): "
                      << (guardOk ? "PASS" : "FAIL")
                      << " [drop=" << dropOk << " throw=" << throwOk << "]\n";
        }
    }

    return (totalFailures == 0 && guardOk) ? 0 : 1;
}

#else // !HAVE_MPI

#include <iostream>
int main()
{
    std::cout << "parallel transpose test skipped (no MPI)\n";
    return 0;
}

#endif // HAVE_MPI
