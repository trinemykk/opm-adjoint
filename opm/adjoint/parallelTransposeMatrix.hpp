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
 * \brief Build an explicit, parallel-CONSISTENT transpose M = A^T of a
 *        distributed BCRS matrix.
 *
 * The forward simulation assembles complete *rows* for interior cells
 * with a single overlap layer. The transposed (adjoint) system needs
 * complete *columns*, i.e. complete interior *rows* of M = A^T: row i of
 * M needs M[i][k] = (A[k][i])^T for every neighbour k. When k is interior
 * on another rank, A[k][i] is assembled there and is missing from this
 * rank's (incomplete) ghost row k. A plain local transpose therefore
 * drops the cross-rank coupling (verified: wrong gradients for np>=3).
 *
 * This routine recovers the missing coupling at the MATRIX level - the
 * exact analogue of what OwnerOverlapCopyCommunication::addOwnerCopyToOwnerCopy
 * does at the action level, but transposed: each rank sends its
 * interior-row blocks A[k][c] that sit in a remotely-owned column c to the
 * owner of c, which stores them as M[c][k]. The result is a complete M
 * whose interior (owned) rows equal the serial transpose exactly, with the
 * ordinary 1-overlap sparsity - so M can be wrapped in the standard
 * Opm::GhostLastMatrixAdapter and handed to opm's parallel CPR unchanged.
 *
 * The ghost rows of M (local index >= interiorSize) are left with only a
 * zero diagonal block here; the caller sets them to identity
 * (invalidateRows_) exactly as the forward parallel solver does. The
 * consistency of the ghost *columns* inside interior rows (which this
 * routine fills) is what makes M actually equal A^T; those are never
 * zeroed.
 */
#ifndef OPM_PARALLEL_TRANSPOSE_MATRIX_HPP
#define OPM_PARALLEL_TRANSPOSE_MATRIX_HPP

#if HAVE_MPI

#include <opm/common/ErrorMacros.hpp>

#include <opm/adjoint/transposeMatrix.hpp>

#include <dune/istl/bcrsmatrix.hh>
#include <dune/istl/matrixindexset.hh>
#include <dune/istl/bvector.hh>
#include <dune/istl/owneroverlapcopy.hh>
#include <dune/common/fvector.hh>

#include <mpi.h>

#include <fmt/format.h>

#include <cstddef>
#include <cstdlib>
#include <iostream>
#include <numeric>
#include <unordered_map>
#include <vector>

namespace Opm {

//! \brief Number of interior (owner-attributed) local cells. Relies on
//!        flow's ghost-last ordering (owner indices are [0, interiorSize)).
template<class Comm>
inline std::size_t adjointInteriorSize(const Comm& comm)
{
    constexpr auto owner = Dune::OwnerOverlapCopyAttributeSet::owner;
    std::size_t maxOwner = 0;
    std::size_t numOwner = 0;
    const auto& indexSet = comm.indexSet();
    for (auto idx = indexSet.begin(); idx != indexSet.end(); ++idx) {
        if (idx->local().attribute() == owner) {
            maxOwner = std::max<std::size_t>(maxOwner, idx->local().local());
            ++numOwner;
        }
    }
    if (numOwner > 0 && numOwner != maxOwner + 1) {
        OPM_THROW(std::logic_error,
                  "Adjoint parallel transpose requires ghost-last cell "
                  "ordering (owner indices contiguous from 0)");
    }
    return numOwner;
}

//! \brief Build the parallel-consistent transpose M = A^T.
//!
//! \tparam Matrix BCRS matrix with square dense blocks (double scalar).
//! \tparam Comm   Dune::OwnerOverlapCopyCommunication<int,int>.
//!
//! On return, the owned rows [0, interiorSize) of \p transpose equal the
//! serial transpose of the global matrix; ghost rows carry only a zero
//! diagonal (the caller makes them identity). \p transpose is rebuilt from
//! scratch (pattern + values) on every call; the pattern is constant
//! across the backward sweep so a caller may cache the result and refill.
//!
//! Exactness note: the matrix sparsity includes every grid neighbour
//! unconditionally, but the parallel overlap only includes neighbours with
//! nonzero transmissibility (NNCs below ~1e-6 trans are zeroed). So a few
//! cross-rank couplings can be structurally present with a partner cell that
//! is not in the receiver's overlap - but for a black-oil model those blocks
//! are exactly zero, so the result is still the exact transpose. Such an
//! entry is dropped only after verifying its block is zero; a NONZERO
//! unplaceable coupling (e.g. diffusion/thermal on a below-trans NNC) throws
//! rather than being silently dropped.
template<class Matrix, class Comm>
void buildConsistentParallelTranspose(const Matrix& forward,
                                      const Comm& comm,
                                      Matrix& transpose)
{
    using Block = typename Matrix::block_type;
    static_assert(static_cast<int>(Block::rows) == static_cast<int>(Block::cols),
                  "buildConsistentParallelTranspose requires square blocks");
    constexpr int bs = static_cast<int>(Block::rows);
    constexpr int bs2 = bs * bs;

    const MPI_Comm mpicomm = comm.communicator();
    const int rank = comm.communicator().rank();
    const int nproc = comm.communicator().size();
    const std::size_t N = forward.N();
    const std::size_t interiorSize = adjointInteriorSize(comm);

    // --- global ids of shared cells + global->local map -----------------
    // (Every cell involved in the exchange is shared and therefore in the
    // parallel index set; purely-interior non-shared cells are never sent
    // nor received, so a partial map is sufficient.)
    std::vector<long> globalId(N, -1);
    std::unordered_map<long, std::size_t> globalToLocal;
    {
        const auto& pis = comm.indexSet();
        for (auto idx = pis.begin(); idx != pis.end(); ++idx) {
            const std::size_t loc = idx->local().local();
            const long gid = static_cast<long>(idx->global());
            globalId[loc] = gid;
            globalToLocal[gid] = loc;
        }
    }
    if (std::getenv("OPM_ADJOINT_TRANSPOSE_DIAG")) {
        std::size_t covered = 0;
        for (std::size_t i = 0; i < N; ++i) {
            if (globalId[i] >= 0) {
                ++covered;
            }
        }
        std::cerr << "[transpose-diag] rank " << rank << ": N=" << N
                  << " interiorSize=" << interiorSize
                  << " indexSet-covered=" << covered
                  << " (uncovered=" << (N - covered) << ")\n";
    }

    // --- owner rank of every local cell via copyOwnerToAll --------------
    // Owned entries keep this rank; copy entries receive their owner's
    // rank. Robust and avoids iterating the remote-index lists.
    std::vector<int> ownerRank(N, rank);
    {
        using OwnerVec = Dune::BlockVector<Dune::FieldVector<double, 1>>;
        OwnerVec ov(N);
        for (std::size_t i = 0; i < N; ++i) {
            ov[i] = static_cast<double>(rank);
        }
        comm.copyOwnerToAll(ov, ov);
        for (std::size_t i = 0; i < N; ++i) {
            ownerRank[i] = static_cast<int>(std::lround(ov[i][0]));
        }
    }

    // --- pack: send interior-row blocks that live in a remote column ----
    // For interior row k and column c owned by another rank, send
    // (globalId[k], globalId[c], block A[k][c]) to owner(c).
    std::vector<std::vector<long>> sendIds(nproc);
    std::vector<std::vector<double>> sendVals(nproc);
    for (auto row = forward.begin();
         row != forward.end() && row.index() < interiorSize; ++row) {
        const std::size_t k = row.index();
        const long gk = globalId[k];
        for (auto col = (*row).begin(); col != (*row).end(); ++col) {
            const std::size_t c = col.index();
            const int owner_c = ownerRank[c];
            if (owner_c == rank) {
                continue; // interior-interior: filled locally below
            }
            sendIds[owner_c].push_back(gk);
            sendIds[owner_c].push_back(globalId[c]);
            const Block& b = *col;
            for (int i = 0; i < bs; ++i) {
                for (int j = 0; j < bs; ++j) {
                    sendVals[owner_c].push_back(b[i][j]);
                }
            }
        }
    }

    // --- exchange (counts, then payload) --------------------------------
    std::vector<int> scId(nproc), scVal(nproc);
    for (int p = 0; p < nproc; ++p) {
        scId[p] = static_cast<int>(sendIds[p].size());
        scVal[p] = static_cast<int>(sendVals[p].size());
    }
    std::vector<int> rcId(nproc), rcVal(nproc);
    MPI_Alltoall(scId.data(), 1, MPI_INT, rcId.data(), 1, MPI_INT, mpicomm);
    MPI_Alltoall(scVal.data(), 1, MPI_INT, rcVal.data(), 1, MPI_INT, mpicomm);

    std::vector<int> sdId(nproc, 0), sdVal(nproc, 0);
    std::vector<int> rdId(nproc, 0), rdVal(nproc, 0);
    for (int p = 1; p < nproc; ++p) {
        sdId[p] = sdId[p - 1] + scId[p - 1];
        sdVal[p] = sdVal[p - 1] + scVal[p - 1];
        rdId[p] = rdId[p - 1] + rcId[p - 1];
        rdVal[p] = rdVal[p - 1] + rcVal[p - 1];
    }
    const int totSId = sdId[nproc - 1] + scId[nproc - 1];
    const int totSVal = sdVal[nproc - 1] + scVal[nproc - 1];
    const int totRId = rdId[nproc - 1] + rcId[nproc - 1];
    const int totRVal = rdVal[nproc - 1] + rcVal[nproc - 1];

    std::vector<long> sId(totSId);
    std::vector<double> sVal(totSVal);
    for (int p = 0; p < nproc; ++p) {
        std::copy(sendIds[p].begin(), sendIds[p].end(), sId.begin() + sdId[p]);
        std::copy(sendVals[p].begin(), sendVals[p].end(), sVal.begin() + sdVal[p]);
    }
    std::vector<long> rId(totRId);
    std::vector<double> rVal(totRVal);
    MPI_Alltoallv(sId.data(), scId.data(), sdId.data(), MPI_LONG,
                  rId.data(), rcId.data(), rdId.data(), MPI_LONG, mpicomm);
    MPI_Alltoallv(sVal.data(), scVal.data(), sdVal.data(), MPI_DOUBLE,
                  rVal.data(), rcVal.data(), rdVal.data(), MPI_DOUBLE, mpicomm);

    // --- build M's sparsity pattern -------------------------------------
    // interior-interior:  M[c][k] = (A[k][c])^T for interior rows k and
    //                     interior columns c (both owned here).
    // received:           M[i][k] = (A[k][i])^T for owned i, ghost k.
    // ghost rows:         zero diagonal only (caller -> identity).
    Dune::MatrixIndexSet indices(N, N);
    for (auto row = forward.begin();
         row != forward.end() && row.index() < interiorSize; ++row) {
        const std::size_t k = row.index();
        for (auto col = (*row).begin(); col != (*row).end(); ++col) {
            const std::size_t c = col.index();
            if (ownerRank[c] == rank) {
                indices.add(c, k);
            }
        }
    }
    const int nRecv = totRId / 2;
    std::vector<std::size_t> recvI(nRecv), recvK(nRecv);
    double maxDroppedBlockAbs = 0.0; // diag: magnitude of dropped coupling blocks
    for (int e = 0; e < nRecv; ++e) {
        const long gk = rId[2 * e];
        const long gc = rId[2 * e + 1];
        const auto itI = globalToLocal.find(gc); // owned here
        const auto itK = globalToLocal.find(gk); // ghost here
        if (itI == globalToLocal.end() || itK == globalToLocal.end()) {
            // Unplaceable coupling: the sender's row-cell k is not in the
            // receiver's index set. This is EXPECTED and HARMLESS for a
            // zero-transmissibility NNC entry: the matrix sparsity includes
            // every grid neighbour unconditionally (tpfalinearizer inserts the
            // neighbour with no trans test), but the parallel overlap only
            // includes neighbours with nonzero transmissibility
            // (addOverlapLayerNoZeroTrans; NNCs below ~1e-6 trans are zeroed).
            // So such an entry is structurally present but its block is exactly
            // zero -> dropping it is EXACT. Guard: if the block is NOT zero
            // (e.g. diffusion/thermal on a below-trans NNC keeps a nonzero
            // block), we cannot represent it and must not silently drop it.
            double blockAbs = 0.0;
            for (int b = 0; b < bs2; ++b) {
                blockAbs = std::max(blockAbs,
                    std::abs(rVal[static_cast<std::size_t>(e) * bs2 + b]));
            }
            maxDroppedBlockAbs = std::max(maxDroppedBlockAbs, blockAbs);
            constexpr double droppedBlockTol = 1e-10;
            if (blockAbs > droppedBlockTol) {
                OPM_THROW(std::logic_error, fmt::format(
                    "Adjoint parallel transpose: a NONZERO cross-rank coupling "
                    "(gk={}, gc={}, |block|={:.3e}) has its partner cell outside "
                    "this rank's transmissibility-based overlap. This can occur "
                    "when a coupling (e.g. diffusion/thermal on a below-trans "
                    "NNC) is not reflected in the overlap. Widen the overlap "
                    "inclusion test or use the matrix-free operator.",
                    gk, gc, blockAbs));
            }
            recvI[e] = static_cast<std::size_t>(-1);
            recvK[e] = static_cast<std::size_t>(-1);
            continue;
        }
        recvI[e] = itI->second;
        recvK[e] = itK->second;
        indices.add(recvI[e], recvK[e]);
    }
    for (std::size_t i = interiorSize; i < N; ++i) {
        indices.add(i, i); // ghost-row diagonal placeholder
    }
    indices.exportIdx(transpose);
    transpose = 0.0;

    // --- fill values ----------------------------------------------------
    for (auto row = forward.begin();
         row != forward.end() && row.index() < interiorSize; ++row) {
        const std::size_t k = row.index();
        for (auto col = (*row).begin(); col != (*row).end(); ++col) {
            const std::size_t c = col.index();
            if (ownerRank[c] == rank) {
                transpose[c][k] = transposeBlock(*col);
            }
        }
    }
    std::size_t skipped = 0;
    for (int e = 0; e < nRecv; ++e) {
        if (recvI[e] == static_cast<std::size_t>(-1)) {
            ++skipped;
            continue;
        }
        Block b;
        for (int i = 0; i < bs; ++i) {
            for (int j = 0; j < bs; ++j) {
                b[i][j] = rVal[static_cast<std::size_t>(e) * bs2 + i * bs + j];
            }
        }
        // received block is A[k][i]; store M[i][k] = (A[k][i])^T
        transpose[recvI[e]][recvK[e]] = transposeBlock(b);
    }
    if (std::getenv("OPM_ADJOINT_TRANSPOSE_DIAG")) {
        std::cerr << "[transpose-diag] rank " << rank << ": received "
                  << nRecv << " couplings, skipped " << skipped
                  << " unplaceable; max |dropped block entry| = "
                  << maxDroppedBlockAbs << "\n";
    }
}

} // namespace Opm

#endif // HAVE_MPI

#endif // OPM_PARALLEL_TRANSPOSE_MATRIX_HPP
