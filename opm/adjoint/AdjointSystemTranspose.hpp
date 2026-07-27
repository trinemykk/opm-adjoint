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
 * \brief Transpose of the coupled reservoir+well system, for the adjoint.
 *
 * The forward simulator can assemble the full 2x2 block system explicitly
 * (opm-simulators' `system_cpr` solver, `linalg/system/`):
 * \code
 *     S = [ A  C ]     A = reservoir-reservoir   (numEq x numEq blocks)
 *         [ B  D ]     C = reservoir-well        B = well-reservoir
 *                      D = well-well
 * \endcode
 * The adjoint of one substep needs S^T = [ A^T  B^T ; C^T  D^T ]. The block
 * shapes map onto each other exactly - B^T has C's shape and C^T has B's -
 * so the transpose is again an `Opm::SystemMatrix` and every existing
 * operator, preconditioner and solver instantiation applies unchanged:
 *
 *   | slot    | forward | adjoint | type       |
 *   |---------|---------|---------|------------|
 *   | (0,0) A | A       | A^T     | RRMatrix   |
 *   | (0,1) C | C       | B^T     | RWMatrix   |
 *   | (1,0) B | B       | C^T     | WRMatrix   |
 *   | (1,1) D | D       | D^T     | WWMatrix   |
 *
 * Keeping the wells in the system means the adjoint no longer folds them
 * through the Schur complement (`addWellContributions`, which requires
 * `--matrix-add-well-contributions=true` to have reserved the cell-to-cell
 * sparsity a multi-perforation well needs).
 *
 * The reservoir block A^T is supplied by the caller: the adjoint linear
 * solver already builds it, with `transposeBlockMatrixInto` in serial and
 * `buildConsistentParallelTranspose` in parallel.
 */
#ifndef OPM_ADJOINT_SYSTEM_TRANSPOSE_HPP
#define OPM_ADJOINT_SYSTEM_TRANSPOSE_HPP

#include <opm/adjoint/transposeMatrix.hpp>

#include <opm/simulators/linalg/FlowLinearSolverParameters.hpp>
#include <opm/simulators/linalg/PropertyTree.hpp>
#include <opm/simulators/linalg/setupPropertyTree.hpp>
#include <opm/simulators/linalg/system/SystemTypes.hpp>
#include <opm/simulators/linalg/system/WellMatrixMerger.hpp>

#include <opm/grid/utility/SparseTable.hpp>

#include <dune/common/fmatrix.hh>
#include <dune/istl/bcrsmatrix.hh>
#include <dune/istl/matrixindexset.hh>

#include <cstddef>
#include <vector>

namespace Opm {

//! \brief The three well-side blocks of the coupled system, in the forward
//!        (untransposed) layout used by `Opm::SystemMatrix`.
template<class Scalar>
struct SystemWellBlocks
{
    WRMatrix<Scalar> B;  //!< (1,0) well-reservoir, numWellDofs x numResDofs blocks
    RWMatrix<Scalar> C;  //!< (0,1) reservoir-well, numResDofs x numWellDofs blocks
    WWMatrix<Scalar> D;  //!< (1,1) well-well

    //! \brief Total number of well degrees of freedom (block rows of D).
    std::size_t numWellDofs() const { return D.N(); }
};

//! \brief The same three blocks transposed, ready to be plugged into the
//!        adjoint `SystemMatrix` (note the swap: B^T goes in the C slot).
template<class Scalar>
struct TransposedWellBlocks
{
    RWMatrix<Scalar> Bt;  //!< B^T - goes in the (0,1) slot
    WRMatrix<Scalar> Ct;  //!< C^T - goes in the (1,0) slot
    WWMatrix<Scalar> Dt;  //!< D^T - goes in the (1,1) slot
};

//! \brief Transpose the well blocks of the coupled system.
//!
//! `transposeBlockMatrixTo` handles rectangular matrices *and* rectangular
//! blocks (it sizes the index set as MxN and swaps the block indices), so
//! the non-square B/C blocks need no special treatment.
template<class Scalar>
void transposeWellBlocks(const SystemWellBlocks<Scalar>& forward,
                         TransposedWellBlocks<Scalar>& transposed)
{
    transposeBlockMatrixInto(forward.B, transposed.Bt);
    transposeBlockMatrixInto(forward.C, transposed.Ct);
    transposeBlockMatrixInto(forward.D, transposed.Dt);
}

//! \brief Assemble a `SystemMatrix` view from its four blocks.
//!
//! The arguments are named by *slot*, not by forward role: pass
//! (A^T, B^T, C^T, D^T) to obtain S^T.
template<class Scalar>
SystemMatrix<Scalar> makeSystemMatrix(const RRMatrix<Scalar>& blockA,
                                      const RWMatrix<Scalar>& blockC,
                                      const WRMatrix<Scalar>& blockB,
                                      const WWMatrix<Scalar>& blockD)
{
    SystemMatrix<Scalar> S;
    S.A = &blockA;
    S.C = &blockC;
    S.B = &blockB;
    S.D = &blockD;
    return S;
}

/*!
 * \brief The `system_cprt` property tree.
 *
 * opm's forward `system_cpr` tree with two changes:
 *  - the reservoir stage runs the TRANSPOSED CPR (`cprt`), since the block
 *    it preconditions is A^T;
 *  - quasi-impes instead of trueimpes weights, because trueimpes needs
 *    forward-simulator internals the adjoint does not carry.
 *
 * Note what is NOT changed: `add_wells` stays false in the reservoir stage,
 * so this uses the ordinary (transpose-capable) PressureTransferPolicy. That
 * is why the coupled adjoint solve needs no `cprwt`.
 *
 * Shared by the solver, the standalone example and the unit test so the three
 * cannot drift apart.
 */
inline PropertyTree setupAdjointSystemCPR(double reduction, int maxIter,
                                          int verbosity)
{
    FlowLinearSolverParameters params;
    params.reset();
    params.linear_solver_reduction_ = reduction;
    params.linear_solver_maxiter_ = maxIter;
    params.linear_solver_verbosity_ = verbosity;

    auto prm = setupSystemCPR("system_cpr", params);
    prm.put("preconditioner.reservoir_solver.preconditioner.type",
            std::string("cprt"));
    prm.put("preconditioner.reservoir_solver.preconditioner.weight_type",
            std::string("quasiimpes"));
    return prm;
}

//! \brief Scalar (1x1-blocked) flattening of a coupled system, for direct
//!        solvers and for file output.
template<class Scalar>
using FlatSystemMatrix = Dune::BCRSMatrix<Dune::FieldMatrix<Scalar, 1, 1>>;

//! \brief Number of scalar unknowns of a coupled system.
template<class Scalar>
std::size_t systemScalarSize(const SystemMatrix<Scalar>& S)
{
    return S.A->N() * numResDofs + S.D->N() * numWellDofs;
}

/*!
 * \brief Flatten [A C; B D] into a single scalar BCRS matrix.
 *
 * The reservoir unknowns come first (numResDofs per cell), the well unknowns
 * after them (numWellDofs per well DOF) - the same ordering as
 * `Opm::SystemVector`. Used by the direct oracle, which needs one matrix
 * rather than four blocks; too expensive for production-size problems, which
 * is exactly why it is only the verification path.
 */
template<class Scalar>
FlatSystemMatrix<Scalar> flattenSystem(const SystemMatrix<Scalar>& S)
{
    const std::size_t dim = systemScalarSize(S);
    const std::size_t wellOffset = S.A->N() * numResDofs;

    Dune::MatrixIndexSet indices(dim, dim);
    auto addBlock = [&indices](const auto& block, std::size_t rowOff,
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
    addBlock(*S.A, 0, 0, numResDofs, numResDofs);
    addBlock(*S.C, 0, wellOffset, numResDofs, numWellDofs);
    addBlock(*S.B, wellOffset, 0, numWellDofs, numResDofs);
    addBlock(*S.D, wellOffset, wellOffset, numWellDofs, numWellDofs);

    FlatSystemMatrix<Scalar> flat;
    indices.exportIdx(flat);
    flat = 0.0;

    auto fillBlock = [&flat](const auto& block, std::size_t rowOff,
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
    fillBlock(*S.A, 0, 0, numResDofs, numResDofs);
    fillBlock(*S.C, 0, wellOffset, numResDofs, numWellDofs);
    fillBlock(*S.B, wellOffset, 0, numWellDofs, numResDofs);
    fillBlock(*S.D, wellOffset, wellOffset, numWellDofs, numWellDofs);
    return flat;
}

//! \brief Scatter a SystemVector into the flat scalar ordering.
template<class Scalar, class FlatVector>
void flattenSystemVector(const SystemVector<Scalar>& v, FlatVector& flat)
{
    const auto& res = v[Dune::Indices::_0];
    const auto& well = v[Dune::Indices::_1];
    flat.resize(res.size() * numResDofs + well.size() * numWellDofs);
    for (std::size_t i = 0; i < res.size(); ++i) {
        for (int j = 0; j < numResDofs; ++j) {
            flat[i * numResDofs + j] = res[i][j];
        }
    }
    const std::size_t offset = res.size() * numResDofs;
    for (std::size_t i = 0; i < well.size(); ++i) {
        for (int j = 0; j < numWellDofs; ++j) {
            flat[offset + i * numWellDofs + j] = well[i][j];
        }
    }
}

//! \brief Gather the flat scalar ordering back into a SystemVector.
template<class Scalar, class FlatVector>
void unflattenSystemVector(const FlatVector& flat, SystemVector<Scalar>& v)
{
    auto& res = v[Dune::Indices::_0];
    auto& well = v[Dune::Indices::_1];
    for (std::size_t i = 0; i < res.size(); ++i) {
        for (int j = 0; j < numResDofs; ++j) {
            res[i][j] = flat[i * numResDofs + j];
        }
    }
    const std::size_t offset = res.size() * numResDofs;
    for (std::size_t i = 0; i < well.size(); ++i) {
        for (int j = 0; j < numWellDofs; ++j) {
            well[i][j] = flat[offset + i * numWellDofs + j];
        }
    }
}

/*!
 * \brief Collects the well blocks of the coupled system from the well model
 *        and keeps their transposes up to date.
 *
 * Mirrors `ISTLSolverSystem::prepareSystemSolver`: per-well B/C/D come from
 * the public `BlackoilWellModel::addBCDMatrix` hook and are concatenated by
 * `Opm::WellMatrixMerger`, which also reports whether the sparsity changed
 * (wells opening/closing, MSW segment topology). When it did not, only the
 * values are refilled - no allocation, and the solver's preconditioner can
 * be updated in place instead of rebuilt.
 *
 * The class owns both the merged forward blocks (needed for the system dump
 * and as the transpose source) and the transposed ones.
 */
template<class Scalar>
class AdjointWellSystem
{
public:
    //! \brief Refresh the merged well blocks and their transposes.
    //!
    //! \param numResDofs number of reservoir cells (block rows of A)
    //! \param wellModel  the simulator's BlackoilWellModel
    //! \return true if the sparsity structure changed (solver must rebuild)
    template<class WellModel>
    bool update(std::size_t numResDofs, const WellModel& wellModel)
    {
        perWellB_.clear();
        perWellC_.clear();
        perWellD_.clear();
        wellCells_.clear();
        wellModel.addBCDMatrix(perWellB_, perWellC_, perWellD_, wellCells_);

        const WellMatrixMerger<Scalar> merger(numResDofs, perWellB_, perWellC_,
                                              perWellD_, wellCells_);
        const bool changed = !initialized_ || !merger.hasSameStructure(structure_);
        if (changed) {
            merger.buildMatrices(forward_.B, forward_.C, forward_.D);
            structure_ = merger.buildStructure();
            initialized_ = true;
        } else {
            merger.updateValues(forward_.B, forward_.C, forward_.D);
        }
        transposeWellBlocks(forward_, transposed_);
        return changed;
    }

    //! \brief Drop the cached structure so the next update() rebuilds.
    void reset()
    {
        initialized_ = false;
        structure_ = WellMatrixStructure{};
    }

    const SystemWellBlocks<Scalar>& forward() const { return forward_; }
    const TransposedWellBlocks<Scalar>& transposed() const { return transposed_; }

    std::size_t numWellDofs() const { return forward_.numWellDofs(); }

    //! \brief S^T = [A^T B^T; C^T D^T] for a caller-supplied A^T.
    SystemMatrix<Scalar> adjointSystem(const RRMatrix<Scalar>& transposedA) const
    {
        return makeSystemMatrix<Scalar>(transposedA, transposed_.Bt,
                                        transposed_.Ct, transposed_.Dt);
    }

    //! \brief S = [A C; B D] for a caller-supplied A (used by the system dump
    //!        and by the transpose-identity check).
    SystemMatrix<Scalar> forwardSystem(const RRMatrix<Scalar>& forwardA) const
    {
        return makeSystemMatrix<Scalar>(forwardA, forward_.C,
                                        forward_.B, forward_.D);
    }

private:
    // Per-well blocks as handed out by the well model (inputs to the merger).
    std::vector<WRMatrix<Scalar>> perWellB_;
    std::vector<RWMatrix<Scalar>> perWellC_;
    std::vector<WWMatrix<Scalar>> perWellD_;
    SparseTable<int> wellCells_;

    SystemWellBlocks<Scalar> forward_;
    TransposedWellBlocks<Scalar> transposed_;

    WellMatrixStructure structure_;
    bool initialized_ = false;
};

} // namespace Opm

#endif // OPM_ADJOINT_SYSTEM_TRANSPOSE_HPP
