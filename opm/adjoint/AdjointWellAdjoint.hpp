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
 * \brief Well-type-agnostic linear-algebra helper for the well adjoint.
 *
 * Kept in its own header (task #16, MSW well-adjoint) so the MSW additions
 * stay isolated from the reservoir sweep body that the performance thread
 * also edits.
 */
#ifndef OPM_ADJOINT_WELL_ADJOINT_HPP
#define OPM_ADJOINT_WELL_ADJOINT_HPP

#include <dune/common/dynmatrix.hh>
#include <dune/common/dynvector.hh>

namespace Opm {
namespace AdjointWellAdjoint {

//! \brief Solve the transposed well system D^T x = rhs for a block-sparse
//!        well matrix D.
//!
//! A StandardWell has a single well "segment" (D is 1x1 of blocks); a
//! MultisegmentWell has D of shape numSeg x numSeg, each block
//! numWellEq x numWellEq, with off-diagonal entries encoding the segment
//! inlet/outlet topology. Both are Dune::BCRSMatrix, so this one routine
//! flattens either into a dense (numSeg*numWellEq) system, transposes it and
//! solves. Segments per well are few (typically O(10)), so the dense solve is
//! cheap and exact — the v1 choice (a sparse transposed factorisation reuse is
//! a later optimisation).
//!
//! \param D        block-sparse well matrix (numSeg x numSeg blocks).
//! \param rhs      flattened right-hand side, length numSeg*numWellEq.
//! \param numWellEq number of well equations per segment (block dimension).
template<class DMat, class Scalar>
Dune::DynamicVector<Scalar>
solveTransposedWellSystem(const DMat& D,
                          const Dune::DynamicVector<Scalar>& rhs,
                          const int numWellEq)
{
    const int numSeg = D.N();
    const int n = numSeg * numWellEq;

    // Assemble Dt = D^T densely. Block (i,j) of D lands at flat rows
    // [i*numWellEq, ..) and columns [j*numWellEq, ..); transposition swaps
    // both the block position and the within-block indices.
    Dune::DynamicMatrix<Scalar> Dt(n, n, 0.0);
    for (auto row = D.begin(); row != D.end(); ++row) {
        const int i = row.index();
        for (auto col = row->begin(); col != row->end(); ++col) {
            const int j = col.index();
            const auto& blk = *col; // numWellEq x numWellEq
            for (int a = 0; a < numWellEq; ++a) {
                for (int b = 0; b < numWellEq; ++b) {
                    Dt[j * numWellEq + b][i * numWellEq + a] = blk[a][b];
                }
            }
        }
    }

    Dune::DynamicVector<Scalar> x(n, 0.0);
    Dt.solve(x, rhs);
    return x;
}

} // namespace AdjointWellAdjoint
} // namespace Opm

#endif // OPM_ADJOINT_WELL_ADJOINT_HPP
