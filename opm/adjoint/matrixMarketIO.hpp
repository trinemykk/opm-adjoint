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
 * \brief MatrixMarket I/O for block matrices/vectors of any block shape.
 *
 * Writing is plain `Dune::storeMatrixMarket`. Reading goes through a scalar
 * (1x1-blocked) intermediate and re-blocks afterwards, for two reasons:
 *
 *  1. dune-istl's `readSparseEntries` (matrixmarket.hh) asserts
 *     `row/bcols < matrix.N()` where it means `row/brows`. For a matrix with
 *     more block rows than block columns - the well-reservoir block B is
 *     4x3 - that assert fires on perfectly valid input. OPM builds Release
 *     with `-UNDEBUG`, so asserts are live and this is not academic.
 *  2. MatrixMarket stores entries in scalar coordinate form regardless of the
 *     block shape, and the reader takes the block dimensions from the target
 *     *type*, not from the `% ISTL_STRUCT blocked i j` comment. So a scalar
 *     read is exact for any writer block shape, and re-blocking is a pure
 *     index remap.
 *
 * The files themselves stay standard MatrixMarket, and so remain loadable by
 * MATLAB/MRST, Julia and scipy.
 */
#ifndef OPM_ADJOINT_MATRIX_MARKET_IO_HPP
#define OPM_ADJOINT_MATRIX_MARKET_IO_HPP

#include <opm/common/ErrorMacros.hpp>

#include <opm/simulators/linalg/MatrixMarketSpecializations.hpp>

#include <dune/common/fmatrix.hh>
#include <dune/common/fvector.hh>
#include <dune/istl/bcrsmatrix.hh>
#include <dune/istl/bvector.hh>
#include <dune/istl/matrixindexset.hh>
#include <dune/istl/matrixmarket.hh>

#include <fmt/format.h>

#include <cstddef>
#include <fstream>
#include <limits>
#include <string>

namespace Opm {

namespace detail {

using ScalarMarketMatrix = Dune::BCRSMatrix<Dune::FieldMatrix<double, 1, 1>>;
using ScalarMarketVector = Dune::BlockVector<Dune::FieldVector<double, 1>>;

inline std::ifstream openMarketFile(const std::string& path)
{
    std::ifstream file(path);
    if (!file) {
        OPM_THROW(std::runtime_error,
                  fmt::format("Could not open MatrixMarket file '{}'", path));
    }
    return file;
}

} // namespace detail

//! \brief Decimal digits needed to round-trip a double exactly.
//!
//! `Dune::storeMatrixMarket` defaults to the stream's precision (6 significant
//! digits), which loses ~10 digits per entry. A dumped system must reproduce
//! the in-situ solve bit for bit, so always write full precision.
constexpr int matrixMarketPrecision = std::numeric_limits<double>::max_digits10;

//! \brief Write a (block) matrix in MatrixMarket format, at full precision.
//!
//! A path ending in .mm or .mtx is used verbatim; anything else gets ".mm"
//! appended (dune's convention).
template<class Matrix>
void writeMatrixMarketFile(const Matrix& matrix, const std::string& path)
{
    Dune::storeMatrixMarket(matrix, path, matrixMarketPrecision);
}

//! \brief Write a (block) vector in MatrixMarket format, at full precision.
template<class Vector>
void writeVectorMarketFile(const Vector& vector, const std::string& path)
{
    Dune::storeMatrixMarket(vector, path, matrixMarketPrecision);
}

//! \brief Read a MatrixMarket file into a BCRS matrix with arbitrary
//!        (possibly rectangular) dense blocks.
//!
//! The scalar dimensions in the file must be exact multiples of the target
//! block dimensions; a mismatch means the file was written for a different
//! block shape and is reported rather than silently reinterpreted.
template<class OutMatrix>
void readBlockedMatrixMarket(OutMatrix& matrix, const std::string& path)
{
    constexpr auto brows = static_cast<std::size_t>(OutMatrix::block_type::rows);
    constexpr auto bcols = static_cast<std::size_t>(OutMatrix::block_type::cols);

    detail::ScalarMarketMatrix scalar;
    {
        auto file = detail::openMarketFile(path);
        Dune::readMatrixMarket(scalar, file);
    }

    if (scalar.N() % brows != 0 || scalar.M() % bcols != 0) {
        OPM_THROW(std::runtime_error,
                  fmt::format("MatrixMarket file '{}' has scalar dimensions "
                              "{}x{}, which are not a multiple of the target "
                              "block size {}x{}",
                              path, scalar.N(), scalar.M(), brows, bcols));
    }
    const std::size_t nRows = scalar.N() / brows;
    const std::size_t nCols = scalar.M() / bcols;

    // A block is structurally present if any of its scalar entries is.
    Dune::MatrixIndexSet indices(nRows, nCols);
    for (auto row = scalar.begin(); row != scalar.end(); ++row) {
        for (auto entry = row->begin(); entry != row->end(); ++entry) {
            indices.add(row.index() / brows, entry.index() / bcols);
        }
    }
    indices.exportIdx(matrix);
    matrix = 0.0; // entries the file omits are structural zeros

    for (auto row = scalar.begin(); row != scalar.end(); ++row) {
        for (auto entry = row->begin(); entry != row->end(); ++entry) {
            matrix[row.index() / brows][entry.index() / bcols]
                  [row.index() % brows][entry.index() % bcols] = (*entry)[0][0];
        }
    }
}

//! \brief Read a MatrixMarket file into a block vector.
template<class OutVector>
void readBlockedVectorMarket(OutVector& vector, const std::string& path)
{
    constexpr auto bs = static_cast<std::size_t>(OutVector::block_type::dimension);

    detail::ScalarMarketVector scalar;
    {
        auto file = detail::openMarketFile(path);
        Dune::readMatrixMarket(scalar, file);
    }

    if (scalar.size() % bs != 0) {
        OPM_THROW(std::runtime_error,
                  fmt::format("MatrixMarket file '{}' has {} scalar entries, "
                              "not a multiple of the target block size {}",
                              path, scalar.size(), bs));
    }
    vector.resize(scalar.size() / bs);
    for (std::size_t i = 0; i < scalar.size(); ++i) {
        vector[i / bs][i % bs] = scalar[i][0];
    }
}

} // namespace Opm

#endif // OPM_ADJOINT_MATRIX_MARKET_IO_HPP
