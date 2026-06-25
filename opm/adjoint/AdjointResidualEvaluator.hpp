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
 * \brief Standalone, value-type-generic, residual-only cell evaluator.
 *
 * Recomputes the converged TPFA black-oil residual one cell at a time,
 * reproducing exactly what TpfaLinearizer::linearize_cell +
 * linearize_source_terms assemble, but WITHOUT the Jacobian and WITHOUT
 * the storage cache (the adjoint always runs --enable-storage-cache=false,
 * so the residual is the pure function R(x_k, x_{k-1})):
 *
 *   R_i = Σ_faces computeFlux(i,j)·faceArea
 *       + (computeStorage(IQ_i^0) − computeStorage(IQ_i^1))·V_i/dt
 *       − V_i·computeSource(i)
 *
 * The per-cell residual only reads the cell's own intensive quantities and
 * those of its TPFA neighbours (the ~7-point stencil from
 * linearizer().getNeighborInfo()), so it is the building block for
 * stencil-local parameter gradients (cell × ~7 instead of cell × Ncells).
 *
 * v1 is a Scalar "value pass" over the model's converged intensive-quantity
 * cache; it is validated bitwise against linearizer().residual(). The
 * evaluation is organised around an *intensive-quantity provider* so the
 * same code can later be instantiated with a seeded AD evaluation type
 * (state-seeded for the Jacobian, parameter-seeded for exact gradients)
 * without touching the assembly logic.
 */
#ifndef OPM_ADJOINT_RESIDUAL_EVALUATOR_HPP
#define OPM_ADJOINT_RESIDUAL_EVALUATOR_HPP

#include <opm/models/utils/propertysystem.hh>

#include <dune/common/fvector.hh>

#include <cstddef>

namespace Opm {

//! \brief Residual-only re-evaluation of the converged TPFA black-oil system,
//!        cell by cell, mirroring linearize_cell without the Jacobian/cache.
template<class TypeTag>
class AdjointResidualEvaluator
{
public:
    using Simulator = GetPropType<TypeTag, Properties::Simulator>;
    using Scalar = GetPropType<TypeTag, Properties::Scalar>;
    using Evaluation = GetPropType<TypeTag, Properties::Evaluation>;
    using LocalResidual = GetPropType<TypeTag, Properties::LocalResidual>;
    using RateVector = GetPropType<TypeTag, Properties::RateVector>;
    using GlobalEqVector = GetPropType<TypeTag, Properties::GlobalEqVector>;
    using IntensiveQuantities = GetPropType<TypeTag, Properties::IntensiveQuantities>;

    static constexpr int numEq = getPropValue<TypeTag, Properties::NumEq>();
    using EqVector = Dune::FieldVector<Scalar, numEq>;

    explicit AdjointResidualEvaluator(Simulator& simulator)
        : simulator_(simulator)
    {}

    //! \brief Default intensive-quantity provider: the model's converged
    //!        cache. iq(globalIdx, timeIdx) returns the stored IQ.
    //!
    //! A perturbed/seeded provider (Steps 2/3) returns a locally-built IQ
    //! for the target cell and delegates to the cache for every other cell.
    struct CacheIQProvider
    {
        const Simulator& simulator;
        const IntensiveQuantities& operator()(unsigned globalIdx,
                                              unsigned timeIdx) const
        { return simulator.model().intensiveQuantities(globalIdx, timeIdx); }
    };

    //! \brief Residual of one cell, Scalar value pass over the cache.
    void evalCellResidual(unsigned globI, EqVector& res) const
    {
        evalCellResidual(globI, CacheIQProvider{simulator_}, res);
    }

    //! \brief Residual of one cell with a caller-supplied IQ provider.
    //!        Reproduces linearize_cell (flux + storage) and
    //!        linearize_source_terms (source) for cell \p globI.
    template<class IQProvider>
    void evalCellResidual(unsigned globI, const IQProvider& iq,
                          EqVector& res) const
    {
        const auto& model = simulator_.model();
        const auto& problem = simulator_.problem();
        const auto& neighborInfo = model.linearizer().getNeighborInfo();

        res = 0.0;

        // --- flux term: Σ_faces computeFlux(i,j)·faceArea ---
        const auto& iqIn = iq(globI, /*timeIdx=*/0);
        RateVector flux(0.0);
        RateVector darcy(0.0);
        for (const auto& nbInfo : neighborInfo[globI]) {
            const unsigned globJ = nbInfo.neighbor;
            const auto& iqEx = iq(globJ, /*timeIdx=*/0);
            flux = 0.0;
            darcy = 0.0;
            LocalResidual::computeFlux(flux, darcy, globI, globJ, iqIn, iqEx,
                                       nbInfo.res_nbinfo, problem.moduleParams());
            const Scalar faceArea = nbInfo.res_nbinfo.faceArea;
            for (int eq = 0; eq < numEq; ++eq) {
                res[eq] += scalarValue_(flux[eq]) * faceArea;
            }
        }

        // --- accumulation term: (storage(x0) − storage(x1))·V/dt ---
        // No storage cache (adjoint forces --enable-storage-cache=false), so
        // the old-time mass is recomputed from IQ at time index 1, exactly
        // like the cache-free branch of linearize_cell.
        EqVector storageNew(0.0);
        EqVector storageOld(0.0);
        LocalResidual::template computeStorage<Scalar>(storageNew, iq(globI, 0));
        LocalResidual::template computeStorage<Scalar>(storageOld, iq(globI, 1));
        const Scalar volume = model.dofTotalVolume(globI);
        const Scalar storefac = volume / simulator_.timeStepSize();
        for (int eq = 0; eq < numEq; ++eq) {
            res[eq] += storefac * (storageNew[eq] - storageOld[eq]);
        }

        // --- source term: −V·computeSource(i) ---
        // computeSource routes through problem.source(), which carries the
        // well perforation rates (SeparateSparseSourceTerms defaults to
        // false), so the reservoir-side well coupling is included exactly as
        // linearize_source_terms does (adres *= -volume).
        RateVector source(0.0);
        LocalResidual::computeSource(source, problem, iqIn, globI, /*timeIdx=*/0);
        for (int eq = 0; eq < numEq; ++eq) {
            res[eq] -= volume * scalarValue_(source[eq]);
        }
    }

    //! \brief Residual of all grid cells (value pass), sized to numGridDof.
    void evalResidual(GlobalEqVector& out) const
    {
        const std::size_t numCells = simulator_.model().numGridDof();
        out.resize(numCells);
        EqVector res(0.0);
        for (std::size_t i = 0; i < numCells; ++i) {
            evalCellResidual(static_cast<unsigned>(i), res);
            out[i] = res;
        }
    }

private:
    //! \brief Scalar value of an Evaluation or a plain Scalar (so the same
    //!        code compiles for the value pass and, later, AD).
    static Scalar scalarValue_(const Evaluation& e) { return e.value(); }
    static Scalar scalarValue_(Scalar s) { return s; }

    Simulator& simulator_;
};

} // namespace Opm

#endif // OPM_ADJOINT_RESIDUAL_EVALUATOR_HPP
