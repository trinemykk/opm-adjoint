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

#include <array>
#include <cstddef>
#include <type_traits>
#include <vector>

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

    //! \brief Null perturbation: every scale is 1 (used for the value pass
    //!        and as the base class of family perturbations).
    struct NoPerturb
    {
        Scalar pvScale(unsigned /*cell*/) const { return Scalar(1.0); }
        Scalar transScale(unsigned /*i*/, unsigned /*j*/) const
        { return Scalar(1.0); }
    };

    //! \brief Reservoir residual of one cell (flux + storage only, NO
    //!        source) with a parameter perturbation applied to the
    //!        transmissibility (flux term) and the pore-volume multiplier
    //!        (storage term).
    //!
    //! This is the stencil-local building block for parameter gradients:
    //! the residual of cell \p globI depends on a parameter θ only through
    //! the flux of its faces (trans) and its own storage (pore volume), and
    //! the well source is θ-independent for the PV/trans/perm families (so
    //! it cancels in a central difference and is omitted here). \p perturb
    //! supplies pvScale(cell) and transScale(i,j).
    //!
    //! The result element type follows the perturbation's scale type: a
    //! Scalar scale gives a Scalar residual (the FD path); a seeded
    //! Evaluation scale gives an Evaluation residual whose derivative is the
    //! exact dR/dθ (the AD path, Step 3). Because the TPFA flux is linear in
    //! trans, the trans scale is applied to the flux *value* post-hoc, which
    //! is exact and lets the scale be an AD type even though the kernel's
    //! transmissibility field is a plain double.
    template<class Perturb, class ResBlock>
    void evalCellResidualReservoir(unsigned globI, const Perturb& perturb,
                                   ResBlock& res) const
    {
        using Elem = std::decay_t<decltype(res[0])>;
        const auto& model = simulator_.model();
        const auto& problem = simulator_.problem();
        const auto& neighborInfo = model.linearizer().getNeighborInfo();

        for (int eq = 0; eq < numEq; ++eq) {
            res[eq] = Elem(0);
        }

        // --- flux term: trans scale applied to the (linear-in-trans) flux ---
        const auto& iqIn = model.intensiveQuantities(globI, /*timeIdx=*/0);
        RateVector flux(0.0);
        RateVector darcy(0.0);
        for (const auto& nbInfo : neighborInfo[globI]) {
            const unsigned globJ = nbInfo.neighbor;
            const auto& iqEx = model.intensiveQuantities(globJ, /*timeIdx=*/0);
            flux = 0.0;
            darcy = 0.0;
            LocalResidual::computeFlux(flux, darcy, globI, globJ, iqIn, iqEx,
                                       nbInfo.res_nbinfo, problem.moduleParams());
            const auto ts = perturb.transScale(globI, globJ);
            const Scalar faceArea = nbInfo.res_nbinfo.faceArea;
            for (int eq = 0; eq < numEq; ++eq) {
                res[eq] += ts * (scalarValue_(flux[eq]) * faceArea);
            }
        }

        // --- storage term with pore-volume scale ---
        EqVector storageNew(0.0);
        EqVector storageOld(0.0);
        LocalResidual::template computeStorage<Scalar>(
            storageNew, model.intensiveQuantities(globI, 0));
        LocalResidual::template computeStorage<Scalar>(
            storageOld, model.intensiveQuantities(globI, 1));
        const Scalar storefac = model.dofTotalVolume(globI)
                              / simulator_.timeStepSize();
        const auto pv = perturb.pvScale(globI);
        for (int eq = 0; eq < numEq; ++eq) {
            res[eq] += pv * (storefac * (storageNew[eq] - storageOld[eq]));
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

    //! \brief Assemble the full reservoir Jacobian into \p jac (which must
    //!        already have the linearizer's sparsity; it is zeroed first),
    //!        reproducing TpfaLinearizer::linearize_cell +
    //!        linearize_source_terms with the model's Evaluation intensive
    //!        quantities and the shared static kernels — no new kernel code,
    //!        no Evaluation-type changes in the simulator. This is the
    //!        forward "analytic derivative": dResidual/dstate, before the
    //!        well Schur complement.
    //!
    //! Block convention (mirrors the linearizer): the flux block of face
    //! (I,J) is added to the diagonal (I,I) and its negative to the
    //! off-diagonal (J,I); storage and source add to (I,I). The two-pass
    //! order (all flux+storage, then all source) matches the linearizer's
    //! linearize_/linearize_source_terms split for a bitwise result.
    template<class Matrix>
    void evalJacobian(Matrix& jac) const
    {
        // Default: use the model's cached intensive quantities.
        const auto& model = simulator_.model();
        evalJacobian(jac, [&model](unsigned cell, unsigned timeIdx)
                          -> const IntensiveQuantities& {
            return model.intensiveQuantities(cell, timeIdx);
        });
    }

    //! \brief Assemble the Jacobian from a caller-supplied intensive-quantity
    //!        source `iq(cell, timeIdx) -> const IntensiveQuantities&`. The
    //!        adjoint passes either the model cache (above) or its own
    //!        separate IQ array (`SeparateIQ`), which it can manipulate.
    template<class Matrix, class IQGet>
    void evalJacobian(Matrix& jac, IQGet&& iq) const
    {
        using Block = typename Matrix::block_type;
        const auto& model = simulator_.model();
        const auto& problem = simulator_.problem();
        const auto& neighborInfo = model.linearizer().getNeighborInfo();
        const std::size_t numCells = model.numGridDof();

        jac = 0.0;

        auto toBlock = [](const RateVector& adres) {
            Block b(0.0);
            for (int eq = 0; eq < numEq; ++eq) {
                for (int pv = 0; pv < numEq; ++pv) {
                    b[eq][pv] = adres[eq].derivative(pv);
                }
            }
            return b;
        };

        // --- pass 1: flux + storage ---
        RateVector flux(0.0);
        RateVector darcy(0.0);
        RateVector storage(0.0);
        for (std::size_t globI = 0; globI < numCells; ++globI) {
            const auto& iqIn = iq(static_cast<unsigned>(globI), 0);
            for (const auto& nb : neighborInfo[globI]) {
                const unsigned globJ = nb.neighbor;
                const auto& iqEx = iq(globJ, 0);
                flux = 0.0;
                darcy = 0.0;
                LocalResidual::computeFlux(flux, darcy, globI, globJ, iqIn,
                                           iqEx, nb.res_nbinfo,
                                           problem.moduleParams());
                flux *= nb.res_nbinfo.faceArea;
                Block b = toBlock(flux);
                jac[globI][globI] += b;
                b *= -1.0;
                jac[globJ][globI] += b;
            }
            storage = 0.0;
            LocalResidual::template computeStorage<Evaluation>(storage, iqIn);
            const Scalar storefac = model.dofTotalVolume(globI)
                                  / simulator_.timeStepSize();
            Block bs(0.0);
            for (int eq = 0; eq < numEq; ++eq) {
                for (int pv = 0; pv < numEq; ++pv) {
                    bs[eq][pv] = storage[eq].derivative(pv) * storefac;
                }
            }
            jac[globI][globI] += bs;
        }

        // --- pass 2: source (well perforation sources via problem.source) ---
        RateVector source(0.0);
        for (std::size_t globI = 0; globI < numCells; ++globI) {
            const auto& iqIn = iq(static_cast<unsigned>(globI), 0);
            source = 0.0;
            LocalResidual::computeSource(source, problem, iqIn, globI, 0);
            source *= -model.dofTotalVolume(globI);
            jac[globI][globI] += toBlock(source);
        }
    }

    //! \brief A separate, adjoint-owned array of intensive quantities,
    //!        populated by the public `IntensiveQuantities::update()` from the
    //!        model's primary variables — reusing the full forward physics
    //!        with no simulator change. It decouples the adjoint from the
    //!        model's IQ cache and is the substrate the adjoint manipulates:
    //!        it controls the primary variables it feeds in (and, once the
    //!        update seeding is made externally controllable, seeded
    //!        parameters). v1 reproduces the converged forward state at both
    //!        time levels (full numEq-slot state AD).
    class SeparateIQ
    {
    public:
        explicit SeparateIQ(const Simulator& simulator)
            : simulator_(simulator)
        { rebuild(); }

        //! \brief (Re)populate the array from the model's current solution.
        void rebuild()
        {
            const auto& model = simulator_.model();
            const auto& problem = simulator_.problem();
            const std::size_t numCells = model.numGridDof();
            for (unsigned timeIdx = 0; timeIdx < 2; ++timeIdx) {
                iq_[timeIdx].resize(numCells);
                const auto& solution = model.solution(timeIdx);
                for (std::size_t i = 0; i < numCells; ++i) {
                    iq_[timeIdx][i].update(problem, solution[i],
                                           static_cast<unsigned>(i), timeIdx);
                }
            }
        }

        const IntensiveQuantities& operator()(unsigned cell,
                                              unsigned timeIdx) const
        { return iq_[timeIdx][cell]; }

    private:
        const Simulator& simulator_;
        std::array<std::vector<IntensiveQuantities>, 2> iq_;
    };

private:
    //! \brief Scalar value of an Evaluation or a plain Scalar (so the same
    //!        code compiles for the value pass and, later, AD).
    static Scalar scalarValue_(const Evaluation& e) { return e.value(); }
    static Scalar scalarValue_(Scalar s) { return s; }

    Simulator& simulator_;
};

} // namespace Opm

#endif // OPM_ADJOINT_RESIDUAL_EVALUATOR_HPP
