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
 * \brief End-of-run warning: scan the deck for quantities the adjoint
 *        does not reproduce/differentiate exactly.
 *
 * The adjoint v1 reproduces the forward residual/Jacobian bitwise and
 * differentiates it exactly for the common black-oil + (group-)well
 * control case. A number of *explicitly-updated* quantities are handled
 * approximately; this scan looks them up in the deck and warns at the
 * end of a gradient/replay run so a user knows whether their case is in
 * the exact regime. The categories mirror the catalog in
 * opm-adjoint/adjoint_plan.md (explicitly-updated-quantities table).
 *
 *  - approx-gradient: the forward state IS serialized so REPLAY is exact
 *    (J is exact), but the cross-step derivative of the explicitly-updated
 *    quantity is neglected, so dJ/dm is approximate;
 *  - approx-replay: the control target is re-derived during the replay
 *    assembly and may not reproduce the forward exactly (reservoir-volume
 *    / voidage-replacement group control);
 *  - neglected: not handled by the adjoint at all (aquifers, networks).
 */
#ifndef OPM_ADJOINT_DECK_WARNINGS_HPP
#define OPM_ADJOINT_DECK_WARNINGS_HPP

#include <opm/common/OpmLog/OpmLog.hpp>

#include <opm/input/eclipse/Schedule/Group/Group.hpp>
#include <opm/input/eclipse/Schedule/OilVaporizationProperties.hpp>
#include <opm/input/eclipse/Schedule/Well/Connection.hpp>

#include <fmt/format.h>

#include <string>
#include <utility>
#include <vector>

namespace Opm {

//! \brief Scan the deck and emit a single categorized warning listing the
//!        explicitly-updated / non-exact quantities present.
template<class Simulator>
inline void warnAdjointDeckApproximations(const Simulator& simulator)
{
    const auto& eclState = simulator.vanguard().eclState();
    const auto& schedule = simulator.vanguard().schedule();
    const std::size_t nstep = schedule.size();

    using Entry = std::pair<std::string, std::string>;
    std::vector<Entry> approxGradient;  // replay exact, dJ/dm cross-term neglected
    std::vector<Entry> approxReplay;    // re-derived control target, replay may drift
    std::vector<Entry> neglected;       // not handled at all

    // --- explicitly-updated quantities: replay exact, gradient approximate ---
    if (eclState.runspec().hysterPar().active()) {
        approxGradient.emplace_back(
            "saturation hysteresis (HYSTER)",
            "state serialized so replay is exact; the hysteresis-state "
            "cross-term in dJ/dm is neglected");
    }

    bool drsdt = false, drvdt = false, vappars = false;
    for (std::size_t i = 0; i < nstep; ++i) {
        const auto& ov = schedule[i].oilvap();
        // Flag DRSDT/DRVDT whenever active, INCLUDING the value 0. A DRSDT
        // rate limit caps Rs_k at Rs_{k-1} + maxDRSDT*dt wherever it binds, an
        // explicit dependence of the step-k residual on the step-(k-1) Rs.
        // Replay reproduces the within-step linearization bitwise, but the
        // adjoint's cross-step coupling (reservoir storage Bdiag) does not
        // carry this term, so dJ/dm is approximate. DRSDT 0 is the *most*
        // binding case (Rs may not increase at all) — measured on SPE1CASE1:
        // removing DRSDT drops the bhp-objective dJ/dPORO FD-vs-adjoint
        // deviation from 2.94% to 0 (see STATUS.md). The earlier assumption
        // that "DRSDT 0 is exact" was wrong.
        for (std::size_t r = 0; r < ov.numPvtRegions(); ++r) {
            if (ov.drsdtActive(r)) drsdt = true;
            if (ov.drvdtActive(r)) drvdt = true;
        }
        vappars |= (ov.getType() ==
                    OilVaporizationProperties::OilVaporization::VAPPARS);
    }
    if (drsdt) {
        approxGradient.emplace_back("DRSDT (incl. DRSDT 0)",
            "the dissolved-gas rate limit caps Rs_k against the step-(k-1) Rs "
            "where it binds; replay is exact but this cross-step term in dJ/dm "
            "is neglected. Material for well objectives on gas-evolving decks");
    }
    if (drvdt) {
        approxGradient.emplace_back("DRVDT (incl. DRVDT 0)",
            "the vaporized-oil rate limit caps Rv_k against the step-(k-1) Rv "
            "where it binds; replay is exact but this cross-step term in dJ/dm "
            "is neglected");
    }
    if (vappars) {
        approxGradient.emplace_back("VAPPARS",
            "replay exact (cache-free storage); the vaporization-parameter "
            "cross-term in dJ/dm is neglected");
    }

    // --- group control: voidage replacement (replay-inexact) and
    //     guide-rate allocation (replay-exact, gradient-inexact) ---
    bool voidage = false;     // RESV/VREP/REIN target re-derived in replay
    bool guideRate = false;   // any active group control => guide-rate sharing
    for (std::size_t i = 0; i < nstep && (!voidage || !guideRate); ++i) {
        for (const auto& gname : schedule.groupNames(i)) {
            const auto& group = schedule.getGroup(gname, i);
            for (const auto& [phase, props] : group.injectionProperties()) {
                static_cast<void>(phase);
                using IM = Group::InjectionCMode;
                if (props.cmode != IM::NONE) {
                    guideRate = true;
                }
                if (props.cmode == IM::VREP || props.cmode == IM::RESV ||
                    props.cmode == IM::REIN) {
                    voidage = true;
                }
            }
            const auto pcmode = group.productionProperties().cmode;
            if (pcmode != Group::ProductionCMode::NONE) {
                guideRate = true;
            }
            if (pcmode == Group::ProductionCMode::RESV) {
                voidage = true;
            }
        }
    }
    if (voidage) {
        approxReplay.emplace_back(
            "reservoir-volume / voidage-replacement group control "
            "(RESV/VREP/REIN)",
            "the effective control target is re-derived from reservoir "
            "voidage during the replay assembly and may not reproduce the "
            "forward bitwise; the gradient is correspondingly approximate");
    }
    if (guideRate) {
        approxGradient.emplace_back(
            "guide-rate group control allocation",
            "the group target is shared among members by guide rates computed "
            "from well potentials; the converged guide rates are restored so "
            "replay is exact, but the d(guide-rate)/d(state) cross-term in "
            "dJ/dm is neglected");
    }

    // --- permeability-derived well connection factors ---
    // A defaulted CTF (COMPDAT item 8) is computed from the cell
    // permeability (Peaceman, Kh), so dJ/dperm has a contribution through
    // the well equations (dCTF/dK) that the cell-based perm gradient does
    // not include. Flag when any connection is perm-derived.
    bool ctfFromPerm = false;
    for (std::size_t i = 0; i < nstep && !ctfFromPerm; ++i) {
        for (const auto& wname : schedule.wellNames(i)) {
            const auto& well = schedule.getWell(wname, i);
            for (const auto& conn : well.getConnections()) {
                if (conn.kind() == Connection::CTFKind::Defaulted) {
                    ctfFromPerm = true;
                    break;
                }
            }
            if (ctfFromPerm) {
                break;
            }
        }
    }
    if (ctfFromPerm) {
        approxGradient.emplace_back(
            "permeability-derived well connection factors (defaulted CTF)",
            "the connection transmissibility factor is computed from the "
            "cell permeability (Peaceman/Kh), so dJ/dperm has a well-equation "
            "contribution dCTF/dK that the cell-based perm gradient neglects");
    }

    // --- neglected entirely ---
    if (eclState.aquifer().active()) {
        neglected.emplace_back("aquifers",
            "aquifer contributions are not assembled during replay and their "
            "cross-term is neglected; exclude aquifer cells from gradients");
    }
    bool network = false;
    for (std::size_t i = 0; i < nstep && !network; ++i) {
        network |= schedule[i].network().active();
    }
    if (network) {
        neglected.emplace_back("extended network (NODEPROP/BRANPROP)",
            "network nodal-pressure coupling is not handled by the adjoint");
    }

    if (approxGradient.empty() && approxReplay.empty() && neglected.empty()) {
        return;
    }

    std::string msg = "Adjoint accuracy notes for this deck — some "
                      "explicitly-updated quantities are not handled exactly:\n";
    auto append = [&msg](const std::vector<Entry>& list, const char* tag) {
        for (const auto& [feature, treatment] : list) {
            msg += fmt::format("  [{}] {}: {}\n", tag, feature, treatment);
        }
    };
    append(approxGradient, "approx-gradient");
    append(approxReplay, "approx-replay");
    append(neglected, "neglected");
    // Standing caveat: every multi-connection standard well carries explicitly-
    // updated quantities (hydrostatic connection pressure drop from prior-state
    // perf densities, wellbore-storage F0_, explicit B-factors). These are
    // recomputed exactly in replay (so J is exact), but their cross-step
    // derivative is neglected in dJ/dm. The hydrostatic term is the material
    // one: perf_pressure_diffs_ = dz*rho*g is a Scalar computed once per step in
    // StandardWell::calculateExplicitQuantities and enters the drawdown, so its
    // state derivative is absent from the well Jacobian the adjoint reuses. This
    // is NEGLIGIBLE for thin / single-perforation wells (~0 head) but can reach
    // O(10%) on well BHP / rate objectives for thick, multi-layer wells
    // (measured ~12% on Norne D-1H). It is NOT practically fixable: the wellbore
    // density depends on the flowing mixture, i.e. on essentially all reservoir
    // state around the well, so an exact derivative is dense and expensive; the
    // explicit treatment is a legacy of the reference simulator. Multisegment
    // (WELSEGS) wells solve segment pressures implicitly and do NOT have this
    // approximation — use an MSW-equivalent deck if exact well-objective
    // gradients on gravity-dominated wells are required (needs MSW adjoint).
    msg += "  [approx-gradient] STANDARD-well explicit hydrostatic head "
           "(connection pressure drop from prior-step perf densities): "
           "recomputed exactly in replay, but its cross-step derivative is "
           "neglected in dJ/dm. Small for thin wells; up to ~10% on BHP/rate "
           "objectives for thick multi-perforation wells (legacy explicit "
           "treatment; not practically fixable — density couples to all "
           "near-well reservoir state; MSW wells avoid it). Also: wellbore-"
           "storage F0 and explicit B-factors (same cross-step caveat).\n";
    msg += "  Replay and gradients are exact for everything else (FD-verified: "
           "porosity/PV, transmissibility, permeability, end-point scaling, "
           "well and group control).";
    OpmLog::warning(msg);
}

} // namespace Opm

#endif // OPM_ADJOINT_DECK_WARNINGS_HPP
