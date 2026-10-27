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
 * \brief Objective functions for the adjoint solver.
 *
 * Per-substep sum objectives J = sum_k J_k(x_k, xw_k), selected with
 * --adjoint-objective:
 *
 *  - "pressure-average": J = (1/N) sum_i x_i[pressureIdx] at the final
 *    substep (reservoir-state validation objective).
 *  - "bhp:<WELL>": J = sum_k dt_k BHP(x_k) (well primary variable;
 *    exact dJ/dx_w, exercises the well coupling without approximation).
 *  - Rate-family objectives, all represented as a sum of rate terms
 *    (single- or multi-term):
 *      "rate:<WELL>:<phase>"                J += dt q
 *      "match:<WELL>:<phase>:<target>"      J += dt (q - q_t)^2
 *      "matchref:<WELL>:<phase>:<refcase>"  J += dt (q - q_obs(t))^2
 *      "matchsum:<refcase>:<W>.<p>[*<w>][+<W>.<p>[*<w>]]..."
 *                                           J += sum_terms w dt (q - q_obs)^2
 *    Observed curves are read from the reference run's summary via
 *    ESmry, interpolated linearly in time and converted from the file's
 *    rate unit to the internal SI/sign convention.
 *
 * Rate-family extensions:
 *  - Well names may be shell patterns ('*', '?'), expanded against the
 *    schedule at construction; one spec term fans out into one RateTerm
 *    per matching well.
 *  - Optional trailing suffix tokens apply to every term of the spec:
 *      :w=<weight>   per-term weight (multiplies matchsum per-term *<w>)
 *      :t0=<days>    window start (substep contributes iff its end time
 *      :t1=<days>    lies inside [t0, t1])
 *    e.g. "matchref:OP*:oil:REFCASE:w=2:t0=0:t1=365".
 *  - A matchref/matchsum term without an explicit window defaults to the
 *    reference summary's time span, so times outside the observed data
 *    contribute nothing (instead of a flat-extrapolated misfit).
 *  - Per-substep term values are accumulated (termObjectiveValues());
 *    the per-term breakdown of J is part of the gradient output.
 *
 * Phase is one of oil|water|gas.
 */
#ifndef OPM_ADJOINT_OBJECTIVE_HPP
#define OPM_ADJOINT_OBJECTIVE_HPP

#include <opm/common/ErrorMacros.hpp>
#include <opm/common/OpmLog/OpmLog.hpp>

#include <opm/io/eclipse/ESmry.hpp>

#include <opm/input/eclipse/Schedule/Schedule.hpp>

#include <fmt/format.h>

#include <opm/models/utils/propertysystem.hh>

#include <opm/adjoint/AdjointParallel.hpp>

#include <algorithm>
#include <cstddef>
#include <limits>
#include <memory>
#include <set>
#include <stdexcept>
#include <string>
#include <vector>

#include <opm/adjoint/AdjointObjectiveSpec.hpp>

namespace Opm {

template<class TypeTag>
class AdjointObjectiveFunction
{
public:
    using Simulator = GetPropType<TypeTag, Properties::Simulator>;
    using Scalar = GetPropType<TypeTag, Properties::Scalar>;
    using Indices = GetPropType<TypeTag, Properties::Indices>;
    using FluidSystem = GetPropType<TypeTag, Properties::FluidSystem>;

    using Kind = Adjoint::ObjectiveKind;
    using TermType = Adjoint::ObjectiveTermType;
    using ObjectiveSpec = Adjoint::ObjectiveSpec;

    //! One rate term of the objective (one concrete well).
    struct RateTerm
    {
        std::string well;
        unsigned canonicalPhaseIdx{0};
        unsigned canonicalCompIdx{0};
        TermType type{TermType::Rate};
        Scalar weight{1.0};
        Scalar targetRate{0.0};        //!< MatchConst (internal units)
        std::string referenceCase{};   //!< MatchRef
        std::string label{};           //!< "<type>:<well>:<phase>" for output
        //! Substep contributes iff its end time [days] is in the window.
        //! For MatchRef without an explicit window this defaults to the
        //! reference summary's time span (set at load).
        mutable Scalar windowStartDays{-std::numeric_limits<Scalar>::max()};
        mutable Scalar windowEndDays{std::numeric_limits<Scalar>::max()};
        bool windowExplicit{false};
        // observed curve (lazily loaded)
        mutable bool loaded{false};
        mutable bool clampWarned{false};        //!< out-of-window warning shown
        mutable std::vector<Scalar> obsTimes;   //!< days
        mutable std::vector<Scalar> obsValues;  //!< internal units/sign
    };

    AdjointObjectiveFunction(const std::string& spec, const Schedule& schedule)
        : AdjointObjectiveFunction(ObjectiveSpec::parse(spec), schedule)
    {}

    AdjointObjectiveFunction(const ObjectiveSpec& spec, const Schedule& schedule)
    {
        kind_ = spec.kind;
        if (kind_ == Kind::WellBhp) {
            wellName_ = spec.wellName;
            bhpWindowStartDays_ = spec.windowStartDays;
            bhpWindowEndDays_ = spec.windowEndDays;
            if (!schedule.hasWell(wellName_)) {
                OPM_THROW(std::runtime_error,
                          "Adjoint objective bhp: unknown well '" +
                          wellName_ + "'");
            }
        }
        for (const auto& ts : spec.terms) {
            for (const auto& wellName : expandWells_(ts.wells, schedule)) {
                RateTerm term;
                term.well = wellName;
                setPhase_(term, ts.phase);
                term.type = ts.type;
                term.weight = ts.weight;
                term.targetRate = ts.targetRateSi;
                term.referenceCase = ts.referenceCase;
                term.windowStartDays = ts.windowStartDays;
                term.windowEndDays = ts.windowEndDays;
                term.windowExplicit = ts.windowExplicit;
                term.label = termLabel_(term, ts.phase);
                terms_.push_back(std::move(term));
            }
        }
        termValues_.assign(terms_.size(), 0.0);
    }

    Kind kind() const
    { return kind_; }

    const std::string& wellName() const
    { return wellName_; }

    const std::vector<RateTerm>& rateTerms() const
    { return terms_; }

    //! \brief Zero the per-term value accumulators (call before a sweep).
    void resetTermValues() const
    { std::fill(termValues_.begin(), termValues_.end(), 0.0); }

    //! \brief Per-term accumulated objective values (local; reduce like J).
    //!        Indexed like rateTerms(); empty for non-rate objectives.
    const std::vector<Scalar>& termObjectiveValues() const
    { return termValues_; }

    //! \brief True if the objective has well terms involving \p well.
    bool involvesWell(const std::string& well) const
    {
        if (kind_ == Kind::WellBhp) {
            return well == wellName_;
        }
        return std::any_of(terms_.begin(), terms_.end(),
                           [&well](const auto& t) { return t.well == well; });
    }

    //! \brief Value contribution of substep \p k; the simulator must hold
    //!        the (recorded) state of substep k.
    Scalar stepValue(const Simulator& simulator, int k, int numSubsteps,
                     double dt) const
    {
        switch (kind_) {
        case Kind::PressureAverage: {
            if (k != numSubsteps - 1) {
                return 0.0;
            }
            const auto& solution = simulator.model().solution(/*timeIdx=*/0);
            Scalar sum = 0.0;
            for (std::size_t i = 0; i < solution.size(); ++i) {
                if (par_ && !par_->interior(i)) {
                    continue;  // skip ghost cells (owned by another rank)
                }
                sum += solution[i][Indices::pressureSwitchIdx];
            }
            if (par_) {
                return par_->sum(sum) / globalNumCells_(simulator);
            }
            return sum / solution.size();
        }
        case Kind::WellBhp: {
            if (!bhpInWindow_(simulator.time() + dt)) {
                return 0.0;
            }
            const auto& wellState =
                simulator.problem().wellModel().wellState();
            return dt * wellState.well(wellName_).bhp;
        }
        case Kind::RateTerms: {
            const auto& wellState =
                simulator.problem().wellModel().wellState();
            Scalar sum = 0.0;
            for (std::size_t ti = 0; ti < terms_.size(); ++ti) {
                const auto& term = terms_[ti];
                const int phasePos =
                    FluidSystem::canonicalToActivePhaseIdx(term.canonicalPhaseIdx);
                const Scalar q =
                    wellState.well(term.well).surface_rates[phasePos];
                const Scalar v = termValue_(simulator, term, dt, q);
                termValues_[ti] += v;
                sum += v;
            }
            return sum;
        }
        }
        return 0.0;
    }

    //! \brief Add dJ_k/dx_r to \p gradient (reservoir part).
    template<class GradientVector>
    void addStepGradient(const Simulator& simulator,
                         int k,
                         int numSubsteps,
                         GradientVector& gradient) const
    {
        if (kind_ != Kind::PressureAverage || k != numSubsteps - 1) {
            return;
        }
        const std::size_t numCells = simulator.model().solution(0).size();
        const Scalar denom = par_ ? globalNumCells_(simulator)
                                  : static_cast<Scalar>(numCells);
        for (std::size_t i = 0; i < numCells; ++i) {
            // ghost cells get the same per-cell weight; their rhs rows are
            // zeroed before the transposed solve, so this is harmless and
            // keeps the interior rows correct.
            gradient[i][Indices::pressureSwitchIdx] += 1.0 / denom;
        }
    }

    //! \brief Connect the parallel context (interior mask + reductions);
    //!        null in serial runs.
    void setParallel(const AdjointParallel<TypeTag>* par)
    { par_ = par; }

    //! \brief dJ_k/d(bhp of \p well) for the bhp objective; \p tEnd is the
    //!        substep end time [s] (window check).
    Scalar bhpGradient(const std::string& well, double dt, Scalar tEnd) const
    {
        if (kind_ == Kind::WellBhp && well == wellName_ &&
            bhpInWindow_(tEnd)) {
            return dt;
        }
        return 0.0;
    }

    //! \brief dJ_k/dq weight of one rate term given the current rate \p q.
    Scalar termWeight(const Simulator& simulator, const RateTerm& term,
                      double dt, Scalar q) const
    {
        const Scalar tEnd = simulator.time() + dt;
        if (!inWindow_(simulator, term, tEnd)) {
            return 0.0;
        }
        switch (term.type) {
        case TermType::Rate:
            return term.weight * dt;
        case TermType::MatchConst:
            return term.weight * 2.0 * dt * (q - term.targetRate);
        case TermType::MatchRef:
            return term.weight * 2.0 * dt *
                   (q - observedRate_(simulator, term, tEnd));
        }
        return 0.0;
    }

private:
    Scalar termValue_(const Simulator& simulator, const RateTerm& term,
                      double dt, Scalar q) const
    {
        const Scalar tEnd = simulator.time() + dt;
        if (!inWindow_(simulator, term, tEnd)) {
            return 0.0;
        }
        switch (term.type) {
        case TermType::Rate:
            return term.weight * dt * q;
        case TermType::MatchConst: {
            const Scalar diff = q - term.targetRate;
            return term.weight * dt * diff * diff;
        }
        case TermType::MatchRef: {
            const Scalar diff = q - observedRate_(simulator, term, tEnd);
            return term.weight * dt * diff * diff;
        }
        }
        return 0.0;
    }

    bool bhpInWindow_(Scalar tEnd) const
    {
        const Scalar tEndDays = tEnd / 86400.0;
        return tEndDays >= bhpWindowStartDays_ &&
               tEndDays <= bhpWindowEndDays_;
    }

    //! \brief Window check on the substep end time \p tEnd [s]. MatchRef
    //!        terms load their observations first so a defaulted window
    //!        (reference summary span) is in place before the check.
    bool inWindow_(const Simulator& simulator, const RateTerm& term,
                   Scalar tEnd) const
    {
        if (term.type == TermType::MatchRef && !term.loaded) {
            loadObservations_(simulator, term);
        }
        const Scalar tEndDays = tEnd / 86400.0;
        return tEndDays >= term.windowStartDays &&
               tEndDays <= term.windowEndDays;
    }

    //! \brief Observed rate (internal sign/units) at time \p t [s].
    Scalar observedRate_(const Simulator& simulator, const RateTerm& term,
                         Scalar t) const
    {
        if (!term.loaded) {
            loadObservations_(simulator, term);
        }
        const Scalar tDays = t / 86400.0;
        const auto& times = term.obsTimes;
        if (times.empty()) {
            OPM_THROW(std::runtime_error, "Empty observed curve");
        }
        if (tDays <= times.front() || tDays >= times.back()) {
            // Flat extrapolation outside the reference window silently
            // biases the misfit; say so once per term.
            if (!term.clampWarned) {
                term.clampWarned = true;
                OpmLog::warning(fmt::format(
                    "Adjoint matchref: simulation time {:.4g} days is outside "
                    "the reference summary window [{:.4g}, {:.4g}] days for "
                    "well {}; the observed rate is held constant there "
                    "(flat extrapolation)",
                    tDays, times.front(), times.back(), term.well));
            }
            return tDays <= times.front() ? term.obsValues.front()
                                          : term.obsValues.back();
        }
        const auto it = std::upper_bound(times.begin(), times.end(), tDays);
        const std::size_t hi = it - times.begin();
        const Scalar w = (tDays - times[hi - 1]) / (times[hi] - times[hi - 1]);
        return (1.0 - w) * term.obsValues[hi - 1] + w * term.obsValues[hi];
    }

    void loadObservations_(const Simulator& simulator, const RateTerm& term) const
    {
        const auto& schedule = simulator.vanguard().schedule();
        const auto& well = schedule.getWell(term.well, simulator.episodeIndex());
        // Producers: W?PR keywords, internal rates negative.
        // Injectors:  W?IR keywords, internal rates positive.
        const bool producer = well.isProducer();
        const Scalar sign = producer ? -1.0 : 1.0;
        std::string keyword;
        if (term.canonicalPhaseIdx == FluidSystem::oilPhaseIdx) {
            keyword = producer ? "WOPR:" : "WOIR:";
        } else if (term.canonicalPhaseIdx == FluidSystem::waterPhaseIdx) {
            keyword = producer ? "WWPR:" : "WWIR:";
        } else {
            keyword = producer ? "WGPR:" : "WGIR:";
        }
        keyword += term.well;

        EclIO::ESmry summary(term.referenceCase);
        summary.loadData();
        const auto& time = summary.get("TIME");      // days
        const auto& rate = summary.get(keyword);     // file units, positive

        // Convert the file's rate unit to internal SI (m3/s).
        const std::string unit = summary.get_unit(keyword);
        Scalar toSi;
        if (unit == "SM3/DAY" || unit == "SM3/D" || unit == "M3/DAY") {
            toSi = 1.0 / 86400.0;
        } else if (unit == "MSCF/DAY" || unit == "MSCF/D") {
            toSi = 28.316846592 / 86400.0;   // 1 Mscf = 1000 ft^3
        } else if (unit == "STB/DAY" || unit == "STB/D") {
            toSi = 0.158987294928 / 86400.0;
        } else {
            OPM_THROW(std::runtime_error,
                      "Unsupported rate unit '" + unit +
                      "' in reference summary for " + keyword +
                      " (supported: SM3/DAY, MSCF/DAY, STB/DAY; reservoir-"
                      "volume units like RM3/DAY are a different quantity)");
        }

        term.obsTimes.assign(time.begin(), time.end());
        term.obsValues.resize(rate.size());
        for (std::size_t i = 0; i < rate.size(); ++i) {
            term.obsValues[i] = sign * static_cast<Scalar>(rate[i]) * toSi;
        }
        // Without a user-given window, restrict the term to the span of the
        // observed data: outside it there is no information, and a flat-
        // extrapolated misfit would silently bias J.
        if (!term.windowExplicit && !term.obsTimes.empty()) {
            term.windowStartDays = term.obsTimes.front();
            term.windowEndDays = term.obsTimes.back();
        }
        term.loaded = true;
    }

    void setPhase_(RateTerm& term, const std::string& phase) const
    {
        if (phase == "oil") {
            term.canonicalPhaseIdx = FluidSystem::oilPhaseIdx;
            term.canonicalCompIdx = FluidSystem::oilCompIdx;
        } else if (phase == "water") {
            term.canonicalPhaseIdx = FluidSystem::waterPhaseIdx;
            term.canonicalCompIdx = FluidSystem::waterCompIdx;
        } else if (phase == "gas") {
            term.canonicalPhaseIdx = FluidSystem::gasPhaseIdx;
            term.canonicalCompIdx = FluidSystem::gasCompIdx;
        } else {
            OPM_THROW(std::runtime_error,
                      "Unknown phase '" + phase + "' in adjoint objective");
        }
    }

    //! \brief Expand exact names and shell patterns against the schedule;
    //!        deduplicated, schedule-independent stable order (sorted).
    static std::vector<std::string>
    expandWells_(const std::vector<std::string>& entries,
                 const Schedule& schedule)
    {
        std::set<std::string> names;
        for (const auto& entry : entries) {
            if (entry.find_first_of("*?") != std::string::npos) {
                const auto matches = schedule.wellNames(entry);
                if (matches.empty()) {
                    OPM_THROW(std::runtime_error,
                              "Adjoint objective well pattern '" + entry +
                              "' matches no well in the schedule");
                }
                names.insert(matches.begin(), matches.end());
            } else {
                if (!schedule.hasWell(entry)) {
                    OPM_THROW(std::runtime_error,
                              "Adjoint objective well '" + entry +
                              "' is not in the schedule");
                }
                names.insert(entry);
            }
        }
        return {names.begin(), names.end()};
    }

    static std::string termLabel_(const RateTerm& term,
                                  const std::string& phase)
    {
        const char* type = term.type == TermType::Rate ? "rate"
                         : term.type == TermType::MatchConst ? "match"
                         : "matchref";
        return std::string(type) + ":" + term.well + ":" + phase;
    }

    //! \brief Global number of interior cells (cached), for the
    //!        pressure-average denominator.
    Scalar globalNumCells_(const Simulator& simulator) const
    {
        if (globalNumCells_cached_ < 0) {
            const auto& solution = simulator.model().solution(/*timeIdx=*/0);
            Scalar localInterior = 0.0;
            for (std::size_t i = 0; i < solution.size(); ++i) {
                if (!par_ || par_->interior(i)) {
                    localInterior += 1.0;
                }
            }
            globalNumCells_cached_ = par_ ? par_->sum(localInterior)
                                          : localInterior;
        }
        return globalNumCells_cached_;
    }

    Kind kind_{Kind::PressureAverage};
    std::string wellName_{};           //!< bhp objective
    Scalar bhpWindowStartDays_{-std::numeric_limits<Scalar>::max()};
    Scalar bhpWindowEndDays_{std::numeric_limits<Scalar>::max()};
    std::vector<RateTerm> terms_;      //!< rate-family objectives
    mutable std::vector<Scalar> termValues_;  //!< per-term J accumulators
    const AdjointParallel<TypeTag>* par_{nullptr};
    mutable Scalar globalNumCells_cached_{-1};
};

} // namespace Opm

#endif // OPM_ADJOINT_OBJECTIVE_HPP
