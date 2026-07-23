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
 * \brief TypeTag-free objective specification for the adjoint solver.
 *
 * A plain data description of an objective (kind, per-term wells/phase/weight/
 * window/reference), independent of the simulator types. Both the CLI string
 * parser (ObjectiveSpec::parse) and the JSON config loader
 * (AdjointConfigFile.hpp) produce it; AdjointObjectiveFunction consumes it.
 * Keeping it in a light header (no ESmry/Schedule/property-system includes)
 * lets AdjointParameters/AdjointConfig carry an ObjectiveSpec directly.
 */
#ifndef OPM_ADJOINT_OBJECTIVE_SPEC_HPP
#define OPM_ADJOINT_OBJECTIVE_SPEC_HPP

#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

namespace Opm::Adjoint {

enum class ObjectiveKind { PressureAverage, WellBhp, RateTerms };

enum class ObjectiveTermType { Rate, MatchConst, MatchRef };

//! One rate term of an objective specification, before well-pattern
//! expansion. TypeTag-free so both the CLI string parser and a config-file
//! loader can produce it.
struct ObjectiveTermSpec
{
    std::vector<std::string> wells;   //!< exact names or shell patterns
    std::string phase;                //!< oil|water|gas
    ObjectiveTermType type{ObjectiveTermType::Rate};
    double targetRateSi{0.0};         //!< MatchConst (SI, internal sign-free)
    std::string referenceCase{};      //!< MatchRef
    double weight{1.0};
    double windowStartDays{-std::numeric_limits<double>::max()};
    double windowEndDays{std::numeric_limits<double>::max()};
    bool windowExplicit{false};
};

//! Parsed objective specification (TypeTag-free).
struct ObjectiveSpec
{
    ObjectiveKind kind{ObjectiveKind::PressureAverage};
    std::string wellName{};                //!< bhp objective (exact name)
    std::vector<ObjectiveTermSpec> terms;  //!< rate-family objectives
    //! bhp objective window (rate terms carry their own)
    double windowStartDays{-std::numeric_limits<double>::max()};
    double windowEndDays{std::numeric_limits<double>::max()};

    static ObjectiveSpec parse(const std::string& spec);
};

namespace detail {

inline std::vector<std::string> splitObjective(const std::string& text,
                                               char sep, std::size_t expected)
{
    std::vector<std::string> parts;
    std::size_t pos = 0;
    while (parts.size() + 1 < expected) {
        const auto next = text.find(sep, pos);
        if (next == std::string::npos) {
            throw std::runtime_error(
                "Malformed adjoint objective specification: " + text);
        }
        parts.push_back(text.substr(pos, next - pos));
        pos = next + 1;
    }
    parts.push_back(text.substr(pos));
    return parts;
}

//! Pop a trailing ":<key>=<value>" token off \p s; true if popped.
inline bool popSuffixToken(std::string& s, const char* key, std::string& value)
{
    const auto pos = s.rfind(':');
    if (pos == std::string::npos) {
        return false;
    }
    const std::string prefix = std::string(key) + "=";
    const std::string tail = s.substr(pos + 1);
    if (tail.rfind(prefix, 0) != 0) {
        return false;
    }
    value = tail.substr(prefix.size());
    s.erase(pos);
    return true;
}

} // namespace detail

inline ObjectiveSpec ObjectiveSpec::parse(const std::string& specIn)
{
    ObjectiveSpec result;
    std::string spec = specIn;

    // Strip trailing :w= / :t0= / :t1= tokens (order-free); they apply to
    // every term parsed from this spec.
    double weight = 1.0;
    double t0 = -std::numeric_limits<double>::max();
    double t1 = std::numeric_limits<double>::max();
    bool windowExplicit = false;
    for (bool popped = true; popped; ) {
        std::string value;
        popped = false;
        if (detail::popSuffixToken(spec, "w", value)) {
            weight = std::stod(value);
            popped = true;
        } else if (detail::popSuffixToken(spec, "t0", value)) {
            t0 = std::stod(value);
            windowExplicit = true;
            popped = true;
        } else if (detail::popSuffixToken(spec, "t1", value)) {
            t1 = std::stod(value);
            windowExplicit = true;
            popped = true;
        }
    }

    const auto makeTerm = [&](std::string well, std::string phase) {
        ObjectiveTermSpec term;
        term.wells.push_back(std::move(well));
        term.phase = std::move(phase);
        term.weight = weight;
        term.windowStartDays = t0;
        term.windowEndDays = t1;
        term.windowExplicit = windowExplicit;
        return term;
    };

    if (spec.empty() || spec == "pressure-average") {
        result.kind = ObjectiveKind::PressureAverage;
        if (windowExplicit || weight != 1.0) {
            throw std::runtime_error(
                "pressure-average is a final-substep objective; :w=/:t0=/:t1= "
                "suffix tokens are not meaningful for it");
        }
    } else if (spec.rfind("bhp:", 0) == 0) {
        result.kind = ObjectiveKind::WellBhp;
        result.wellName = spec.substr(4);
        result.windowStartDays = t0;
        result.windowEndDays = t1;
        if (weight != 1.0) {
            throw std::runtime_error(
                "bhp objective does not support :w= (a single-term J scales "
                "trivially)");
        }
        if (result.wellName.empty()) {
            throw std::runtime_error(
                "--adjoint-objective=bhp:<WELL> needs a well name");
        }
    } else if (spec.rfind("rate:", 0) == 0) {
        result.kind = ObjectiveKind::RateTerms;
        const auto parts = detail::splitObjective(spec.substr(5), ':', 2);
        auto term = makeTerm(parts[0], parts[1]);
        term.type = ObjectiveTermType::Rate;
        result.terms.push_back(std::move(term));
    } else if (spec.rfind("match:", 0) == 0) {
        result.kind = ObjectiveKind::RateTerms;
        const auto parts = detail::splitObjective(spec.substr(6), ':', 3);
        auto term = makeTerm(parts[0], parts[1]);
        term.type = ObjectiveTermType::MatchConst;
        term.targetRateSi = std::stod(parts[2]) / 86400.0; // sm3/day -> SI
        result.terms.push_back(std::move(term));
    } else if (spec.rfind("matchref:", 0) == 0) {
        result.kind = ObjectiveKind::RateTerms;
        const auto parts = detail::splitObjective(spec.substr(9), ':', 3);
        auto term = makeTerm(parts[0], parts[1]);
        term.type = ObjectiveTermType::MatchRef;
        term.referenceCase = parts[2];
        result.terms.push_back(std::move(term));
    } else if (spec.rfind("matchsum:", 0) == 0) {
        // matchsum:<refcase>:<W>.<p>[*<w>][+<W>.<p>[*<w>]]...
        result.kind = ObjectiveKind::RateTerms;
        const auto rest = spec.substr(9);
        const auto lastColon = rest.rfind(':');
        if (lastColon == std::string::npos) {
            throw std::runtime_error(
                "--adjoint-objective=matchsum:<refcase>:<W>.<p>[+<W>.<p>]...");
        }
        const std::string refcase = rest.substr(0, lastColon);
        const std::string termsSpec = rest.substr(lastColon + 1);
        std::size_t pos = 0;
        while (pos != std::string::npos && pos < termsSpec.size()) {
            const auto plus = termsSpec.find('+', pos);
            std::string one = termsSpec.substr(
                pos, plus == std::string::npos ? std::string::npos : plus - pos);
            double termWeight = 1.0;
            if (const auto star = one.rfind('*'); star != std::string::npos) {
                termWeight = std::stod(one.substr(star + 1));
                one.erase(star);
            }
            const auto dot = one.find('.');
            if (dot == std::string::npos) {
                throw std::runtime_error(
                    "matchsum term '" + one + "' must be <WELL>.<phase>");
            }
            auto term = makeTerm(one.substr(0, dot), one.substr(dot + 1));
            term.type = ObjectiveTermType::MatchRef;
            term.referenceCase = refcase;
            term.weight = weight * termWeight;
            result.terms.push_back(std::move(term));
            pos = (plus == std::string::npos) ? std::string::npos : plus + 1;
        }
        if (result.terms.empty()) {
            throw std::runtime_error("matchsum objective without terms");
        }
    } else {
        throw std::runtime_error("Unknown adjoint objective '" + spec + "'");
    }
    return result;
}

} // namespace Opm::Adjoint

#endif // OPM_ADJOINT_OBJECTIVE_SPEC_HPP
