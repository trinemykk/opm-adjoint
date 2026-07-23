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
 * \brief JSON configuration file for adjoint/gradient runs (--adjoint-config).
 *
 * A single file replaces the grown set of --adjoint-* flag strings, and can
 * express a structured objective (per-term wells/phase/weight/time-window,
 * wildcards) that the CLI string cannot. Loaded as the BASE configuration;
 * explicitly-set CLI flags override it (applied by AdjointConfig::fromParameters).
 *
 * Schema (v1; every section optional):
 * {
 *   "schema_version": 1,
 *   "mode": "gradient",                     // "" | replay | gradient | objective
 *   "archive": { "path": "...", "save": true, "save_system": false },
 *   "gradient": { "method": "general-ad", "fd_step": 1e-7,
 *                 "endpoint_method": "stencil" },
 *   "solver": { "linear_solver": "cprt", "reduction": 1e-14, "max_iter": 300 },
 *   "endpoints": "SWL,KRW",
 *   "objective": {                          // one of the three forms:
 *     "kind": "pressure-average"            //  (a) reservoir-state
 *     // or "kind": "bhp", "well": "PROD", "window_days": [0, 900]
 *     // or "terms": [                      //  (c) rate family
 *     //   { "kind": "matchref", "wells": ["OP_*"], "phase": "oil",
 *     //     "reference_case": "/path/REF", "weight": 2.0,
 *     //     "window_days": [0, 1826] },
 *     //   { "kind": "match", "wells": ["WI_1"], "phase": "water",
 *     //     "target_sm3_per_day": 500.0 },
 *     //   { "kind": "rate", "wells": ["PROD"], "phase": "gas" } ]
 *   }
 * }
 */
#ifndef OPM_ADJOINT_CONFIG_FILE_HPP
#define OPM_ADJOINT_CONFIG_FILE_HPP

#include <opm/adjoint/AdjointParameters.hpp>
#include <opm/adjoint/AdjointObjectiveSpec.hpp>

#include <boost/property_tree/json_parser.hpp>
#include <boost/property_tree/ptree.hpp>

#include <limits>
#include <stdexcept>
#include <string>

namespace Opm {

namespace detail {

//! Read an optional [t0, t1] window (days) into a term.
inline void readWindow(const boost::property_tree::ptree& node,
                       Adjoint::ObjectiveTermSpec& term)
{
    if (const auto w = node.get_child_optional("window_days")) {
        std::vector<double> t;
        for (const auto& [k, v] : *w) {
            static_cast<void>(k);
            t.push_back(v.get_value<double>());
        }
        if (t.size() != 2) {
            throw std::runtime_error(
                "adjoint config: window_days must be [t0, t1]");
        }
        term.windowStartDays = t[0];
        term.windowEndDays   = t[1];
        term.windowExplicit  = true;
    }
}

inline Adjoint::ObjectiveTermType termType(const std::string& kind)
{
    if (kind == "rate")     return Adjoint::ObjectiveTermType::Rate;
    if (kind == "match")    return Adjoint::ObjectiveTermType::MatchConst;
    if (kind == "matchref") return Adjoint::ObjectiveTermType::MatchRef;
    throw std::runtime_error("adjoint config: unknown term kind '" + kind + "'");
}

inline Adjoint::ObjectiveSpec
parseObjective(const boost::property_tree::ptree& obj)
{
    using namespace Adjoint;
    ObjectiveSpec spec;

    // (c) rate family: an explicit list of terms.
    if (const auto terms = obj.get_child_optional("terms")) {
        spec.kind = ObjectiveKind::RateTerms;
        for (const auto& [k, tnode] : *terms) {
            static_cast<void>(k);
            ObjectiveTermSpec term;
            term.type = termType(tnode.get<std::string>("kind"));
            term.phase = tnode.get<std::string>("phase");
            for (const auto& [wk, wv] : tnode.get_child("wells")) {
                static_cast<void>(wk);
                term.wells.push_back(wv.get_value<std::string>());
            }
            if (term.wells.empty()) {
                throw std::runtime_error("adjoint config: term has no wells");
            }
            term.weight = tnode.get<double>("weight", 1.0);
            if (term.type == ObjectiveTermType::MatchConst) {
                term.targetRateSi =
                    tnode.get<double>("target_sm3_per_day") / 86400.0;
            } else if (term.type == ObjectiveTermType::MatchRef) {
                term.referenceCase = tnode.get<std::string>("reference_case");
            }
            readWindow(tnode, term);
            spec.terms.push_back(std::move(term));
        }
        if (spec.terms.empty()) {
            throw std::runtime_error("adjoint config: objective.terms is empty");
        }
        return spec;
    }

    // (a)/(b) scalar kinds.
    const std::string kind = obj.get<std::string>("kind", "pressure-average");
    if (kind == "pressure-average") {
        spec.kind = ObjectiveKind::PressureAverage;
    } else if (kind == "bhp") {
        spec.kind = ObjectiveKind::WellBhp;
        spec.wellName = obj.get<std::string>("well");
        if (const auto w = obj.get_child_optional("window_days")) {
            std::vector<double> t;
            for (const auto& [k, v] : *w) { static_cast<void>(k);
                t.push_back(v.get_value<double>()); }
            if (t.size() != 2) {
                throw std::runtime_error(
                    "adjoint config: bhp window_days must be [t0, t1]");
            }
            spec.windowStartDays = t[0];
            spec.windowEndDays   = t[1];
        }
    } else {
        throw std::runtime_error(
            "adjoint config: objective.kind '" + kind + "' needs a terms list");
    }
    return spec;
}

} // namespace detail

//! \brief Load a JSON config file into \p config as the base configuration.
//!        Only fields present in the file are set; the caller then applies
//!        explicit CLI overrides.
inline void loadAdjointConfigFile(const std::string& path, AdjointConfig& config)
{
    boost::property_tree::ptree pt;
    try {
        boost::property_tree::read_json(path, pt);
    } catch (const std::exception& e) {
        throw std::runtime_error("adjoint config '" + path + "': " + e.what());
    }

    const int version = pt.get<int>("schema_version", 1);
    if (version != 1) {
        throw std::runtime_error(
            "adjoint config: unsupported schema_version " +
            std::to_string(version) + " (this build understands 1)");
    }

    if (auto v = pt.get_optional<std::string>("mode"))       config.mode = *v;
    if (auto v = pt.get_optional<std::string>("endpoints"))  config.endpoints = *v;

    if (const auto arc = pt.get_child_optional("archive")) {
        if (auto v = arc->get_optional<std::string>("path"))  config.path = *v;
        if (auto v = arc->get_optional<bool>("save"))         config.save = *v;
        if (auto v = arc->get_optional<bool>("save_system"))  config.saveSystem = *v;
    }
    if (const auto grad = pt.get_child_optional("gradient")) {
        if (auto v = grad->get_optional<std::string>("method"))
            config.gradientMethod = *v;
        if (auto v = grad->get_optional<double>("fd_step"))   config.fdStep = *v;
        if (auto v = grad->get_optional<std::string>("endpoint_method"))
            config.endpointMethod = *v;
    }
    if (const auto sol = pt.get_child_optional("solver")) {
        if (auto v = sol->get_optional<std::string>("linear_solver"))
            config.linearSolver = *v;
        if (auto v = sol->get_optional<double>("reduction"))
            config.linearSolverReduction = *v;
        if (auto v = sol->get_optional<int>("max_iter"))
            config.linearSolverMaxIter = *v;
    }
    if (const auto obj = pt.get_child_optional("objective")) {
        config.objectiveSpec = detail::parseObjective(*obj);
    }
}

} // namespace Opm

#endif // OPM_ADJOINT_CONFIG_FILE_HPP
