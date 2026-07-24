# -*- mode: cmake; tab-width: 2; indent-tabs-mode: nil; -*-

###########################################################################
#                                                                         #
# Note: The bulk of the build system is located in the cmake/ directory   #
#       of opm-common. This file only contains the specializations for    #
#       this particular project (modeled on current opm-simulators).      #
#                                                                         #
#       dune.module                Name and version number                #
#       CMakeLists_files.cmake     Path of source files                   #
#       opm-adjoint-prereqs.cmake  Dependencies                           #
#                                                                         #
###########################################################################

cmake_minimum_required(VERSION 3.26)

project(opm-adjoint C CXX)

option(SIBLING_SEARCH "Search for other modules in sibling directories?" ON)

# Sibling-directory guess for opm-common (copied from opm-simulators).
if(SIBLING_SEARCH AND NOT opm-common_DIR)
  get_filename_component(_leaf_dir_name ${PROJECT_BINARY_DIR} NAME)
  get_filename_component(_parent_full_dir ${PROJECT_BINARY_DIR} DIRECTORY)
  get_filename_component(_parent_dir_name ${_parent_full_dir} NAME)
  get_filename_component(_modules_dir ${_parent_full_dir} DIRECTORY)
  if(IS_DIRECTORY ${_modules_dir}/opm-common/${_leaf_dir_name})
    set(opm-common_DIR ${_modules_dir}/opm-common/${_leaf_dir_name})
  else()
    string(REPLACE ${PROJECT_NAME} opm-common _opm_common_leaf ${_leaf_dir_name})
    if(NOT _leaf_dir_name STREQUAL _opm_common_leaf
        AND IS_DIRECTORY ${_parent_full_dir}/${_opm_common_leaf})
      set(opm-common_DIR ${_parent_full_dir}/${_opm_common_leaf})
    elseif(IS_DIRECTORY ${_parent_full_dir}/opm-common)
      set(opm-common_DIR "${_parent_full_dir}/opm-common")
    endif()
  endif()
endif()
if(opm-common_DIR AND NOT IS_DIRECTORY ${opm-common_DIR})
  message(WARNING "Value ${opm-common_DIR} passed to variable"
    " opm-common_DIR is not a directory")
endif()

find_package(opm-common REQUIRED)

# project information is in dune.module. Read this file and set variables.
include (OpmInit)

# all setup common to the OPM library modules is done here
# (includes ${project}-prereqs and CMakeLists_files.cmake).
include (OpmLibMain)

target_link_libraries(opmadjoint
  PUBLIC
    opmsimulators
)

# Match the compile environment of the opm-simulators build: HYPRE's
# imported target carries HAVE_HYPRE=1 on its interface, which
# opm-simulators links privately; mismatching definitions between the
# library and our header-instantiated code must be avoided.
if(USE_HYPRE AND USE_MPI)
  find_package(HYPRE)
  if(HYPRE_FOUND)
    target_link_libraries(opmadjoint PUBLIC HYPRE::HYPRE)
  endif()
endif()

# moduleVersion.cpp provides the version strings the flow banner needs;
# compiled here against this module's generated project-version.h
# (same pattern as opm-flowgeomechanics).
add_library(moduleVersionAdjoint OBJECT opm/simulators/utils/moduleVersion.cpp)
set_property(TARGET moduleVersionAdjoint PROPERTY POSITION_INDEPENDENT_CODE ON)
target_include_directories(moduleVersionAdjoint PRIVATE "${PROJECT_BINARY_DIR}")
# for opm/simulators/utils/moduleVersion.hpp and its dependencies
target_link_libraries(moduleVersionAdjoint PRIVATE opmsimulators)
add_dependencies(moduleVersionAdjoint opmadjoint)

add_executable(flow_adjoint
  examples/flow_adjoint.cpp
)
target_link_libraries(flow_adjoint
  PUBLIC
    opmadjoint
  PRIVATE
    moduleVersionAdjoint
)

# Look for the opm-tests repository; sets HAVE_OPM_TESTS / OPM_TESTS_ROOT.
include(Findopm-tests)

# Script-driven verification tests (see adjoint_testing.md). The FD
# tests run many small forward simulations; all of them use the tiny
# 3-cell MODEL_1D_DEBUG deck from opm-tests (branch with adjoint_tests/).
if(HAVE_OPM_TESTS)
  set(_m1d ${OPM_TESTS_ROOT}/adjoint_tests/MODEL_1D_DEBUG/inputfiles)
  set(_testout ${PROJECT_BINARY_DIR}/script-tests)
  if(EXISTS ${_m1d})
    add_test(NAME adjoint_replay_m1d
      COMMAND ${PROJECT_SOURCE_DIR}/tests/run-adjoint-replay-test.sh
              $<TARGET_FILE:flow_adjoint>
              ${_m1d}/MODEL_1D_DEBUG.DATA ${_testout}/replay_m1d)
    add_test(NAME adjoint_residual_eval_m1d
      COMMAND ${PROJECT_SOURCE_DIR}/tests/run-adjoint-residual-eval-test.sh
              $<TARGET_FILE:flow_adjoint>
              ${_m1d}/MODEL_1D_DEBUG.DATA ${_testout}/reseval_m1d)
    add_test(NAME adjoint_jacobian_eval_m1d
      COMMAND ${PROJECT_SOURCE_DIR}/tests/run-adjoint-jacobian-eval-test.sh
              $<TARGET_FILE:flow_adjoint>
              ${_m1d}/MODEL_1D_DEBUG.DATA ${_testout}/jaceval_m1d)
    add_test(NAME adjoint_gradient_regression
      COMMAND ${PROJECT_SOURCE_DIR}/tests/run-adjoint-gradient-regression.sh
              $<TARGET_FILE:flow_adjoint> ${_m1d} ${_testout}/gradreg)
    add_test(NAME adjoint_fd_pv
      COMMAND ${PROJECT_SOURCE_DIR}/tests/run-adjoint-fd-test.sh
              $<TARGET_FILE:flow_adjoint> ${_m1d} MODEL_1D_DEBUG
              ${_testout}/fd_pv)
    # General (stencil-local) gradient mechanism: random directional-
    # derivative FD on the full gradient vector computed in one sweep.
    add_test(NAME adjoint_dirderiv_pv_m1d
      COMMAND python3 ${PROJECT_SOURCE_DIR}/tests/run-adjoint-dirderiv-test.py
              $<TARGET_FILE:flow_adjoint> ${_m1d} MODEL_1D_DEBUG pv
              ${_testout}/dirderiv_pv)
    add_test(NAME adjoint_dirderiv_perm_m1d
      COMMAND python3 ${PROJECT_SOURCE_DIR}/tests/run-adjoint-dirderiv-test.py
              $<TARGET_FILE:flow_adjoint> ${_m1d} MODEL_1D_DEBUG perm
              ${_testout}/dirderiv_perm 1e-2)
    # General gradient vs analytic oracle (PV, TRANS, PERM) on one deck.
    add_test(NAME adjoint_general_oracle_m1d
      COMMAND ${PROJECT_SOURCE_DIR}/tests/run-adjoint-general-oracle-test.sh
              $<TARGET_FILE:flow_adjoint> ${_m1d}/MODEL_1D_DEBUG.DATA
              ${_testout}/general_oracle_m1d)
    # Seeded-AD gradient must reproduce the analytic gradient to ~machine
    # precision (exact dR/dθ in one pass).
    add_test(NAME adjoint_general_ad_oracle_m1d
      COMMAND ${PROJECT_SOURCE_DIR}/tests/run-adjoint-general-oracle-test.sh
              $<TARGET_FILE:flow_adjoint> ${_m1d}/MODEL_1D_DEBUG.DATA
              ${_testout}/general_ad_oracle_m1d 1e-10 general-ad)
    add_test(NAME adjoint_fd_trans
      COMMAND ${PROJECT_SOURCE_DIR}/tests/run-adjoint-fd-trans-test.sh
              $<TARGET_FILE:flow_adjoint> ${_m1d} MODEL_1D_DEBUG
              ${_testout}/fd_trans)
    add_test(NAME adjoint_fd_perm
      COMMAND ${PROJECT_SOURCE_DIR}/tests/run-adjoint-fd-perm-test.sh
              $<TARGET_FILE:flow_adjoint> ${_m1d} MODEL_1D_DEBUG
              ${_testout}/fd_perm)
    add_test(NAME adjoint_fd_wellctrl
      COMMAND ${PROJECT_SOURCE_DIR}/tests/run-adjoint-fd-ctrl-test.sh
              $<TARGET_FILE:flow_adjoint> ${_m1d} MODEL_1D_DEBUG
              ${_testout}/fd_ctrl)
    add_test(NAME adjoint_fd_endpoint_swl
      COMMAND ${PROJECT_SOURCE_DIR}/tests/run-adjoint-fd-endpoint-test.sh
              $<TARGET_FILE:flow_adjoint> ${_m1d}
              ${_testout}/fd_endpoint_swl)
    # Same FD verification against the stencil-local endpoint method
    # (cell x ~7 instead of cell x Ncells; whole-domain fallback for
    # perforated cells).
    add_test(NAME adjoint_fd_endpoint_swl_stencil
      COMMAND ${PROJECT_SOURCE_DIR}/tests/run-adjoint-fd-endpoint-test.sh
              $<TARGET_FILE:flow_adjoint> ${_m1d}
              ${_testout}/fd_endpoint_swl_stencil
              SWL 0.0 1e-5 1e-3 pressure-average "" stencil)
    add_test(NAME adjoint_solver_m1d
      COMMAND ${PROJECT_SOURCE_DIR}/tests/run-adjoint-solver-test.sh
              $<TARGET_FILE:flow_adjoint> ${_m1d}/MODEL_1D_DEBUG.DATA
              ${_testout}/solver_m1d)
  endif()
  # Slim archive (no serialized schedule) == full, byte-identical gradients,
  # on the group-control/guide-rate deck (the state most at risk if slim drops
  # something). Guarded here; gated on the SPE1_GRPCTRL deck existing.
  if(EXISTS ${OPM_TESTS_ROOT}/adjoint_tests/SPE1_GRPCTRL/SPE1_GRPCTRL.DATA)
    add_test(NAME adjoint_slim_grpctrl
      COMMAND ${PROJECT_SOURCE_DIR}/tests/run-adjoint-slim-test.sh
              $<TARGET_FILE:flow_adjoint>
              ${OPM_TESTS_ROOT}/adjoint_tests/SPE1_GRPCTRL/SPE1_GRPCTRL.DATA
              ${_testout}/slim_grpctrl)
  endif()
  # Group-control gradient: FD-verified on SPE1 under GCONPROD group
  # control - proves the adjoint gradient is correct when wells follow a
  # group target, not just single-well control (the field-relevant case).
  set(_grpctrl ${OPM_TESTS_ROOT}/adjoint_tests/SPE1_GRPCTRL/SPE1_GRPCTRL.DATA)
  if(EXISTS ${_grpctrl})
    add_test(NAME adjoint_fd_grpctrl
      COMMAND ${PROJECT_SOURCE_DIR}/tests/run-adjoint-fd-grpctrl-test.sh
              $<TARGET_FILE:flow_adjoint> ${_grpctrl} ${_testout}/fd_grpctrl)
  endif()
  # Scale regression: separate-IQ Jacobian (and general-ad gradients) on
  # SPE9 (9000 cells, DISGAS, multi-perforation wells).
  if(EXISTS ${OPM_TESTS_ROOT}/spe9/SPE9_CP_SHORT.DATA)
    add_test(NAME adjoint_jacobian_eval_spe9
      COMMAND ${PROJECT_SOURCE_DIR}/tests/run-adjoint-jacobian-eval-test.sh
              $<TARGET_FILE:flow_adjoint>
              ${OPM_TESTS_ROOT}/spe9/SPE9_CP_SHORT.DATA ${_testout}/jaceval_spe9)
    add_test(NAME adjoint_general_ad_oracle_spe9
      COMMAND ${PROJECT_SOURCE_DIR}/tests/run-adjoint-general-oracle-test.sh
              $<TARGET_FILE:flow_adjoint>
              ${OPM_TESTS_ROOT}/spe9/SPE9_CP_SHORT.DATA
              ${_testout}/general_ad_oracle_spe9 1e-5 general-ad)
    # LARGER-CASE gradient verification by directional FD (per-cell FD is
    # infeasible at 9000 cells). Deck-general MULTPV perturbation vs the adjoint
    # dJ/d(pvmult); one random direction, strict tolerances, eps=1e-6.
    add_test(NAME adjoint_dirderiv_general_spe9
      COMMAND python3 ${PROJECT_SOURCE_DIR}/tests/run-adjoint-dirderiv-general.py
              $<TARGET_FILE:flow_adjoint>
              ${OPM_TESTS_ROOT}/spe9/SPE9_CP_SHORT.DATA
              ${_testout}/dirderiv_general_spe9 pv pressure-average 1e-6 5e-3 1 1)
  endif()
  if(EXISTS ${OPM_TESTS_ROOT}/spe1/SPE1CASE1.DATA)
    add_test(NAME adjoint_replay_spe1
      COMMAND ${PROJECT_SOURCE_DIR}/tests/run-adjoint-replay-test.sh
              $<TARGET_FILE:flow_adjoint>
              ${OPM_TESTS_ROOT}/spe1/SPE1CASE1.DATA ${_testout}/replay_spe1)
    add_test(NAME adjoint_residual_eval_spe1
      COMMAND ${PROJECT_SOURCE_DIR}/tests/run-adjoint-residual-eval-test.sh
              $<TARGET_FILE:flow_adjoint>
              ${OPM_TESTS_ROOT}/spe1/SPE1CASE1.DATA ${_testout}/reseval_spe1)
    add_test(NAME adjoint_jacobian_eval_spe1
      COMMAND ${PROJECT_SOURCE_DIR}/tests/run-adjoint-jacobian-eval-test.sh
              $<TARGET_FILE:flow_adjoint>
              ${OPM_TESTS_ROOT}/spe1/SPE1CASE1.DATA ${_testout}/jaceval_spe1)
    add_test(NAME adjoint_general_oracle_spe1
      COMMAND ${PROJECT_SOURCE_DIR}/tests/run-adjoint-general-oracle-test.sh
              $<TARGET_FILE:flow_adjoint>
              ${OPM_TESTS_ROOT}/spe1/SPE1CASE1.DATA ${_testout}/general_oracle_spe1)
    # Well-objective (bhp) FD gradient on the real DISGAS deck — the FD suite
    # otherwise only runs on the 3-cell MODEL_1D_DEBUG. Asserts the gradient is
    # exact once DRSDT is removed (well/bhp coupling correct), and documents the
    # DRSDT-0 cross-step gap (~3%) that is otherwise missing from dJ/dm.
    add_test(NAME adjoint_drsdt_crossterm_spe1
      COMMAND ${PROJECT_SOURCE_DIR}/tests/run-adjoint-drsdt-crossterm-test.sh
              $<TARGET_FILE:flow_adjoint>
              ${OPM_TESTS_ROOT}/spe1/SPE1CASE1.DATA ${_testout}/drsdt_crossterm_spe1)
    # dCTF/dperm well-coupling term (B1): perm gradient at a defaulted-CTF
    # perforation matches FD (was ~95% off without the well term).
    add_test(NAME adjoint_fd_perm_well_spe1
      COMMAND ${PROJECT_SOURCE_DIR}/tests/run-adjoint-fd-perm-well-test.sh
              $<TARGET_FILE:flow_adjoint>
              ${OPM_TESTS_ROOT}/spe1 ${_testout}/fd_perm_well_spe1)
    # Reservoir PERM gradient by directional FD on a real DISGAS deck (DRSDT
    # stripped, perforations masked, so it is independent of tasks #18 and B1).
    add_test(NAME adjoint_dirderiv_perm_spe1
      COMMAND ${PROJECT_SOURCE_DIR}/tests/run-adjoint-dirderiv-perm-test.sh
              $<TARGET_FILE:flow_adjoint>
              ${OPM_TESTS_ROOT}/spe1/SPE1CASE1.DATA ${_testout}/dirderiv_perm_spe1)
    # --adjoint-config JSON == equivalent flag string; CLI overrides JSON.
    add_test(NAME adjoint_config_spe1
      COMMAND ${PROJECT_SOURCE_DIR}/tests/run-adjoint-config-test.sh
              $<TARGET_FILE:flow_adjoint>
              ${OPM_TESTS_ROOT}/spe1/SPE1CASE1.DATA ${_testout}/config_spe1)
    # Machine-readable field output: cartesian-keyed CSV + JSON metadata sidecar.
    add_test(NAME adjoint_field_output_spe1
      COMMAND ${PROJECT_SOURCE_DIR}/tests/run-adjoint-field-output-test.sh
              $<TARGET_FILE:flow_adjoint>
              ${OPM_TESTS_ROOT}/spe1/SPE1CASE1.DATA ${_testout}/field_output_spe1)
    # One-stop (forward + backward in one process, in-memory archive) ==
    # two-process, byte-identical gradients, no state files.
    add_test(NAME adjoint_onestop_spe1
      COMMAND ${PROJECT_SOURCE_DIR}/tests/run-adjoint-onestop-test.sh
              $<TARGET_FILE:flow_adjoint>
              ${OPM_TESTS_ROOT}/spe1/SPE1CASE1.DATA ${_testout}/onestop_spe1)
    add_test(NAME adjoint_general_ad_oracle_spe1
      COMMAND ${PROJECT_SOURCE_DIR}/tests/run-adjoint-general-oracle-test.sh
              $<TARGET_FILE:flow_adjoint>
              ${OPM_TESTS_ROOT}/spe1/SPE1CASE1.DATA
              ${_testout}/general_ad_oracle_spe1 1e-10 general-ad)
    add_test(NAME adjoint_matchref_spe1
      COMMAND ${PROJECT_SOURCE_DIR}/tests/run-adjoint-matchref-test.sh
              $<TARGET_FILE:flow_adjoint>
              ${OPM_TESTS_ROOT}/spe1/SPE1CASE1.DATA ${_testout}/matchref)
    # Weighted (:w=), time-windowed (:t0=/:t1=) and wildcard-well matchref:
    # same FD-vs-adjoint check, guarding the objective-layer extensions.
    add_test(NAME adjoint_matchref_extended_spe1
      COMMAND ${PROJECT_SOURCE_DIR}/tests/run-adjoint-matchref-test.sh
              $<TARGET_FILE:flow_adjoint>
              ${OPM_TESTS_ROOT}/spe1/SPE1CASE1.DATA ${_testout}/matchref_ext
              2e-5 1e-3 0.285 "matchref:PROD*:gas:%REF%:w=2.0:t0=0:t1=1800")
    # SPE1 needs ~125 backward solves; iterative agreement vs umfpack is
    # bounded by the accumulated lambda error (reduction 1e-14 -> ~1e-6).
    add_test(NAME adjoint_solver_spe1
      COMMAND ${PROJECT_SOURCE_DIR}/tests/run-adjoint-solver-test.sh
              $<TARGET_FILE:flow_adjoint>
              ${OPM_TESTS_ROOT}/spe1/SPE1CASE1.DATA ${_testout}/solver_spe1
              1e-5 cprt)
    # Parallel (np=2) vs serial gradient agreement; the script skips
    # itself when no MPI launcher is available.
    add_test(NAME adjoint_mpi_spe1
      COMMAND ${PROJECT_SOURCE_DIR}/tests/run-adjoint-mpi-test.sh
              $<TARGET_FILE:flow_adjoint>
              ${OPM_TESTS_ROOT}/spe1/SPE1CASE1.DATA ${_testout}/mpi_spe1 2)
  endif()
endif()
# Match the opm build conventions (notably -UNDEBUG in Release when
# WITH_NDEBUG=OFF; an NDEBUG mismatch against the opm-simulators library
# is an ABI break in classes with assertion-only members).
include(UseOptimization)
use_additional_optimization(TARGET flow_adjoint)
