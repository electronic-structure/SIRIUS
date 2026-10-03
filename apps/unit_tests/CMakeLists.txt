set(unit_tests "test_init;test_nan;test_ylm;test_rlm;test_sinx_cosx;test_gvec;test_fft_correctness_1;\
test_fft_correctness_2;test_fft_correctness_3;test_rlm_deriv;test_spline_1;\
test_spline_2;test_rot_ylm;test_linalg;test_wf_ortho_1;test_serialize;test_mempool;test_sim_ctx;test_roundoff;\
test_sht_lapl;test_sht;test_spheric_function;test_splindex;test_gaunt_coeff_1;test_gaunt_coeff_2;test_gaunt_coeff_3;\
test_gaunt_coeff_4;test_init_ctx;test_cmd_args;test_geom3d;test_any_ptr;test_kinetic_density;test_spheric_kinetic_density;test_spheric_interpolation;test_smooth_periodic_sampling;test_paw_xc_correction;test_lapw_xc_field_map;test_lapw_external_xc;test_core_kinetic_density;test_external_xc")

foreach(name ${unit_tests})
  add_executable(${name} "${name}.cpp")
  target_link_libraries(${name} PRIVATE sirius_cxx)
  install(TARGETS ${name} RUNTIME DESTINATION "${CMAKE_INSTALL_PREFIX}/bin")
  add_test(NAME ${name} COMMAND ${name})
endforeach()

function(add_mpi_unit_test name target timeout)
  add_test(NAME ${name}
    COMMAND ${MPIEXEC_EXECUTABLE} ${MPIEXEC_NUMPROC_FLAG} 2 ${MPIEXEC_PREFLAGS}
            $<TARGET_FILE:${target}> ${ARGN} ${MPIEXEC_POSTFLAGS})
  set_tests_properties(${name} PROPERTIES
    TIMEOUT ${timeout} PROCESSORS 2 ENVIRONMENT "OMP_NUM_THREADS=1;OPENBLAS_NUM_THREADS=1")
endfunction()

add_executable(test_lapw_atom_input test_lapw_atom_input.cpp)
target_link_libraries(test_lapw_atom_input PRIVATE sirius_cxx)
add_test(NAME test_lapw_atom_input COMMAND test_lapw_atom_input)
set_tests_properties(test_lapw_atom_input PROPERTIES
  TIMEOUT 30 ENVIRONMENT "OMP_NUM_THREADS=1;OPENBLAS_NUM_THREADS=1")

add_executable(test_radial_core_solver test_radial_core_solver.cpp)
target_link_libraries(test_radial_core_solver PRIVATE sirius_cxx)
add_test(NAME test_radial_core_solver COMMAND test_radial_core_solver)
set_tests_properties(test_radial_core_solver PROPERTIES
  TIMEOUT 120 ENVIRONMENT "OMP_NUM_THREADS=1;OPENBLAS_NUM_THREADS=1")

include(CheckCXXSymbolExists)
include(CMakePushCheckState)
cmake_push_check_state(RESET)
set(CMAKE_REQUIRED_LIBRARIES sirius::libxc)
check_cxx_symbol_exists(xc_func_set_fhc_enforcement "xc.h" SIRIUS_LIBXC_HAS_FHC_CONTROL)
cmake_pop_check_state()

if(SIRIUS_LIBXC_HAS_FHC_CONTROL)
  add_executable(test_meta_gga test_meta_gga.cpp)
  target_link_libraries(test_meta_gga PRIVATE sirius_cxx)
  add_test(NAME test_meta_gga COMMAND test_meta_gga)
  set_tests_properties(test_meta_gga PROPERTIES TIMEOUT 120 ENVIRONMENT "OMP_NUM_THREADS=1;OPENBLAS_NUM_THREADS=1")
  add_test(NAME test_meta_gga_paw COMMAND test_meta_gga --paw)
  set_tests_properties(test_meta_gga_paw PROPERTIES TIMEOUT 120 ENVIRONMENT "OMP_NUM_THREADS=1;OPENBLAS_NUM_THREADS=1")
  add_executable(test_lapw_meta_gga test_lapw_meta_gga.cpp)
  target_link_libraries(test_lapw_meta_gga PRIVATE sirius_cxx)
  add_test(NAME test_lapw_meta_gga COMMAND test_lapw_meta_gga)
  set_tests_properties(test_lapw_meta_gga PROPERTIES
    TIMEOUT 120 ENVIRONMENT "OMP_NUM_THREADS=1;OPENBLAS_NUM_THREADS=1")
  add_test(NAME test_lapw_meta_gga_scf COMMAND test_lapw_meta_gga scf)
  set_tests_properties(test_lapw_meta_gga_scf PROPERTIES
    TIMEOUT 300 ENVIRONMENT "OMP_NUM_THREADS=1;OPENBLAS_NUM_THREADS=1")
  add_test(NAME test_lapw_meta_gga_core COMMAND test_lapw_meta_gga core single)
  set_tests_properties(test_lapw_meta_gga_core PROPERTIES
    TIMEOUT 600 ENVIRONMENT "OMP_NUM_THREADS=1;OPENBLAS_NUM_THREADS=1")
  if(MPIEXEC_EXECUTABLE)
    add_mpi_unit_test(test_lapw_meta_gga_mpi test_lapw_meta_gga 120)
    add_mpi_unit_test(test_lapw_meta_gga_scf_mpi test_lapw_meta_gga 600 scf)
    add_mpi_unit_test(test_lapw_meta_gga_core_mpi test_lapw_meta_gga 600 core spin)
    add_mpi_unit_test(test_meta_gga_mpi test_meta_gga 120)
    add_mpi_unit_test(test_meta_gga_paw_mpi test_meta_gga 120 --paw)
  endif()
endif()

set(h_paw_setup "${PROJECT_SOURCE_DIR}/examples/pp-pw/C42H58ClNO2Si_420_atoms/H.pz-kjpaw_psl.0.1.UPF.json")
add_test(NAME test_paw_xc_correction_h_setup COMMAND test_paw_xc_correction "${h_paw_setup}")
set_tests_properties(test_paw_xc_correction_h_setup PROPERTIES
  TIMEOUT 60 ENVIRONMENT "OMP_NUM_THREADS=1;OPENBLAS_NUM_THREADS=1")

set_tests_properties(test_kinetic_density test_spheric_kinetic_density test_spheric_interpolation test_smooth_periodic_sampling test_paw_xc_correction test_lapw_xc_field_map test_lapw_external_xc test_core_kinetic_density test_external_xc PROPERTIES
  TIMEOUT 60 ENVIRONMENT "OMP_NUM_THREADS=1;OPENBLAS_NUM_THREADS=1")
add_test(NAME test_lapw_external_xc_core COMMAND test_lapw_external_xc core)
set_tests_properties(test_lapw_external_xc_core PROPERTIES
  TIMEOUT 240 ENVIRONMENT "OMP_NUM_THREADS=1;OPENBLAS_NUM_THREADS=1")
if(MPIEXEC_EXECUTABLE)
  add_mpi_unit_test(test_lapw_external_xc_core_mpi test_lapw_external_xc 240 core)
  add_mpi_unit_test(test_paw_xc_correction_h_setup_mpi test_paw_xc_correction 60 "${h_paw_setup}")
  foreach(name test_kinetic_density test_smooth_periodic_sampling test_paw_xc_correction test_lapw_xc_field_map test_lapw_external_xc test_core_kinetic_density test_external_xc)
    add_mpi_unit_test(${name}_mpi ${name} 60)
  endforeach()
endif()

if(SIRIUS_CREATE_FORTRAN_BINDINGS)
  add_executable(test_core_kinetic_density_api test_core_kinetic_density_api.f90)
  target_link_libraries(test_core_kinetic_density_api PRIVATE sirius MPI::MPI_Fortran)
  add_test(NAME test_core_kinetic_density_api COMMAND test_core_kinetic_density_api)
  set_tests_properties(test_core_kinetic_density_api PROPERTIES
    TIMEOUT 60 ENVIRONMENT "OMP_NUM_THREADS=1;OPENBLAS_NUM_THREADS=1")
  add_executable(test_external_xc_api test_external_xc_api.f90)
  target_link_libraries(test_external_xc_api PRIVATE sirius MPI::MPI_Fortran)
  add_test(NAME test_external_xc_api COMMAND test_external_xc_api)
  set_tests_properties(test_external_xc_api PROPERTIES
    TIMEOUT 60 ENVIRONMENT "OMP_NUM_THREADS=1;OPENBLAS_NUM_THREADS=1")
  if(MPIEXEC_EXECUTABLE)
    add_mpi_unit_test(test_external_xc_api_mpi test_external_xc_api 60)
  endif()
endif()

file(COPY "${CMAKE_CURRENT_SOURCE_DIR}/unit_tests.x" DESTINATION "${CMAKE_CURRENT_BINARY_DIR}")
install(FILES "unit_tests.x" DESTINATION "${CMAKE_INSTALL_PREFIX}/bin")

add_subdirectory(multi_cg)
