set(unit_tests "test_init;test_nan;test_ylm;test_rlm;test_sinx_cosx;test_gvec;test_fft_correctness_1;\
test_fft_correctness_2;test_fft_correctness_3;test_rlm_deriv;test_spline_1;\
test_spline_2;test_rot_ylm;test_linalg;test_wf_ortho_1;test_serialize;test_mempool;test_sim_ctx;test_roundoff;\
test_sht_lapl;test_sht;test_spheric_function;test_splindex;test_gaunt_coeff_1;test_gaunt_coeff_2;test_gaunt_coeff_3;\
test_gaunt_coeff_4;test_init_ctx;test_cmd_args;test_geom3d;test_any_ptr;test_kinetic_density;test_spheric_kinetic_density;test_core_kinetic_density")

foreach(name ${unit_tests})
  add_executable(${name} "${name}.cpp")
  target_link_libraries(${name} PRIVATE sirius_cxx)
  install(TARGETS ${name} RUNTIME DESTINATION "${CMAKE_INSTALL_PREFIX}/bin")
  add_test(NAME ${name} COMMAND ${name})
endforeach()

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
    add_test(NAME test_lapw_meta_gga_mpi
      COMMAND ${MPIEXEC_EXECUTABLE} ${MPIEXEC_NUMPROC_FLAG} 2 ${MPIEXEC_PREFLAGS}
              $<TARGET_FILE:test_lapw_meta_gga> ${MPIEXEC_POSTFLAGS})
    set_tests_properties(test_lapw_meta_gga_mpi PROPERTIES
      TIMEOUT 120 PROCESSORS 2 ENVIRONMENT "OMP_NUM_THREADS=1;OPENBLAS_NUM_THREADS=1")
    add_test(NAME test_lapw_meta_gga_scf_mpi
      COMMAND ${MPIEXEC_EXECUTABLE} ${MPIEXEC_NUMPROC_FLAG} 2 ${MPIEXEC_PREFLAGS}
              $<TARGET_FILE:test_lapw_meta_gga> scf ${MPIEXEC_POSTFLAGS})
    set_tests_properties(test_lapw_meta_gga_scf_mpi PROPERTIES
      TIMEOUT 600 PROCESSORS 2 ENVIRONMENT "OMP_NUM_THREADS=1;OPENBLAS_NUM_THREADS=1")
    add_test(NAME test_lapw_meta_gga_core_mpi
      COMMAND ${MPIEXEC_EXECUTABLE} ${MPIEXEC_NUMPROC_FLAG} 2 ${MPIEXEC_PREFLAGS}
              $<TARGET_FILE:test_lapw_meta_gga> core spin ${MPIEXEC_POSTFLAGS})
    set_tests_properties(test_lapw_meta_gga_core_mpi PROPERTIES
      TIMEOUT 600 PROCESSORS 2 ENVIRONMENT "OMP_NUM_THREADS=1;OPENBLAS_NUM_THREADS=1")
    add_test(NAME test_meta_gga_mpi
      COMMAND ${MPIEXEC_EXECUTABLE} ${MPIEXEC_NUMPROC_FLAG} 2 ${MPIEXEC_PREFLAGS}
              $<TARGET_FILE:test_meta_gga> ${MPIEXEC_POSTFLAGS})
    set_tests_properties(test_meta_gga_mpi PROPERTIES
      TIMEOUT 120 PROCESSORS 2 ENVIRONMENT "OMP_NUM_THREADS=1;OPENBLAS_NUM_THREADS=1")
    add_test(NAME test_meta_gga_paw_mpi
      COMMAND ${MPIEXEC_EXECUTABLE} ${MPIEXEC_NUMPROC_FLAG} 2 ${MPIEXEC_PREFLAGS}
              $<TARGET_FILE:test_meta_gga> --paw ${MPIEXEC_POSTFLAGS})
    set_tests_properties(test_meta_gga_paw_mpi PROPERTIES
      TIMEOUT 120 PROCESSORS 2 ENVIRONMENT "OMP_NUM_THREADS=1;OPENBLAS_NUM_THREADS=1")
  endif()
endif()

set_tests_properties(test_kinetic_density test_spheric_kinetic_density test_core_kinetic_density PROPERTIES
  TIMEOUT 60 ENVIRONMENT "OMP_NUM_THREADS=1;OPENBLAS_NUM_THREADS=1")
if(MPIEXEC_EXECUTABLE)
  foreach(name test_kinetic_density test_core_kinetic_density)
    add_test(NAME ${name}_mpi
      COMMAND ${MPIEXEC_EXECUTABLE} ${MPIEXEC_NUMPROC_FLAG} 2 ${MPIEXEC_PREFLAGS}
              $<TARGET_FILE:${name}> ${MPIEXEC_POSTFLAGS})
    set_tests_properties(${name}_mpi PROPERTIES
      TIMEOUT 60 PROCESSORS 2 ENVIRONMENT "OMP_NUM_THREADS=1;OPENBLAS_NUM_THREADS=1")
  endforeach()
endif()

if(SIRIUS_CREATE_FORTRAN_BINDINGS)
  add_executable(test_core_kinetic_density_api test_core_kinetic_density_api.f90)
  target_link_libraries(test_core_kinetic_density_api PRIVATE sirius MPI::MPI_Fortran)
  add_test(NAME test_core_kinetic_density_api COMMAND test_core_kinetic_density_api)
  set_tests_properties(test_core_kinetic_density_api PROPERTIES
    TIMEOUT 60 ENVIRONMENT "OMP_NUM_THREADS=1;OPENBLAS_NUM_THREADS=1")
endif()

file(COPY "${CMAKE_CURRENT_SOURCE_DIR}/unit_tests.x" DESTINATION "${CMAKE_CURRENT_BINARY_DIR}")
install(FILES "unit_tests.x" DESTINATION "${CMAKE_INSTALL_PREFIX}/bin")

add_subdirectory(multi_cg)
