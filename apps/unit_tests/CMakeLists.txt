# FILE(GLOB _tests RELATIVE "${CMAKE_CURRENT_SOURCE_DIR}" "*.cpp")
set(unit_tests "test_init;test_nan;test_ylm;test_sinx_cosx;test_gvec;test_fft_correctness_1;\
test_fft_correctness_2;test_fft_real_1;test_fft_real_2;test_fft_real_3;\
test_spline;test_rot_ylm;test_linalg;test_wf_ortho;test_serialize;test_mempool;test_sim_ctx;test_roundoff")

foreach(name ${unit_tests})
  add_executable(${name} "${name}.cpp")
  SIRIUS_SETUP_TARGET(${name})
  install(TARGETS ${name} RUNTIME DESTINATION "${CMAKE_INSTALL_PREFIX}/bin")
endforeach()
file(COPY "${CMAKE_CURRENT_SOURCE_DIR}/unit_tests.x" DESTINATION "${CMAKE_CURRENT_BINARY_DIR}")
install(FILES "unit_tests.x" DESTINATION "${CMAKE_INSTALL_PREFIX}/bin")
