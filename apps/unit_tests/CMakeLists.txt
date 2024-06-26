set(unit_tests "test_init;test_nan;test_ylm;test_rlm;test_sinx_cosx;test_gvec;test_fft_correctness_1;\
test_fft_correctness_2;test_fft_correctness_3;test_rlm_deriv;test_spline_1;\
test_spline_2;test_rot_ylm;test_linalg;test_wf_ortho_1;test_serialize;test_mempool;test_sim_ctx;test_roundoff;\
test_sht_lapl;test_sht;test_spheric_function;test_splindex;test_gaunt_coeff_1;test_gaunt_coeff_2;\
test_init_ctx;test_cmd_args;test_geom3d;test_any_ptr")

foreach(name ${unit_tests})
  add_executable(${name} "${name}.cpp")
  target_link_libraries(${name} PRIVATE sirius_cxx)
  install(TARGETS ${name} RUNTIME DESTINATION "${CMAKE_INSTALL_PREFIX}/bin")
  add_test(NAME ${name} COMMAND ${name})
endforeach()

file(COPY "${CMAKE_CURRENT_SOURCE_DIR}/unit_tests.x" DESTINATION "${CMAKE_CURRENT_BINARY_DIR}")
install(FILES "unit_tests.x" DESTINATION "${CMAKE_INSTALL_PREFIX}/bin")

add_subdirectory(multi_cg)
