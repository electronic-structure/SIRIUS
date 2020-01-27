set(_tests "test_hdf5;test_allgather;mt_function;splindex;hydrogen;\
read_atom;test_mdarray;test_xc;test_hloc;\
test_mpi_grid;test_enu;test_eigen_v2;test_gemm;test_gemm2;test_wf_inner_v3;test_memop;\
test_mem_pool;test_mem_alloc;test_examples;test_fft_full_grid;test_wf_inner_v4;test_bcast_v2;test_p2p_cyclic;\
test_wf_ortho_6;test_mixer_v1;test_davidson;test_lapw_xc")

foreach(_test ${_tests})
  add_executable(${_test} ${_test}.cpp)
  target_link_libraries(${_test} PRIVATE sirius)
  install(TARGETS ${_test} RUNTIME DESTINATION "${CMAKE_INSTALL_PREFIX}/bin")
endforeach()

#set(_test test_gemm)
#add_executable("${_test}_real" "${_test}.cpp")
#target_link_libraries(${_test} PRIVATE sirius)
#target_compile_definitions(${_test} PRIVATE COMPILE_DEFINITIONS "-D__TEST_REAL")
#install(TARGETS ${_test} RUNTIME DESTINATION "${CMAKE_INSTALL_PREFIX}/bin")
#
#set(_test test_gemm2)
#add_executable("${_test}_real" "${_test}.cpp")
#target_link_libraries(${_test} PRIVATE sirius)
#target_compile_definitions(${_test} PRIVATE COMPILE_DEFINITIONS "-D__TEST_REAL")
#install(TARGETS ${_test} RUNTIME DESTINATION "${CMAKE_INSTALL_PREFIX}/bin")
