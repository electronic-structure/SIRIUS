set(_tests "test_hdf5;test_allgather;mt_function;splindex;hydrogen;\
read_atom;test_mdarray;test_xc;test_hloc;\
test_mpi_grid;test_enu;test_eigen_v2;test_gemm;test_gemm2")

foreach(_test ${_tests})
  add_executable(${_test} "${_test}.cpp")
  SIRIUS_SETUP_TARGET(${_test})
  install(TARGETS ${_test} RUNTIME DESTINATION "${CMAKE_INSTALL_PREFIX}/bin")
endforeach()

#set(_test test_gemm)
#add_executable("${_test}_real" "${_test}.cpp")
#SIRIUS_SETUP_TARGET(${_test})
#target_compile_definitions(${_test} PRIVATE COMPILE_DEFINITIONS "-D__TEST_REAL")
#install(TARGETS ${_test} RUNTIME DESTINATION "${CMAKE_INSTALL_PREFIX}/bin")
#
#set(_test test_gemm2)
#add_executable("${_test}_real" "${_test}.cpp")
#SIRIUS_SETUP_TARGET(${_test})
#target_compile_definitions(${_test} PRIVATE COMPILE_DEFINITIONS "-D__TEST_REAL")
#install(TARGETS ${_test} RUNTIME DESTINATION "${CMAKE_INSTALL_PREFIX}/bin")
