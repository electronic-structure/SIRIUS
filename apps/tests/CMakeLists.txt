set(_tests "test_alloc;test_hdf5;test_allgather;\
read_atom;test_mdarray;test_xc;test_hloc;test_vector_calculus;\
test_mpi_grid;test_enu;test_eigen;test_gemm;test_gemm2;test_wf_inner;test_memop;\
test_mem_pool;test_mem_alloc;test_examples;test_bcast_v2;test_p2p_cyclic;\
test_wf_ortho;test_mixer;test_davidson;test_lapw_xc;test_phase;test_bessel;test_fp;test_pppw_xc;\
test_exc_vxc;test_atomic_orbital_index;test_sym;test_blacs;test_reduce;test_comm_split;test_wf_trans;\
test_wf_fft;test_potrf;test_lr_solver;test_radial_solver")

foreach(_test ${_tests})
  add_executable(${_test} ${_test}.cpp)
  target_link_libraries(${_test} PRIVATE sirius)
  install(TARGETS ${_test} RUNTIME DESTINATION "${CMAKE_INSTALL_PREFIX}/bin")
endforeach()

if(SIRIUS_CREATE_FORTRAN_BINDINGS)
  add_executable(test_fortran_api test_fortran_api.f90)
  target_link_libraries(test_fortran_api PRIVATE sirius MPI::MPI_Fortran)
  install(TARGETS test_fortran_api RUNTIME DESTINATION "${CMAKE_INSTALL_PREFIX}/bin")

  add_executable(test_srvo3_pwpp test_srvo3_pwpp.f90)
  target_link_libraries(test_srvo3_pwpp PRIVATE sirius MPI::MPI_Fortran)
  install(TARGETS test_srvo3_pwpp RUNTIME DESTINATION "${CMAKE_INSTALL_PREFIX}/bin")
endif(SIRIUS_CREATE_FORTRAN_BINDINGS)
