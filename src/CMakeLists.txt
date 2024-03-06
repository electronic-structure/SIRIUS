# workaround for set_target_properties(... POSITION_INDEPENDENT_CODE ON) not
# working correctly
# list(APPEND CUDA_NVCC_FLAGS "-Xcompiler -fPIC")

if(SIRIUS_CREATE_FORTRAN_BINDINGS)
  set(_FSOURCES "api/sirius_api.cpp;api/sirius.f90")
endif()

if(SIRIUS_USE_CUDA OR SIRIUS_USE_ROCM)
  set(_CUSOURCES
    "core/gpu_kernels/augmentation_operator.cu"
    "core/gpu_kernels/create_beta_gk.cu"
    "core/gpu_kernels/checksum.cu"
    "core/gpu_kernels/density_rg.cu"
    "core/gpu_kernels/generate_dm_pw.cu"
    "core/gpu_kernels/generate_phase_factors.cu"
    "core/gpu_kernels/local_operator.cu"
    "core/gpu_kernels/mul_veff_with_phase_factors.cu"
    "core/gpu_kernels/residuals_aux.cu"
    "core/gpu_kernels/scale_matrix.cu"
    "core/gpu_kernels/spherical_harmonics.cu"
    "core/gpu_kernels/spline.cu"
    "core/gpu_kernels/sum_q_pw_dm_pw.cu"
    "nlcglib/preconditioner/diag_mm.cu"
  )
endif()

set(_SOURCES
  "hamiltonian/hamiltonian.cpp"
  "hamiltonian/hamiltonian_k.cpp"
  "hamiltonian/local_operator.cpp"
  "hamiltonian/non_local_operator.cpp"
  "hamiltonian/s_u_operator.cpp"
  "density/density.cpp"
  "density/augmentation_operator.cpp"
  "density/occupation_matrix.cpp"
  "dft/dft_ground_state.cpp"
  "dft/energy.cpp"
  "dft/smearing.cpp"
  "beta_projectors/beta_projectors_base.cpp"
  "hubbard/hubbard.cpp"
  "hubbard/hubbard_occupancies_derivatives.cpp"
  "hubbard/hubbard_potential_energy.cpp"
  "hubbard/hubbard_matrix.cpp"
  "potential/generate_d_operator_matrix.cpp"
  "potential/generate_pw_coeffs.cpp"
  "potential/paw_potential.cpp"
  "potential/poisson.cpp"
  "potential/xc.cpp"
  "potential/xc_mt.cpp"
  "potential/potential.cpp"
  "potential/check_xc_potential.cpp"
  "unit_cell/unit_cell.cpp"
  "unit_cell/atom_type.cpp"
  "unit_cell/atom_symmetry_class.cpp"
  "symmetry/crystal_symmetry.cpp"
  "geometry/force.cpp"
  "geometry/stress.cpp"
  "k_point/generate_fv_states.cpp"
  "k_point/generate_spinor_wave_functions.cpp"
  "k_point/k_point.cpp"
  "k_point/k_point_set.cpp"
  "radial/radial_integrals.cpp"
  "mixer/mixer_functions.cpp"
  "nlcglib/adaptor.cpp"
  "context/simulation_context.cpp"
  "context/simulation_parameters.cpp"
  "core/cmd_args.cpp"
  "core/sht/sht.cpp"
  "core/fft/gvec.cpp"
  "core/mpi/communicator.cpp"
  "core/sf/sbessel.cpp"
  "core/acc/acc.cpp"
  "core/acc/acc_blas.cpp"
  "core/acc/cusolver.cpp"
  "core/acc/rocsolver.cpp"
  "core/profiler.cpp"
  "core/rt_graph.cpp"
  "core/ostream_tools.cpp"
  "core/sirius_version.cpp"
  "core/la/eigensolver.cpp"
  "core/la/eigenproblem.cpp"
  "core/la/linalg_spla.cpp"
  "core/la/dmatrix.cpp"
  "core/memory.cpp"
  )


  # create library with .cpp, .cu and .f90 sources
add_library(sirius_cxx "${_SOURCES};${_CUSOURCES};")
target_compile_features(sirius_cxx PUBLIC cxx_std_17)
target_compile_features(sirius_cxx PUBLIC c_std_99)
target_compile_features(sirius_cxx PUBLIC cuda_std_17)
set_target_properties(sirius_cxx PROPERTIES POSITION_INDEPENDENT_CODE ON)

if(SIRIUS_USE_ROCM)
  set_source_files_properties(${_CUSOURCES} PROPERTIES LANGUAGE HIP)
  set_target_properties(sirius_cxx PROPERTIES HIP_ARCHITECTURES ${CMAKE_HIP_ARCHITECTURES})
endif()

#if(SIRIUS_USE_CUDA)
#  set_target_properties(sirius PROPERTIES CUDA_ARCHITECTURES ${CMAKE_CUDA_ARCHITECTURES})
#endif()

target_link_libraries(sirius_cxx PUBLIC ${GSL_LIBRARY}
  MPI::MPI_CXX
  sirius::libxc
  sirius::libspg
  sirius::hdf5
  costa::costa
  $<TARGET_NAME_IF_EXISTS:sirius::elpa>
  $<$<BOOL:${SIRIUS_USE_DLAF}>:DLAF::DLAF>
  $<TARGET_NAME_IF_EXISTS:sirius::magma>
  $<TARGET_NAME_IF_EXISTS:sirius::libvdwxc>
  $<TARGET_NAME_IF_EXISTS:nlcglib::nlcglib>
  $<TARGET_NAME_IF_EXISTS:kokkos::kokkos>
  SpFFT::spfft
  SPLA::spla
  "${SIRIUS_LINALG_LIB}"
  $<$<BOOL:${SIRIUS_USE_MEMORY_POOL}>:umpire>
  $<$<BOOL:${SIRIUS_USE_NVTX}>:nvToolsExt>
  $<TARGET_NAME_IF_EXISTS:sirius::cudalibs>
  $<$<BOOL:${SIRIUS_USE_ROCM}>:roc::rocsolver>
  $<$<BOOL:${SIRIUS_USE_ROCM}>:roc::rocblas>
  $<$<BOOL:${SIRIUS_USE_ROCM}>:hip::host>
  $<TARGET_NAME_IF_EXISTS:OpenMP::OpenMP_CXX>
)

target_include_directories(sirius_cxx PUBLIC $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/src>
                                             $<BUILD_INTERFACE:${PROJECT_BINARY_DIR}/src>)
target_compile_definitions(sirius_cxx PUBLIC
  $<$<BOOL:${SIRIUS_USE_PROFILER}>:SIRIUS_PROFILE>
  $<$<BOOL:${SIRIUS_USE_SCALAPACK}>:SIRIUS_SCALAPACK>
  $<$<BOOL:${SIRIUS_USE_MEMORY_POOL}>:SIRIUS_USE_MEMORY_POOL>
  $<$<BOOL:${SIRIUS_USE_POWER_COUNTER}>:SIRIUS_SIRIUS_USE_POWER_COUNTER>
  $<$<BOOL:${SIRIUS_USE_DLAF}>:SIRIUS_DLAF>
  $<$<BOOL:${SIRIUS_USE_ELPA}>:SIRIUS_ELPA>
  $<$<BOOL:${SIRIUS_USE_NLCGLIB}>:SIRIUS_NLCGLIB>
  $<$<BOOL:${SIRIUS_USE_CUDA}>:SIRIUS_GPU SIRIUS_CUDA>
  $<$<BOOL:${SIRIUS_USE_NVTX}>:SIRIUS_CUDA_NVTX>
  $<$<BOOL:${SIRIUS_USE_MAGMA}>:SIRIUS_MAGMA>
  $<$<BOOL:${SIRIUS_USE_ROCM}>:SIRIUS_GPU SIRIUS_ROCM>
  $<$<BOOL:${SIRIUS_USE_VDWXC}>:SIRIUS_USE_VDWXC>
  $<$<BOOL:${SIRIUS_USE_FP32_BOOL}>:SIRIUS_USE_FP32>
  $<$<BOOL:${SIRIUS_USE_VCSQNM}>:SIRIUS_VCSQNM>
  $<$<BOOL:${SIRIUS_HAVE_LIBVDW_WITH_MPI}>:SIRIUS_HAVE_VDWXC_MPI>
  $<$<AND:$<BOOL:${SIRIUS_USE_MAGMA}>,$<BOOL:${SIRIUS_USE_ROCM}>>:HAVE_HIP> # Required for magma headers
)

if(SIRIUS_CREATE_FORTRAN_BINDINGS)
  add_library(sirius "${_FSOURCES}")
  target_link_libraries(sirius PRIVATE sirius_cxx)
  target_include_directories(sirius PUBLIC
    $<BUILD_INTERFACE:${PROJECT_BINARY_DIR}/src/mod_files>)

  set_target_properties(sirius PROPERTIES Fortran_MODULE_DIRECTORY mod_files)
  install(FILES "${PROJECT_BINARY_DIR}/src/mod_files/sirius.mod"
          DESTINATION "${CMAKE_INSTALL_INCLUDEDIR}/sirius")
  install(TARGETS sirius
    EXPORT sirius_targets
    ARCHIVE DESTINATION "${CMAKE_INSTALL_LIBDIR}"
    LIBRARY DESTINATION "${CMAKE_INSTALL_LIBDIR}"
    INCLUDES DESTINATION "${CMAKE_INSTALL_INCLUDEDIR}/sirius")
  install(EXPORT sirius_targets
    FILE siriusTargets.cmake
    NAMESPACE sirius::
    DESTINATION "${CMAKE_INSTALL_LIBDIR}/cmake/sirius")
endif()

install(TARGETS sirius_cxx
  EXPORT sirius_cxx_targets
  ARCHIVE DESTINATION "${CMAKE_INSTALL_LIBDIR}"
  LIBRARY DESTINATION "${CMAKE_INSTALL_LIBDIR}"
  INCLUDES DESTINATION "${CMAKE_INSTALL_INCLUDEDIR}/sirius_cxx")
install(EXPORT sirius_cxx_targets
  FILE sirius_cxxTargets.cmake
  NAMESPACE sirius::
  DESTINATION "${CMAKE_INSTALL_LIBDIR}/cmake/sirius_cxx")
install(DIRECTORY "${PROJECT_SOURCE_DIR}/src"
  DESTINATION "${CMAKE_INSTALL_INCLUDEDIR}/sirius"
  FILES_MATCHING REGEX ".*(hpp|h)$")
install(FILES "${PROJECT_BINARY_DIR}/src/core/version.hpp"
  "${PROJECT_BINARY_DIR}/src/context/input_schema.hpp"
  DESTINATION "${CMAKE_INSTALL_INCLUDEDIR}/sirius")
