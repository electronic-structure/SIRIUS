# workaround for set_target_properties(... POSITION_INDEPENDENT_CODE ON) not
# working correctly
# list(APPEND CUDA_NVCC_FLAGS "-Xcompiler -fPIC")

if(CREATE_FORTRAN_BINDINGS)
  set(_FSOURCES "api/sirius_api.cpp;api/sirius.f90")
endif()

if(USE_CUDA OR USE_ROCM)
  file(GLOB_RECURSE CUFILES "gpu/*.cu")
  set(_CUSOURCES "${CUFILES}")
endif()

set(_SOURSES
  "hamiltonian/hamiltonian.cpp"
  "hamiltonian/hamiltonian_k.cpp"
  "hamiltonian/local_operator.cpp"
  "hamiltonian/non_local_operator.cpp"
  "band/band.cpp"
  "band/solve.cpp"
  "band/diag_pseudo_potential.cpp"
  "band/diag_full_potential.cpp"
  "band/residuals.cpp"
  "density/density.cpp"
  "density/augmentation_operator.cpp"
  "dft/dft_ground_state.cpp"
  "dft/energy.cpp"
  "beta_projectors/beta_projectors_base.cpp"
  "hubbard/apply_hubbard_potential.cpp"
  "hubbard/hubbard.cpp"
  "hubbard/hubbard_occupancies_derivatives.cpp"
  "hubbard/hubbard_occupancy.cpp"
  "hubbard/hubbard_potential_energy.cpp"
  "potential/generate_d_operator_matrix.cpp"
  "potential/generate_pw_coeffs.cpp"
  "potential/paw_potential.cpp"
  "potential/poisson.cpp"
  "potential/xc.cpp"
  "unit_cell/unit_cell.cpp"
  "unit_cell/unit_cell_symmetry.cpp"
  "simulation_context.cpp"
  "simulation_parameters.cpp"
  "geometry/force.cpp"
  "geometry/stress.cpp"
  "geometry/non_local_functor.cpp"
  "k_point/generate_atomic_wave_functions.cpp"
  "k_point/generate_fv_states.cpp"
  "k_point/generate_spinor_wave_functions.cpp"
  "k_point/k_point.cpp"
  "k_point/k_point_set.cpp"
  "k_point/generate_gklo_basis.cpp"
  "k_point/test_fv_states.cpp"
  "function3d/field4d.cpp"
  "radial/radial_integrals.cpp"
  "specfunc/sbessel.cpp"
  "utils/cmd_args.cpp"
  "utils/utils.cpp"
  "utils/rt_graph.cpp"
  "utils/profiler.cpp"
  "SDDK/wf_inner.cpp"
  "SDDK/wf_ortho.cpp"
  "SDDK/wf_trans.cpp"
  "SDDK/wave_functions.cpp"
  "mpi/communicator.cpp"
  "SDDK/dmatrix.cpp"
  "SDDK/gvec.cpp"
  "SDDK/matrix_storage.cpp"
  "gpu/acc.cpp"
  "gpu/acc_blas.cpp"
  "gpu/cusolver.cpp"
  "gpu/cublas.cpp"
  "sht/sht.cpp"
  "sirius_version.cpp"
  "mixer/mixer_functions.cpp"
  "linalg/eigensolver.cpp"
  "nlcglib/adaptor.cpp"
  )


# create library with .cpp, .cu and .f90 sources

if(USE_ROCM)
  # required for linking with compiler other than hcc
  set(HIP_HCC_FLAGS ${HIP_HCC_FLAGS} -fno-gpu-rdc)
  # Dependency generation of FindHIP module creates a lot of warnings - disable for release mode
  set(HIP_HCC_FLAGS_RELEASE ${HIP_HCC_FLAGS_RELEASE} -Wno-everything)
  # FindHIP does not account for C++ standard
  if(CMAKE_CXX_STANDARD)
    set(HIP_HCC_FLAGS ${HIP_HCC_FLAGS} -std=gnu++${CMAKE_CXX_STANDARD})
  endif()
  # macro from FindHIP package, which compiles all .cu files with hcc and cpp files with the set c++ compiler
  HIP_ADD_LIBRARY(sirius SHARED "${_SOURSES};${_CUSOURCES};${_FSOURCES}")
else()
  add_library(sirius "${_SOURSES};${_CUSOURCES};${_FSOURCES}")
endif()


target_link_libraries(sirius PUBLIC OpenMP::OpenMP_CXX
                                    SpFFT::spfft
                                    ${GSL_LIBRARY}
                                    "${LINALG_LIB}"
                                    MPI::MPI_CXX
                                    sirius::libxc
                                    sirius::libspg
                                    sirius::hdf5
                                    $<TARGET_NAME_IF_EXISTS:OpenMP::OpenMP_CXX>
                                    $<TARGET_NAME_IF_EXISTS:sirius::elpa>
                                    $<TARGET_NAME_IF_EXISTS:sirius::magma>
                                    $<TARGET_NAME_IF_EXISTS:sirius::libvdwxc>
                                    $<TARGET_NAME_IF_EXISTS:sirius::cudalibs>
                                    $<TARGET_NAME_IF_EXISTS:sirius_rocm>
                                    $<TARGET_NAME_IF_EXISTS:sirius_rocm_interface>
                                    $<TARGET_NAME_IF_EXISTS:nlcglib::nlcglib>
                                    $<TARGET_NAME_IF_EXISTS:kokkos::kokkos>
                                    $<$<BOOL:${USE_NVTX}>:nvToolsExt>
                                    $<$<BOOL:${USE_ROCM}>:ROCBLAS::rocblas>
                                    $<$<BOOL:${USE_ROCM}>:HIPLIBS::hiplibs>
                                    )

target_include_directories(sirius PUBLIC $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/src>
                                         $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/src/SDDK>
                                         $<BUILD_INTERFACE:${PROJECT_BINARY_DIR}/src>)
target_compile_definitions(sirius PUBLIC
  $<$<BOOL:${USE_PROFILER}>:__PROFILE>
  $<$<BOOL:${USE_SCALAPACK}>:__SCALAPACK>
  $<$<BOOL:${USE_MEMORY_POOL}>:__USE_MEMORY_POOL>
  $<$<BOOL:${DEBUG_MEMORY_POOL}>:__DEBUG_MEMORY_POOL>
  $<$<BOOL:${USE_ELPA}>:__ELPA>
  $<$<BOOL:${USE_NLCGLIB}>:__NLCGLIB>
  $<$<BOOL:${USE_CUDA}>:__GPU __CUDA>
  $<$<BOOL:${USE_ROCM}>:__HIP_PLATFORM_HCC__>
  $<$<BOOL:${USE_NVTX}>:__CUDA_NVTX>
  $<$<BOOL:${USE_MAGMA}>:__MAGMA>
  $<$<BOOL:${USE_ROCM}>:__GPU __ROCM>
  $<$<BOOL:${USE_VDWXC}>:__USE_VDWXC>
  $<$<BOOL:${HAVE_LIBVDW_WITH_MPI}>:__HAVE_VDWXC_MPI>
  $<$<BOOL:${USE_MAGMA} AND ${USE_ROCM}>:HAVE_HIP> # Required for magma headers
)

set_target_properties(sirius PROPERTIES POSITION_INDEPENDENT_CODE ON)

if(CREATE_FORTRAN_BINDINGS)
  set_target_properties(sirius PROPERTIES Fortran_MODULE_DIRECTORY mod_files)
  install(FILES "${PROJECT_BINARY_DIR}/src/mod_files/sirius.mod"
          DESTINATION "${CMAKE_INSTALL_INCLUDEDIR}/sirius")
endif()

install(TARGETS sirius
        EXPORT sirius_targets
        ARCHIVE DESTINATION "${CMAKE_INSTALL_LIBDIR}"
        LIBRARY DESTINATION "${CMAKE_INSTALL_LIBDIR}"
        INCLUDES DESTINATION "${CMAKE_INSTALL_INCLUDEDIR}/sirius"
                             "${CMAKE_INSTALL_INCLUDEDIR}/sirius/SDDK")
install(EXPORT sirius_targets
        FILE siriusTargets.cmake
        NAMESPACE sirius::
        DESTINATION "${CMAKE_INSTALL_LIBDIR}/cmake/sirius")
install(DIRECTORY "${PROJECT_SOURCE_DIR}/src"
        DESTINATION "${CMAKE_INSTALL_INCLUDEDIR}/sirius"
        FILES_MATCHING REGEX ".*(hpp|h)$")
install(FILES "${PROJECT_BINARY_DIR}/src/version.hpp"
              "${PROJECT_BINARY_DIR}/src/runtime_options_json.hpp"
        DESTINATION "${CMAKE_INSTALL_INCLUDEDIR}/sirius")
