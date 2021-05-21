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
  "density/occupation_matrix.cpp"
  "dft/dft_ground_state.cpp"
  "dft/energy.cpp"
  "beta_projectors/beta_projectors_base.cpp"
  "hubbard/apply_hubbard_potential.cpp"
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
  "unit_cell/unit_cell_symmetry.cpp"
  "unit_cell/atom_type.cpp"
  "geometry/force.cpp"
  "geometry/stress.cpp"
  "geometry/non_local_functor.cpp"
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
  "sht/sht.cpp"
  "sirius_version.cpp"
  "mixer/mixer_functions.cpp"
  "linalg/eigensolver.cpp"
  "linalg/linalg_spla.cpp"
  "nlcglib/adaptor.cpp"
  "context/simulation_context.cpp"
  "context/simulation_parameters.cpp"
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

if(USE_CUDA AND ${CMAKE_VERSION} VERSION_GREATER_EQUAL "3.18.0")
  if(${CUDA_ARCH})
    set_property(TARGET sirius PROPERTY CUDA_ARCHITECTURES ${CUDA_ARCH})
  endif()
endif()

target_link_libraries(sirius PUBLIC SpFFT::spfft
                                    SPLA::spla
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
                                    $<$<BOOL:${USE_ROCM}>:roc::rocblas>
                                    $<$<BOOL:${USE_ROCM}>:hip::host>
                                    )

target_include_directories(sirius PUBLIC $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/src>
                                         $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/src/SDDK>
                                         $<BUILD_INTERFACE:${PROJECT_BINARY_DIR}/src>
                                         $<BUILD_INTERFACE:${PROJECT_BINARY_DIR}/src/mod_files>)
target_compile_definitions(sirius PUBLIC
  $<$<BOOL:${USE_PROFILER}>:SIRIUS_PROFILE>
  $<$<BOOL:${USE_SCALAPACK}>:SIRIUS_SCALAPACK>
  $<$<BOOL:${USE_MEMORY_POOL}>:SIRIUS_USE_MEMORY_POOL>
  $<$<BOOL:${DEBUG_MEMORY_POOL}>:SIRIUS_DEBUG_MEMORY_POOL>
  $<$<BOOL:${USE_POWER_COUNTER}>:SIRIUS_USE_POWER_COUNTER>
  $<$<BOOL:${USE_ELPA}>:SIRIUS_ELPA>
  $<$<BOOL:${USE_NLCGLIB}>:SIRIUS_NLCGLIB>
  $<$<BOOL:${USE_CUDA}>:SIRIUS_GPU SIRIUS_CUDA>
  $<$<BOOL:${USE_NVTX}>:SIRIUS_CUDA_NVTX>
  $<$<BOOL:${USE_MAGMA}>:SIRIUS_MAGMA>
  $<$<BOOL:${USE_ROCM}>:SIRIUS_GPU SIRIUS_ROCM>
  $<$<BOOL:${USE_VDWXC}>:SIRIUS_USE_VDWXC>
  $<$<BOOL:${HAVE_LIBVDW_WITH_MPI}>:SIRIUS_HAVE_VDWXC_MPI>
  $<$<AND:$<BOOL:${USE_MAGMA}>,$<BOOL:${USE_ROCM}>>:HAVE_HIP> # Required for magma headers
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
                             "${CMAKE_INSTALL_INCLUDEDIR}/sirius/src/SDDK")
install(EXPORT sirius_targets
        FILE siriusTargets.cmake
        NAMESPACE sirius::
        DESTINATION "${CMAKE_INSTALL_LIBDIR}/cmake/sirius")
install(DIRECTORY "${PROJECT_SOURCE_DIR}/src"
        DESTINATION "${CMAKE_INSTALL_INCLUDEDIR}/sirius"
        FILES_MATCHING REGEX ".*(hpp|h)$")
install(FILES "${PROJECT_BINARY_DIR}/src/version.hpp"
              "${PROJECT_BINARY_DIR}/src/context/runtime_options_json.hpp"
              "${PROJECT_BINARY_DIR}/src/context/input_schema.hpp"
        DESTINATION "${CMAKE_INSTALL_INCLUDEDIR}/sirius")
