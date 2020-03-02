# workaround for set_target_properties(... POSITION_INDEPENDENT_CODE ON) not
# working correctly
# list(APPEND CUDA_NVCC_FLAGS "-Xcompiler -fPIC")

if(CREATE_FORTRAN_BINDINGS)
  set(_FSOURCES "api/sirius_api.cpp;api/sirius.f90")
endif()

if(USE_CUDA)
  file(GLOB_RECURSE CUFILES_SDDK "SDDK/*.cu")
  file(GLOB_RECURSE CUFILES_KERNELS "Kernels/*.cu")
  set(_CUSOURCES "${CUFILES_KERNELS};${CUFILES_SDDK}")
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
  "sht/sht.cpp"
  "sirius_version.cpp"
  "mixer/mixer_functions.cpp"
  )

if(USE_ROCM)
  add_library(sirius_rocm_interface STATIC ./gpu/rocfft_interface.cpp)
  set_target_properties(sirius_rocm_interface PROPERTIES POSITION_INDEPENDENT_CODE ON)
  # don't install as it is STATIC and not meant to be used by external projects
  target_link_libraries(sirius_rocm_interface PUBLIC OpenMP::OpenMP_CXX)

  # use include directories and definitions of sirius_rocm_interface (using custom targets not possible with generators)
  set(DEFINITIONS_PROP "$<TARGET_PROPERTY:sirius_rocm_interface,COMPILE_DEFINITIONS>")
  set(DEFINITIONS_GENERATOR "$<$<BOOL:${DEFINITIONS_PROP}>:-D$<JOIN:${DEFINITIONS_PROP}, -D>>")
  set(INCLUDE_DIR_PROP "$<TARGET_PROPERTY:sirius_rocm_interface,INCLUDE_DIRECTORIES>")
  set(INCLUDE_DIR_GENERATOR "$<$<BOOL:${INCLUDE_DIR_PROP}>:-I$<JOIN:${INCLUDE_DIR_PROP}, -I>>")

  # create gpu library compiled with hip
  file(GLOB_RECURSE HIPFILES_KERNELS "gpu/*.cu")
  file(GLOB_RECURSE HIPFILES_SDDK "SDDK/*.cu")
  rocm_hip_add_library(sirius_rocm SHARED ${HIPFILES_SDDK} ${HIPFILES_KERNELS}
    FLAGS ${DEFINITIONS_GENERATOR} ${INCLUDE_DIR_GENERATOR} "-Wno-macro-redefined" OUTPUT_DIR ${CMAKE_CURRENT_BINARY_DIR})
endif()

# create library with .cpp, .cu and .f90 sources
add_library(sirius "${_SOURSES};${_CUSOURCES};${_FSOURCES}")
target_link_libraries(sirius PUBLIC OpenMP::OpenMP_CXX
                                    SpFFT::spfft
                                    GSL::gsl
                                    "${LINALG_LIB}"
                                    MPI::MPI_CXX
                                    sirius::libxc
                                    sirius::libspg
                                    sirius::hdf5
                                    $<TARGET_NAME_IF_EXISTS:sirius::elpa>
                                    $<TARGET_NAME_IF_EXISTS:sirius::magma>
                                    $<TARGET_NAME_IF_EXISTS:sirius::libvdwxc>
                                    $<TARGET_NAME_IF_EXISTS:sirius::cudalibs>
                                    $<TARGET_NAME_IF_EXISTS:sirius_rocm>
                                    $<TARGET_NAME_IF_EXISTS:sirius_rocm_interface>
                                    $<TARGET_NAME_IF_EXISTS:hipblas_port>
                                    $<$<BOOL:${USE_ROCM}>:${ROCM_LIBRARIES}>
                                    $<$<BOOL:${USE_NVTX}>:nvToolsExt>
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
  $<$<BOOL:${USE_CUDA}>:__GPU __CUDA>
  $<$<BOOL:${USE_NVTX}>:__CUDA_NVTX>
  $<$<BOOL:${USE_MAGMA}>:__MAGMA>
  $<$<BOOL:${USE_ROCM}>:__GPU __ROCM>
  $<$<BOOL:${USE_VDWXC}>:__USE_VDWXC>
  $<$<BOOL:${HAVE_LIBVDW_WITH_MPI}>:__HAVE_VDWXC_MPI>
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
