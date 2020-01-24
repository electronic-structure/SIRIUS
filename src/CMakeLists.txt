# workaround for set_target_properties(... POSITION_INDEPENDENT_CODE ON) not
# working correctly
# list(APPEND CUDA_NVCC_FLAGS "-Xcompiler -fPIC")

include(GNUInstallDirs)  # required to get a proper LIBDIR variable

if(CREATE_FORTRAN_BINDINGS)
  set(_FSOURCES "sirius_api.cpp;sirius.f90")
endif()

if(USE_CUDA)
  file(GLOB_RECURSE CUFILES_SDDK "SDDK/*.cu")
  file(GLOB_RECURSE CUFILES_KERNELS "Kernels/*.cu")
  set(_CUSOURCES "${CUFILES_KERNELS};${CUFILES_SDDK}")
endif()

set(_SOURSES "Hamiltonian/hamiltonian.cpp"
  "Hamiltonian/hamiltonian_k.cpp"
  "Band/band.cpp"
  "Band/solve.cpp"
  "Density/density.cpp"
  "Density/augmentation_operator.cpp"
  "dft_ground_state.cpp"
  "Band/diag_pseudo_potential.cpp"
  "Band/diag_full_potential.cpp"
  "Band/residuals.cpp"
  "Beta_projectors/beta_projectors_base.cpp"
  "Hubbard/apply_hubbard_potential.cpp"
  "Hubbard/hubbard.cpp"
  "Hubbard/hubbard_occupancies_derivatives.cpp"
  "Hubbard/hubbard_occupancy.cpp"
  "Hubbard/hubbard_potential_energy.cpp"
  "Potential/generate_d_operator_matrix.cpp"
  "Potential/generate_pw_coeffs.cpp"
  "Potential/paw_potential.cpp"
  "Potential/poisson.cpp"
  "Potential/xc.cpp"
  "Unit_cell/unit_cell.cpp"
  "Unit_cell/unit_cell_symmetry.cpp"
  "simulation_context.cpp"
  "simulation_parameters.cpp"
  "Geometry/force.cpp"
  "Geometry/stress.cpp"
  "Geometry/non_local_functor.cpp"
  "Hamiltonian/local_operator.cpp"
  "Hamiltonian/non_local_operator.cpp"
  "K_point/generate_atomic_wave_functions.cpp"
  "K_point/generate_fv_states.cpp"
  "K_point/generate_spinor_wave_functions.cpp"
  "K_point/k_point.cpp"
  "K_point/k_point_set.cpp"
  "K_point/generate_gklo_basis.cpp"
  "K_point/test_fv_states.cpp"
  "energy.cpp"
  "field4d.cpp"
  "radial_integrals.cpp"
  "sbessel.cpp"
  "utils/cmd_args.cpp"
  "utils/timer.cpp"
  "utils/utils.cpp"
  "utils/rt_graph.cpp"
  "utils/profiler.cpp"
  "SDDK/wf_inner.cpp"
  "SDDK/wf_ortho.cpp"
  "SDDK/wf_trans.cpp"
  "SDDK/wave_functions.cpp"
  "SDDK/communicator.cpp"
  "SDDK/dmatrix.cpp"
  "SDDK/gvec.cpp"
  "SDDK/matrix_storage.cpp"
  "SDDK/GPU/acc.cpp"
  "SHT/sht.cpp"
  "sirius_version.cpp"
  "Mixer/mixer_functions.cpp"
  )

# create library with .cpp, .cu and .f90 sources
add_library(sirius "${_SOURSES};${_CUSOURCES};${_FSOURCES}")
target_link_libraries(sirius PUBLIC OpenMP::OpenMP_CXX SpFFT::spfft)
target_include_directories(sirius PUBLIC "${PROJECT_SOURCE_DIR}/src")

# add dependency on auto-generated headers
add_dependencies(sirius generate_version_hpp runtime_options_json_hpp)
set_target_properties(sirius PROPERTIES POSITION_INDEPENDENT_CODE ON)

if(CREATE_FORTRAN_BINDINGS)
  set_target_properties(sirius PROPERTIES Fortran_MODULE_DIRECTORY mod_files)
  install(FILES ${PROJECT_BINARY_DIR}/src/mod_files/sirius.mod 
          DESTINATION "${CMAKE_INSTALL_PREFIX}/include/sirius")
endif()

if(USE_ROCM)
  add_library(sirius_rocm_interface STATIC ./SDDK/GPU/rocfft_interface.cpp)
  set_target_properties(sirius_rocm_interface PROPERTIES POSITION_INDEPENDENT_CODE ON)
  # don't install as it is STATIC and not meant to be used by external projects
  target_link_libraries(sirius_rocm_interface PUBLIC OpenMP::OpenMP_CXX)

  # use include directories and definitions of sirius_rocm_interface (using custom targets not possible with generators)
  set(DEFINITIONS_PROP "$<TARGET_PROPERTY:sirius_rocm_interface,COMPILE_DEFINITIONS>")
  set(DEFINITIONS_GENERATOR "$<$<BOOL:${DEFINITIONS_PROP}>:-D$<JOIN:${DEFINITIONS_PROP}, -D>>")
  set(INCLUDE_DIR_PROP "$<TARGET_PROPERTY:sirius_rocm_interface,INCLUDE_DIRECTORIES>")
  set(INCLUDE_DIR_GENERATOR "$<$<BOOL:${INCLUDE_DIR_PROP}>:-I$<JOIN:${INCLUDE_DIR_PROP}, -I>>")

  # create gpu library compiled with hip
  file(GLOB_RECURSE HIPFILES_KERNELS "Kernels/*.cu")
  file(GLOB_RECURSE HIPFILES_SDDK "SDDK/*.cu")
  rocm_hip_add_library(sirius_rocm SHARED ${HIPFILES_SDDK} ${HIPFILES_KERNELS}
    FLAGS ${DEFINITIONS_GENERATOR} ${INCLUDE_DIR_GENERATOR} "-Wno-macro-redefined" OUTPUT_DIR ${CMAKE_CURRENT_BINARY_DIR})
endif()

install(TARGETS sirius 
        ARCHIVE DESTINATION "${CMAKE_INSTALL_LIBDIR}"
        LIBRARY DESTINATION "${CMAKE_INSTALL_LIBDIR}")
install(DIRECTORY "${PROJECT_SOURCE_DIR}/src" 
        DESTINATION "${CMAKE_INSTALL_PREFIX}/include/sirius" 
        FILES_MATCHING REGEX ".*(hpp|h)$")
install(FILES ${PROJECT_BINARY_DIR}/src/version.hpp 
        DESTINATION "${CMAKE_INSTALL_PREFIX}/include/sirius")
