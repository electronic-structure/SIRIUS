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

set(_SOURSES "Band/band.cpp;Band/solve.cpp;Density/density.cpp;dft_ground_state.cpp"
  "Band/diag_pseudo_potential.cpp;Band/diag_full_potential.cpp;Band/residuals.cpp"
  "Hubbard/apply_hubbard_potential.cpp"
  "Hubbard/Hubbard.cpp"
  "Hubbard/hubbard_generate_atomic_orbitals.cpp"
  "Hubbard/hubbard_occupancies_derivatives.cpp"
  "Hubbard/hubbard_occupancy.cpp"
  "Hubbard/hubbard_potential_energy.cpp"
  "Potential/generate_d_operator_matrix.cpp"
  "Potential/generate_pw_coeffs.cpp"
  "Potential/paw_potential.cpp"
  "Potential/poisson.cpp"
  "Potential/xc.cpp"
  "Unit_cell/unit_cell.cpp"
  "simulation_context.cpp"
  "simulation_parameters.cpp"
  "Geometry/force.cpp"
  "Geometry/stress.cpp"
  "Geometry/non_local_functor.cpp")

# create library with .cpp, .cu and .f90 sources
add_library(sirius STATIC "${_SOURSES};${_CUSOURCES};${_FSOURCES}")

# add dependency on auto-generated headers
add_dependencies(sirius generate_version_hpp)
add_dependencies(sirius runtime_options_json_hpp)

set_target_properties(sirius PROPERTIES POSITION_INDEPENDENT_CODE ON)

include_directories(".")

INSTALL (TARGETS sirius ARCHIVE DESTINATION
  ARCHIVE DESTINATION "${CMAKE_INSTALL_LIBDIR}"
  LIBRARY DESTINATION "${CMAKE_INSTALL_LIBDIR}"
  )
target_link_libraries(sirius PUBLIC OpenMP::OpenMP_CXX)

if(CREATE_FORTRAN_BINDINGS)
  set_target_properties(sirius PROPERTIES Fortran_MODULE_DIRECTORY mod_files)
  INSTALL ( CODE
    "EXECUTE_PROCESS (COMMAND \"${CMAKE_COMMAND}\" -E copy_directory \"${PROJECT_BINARY_DIR}/src/mod_files\" \"${CMAKE_INSTALL_PREFIX}/include/sirius/\")"
    )
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

install(DIRECTORY ./ DESTINATION "${CMAKE_INSTALL_PREFIX}/include/sirius" FILES_MATCHING REGEX ".*(hpp|h)$")
install(FILES ${PROJECT_BINARY_DIR}/src/version.hpp DESTINATION "${CMAKE_INSTALL_PREFIX}/include/sirius")
