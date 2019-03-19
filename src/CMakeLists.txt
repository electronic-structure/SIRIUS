# workaround for set_target_properties(... POSITION_INDEPENDENT_CODE ON) not
# working correctly
# list(APPEND CUDA_NVCC_FLAGS "-Xcompiler -fPIC")

# keep two libraries: libsirius and libsirius_f

if(USE_CUDA)
  file(GLOB_RECURSE CUFILES_SDDK "SDDK/*.cu")
  file(GLOB_RECURSE CUFILES_KERNELS "Kernels/*.cu")
  add_library(sirius_cu "${CUFILES_KERNELS};${CUFILES_SDDK}")
  set_target_properties(sirius_cu PROPERTIES POSITION_INDEPENDENT_CODE ON)
  INSTALL (TARGETS sirius_cu ARCHIVE DESTINATION ${CMAKE_INSTALL_PREFIX}/lib/)
endif()
if(CREATE_FORTRAN_BINDINGS)
  add_library(sirius_f "sirius_api.cpp;sirius.f90")
  add_dependencies(sirius_f generate_version_hpp runtime_options_json_hpp)
  INSTALL (TARGETS sirius_f ARCHIVE DESTINATION
    ${CMAKE_INSTALL_PREFIX}/lib/)
  set_target_properties(sirius_f PROPERTIES POSITION_INDEPENDENT_CODE ON)
  set_target_properties(sirius_f PROPERTIES Fortran_MODULE_DIRECTORY mod_files)
  INSTALL ( CODE
    "EXECUTE_PROCESS (COMMAND \"${CMAKE_COMMAND}\" -E copy_directory \"${PROJECT_BINARY_DIR}/src/mod_files\" \"${CMAKE_INSTALL_PREFIX}/fortran\")"
    )
endif()

if(USE_ROCM)
  add_library(sirius_rocm_interface STATIC ./SDDK/GPU/rocfft_interface.cpp)
  set_target_properties(sirius_rocm_interface PROPERTIES POSITION_INDEPENDENT_CODE ON)
  # don't install as it is STATIC and not meant to be used by external projects

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
