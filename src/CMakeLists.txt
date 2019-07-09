# workaround for set_target_properties(... POSITION_INDEPENDENT_CODE ON) not
# working correctly
# list(APPEND CUDA_NVCC_FLAGS "-Xcompiler -fPIC")

include(GNUInstallDirs)  # required to get a proper LIBDIR variable

# keep two libraries: libsirius and libsirius_f

if(CREATE_FORTRAN_BININGS)
  set(_FSOURCES "sirius_api.cpp;sirius.f90")
endif()

if(USE_CUDA)
  file(GLOB_RECURSE CUFILES_SDDK "SDDK/*.cu")
  file(GLOB_RECURSE CUFILES_KERNELS "Kernels/*.cu")
  set(_CUSOURCES "${CUFILES_KERNELS};${CUFILES_SDDK}")
endif()


if(CREATE_FORTRAN_BININGS OR USE_CUDA)
  add_library(sirius STATIC "${_CUSOURCES};${_FSOURCES}")
  set_target_properties(sirius PROPERTIES POSITION_INDEPENDENT_CODE ON)
  set_target_properties(sirius PROPERTIES Fortran_MODULE_DIRECTORY mod_files)
  INSTALL (TARGETS sirius ARCHIVE DESTINATION
    ARCHIVE DESTINATION "${CMAKE_INSTALL_LIBDIR}"
    LIBRARY DESTINATION "${CMAKE_INSTALL_LIBDIR}"
    )
  target_link_libraries(sirius PUBLIC OpenMP::OpenMP_CXX)

endif()

if(CREATE_FORTRAN_BININGS)
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
