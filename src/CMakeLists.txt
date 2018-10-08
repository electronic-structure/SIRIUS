# workaround for set_target_properties(... POSITION_INDEPENDENT_CODE ON) not
# working correctly
# list(APPEND CUDA_NVCC_FLAGS "-Xcompiler -fPIC")

# keep two libraries: libsirius and libsirius_f

if(USE_CUDA)
  file(GLOB_RECURSE CUFILES_SDDK "SDDK/*.cu")
  file(GLOB_RECURSE CUFILES_KERNELS "Kernels/*.cu")
  add_library(sirius_f "sirius_api.cpp;sirius.f90;${CUFILES_KERNELS};${CUFILES_SDDK}")
  add_library(sirius "${CUFILES_KERNELS};${CUFILES_SDDK}")
else()
  add_library(sirius "dummy.cpp")
  add_library(sirius_f "sirius_api.cpp;sirius.f90")
endif()

INSTALL (TARGETS sirius ARCHIVE DESTINATION
  ${CMAKE_INSTALL_PREFIX}/lib/)
INSTALL (TARGETS sirius_f ARCHIVE DESTINATION
  ${CMAKE_INSTALL_PREFIX}/lib/)

set_target_properties(sirius PROPERTIES POSITION_INDEPENDENT_CODE ON)
set_target_properties(sirius_f PROPERTIES POSITION_INDEPENDENT_CODE ON)
set_target_properties(sirius_f PROPERTIES Fortran_MODULE_DIRECTORY mod_files)

# ugly hack, path below contains ${PROJECT_BINARY_DIR}/src/mod_files. In case the name of the current directory is
# changed, the path below needs to be adapted as well...
INSTALL ( CODE
  "EXECUTE_PROCESS (COMMAND \"${CMAKE_COMMAND}\" -E copy_directory \"${PROJECT_BINARY_DIR}/src/mod_files\" \"${CMAKE_INSTALL_PREFIX}/fortran\")"
)
install(DIRECTORY ./ DESTINATION "${CMAKE_INSTALL_PREFIX}/include/sirius" FILES_MATCHING REGEX ".*(hpp|h)$")
