# workaround for set_target_properties(... POSITION_INDEPENDENT_CODE ON) not
# working correctly
# list(APPEND CUDA_NVCC_FLAGS "-Xcompiler -fPIC")

if(USE_CUDA)
  file(GLOB_RECURSE CUFILES_SDDK "SDDK/*.cu")
  file(GLOB_RECURSE CUFILES_KERNELS "Kernels/*.cu")
  add_library(sirius "sirius_api.cpp;sirius.f90;${CUFILES_KERNELS};${CUFILES_SDDK}")
else()
  add_library(sirius "sirius_api.cpp;sirius.f90;${CUFILES_KERNELS};${CUFILES_SDDK}")
endif()

INSTALL (TARGETS sirius ARCHIVE DESTINATION
  ${CMAKE_INSTALL_PREFIX}/lib/)

set_target_properties(sirius PROPERTIES POSITION_INDEPENDENT_CODE ON)
set_target_properties(sirius PROPERTIES Fortran_MODULE_DIRECTORY mod_files)

# ugly hack, path below contains ${PROJECT_BINARY_DIR}/src/mod_files. In case the name of the current directory is
# changed, the path below needs to be adapted as well...
INSTALL ( CODE
  "EXECUTE_PROCESS (COMMAND \"${CMAKE_COMMAND}\" -E copy_directory \"${PROJECT_BINARY_DIR}/src/mod_files\" \"${CMAKE_INSTALL_PREFIX}/fortran\")"
)
install(DIRECTORY ./ DESTINATION "${CMAKE_INSTALL_PREFIX}/include/sirius" FILES_MATCHING REGEX ".*(hpp|h)$")
