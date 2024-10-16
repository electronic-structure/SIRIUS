# find roctracer roctracer

include(FindPackageHandleStandardArgs)

find_library(SIRIUS_ROCTRACER_LIBRARIES
  NAMES roctracer64 roctracer
  PATH_SUFFIXES lib
  DOC "roctracer libraries list")

find_path(SIRIUS_ROCTRACER_INCLUDE_DIR
  NAMES roctracer.h
  PATH_SUFFIXES include include/roctracer
)

find_package_handle_standard_args(Roctracer "DEFAULT_MSG" SIRIUS_ROCTRACER_LIBRARIES SIRIUS_ROCTRACER_INCLUDE_DIR)

if(Roctracer_FOUND AND NOT TARGET sirius::roctracer)
  add_library(sirius::roctracer INTERFACE IMPORTED)
  set_target_properties(sirius::roctracer PROPERTIES
                                     INTERFACE_INCLUDE_DIRECTORIES "${SIRIUS_ROCTRACER_INCLUDE_DIR}"
                                     INTERFACE_LINK_LIBRARIES "${SIRIUS_ROCTRACER_LIBRARIES}")
endif()
