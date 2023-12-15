# find roctracer roctx

include(FindPackageHandleStandardArgs)

find_library(SIRIUS_ROCTX_LIBRARIES
  NAMES roctx64 roctx
  PATH_SUFFIXES lib
  DOC "roctracer/roctx libraries list")

find_path(SIRIUS_ROCTX_INCLUDE_DIR
  NAMES roctx.h
  PATH_SUFFIXES include include/roctracer
)

find_package_handle_standard_args(RocTX "DEFAULT_MSG" SIRIUS_ROCTX_LIBRARIES SIRIUS_ROCTX_INCLUDE_DIR)

if(RocTX_FOUND AND NOT TARGET sirius::roctx)
  add_library(sirius::roctx INTERFACE IMPORTED)
  set_target_properties(sirius::roctx PROPERTIES
                                     INTERFACE_INCLUDE_DIRECTORIES "${SIRIUS_ROCTX_INCLUDE_DIR}"
                                     INTERFACE_LINK_LIBRARIES "${SIRIUS_ROCTX_LIBRARIES}")
endif()
