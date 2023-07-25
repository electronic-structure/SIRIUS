# find spglib
# if in non-standard location set environment variabled `SPG_DIR` to the root directory

include(FindPackageHandleStandardArgs)

find_path(SIRIUS_LIBSPG_INCLUDE_DIR
  NAMES spglib.h
  PATH_SUFFIXES include include/spglib spglib
  HINTS
  ENV EBROOTSPGLIB
  ENV SPG_DIR
  ENV LIBSPGROOT
  DOC "spglib include directory")

find_library(SIRIUS_LIBSPG_LIBRARIES
  NAMES symspg
  PATH_SUFFIXES lib
  HINTS
  ENV EBROOTSPGLIB
  ENV SPG_DIR
  ENV LIBSPGROOT
  DOC "spglib libraries list")

find_package_handle_standard_args(LibSPG DEFAULT_MSG SIRIUS_LIBSPG_LIBRARIES SIRIUS_LIBSPG_INCLUDE_DIR)

if(LibSPG_FOUND AND NOT TARGET sirius::libspg)
  add_library(sirius::libspg INTERFACE IMPORTED)
  set_target_properties(sirius::libspg PROPERTIES
                                       INTERFACE_INCLUDE_DIRECTORIES "${SIRIUS_LIBSPG_INCLUDE_DIR}"
                                       INTERFACE_LINK_LIBRARIES "${SIRIUS_LIBSPG_LIBRARIES}")
endif()
