# find spglib
# if in non-standard location set environment variabled `SPG_DIR` to the root directory

include(FindPackageHandleStandardArgs)

find_path(LIBSPG_INCLUDE_DIR
  NAMES spglib/spglib.h
  PATH_SUFFIXES include inc
  HINTS
  ENV EBROOTSPGLIB
  ENV SPG_DIR
  ENV LIBSPGROOT
  DOC "spglib include directory")


find_library(LIBSPG_LIBRARIES
  NAMES symspg
  PATH_SUFFIXES lib
  HINTS
  ENV EBROOTSPGLIB
  ENV SPG_DIR
  ENV LIBSPGROOT
  DOC "spglib libraries list")

find_package_handle_standard_args(LibSPG DEFAULT_MSG LIBSPG_LIBRARIES LIBSPG_INCLUDE_DIR)
