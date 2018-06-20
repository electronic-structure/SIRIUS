include(FindPackageHandleStandardArgs)
find_package(PkgConfig REQUIRED)

pkg_search_module(_SCALAPACK scalapack)
find_library(SCALAPACK_LIBRARIES NAMES scalapack
  HINTS
  ${_SCALAPACK_LIBRARY_DIRS}
  ENV SCALAPACKROOT
  /usr/lib
  PATH_SUFFIXES
  lib
  DOC "scalapack library path")

find_path(SCALAPACK_INCLUDE_DIR
  pblas.h
  PATH_SUFFIXES
  include
  HINTS
  ENV SCALAPACKROOT
  ${_SCALAPACK_INCLUDE_DIRS}
  /usr/lib)

find_package_handle_standard_args(ScaLAPACK DEFAULT_MSG SCALAPACK_LIBRARIES SCALAPACK_INCLUDE_DIR)
