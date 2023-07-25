include(FindPackageHandleStandardArgs)
find_package(PkgConfig REQUIRED)

pkg_search_module(_SCALAPACK scalapack)
find_library(SIRIUS_SCALAPACK_LIBRARIES
  NAMES scalapack scalapack-openmpi
  HINTS
  ${_SCALAPACK_LIBRARY_DIRS}
  ENV SCALAPACKROOT
  /usr
  PATH_SUFFIXES lib
  DOC "scalapack library path")

find_package_handle_standard_args(SCALAPACK DEFAULT_MSG SIRIUS_SCALAPACK_LIBRARIES)
