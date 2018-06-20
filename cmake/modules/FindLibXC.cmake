include(FindPackageHandleStandardArgs)
find_package(PkgConfig REQUIRED)

pkg_search_module(_LIBXC libxc>=${LibXC_FIND_VERSION})

find_library(LIBXC_LIBRARIES NAMES xc
  PATH_SUFFIXES lib
  HINTS
  ENV EBROOTLIBXC
  ENV LIBXCROOT
  ${_LIBXC_LIBRARY_DIRS}
  DOC "libxc libraries list")

find_path(LIBXC_INCLUDE_DIR NAMES xc.h xc_f90_types_m.mod
  PATH_SUFFIXES inc include
  HINTS
  ${_LIBXC_INCLUDE_DIRS}
  ENV EBROOTLIBXC
  ENV LIBXCROOT
  )

find_package_handle_standard_args(LibXC DEFAULT_MSG LIBXC_LIBRARIES LIBXC_INCLUDE_DIR)
