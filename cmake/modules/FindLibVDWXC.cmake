# find libvdwxc
# if in non-standard location set environment variabled `VDWCXC_DIR` to the root directory

include(FindPackageHandleStandardArgs)
find_package(PkgConfig REQUIRED)

pkg_search_module(_LIBVDWXC libvdwxc>=${LibVDWXC_FIND_VERSION})

find_path(LIBVDWXC_INCLUDE_DIR
  NAMES vdwxc.h vdwxc_mpi.h
  PATH_SUFFIXES include inc
  HINTS
  ENV EBROOTVDWXCLIB
  ENV VDWXC_DIR
  ENV LIBVDWXCROOT
  ${_LIBVDWXC_INCLUDE_DIRS}
  DOC "vdwxc include directory")


find_library(LIBVDWXC_LIBRARIES
  NAMES vdwxc
  PATH_SUFFIXES lib
  HINTS
  ENV EBROOTVDWXC
  ENV VDWXC_DIR
  ENV VDWXCROOT
  ${_LIBVDWXC_LIBRARY_DIRS}
  DOC "vdwxc libraries list")

find_package_handle_standard_args(LibVDWXC DEFAULT_MSG LIBVDWXC_LIBRARIES LIBVDWXC_INCLUDE_DIR)
