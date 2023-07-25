# find libvdwxc
# if in non-standard location set environment variabled `VDWXC_DIR` to the root directory

include(FindPackageHandleStandardArgs)
include(CheckSymbolExists)
find_package(PkgConfig REQUIRED)

pkg_search_module(_LIBVDWXC libvdwxc>=${LibVDWXC_FIND_VERSION})

find_path(SIRIUS_LIBVDWXC_INCLUDE_DIR
  NAMES vdwxc.h vdwxc_mpi.h
  PATH_SUFFIXES include inc
  HINTS
  ENV EBROOTLIBVDWXC
  ENV VDWXC_DIR
  ENV LIBVDWXCROOT
  ${_LIBVDWXC_INCLUDE_DIRS}
  DOC "vdwxc include directory")

find_library(SIRIUS_LIBVDWXC_LIBRARIES
  NAMES vdwxc
  PATH_SUFFIXES lib
  HINTS
  ENV EBROOTLIBVDWXC
  ENV VDWXC_DIR
  ENV LIBVDWXCROOT
  ${_LIBVDWXC_LIBRARY_DIRS}
  DOC "vdwxc libraries list")

# try linking in C (C++ fails because vdwxc_mpi.h includes mpi.h inside extern "C"{...})
set(CMAKE_REQUIRED_LIBRARIES "${SIRIUS_LIBVDWXC_LIBRARIES}")
check_symbol_exists(vdwxc_init_mpi "${SIRIUS_LIBVDWXC_INCLUDE_DIR}/vdwxc_mpi.h" HAVE_LIBVDW_WITH_MPI)

find_package_handle_standard_args(LibVDWXC DEFAULT_MSG SIRIUS_LIBVDWXC_LIBRARIES SIRIUS_LIBVDWXC_INCLUDE_DIR)

if(LibVDWXC_FOUND AND NOT TARGET sirius::libvdwxc)
  add_library(sirius::libvdwxc INTERFACE IMPORTED)
  set_target_properties(sirius::libvdwxc PROPERTIES
                                         INTERFACE_INCLUDE_DIRECTORIES "${SIRIUS_LIBVDWXC_INCLUDE_DIR}"
                                         INTERFACE_LINK_LIBRARIES "${SIRIUS_LIBVDWXC_LIBRARIES}")
endif()
