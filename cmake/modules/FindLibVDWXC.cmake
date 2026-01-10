# find libvdwxc
# if in non-standard location set environment variabled `VDWXC_DIR` to the root directory

include(FindPackageHandleStandardArgs)
include(CheckSymbolExists)
find_package(PkgConfig REQUIRED)

pkg_check_modules(SIRIUS_LIBVDWXC IMPORTED_TARGET GLOBAL libvdwxc>=${LibVDWXC_FIND_VERSION})
pkg_check_modules(SIRIUS_FFTW3 IMPORTED_TARGET GLOBAL fftw3)

find_library(SIRIUS_FFTW3_MPI_LINK_LIBRARIES
             NAME fftw3_mpi
             HINTS
             SIRIUS_FFTW3_LIBRARIES_DIRS
             DOC "fftw3_mpi library")

find_path(SIRIUS_LIBVDWXC_INCLUDE_DIR
          NAMES vdwxc_mpi.h
          HINTS ${SIRIUS_LIBVDWXC_INCLUDE_DIRS}
          DOC "vdwxc include directory")

# try linking in C (C++ fails because vdwxc_mpi.h includes mpi.h inside extern "C"{...})
set(CMAKE_REQUIRED_LIBRARIES "${SIRIUS_LIBVDWXC_LINK_LIBRARIES}")

find_package_handle_standard_args(LibVDWXC DEFAULT_MSG SIRIUS_LIBVDWXC_LIBRARIES SIRIUS_LIBVDWXC_INCLUDE_DIR)

if(LibVDWXC_FOUND AND NOT TARGET sirius::libvdwxc)
  add_library(sirius::libvdwxc INTERFACE IMPORTED)
  set_target_properties(sirius::libvdwxc PROPERTIES
                                         INTERFACE_INCLUDE_DIRECTORIES "${SIRIUS_LIBVDWXC_INCLUDE_DIR};${SIRIUS_LIBVDWXC_INCLUDE_DIRS};${SIRIUS_FFTW3_INCLUDE_DIRS}"
                                         INTERFACE_LINK_LIBRARIES "${SIRIUS_LIBVDWXC_LINK_LIBRARIES};${SIRIUS_FFTW3_MPI_LINK_LIBRARIES};${SIRIUS_FFTW3_LINK_LIBRARIES}")
endif()

