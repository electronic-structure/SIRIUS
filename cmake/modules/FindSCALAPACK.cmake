include(FindPackageHandleStandardArgs)
find_package(PkgConfig REQUIRED)

pkg_search_module(_SCALAPACK scalapack)
find_library(SIRIUS_SCALAPACK_LIBRARIES
  NAMES scalapack scalapack-openmpi scalapack-mpich
  HINTS
  ${_SCALAPACK_LIBRARY_DIRS}
  ENV SCALAPACKROOT
  /usr /usr/local
  PATH_SUFFIXES lib lib64
  DOC "scalapack library path")

if(NOT TARGET sirius::scalapack)
  add_library(sirius::scalapack INTERFACE IMPORTED)
  set_target_properties(sirius::scalapack PROPERTIES
    INTERFACE_LINK_LIBRARIES "${SIRIUS_SCALAPACK_LIBRARIES}"
  )
  if(_SCALAPACK_INCLUDE_DIRS)
    set_target_properties(sirius::scalapack PROPERTIES
      INTERFACE_INCLUDE_DIRECTORIES "${_SCALAPACK_INCLUDE_DIRS}"
    )
  endif()
endif()

if(CMAKE_Fortran_IMPLICIT_LINK_LIBRARIES)
  set_property(TARGET sirius::scalapack APPEND PROPERTY
    INTERFACE_LINK_LIBRARIES ${CMAKE_Fortran_IMPLICIT_LINK_LIBRARIES})
endif()

find_package_handle_standard_args(SCALAPACK DEFAULT_MSG SIRIUS_SCALAPACK_LIBRARIES)
mark_as_advanced(SIRIUS_SCALAPACK_LIBRARIES)
