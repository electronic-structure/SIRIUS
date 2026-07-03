include(FindPackageHandleStandardArgs)

# Prefer a package-provided target: it is the only route that can faithfully
# carry a vendor's complete BLACS / BLAS / LAPACK closure without guessing.
find_package(scalapack CONFIG QUIET)

set(_scalapack_provider_target "")
foreach(_target IN ITEMS scalapack::scalapack ScaLAPACK::ScaLAPACK scalapack)
  if(TARGET ${_target})
    set(_scalapack_provider_target ${_target})
    break()
  endif()
endforeach()

# Most distribution packages provide a .pc file.  Query its static closure so
# the imported target also works when libscalapack itself is an archive.
if(NOT _scalapack_provider_target)
  find_package(PkgConfig QUIET)
  if(PkgConfig_FOUND)
    set(_scalapack_pkg_config_argn "${PKG_CONFIG_ARGN}")
    list(PREPEND PKG_CONFIG_ARGN --static)
    pkg_search_module(_SCALAPACK QUIET IMPORTED_TARGET
      scalapack scalapack-openmpi scalapack-mpich)
    set(PKG_CONFIG_ARGN "${_scalapack_pkg_config_argn}")

    if(_SCALAPACK_FOUND)
      set(_scalapack_provider_target PkgConfig::_SCALAPACK)
    endif()
  endif()
endif()

# Last resort for installations with neither a Config package nor pkg-config.
# SIRIUS_SCALAPACK_LIBRARIES is deliberately a *list*: a caller can supply the
# full ordered static closure (ScaLAPACK, BLACS and any init archives) instead
# of losing it through a one-library find_library() result.
set(SIRIUS_SCALAPACK_LIBRARIES "" CACHE STRING
  "Ordered ScaLAPACK link items for the raw fallback")
if(NOT _scalapack_provider_target)
  if(NOT SIRIUS_SCALAPACK_LIBRARIES)
    find_library(SIRIUS_SCALAPACK_LIBRARY
      NAMES scalapack scalapack-openmpi scalapack-mpich
      HINTS ENV SCALAPACKROOT
      PATH_SUFFIXES lib lib64
      DOC "ScaLAPACK library path")
    set(SIRIUS_SCALAPACK_LIBRARIES "${SIRIUS_SCALAPACK_LIBRARY}")
  endif()
  set(_scalapack_provider_target "${SIRIUS_SCALAPACK_LIBRARIES}")
  set(_scalapack_raw_fallback TRUE)
endif()

find_package_handle_standard_args(SCALAPACK
  REQUIRED_VARS _scalapack_provider_target)

if(SCALAPACK_FOUND AND NOT TARGET sirius::scalapack)
  add_library(sirius::scalapack INTERFACE IMPORTED)
  target_link_libraries(sirius::scalapack INTERFACE
    "${_scalapack_provider_target}")

  # A raw shared library records its own dependencies, but a raw static archive
  # does not.  Append the universal part of the dependency graph in the only
  # safe direction: ScaLAPACK first, then MPI and LAPACK.  For a split static
  # BLACS installation users must set SIRIUS_SCALAPACK_LIBRARIES to its complete
  # ordered archive list; Config and pkg-config providers need no guessing here.
  if(_scalapack_raw_fallback)
    find_package(MPI REQUIRED COMPONENTS CXX)
    find_package(LAPACK REQUIRED)
    target_link_libraries(sirius::scalapack INTERFACE
      MPI::MPI_CXX
      LAPACK::LAPACK)
  endif()

  if(CMAKE_Fortran_IMPLICIT_LINK_LIBRARIES)
    target_link_libraries(sirius::scalapack INTERFACE
      ${CMAKE_Fortran_IMPLICIT_LINK_LIBRARIES})
  endif()
endif()

mark_as_advanced(SIRIUS_SCALAPACK_LIBRARY)
