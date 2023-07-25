# find Elpa via pkg-config or easybuild

include(FindPackageHandleStandardArgs)
find_package(PkgConfig)

pkg_search_module(_ELPA
  elpa
  elpa_openmp
  elpa-openmp-2019.05.001
  elpa_openmp-2019.11.001
  elpa_openmp-2020.05.001
  elpa-2019.05.001
  elpa-2019.11.001
  elpa-2020.05.001)

find_library(SIRIUS_ELPA_LIBRARIES
  NAMES elpa elpa_openmp
  PATH_SUFFIXES lib
  HINTS
  ENV EBROOTELPA
  ENV ELPAROOT
  ${_ELPA_LIBRARY_DIRS}
  DOC "elpa libraries list")

find_path(SIRIUS_ELPA_INCLUDE_DIR
  NAMES elpa/elpa.h elpa/elpa_constants.h
  PATH_SUFFIXES include/elpa_openmp-$ENV{EBVERSIONELPA} include/elpa-$ENV{EBVERSIONELPA}
  HINTS
  ${_ELPA_INCLUDE_DIRS}
  ENV ELPAROOT
  ENV EBROOTELPA)

find_package_handle_standard_args(Elpa "DEFAULT_MSG" SIRIUS_ELPA_LIBRARIES SIRIUS_ELPA_INCLUDE_DIR)

message("ELPA_INCLUDE_DIR: ${SIRIUS_ELPA_INCLUDE_DIR}")

if(Elpa_FOUND AND NOT TARGET sirius::elpa)
  add_library(sirius::elpa INTERFACE IMPORTED)
  set_target_properties(sirius::elpa PROPERTIES
                                     INTERFACE_INCLUDE_DIRECTORIES "${SIRIUS_ELPA_INCLUDE_DIR}"
                                     INTERFACE_LINK_LIBRARIES "${SIRIUS_ELPA_LIBRARIES}")
endif()
