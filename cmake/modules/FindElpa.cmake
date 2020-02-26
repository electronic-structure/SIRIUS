# find Elpa via pkg-config or easybuild

include(FindPackageHandleStandardArgs)
find_package(PkgConfig)

pkg_search_module(_ELPA elpa elpa_openmp)

find_library(ELPA_LIBRARIES
  NAMES elpa elpa_openmp
  PATH_SUFFIXES lib
  HINTS
  ENV EBROOTELPA
  ENV ELPAROOT
  ${_ELPA_LIBRARY_DIRS}
  DOC "elpa libraries list")

find_path(ELPA_INCLUDE_DIR
  NAMES elpa.h elpa_constants.h
  PATH_SUFFIXES include/elpa_openmp-$ENV{EBVERSIONELPA}/elpa include/elpa_openmp-$ENV{EBVERSIONELPA} elpa
  HINTS
  ${_ELPA_INCLUDE_DIRS}
  ENV ELPAROOT
  ENV EBROOTELPA)

find_package_handle_standard_args(Elpa "DEFAULT_MSG" ELPA_LIBRARIES ELPA_INCLUDE_DIR)

if(Elpa_FOUND AND NOT TARGET sirius::elpa)
  add_library(sirius::elpa INTERFACE IMPORTED)
  set_target_properties(sirius::elpa PROPERTIES
                                     INTERFACE_INCLUDE_DIRECTORIES "${ELPA_INCLUDE_DIR}"
                                     INTERFACE_LINK_LIBRARIES "${ELPA_LIBRARIES}")
endif()
